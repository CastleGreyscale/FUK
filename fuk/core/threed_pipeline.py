"""
3D Reconstruction Pipeline Runner for FUK

Turns images into proxy geometry — collision meshes, depth references, NeRF
initialisation, geometry-informed AOVs for compositing. Not a replacement for
dedicated photogrammetry.

Two tiers, both loading from their own vendored repos rather than through the
DiffSynth hub:

    trellis   single image  → mesh          (isolated env, subprocess)
    vggt      1..N images   → point cloud + mesh  (main venv)

The first-class input is the LoRA Dataset Builder's Qwen orbital views:
synthetic multi-view → 3D is the workflow this exists for. Output is proxy
geometry — Qwen is a generative model, not a physically consistent renderer,
so cross-view geometry is approximate by construction.

See docs/3D_RECONSTRUCTION_SYSTEM.md
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pipeline_base import PipelineRunner, _log

from core.threed import trellis_backend, vggt_backend

# Below this, VGGT's angular coverage is too thin for a clean surface. Not
# enforced — the UI warns and the user decides.
MIN_RECOMMENDED_VIEWS = 6

SUPPORTED_FORMATS = ("glb", "ply", "obj")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ThreeDPipelineRunner(PipelineRunner):
    """Reconstruction runner. Dispatches by the model entry's `type` field."""

    pipeline_family = "threed"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_available(self) -> Dict[str, dict]:
        """Every threed model, annotated with whether it can actually run."""
        models = {}
        for key, entry in self.models_config.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            if entry.get("pipeline") != "threed":
                continue

            model_type = entry.get("type", "single_image")
            if model_type == "single_image":
                status = trellis_backend.availability()
            else:
                status = {
                    "available": vggt_backend.vendor_available(),
                    "missing": [] if vggt_backend.vendor_available() else ["VGGT source (vendor/VGGT)"],
                    "hint": None if vggt_backend.vendor_available()
                            else "git clone https://github.com/facebookresearch/vggt.git fuk/vendor/VGGT",
                }

            models[key] = {
                "key": key,
                "name": entry.get("name", key),
                "description": entry.get("description", ""),
                "type": model_type,
                "supports": entry.get("supports", ["glb"]),
                "vram_gb_estimate": entry.get("vram_gb_estimate"),
                "max_input_dim": entry.get("max_input_dim"),
                **status,
            }
        return models

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_images: Union[str, List[str]],
        output_path: Union[str, Path],
        model: str = "trellis",
        export_formats: Optional[List[str]] = None,
        vram_preset: str = None,
        progress_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Reconstruct geometry from one or more images.

        Args:
            input_images: One path, or a list. A directory expands to the
                images inside it.
            output_path: Directory to write mesh.glb / pointcloud.ply /
                reconstruction_meta.json into.
            model: A models.json key or alias — "trellis" or "vggt".
            export_formats: Any of glb, ply, obj. Defaults to ["glb"].
        """
        started = time.perf_counter()

        paths = self._resolve_inputs(input_images)
        if not paths:
            raise ValueError("No readable input images")

        formats = [f.lower() for f in (export_formats or ["glb"])]
        unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
        if unknown:
            raise ValueError(
                f"Unsupported export format(s): {unknown}. Supported: {list(SUPPORTED_FORMATS)}"
            )

        model_type = self.resolve_model_type(model)
        entry = self.get_model_entry(model_type)
        if entry.get("pipeline") != "threed":
            raise ValueError(f"'{model}' is not a 3D reconstruction model")

        output_dir = Path(output_path)
        # Tolerate being handed a file path — every other runner takes one.
        if output_dir.suffix:
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        kind = entry.get("type", "single_image")

        self.log_generation_header(
            "3D RECONSTRUCTION",
            model_type,
            entry,
            {
                "inputs": len(paths),
                "type": kind,
                "export_formats": formats,
                "output_dir": str(output_dir),
            },
        )

        if kind == "single_image" and len(paths) > 1:
            _log(self.log_prefix,
                 f"{model_type} is single-image — using the first of {len(paths)} images", "warning")
        if kind == "multi_view" and len(paths) < MIN_RECOMMENDED_VIEWS:
            _log(self.log_prefix,
                 f"Only {len(paths)} views — {MIN_RECOMMENDED_VIEWS}+ recommended for clean geometry",
                 "warning")

        # These models sit at 12–16GB peak, which will not coexist with a
        # resident Qwen or Wan pipeline on a 24GB card. Same single-cached-
        # pipeline rule the rest of the backend follows.
        self._free_vram_for_reconstruction()

        if kind == "single_image":
            result = trellis_backend.reconstruct(
                image=paths[0],
                output_dir=output_dir,
                entry=entry,
                export_formats=formats,
                seed=int(kwargs.get("seed", 42)),
                steps=int(kwargs.get("steps", 12)),
                cfg_strength=float(kwargs.get("cfg_strength", 7.5)),
                simplify=float(kwargs.get("simplify", 0.95)),
                texture_size=int(kwargs.get("texture_size", 1024)),
                fill_holes=bool(kwargs.get("fill_holes", True)),
                progress_callback=progress_callback,
            )
        else:
            result = vggt_backend.reconstruct(
                input_images=paths,
                output_dir=output_dir,
                entry=entry,
                export_formats=formats,
                conf_percentile=float(kwargs.get("conf_percentile", 50.0)),
                max_points=int(kwargs.get("max_points", 500_000)),
                poisson_depth=int(kwargs.get("poisson_depth", 9)),
                build_mesh=bool(kwargs.get("build_mesh", True)),
                remove_background=str(kwargs.get("remove_background", "auto")),
                bg_luma=float(kwargs.get("bg_luma", 0.62)),
                bg_sat=float(kwargs.get("bg_sat", 0.10)),
                progress_callback=progress_callback,
            )

        elapsed = time.perf_counter() - started

        meta_path = self._write_meta(
            output_dir=output_dir,
            input_paths=paths,
            model_type=model_type,
            entry=entry,
            formats=formats,
            elapsed=elapsed,
            result=result,
            settings=kwargs,
        )

        _log(self.log_prefix,
             f"Reconstruction complete in {elapsed:.1f}s → {output_dir}", "success")

        return {
            "success": True,
            "status": "success",
            "model": model_type,
            "type": kind,
            "outputs": result["outputs"],
            "input_count": result.get("input_count", len(paths)),
            "point_count": result.get("point_count", 0),
            "vertex_count": result.get("vertex_count", 0),
            "face_count": result.get("face_count", 0),
            "has_mesh": result.get("has_mesh", False),
            "peak_vram_gb": result.get("peak_vram_gb"),
            "inference_seconds": result.get("inference_seconds"),
            "elapsed": round(elapsed, 1),
            "output_dir": str(output_dir),
            "meta": str(meta_path),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_inputs(self, input_images: Union[str, List[str]]) -> List[str]:
        """Normalise to a sorted list of existing image paths.

        Directories expand to the images inside them, which is what makes the
        UI's "Use Dataset Builder output" shortcut a one-click action.
        """
        if isinstance(input_images, (str, Path)):
            input_images = [input_images]

        resolved: List[str] = []
        for item in input_images:
            path = Path(str(item)).expanduser()
            if path.is_dir():
                found = sorted(
                    p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
                )
                _log(self.log_prefix, f"Expanded {path.name}/ → {len(found)} images")
                resolved.extend(str(p) for p in found)
            elif path.exists():
                resolved.append(str(path))
            else:
                _log(self.log_prefix, f"Input not found, skipping: {path}", "warning")

        # De-duplicate while holding order — repeated picks are common in the
        # file browser and would otherwise weight those views twice.
        seen, unique = set(), []
        for p in resolved:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _free_vram_for_reconstruction(self):
        """Evict cached DiffSynth pipelines before a reconstruction runs."""
        import gc
        import torch

        cached = list(getattr(self.backend, "pipelines", {}).keys())
        if cached:
            _log(self.log_prefix, f"Evicting {len(cached)} cached pipeline(s) to free VRAM")
            for key in cached:
                self.backend._evict_pipeline(key)
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Reset unconditionally — otherwise the peak reported back is whatever
        # a previous generation hit, not this reconstruction's.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _write_meta(
        self,
        output_dir: Path,
        input_paths: List[str],
        model_type: str,
        entry: dict,
        formats: List[str],
        elapsed: float,
        result: dict,
        settings: dict,
    ) -> Path:
        """Record everything needed to reproduce or re-use the reconstruction.

        The input list is the point of this file — it lets the same image set
        be re-run against the other tier, which the docs call for when
        comparing TRELLIS and VGGT on synthetic views.
        """
        meta = {
            "timestamp": datetime.now().isoformat(),
            "model": model_type,
            "model_name": entry.get("name", model_type),
            "type": entry.get("type", "single_image"),
            "input_images": input_paths,
            "input_count": len(input_paths),
            "export_formats": formats,
            "outputs": {k: Path(v).name for k, v in result["outputs"].items()},
            "elapsed_seconds": round(elapsed, 2),
            "inference_seconds": result.get("inference_seconds"),
            "peak_vram_gb": result.get("peak_vram_gb"),
            "point_count": result.get("point_count", 0),
            "vertex_count": result.get("vertex_count", 0),
            "face_count": result.get("face_count", 0),
            "settings": {k: v for k, v in settings.items() if _jsonable(v)},
        }
        if "cameras" in result:
            meta["cameras"] = result["cameras"]
        if "normalization" in result:
            meta["normalization"] = result["normalization"]

        path = output_dir / "reconstruction_meta.json"
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
        return path


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))
