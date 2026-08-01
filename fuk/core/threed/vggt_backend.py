"""
VGGT multi-view reconstruction (Tier 2).

VGGT is a single feed-forward transformer: one to hundreds of images in,
camera parameters + dense depth + a 3D point cloud out, in one pass. No
pairwise matching, no global bundle adjustment.

Runs in FUK's main venv — it is plain PyTorch with no CUDA extensions, so
unlike TRELLIS it needs no isolation.

Source: https://github.com/facebookresearch/vggt   (Meta, CVPR 2025)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from . import mesh_export
from .mesh_export import _log

# VGGT resizes every input so its longest side is 518px. This is fixed by the
# model's patch grid, not a quality setting.
MAX_INPUT_DIM = 518

_VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor"
_VGGT_DIR = _VENDOR_DIR / "VGGT"

_model = None
_model_path: Optional[str] = None


def vendor_available() -> bool:
    return (_VGGT_DIR / "vggt" / "models" / "vggt.py").exists()


def _ensure_vendor_on_path():
    if not vendor_available():
        raise RuntimeError(
            f"VGGT source not found at {_VGGT_DIR}\n"
            f"  git clone https://github.com/facebookresearch/vggt.git {_VGGT_DIR}"
        )
    if str(_VGGT_DIR) not in sys.path:
        sys.path.insert(0, str(_VGGT_DIR))


def resolve_weights(entry: dict) -> str:
    """Local weights directory if present, else the HuggingFace repo id."""
    configured = entry.get("path")
    if configured:
        expanded = Path(configured).expanduser()
        if expanded.exists():
            return str(expanded)
    return entry.get("model_id", "facebook/VGGT-1B")


def load_model(entry: dict, device: str = "cuda"):
    """Load VGGT, reusing the cached instance when the weights haven't changed."""
    global _model, _model_path


    _ensure_vendor_on_path()
    from vggt.models.vggt import VGGT

    weights = resolve_weights(entry)
    if _model is not None and _model_path == weights:
        return _model

    unload()
    _log(f"Loading VGGT from {weights}")
    t0 = time.perf_counter()
    model = VGGT.from_pretrained(weights).to(device).eval()
    _log(f"VGGT loaded in {time.perf_counter() - t0:.1f}s", "success")

    _model, _model_path = model, weights
    return model


def unload():
    """Drop the cached model and return its VRAM."""
    global _model, _model_path
    if _model is None:
        return
    import torch

    del _model
    _model, _model_path = None, None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _log("VGGT unloaded")


def reconstruct(
    input_images: List[str],
    output_dir: Path,
    entry: dict,
    export_formats: List[str],
    conf_percentile: float = 50.0,
    max_points: int = 500_000,
    poisson_depth: int = 9,
    build_mesh: bool = True,
    remove_background: str = "auto",   # auto | alpha | none
    bg_luma: float = 0.62,
    bg_sat: float = 0.10,
    keep_loaded: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run a multi-view reconstruction.

    Source images are never modified — VGGT's loader resizes copies in memory.
    """
    import torch

    _ensure_vendor_on_path()
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [str(p) for p in input_images if Path(p).exists()]
    if not paths:
        raise ValueError("No readable input images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 on Ampere and later, fp16 below — matches VGGT's own demo.
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    if progress_callback:
        progress_callback(0.05, f"Loading VGGT ({len(paths)} images)")
    model = load_model(entry, device)

    if progress_callback:
        progress_callback(0.20, f"Preprocessing {len(paths)} images → {MAX_INPUT_DIM}px")
    _log(f"Preprocessing {len(paths)} images (resized to {MAX_INPUT_DIM}px for inference)")
    images = load_and_preprocess_images(paths).to(device)

    if progress_callback:
        progress_callback(0.35, "Running reconstruction")
    _log(f"Reconstructing from {len(paths)} views…")

    t0 = time.perf_counter()
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    if device == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - t0
    _log(f"Inference: {inference_time:.2f}s", "success")

    peak_vram = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None
    )

    if progress_callback:
        progress_callback(0.55, "Extracting geometry")

    # Camera solve — kept in the metadata now, exposed as usable data in Phase 3.
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    points = predictions["world_points"][0].float().cpu().numpy()
    conf = predictions["world_points_conf"][0].float().cpu().numpy()
    # Per-pixel colours come straight from the (already resized) input images,
    # so they align with world_points without re-deriving VGGT's crop geometry.
    colors = predictions["images"][0].float().cpu().numpy().transpose(0, 2, 3, 1)
    image_shape = tuple(images.shape[-2:])

    cameras = {
        "extrinsics": extrinsics[0].float().cpu().numpy().tolist(),
        "intrinsics": intrinsics[0].float().cpu().numpy().tolist(),
        "convention": "world-to-camera 3x4 [R|t], OpenCV intrinsics at 518px",
    }

    del predictions, images
    if not keep_loaded:
        unload()
    elif device == "cuda":
        torch.cuda.empty_cache()

    if progress_callback:
        progress_callback(0.62, "Filtering point cloud")

    raw_count = points.reshape(-1, 3).shape[0]

    # Background rejection has to happen before anything else: backdrop points
    # otherwise set the bounding box, survive the confidence cut (VGGT is
    # confident about smooth backdrops), and give Poisson a shell to close
    # over. See mesh_export.neutral_background_mask for why.
    foreground = None
    mode = (remove_background or "none").lower()

    if mode == "alpha":
        foreground = mesh_export.alpha_masks_for(paths, image_shape)
        if foreground is None:
            _log("No usable alpha in the inputs — falling back to backdrop keying", "warning")
            mode = "auto"

    if mode == "auto":
        foreground = mesh_export.neutral_background_mask(colors, bg_luma, bg_sat)

    if foreground is not None:
        foreground = foreground.reshape(-1)
        kept = int(foreground.sum())
        # A dark subject, a non-neutral backdrop, or a genuine scene capture
        # can all make the mask meaningless. Better a noisy cloud than none.
        if kept < 1000:
            _log(
                f"Background mask would keep only {kept:,} points — ignoring it. "
                f"Set background removal to 'none' if this is a scene, not an object.",
                "warning",
            )
        else:
            points = points.reshape(-1, 3)[foreground]
            colors = colors.reshape(-1, 3)[foreground]
            conf = conf.reshape(-1)[foreground]
            _log(
                f"Background removed ({mode}): {raw_count:,} → {kept:,} foreground points",
                "success",
            )

    points, colors = mesh_export.filter_by_confidence(points, colors, conf, conf_percentile)
    points, colors = mesh_export.trim_outliers(points, colors)
    points, transform = mesh_export.normalize_to_unit_cube(points)
    points, colors = mesh_export.voxel_downsample(points, colors, max_points)
    _log(f"Point cloud: {raw_count:,} → {len(points):,} after filtering")

    if len(points) == 0:
        raise RuntimeError("Reconstruction produced no usable points")

    outputs: Dict[str, str] = {}
    formats = [f.lower() for f in export_formats]

    if "ply" in formats:
        if progress_callback:
            progress_callback(0.68, "Writing point cloud")
        outputs["ply"] = str(mesh_export.write_ply(points, colors, output_dir / "pointcloud.ply"))

    mesh = None
    if build_mesh and any(f in formats for f in ("glb", "obj")):
        try:
            mesh = mesh_export.pointcloud_to_mesh(
                points, colors, depth=poisson_depth, progress_callback=progress_callback
            )
        except Exception as e:
            _log(f"Meshing failed ({type(e).__name__}: {e}) — exporting point cloud only", "warning")

    if progress_callback:
        progress_callback(0.92, "Writing outputs")

    if "glb" in formats:
        if mesh is not None:
            outputs["glb"] = str(mesh_export.write_glb(mesh, output_dir / "mesh.glb"))
        else:
            # Still give the user something to orbit — a point-cloud GLB.
            outputs["glb"] = str(
                mesh_export.pointcloud_as_glb(points, colors, output_dir / "mesh.glb")
            )
            _log("No mesh — GLB contains the point cloud instead", "warning")

    if "obj" in formats and mesh is not None:
        outputs["obj"] = str(mesh_export.write_obj(mesh, output_dir / "mesh.obj"))

    return {
        "outputs": outputs,
        "input_count": len(paths),
        "point_count": int(len(points)),
        "vertex_count": int(len(mesh.vertices)) if mesh is not None else 0,
        "face_count": int(len(mesh.faces)) if mesh is not None else 0,
        "has_mesh": mesh is not None,
        "inference_seconds": round(inference_time, 2),
        "peak_vram_gb": peak_vram,
        "cameras": cameras,
        "normalization": transform,
    }
