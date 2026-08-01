"""
3D Reconstruction API Endpoints

Images in, proxy geometry out. Two tiers behind one endpoint: TRELLIS for a
single image, VGGT for a multi-view set. The endpoint takes an arbitrary list
of image paths, which is what lets the LoRA Dataset Builder's orbital views,
a hand-picked selection, and (later) the Pi capture rig all feed the same
route with no changes.

Wire up in fuk_web_server.py:
    from threed_endpoints import setup_threed_routes
    setup_threed_routes(app, generation_backend=..., ...)

See docs/3D_RECONSTRUCTION_SYSTEM.md
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel

# Below this VGGT's angular coverage gets thin — the UI surfaces it as a
# warning rather than a block, since sparse sets are still useful as proxies.
MIN_RECOMMENDED_VIEWS = 6

_ctx: Dict[str, Any] = {}


# ============================================================================
# Request models
# ============================================================================

class ThreeDReconstructRequest(BaseModel):
    input_images: List[str]
    model: str = "vggt"
    export_formats: List[str] = ["glb"]
    output_path: Optional[str] = None          # defaults to a new project cache dir

    # VGGT tuning
    conf_percentile: float = 50.0              # keep the top N% most confident points
    max_points: int = 500_000
    poisson_depth: int = 9
    build_mesh: bool = True
    # VGGT reconstructs the whole scene, backdrop included, and an untextured
    # studio background comes back as invented geometry wrapped around the
    # subject. "auto" keys it out by brightness/saturation, "alpha" uses the
    # source images' matte, "none" keeps everything (correct for scenes).
    remove_background: str = "auto"            # auto | alpha | none
    bg_luma: float = 0.62
    bg_sat: float = 0.10

    # TRELLIS tuning
    seed: int = 42
    steps: int = 12
    cfg_strength: float = 7.5
    simplify: float = 0.95
    texture_size: int = 1024
    # Hole filling renders the mesh from 1000 viewpoints; on a dense SLAT
    # mesh it dominates runtime. Off is the escape hatch for a slow run.
    fill_holes: bool = True

    vram_preset: Optional[str] = None


# ============================================================================
# Core handler — shared by the route and the task dispatcher
# ============================================================================

async def run_reconstruction(
    request: ThreeDReconstructRequest,
    generation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a reconstruction and return the standard generation result shape."""
    backend = _ctx["generation_backend"]
    log = _ctx["log"]
    resolve_input_path = _ctx["resolve_input_path"]
    get_generation_output_dir = _ctx["get_generation_output_dir"]
    get_project_relative_url = _ctx["get_project_relative_url"]
    active_generations = _ctx["active_generations"]
    clear_vram = _ctx["clear_vram"]

    runner = backend.runners.get("threed")
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="3D reconstruction runner not loaded — check the server log for import errors",
        )

    if not request.input_images:
        raise HTTPException(status_code=400, detail="No input images provided")

    # UI paths arrive as api/project/cache/… URLs; the runner needs real ones.
    # Directories are passed through untouched — the runner expands them.
    resolved: List[str] = []
    for raw in request.input_images:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            resolved.append(str(candidate))
            continue
        path = resolve_input_path(raw)
        if path and Path(path).exists():
            resolved.append(str(path))
        else:
            log.warning("ThreeD", f"Input not found, skipping: {raw}")

    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="None of the supplied input images could be resolved on disk",
        )

    try:
        model_type = backend.resolve_model_type(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entry = backend.get_model_entry(model_type)
    if entry.get("pipeline") != "threed":
        raise HTTPException(
            status_code=400, detail=f"'{request.model}' is not a 3D reconstruction model"
        )

    availability = runner.list_available().get(model_type, {})
    if not availability.get("available", False):
        missing = ", ".join(availability.get("missing", ["unknown"]))
        hint = availability.get("hint")
        raise HTTPException(
            status_code=503,
            detail=f"{entry.get('name', model_type)} is not installed — missing: {missing}"
                   + (f"\n{hint}" if hint else ""),
        )

    if request.output_path:
        output_dir = Path(request.output_path).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_generation_output_dir(f"threed_{model_type}")

    # Progress arrives from the backends as (fraction, label); the SSE stream
    # and GenerationModal expect the active_generations shape.
    def progress_callback(fraction: float, label: str):
        if generation_id and generation_id in active_generations:
            active_generations[generation_id].update({
                "phase": label,
                "progress": float(fraction),
                "updated_at": datetime.now().isoformat(),
            })

    log.info(
        "ThreeD",
        f"Reconstructing with {model_type}: {len(resolved)} input(s) → {output_dir.name}",
    )

    try:
        result = await asyncio.to_thread(
            runner.generate,
            input_images=resolved,
            output_path=output_dir,
            model=model_type,
            export_formats=request.export_formats,
            vram_preset=request.vram_preset,
            progress_callback=progress_callback,
            conf_percentile=request.conf_percentile,
            max_points=request.max_points,
            poisson_depth=request.poisson_depth,
            build_mesh=request.build_mesh,
            remove_background=request.remove_background,
            bg_luma=request.bg_luma,
            bg_sat=request.bg_sat,
            seed=request.seed,
            steps=request.steps,
            cfg_strength=request.cfg_strength,
            simplify=request.simplify,
            texture_size=request.texture_size,
            fill_holes=request.fill_holes,
        )
    except Exception as e:
        clear_vram(full=True)
        log.error("ThreeD", f"Reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # These models peak well above their resting footprint — hand the memory
    # back so the next image or video generation isn't starved.
    clear_vram(full=True)

    # URLs for the viewer and the download buttons.
    urls = {
        fmt: get_project_relative_url(Path(path))
        for fmt, path in result["outputs"].items()
    }
    result["urls"] = urls
    result["output_dir_url"] = get_project_relative_url(output_dir)

    log.info(
        "ThreeD",
        f"Done in {result['elapsed']}s — " + ", ".join(f"{k}: {Path(v).name}"
                                                       for k, v in result["outputs"].items()),
    )
    return result


# ============================================================================
# Route registration
# ============================================================================

def setup_threed_routes(
    app,
    *,
    generation_backend,
    resolve_input_path: Callable[[str], Optional[Path]],
    get_generation_output_dir: Callable[[str], Path],
    get_project_relative_url: Callable[[Path], str],
    active_generations: dict,
    clear_vram: Callable[..., None],
    datasets_root: Path,
    log,
):
    """Register /api/threed/* routes."""
    _ctx.update(
        generation_backend=generation_backend,
        resolve_input_path=resolve_input_path,
        get_generation_output_dir=get_generation_output_dir,
        get_project_relative_url=get_project_relative_url,
        active_generations=active_generations,
        clear_vram=clear_vram,
        datasets_root=Path(datasets_root),
        log=log,
    )

    @app.post("/api/threed/reconstruct")
    async def threed_reconstruct(request: ThreeDReconstructRequest):
        """Reconstruct geometry from one or more images.

        Prefer POST /api/task/start with task_type "threed_reconstruct" from
        the UI — that route streams console output into GenerationModal. This
        one runs synchronously and is handy for scripting and curl.
        """
        return await run_reconstruction(request)

    @app.get("/api/threed/models")
    async def threed_models():
        """Available reconstruction models and whether each can actually run."""
        runner = generation_backend.runners.get("threed")
        if runner is None:
            return {"models": {}, "error": "3D runner not loaded"}
        return {
            "models": runner.list_available(),
            "min_recommended_views": MIN_RECOMMENDED_VIEWS,
        }

    @app.get("/api/threed/datasets")
    async def threed_datasets():
        """LoRA Dataset Builder jobs that can be used as reconstruction input.

        Orbital views from the builder are the intended synthetic-to-3D path,
        so the UI offers them directly rather than making the user hunt through
        a file browser. Approved images win when present — they are the set the
        user already vetted; otherwise every generated view is offered.
        """
        root = _ctx["datasets_root"]
        if not root.exists():
            return {"datasets": []}

        image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
        datasets = []

        for job_dir in sorted(root.iterdir(), reverse=True):
            manifest_path = job_dir / "dataset_manifest.json"
            if not job_dir.is_dir() or not manifest_path.exists():
                continue
            try:
                with open(manifest_path) as fh:
                    manifest = json.load(fh)
            except Exception:
                continue

            approved_dir = job_dir / "approved"
            approved = sorted(
                p for p in approved_dir.glob("*") if p.suffix.lower() in image_suffixes
            ) if approved_dir.exists() else []

            generated = sorted(
                p for p in (job_dir / "generated").glob("*/generated.png")
            ) if (job_dir / "generated").exists() else []

            images = approved or generated
            if not images:
                continue

            datasets.append({
                "job_id": job_dir.name,
                "subject_name": manifest.get("subject_name", job_dir.name),
                "subject_type": manifest.get("subject_type", ""),
                "created": manifest.get("created", ""),
                "source": "approved" if approved else "generated",
                "image_count": len(images),
                "approved_count": len(approved),
                "generated_count": len(generated),
                "directory": str(approved_dir if approved else job_dir / "generated"),
                "images": [str(p) for p in images],
            })

        return {"datasets": datasets}

    log.info("ThreeD", "3D reconstruction routes registered")
