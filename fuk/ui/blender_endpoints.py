"""
Blender bridge endpoints.

The Blender addon renders control passes and generates through the normal
/api/generate/image path (which always creates an img_gen entry). These routes
let the addon turn a render into a *complete, native-looking* history entry:

    POST /api/blender/save-entry   enrich an existing entry, or create a new one,
                                    with the control map + Blender beauty as source,
                                    and register the control map as a preprocess
                                    entry so it shows in FUK's control/history panels.
    GET  /api/blender/signal       a cheap version counter the web UI polls so it
                                    auto-refreshes when Blender saves something.

Same-machine assumption: paths in the request are absolute local files the server
can read directly (no upload).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from project_endpoints import (
    get_cache_root,
    get_generation_output_dir,
    get_project_relative_url,
)

router = APIRouter(prefix="/api/blender", tags=["blender"])

# Bumped on every save so the web UI's poller knows to refetch history.
_VERSION = {"n": 0}


def bump_version() -> int:
    _VERSION["n"] += 1
    return _VERSION["n"]


class SaveEntryRequest(BaseModel):
    # Enrich this existing entry (cache-relative path, e.g. "proj_shot01_v/img_gen_005").
    # If absent, a new img_gen entry is created from result_path.
    generation_id: Optional[str] = None
    control_path: Optional[str] = None    # absolute path to the control map
    beauty_path: Optional[str] = None     # absolute path to the Blender beauty render
    result_path: Optional[str] = None     # absolute path to the generated image (create mode)
    control_kind: str = "control"         # depth | normals | openpose | canny
    register_control: bool = True         # also create a preprocess entry for the control
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    seed: Optional[int] = None
    width: int = 0
    height: int = 0


def _thumbnail(src: Path, dest: Path) -> None:
    try:
        from PIL import Image
        with Image.open(src) as im:
            im.thumbnail((400, 400), Image.Resampling.LANCZOS)
            im.convert("RGB").save(dest, "JPEG", quality=85)
    except Exception:
        pass


@router.get("/signal")
async def signal():
    """Cheap version counter; increments whenever Blender saves an entry."""
    return {"version": _VERSION["n"]}


@router.post("/save-entry")
async def save_entry(req: SaveEntryRequest):
    cache_root = get_cache_root()
    if not cache_root:
        raise HTTPException(status_code=400, detail="No project loaded")
    cache_root = Path(cache_root)

    # --- locate (enrich) or create the generation entry ---
    if req.generation_id:
        gen_dir = cache_root / req.generation_id
        try:
            gen_dir.resolve().relative_to(cache_root.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid generation_id")
        if not gen_dir.exists():
            raise HTTPException(status_code=404, detail=f"Generation not found: {req.generation_id}")
    else:
        if not req.result_path or not Path(req.result_path).exists():
            raise HTTPException(status_code=400, detail="result_path required to create a new entry")
        gen_dir = get_generation_output_dir("img_gen")
        shutil.copy(req.result_path, gen_dir / "generated.png")

    # --- copy the control map + beauty into the entry (self-contained) ---
    control_rel = None
    if req.control_path and Path(req.control_path).exists():
        shutil.copy(req.control_path, gen_dir / "control.png")
        control_rel = get_project_relative_url(gen_dir / "control.png")
    source_rel = None
    if req.beauty_path and Path(req.beauty_path).exists():
        shutil.copy(req.beauty_path, gen_dir / "source.png")
        source_rel = get_project_relative_url(gen_dir / "source.png")

    # --- write / merge metadata ---
    meta_path = gen_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta.setdefault("timestamp", datetime.now().isoformat())
    if req.prompt:
        meta["prompt"] = req.prompt
    if req.negative_prompt:
        meta["negative_prompt"] = req.negative_prompt
    if req.model:
        meta["model"] = req.model
    if req.seed is not None:
        meta["seed"] = req.seed
    if req.width and req.height:
        meta["image_size"] = [req.width, req.height]
    if control_rel:
        meta["control_image"] = control_rel
    if source_rel:
        meta["source"] = source_rel
    meta["source_app"] = "blender"
    meta["blender"] = {"control_kind": req.control_kind}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- register the control map as a preprocess entry (shows in control panel) ---
    control_entry = None
    if req.register_control and req.control_path and Path(req.control_path).exists():
        pdir = get_generation_output_dir("preprocess")
        shutil.copy(req.control_path, pdir / "processed.png")
        if req.beauty_path and Path(req.beauty_path).exists():
            shutil.copy(req.beauty_path, pdir / "source.png")
        with open(pdir / "metadata.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "prompt": f"Blender {req.control_kind}",
                "model": req.control_kind,
                "method": req.control_kind,
                "seed": None,
                "image_size": [req.width, req.height],
                "source_image": str(req.beauty_path or ""),
                "source_app": "blender",
            }, f, indent=2)
        _thumbnail(pdir / "processed.png", pdir / "thumbnail.jpg")
        try:
            control_entry = str(pdir.relative_to(cache_root))
        except ValueError:
            control_entry = pdir.name

    version = bump_version()
    try:
        gen_rel = str(gen_dir.relative_to(cache_root))
    except ValueError:
        gen_rel = gen_dir.name

    return {
        "success": True,
        "generation_id": gen_rel,
        "control_entry": control_entry,
        "version": version,
    }


def setup_blender_routes(app):
    app.include_router(router)
