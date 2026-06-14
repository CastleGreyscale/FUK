"""
Storyboard endpoints — manifest CRUD over the active project folder.

All routes assume a project folder is set (via /api/project/set-folder).
The manifest is auto-created on first read so the frontend doesn't have to
distinguish "no storyboard" from "empty storyboard."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

import storyboard_manager as sm
from project_endpoints import get_project_folder


router = APIRouter(prefix="/api/storyboard", tags=["storyboard"])


def _require_folder() -> Path:
    folder = get_project_folder()
    if not folder:
        raise HTTPException(status_code=400, detail="No project folder set")
    return folder


def _load_or_create() -> Dict[str, Any]:
    folder = _require_folder()
    return sm.ensure_manifest(folder)


def _save(manifest: Dict[str, Any]) -> Dict[str, Any]:
    folder = _require_folder()
    sm.save_manifest(folder, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@router.get("")
async def get_manifest():
    return _load_or_create()


@router.put("")
async def put_manifest(payload: Dict[str, Any] = Body(...)):
    """Full replace. The frontend's autosave hook can use this for bulk writes."""
    folder = _require_folder()
    sm.save_manifest(folder, payload or {})
    return sm.load_manifest(folder)


@router.put("/specs")
async def put_specs(payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    m["specs"] = {**m.get("specs", {}), **(payload or {})}
    return _save(m)


# ---------------------------------------------------------------------------
# Globals — project tags (live `#marker` references) + mood + active LoRAs
# ---------------------------------------------------------------------------

@router.get("/globals")
async def get_globals():
    m = _load_or_create()
    return m.get(
        "globals",
        {"tags": [], "mood": "", "active_loras": [], "image_seed": None, "video_seed": None},
    )


@router.post("/globals/tags")
async def create_tag(payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    try:
        entry = sm.add_tag(m, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return entry


@router.put("/globals/tags/{tag_id}")
async def update_tag(tag_id: str, payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    try:
        entry = sm.update_tag(m, tag_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="Tag not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return entry


@router.delete("/globals/tags/{tag_id}")
async def delete_tag(tag_id: str):
    m = _load_or_create()
    if not sm.remove_tag(m, tag_id):
        raise HTTPException(status_code=404, detail="Tag not found")
    _save(m)
    return {"success": True}


@router.put("/globals/mood")
async def put_mood(payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    cleaned = sm.set_mood(m, payload.get("mood", ""))
    _save(m)
    return {"mood": cleaned}


@router.put("/globals/image_seed")
async def put_image_seed(payload: Dict[str, Any] = Body(...)):
    """Set the project-wide image seed inherited by new shots on first send-to-image.
    Pass `seed: null` to clear (shots fall back to per-shot defaults)."""
    m = _load_or_create()
    try:
        cleaned = sm.set_image_seed(m, payload.get("seed"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return {"image_seed": cleaned}


@router.put("/globals/video_seed")
async def put_video_seed(payload: Dict[str, Any] = Body(...)):
    """Set the project-wide video seed inherited by new shots on first send-to-video.
    Pass `seed: null` to clear."""
    m = _load_or_create()
    try:
        cleaned = sm.set_video_seed(m, payload.get("seed"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return {"video_seed": cleaned}


@router.put("/globals/loras")
async def put_active_loras(payload: Dict[str, Any] = Body(...)):
    """Replace the project-wide active LoRA list. Drives caption autocomplete
    across all storyboard textareas (and merges into the Image/Video tab
    autocomplete on top of whatever those tabs already pass)."""
    m = _load_or_create()
    cleaned = sm.set_active_loras(m, list(payload.get("loras") or []))
    _save(m)
    return {"active_loras": cleaned}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

@router.put("/panels/{shot_id}")
async def upsert_panel(shot_id: str, payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    try:
        entry = sm.upsert_panel(m, shot_id, payload or {})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return entry


@router.delete("/panels/{shot_id}")
async def delete_panel(shot_id: str):
    m = _load_or_create()
    if not sm.remove_panel(m, shot_id):
        raise HTTPException(status_code=404, detail="Panel not found")
    _save(m)
    return {"success": True}


@router.post("/snapshot")
async def create_snapshot():
    """Save a dated copy of the current manifest for storyboard versioning."""
    folder = _require_folder()
    filename = sm.snapshot_manifest(folder)
    return {"success": True, "filename": filename}


@router.get("/snapshots")
async def get_snapshots():
    """List existing storyboard snapshots, newest first."""
    folder = _require_folder()
    return {"snapshots": sm.list_snapshots(folder)}


@router.post("/snapshots/{filename}/restore")
async def restore_snapshot(filename: str):
    """Replace the live manifest with a snapshot file.

    The caller is responsible for auto-snapshotting current state first if
    they want an undo trail (the frontend does this before calling restore).
    Returns the full restored manifest.
    """
    folder = _require_folder()
    try:
        m = sm.restore_snapshot(folder, filename)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return m


@router.put("/sequence")
async def put_sequence(payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    new_seq: List[str] = list(payload.get("sequence") or [])
    cleaned = sm.reorder_sequence(m, new_seq)
    _save(m)
    return {"sequence": cleaned}


# ---------------------------------------------------------------------------
# Send-to-tab — push a panel's prompt draft down into the shot file.
# ---------------------------------------------------------------------------

def _latest_shot_file(folder: Path, shot_id: str) -> Optional[Path]:
    """Return the most-recent `*_shot{NN}_*.json` for this shot, or None.

    Storyboard panels can exist before a shot file does (the user might be
    blocking out the spine of a piece). In that case Send returns 404 and
    the frontend prompts the user to create the shot first.
    """
    candidates = sorted(
        folder.glob(f"*_shot{shot_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _write_prompt_to_tab(
    shot_path: Path,
    tab_name: str,
    prompt: str,
    target_model: Optional[str] = None,
    global_seed: Optional[int] = None,
) -> str:
    """Write `prompt` into the shot file's `tabs.{tab_name}.modelSettings[activeModel].prompt`.

    The raw markered draft goes in — resolution happens at generation time.
    If `target_model` is given, the tab's `activeModel` is switched to it so
    the prompt lands on the storyboard's chosen model.

    If `global_seed` is provided AND this model slot's seed is still untouched
    (no seed value yet), the global seed gets baked in and the mode is flipped
    to `fixed`. That gives new shots the storyboard's consistency seed without
    overwriting a seed the user has already customized.
    Returns the active model the prompt landed on, for the response payload.
    """
    with open(shot_path) as f:
        state = json.load(f)

    tabs = state.setdefault("tabs", {})
    tab = tabs.setdefault(tab_name, {})
    default_model = "qwen_image" if tab_name == "image" else "i2v-A14B"
    settings_map = tab.setdefault("modelSettings", {})
    # Priority: explicit target → whatever the tab already had → first present → default.
    active_model = (
        target_model
        or tab.get("activeModel")
        or next(iter(settings_map), default_model)
    )
    tab["activeModel"] = active_model
    model_settings = settings_map.setdefault(active_model, {})
    model_settings["prompt"] = prompt or ""

    # First-touch seed inheritance: only seed if the slot has never been set.
    # `seed: null` + missing/`random` mode is the "fresh" state SeedControl renders.
    if global_seed is not None and model_settings.get("seed") in (None, ""):
        model_settings["seed"] = int(global_seed)
        model_settings["seedMode"] = "fixed"

    with open(shot_path, "w") as f:
        json.dump(state, f, indent=2)
    return active_model


def _send_to_tab(
    shot_id: str, tab_name: str, prompt_field: str, model_field: str
) -> Dict[str, Any]:
    folder = _require_folder()
    manifest = _load_or_create()
    panel = (manifest.get("panels") or {}).get(shot_id)
    if not panel:
        raise HTTPException(status_code=404, detail=f"No panel for shot {shot_id}")
    shot_path = _latest_shot_file(folder, shot_id)
    if not shot_path:
        raise HTTPException(
            status_code=404,
            detail=f"No shot file exists for shot {shot_id}. Create the shot first.",
        )
    prompt = panel.get(prompt_field, "") or ""
    target_model = panel.get(model_field) or None
    global_seed = (
        sm.get_image_seed(manifest) if tab_name == "image" else sm.get_video_seed(manifest)
    )
    active_model = _write_prompt_to_tab(
        shot_path, tab_name, prompt, target_model, global_seed=global_seed
    )
    return {
        "success": True,
        "shot_file": shot_path.name,
        "tab": tab_name,
        "active_model": active_model,
        "prompt_chars": len(prompt),
    }


@router.post("/panels/{shot_id}/preview")
async def set_panel_preview(shot_id: str, payload: Dict[str, Any] = Body(...)):
    """Pin a generated image/video as the panel's preview thumbnail.

    Payload: `{kind: "image"|"video", path: "..."}`. `path` may be `null` or
    empty to clear the slot.
    """
    m = _load_or_create()
    try:
        entry = sm.set_panel_preview(
            m,
            shot_id,
            (payload or {}).get("kind", ""),
            (payload or {}).get("path"),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return entry


@router.post("/panels/{shot_id}/send/image")
async def send_panel_to_image(shot_id: str):
    """Write `panel.imagery_prompt` into the shot's image-tab positive prompt."""
    return _send_to_tab(shot_id, "image", "imagery_prompt", "image_model")


@router.post("/panels/{shot_id}/send/video")
async def send_panel_to_video(shot_id: str):
    """Write `panel.action_prompt` into the shot's video-tab positive prompt."""
    return _send_to_tab(shot_id, "video", "action_prompt", "video_model")


def setup_storyboard_routes(app):
    app.include_router(router)
