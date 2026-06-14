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
# Globals — subjects (live `#marker` references)
# ---------------------------------------------------------------------------

@router.get("/globals")
async def get_globals():
    m = _load_or_create()
    return m.get("globals", {"subjects": [], "mood": ""})


@router.post("/globals/subjects")
async def create_subject(payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    try:
        entry = sm.add_subject(m, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return entry


@router.put("/globals/subjects/{subject_id}")
async def update_subject(subject_id: str, payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    try:
        entry = sm.update_subject(m, subject_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="Subject not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save(m)
    return entry


@router.delete("/globals/subjects/{subject_id}")
async def delete_subject(subject_id: str):
    m = _load_or_create()
    if not sm.remove_subject(m, subject_id):
        raise HTTPException(status_code=404, detail="Subject not found")
    _save(m)
    return {"success": True}


@router.put("/globals/mood")
async def put_mood(payload: Dict[str, Any] = Body(...)):
    m = _load_or_create()
    cleaned = sm.set_mood(m, payload.get("mood", ""))
    _save(m)
    return {"mood": cleaned}


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


def _write_prompt_to_tab(shot_path: Path, tab_name: str, prompt: str) -> str:
    """Write `prompt` into the shot file's `tabs.{tab_name}.modelSettings[activeModel].prompt`.

    The raw markered draft goes in — resolution happens at generation time.
    Returns the active model the prompt landed on, for the response payload.
    """
    with open(shot_path) as f:
        state = json.load(f)

    tabs = state.setdefault("tabs", {})
    tab = tabs.setdefault(tab_name, {})
    # `activeModel` may be unset on freshly created files; fall back to the
    # first model with settings, then to a known default per tab.
    default_model = "qwen_image" if tab_name == "image" else "i2v-A14B"
    settings_map = tab.setdefault("modelSettings", {})
    active_model = tab.get("activeModel") or next(iter(settings_map), default_model)
    tab["activeModel"] = active_model
    model_settings = settings_map.setdefault(active_model, {})
    model_settings["prompt"] = prompt or ""

    with open(shot_path, "w") as f:
        json.dump(state, f, indent=2)
    return active_model


def _send_to_tab(shot_id: str, tab_name: str, prompt_field: str) -> Dict[str, Any]:
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
    active_model = _write_prompt_to_tab(shot_path, tab_name, prompt)
    return {
        "success": True,
        "shot_file": shot_path.name,
        "tab": tab_name,
        "active_model": active_model,
        "prompt_chars": len(prompt),
    }


@router.post("/panels/{shot_id}/send/image")
async def send_panel_to_image(shot_id: str):
    """Write `panel.imagery_prompt` into the shot's image-tab positive prompt."""
    return _send_to_tab(shot_id, "image", "imagery_prompt")


@router.post("/panels/{shot_id}/send/video")
async def send_panel_to_video(shot_id: str):
    """Write `panel.action_prompt` into the shot's video-tab positive prompt."""
    return _send_to_tab(shot_id, "video", "action_prompt")


def setup_storyboard_routes(app):
    app.include_router(router)
