"""
Storyboard manager — manifest load/save and globals catalog.

The storyboard manifest is a project-level file (`{projectname}_storyboard.json`)
that sits alongside the per-shot project files. It stores:

  - specs (TRT, aspect ratio, resolution)
  - globals.subjects: named `#marker` references (characters, props, locations)
  - globals.mood: a single freeform sentence applied to every shot at generation
  - sequence: ordered shot IDs
  - panels: per-shot metadata (imagery prompt, action prompt, duration, ...)

Globals are *live*: editing a subject's description changes how `#sarah`
resolves the next time any shot is generated. Markers are NOT expanded at
"send to tab" time — the raw `#sarah is doing X` is what's saved to the shot
file, and resolution happens server-side just before generation.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


MANIFEST_VERSION = "1.0"

# Mirror the marker rules used in llm_endpoints._name_to_marker so a subject's
# `#marker` and a tag's `#marker` follow identical slugging.
_MARKER_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-]+")
_SUBJECT_KINDS = ("character", "prop", "location", "other")

# Same shape as project_endpoints._SHOT_ID_RE — shot ids are part of the
# filename `{project}_shot{id}_{version}.json` so the regex MUST match.
_SHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,31}$")


def _slug_marker(name: str) -> str:
    slug = _MARKER_CHAR_RE.sub("_", (name or "").strip()).strip("_")
    return f"#{slug}" if slug else ""


def _now() -> str:
    return datetime.now().isoformat()


def manifest_filename(project_name: str) -> str:
    return f"{project_name}_storyboard.json"


def find_manifest(project_folder: Path) -> Optional[Path]:
    """Return the first *_storyboard.json in the folder, or None."""
    if not project_folder or not project_folder.exists():
        return None
    for p in sorted(project_folder.glob("*_storyboard.json")):
        return p
    return None


def project_name_from_folder(project_folder: Path) -> Optional[str]:
    """Infer the project name from any existing shot file in the folder.

    Shot files follow `{projectname}_shotNN_VERSION.json`. We look for the most
    recent one and split off the suffix.
    """
    if not project_folder or not project_folder.exists():
        return None
    for f in sorted(
        project_folder.glob("*_shot*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        name = f.stem
        idx = name.find("_shot")
        if idx > 0:
            return name[:idx]
    return None


def empty_manifest() -> Dict[str, Any]:
    return {
        "meta": {
            "version": MANIFEST_VERSION,
            "createdAt": _now(),
            "updatedAt": _now(),
        },
        "specs": {
            "trt_seconds": None,
            "aspect_ratio": "16:9",
            "resolution": [1920, 1080],
        },
        "globals": {
            "subjects": [],     # [{id, name, marker, description, kind}]
            "mood": "",         # freeform style+environment sentence
        },
        "sequence": [],
        "panels": {},
    }


def load_manifest(project_folder: Path) -> Optional[Dict[str, Any]]:
    """Load the storyboard manifest for this project folder, or None if absent.

    A missing file is not an error — the storyboard tab can prompt the user
    to create one. A malformed file *is* an error and surfaces to the caller.
    """
    if not project_folder:
        return None
    path = find_manifest(project_folder)
    if not path or not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return _migrate(data)


def save_manifest(project_folder: Path, manifest: Dict[str, Any]) -> Path:
    """Persist the manifest. Filename is derived from the project name."""
    if not project_folder:
        raise ValueError("No project folder set")
    project_name = project_name_from_folder(project_folder) or "untitled"
    existing = find_manifest(project_folder)
    path = existing or (project_folder / manifest_filename(project_name))

    manifest = dict(manifest)
    manifest.setdefault("meta", {})
    manifest["meta"] = {
        **manifest["meta"],
        "version": MANIFEST_VERSION,
        "updatedAt": _now(),
    }
    manifest["meta"].setdefault("createdAt", manifest["meta"]["updatedAt"])

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def ensure_manifest(project_folder: Path) -> Dict[str, Any]:
    """Load existing manifest or create an empty one on disk."""
    m = load_manifest(project_folder)
    if m is not None:
        return m
    m = empty_manifest()
    save_manifest(project_folder, m)
    return m


def _migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Forward-compat shim. Fill in any missing keys with empty defaults."""
    base = empty_manifest()
    out = {**base, **(data or {})}
    out["meta"] = {**base["meta"], **(data.get("meta") or {})}
    out["specs"] = {**base["specs"], **(data.get("specs") or {})}
    globals_in = data.get("globals") or {}
    out["globals"] = {
        "subjects": list(globals_in.get("subjects") or []),
        "mood": str(globals_in.get("mood") or ""),
    }
    out["sequence"] = list(data.get("sequence") or [])
    out["panels"] = dict(data.get("panels") or {})
    return out


# ---------------------------------------------------------------------------
# Globals: subjects (live `#marker` references)
# ---------------------------------------------------------------------------

def _normalize_subject_name(name: str) -> str:
    return " ".join((name or "").strip().split()).lower()


def validate_subject(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _normalize_subject_name(payload.get("name", ""))
    description = (payload.get("description") or "").strip()
    kind = (payload.get("kind") or "character").strip().lower()
    if not name:
        raise ValueError("Subject name is required")
    if not description:
        raise ValueError("Subject description is required")
    if kind not in _SUBJECT_KINDS:
        raise ValueError(f"Unknown subject kind '{kind}'. Allowed: {', '.join(_SUBJECT_KINDS)}")
    marker = _slug_marker(name)
    if not marker:
        raise ValueError("Subject name cannot be slugged into a marker")
    return {"name": name, "description": description, "kind": kind, "marker": marker}


def add_subject(manifest: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = validate_subject(payload)
    subjects = manifest["globals"].setdefault("subjects", [])
    if any(s.get("marker") == clean["marker"] for s in subjects):
        raise ValueError(f"Subject with marker {clean['marker']} already exists")
    entry = {"id": uuid.uuid4().hex, **clean}
    subjects.append(entry)
    return entry


def update_subject(manifest: Dict[str, Any], subject_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = validate_subject(payload)
    subjects = manifest["globals"].setdefault("subjects", [])
    for i, s in enumerate(subjects):
        if s.get("id") == subject_id:
            if any(o.get("marker") == clean["marker"] and o.get("id") != subject_id for o in subjects):
                raise ValueError(f"Subject with marker {clean['marker']} already exists")
            subjects[i] = {"id": subject_id, **clean}
            return subjects[i]
    raise KeyError(f"Subject {subject_id} not found")


def remove_subject(manifest: Dict[str, Any], subject_id: str) -> bool:
    subjects = manifest["globals"].setdefault("subjects", [])
    before = len(subjects)
    manifest["globals"]["subjects"] = [s for s in subjects if s.get("id") != subject_id]
    return len(manifest["globals"]["subjects"]) < before


def set_mood(manifest: Dict[str, Any], mood: str) -> str:
    cleaned = (mood or "").strip()
    manifest["globals"]["mood"] = cleaned
    return cleaned


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _default_panel(shot_id: str) -> Dict[str, Any]:
    return {
        "shot_id": shot_id,
        "thumbnail_ref": None,
        "imagery_prompt": "",
        "action_prompt": "",
        "duration_seconds": None,
        "notes": "",
    }


def upsert_panel(manifest: Dict[str, Any], shot_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    if not shot_id:
        raise ValueError("shot_id is required")
    if not _SHOT_ID_RE.match(shot_id):
        raise ValueError(
            "Shot id: 1-32 chars, alphanumeric or hyphen, must start alphanumeric. No underscores."
        )
    panels = manifest.setdefault("panels", {})
    sequence = manifest.setdefault("sequence", [])
    existing = panels.get(shot_id) or _default_panel(shot_id)
    merged = {**existing, **(patch or {}), "shot_id": shot_id}
    panels[shot_id] = merged
    if shot_id not in sequence:
        sequence.append(shot_id)
    return merged


def remove_panel(manifest: Dict[str, Any], shot_id: str) -> bool:
    panels = manifest.setdefault("panels", {})
    if shot_id not in panels:
        return False
    del panels[shot_id]
    manifest["sequence"] = [s for s in manifest.get("sequence", []) if s != shot_id]
    return True


def reorder_sequence(manifest: Dict[str, Any], new_sequence: List[str]) -> List[str]:
    panels = manifest.get("panels", {})
    cleaned = [s for s in (new_sequence or []) if s in panels]
    # Preserve any panels missing from the supplied order so we don't silently drop them.
    for shot_id in panels:
        if shot_id not in cleaned:
            cleaned.append(shot_id)
    manifest["sequence"] = cleaned
    return cleaned


# ---------------------------------------------------------------------------
# Token surface — subjects rendered as prompt tokens for /api/prompt/tokens
# ---------------------------------------------------------------------------

def subject_tokens(manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render globals.subjects as unified prompt tokens.

    These tokens take precedence over workspace tags on marker collision —
    see llm_endpoints._collate_tokens.
    """
    if not manifest:
        return []
    out: List[Dict[str, Any]] = []
    for s in manifest.get("globals", {}).get("subjects", []) or []:
        marker = s.get("marker") or _slug_marker(s.get("name", ""))
        name = s.get("name", "").strip()
        desc = (s.get("description") or "").strip()
        if not marker or not name or not desc:
            continue
        out.append({
            "id": f"global:{s.get('id') or marker}",
            "source": "global",
            "name": name,
            "marker": marker,
            "expansion": desc,
            "category": "subject",
            "kind": s.get("kind") or "character",
        })
    return out


def get_mood(manifest: Optional[Dict[str, Any]]) -> str:
    if not manifest:
        return ""
    return (manifest.get("globals", {}).get("mood") or "").strip()
