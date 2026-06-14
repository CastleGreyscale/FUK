"""
Storyboard manager — manifest load/save and globals catalog.

The storyboard manifest is a project-level file (`{projectname}_storyboard.json`)
that sits alongside the per-shot project files. It stores:

  - specs (TRT, aspect ratio, resolution)
  - globals.tags: project-local `#marker` shorthand (characters, props, etc.)
  - globals.mood: a single freeform sentence applied to every shot at generation
  - globals.active_loras: project-wide active LoRA keys
  - sequence: ordered shot IDs
  - panels: per-shot metadata (imagery prompt, action prompt, duration, ...)

Globals are *live*: editing a tag's value changes how `#sarah` resolves the
next time any shot is generated. Markers are NOT expanded at "send to tab"
time — the raw `#sarah is doing X` is what's saved to the shot file, and
resolution happens server-side just before generation.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


MANIFEST_VERSION = "1.0"

# Mirror the marker rules used in llm_endpoints._name_to_marker so project
# tag markers slug identically to workspace tag markers.
_MARKER_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-]+")

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
            "tags": [],           # [{id, name, marker, value, category, created, updated}]
            "mood": "",           # freeform style+environment sentence
            "active_loras": [],   # LoRA names whose triggers/captions surface in `#` autocomplete
            "image_seed": None,   # uint32 seeded into a new shot's image tab on first send; null = no inheritance
            "video_seed": None,   # uint32 seeded into a new shot's video tab on first send; null = no inheritance
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
    """Forward-compat shim. Fill in missing keys; migrate legacy `subjects`
    entries into `tags` so older manifests keep their `#markers`."""
    base = empty_manifest()
    out = {**base, **(data or {})}
    out["meta"] = {**base["meta"], **(data.get("meta") or {})}
    out["specs"] = {**base["specs"], **(data.get("specs") or {})}
    globals_in = data.get("globals") or {}
    tags = list(globals_in.get("tags") or [])
    # Legacy subjects → tags: description becomes value, kind becomes category.
    # Skip any subject whose marker is already claimed by a real tag.
    seen_markers = {t.get("marker") for t in tags if t.get("marker")}
    for s in globals_in.get("subjects") or []:
        marker = s.get("marker") or _slug_marker(s.get("name", ""))
        name = (s.get("name") or "").strip()
        value = (s.get("description") or "").strip()
        if not marker or not name or not value or marker in seen_markers:
            continue
        seen_markers.add(marker)
        ts = _now()
        tags.append({
            "id": s.get("id") or uuid.uuid4().hex,
            "name": name,
            "marker": marker,
            "value": value,
            "category": (s.get("kind") or "other").strip().lower() or "other",
            "created": ts,
            "updated": ts,
        })
    out["globals"] = {
        "tags": tags,
        "mood": str(globals_in.get("mood") or ""),
        "active_loras": [str(x) for x in (globals_in.get("active_loras") or []) if str(x).strip()],
        "image_seed": _coerce_seed(globals_in.get("image_seed")),
        "video_seed": _coerce_seed(globals_in.get("video_seed")),
    }
    out["sequence"] = list(data.get("sequence") or [])
    out["panels"] = dict(data.get("panels") or {})
    return out


# ---------------------------------------------------------------------------
# Globals: project-local tags (mirror workspace tags but scoped to manifest)
# ---------------------------------------------------------------------------

def _normalize_tag_name(name: str) -> str:
    return " ".join((name or "").strip().split()).lower()


def validate_tag(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _normalize_tag_name(payload.get("name", ""))
    value = (payload.get("value") or "").strip()
    category = (payload.get("category") or "").strip() or None
    if not name:
        raise ValueError("Tag name is required")
    if not value:
        raise ValueError("Tag value is required")
    marker = _slug_marker(name)
    if not marker:
        raise ValueError("Tag name cannot be slugged into a marker")
    return {"name": name, "value": value, "category": category, "marker": marker}


def add_tag(manifest: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = validate_tag(payload)
    tags = manifest["globals"].setdefault("tags", [])
    if any(t.get("marker") == clean["marker"] for t in tags):
        raise ValueError(f"Tag with marker {clean['marker']} already exists")
    now = _now()
    entry = {"id": uuid.uuid4().hex, **clean, "created": now, "updated": now}
    tags.append(entry)
    return entry


def update_tag(manifest: Dict[str, Any], tag_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = validate_tag(payload)
    tags = manifest["globals"].setdefault("tags", [])
    for i, t in enumerate(tags):
        if t.get("id") == tag_id:
            if any(o.get("marker") == clean["marker"] and o.get("id") != tag_id for o in tags):
                raise ValueError(f"Tag with marker {clean['marker']} already exists")
            tags[i] = {
                "id": tag_id,
                **clean,
                "created": t.get("created") or _now(),
                "updated": _now(),
            }
            return tags[i]
    raise KeyError(f"Tag {tag_id} not found")


def remove_tag(manifest: Dict[str, Any], tag_id: str) -> bool:
    tags = manifest["globals"].setdefault("tags", [])
    before = len(tags)
    manifest["globals"]["tags"] = [t for t in tags if t.get("id") != tag_id]
    return len(manifest["globals"]["tags"]) < before


_SEED_MAX = 2**32 - 1


def _coerce_seed(raw: Any) -> Optional[int]:
    """Normalize a seed payload to uint32 or None. Empty string and null both clear."""
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Seed must be an integer")
    if n < 0 or n > _SEED_MAX:
        raise ValueError(f"Seed must be in [0, {_SEED_MAX}]")
    return n


def set_image_seed(manifest: Dict[str, Any], seed: Any) -> Optional[int]:
    cleaned = _coerce_seed(seed)
    manifest["globals"]["image_seed"] = cleaned
    return cleaned


def set_video_seed(manifest: Dict[str, Any], seed: Any) -> Optional[int]:
    cleaned = _coerce_seed(seed)
    manifest["globals"]["video_seed"] = cleaned
    return cleaned


def get_image_seed(manifest: Optional[Dict[str, Any]]) -> Optional[int]:
    if not manifest:
        return None
    return _coerce_seed(manifest.get("globals", {}).get("image_seed"))


def get_video_seed(manifest: Optional[Dict[str, Any]]) -> Optional[int]:
    if not manifest:
        return None
    return _coerce_seed(manifest.get("globals", {}).get("video_seed"))


def set_mood(manifest: Dict[str, Any], mood: str) -> str:
    cleaned = (mood or "").strip()
    manifest["globals"]["mood"] = cleaned
    return cleaned


def set_active_loras(manifest: Dict[str, Any], loras: List[str]) -> List[str]:
    """Replace the active-LoRA list. Dedupes and preserves insertion order."""
    seen = set()
    cleaned: List[str] = []
    for raw in loras or []:
        v = str(raw or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        cleaned.append(v)
    manifest["globals"]["active_loras"] = cleaned
    return cleaned


def get_active_loras(manifest: Optional[Dict[str, Any]]) -> List[str]:
    if not manifest:
        return []
    return list(manifest.get("globals", {}).get("active_loras") or [])


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
        "image_model": None,
        "video_model": None,
        "image_preview": None,
        "video_preview": None,
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


_PREVIEW_KINDS = {"image", "video"}


def set_panel_preview(
    manifest: Dict[str, Any], shot_id: str, kind: str, path: Optional[str]
) -> Dict[str, Any]:
    """Pin a generated asset as the panel's image/video preview.

    `path` is whatever the history API returns for a generation — relative to
    the project folder. `None` clears the slot. Auto-creates the panel if the
    user is pinning to a shot that only exists as a shot file (no panel yet).
    """
    if kind not in _PREVIEW_KINDS:
        raise ValueError(f"kind must be one of {sorted(_PREVIEW_KINDS)}")
    if not _SHOT_ID_RE.match(shot_id or ""):
        raise ValueError("Invalid shot id")
    field = "image_preview" if kind == "image" else "video_preview"
    clean = (path or "").strip() or None
    upsert_panel(manifest, shot_id, {field: clean})
    return manifest["panels"][shot_id]


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
# Token surface — project tags rendered as prompt tokens for /api/prompt/tokens
# ---------------------------------------------------------------------------

def tag_tokens(manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render globals.tags as unified prompt tokens.

    Same shape as workspace tags but `source: "tag"` and the id is namespaced
    `sbtag:` so the frontend can distinguish project-local from workspace tags
    if it ever wants to. Precedence is set in llm_endpoints._vocabulary_map.
    """
    if not manifest:
        return []
    out: List[Dict[str, Any]] = []
    for t in manifest.get("globals", {}).get("tags", []) or []:
        marker = t.get("marker") or _slug_marker(t.get("name", ""))
        name = (t.get("name") or "").strip()
        value = (t.get("value") or "").strip()
        if not marker or not name or not value:
            continue
        out.append({
            "id": f"sbtag:{t.get('id') or marker}",
            "source": "tag",
            "scope": "storyboard",
            "name": name,
            "marker": marker,
            "expansion": value,
            "category": t.get("category"),
        })
    return out


def get_mood(manifest: Optional[Dict[str, Any]]) -> str:
    if not manifest:
        return ""
    return (manifest.get("globals", {}).get("mood") or "").strip()


# ---------------------------------------------------------------------------
# Snapshots — dated copies of the manifest for storyboard versioning
# ---------------------------------------------------------------------------

def _snapshot_project_name(project_folder: Path) -> str:
    """Best-effort project name for snapshot filenames."""
    name = project_name_from_folder(project_folder)
    if name:
        return name
    existing = find_manifest(project_folder)
    if existing:
        stem = existing.stem  # "{project}_storyboard"
        return stem[: stem.rfind("_storyboard")] if "_storyboard" in stem else stem
    return "untitled"


def snapshot_manifest(project_folder: Path) -> str:
    """Save a dated snapshot of the current manifest alongside the main file.

    Snapshot files follow `{project}_storyboard_{YYMMDD}[a-z].json` so they
    are never confused with the live manifest (which ends in `_storyboard.json`)
    and are never picked up by the shot-file scanner.

    Returns the snapshot filename.
    """
    m = load_manifest(project_folder) or empty_manifest()
    project_name = _snapshot_project_name(project_folder)
    date_str = datetime.now().strftime("%y%m%d")

    base = f"{project_name}_storyboard_{date_str}"
    filename = f"{base}.json"
    path = project_folder / filename

    if path.exists():
        suffix = "a"
        while (project_folder / f"{base}{suffix}.json").exists():
            suffix = chr(ord(suffix) + 1)
        filename = f"{base}{suffix}.json"
        path = project_folder / filename

    snapshot = {**m, "meta": {**m.get("meta", {}), "snapshot": True, "snapshotAt": _now()}}
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return filename


def list_snapshots(project_folder: Path) -> List[Dict[str, Any]]:
    """List storyboard snapshot files in the folder, newest first."""
    if not project_folder or not project_folder.exists():
        return []
    results = []
    for p in project_folder.glob("*_storyboard_*.json"):
        try:
            results.append({
                "filename": p.name,
                "modifiedAt": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
        except Exception:
            pass
    results.sort(key=lambda x: x["modifiedAt"], reverse=True)
    return results


def restore_snapshot(project_folder: Path, filename: str) -> Dict[str, Any]:
    """Replace the live manifest with a snapshot.

    Strips the snapshot metadata fields before writing so the restored file
    looks like a normal manifest. Returns the restored manifest.
    """
    if not project_folder:
        raise ValueError("No project folder set")
    # Basic path-traversal guard
    if any(c in filename for c in ("/", "\\", "..")):
        raise ValueError("Invalid snapshot filename")
    path = project_folder / filename
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {filename}")
    with open(path) as f:
        data = json.load(f)
    m = _migrate(data)
    # Drop the snapshot-only meta fields before writing as the live manifest
    m["meta"].pop("snapshot", None)
    m["meta"].pop("snapshotAt", None)
    save_manifest(project_folder, m)
    return m
