"""
Tag version history — shared by workspace tags (prompt_tags.json) and project
tags (the storyboard manifest).

A history store is a plain dict `{tag_id: [version, ...]}`, oldest first. Each
version is a superseded copy of a tag:

    {vid, name, value, category, updated, replaced_at, reason}

`updated` is when that version was written, `replaced_at` when it was
superseded, and `reason` what superseded it (edit / delete / moved / restore /
replaced / snapshot). Apart from the per-tag cap the store is append-only, so a deleted
tag keeps its versions and can be brought back from its last one.

The caller owns persistence: workspace history lives in its own JSON file,
project history rides inside the manifest so storyboard snapshots carry it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

MAX_VERSIONS = 30

# Reasons that end a tag's life in its store — always recorded, even when the
# content matches the previous version, so the deleted list knows when and why.
_TERMINAL_REASONS = {"delete", "moved", "snapshot"}


def _now() -> str:
    return datetime.now().isoformat()


def _content(tag: Dict[str, Any]) -> tuple:
    return (tag.get("name") or "", tag.get("value") or "", tag.get("category") or None)


def same_content(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _content(a) == _content(b)


def _age_seconds(iso: Optional[str]) -> Optional[float]:
    try:
        return (datetime.now() - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def record(
    history: Dict[str, List[dict]],
    tag: Dict[str, Any],
    reason: str,
    coalesce_seconds: float = 0.0,
) -> bool:
    """Push `tag` — the version about to be superseded — onto its history.

    `coalesce_seconds` drops a version that an edit wrote less than that long
    ago. The storyboard autosaves while typing, so without it every pause
    mid-sentence would become a version. The state from before the edit
    session is still kept, and so is a version a restore or push wrote, since
    neither was written by an edit. Returns True if recorded.
    """
    tag_id = tag.get("id")
    if not tag_id:
        return False
    versions = history.setdefault(tag_id, [])
    if reason not in _TERMINAL_REASONS:
        if coalesce_seconds and versions and versions[-1].get("reason") == "edit":
            age = _age_seconds(tag.get("updated"))
            if age is not None and age < coalesce_seconds:
                return False
        if versions and same_content(versions[-1], tag):
            return False
    versions.append({
        "vid": uuid.uuid4().hex[:12],
        "name": tag.get("name") or "",
        "value": tag.get("value") or "",
        "category": tag.get("category") or None,
        "updated": tag.get("updated"),
        "replaced_at": _now(),
        "reason": reason,
    })
    del versions[:-MAX_VERSIONS]
    return True


def versions(history: Dict[str, List[dict]], tag_id: str) -> List[dict]:
    """A tag's versions, newest first."""
    return list(reversed(history.get(tag_id) or []))


def find_version(history: Dict[str, List[dict]], tag_id: str, vid: str) -> Optional[dict]:
    return next((v for v in history.get(tag_id) or [] if v.get("vid") == vid), None)


def deleted(history: Dict[str, List[dict]], live_ids: Iterable[str]) -> List[dict]:
    """Tags with history but no live entry, newest deletion first.

    Each row is the tag's last version (what a restore brings back) plus its
    id, when and why it went, and how many versions are on file.
    """
    live = set(live_ids)
    out = []
    for tag_id, vs in history.items():
        if tag_id in live or not vs:
            continue
        last = vs[-1]
        out.append({
            "id": tag_id,
            "vid": last.get("vid"),
            "name": last.get("name"),
            "value": last.get("value"),
            "category": last.get("category"),
            "deleted_at": last.get("replaced_at"),
            "reason": last.get("reason"),
            "version_count": len(vs),
        })
    out.sort(key=lambda r: r.get("deleted_at") or "", reverse=True)
    return out
