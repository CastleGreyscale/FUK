"""
LLM Endpoints — Ollama integration for FUK

Phase 1: image description only. Prompt construction and storyboard assist
land later (see docs/LLM_INTEGRATION.md).

The Ollama server is expected to be running externally at OLLAMA_HOST
(default http://localhost:11434). FUK does not spawn or manage it.

VRAM strategy: every request passes keep_alive=0 so the model is evicted
from VRAM immediately after the response — the generation pipeline is the
primary VRAM tenant, the LLM is a transient guest.
"""

import base64
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import requests
from fastapi import HTTPException
from PIL import Image
from pydantic import BaseModel


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.environ.get("FUK_LLM_MODEL", "qwen3-vl:8b")
DESCRIBE_MAX_SIZE = 1024
VIDEO_FRAME_MAX_SIZE = 512   # per-frame size for video describe — smaller because we send several
VIDEO_FRAME_COUNT = 6        # frames sampled evenly across the clip
VIDEO_REQUEST_TIMEOUT = 120  # multi-frame requests take longer
REQUEST_TIMEOUT = 60

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

DESCRIBE_SYSTEM = (
    "You are a visual description assistant for a VFX production pipeline. "
    "Describe images with the precision and language of a cinematographer or "
    "VFX supervisor.\n\n"
    "Focus on: subject, action, lighting quality and direction, color palette, "
    "atmosphere, camera angle/lens feel, depth of field, and any notable visual "
    "effects or stylistic choices.\n\n"
    "Be concrete. Avoid vague qualitative adjectives (\"beautiful\", \"stunning\"). "
    "Write in present tense. 2-4 sentences. No preamble."
)

DESCRIBE_VIDEO_SYSTEM = (
    "You are a visual description assistant for a VFX production pipeline. "
    "You are given several frames sampled evenly across a short video clip, "
    "ordered from earliest to latest. Describe the clip with the language of a "
    "cinematographer or VFX supervisor.\n\n"
    "Cover, in this order: subject and action, how the action evolves across the "
    "clip, camera movement (push, pull, pan, tilt, handheld, static), lighting and "
    "color, atmosphere, and any notable stylistic choices.\n\n"
    "Treat the frames as a single continuous shot — do not describe them as separate "
    "images. Be concrete. Avoid vague qualitative adjectives. Write in present tense. "
    "3-5 sentences. No preamble."
)


class DescribeRequest(BaseModel):
    image_path: str
    max_size: Optional[int] = DESCRIBE_MAX_SIZE


# ---------------------------------------------------------------------------
# Prompt tags — shorthand → full description expansion.
# Stored as a single JSON file under fuk/ui/data/prompt_tags.json. The format
# is intentionally simple so the same file can be consumed later by the
# storyboard / prompt-construction features without migration.
# ---------------------------------------------------------------------------

TAGS_FILE = Path(__file__).parent / "data" / "prompt_tags.json"
CATEGORIES_FILE = Path(__file__).parent / "data" / "prompt_tag_categories.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-]{0,63}$")

# Starter category set. Auto-written to CATEGORIES_FILE on first run; the user
# can edit that file freely afterwards — it is the source of truth from then on.
DEFAULT_CATEGORIES = [
    "composition",
    "subject",
    "lighting",
    "camera",
    "color",
    "style",
    "atmosphere",
    "setting",
]


class TagPayload(BaseModel):
    name: str
    value: str
    category: Optional[str] = None


def _load_tags() -> List[dict]:
    if not TAGS_FILE.exists():
        return []
    try:
        with open(TAGS_FILE) as f:
            data = json.load(f)
        return data.get("tags", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _save_tags(tags: List[dict]) -> None:
    TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TAGS_FILE, "w") as f:
        json.dump({"tags": tags}, f, indent=2)


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def _load_categories() -> List[str]:
    """Read user-editable category list, seeding the file on first run."""
    if not CATEGORIES_FILE.exists():
        CATEGORIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CATEGORIES_FILE, "w") as f:
            json.dump({"categories": DEFAULT_CATEGORIES}, f, indent=2)
        return list(DEFAULT_CATEGORIES)
    try:
        with open(CATEGORIES_FILE) as f:
            data = json.load(f)
        cats = data.get("categories", []) if isinstance(data, dict) else []
        return [str(c).strip().lower() for c in cats if str(c).strip()]
    except Exception:
        return list(DEFAULT_CATEGORIES)


def _validate_payload(payload: TagPayload) -> tuple[str, str, Optional[str]]:
    name = _normalize_name(payload.name)
    value = payload.value.strip()
    category = (payload.category or "").strip().lower() or None
    if not name:
        raise HTTPException(status_code=400, detail="Tag name is required")
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Tag name: 1-64 chars, letters/digits/space/underscore/hyphen, must start alphanumeric",
        )
    if not value:
        raise HTTPException(status_code=400, detail="Tag value is required")
    if len(value) > 4000:
        raise HTTPException(status_code=400, detail="Tag value too long (max 4000 chars)")
    if category is not None:
        allowed = _load_categories()
        if category not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category '{category}'. Allowed: {', '.join(allowed)}",
            )
    return name, value, category


def _encode_image_for_describe(image_path: Path, max_size: int) -> str:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _encode_pil_for_describe(img: "Image.Image", max_size: int) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _sample_video_frames(video_path: Path, n: int, max_size: int) -> List[str]:
    """Pull N evenly-spaced frames from a video, return them as base64 JPEGs."""
    # cv2 is a heavy import — defer it so the LLM module doesn't pay the cost
    # unless a video is actually being described.
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path.name}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    # Even spacing, biased slightly inward so we don't waste a sample on a
    # black first/last frame.
    if total <= n:
        indices = list(range(total))
    else:
        indices = [int((i + 0.5) * total / n) for i in range(n)]

    frames_b64: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        # cv2 gives BGR; PIL expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        frames_b64.append(_encode_pil_for_describe(img, max_size))

    cap.release()
    if not frames_b64:
        raise RuntimeError("Could not decode any frames from the video")
    return frames_b64


def _ollama_health() -> dict:
    """Probe Ollama and report whether the configured model is available."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException as e:
        return {"available": False, "model_present": False, "error": str(e)}

    tags = r.json().get("models", []) or []
    names = {m.get("name", "") for m in tags}
    return {
        "available": True,
        "model": LLM_MODEL,
        "model_present": LLM_MODEL in names,
        "models": sorted(names),
    }


def setup_llm_routes(app, *, resolve_input_path: Callable[[str], Path], log):
    """Register /api/llm/* endpoints."""

    @app.get("/api/llm/health")
    async def llm_health():
        return _ollama_health()

    @app.post("/api/llm/describe")
    async def llm_describe(request: DescribeRequest):
        try:
            image_path = resolve_input_path(request.image_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image path: {e}")

        if image_path is None or not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Media not found: {request.image_path}")

        is_video = image_path.suffix.lower() in VIDEO_EXTS
        max_size = request.max_size or (VIDEO_FRAME_MAX_SIZE if is_video else DESCRIBE_MAX_SIZE)
        timeout = VIDEO_REQUEST_TIMEOUT if is_video else REQUEST_TIMEOUT

        try:
            if is_video:
                images = _sample_video_frames(image_path, VIDEO_FRAME_COUNT, max_size)
                system_prompt = DESCRIBE_VIDEO_SYSTEM
                user_text = (
                    f"{len(images)} frames from a short clip, ordered earliest to latest. "
                    "Describe the clip as a single continuous shot."
                )
            else:
                images = [_encode_image_for_describe(image_path, max_size)]
                system_prompt = DESCRIBE_SYSTEM
                user_text = "Describe this image."
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read media: {e}")

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_text,
                    "images": images,
                },
            ],
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0.3},
        }

        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=timeout,
            )
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except requests.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama unavailable at {OLLAMA_HOST}. Is `ollama serve` running?",
            )

        if r.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Model {LLM_MODEL} not pulled. Run: ollama pull {LLM_MODEL}",
            )
        if not r.ok:
            raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text[:300]}")

        data = r.json()
        content = (data.get("message") or {}).get("content", "").strip()
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()

        if not content:
            raise HTTPException(status_code=502, detail="LLM returned empty response")

        kind = "video" if is_video else "image"
        log.info("LLM", f"describe ({kind}): {image_path.name} -> {len(content)} chars")

        return {
            "description": content,
            "model": LLM_MODEL,
            "image_path": str(image_path),
            "kind": kind,
            "frames_used": len(images) if is_video else None,
        }

    # -----------------------------------------------------------------------
    # Prompt tags
    # -----------------------------------------------------------------------

    @app.get("/api/tags/categories")
    async def list_tag_categories():
        return {"categories": _load_categories()}

    @app.get("/api/tags")
    async def list_tags():
        tags = _load_tags()
        tags.sort(key=lambda t: ((t.get("category") or "~"), t.get("name", "")))
        return {"tags": tags}

    @app.post("/api/tags")
    async def create_tag(payload: TagPayload):
        name, value, category = _validate_payload(payload)
        tags = _load_tags()
        if any(_normalize_name(t.get("name", "")) == name for t in tags):
            raise HTTPException(status_code=409, detail=f"Tag '{name}' already exists")
        now = datetime.now().isoformat()
        entry = {
            "id": uuid.uuid4().hex,
            "name": name,
            "value": value,
            "category": category,
            "created": now,
            "updated": now,
        }
        tags.append(entry)
        _save_tags(tags)
        log.info("LLM", f"tag created: {name}")
        return entry

    @app.put("/api/tags/{tag_id}")
    async def update_tag(tag_id: str, payload: TagPayload):
        name, value, category = _validate_payload(payload)
        tags = _load_tags()
        idx = next((i for i, t in enumerate(tags) if t.get("id") == tag_id), -1)
        if idx < 0:
            raise HTTPException(status_code=404, detail="Tag not found")
        if any(
            i != idx and _normalize_name(t.get("name", "")) == name
            for i, t in enumerate(tags)
        ):
            raise HTTPException(status_code=409, detail=f"Tag '{name}' already exists")
        tags[idx].update({
            "name": name,
            "value": value,
            "category": category,
            "updated": datetime.now().isoformat(),
        })
        _save_tags(tags)
        return tags[idx]

    @app.delete("/api/tags/{tag_id}")
    async def delete_tag(tag_id: str):
        tags = _load_tags()
        new_tags = [t for t in tags if t.get("id") != tag_id]
        if len(new_tags) == len(tags):
            raise HTTPException(status_code=404, detail="Tag not found")
        _save_tags(new_tags)
        return {"success": True}
