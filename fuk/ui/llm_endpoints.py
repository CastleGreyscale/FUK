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
from fastapi import HTTPException, Query
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

EXPAND_SYSTEM_BASE = (
    "You are a prompt engineer for an AI media generation pipeline used in "
    "professional VFX and animation production. Your job is to refine the "
    "author's working draft into a clean, directive, generation-ready prompt "
    "while preserving their intent and any specific named phrases.\n\n"
    "Rules:\n"
    "- Treat the input as the author's draft. Refine and integrate; do not "
    "rewrite from scratch.\n"
    "- Write as a visual description, not an instruction. Subject, action, "
    "framing, lighting, mood, style. Concrete language. No filler.\n"
    "- Preserve the existing 'Style: ...' sentence if present — the listed "
    "phrases are authoritative style intent. Integrate them naturally rather "
    "than reworording them.\n"
    "- Use vocabulary phrases verbatim where they apply. Prefer them over "
    "inventing new style claims.\n"
    "- Negative prompt: technical failure modes relevant to the shot. No vague "
    "catch-alls.\n"
    "{mode_rule}\n"
    "- Output JSON only. No preamble, no markdown fences.\n\n"
    "Output schema:\n"
    "{{\"positive\": \"...\", \"negative\": \"...\", \"notes\": \"...\"}}\n\n"
    "The \"notes\" field is optional — use it only if there's a meaningful "
    "caveat or suggestion the author should know about."
)

EXPAND_MODE_RULE_IMAGE = (
    "- This is a still-image prompt. Do not describe motion or temporal action."
)
EXPAND_MODE_RULE_VIDEO = (
    "- This is a video prompt. Include motion description: subject action, "
    "camera move (push, pull, pan, tilt, handheld, static). Keep action "
    "language grounded and physically plausible."
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


# ---------------------------------------------------------------------------
# Prompt token unification — tags + LoRA triggers + caption-analysis phrases
# all surface through a single /api/prompt/tokens endpoint so the frontend
# autocomplete and slot picker only need to know one shape.
# ---------------------------------------------------------------------------

CAPTIONS_DIR = Path(__file__).parent / "data" / "lora_captions"

# Markers must be safe in a prompt textarea. Underscores, digits, hyphen.
_MARKER_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-]+")
_MARKER_SCAN_RE = re.compile(r"#([A-Za-z0-9][A-Za-z0-9_\-]*)")

# Min share of captions a phrase must appear in to be surfaced as a token.
CAPTION_MIN_DOC_SHARE = 0.05
CAPTION_TOP_N = 30  # per LoRA, taken from bigrams+trigrams pool

# Small keyword map for auto-categorizing caption phrases. Cheap but covers
# the obvious cases; anything that doesn't match falls into "uncategorized".
_CAPTION_CATEGORY_KEYWORDS = [
    ("lighting",    ("light", "lit", "shadow", "key", "backlit", "rim", "glow", "silhouette")),
    ("camera",      ("lens", "frame", "framing", "angle", "shot", "focus", "dof", "depth of field", "telephoto", "wide-angle", "close-up")),
    ("composition", ("composition", "centered", "rule of thirds", "negative space", "symmetr")),
    ("color",       ("color", "colour", "palette", "tone", "hue", "saturation", "muted", "warm", "cool")),
    ("style",       ("cinematic", "aesthetic", "grain", "film", "noir", "style", "stylized", "vintage", "retro", "nostalgic")),
    ("atmosphere",  ("mood", "atmosphere", "ambience", "ambient", "haze", "fog", "smoke")),
    ("setting",     ("interior", "exterior", "scene", "backdrop", "environment", "urban", "rural")),
    ("subject",     ("subject", "figure", "person", "character", "portrait")),
]


def _name_to_marker(name: str) -> str:
    """Slug a token name into a `#marker` form."""
    slug = _MARKER_CHAR_RE.sub("_", name.strip()).strip("_")
    return f"#{slug}" if slug else ""


def _model_family(model: Optional[str]) -> Optional[str]:
    """qwen_image -> qwen, wan_2_2 -> wan, etc. None passes through."""
    if not model:
        return None
    return str(model).split("_", 1)[0].lower()


def _categorize_caption_phrase(phrase: str) -> Optional[str]:
    p = phrase.lower()
    for cat, kws in _CAPTION_CATEGORY_KEYWORDS:
        if any(kw in p for kw in kws):
            return cat
    return None


def _load_loras_config(config_dir: Path) -> List[dict]:
    """Read defaults_loras.json -> list of lora dicts. Empty list if missing."""
    path = config_dir / "defaults_loras.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        loras = data.get("loras", [])
        return loras if isinstance(loras, list) else []
    except Exception:
        return []


def _tag_tokens() -> List[dict]:
    """Render saved prompt tags into the unified token shape."""
    out = []
    for t in _load_tags():
        name = t.get("name", "").strip()
        value = t.get("value", "").strip()
        if not name or not value:
            continue
        marker = _name_to_marker(name)
        if not marker:
            continue
        out.append({
            "id": f"tag:{t.get('id') or name}",
            "source": "tag",
            "name": name,
            "marker": marker,
            "expansion": value,
            "category": t.get("category"),
        })
    return out


def _lora_tokens(config_dir: Path, model_family: Optional[str]) -> List[dict]:
    """Synthesize tokens from defaults_loras.json trigger words.

    The same trigger often appears across multiple model variants (e.g. an
    Italian_Horror LoRA shipped for both qwen_image and qwen_image_2512).
    They collapse to one token here so the dropdown doesn't repeat itself —
    the resolved entry just records that more than one variant exists.
    """
    by_marker: dict[str, dict] = {}
    for entry in _load_loras_config(config_dir):
        trigger = (entry.get("trigger_word") or "").strip()
        inject = (entry.get("inject_text") or "").strip()
        if not trigger or not inject:
            continue
        entry_model = entry.get("model") or ""
        if model_family and _model_family(entry_model) != model_family:
            continue
        marker = _name_to_marker(trigger)
        if not marker:
            continue
        lora_name = entry.get("name") or trigger
        lora_key = f"{entry_model}/{lora_name}" if entry_model else lora_name
        existing = by_marker.get(marker)
        if existing:
            existing.setdefault("lora_keys", []).append(lora_key)
            continue
        by_marker[marker] = {
            "id": f"lora:{trigger}",
            "source": "lora",
            "name": trigger,
            "marker": marker,
            "expansion": inject,
            "category": "style",
            "lora": lora_name,
            "lora_key": lora_key,
            "lora_keys": [lora_key],
            "model": entry_model or None,
            "default_strength": entry.get("default_strength"),
        }
    return list(by_marker.values())


def _read_caption_analysis(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _caption_tokens_for_lora(analysis: dict) -> List[dict]:
    """Extract top bigrams/trigrams as tokens, filtered by doc_freq share."""
    captions_total = int(analysis.get("captions_total") or 0)
    trigger = (analysis.get("trigger_word") or "").strip()
    if captions_total <= 0 or not trigger:
        return []

    # Pool bigrams + trigrams; rank by doc_freq descending.
    pool = []
    for src in ("bigrams", "trigrams"):
        for item in analysis.get(src) or []:
            term = (item.get("term") or "").strip()
            doc_freq = int(item.get("doc_freq") or 0)
            if not term or doc_freq <= 0:
                continue
            share = doc_freq / captions_total
            if share < CAPTION_MIN_DOC_SHARE:
                continue
            pool.append((term, doc_freq, share, item.get("count") or doc_freq))

    pool.sort(key=lambda x: x[1], reverse=True)
    pool = pool[:CAPTION_TOP_N]

    out = []
    for term, doc_freq, share, count in pool:
        out.append({
            "id": f"caption:{trigger}:{term}",
            "source": "caption",
            "name": term,
            # Caption phrases are already short natural language — insert
            # them as raw text rather than a marker. The frontend treats
            # marker=null as "paste expansion directly, no compile step."
            "marker": None,
            "expansion": term,
            "category": _categorize_caption_phrase(term),
            "lora_trigger": trigger,
            "score": round(share, 3),
            "count": count,
        })
    return out


def _caption_tokens(active_lora_triggers: set) -> List[dict]:
    """Load caption analyses, filter to active LoRA trigger words."""
    if not CAPTIONS_DIR.exists() or not active_lora_triggers:
        return []
    out = []
    for path in sorted(CAPTIONS_DIR.glob("*.json")):
        analysis = _read_caption_analysis(path)
        if not analysis:
            continue
        trigger = (analysis.get("trigger_word") or "").strip()
        if trigger and trigger in active_lora_triggers:
            out.extend(_caption_tokens_for_lora(analysis))
    return out


def _resolve_active_lora_triggers(
    config_dir: Path,
    active_loras: List[str],
) -> set:
    """Map active LoRA identifiers (name, key, or trigger) to their triggers.

    Anything that doesn't resolve via defaults_loras.json falls through as
    its raw form — that lets a caption analysis file work even when its LoRA
    isn't registered yet, which is common during LoRA-builder iteration.
    """
    if not active_loras:
        return set()
    cfg = _load_loras_config(config_dir)
    requested = {s.strip() for s in active_loras if s and s.strip()}
    triggers = set()
    matched = set()
    for entry in cfg:
        trigger = (entry.get("trigger_word") or "").strip()
        if not trigger:
            continue
        name = entry.get("name") or ""
        model = entry.get("model") or ""
        composite = f"{model}/{name}" if model else name
        for r in requested:
            if r in (name, composite, trigger):
                triggers.add(trigger)
                matched.add(r)
    # Anything the user passed that didn't match a registered LoRA still
    # gets a shot at lining up with a caption analysis file by trigger word.
    triggers.update(requested - matched)
    return triggers


class CompileRequest(BaseModel):
    text: str
    model: Optional[str] = None
    active_loras: Optional[List[str]] = None
    # Style chips assembled via the slot picker. Each entry is either a
    # `#marker` (resolved against the same vocabulary as inline expansion)
    # or a literal phrase (caption-derived tokens have no marker). They
    # are joined into one trailing "Style: …" sentence so the compiled
    # prompt retains a disciplined structure even without the LLM path.
    style_chips: Optional[List[str]] = None
    style_label: Optional[str] = "Style"


_SENTENCE_END_RE = re.compile(r"[.!?]['\")\]]*\s*$")
_MOOD_LABEL = "Mood"


def _storyboard_context() -> tuple[List[dict], str]:
    """Lazy-load storyboard project-tag tokens and mood for the active project.

    Imported lazily so this module stays usable in test contexts where the
    project folder isn't set. Failures are silent — the storyboard is optional.
    """
    try:
        import storyboard_manager as sm
        from project_endpoints import get_project_folder
        folder = get_project_folder()
        if not folder:
            return [], ""
        manifest = sm.load_manifest(folder)
        return sm.tag_tokens(manifest), sm.get_mood(manifest)
    except Exception:
        return [], ""


def _storyboard_active_loras() -> List[str]:
    """Project-wide active LoRAs from the storyboard manifest.

    These are unioned with whatever the caller passes so the storyboard's
    LoRA picker activates caption autocomplete across the storyboard tab
    AND adds context on the per-shot Image/Video panels.
    """
    try:
        import storyboard_manager as sm
        from project_endpoints import get_project_folder
        folder = get_project_folder()
        if not folder:
            return []
        manifest = sm.load_manifest(folder)
        return sm.get_active_loras(manifest)
    except Exception:
        return []


def _vocabulary_map(config_dir: Path, model_family: Optional[str]) -> dict:
    """Build the `marker -> expansion` map used by compile and resolve.

    Precedence (first one to claim a marker wins):
      1. Storyboard tags — project-local, live with the manifest.
      2. Workspace tags (prompt_tags.json) — shared across projects.
      3. LoRA triggers (defaults_loras.json) — auto-generated shorthand.

    Project tags shadow workspace tags by design — the user opted into the
    project, so their named entities override generic shorthand.
    """
    sb_tag_tokens, _ = _storyboard_context()
    marker_to_expansion: dict[str, str] = {}
    for tok in sb_tag_tokens:
        m = tok.get("marker")
        if m and m not in marker_to_expansion:
            marker_to_expansion[m] = tok["expansion"]
    for tok in _tag_tokens():
        m = tok.get("marker")
        if m and m not in marker_to_expansion:
            marker_to_expansion[m] = tok["expansion"]
    for tok in _lora_tokens(config_dir, model_family):
        m = tok.get("marker")
        if m and m not in marker_to_expansion:
            marker_to_expansion[m] = tok["expansion"]
    return marker_to_expansion


def _compile_prompt(
    text: str,
    config_dir: Path,
    model: Optional[str],
    active_loras: Optional[List[str]],
    style_chips: Optional[List[str]] = None,
    style_label: str = "Style",
) -> dict:
    """Drain style chips into a trailing "Style: …" sentence.

    Inline `#markers` in `text` are NOT expanded here — they stay raw and are
    resolved at generation time (see `_resolve_prompt`). This keeps subjects
    like `#sarah` live: editing the global description changes every shot's
    output without rewriting the prompt textareas.
    """
    family = _model_family(model)
    marker_to_expansion = _vocabulary_map(config_dir, family)

    expanded_markers: List[str] = []   # only chips that were resolved
    unknown_markers: List[str] = []

    compiled_main = (text or "").strip()

    # Style chips: a chip is either a marker (`#noir`) or a raw phrase.
    # Marker chips resolve through the same vocabulary as inline markers.
    style_parts: List[str] = []
    seen_style: set = set()
    for raw in (style_chips or []):
        chip = (raw or "").strip()
        if not chip:
            continue
        if chip.startswith("#"):
            exp = marker_to_expansion.get(chip)
            if exp is None:
                unknown_markers.append(chip)
                continue
            expanded_markers.append(chip)
            piece = exp.strip()
        else:
            piece = chip
        # De-dupe case-insensitively so two LoRAs with the same inject text
        # don't double up in the style sentence.
        key = piece.lower()
        if not piece or key in seen_style:
            continue
        seen_style.add(key)
        style_parts.append(piece)

    label = (style_label or "Style").strip() or "Style"
    style_sentence = ""
    if style_parts:
        style_sentence = f"{label}: " + ", ".join(style_parts) + "."

    if compiled_main and style_sentence:
        if not _SENTENCE_END_RE.search(compiled_main):
            compiled_main = compiled_main + "."
        compiled = f"{compiled_main}\n\n{style_sentence}"
    elif style_sentence:
        compiled = style_sentence
    else:
        compiled = compiled_main

    # Trigger words for the active LoRAs that didn't make it into the text.
    # We surface this so the UI can flag "you have LoRA X enabled but didn't
    # reference its trigger" — no auto-injection.
    triggers = _resolve_active_lora_triggers(config_dir, active_loras or [])
    missing_triggers = sorted(t for t in triggers if t.lower() not in compiled.lower())

    return {
        "compiled": compiled,
        "expanded_markers": expanded_markers,
        "unknown_markers": sorted(set(unknown_markers)),
        "missing_lora_triggers": missing_triggers,
        "style_sentence": style_sentence,
    }


class ResolveRequest(BaseModel):
    text: str
    model: Optional[str] = None
    active_loras: Optional[List[str]] = None
    apply_mood: Optional[bool] = True


def _resolve_prompt(
    text: str,
    config_dir: Path,
    model: Optional[str],
    active_loras: Optional[List[str]],
    apply_mood: bool = True,
) -> dict:
    """Expand `#markers` and (optionally) append the storyboard's mood sentence.

    Called server-side just before generation. The result is what's actually
    sent to the model. Returns the resolved text plus enough provenance for
    metadata capture.
    """
    family = _model_family(model)
    marker_to_expansion = _vocabulary_map(config_dir, family)

    expanded_markers: List[str] = []
    unknown_markers: List[str] = []

    def _replace(match: re.Match) -> str:
        marker = match.group(0)
        if marker in marker_to_expansion:
            expanded_markers.append(marker)
            return marker_to_expansion[marker]
        unknown_markers.append(marker)
        return marker

    resolved = _MARKER_SCAN_RE.sub(_replace, text or "").strip()

    _, mood = _storyboard_context()
    mood_applied = ""
    if apply_mood and mood:
        mood_sentence = f"{_MOOD_LABEL}: {mood.rstrip('.')}."
        # Only append if the mood isn't already present verbatim — author may
        # have written it explicitly in the prompt and we shouldn't double up.
        if mood.lower() not in resolved.lower():
            if resolved and not _SENTENCE_END_RE.search(resolved):
                resolved = resolved + "."
            resolved = f"{resolved}\n\n{mood_sentence}" if resolved else mood_sentence
            mood_applied = mood

    return {
        "resolved": resolved,
        "expanded_markers": expanded_markers,
        "unknown_markers": sorted(set(unknown_markers)),
        "mood_applied": mood_applied,
    }


class ExpandRequest(BaseModel):
    text: str
    model: Optional[str] = None
    active_loras: Optional[List[str]] = None
    style_chips: Optional[List[str]] = None
    mode: Optional[str] = "image"   # "image" | "video"
    intent: Optional[str] = None    # optional free-text steer; appended to the user message


def _vocabulary_list(config_dir: Path, model_family: Optional[str], char_budget: int = 4000) -> str:
    """Render a compact vocabulary list for the LLM to consume verbatim.

    Format: [source:category] name: "expansion"
    Truncated to keep the system message bounded — most prompts won't need
    more than a few dozen entries.
    """
    lines: List[str] = []
    total = 0

    def _add(prefix: str, name: str, expansion: str) -> bool:
        nonlocal total
        line = f'{prefix} {name}: "{expansion}"'
        if total + len(line) + 1 > char_budget:
            return False
        lines.append(line)
        total += len(line) + 1
        return True

    for t in _tag_tokens():
        cat = t.get("category") or "tag"
        if not _add(f"[tag:{cat}]", t["name"], t["expansion"]):
            break

    for t in _lora_tokens(config_dir, model_family):
        if not _add("[lora]", t["name"], t["expansion"]):
            break

    return "\n".join(lines)


def _build_expand_messages(
    text: str,
    config_dir: Path,
    mode: str,
    model: Optional[str],
    active_loras: Optional[List[str]],
    style_chips: Optional[List[str]],
    intent: Optional[str],
) -> List[dict]:
    """Compose the chat messages for /api/prompt/expand."""
    mode_norm = (mode or "image").lower()
    mode_rule = EXPAND_MODE_RULE_VIDEO if mode_norm == "video" else EXPAND_MODE_RULE_IMAGE
    system = EXPAND_SYSTEM_BASE.format(mode_rule=mode_rule)

    vocab = _vocabulary_list(config_dir, _model_family(model))
    if vocab:
        system = f"{system}\n\nVocabulary:\n{vocab}"

    # Pre-resolve the draft so the LLM sees the author's intent in concrete
    # form. Two passes: Compile drains style chips into a "Style:" sentence,
    # then Resolve expands `#markers` against the live vocabulary (globals +
    # tags + LoRA triggers). Mood is *not* applied here — it's appended at
    # generation time, and the LLM shouldn't try to refine it.
    pre = _compile_prompt(
        text=text,
        config_dir=config_dir,
        model=model,
        active_loras=active_loras,
        style_chips=style_chips,
    )
    resolved = _resolve_prompt(
        text=pre["compiled"],
        config_dir=config_dir,
        model=model,
        active_loras=active_loras,
        apply_mood=False,
    )
    draft = resolved["resolved"]

    triggers = sorted(_resolve_active_lora_triggers(config_dir, active_loras or []))

    user_parts = []
    if intent and intent.strip():
        user_parts.append(f"Intent: {intent.strip()}")
    user_parts.append("Draft:\n" + (draft or "(empty)"))
    if triggers:
        user_parts.append("Active LoRA triggers: " + ", ".join(triggers))
    user_parts.append(f"Mode: {mode_norm}")
    user_parts.append("/no_think")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _parse_expand_response(content: str) -> dict:
    """Extract JSON {positive, negative?, notes?} from the LLM raw output."""
    raw = (content or "").strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    # Strip a code fence if the model returned one despite instructions.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to locate the first JSON object in the response.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {"positive": raw, "negative": "", "notes": "Model did not return valid JSON; raw text used."}
        else:
            return {"positive": raw, "negative": "", "notes": "Model did not return valid JSON; raw text used."}

    return {
        "positive": str(data.get("positive") or "").strip(),
        "negative": str(data.get("negative") or "").strip(),
        "notes": str(data.get("notes") or "").strip() or None,
    }


def setup_llm_routes(app, *, resolve_input_path: Callable[[str], Path], log, config_dir: Optional[Path] = None):
    """Register /api/llm/* and /api/prompt/* endpoints."""
    # Fall back to fuk/config relative to this module if not provided.
    cfg_dir = config_dir or (Path(__file__).resolve().parent.parent / "config")

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

    # -----------------------------------------------------------------------
    # Unified prompt tokens — tags + LoRA triggers + caption phrases
    # -----------------------------------------------------------------------

    @app.get("/api/prompt/tokens")
    async def list_prompt_tokens(
        model: Optional[str] = Query(None, description="Model id, e.g. qwen_image; used to filter LoRA tokens by family"),
        active_loras: Optional[str] = Query(None, description="Comma-separated LoRA keys/names/triggers currently enabled"),
    ):
        family = _model_family(model)
        active_list = [s for s in (active_loras or "").split(",") if s.strip()]
        # Union the storyboard's project-wide LoRA picks. Preserves caller
        # order first, then appends any storyboard additions.
        sb_loras = _storyboard_active_loras()
        seen = {s for s in active_list}
        for s in sb_loras:
            if s and s not in seen:
                active_list.append(s)
                seen.add(s)
        active_triggers = _resolve_active_lora_triggers(cfg_dir, active_list)

        # Project tags shadow workspace tags on marker collision; emit them
        # first so the autocomplete surfaces the project version.
        sb_tag_toks, mood = _storyboard_context()

        tokens: List[dict] = []
        seen_markers = {t.get("marker") for t in sb_tag_toks if t.get("marker")}
        tokens.extend(sb_tag_toks)
        tokens.extend(t for t in _tag_tokens() if t.get("marker") not in seen_markers)

        lora_toks = _lora_tokens(cfg_dir, family)
        # Mark which LoRA tokens correspond to currently-active LoRAs so the
        # UI can highlight or auto-include them.
        for t in lora_toks:
            t["active"] = t.get("name") in active_triggers
        tokens.extend(lora_toks)

        tokens.extend(_caption_tokens(active_triggers))

        categories = _load_categories()
        # Surface any category the storyboard tags use (character, prop, etc.)
        # even if the workspace category list doesn't include it — otherwise
        # the slot picker would dump them all into "other".
        extra = [
            c for c in dict.fromkeys(t.get("category") for t in sb_tag_toks if t.get("category"))
            if c not in categories
        ]
        if extra:
            categories = [*extra, *categories]

        return {
            "tokens": tokens,
            "categories": categories,
            "model_family": family,
            "active_lora_triggers": sorted(active_triggers),
            "mood": mood,
        }

    @app.post("/api/prompt/compile")
    async def compile_prompt(req: CompileRequest):
        return _compile_prompt(
            text=req.text,
            config_dir=cfg_dir,
            model=req.model,
            active_loras=req.active_loras,
            style_chips=req.style_chips,
            style_label=req.style_label or "Style",
        )

    @app.post("/api/prompt/resolve")
    async def resolve_prompt(req: ResolveRequest):
        """Resolve `#markers` and append the storyboard mood. Called by the
        generation pipeline (server-side) and exposed as an endpoint so the
        frontend can preview the final string before submit."""
        return _resolve_prompt(
            text=req.text,
            config_dir=cfg_dir,
            model=req.model,
            active_loras=req.active_loras,
            apply_mood=bool(req.apply_mood) if req.apply_mood is not None else True,
        )

    @app.post("/api/prompt/expand")
    async def expand_prompt(req: ExpandRequest):
        if not (req.text or "").strip() and not (req.style_chips or []) and not (req.intent or "").strip():
            raise HTTPException(status_code=400, detail="Nothing to expand — provide text, style chips, or intent.")

        messages = _build_expand_messages(
            text=req.text,
            config_dir=cfg_dir,
            mode=req.mode or "image",
            model=req.model,
            active_loras=req.active_loras,
            style_chips=req.style_chips,
            intent=req.intent,
        )

        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0.7},
            "think": False,
        }

        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT,
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
        content = (data.get("message") or {}).get("content", "")
        parsed = _parse_expand_response(content)
        if not parsed["positive"]:
            raise HTTPException(status_code=502, detail="LLM returned empty positive prompt")

        log.info("LLM", f"expand ({req.mode}): -> {len(parsed['positive'])} chars positive")

        return {
            **parsed,
            "model": LLM_MODEL,
            "mode": req.mode or "image",
        }
