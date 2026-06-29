"""
Shot synchronization — keep the shot JSON (source of truth) and FukProps aligned.

- `pull` loads a shot via the server, mirrors `tabs.image` into props, and caches
  the full JSON so we can merge edits back without losing other fields.
- `push` merges props back into the cached JSON and saves it via the server.
"""

from __future__ import annotations

from . import props as props_mod

# Full shot JSON keyed by filename, captured on load so push() can merge.
_LOADED: dict[str, dict] = {}


def _aspect_to_height(width: int, aspect: str) -> int:
    """Parse 'W:H' (e.g. '2.39:1') and return the even height for `width`."""
    try:
        a, b = aspect.split(":")
        ratio = float(a) / float(b)
        if ratio > 0:
            h = int(round(width / ratio))
            return h - (h % 2)
    except (ValueError, ZeroDivisionError):
        pass
    return 0


def pull(client, filename: str, props) -> dict:
    """Load `filename` via the server and copy tabs.image into props."""
    resp = client.load_shot(filename)
    data = resp.get("data", resp)
    _LOADED[filename] = data
    image = (data.get("tabs", {}) or {}).get("image", {}) or {}

    props.prompt = image.get("prompt", "") or ""
    props.negative_prompt = image.get("negative_prompt", "") or ""

    model = image.get("model")
    valid_models = {m[0] for m in props_mod.MODEL_ITEMS}
    if model in valid_models:
        props.model = model

    props.steps = int(image.get("steps", props.steps) or props.steps)
    props.guidance_scale = float(image.get("guidance_scale", props.guidance_scale) or props.guidance_scale)

    seed = image.get("seed", None)
    if seed in (None, "", "null"):
        props.seed_mode = "random"
    else:
        props.seed_mode = "fixed"
        try:
            props.seed = max(0, int(seed))
        except (ValueError, TypeError):
            props.seed_mode = "random"

    of = image.get("output_format", "png")
    if of in {"png", "exr", "both"}:
        props.output_format = of

    # Blender-specific fields persisted alongside the shot.
    cs = image.get("blender_control_source", "depth")
    if cs in {c[0] for c in props_mod.CONTROL_SOURCE_ITEMS}:
        props.control_source = cs
    props.openpose_view_layer = image.get("blender_openpose_view_layer", "") or ""

    info = (data.get("project", {}) or {})
    label = f"{info.get('name','?')} shot{info.get('shot','?')} ({info.get('version','?')})"
    props.status = f"Loaded {label}"
    return data


def push(client, filename: str, props) -> dict:
    """Merge props back into the shot JSON and save via the server."""
    data = _LOADED.get(filename)
    if data is None:
        # Not cached this session — fetch current state first so we don't clobber.
        resp = client.load_shot(filename)
        data = resp.get("data", resp)
        _LOADED[filename] = data

    tabs = data.setdefault("tabs", {})
    image = tabs.setdefault("image", {})

    image["prompt"] = props.prompt
    image["negative_prompt"] = props.negative_prompt
    image["model"] = props.model
    image["steps"] = int(props.steps)
    image["guidance_scale"] = float(props.guidance_scale)
    image["seed"] = None if props.seed_mode == "random" else int(props.seed)
    image["output_format"] = props.output_format
    image["blender_control_source"] = props.control_source
    image["blender_openpose_view_layer"] = props.openpose_view_layer

    client.save_shot(filename, data)
    props.status = f"Saved {filename}"
    return data
