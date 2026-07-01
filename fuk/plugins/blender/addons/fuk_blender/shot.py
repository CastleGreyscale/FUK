"""
Shot synchronization — keep the shot JSON (source of truth) and FukProps aligned.

The FUK web UI stores image settings per-model:
    tabs.image = { activeModel, modelSettings: { <model>: {prompt, seed, ...} } }
(older shots use a flat `tabs.image`). `pull`/`push` resolve the active model's
settings the same way the web Image tab does, so prompt/seed/etc. flow through both
ways. Blender's own generation model (control_union) is kept independent of the
shot's active model so the control round trip isn't broken; Blender-only fields are
stored flat on `tabs.image` (shot-global).
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


def _active_settings(image: dict):
    """Return (settings, active_model) mirroring the web Image tab's resolution."""
    model_settings = image.get("modelSettings")
    if isinstance(model_settings, dict):
        active = image.get("activeModel") or image.get("model") or ""
        return dict(model_settings.get(active, {}) or {}), active
    # Old flat format.
    return image, image.get("model", "")


def pull(client, filename: str, props) -> dict:
    """Load `filename` via the server and copy the active model's settings into props."""
    resp = client.load_shot(filename)
    data = resp.get("data", resp)
    _LOADED[filename] = data
    image = (data.get("tabs", {}) or {}).get("image", {}) or {}
    settings, active = _active_settings(image)

    # Prefer the active model's slot; fall back to the flat image dict if a key
    # is absent there (covers shots that store prompt/negative in either place).
    def _field(key, default=""):
        val = settings.get(key, None)
        if val is None:
            val = image.get(key, default)
        return val

    props.prompt = _field("prompt") or ""
    props.negative_prompt = _field("negative_prompt") or ""

    # Adopt the shot's model only if it's a control-capable Blender model; otherwise
    # keep Blender's current generation model so the control round trip still works.
    if active in {"qwen_image_control_union_2512", "qwen_image_control_union"}:
        props.model = active

    if _field("steps", None) is not None:
        props.steps = int(_field("steps"))
    if _field("guidance_scale", None) is not None:
        props.guidance_scale = float(_field("guidance_scale"))

    seed = _field("seed", None)
    last_used = _field("lastUsedSeed", None)
    mode = (_field("seedMode", "") or "").lower()
    if mode == "random":
        props.seed_mode = "random"
    elif mode in ("fixed", "increment"):
        props.seed_mode = "fixed"
    else:
        props.seed_mode = "random" if seed in (None, "", "null") else "fixed"

    # Always surface a concrete number: the explicit seed if set, else the last
    # one actually used. Keeps the panel showing the real value in every mode.
    display_seed = seed if seed not in (None, "", "null") else last_used
    if display_seed not in (None, "", "null"):
        try:
            props.seed = max(0, int(display_seed))
        except (ValueError, TypeError):
            pass
    if last_used not in (None, "", "null"):
        try:
            props.last_used_seed = max(0, int(last_used))
        except (ValueError, TypeError):
            pass

    of = settings.get("output_format", "png")
    if of in {"png", "exr", "both"}:
        props.output_format = of

    # Blender-specific fields persisted flat on the shot (shot-global).
    cs = image.get("blender_control_source", "depth")
    if cs in {c[0] for c in props_mod.CONTROL_SOURCE_ITEMS}:
        props.control_source = cs
    props.openpose_view_layer = image.get("blender_openpose_view_layer", "") or ""

    info = (data.get("project", {}) or {})
    label = f"{info.get('name','?')} shot{info.get('shot','?')} ({info.get('version','?')})"
    props.status = f"Loaded {label}"
    return data


def push(client, filename: str, props) -> dict:
    """Merge props back into the active model's settings and save via the server."""
    data = _LOADED.get(filename)
    if data is None:
        # Not cached this session — fetch current state first so we don't clobber.
        resp = client.load_shot(filename)
        data = resp.get("data", resp)
        _LOADED[filename] = data

    tabs = data.setdefault("tabs", {})
    image = tabs.setdefault("image", {})

    # Write into the same slot the web Image tab reads from.
    model_settings = image.get("modelSettings")
    if isinstance(model_settings, dict):
        active = image.get("activeModel") or image.get("model") or props.model
        image.setdefault("activeModel", active)
        target = model_settings.setdefault(active, {})
    else:
        target = image

    target["prompt"] = props.prompt
    target["negative_prompt"] = props.negative_prompt
    target["steps"] = int(props.steps)
    target["guidance_scale"] = float(props.guidance_scale)
    target["seed"] = None if props.seed_mode == "random" else int(props.seed)
    target["seedMode"] = "random" if props.seed_mode == "random" else "fixed"
    if props.last_used_seed:
        target["lastUsedSeed"] = int(props.last_used_seed)
    target["output_format"] = props.output_format

    # Blender-only fields stay flat on image (shot-global, model-independent).
    image["blender_control_source"] = props.control_source
    image["blender_openpose_view_layer"] = props.openpose_view_layer

    client.save_shot(filename, data)
    props.status = f"Saved {filename}"
    return data


def load_current(client, filename: str) -> dict:
    """Load the shot from the server (also sets the server's project state) and refresh
    the local cache — so advanced settings edited in the web UI (LoRAs, detail, VRAM)
    are picked up at generation time WITHOUT a manual Load Shot. Does not touch props;
    the prompt/seed you edit in Blender stay authoritative.
    """
    resp = client.load_shot(filename)
    data = resp.get("data", resp)
    _LOADED[filename] = data
    return data


def generation_extras(client, filename: str) -> dict:
    """Advanced generation fields carried by the shot that the Blender panel doesn't
    expose (LoRAs, detail bias, EliGen, VRAM preset). These are set in FUK's web UI
    and must NOT be dropped when generating from Blender.

    Returns a dict of payload fields to merge (mirrors the web Image tab's mapping).
    """
    data = _LOADED.get(filename)
    if data is None:
        try:
            resp = client.load_shot(filename)
            data = resp.get("data", resp)
            _LOADED[filename] = data
        except Exception:
            return {}

    image = (data.get("tabs", {}) or {}).get("image", {}) or {}
    settings, _ = _active_settings(image)
    extras = {}

    # LoRAs — forward the array as the web UI stores/sends it.
    loras = settings.get("loras")
    if loras:
        extras["loras"] = loras

    # Detail bias -> denoising_strength (web UI: only when < 1.0).
    detail = settings.get("detail_bias")
    try:
        if detail is not None and float(detail) < 1.0:
            extras["denoising_strength"] = float(detail)
    except (TypeError, ValueError):
        pass

    # Sampling/detail timestep control.
    if settings.get("exponential_shift_mu") is not None:
        extras["exponential_shift_mu"] = settings["exponential_shift_mu"]

    # EliGen entity control.
    if settings.get("eligen_source"):
        extras["eligen_source"] = settings["eligen_source"]
    if settings.get("eligen_alpha") is not None:
        extras["eligen_alpha"] = settings["eligen_alpha"]

    # VRAM preset travels with the shot.
    if settings.get("vram_preset"):
        extras["vram_preset"] = settings["vram_preset"]

    return extras
