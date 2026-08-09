"""
Shot synchronization — keep the shot JSON (source of truth) and FukProps aligned.

The FUK web UI stores image settings per-model:
    tabs.image = { activeModel, modelSettings: { <model>: {prompt, seed, ...} } }
(older shots use a flat `tabs.image`).

`pull`/`push` read and write the slot of the model Blender actually generates with —
not whatever the web UI happens to have active — so the seed Blender sends is the seed
recorded against that model. A shot authored in the web UI on some other model has no
slot for Blender's model yet; that first Load inherits the active model's settings so
the prompt still carries over. `activeModel` itself is never rewritten, so Blender does
not move the web UI's model tab under the user.

Blender-only fields (control source, openpose layer) are stored flat on `tabs.image`
(shot-global, model-independent).
"""

from __future__ import annotations

from . import props as props_mod
from .client import FukError

# Full shot JSON keyed by filename, captured on load so push() can merge.
_LOADED: dict[str, dict] = {}

_SEED_MODES = {m[0] for m in props_mod.SEED_MODE_ITEMS}


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


def _active_model(image: dict) -> str:
    """The model the web Image tab currently has selected."""
    return image.get("activeModel") or image.get("model") or ""


def _is_blank(settings: dict) -> bool:
    """True for a slot the web UI created but never filled in.

    Switching models in the web UI materialises a slot for every model it has ever
    shown, all at defaults. Treating those as real would hand Blender an empty prompt
    instead of inheriting the one the user actually wrote.
    """
    return not (settings.get("prompt") or settings.get("seed")
                or settings.get("lastUsedSeed"))


def _settings_for(image: dict, model: str) -> dict:
    """Settings for `model` — its own slot, else the active model's (inherited once)."""
    model_settings = image.get("modelSettings")
    if not isinstance(model_settings, dict):
        return image  # old flat format: one slot for everything
    own = model_settings.get(model)
    if isinstance(own, dict) and not _is_blank(own):
        return dict(own)
    return dict(model_settings.get(_active_model(image), {}) or {})


def _fielder(settings: dict, tab: dict):
    """Read a key from the model slot, falling back to the flat tab dict.

    Covers shots that store prompt/negative in either place.
    """
    def _field(key, default=""):
        val = settings.get(key, None)
        if val is None:
            val = tab.get(key, default)
        return val
    return _field


def _pull_seed(props, fields, _field) -> None:
    """Copy a tab's seed/seedMode/lastUsedSeed into the props named by `fields`."""
    mode_attr, seed_attr, last_attr = fields
    seed = props_mod.parse_seed(_field("seed", None))
    last_used = props_mod.parse_seed(_field("lastUsedSeed", None))
    mode = (_field("seedMode", "") or "").lower()
    setattr(props, mode_attr,
            mode if mode in _SEED_MODES else ("random" if seed is None else "fixed"))
    # Always surface a concrete number: the explicit seed if set, else the last
    # one actually used. Keeps the panel showing the real value in every mode.
    setattr(props, seed_attr, props_mod.seed_text(seed if seed is not None else last_used))
    setattr(props, last_attr, props_mod.seed_text(last_used))


def _pull_image(image: dict, props) -> None:
    # Resolve the generation model FIRST — it decides which slot everything below is
    # read from. Adopt the shot's model only if it's control-capable; otherwise keep
    # Blender's, so the control round trip isn't broken by the web UI's tab.
    active = _active_model(image)
    if active in props_mod.CONTROL_MODELS:
        props.model = active
    settings = _settings_for(image, props.model)
    _field = _fielder(settings, image)

    props.prompt = _field("prompt") or ""
    props.negative_prompt = _field("negative_prompt") or ""

    if _field("steps", None) is not None:
        props.steps = int(_field("steps"))
    if _field("guidance_scale", None) is not None:
        props.guidance_scale = float(_field("guidance_scale"))

    _pull_seed(props, props_mod.IMAGE_SEED_FIELDS, _field)

    of = settings.get("output_format", "png")
    if of in {"png", "exr", "both"}:
        props.output_format = of

    # Blender-specific fields persisted flat on the tab (shot-global). An absent key
    # keeps whatever the .blend already has — a shot authored in the web UI carries no
    # blender_* keys at all, and defaulting here reset the panel on every Load.
    cs = image.get("blender_control_source")
    if cs in props_mod.CONTROL_SOURCES:
        props.control_source = cs
    layer = image.get("blender_openpose_view_layer")
    if layer is not None:
        props.openpose_view_layer = layer or ""


def _pull_video(video: dict, props) -> None:
    """Copy `tabs.video`'s VACE slot into the video props.

    Same slot rule as the image side: Blender reads the task it generates with
    (VIDEO_TASK), inheriting the web UI's active task once if that slot is empty.
    """
    settings = _settings_for(video, props_mod.VIDEO_TASK)
    _field = _fielder(settings, video)

    props.video_prompt = _field("prompt") or ""
    props.video_negative_prompt = _field("negative_prompt") or ""

    if _field("steps", None) is not None:
        props.video_steps = int(_field("steps"))
    if _field("guidance_scale", None) is not None:
        props.video_guidance = float(_field("guidance_scale"))

    _pull_seed(props, props_mod.VIDEO_SEED_FIELDS, _field)

    pct = video.get("blender_video_percentage")
    if isinstance(pct, int) and 10 <= pct <= 100:
        props.video_percentage = pct


def pull(client, filename: str, props) -> dict:
    """Load `filename` via the server and copy its settings for Blender's models into props."""
    resp = client.load_shot(filename)
    data = resp.get("data", resp)
    _LOADED[filename] = data
    tabs = data.get("tabs", {}) or {}
    _pull_image(tabs.get("image", {}) or {}, props)
    _pull_video(tabs.get("video", {}) or {}, props)

    info = (data.get("project", {}) or {})
    label = f"{info.get('name','?')} shot{info.get('shot','?')} ({info.get('version','?')})"
    props.status = f"Loaded {label}"
    return data


def _writable(client, filename: str, props, video=False):
    """(data, tab, target_slot) for writing — fetches the shot if not cached."""
    data = _LOADED.get(filename)
    if data is None:
        # Not cached this session — fetch current state first so we don't clobber.
        resp = client.load_shot(filename)
        data = resp.get("data", resp)
        _LOADED[filename] = data

    name = "video" if video else "image"
    model = props_mod.VIDEO_TASK if video else props.model
    tab = data.setdefault("tabs", {}).setdefault(name, {})

    # Write into Blender's own model slot. `activeModel` is only filled in when the
    # tab has none at all — rewriting it would move the web UI's model tab.
    model_settings = tab.get("modelSettings")
    if isinstance(model_settings, dict):
        tab.setdefault("activeModel", model)
        target = model_settings.setdefault(model, {})
    else:
        target = tab  # old flat format
    return data, tab, target


def _write_seed(target, props, fields, seed_used=None):
    """Record the seed props named by `fields` the way the web tabs store them."""
    mode_attr, seed_attr, last_attr = fields
    mode = getattr(props, mode_attr)
    used = props_mod.parse_seed(seed_used)
    target["seedMode"] = mode
    target["seed"] = None if mode == "random" else props_mod.parse_seed(getattr(props, seed_attr))
    last_used = used if used is not None else props_mod.parse_seed(getattr(props, last_attr))
    if last_used is not None:
        target["lastUsedSeed"] = last_used


def push(client, filename: str, props) -> dict:
    """Merge props into Blender's slots on both tabs and save via the server."""
    data, image, target = _writable(client, filename, props)
    target["model"] = props.model
    target["prompt"] = props.prompt
    target["negative_prompt"] = props.negative_prompt
    target["steps"] = int(props.steps)
    target["guidance_scale"] = float(props.guidance_scale)
    _write_seed(target, props, props_mod.IMAGE_SEED_FIELDS)
    target["output_format"] = props.output_format
    # Deliberately NOT written: width/height/aspectRatio. Those are the shot's
    # text-to-image framing. Any control-driven generation inherits its size from the
    # input image instead (the server overrides to match), so Blender's render size
    # is already authoritative at generation time — writing it here would only
    # destroy the T2I framing the shot was authored with.

    # Blender-only fields stay flat on the tab (shot-global, model-independent).
    image["blender_control_source"] = props.control_source
    image["blender_openpose_view_layer"] = props.openpose_view_layer

    _, video, vtarget = _writable(client, filename, props, video=True)
    vtarget["task"] = props_mod.VIDEO_TASK
    vtarget["prompt"] = props.video_prompt
    vtarget["negative_prompt"] = props.video_negative_prompt
    vtarget["steps"] = int(props.video_steps)
    vtarget["guidance_scale"] = float(props.video_guidance)
    _write_seed(vtarget, props, props_mod.VIDEO_SEED_FIELDS)
    video["blender_video_percentage"] = int(props.video_percentage)

    client.save_shot(filename, data)
    props.status = f"Saved {filename}"
    return data


def record_run(client, filename: str, props, seed_used, video=False) -> None:
    """Persist what a generation actually used — the seed FUK rolled and the control
    source that drove it — back into the shot.

    The web tabs record `lastUsedSeed` after every generation; without this the shot
    file kept whatever seed it had before Blender ran, so nothing downstream could
    reproduce the frame. Deliberately narrow: prompts, steps and guidance still need an
    explicit Save to Shot, so this can't overwrite edits made in the web UI meanwhile.
    Best-effort — a generation must not fail because the shot couldn't be written.
    """
    try:
        data, tab, target = _writable(client, filename, props, video=video)
        fields = props_mod.VIDEO_SEED_FIELDS if video else props_mod.IMAGE_SEED_FIELDS
        _write_seed(target, props, fields, seed_used)
        if video:
            tab["blender_video_percentage"] = int(props.video_percentage)
        else:
            tab["blender_control_source"] = props.control_source
            tab["blender_openpose_view_layer"] = props.openpose_view_layer
        client.save_shot(filename, data)
    except (FukError, OSError, KeyError, ValueError, TypeError):
        pass


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


def generation_extras(client, filename: str, model: str) -> dict:
    """Advanced generation fields carried by the shot that the Blender panel doesn't
    expose (LoRAs, detail bias, EliGen, VRAM preset). These are set in FUK's web UI
    and must NOT be dropped when generating from Blender.

    Read from `model`'s slot, the same one `pull`/`push` use, so the LoRAs sent are the
    ones configured against the model being generated with.

    Returns a dict of payload fields to merge (mirrors the web Image tab's mapping).
    """
    data = _LOADED.get(filename)
    if data is None:
        try:
            resp = client.load_shot(filename)
            data = resp.get("data", resp)
            _LOADED[filename] = data
        except (FukError, OSError, KeyError, ValueError, TypeError):
            return {}

    image = (data.get("tabs", {}) or {}).get("image", {}) or {}
    settings = _settings_for(image, model)
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
