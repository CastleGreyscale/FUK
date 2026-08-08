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


def pull(client, filename: str, props) -> dict:
    """Load `filename` via the server and copy its settings for Blender's model into props."""
    resp = client.load_shot(filename)
    data = resp.get("data", resp)
    _LOADED[filename] = data
    image = (data.get("tabs", {}) or {}).get("image", {}) or {}

    # Resolve the generation model FIRST — it decides which slot everything below is
    # read from. Adopt the shot's model only if it's control-capable; otherwise keep
    # Blender's, so the control round trip isn't broken by the web UI's tab.
    active = _active_model(image)
    if active in props_mod.CONTROL_MODELS:
        props.model = active
    settings = _settings_for(image, props.model)

    # Fall back to the flat image dict when a key is absent from the slot (covers
    # shots that store prompt/negative in either place).
    def _field(key, default=""):
        val = settings.get(key, None)
        if val is None:
            val = image.get(key, default)
        return val

    props.prompt = _field("prompt") or ""
    props.negative_prompt = _field("negative_prompt") or ""

    if _field("steps", None) is not None:
        props.steps = int(_field("steps"))
    if _field("guidance_scale", None) is not None:
        props.guidance_scale = float(_field("guidance_scale"))

    seed = props_mod.parse_seed(_field("seed", None))
    last_used = props_mod.parse_seed(_field("lastUsedSeed", None))
    mode = (_field("seedMode", "") or "").lower()
    if mode in {"random", "fixed", "increment"}:
        props.seed_mode = mode
    else:
        props.seed_mode = "random" if seed is None else "fixed"

    # Always surface a concrete number: the explicit seed if set, else the last
    # one actually used. Keeps the panel showing the real value in every mode.
    props.seed = props_mod.seed_text(seed if seed is not None else last_used)
    props.last_used_seed = props_mod.seed_text(last_used)

    of = settings.get("output_format", "png")
    if of in {"png", "exr", "both"}:
        props.output_format = of

    # Blender-specific fields persisted flat on the shot (shot-global). An absent key
    # keeps whatever the .blend already has — a shot authored in the web UI carries no
    # blender_* keys at all, and defaulting here reset the panel on every Load.
    cs = image.get("blender_control_source")
    if cs in props_mod.CONTROL_SOURCES:
        props.control_source = cs
    layer = image.get("blender_openpose_view_layer")
    if layer is not None:
        props.openpose_view_layer = layer or ""

    info = (data.get("project", {}) or {})
    label = f"{info.get('name','?')} shot{info.get('shot','?')} ({info.get('version','?')})"
    props.status = f"Loaded {label}"
    return data


def _writable(client, filename: str, props):
    """(data, image, target_slot) for writing — fetches the shot if not cached."""
    data = _LOADED.get(filename)
    if data is None:
        # Not cached this session — fetch current state first so we don't clobber.
        resp = client.load_shot(filename)
        data = resp.get("data", resp)
        _LOADED[filename] = data

    image = data.setdefault("tabs", {}).setdefault("image", {})

    # Write into Blender's own model slot. `activeModel` is only filled in when the
    # shot has none at all — rewriting it would move the web UI's model tab.
    model_settings = image.get("modelSettings")
    if isinstance(model_settings, dict):
        image.setdefault("activeModel", props.model)
        target = model_settings.setdefault(props.model, {})
    else:
        target = image  # old flat format
    return data, image, target


def _write_seed(target, props, seed_used=None):
    """Record the seed fields the way the web Image tab stores them."""
    used = props_mod.parse_seed(seed_used)
    seed = props_mod.parse_seed(props.seed)
    target["seedMode"] = props.seed_mode
    target["seed"] = None if props.seed_mode == "random" else seed
    last_used = used if used is not None else props_mod.parse_seed(props.last_used_seed)
    if last_used is not None:
        target["lastUsedSeed"] = last_used


def push(client, filename: str, props) -> dict:
    """Merge props into Blender's model slot and save via the server."""
    data, image, target = _writable(client, filename, props)

    target["model"] = props.model
    target["prompt"] = props.prompt
    target["negative_prompt"] = props.negative_prompt
    target["steps"] = int(props.steps)
    target["guidance_scale"] = float(props.guidance_scale)
    _write_seed(target, props)
    target["output_format"] = props.output_format

    # Blender-only fields stay flat on image (shot-global, model-independent).
    image["blender_control_source"] = props.control_source
    image["blender_openpose_view_layer"] = props.openpose_view_layer

    client.save_shot(filename, data)
    props.status = f"Saved {filename}"
    return data


def record_run(client, filename: str, props, seed_used) -> None:
    """Persist what a generation actually used — the seed FUK rolled and the control
    source that drove it — back into the shot.

    The web Image tab records `lastUsedSeed` after every generation; without this the
    shot file kept whatever seed it had before Blender ran, so nothing downstream could
    reproduce the frame. Deliberately narrow: prompt, steps and guidance still need an
    explicit Save to Shot, so this can't overwrite edits made in the web UI meanwhile.
    Best-effort — a generation must not fail because the shot couldn't be written.
    """
    try:
        data, image, target = _writable(client, filename, props)
        _write_seed(target, props, seed_used)
        image["blender_control_source"] = props.control_source
        image["blender_openpose_view_layer"] = props.openpose_view_layer
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
