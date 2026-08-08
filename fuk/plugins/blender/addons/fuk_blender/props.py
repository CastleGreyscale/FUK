"""
FukProps — the in-Blender working copy of a shot's image settings.

The shot JSON on disk (served by FUK) stays the source of truth; these props are
synced from it on Load and merged back on Save (see shot.py). Blender-specific
fields (control source, openpose layer) are persisted into the shot JSON under
`tabs.image.blender_*` keys so they travel with the shot.
"""

from __future__ import annotations

import bpy

# Populated by the Connect/Refresh operator: list of {"name","path","modifiedAt"}.
SHOT_CACHE: list[dict] = []

# Populated by the Refresh Tags operator: prompt tokens from /api/prompt/tokens.
# Each: {"marker": "#hero", "name": "...", "expansion": "...", "category": "..."}.
TOKEN_CACHE: list[dict] = []

# Blender does not keep its own reference to the strings a dynamic enum callback
# returns, so we must hold them here — otherwise they can be garbage collected and
# the dropdown renders corrupted labels. Matters more now that the shot list is
# re-listed on demand (Refresh) rather than only once at Connect.
_SHOT_ENUM_ITEMS: list[tuple] = []


def shot_enum_items(self, context):
    """EnumProperty items from the cached shot list (filenames)."""
    items = []
    for entry in SHOT_CACHE:
        name = entry.get("name", "")
        if name:
            items.append((name, name, entry.get("modifiedAt", "")))
    if not items:
        items = [("", "<no shots — Connect first>", "")]
    _SHOT_ENUM_ITEMS[:] = items
    return _SHOT_ENUM_ITEMS


def _on_shot_change(self, context):
    """Mirror the picked shot into a plain string that survives file save/load.

    The shot dropdown is a dynamic enum whose value can't persist while the shot
    cache is empty (right after a file loads), so we stash the name here and restore
    it on (auto)connect.
    """
    if self.shot_file:
        self.last_shot = self.shot_file


def token_enum_items(self, context):
    """EnumProperty items (searchable) for the #marker tag inserter."""
    items = []
    seen = set()
    for tok in TOKEN_CACHE:
        marker = tok.get("marker", "")
        if not marker or marker in seen:
            continue
        seen.add(marker)
        name = tok.get("name") or marker
        category = tok.get("category") or tok.get("source") or ""
        label = f"{marker}  ({name})" if name != marker else marker
        desc = (tok.get("expansion") or "")[:120]
        if category:
            desc = f"[{category}] {desc}"
        items.append((marker, label, desc))
    if not items:
        items = [("", "<no tags — Refresh Tags>", "")]
    return items


# Generation models the Blender flow can drive. control_union variants are the
# ControlNet-style structural-control path (FUK routes control_image -> context_image).
MODEL_ITEMS = [
    ("qwen_image_control_union_2512", "Control Union 2512", "ControlNet-style structural control (2512 base)"),
    ("qwen_image_control_union", "Control Union", "ControlNet-style structural control"),
]

CONTROL_SOURCE_ITEMS = [
    ("depth", "Depth", "Blender Z-pass, normalized (native, exact)"),
    ("normals", "Normals", "Blender normal pass, remapped (native; experimental encoding)"),
    ("openpose", "OpenPose", "From a rig view layer (native) or FUK estimate (fallback)"),
    ("canny", "Canny", "Edge map derived by FUK from the beauty render"),
]

CONTROL_MODELS = {m[0] for m in MODEL_ITEMS}
CONTROL_SOURCES = {c[0] for c in CONTROL_SOURCE_ITEMS}

# FUK seeds are uint32 (the web UI rolls them with crypto.getRandomValues on a
# Uint32Array). Blender's IntProperty is a C int — assigning anything above
# 0x7fffffff raises ValueError, and a max above it won't even register. So seeds
# live in StringProperty and are parsed on the way in and out; storing them as ints
# silently dropped over half of all seeds FUK produces.
SEED_MAX = 2**32 - 1
_SEED_DIGITS = len(str(SEED_MAX))


def parse_seed(value):
    """Coerce a seed from the shot JSON or a props field to a uint32, or None."""
    if value is None:
        return None
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return max(0, min(SEED_MAX, n))


def seed_text(value) -> str:
    """Format a seed for a props field ('' when there is no seed)."""
    n = parse_seed(value)
    return "" if n is None else str(n)


def _sanitize_seed(self, context):
    """Hold the seed field to digits inside the uint32 range as it is typed."""
    for field in ("seed", "last_used_seed"):
        raw = getattr(self, field)
        digits = "".join(c for c in raw if c.isdigit())[:_SEED_DIGITS]
        if digits:
            digits = str(min(int(digits), SEED_MAX))
        if raw != digits:
            setattr(self, field, digits)


def _update_bg_alpha(self, context):
    """Live-apply the opacity slider to the current camera-background overlay,
    so it can be adjusted after a render without re-running anything."""
    cam = context.scene.camera
    if cam is None or getattr(cam, "type", None) != "CAMERA":
        return
    for bg in cam.data.background_images:
        if bg.image and bpy.path.basename(bg.image.filepath) == "result.png":
            bg.alpha = self.bg_alpha


class FukProps(bpy.types.PropertyGroup):
    # --- connection / shot binding (runtime, not saved to shot) ---
    project_folder: bpy.props.StringProperty(
        name="Project Folder",
        description="FUK project folder (the one containing your shot .json files)",
        subtype="DIR_PATH",
        default="",
    )
    shot_file: bpy.props.EnumProperty(
        name="Shot",
        description="Shot .json to drive (source of truth)",
        items=shot_enum_items,
        update=_on_shot_change,
    )
    # Persisted copy of the selected shot filename, restored on (auto)connect.
    last_shot: bpy.props.StringProperty(default="")

    # --- mirrored shot.tabs.image fields ---
    prompt: bpy.props.StringProperty(name="Prompt", default="")
    negative_prompt: bpy.props.StringProperty(name="Negative", default="")
    model: bpy.props.EnumProperty(name="Model", items=MODEL_ITEMS, default="qwen_image_control_union_2512")
    steps: bpy.props.IntProperty(name="Steps", default=20, min=1, max=200)
    guidance_scale: bpy.props.FloatProperty(name="Guidance", default=2.5, min=0.0, max=30.0)
    seed_mode: bpy.props.EnumProperty(
        name="Seed Mode",
        items=[("random", "Random", "New random seed each run"),
               ("fixed", "Fixed", "Reuse the seed below"),
               ("increment", "Increment", "Seed +1 after each run")],
        default="random",
    )
    # Text, not int — see SEED_MAX above. Empty means "no seed set".
    seed: bpy.props.StringProperty(
        name="Seed", default="", update=_sanitize_seed,
        description=f"Seed value (0–{SEED_MAX}); blank lets FUK roll one",
    )
    output_format: bpy.props.EnumProperty(
        name="Output",
        items=[("png", "PNG", ""), ("exr", "EXR", ""), ("both", "Both", "")],
        default="png",
    )

    # --- control wiring (Blender-specific; saved as tabs.image.blender_*) ---
    control_source: bpy.props.EnumProperty(
        name="Control", items=CONTROL_SOURCE_ITEMS, default="depth",
        description="Which structural control map drives the generation",
    )
    openpose_view_layer: bpy.props.StringProperty(
        name="OpenPose Layer",
        description="View layer that renders your OpenPose rig skeleton (leave blank to use FUK estimation)",
        default="",
    )
    preview_percentage: bpy.props.IntProperty(
        name="Preview %", default=50, min=10, max=100,
        description="Resolution percentage used for Quick Preview renders",
    )
    preview_steps: bpy.props.IntProperty(
        name="Preview Steps", default=8, min=1, max=100,
        description="Diffusion steps used for Quick Preview generations",
    )
    show_diffusion: bpy.props.BoolProperty(
        name="Diffusion Preview", default=True,
        description="Show the image forming step-by-step during generation "
                    "(decodes a few mid-diffusion frames — adds a little time)",
    )

    # --- video (Wan-VACE) ---
    video_percentage: bpy.props.IntProperty(
        name="Video %", default=50, min=10, max=100,
        description="Resolution percentage for the video render (scales the control "
                    "sequence and the output video — lower is much faster)",
    )
    video_steps: bpy.props.IntProperty(
        name="Video Steps", default=20, min=1, max=100,
        description="Diffusion steps for the Wan-VACE video generation",
    )
    video_guidance: bpy.props.FloatProperty(
        name="Video Guidance", default=5.0, min=0.0, max=30.0,
        description="CFG scale for the video generation",
    )

    # --- live (IPR-style auto-update) ---
    live_mode: bpy.props.BoolProperty(
        name="Live", default=False,
        description="Auto-run a Quick Preview after camera/object edits settle",
    )
    live_delay: bpy.props.FloatProperty(
        name="Delay", default=0.6, min=0.1, max=5.0,
        description="Seconds of stillness after a change before auto-previewing",
    )
    live_interrupt: bpy.props.BoolProperty(
        name="Interrupt", default=True,
        description="In Live mode, cancel an in-flight preview and restart when you "
                    "make a new edit (instead of waiting for it to finish)",
    )

    # --- result display ---
    result_display: bpy.props.EnumProperty(
        name="Show Result",
        items=[
            ("viewport", "Viewport", "Overlay the result as the camera's background and switch to camera view"),
            ("window", "New Window", "Open the result in a standalone Image Editor window"),
            ("editor", "Image Editor", "Use an open Image Editor (or a new window if none)"),
        ],
        default="viewport",
    )
    bg_alpha: bpy.props.FloatProperty(
        name="Overlay Opacity", default=1.0, min=0.0, max=1.0,
        description="Opacity of the camera-background result overlay (live-adjustable after a render)",
        update=_update_bg_alpha,
    )

    # --- prompt resolution preview (read-only) ---
    resolved_preview: bpy.props.StringProperty(name="Resolved", default="")

    # The actual seed FUK used last (random or fixed); shown so the panel always
    # surfaces a concrete number even in random mode. Text, like `seed`.
    last_used_seed: bpy.props.StringProperty(
        name="Last Seed", default="", update=_sanitize_seed)

    # --- runtime status (not saved) ---
    status: bpy.props.StringProperty(name="Status", default="Not connected")
    connected: bpy.props.BoolProperty(name="Connected", default=False)
    busy: bpy.props.BoolProperty(name="Busy", default=False)
    last_result: bpy.props.StringProperty(name="Last Result", default="", subtype="FILE_PATH")
    # Whether the current result is already a FUK history entry (Full auto-saves;
    # Quick Preview does not). Drives the Save-to-History button.
    result_persisted: bpy.props.BoolProperty(name="Saved", default=False)
    # The per-shot working dir holding the latest beauty/control/result/meta.
    working_dir: bpy.props.StringProperty(name="Working Dir", default="", subtype="DIR_PATH")
