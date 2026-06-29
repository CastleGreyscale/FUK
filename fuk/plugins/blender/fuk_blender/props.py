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


def shot_enum_items(self, context):
    """EnumProperty items from the cached shot list (filenames)."""
    items = []
    for entry in SHOT_CACHE:
        name = entry.get("name", "")
        if name:
            items.append((name, name, entry.get("modifiedAt", "")))
    if not items:
        items = [("", "<no shots — Connect first>", "")]
    return items


# Generation models the Blender flow can drive. control_union variants are the
# ControlNet-style structural-control path (FUK routes control_image -> context_image).
MODEL_ITEMS = [
    ("qwen_image_control_union_2512", "Control Union 2512", "ControlNet-style structural control (2512 base)"),
    ("qwen_image_control_union", "Control Union", "ControlNet-style structural control"),
    ("qwen_image_2512", "Qwen Image 2512", "Plain text-to-image (no control)"),
    ("qwen_image", "Qwen Image", "Plain text-to-image (no control)"),
]

CONTROL_SOURCE_ITEMS = [
    ("depth", "Depth", "Blender Z-pass, normalized (native, exact)"),
    ("normals", "Normals", "Blender normal pass, remapped (native; experimental encoding)"),
    ("openpose", "OpenPose", "From a rig view layer (native) or FUK estimate (fallback)"),
    ("canny", "Canny", "Edge map derived by FUK from the beauty render"),
]


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
    )

    # --- mirrored shot.tabs.image fields ---
    prompt: bpy.props.StringProperty(name="Prompt", default="")
    negative_prompt: bpy.props.StringProperty(name="Negative", default="")
    model: bpy.props.EnumProperty(name="Model", items=MODEL_ITEMS, default="qwen_image_control_union_2512")
    steps: bpy.props.IntProperty(name="Steps", default=20, min=1, max=200)
    guidance_scale: bpy.props.FloatProperty(name="Guidance", default=2.5, min=0.0, max=30.0)
    seed_mode: bpy.props.EnumProperty(
        name="Seed Mode",
        items=[("random", "Random", "New random seed each run"),
               ("fixed", "Fixed", "Reuse the seed below")],
        default="random",
    )
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0)
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

    # --- runtime status (not saved) ---
    status: bpy.props.StringProperty(name="Status", default="Not connected")
    busy: bpy.props.BoolProperty(name="Busy", default=False)
    last_result: bpy.props.StringProperty(name="Last Result", default="", subtype="FILE_PATH")
