"""
FUK Blender addon — drive the FUK generation server from inside Blender.

Render a shot's control passes (Blender-native depth/normals/openpose, FUK canny)
and round-trip them through FUK to get a generated still back in Blender. Images
first; video later. See README.md.
"""

bl_info = {
    "name": "FUK Bridge",
    "author": "FUK",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > FUK",
    "description": "Render control passes and round-trip them through the FUK server",
    "category": "Render",
}

import importlib

from . import prefs, props, client, shot, render, controls, ops, ui

# Re-import submodules on addon reload so edits take effect without restarting Blender.
for _m in (prefs, props, client, shot, render, controls, ops, ui):
    importlib.reload(_m)

import bpy

_CLASSES = (
    prefs.FukAddonPreferences,
    props.FukProps,
    *ops.CLASSES,
    ui.FUK_PT_main,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fuk = bpy.props.PointerProperty(type=props.FukProps)


def unregister():
    del bpy.types.Scene.fuk
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
