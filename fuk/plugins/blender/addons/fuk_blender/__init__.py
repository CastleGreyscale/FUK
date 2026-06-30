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

from . import prefs, props, client, shot, render, controls, viewer, ops, ui

# Re-import submodules on addon reload so edits take effect without restarting Blender.
for _m in (prefs, props, client, shot, render, controls, viewer, ops, ui):
    importlib.reload(_m)

import bpy

_CLASSES = (
    prefs.FukAddonPreferences,
    props.FukProps,
    *ops.CLASSES,
    ui.FUK_PT_main,
)


_addon_keymaps = []


def _register_keymaps():
    kc = bpy.context.window_manager.keyconfigs.addon
    if not kc:
        return
    km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    # Ctrl+Shift+P — run a Quick Preview; Ctrl+Shift+L — toggle Live mode.
    kmi = km.keymap_items.new("fuk.generate", "P", "PRESS", ctrl=True, shift=True)
    kmi.properties.mode = "preview"
    _addon_keymaps.append((km, kmi))
    kmi = km.keymap_items.new("fuk.live", "L", "PRESS", ctrl=True, shift=True)
    _addon_keymaps.append((km, kmi))


def _unregister_keymaps():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fuk = bpy.props.PointerProperty(type=props.FukProps)
    _register_keymaps()


def unregister():
    _unregister_keymaps()
    del bpy.types.Scene.fuk
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
