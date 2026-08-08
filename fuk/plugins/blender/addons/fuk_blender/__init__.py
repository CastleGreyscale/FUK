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

# Panel order matters: FUK_PT_main must register before the sub-panels that name it
# as their bl_parent_id.
_CLASSES = (
    prefs.FukAddonPreferences,
    props.FukProps,
    *ops.CLASSES,
    *ui.CLASSES,
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


# --- best-effort auto-connect at launch / file load ---------------------------
def _try_autoconnect():
    """One-shot: connect if a project folder is set and we're not already connected.
    Best-effort — silently no-ops if the server is down."""
    try:
        scene = bpy.context.scene
        p = getattr(scene, "fuk", None)
        if p and p.project_folder and not p.connected and not p.busy:
            ops.connect_to_server(bpy.context)
    except Exception:
        pass
    return None  # don't repeat


@bpy.app.handlers.persistent
def _on_load_post(_dummy):
    # After a .blend loads, the scene's saved project folder is available — attempt
    # a connect shortly after so the shot list is ready without pressing Connect.
    bpy.app.timers.register(_try_autoconnect, first_interval=0.5)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fuk = bpy.props.PointerProperty(type=props.FukProps)
    _register_keymaps()
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    # And once for the file that's already open when the addon is enabled.
    bpy.app.timers.register(_try_autoconnect, first_interval=1.0)


def unregister():
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if bpy.app.timers.is_registered(_try_autoconnect):
        bpy.app.timers.unregister(_try_autoconnect)
    _unregister_keymaps()
    del bpy.types.Scene.fuk
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
