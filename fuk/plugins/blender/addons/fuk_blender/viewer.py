"""
Show a generated result back inside Blender.

- viewport : set the image as the active camera's background overlay (display in
             front), and switch the 3D view to camera view so it's visible.
- window   : open the result in a standalone Image Editor window.
- editor   : reuse an open Image Editor, else fall back to a new window.

All three first load/refresh the image datablock so re-runs update in place.
"""

from __future__ import annotations

import bpy


def load_result_image(path: str):
    """Load (or refresh) the result image datablock from `path`."""
    img = bpy.data.images.load(path, check_existing=True)
    img.reload()
    return img


def _find_image_editor(context):
    for area in context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            return area
    return None


def _switch_to_camera_view(context):
    for area in context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D" and space.region_3d is not None:
                space.region_3d.view_perspective = "CAMERA"
        return True
    return False


def _set_camera_background(context, img, alpha=1.0):
    cam = context.scene.camera
    if cam is None or cam.type != "CAMERA":
        return False
    cam_data = cam.data
    cam_data.show_background_images = True

    # Drop any stale FUK result overlays (different file, same role) to avoid stacking.
    for bg in list(cam_data.background_images):
        if bg.image and bg.image is not img and bpy.path.basename(bg.image.filepath) == "result.png":
            cam_data.background_images.remove(bg)

    bg = next((b for b in cam_data.background_images if b.image is img), None)
    if bg is None:
        bg = cam_data.background_images.new()
    bg.image = img
    bg.display_depth = "FRONT"
    bg.frame_method = "FIT"
    bg.alpha = alpha
    return True


def _open_image_window(context, img):
    try:
        bpy.ops.wm.window_new()
    except RuntimeError:
        return False
    win = context.window_manager.windows[-1]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "IMAGE_EDITOR"
    area.spaces.active.image = img
    return True


def show_result(context, img, mode="viewport", alpha=1.0):
    """Display `img` per `mode`. Returns a short human label of what happened."""
    if mode == "viewport":
        if _set_camera_background(context, img, alpha):
            _switch_to_camera_view(context)
            return "viewport (camera background)"
        # No camera — fall through to an editor.

    if mode in ("editor", "viewport"):
        area = _find_image_editor(context)
        if area is not None:
            area.spaces.active.image = img
            return "Image Editor"

    if _open_image_window(context, img):
        return "new window"

    return None
