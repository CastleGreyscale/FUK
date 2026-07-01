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

    # Drop any stale FUK overlays (result + live-diffusion preview) to avoid stacking,
    # so switching preview <-> final reuses a single background slot.
    for bg in list(cam_data.background_images):
        if bg.image and bg.image is not img and bpy.path.basename(bg.image.filepath) in ("result.png", "preview.png"):
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


def _sequencer_strips(scene):
    se = scene.sequence_editor_create()
    if hasattr(se, "strips"):      # Blender 5.x
        return se.strips
    if hasattr(se, "sequences"):   # older
        return se.sequences
    return None


def _open_sequencer_window(context):
    try:
        bpy.ops.wm.window_new()
    except RuntimeError:
        return False
    win = context.window_manager.windows[-1]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "SEQUENCE_EDITOR"
    for space in area.spaces:
        if space.type == "SEQUENCE_EDITOR":
            try:
                space.view_type = "SEQUENCER_PREVIEW"
            except Exception:
                pass
            break
    return True


def show_video(context, mp4_path):
    """Load the result mp4 into the scene's Video Sequencer and reveal it."""
    scene = context.scene
    strips = _sequencer_strips(scene)
    if strips is None:
        return None
    # Replace any prior FUK result strip.
    for s in list(strips):
        if s.name.startswith("FUK_result"):
            try:
                strips.remove(s)
            except Exception:
                pass
    try:
        strip = strips.new_movie(name="FUK_result", filepath=mp4_path,
                                 channel=1, frame_start=scene.frame_start)
    except Exception:
        return None
    try:  # extend the scene range so playback covers the whole clip
        scene.frame_end = max(scene.frame_end, int(strip.frame_final_end) - 1)
    except Exception:
        pass
    # Reuse an open sequencer if there is one; else pop a window.
    if not any(a.type == "SEQUENCE_EDITOR" for a in context.screen.areas):
        _open_sequencer_window(context)
    return "Video Sequencer"


def show_result(context, img, mode="viewport", alpha=1.0, reuse_only=False):
    """Display `img` per `mode`. Returns a short human label of what happened.

    `reuse_only` (used for mid-diffusion previews) updates only an already-open
    display — it never opens a new window, so previews don't spawn one per frame.
    """
    if mode == "viewport":
        if _set_camera_background(context, img, alpha):
            _switch_to_camera_view(context)
            return "viewport (camera background)"
        # No camera — fall through to an editor.

    if mode in ("editor", "viewport") or reuse_only:
        area = _find_image_editor(context)
        if area is not None:
            area.spaces.active.image = img
            return "Image Editor"

    if reuse_only:
        return None  # don't pop a new window for every preview frame

    if _open_image_window(context, img):
        return "new window"

    return None
