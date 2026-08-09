"""
Show a generated result back inside Blender.

- viewport : set the result as the active camera's background overlay (display in
             front), and switch the 3D view to camera view so it's visible. Video
             results go in as a movie clip, so they play with the timeline.
- window   : open the result in a standalone Image Editor / Sequencer window.
- editor   : reuse an open editor, else fall back to a new window.

All paths first load/refresh the datablock so re-runs update in place, then tag the
viewport for redraw — see _tag_redraw.
"""

from __future__ import annotations

import os
import bpy

# Basenames of the artifacts FUK parks in the camera's background slots. Used to clear
# our own overlays without touching background images the user set up themselves.
_FUK_ARTIFACTS = {"result.png", "preview.png", "result.mp4"}


def _tag_redraw(context):
    """Force the 3D viewports and image editors to repaint.

    Nothing else will. The generate operators are modal timer loops, and a timer event
    only redraws the area that handles it — so swapping the camera-background image
    mid-run changed the datablock but left the viewport showing its last paint. That is
    why live diffusion previews looked blank: they were arriving and being installed,
    just never drawn until some unrelated event (a mouse move) forced a repaint, by
    which point the final result had already replaced them.
    """
    wm = getattr(context, "window_manager", None)
    windows = list(getattr(wm, "windows", ())) if wm else []
    if not windows and getattr(context, "window", None):
        windows = [context.window]
    for win in windows:
        screen = getattr(win, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in ("VIEW_3D", "IMAGE_EDITOR", "SEQUENCE_EDITOR"):
                area.tag_redraw()


def load_result_image(path: str):
    """Load (or refresh) the result image datablock from `path`."""
    img = bpy.data.images.load(path, check_existing=True)
    img.reload()
    return img


def load_result_clip(path: str):
    """Load the result movie clip, replacing any stale datablock for the same file.

    Unlike Image, MovieClip has no reload() — refreshing one is an operator needing a
    Clip Editor context. Since result.mp4 is overwritten every run, the only reliable
    way to avoid showing the previous video is to drop the old datablock and load the
    file again. Any background slot referencing it is re-pointed straight after.
    """
    target = os.path.normpath(bpy.path.abspath(path))
    for clip in list(bpy.data.movieclips):
        if os.path.normpath(bpy.path.abspath(clip.filepath)) == target:
            bpy.data.movieclips.remove(clip)
    return bpy.data.movieclips.load(target)


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


def _is_fuk_background(bg):
    """True for a background slot holding one of our own artifacts."""
    for datablock in (bg.image, bg.clip):
        if datablock and bpy.path.basename(datablock.filepath) in _FUK_ARTIFACTS:
            return True
    return False


def _camera_background(context, alpha, keep=None):
    """Clear our stale overlays and return a background slot ready to fill.

    Returns None when there's no camera. `keep` is the datablock the caller is about to
    install — its existing slot is reused rather than removed, so preview → preview
    doesn't churn a slot every poll.
    """
    cam = context.scene.camera
    if cam is None or cam.type != "CAMERA":
        return None
    cam_data = cam.data
    cam_data.show_background_images = True

    # Drop stale FUK overlays (still, live-diffusion preview, video) so switching
    # between them reuses a single slot instead of stacking.
    existing = None
    for bg in list(cam_data.background_images):
        if keep is not None and (bg.image is keep or bg.clip is keep):
            existing = bg
        elif _is_fuk_background(bg):
            cam_data.background_images.remove(bg)

    bg = existing or cam_data.background_images.new()
    bg.display_depth = "FRONT"
    bg.frame_method = "FIT"
    bg.alpha = alpha
    return bg


def _set_camera_background(context, img, alpha=1.0):
    bg = _camera_background(context, alpha, keep=img)
    if bg is None:
        return False
    bg.source = "IMAGE"
    bg.image = img
    return True


def _set_camera_background_clip(context, clip, alpha=1.0):
    """Overlay a movie clip, so the result plays as you scrub the timeline."""
    bg = _camera_background(context, alpha, keep=clip)
    if bg is None:
        return False
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    # Line the clip up with the frames it was rendered from, so scrubbing shows the
    # generated frame for the scene frame under the playhead. frame_start lives on the
    # clip datablock — MovieClipUser only carries frame_current/proxy settings.
    clip.frame_start = context.scene.frame_start
    bg.clip_user.use_render_undistorted = True
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


def _show_video_in_sequencer(context, mp4_path):
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


def show_video(context, mp4_path, mode="viewport", alpha=1.0):
    """Display the result mp4 per `mode`, mirroring show_result for stills.

    In `viewport` the clip becomes the camera background, so scrubbing the timeline
    plays the generated video straight over your scene — the same overlay the still
    flow uses, and the opacity slider works on it identically.
    """
    label = None
    if mode == "viewport":
        try:
            clip = load_result_clip(mp4_path)
        except RuntimeError:
            clip = None
        if clip is not None and _set_camera_background_clip(context, clip, alpha):
            _switch_to_camera_view(context)
            label = "viewport (camera background)"

    if label is None:
        label = _show_video_in_sequencer(context, mp4_path)

    _tag_redraw(context)
    return label


def show_result(context, img, mode="viewport", alpha=1.0, reuse_only=False):
    """Display `img` per `mode`. Returns a short human label of what happened.

    `reuse_only` (used for mid-diffusion previews) updates only an already-open
    display — it never opens a new window, so previews don't spawn one per frame.
    """
    if mode == "viewport":
        if _set_camera_background(context, img, alpha):
            _switch_to_camera_view(context)
            _tag_redraw(context)
            return "viewport (camera background)"
        # No camera — fall through to an editor.

    if mode in ("editor", "viewport") or reuse_only:
        area = _find_image_editor(context)
        if area is not None:
            area.spaces.active.image = img
            _tag_redraw(context)
            return "Image Editor"

    if reuse_only:
        return None  # don't pop a new window for every preview frame

    if _open_image_window(context, img):
        _tag_redraw(context)
        return "new window"

    return None
