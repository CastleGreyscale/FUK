"""
Blender-native control-pass rendering.

Renders the beauty image plus one structural control map in a single render:
  - beauty   : direct PNG via render.filepath (write_still)
  - depth    : Z pass -> Normalize -> File Output (EXR) -> OIIO -> normalized PNG
  - normals  : Normal pass -> File Output (EXR) -> OIIO (remap -1..1 -> 0..1) -> PNG
  - openpose : a user rig view-layer's Image -> File Output (EXR) -> OIIO -> PNG

Blender 5.x notes (validated on 5.1):
  * The compositor is a node-group datablock assigned to `scene.compositing_node_group`
    (there is no `scene.node_tree`).
  * `CompositorNodeOutputFile` is locked to multilayer-EXR output, so we write EXR and
    convert to PNG with OpenImageIO (bundled with Blender).

Everything mutated on the scene is snapshotted and restored in a `finally`.
"""

from __future__ import annotations

import os
import glob
import bpy

# Controls FUK can derive itself (no native Blender pass written).
FUK_DERIVED = {"canny"}


def _eevee_engine():
    ids = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    for cand in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if cand in ids:
            return cand
    return None


def _add_exr_output(ng, name, out_dir, socket="FLOAT"):
    fo = ng.nodes.new("CompositorNodeOutputFile")
    fo.directory = out_dir
    fo.file_name = name
    fo.file_output_items.new(socket, name)  # creates the single real input socket
    return fo


def _clear_exr(out_dir, *names):
    """Delete previous EXRs for `names` so a stale one can't be mistaken for fresh.

    The compositor's File Output node is fire-and-forget: if it doesn't run — the
    node group failing to evaluate, the pass being unavailable on the active engine,
    a write erroring — nothing raises. The previous render's EXR is then still sitting
    in out_dir, and _find_exr would return it as the newest match. Generation would
    proceed from a control map minutes or hours out of date, which downstream looks
    exactly like a caching bug: the beauty updates, the control doesn't, and the VAE
    encode cache correctly reports a hit on unchanged bytes. Clearing first turns that
    silent wrong result into a visible "no control map produced" warning.
    """
    for name in names:
        for path in glob.glob(os.path.join(out_dir, f"{name}*.exr")):
            try:
                os.remove(path)
            except OSError:
                pass


def _find_exr(out_dir, name):
    matches = sorted(glob.glob(os.path.join(out_dir, f"{name}*.exr")), key=os.path.getmtime)
    return matches[-1] if matches else None


def _read_exr(exr_path):
    import OpenImageIO as oiio
    import numpy as np
    src = oiio.ImageInput.open(exr_path)
    if src is None:
        raise RuntimeError(f"OIIO could not open {exr_path}: {oiio.geterror()}")
    spec = src.spec()
    pixels = src.read_image(format="float")
    src.close()
    if pixels is None:
        raise RuntimeError(f"OIIO read failed for {exr_path}: {oiio.geterror()}")
    return np.array(pixels).reshape(spec.height, spec.width, spec.nchannels)


def _depth_valid_mask(ch, far):
    import numpy as np
    return np.isfinite(ch) & (ch < far * 0.999) & (ch < 1e9)


def _linear_to_srgb(x):
    """Scene-linear -> sRGB (the IEC 61966-2-1 OETF, i.e. Blender's 'Standard')."""
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _global_depth_range(exr_paths, far):
    """Min/max of valid depth across a whole sequence — so per-frame normalization
    doesn't flicker as the object's depth range changes frame to frame."""
    import numpy as np
    lo, hi = float("inf"), float("-inf")
    for p in exr_paths:
        ch = _read_exr(p)[..., 0].astype(np.float32)
        valid = _depth_valid_mask(ch, far)
        if valid.any():
            lo = min(lo, float(ch[valid].min()))
            hi = max(hi, float(ch[valid].max()))
    return (lo, hi) if hi > lo else None


def _exr_to_png(exr_path, png_path, mode, far=1e9, depth_range=None):
    """Convert a single-pass EXR to an 8-bit PNG via OpenImageIO + numpy.

    `far` is the camera clip-end; depth pixels at/beyond it are the empty
    background (EEVEE writes Z=clip_end there, Cycles ~1e10) and are excluded
    from the range so geometry keeps its gradient. `depth_range` (lo, hi) forces a
    fixed normalization range (used for temporally-consistent video sequences).
    """
    import OpenImageIO as oiio
    import numpy as np

    a = _read_exr(exr_path)
    if mode == "depth":
        # Raw camera Z. Background/sky has no geometry -> a huge sentinel (~1e10)
        # or non-finite value. Exclude it from the range so the actual geometry
        # keeps a smooth gradient instead of crushing to near-black against a
        # white void (the "pure black and white" failure).
        ch = a[..., 0].astype(np.float32)
        valid = _depth_valid_mask(ch, far)
        if depth_range is not None:
            lo, hi = depth_range
        elif valid.any():
            lo, hi = float(ch[valid].min()), float(ch[valid].max())
        else:
            lo, hi = 0.0, 1.0
        norm = np.clip((ch - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(ch)
        # Match the Depth-Anything convention the control model expects:
        # near = white, far = black. Invert, and push the empty background to far.
        depth = 1.0 - norm
        depth[~valid] = 0.0
        rgb = np.stack([depth, depth, depth], axis=-1)
    elif mode == "normals":
        rgb = np.clip(a[..., :3] * 0.5 + 0.5, 0.0, 1.0)
    else:  # colour (openpose rig render)
        # The compositor's File Output writes OpenEXR scene-LINEAR — no view
        # transform. Blender's own F12 PNG does get one ('Standard' == the sRGB
        # OETF), which is why the rig looks right rendered from Blender and wrong
        # coming through here. Writing those linear floats straight to 8-bit
        # linearizes the rig's colours a second time: #ff5500 lands at #ff1700,
        # #aaff00 at #67ff00, and every 0.6-attenuated limb at 81 instead of 153.
        # draw_bodypose's 18-colour palette IS the pose encoder's whole vocabulary
        # — a hue shift that large reads as a different keypoint, or as none — so
        # the control map silently fails to latch instead of erroring. Encode.
        rgb = _linear_to_srgb(a[..., :3])

    out8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w = out8.shape[:2]
    out = oiio.ImageOutput.create(png_path)
    if out is None:
        raise RuntimeError(f"OIIO could not create {png_path}: {oiio.geterror()}")
    out.open(png_path, oiio.ImageSpec(w, h, 3, "uint8"))
    out.write_image(out8)
    out.close()
    return png_path


def render_passes(context, out_dir, control_source, preview=False,
                  preview_percentage=50, openpose_view_layer=""):
    """
    Render beauty + the native control map for `control_source` into `out_dir`.

    Returns dict: {beauty, control, control_kind, width, height}.
    `control` is None when FUK must derive the map (canny, or openpose without a rig layer).
    """
    os.makedirs(out_dir, exist_ok=True)
    scene = context.scene
    view_layer = context.view_layer

    # --- snapshot ---
    snap = {
        "engine": scene.render.engine,
        "pct": scene.render.resolution_percentage,
        "filepath": scene.render.filepath,
        "ffmt": scene.render.image_settings.file_format,
        "cmode": scene.render.image_settings.color_mode,
        "cdepth": scene.render.image_settings.color_depth,
        "use_nodes": scene.use_nodes,
        "use_comp": scene.render.use_compositing,
        "use_seq": scene.render.use_sequencer,
        "comp_group": scene.compositing_node_group,
        "pass_z": view_layer.use_pass_z,
        "pass_n": view_layer.use_pass_normal,
        "layer_use": view_layer.use,
    }
    temp_group = None
    pose_layer_snap = None

    try:
        # Blender renders Render Layers -> Compositor -> Sequencer. With any strip in
        # the sequencer, its output REPLACES the render: the compositor is skipped (so
        # no control EXR is written) and the beauty PNG becomes a frame of the strip.
        # We add a strip ourselves in viewer.show_video, so generating a video silently
        # poisoned every later still — stale beauty AND stale control, which looks
        # exactly like a caching bug. Our renders must never route through it.
        scene.render.use_sequencer = False

        if preview:
            eevee = _eevee_engine()
            if eevee:
                scene.render.engine = eevee
            scene.render.resolution_percentage = max(10, min(100, preview_percentage))

        want_depth = control_source == "depth"
        want_norm = control_source == "normals"
        want_pose_layer = (
            control_source == "openpose"
            and openpose_view_layer
            and openpose_view_layer in scene.view_layers
        )

        if want_depth:
            view_layer.use_pass_z = True
        if want_norm:
            view_layer.use_pass_normal = True

        # The active view layer must actually render, or its Render Layers node feeds
        # the File Output node nothing and no EXR is written — silently. (The pose rig
        # layer below already gets this treatment; the main layer never did.)
        if not view_layer.use:
            view_layer.use = True

        # Build a fresh compositor node group for the native passes we need.
        if want_depth or want_norm or want_pose_layer:
            # Drop last run's EXRs first — see _clear_exr.
            _clear_exr(out_dir, "ctl_depth", "ctl_normals", "ctl_pose")
            temp_group = bpy.data.node_groups.new("FUK_BLENDER_COMP", "CompositorNodeTree")
            rl = temp_group.nodes.new("CompositorNodeRLayers")
            rl.scene = scene
            rl.layer = view_layer.name

            if want_depth:
                # Write the RAW Z pass (no Normalize node) so the EXR keeps real
                # distances; we mask the void and normalize in numpy on convert.
                fo = _add_exr_output(temp_group, "ctl_depth", out_dir, "FLOAT")
                temp_group.links.new(rl.outputs["Depth"], fo.inputs[0])
            if want_norm:
                fo = _add_exr_output(temp_group, "ctl_normals", out_dir, "RGBA")
                temp_group.links.new(rl.outputs["Normal"], fo.inputs[0])
            if want_pose_layer:
                pose_vl = scene.view_layers[openpose_view_layer]
                pose_layer_snap = pose_vl.use
                pose_vl.use = True
                rl_pose = temp_group.nodes.new("CompositorNodeRLayers")
                rl_pose.scene = scene
                rl_pose.layer = openpose_view_layer
                fo = _add_exr_output(temp_group, "ctl_pose", out_dir, "RGBA")
                temp_group.links.new(rl_pose.outputs["Image"], fo.inputs[0])

            scene.use_nodes = True
            scene.render.use_compositing = True
            scene.compositing_node_group = temp_group

        # Beauty straight to PNG (active view layer's render result).
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.filepath = os.path.join(out_dir, "beauty")

        bpy.ops.render.render(write_still=True)

        # write_still writes "<filepath><ext>" (no frame number for a still here);
        # fall back to a glob in case the Blender build pads the frame.
        beauty = bpy.path.abspath(scene.render.filepath) + scene.render.file_extension
        if not os.path.exists(beauty):
            cand = sorted(glob.glob(os.path.join(out_dir, "beauty*.png")), key=os.path.getmtime)
            beauty = cand[-1] if cand else beauty
        rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
        ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

        cam = scene.camera
        far_clip = cam.data.clip_end if (cam and cam.type == "CAMERA") else 1e9

        control_path = None
        control_kind = control_source
        if want_depth:
            exr = _find_exr(out_dir, "ctl_depth")
            if exr:
                control_path = _exr_to_png(exr, os.path.join(out_dir, "depth.png"), "depth", far=far_clip)
        elif want_norm:
            exr = _find_exr(out_dir, "ctl_normals")
            if exr:
                control_path = _exr_to_png(exr, os.path.join(out_dir, "normals.png"), "normals")
        elif want_pose_layer:
            exr = _find_exr(out_dir, "ctl_pose")
            if exr:
                control_path = _exr_to_png(exr, os.path.join(out_dir, "openpose.png"), "color")

        return {
            "beauty": beauty,
            "control": control_path,
            "control_kind": control_kind,
            "width": rx,
            "height": ry,
        }

    finally:
        # --- restore everything we touched ---
        scene.render.engine = snap["engine"]
        scene.render.resolution_percentage = snap["pct"]
        scene.render.filepath = snap["filepath"]
        scene.render.image_settings.file_format = snap["ffmt"]
        scene.render.image_settings.color_mode = snap["cmode"]
        scene.render.image_settings.color_depth = snap["cdepth"]
        scene.compositing_node_group = snap["comp_group"]
        scene.use_nodes = snap["use_nodes"]
        scene.render.use_compositing = snap["use_comp"]
        scene.render.use_sequencer = snap["use_seq"]
        view_layer.use_pass_z = snap["pass_z"]
        view_layer.use_pass_normal = snap["pass_n"]
        view_layer.use = snap["layer_use"]
        if pose_layer_snap is not None and openpose_view_layer in scene.view_layers:
            scene.view_layers[openpose_view_layer].use = pose_layer_snap
        if temp_group is not None:
            bpy.data.node_groups.remove(temp_group)


# Native controls that can be rendered as a sequence (no per-frame server work).
SEQUENCE_CONTROLS = {"depth", "normals", "openpose"}


def render_control_sequence(context, out_dir, control_source, openpose_view_layer="", percentage=100):
    """Render the control pass over the scene frame range into a folder of PNGs — the
    VACE control 'video'. Returns {control_dir, frames, width, height, fps}.
    `percentage` scales the render resolution (and thus the output video size).

    Depth is normalized over a GLOBAL range across the sequence so it doesn't flicker.
    Supports native controls (depth, normals, openpose rig layer); canny / estimated
    openpose need per-frame server preprocessing and aren't supported here.
    """
    import shutil
    scene = context.scene
    view_layer = context.view_layer

    if control_source == "depth":
        mode, socket, socket_type = "depth", "Depth", "FLOAT"
    elif control_source == "normals":
        mode, socket, socket_type = "normals", "Normal", "RGBA"
    elif control_source == "openpose" and openpose_view_layer and openpose_view_layer in scene.view_layers:
        mode, socket, socket_type = "color", "Image", "RGBA"
    else:
        raise RuntimeError(
            "Video control supports depth, normals, or openpose (with a rig layer). "
            "Canny / estimated openpose aren't supported for sequences."
        )

    exr_dir = os.path.join(out_dir, "_ctl_exr")
    control_dir = os.path.join(out_dir, "control")
    # Clear both first. exr_dir is normally removed in the `finally` below, but a run
    # that died before it (or was force-quit) leaves frames behind — and `frames` is
    # just len(glob("seq*.exr")), so leftovers would silently inflate the video length
    # and pair the wrong control frame with each output frame.
    shutil.rmtree(exr_dir, ignore_errors=True)
    for d in (exr_dir, control_dir):
        os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(control_dir, "*.png")):
        try:
            os.remove(f)
        except OSError:
            pass

    snap = {
        "pct": scene.render.resolution_percentage,
        "filepath": scene.render.filepath,
        "ffmt": scene.render.image_settings.file_format,
        "use_nodes": scene.use_nodes,
        "use_comp": scene.render.use_compositing,
        "use_seq": scene.render.use_sequencer,
        "comp_group": scene.compositing_node_group,
        "pass_z": view_layer.use_pass_z,
        "pass_n": view_layer.use_pass_normal,
        "frame": scene.frame_current,
    }
    temp_group = None
    pose_layer_snap = None
    try:
        # A sequencer strip would replace the render and skip the compositor entirely —
        # see the note in render_passes. Doubly important here: the FUK_result strip we
        # add after a video generation would otherwise feed the next one its own output.
        scene.render.use_sequencer = False

        scene.render.resolution_percentage = max(10, min(100, int(percentage)))
        if mode == "depth":
            view_layer.use_pass_z = True
        elif mode == "normals":
            view_layer.use_pass_normal = True

        layer_name = view_layer.name
        if mode == "color":
            pose_vl = scene.view_layers[openpose_view_layer]
            pose_layer_snap = pose_vl.use
            pose_vl.use = True
            layer_name = openpose_view_layer

        temp_group = bpy.data.node_groups.new("FUK_SEQ_COMP", "CompositorNodeTree")
        rl = temp_group.nodes.new("CompositorNodeRLayers")
        rl.scene = scene
        rl.layer = layer_name
        fo = _add_exr_output(temp_group, "seq", exr_dir, socket_type)
        temp_group.links.new(rl.outputs[socket], fo.inputs[0])

        scene.use_nodes = True
        scene.render.use_compositing = True
        scene.compositing_node_group = temp_group
        scene.render.image_settings.file_format = "PNG"      # throwaway main output
        scene.render.filepath = os.path.join(exr_dir, "_beauty_")

        bpy.ops.render.render(animation=True)

        exrs = sorted(glob.glob(os.path.join(exr_dir, "seq*.exr")))

        # Wan only runs at 4n+1 frames and rounds UP internally. A control sequence of
        # any other length lands on a different temporal grid than the latents, and
        # VaceWanModel silently zero-pads the control tokens to fit — the control then
        # drifts out of sync instead of erroring. Drop the trailing frames (at most 3)
        # so what we send is exactly what the model will run.
        usable = len(exrs) - ((len(exrs) - 1) % 4) if len(exrs) >= 5 else len(exrs)
        exrs = exrs[:usable]

        cam = scene.camera
        far = cam.data.clip_end if (cam and cam.type == "CAMERA") else 1e9
        depth_range = _global_depth_range(exrs, far) if mode == "depth" else None
        for i, exr in enumerate(exrs):
            _exr_to_png(exr, os.path.join(control_dir, f"f_{i:04d}.png"),
                        mode, far=far, depth_range=depth_range)

        rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
        ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
        return {
            "control_dir": control_dir,
            "frames": len(exrs),
            "width": rx,
            "height": ry,
            "fps": scene.render.fps,
        }

    finally:
        scene.render.resolution_percentage = snap["pct"]
        scene.render.filepath = snap["filepath"]
        scene.render.image_settings.file_format = snap["ffmt"]
        scene.compositing_node_group = snap["comp_group"]
        scene.use_nodes = snap["use_nodes"]
        scene.render.use_compositing = snap["use_comp"]
        scene.render.use_sequencer = snap["use_seq"]
        view_layer.use_pass_z = snap["pass_z"]
        view_layer.use_pass_normal = snap["pass_n"]
        scene.frame_current = snap["frame"]
        if pose_layer_snap is not None and openpose_view_layer in scene.view_layers:
            scene.view_layers[openpose_view_layer].use = pose_layer_snap
        if temp_group is not None:
            bpy.data.node_groups.remove(temp_group)
        shutil.rmtree(exr_dir, ignore_errors=True)
