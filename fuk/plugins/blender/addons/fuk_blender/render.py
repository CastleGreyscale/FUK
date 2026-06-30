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


def _find_exr(out_dir, name):
    matches = sorted(glob.glob(os.path.join(out_dir, f"{name}*.exr")), key=os.path.getmtime)
    return matches[-1] if matches else None


def _exr_to_png(exr_path, png_path, mode, far=1e9):
    """Convert a single-pass EXR to an 8-bit PNG via OpenImageIO + numpy.

    `far` is the camera clip-end; depth pixels at/beyond it are the empty
    background (EEVEE writes Z=clip_end there, Cycles ~1e10) and are excluded
    from the range so geometry keeps its gradient.
    """
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

    a = np.array(pixels).reshape(spec.height, spec.width, spec.nchannels)
    if mode == "depth":
        # Raw camera Z. Background/sky has no geometry -> a huge sentinel (~1e10)
        # or non-finite value. Exclude it from the range so the actual geometry
        # keeps a smooth gradient instead of crushing to near-black against a
        # white void (the "pure black and white" failure).
        ch = a[..., 0].astype(np.float32)
        valid = np.isfinite(ch) & (ch < far * 0.999) & (ch < 1e9)
        if valid.any():
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
    else:  # passthrough colour (openpose rig render)
        rgb = np.clip(a[..., :3], 0.0, 1.0)

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
        "comp_group": scene.compositing_node_group,
        "pass_z": view_layer.use_pass_z,
        "pass_n": view_layer.use_pass_normal,
    }
    temp_group = None
    pose_layer_snap = None

    try:
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

        # Build a fresh compositor node group for the native passes we need.
        if want_depth or want_norm or want_pose_layer:
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
        view_layer.use_pass_z = snap["pass_z"]
        view_layer.use_pass_normal = snap["pass_n"]
        if pose_layer_snap is not None and openpose_view_layer in scene.view_layers:
            scene.view_layers[openpose_view_layer].use = pose_layer_snap
        if temp_group is not None:
            bpy.data.node_groups.remove(temp_group)
