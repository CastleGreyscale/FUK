"""
Blender-native control-pass rendering.

Renders the beauty image plus one structural control map in a single render:
  - beauty   : direct PNG via render.filepath (write_still)
  - depth    : Z pass -> Normalize -> File Output (EXR) -> OIIO -> normalized PNG
  - normals  : Normal pass -> File Output (EXR) -> OIIO (remap -1..1 -> 0..1) -> PNG
  - openpose : a user rig view-layer's Image -> File Output (EXR) -> OIIO -> PNG
  - canny    : a user line-work view-layer's Image, as rendered (no edge detection)
  - combined : any of depth / canny / openpose from the one render, layered into a
               single control image (depth_openpose, depth_canny_openpose, ...)

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

# Depth for the set, skeleton for the figure, in ONE map. The control-union pipeline
# takes a single context image (only control_image[0] reaches the model), so two
# controls can only be combined by compositing them before they leave Blender.
COMBINED = "depth_openpose"
# A control source names the maps that go into that one image, joined by "_" —
# "depth_canny_openpose" is all three. They are layered in this order, bottom to
# top: depth is the solid ground, canny lines sit on it, and the skeleton goes
# over both so a limb is never broken by an edge that crosses it.
PART_ORDER = ("depth", "normals", "canny", "openpose")
COMBINED_SOURCES = (COMBINED, "depth_canny", "canny_openpose", "depth_canny_openpose")


def control_parts(control_source):
    """The maps a control source is made of, in compositing order."""
    named = set((control_source or "").split("_"))
    return tuple(p for p in PART_ORDER if p in named)


# Controls that read the rig view layer / the canny view layer.
POSE_SOURCES = {"openpose"} | {c for c in COMBINED_SOURCES if "openpose" in control_parts(c)}
CANNY_SOURCES = {"canny"} | {c for c in COMBINED_SOURCES if "canny" in control_parts(c)}

# The rig's limb sticks are drawn at 0.6 of their joint colour (as controlnet_aux
# does), so 0.6 is the dimmest fully-covered skeleton pixel — see _pose_over.
_POSE_KEY = 0.6

_GEOMETRY_TYPES = {"MESH", "CURVE", "CURVES", "SURFACE", "META", "FONT",
                   "VOLUME", "GREASEPENCIL", "GPENCIL", "POINTCLOUD"}
# Objects that render nothing themselves, so losing them from a layer costs nothing
# unless something there depends on them (see _isolate_pose_geometry).
_KEEP_TYPES = {"LIGHT", "CAMERA", "LIGHT_PROBE", "SPEAKER"}


def _pose_layer(scene, name):
    """The rig view layer named `name`, or None if it is unset or gone."""
    return scene.view_layers.get(name) if name else None


def _require_layers(parts, pose_vl, canny_vl):
    """Refuse a combined control whose view layers are not all there.

    A single control can fall back — FUK estimates a pose or derives canny from
    the beauty — but a composite has no such route: the estimate would have to
    come from a beauty that need not hold a figure, and sending the parts that
    did render would pass for a weak control rather than a missing one.
    """
    if len(parts) < 2:
        return
    if "openpose" in parts and pose_vl is None:
        raise RuntimeError("This control needs a Rig Layer — pick the view layer "
                           "that renders the OpenPose skeleton")
    if "canny" in parts and canny_vl is None:
        raise RuntimeError("This control needs a Canny Layer — pick the view layer "
                           "that renders the line work")


def _layer_collections(lc):
    yield lc
    for child in lc.children:
        yield from _layer_collections(child)


def _isolate_pose_geometry(view_layer, pose_vl):
    """Drop the rig layer's skeleton geometry from the active layer for this render.

    A view layer only isolates in one direction: the rig layer excludes the set, but
    nothing stops the skeleton's collection from also being enabled in the main layer
    — it is, by default, for any collection created after the layer was. The emission
    sticks are then real geometry in the main render: they land in the Z pass (and so
    in the depth map) and in the beauty. hide_render can't fix it, being per-object
    rather than per-layer — it would empty the rig layer too.

    So: every collection that contributes geometry to the rig layer is excluded from
    the active layer for the render. Returns [(layer_collection, exclude)] to hand to
    _restore_layer_collections.

    A collection is left alone when it holds lights or a camera (it isn't a pure
    control collection), or an armature/empty that the active layer has no other
    route to — excluding that would drop it from this layer's depsgraph and silently
    freeze whatever it deforms at rest.
    """
    snap = []
    if pose_vl is None or pose_vl.name == view_layer.name:
        return snap
    main = {lc.collection: lc for lc in _layer_collections(view_layer.layer_collection)}
    for pose_lc in _layer_collections(pose_vl.layer_collection):
        col = pose_lc.collection
        main_lc = main.get(col)
        if (pose_lc.exclude or main_lc is None or main_lc.exclude
                or main_lc is view_layer.layer_collection):
            continue
        objs = list(col.objects)
        if not any(o.type in _GEOMETRY_TYPES for o in objs):
            continue
        if any(o.type in _KEEP_TYPES for o in col.all_objects):
            continue
        inside = set(col.children_recursive) | {col}
        helpers = [o for o in col.all_objects if o.type not in _GEOMETRY_TYPES]
        if any(all(c in inside for c in o.users_collection) for o in helpers):
            continue
        # Excluding a parent flips its children too; record the whole subtree.
        snap.extend((lc, lc.exclude) for lc in _layer_collections(main_lc))
        main_lc.exclude = True
    return snap


def _restore_layer_collections(view_layer, snap):
    if not snap:
        return
    for lc, exclude in snap:          # parents first, as recorded
        if lc.exclude != exclude:
            lc.exclude = exclude
    # Flush the re-inclusion now, while the caller still has live change-detection
    # suspended. Left to the next redraw, the re-added skeleton mesh reports a
    # geometry update that reads as a user edit and re-triggers Live forever.
    view_layer.update()


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


def _exr_to_rgb(exr_path, mode, far=1e9, depth_range=None):
    """Read a single-pass EXR as a display-ready float RGB array (0..1).

    `far` is the camera clip-end; depth pixels at/beyond it are the empty
    background (EEVEE writes Z=clip_end there, Cycles ~1e10) and are excluded
    from the range so geometry keeps its gradient. `depth_range` (lo, hi) forces a
    fixed normalization range (used for temporally-consistent video sequences).
    """
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
    return rgb


def _pose_over(pose_rgb, base_rgb):
    """Key the rig-layer skeleton over another control map.

    The rig layer renders on an opaque black world, so there is no alpha to use
    (film_transparent would give one, but it is scene-wide and would strip the
    world out of the beauty). Coverage is recovered from brightness instead: an
    antialiased edge pixel is the stick colour scaled by its coverage, and the
    dimmest solid pixel the rig draws is a limb at _POSE_KEY. The skeleton is
    already premultiplied by that coverage, hence pose + (1 - a) * base.
    """
    import numpy as np
    a = np.clip(pose_rgb.max(axis=-1, keepdims=True) / _POSE_KEY, 0.0, 1.0)
    return np.clip(pose_rgb + (1.0 - a) * base_rgb, 0.0, 1.0)


def _lines_over(lines_rgb, base_rgb):
    """Key a view layer of bright line work over another control map.

    The same idea as _pose_over with a different key: a canny map is white on
    black, so a pixel's own brightness is its coverage and there is no dimmer
    "solid" level to normalise against. The layer is expected to render its
    lines light on a black world; anything it leaves black lets the map
    underneath through.
    """
    import numpy as np
    a = np.clip(lines_rgb.max(axis=-1, keepdims=True), 0.0, 1.0)
    return np.clip(lines_rgb + (1.0 - a) * base_rgb, 0.0, 1.0)


def _composite(parts, maps):
    """Layer the rendered maps for `parts` into one image, or None if any is missing."""
    import numpy as np
    if any(maps.get(p) is None for p in parts):
        return None
    out = None
    for part in parts:                      # PART_ORDER: bottom to top
        rgb = maps[part]
        if out is None:
            out = rgb
        elif part == "openpose":
            out = _pose_over(rgb, out)
        elif part == "canny":
            out = _lines_over(rgb, out)
        else:
            out = np.clip(rgb, 0.0, 1.0)
    return out


def _write_png(rgb, png_path):
    import OpenImageIO as oiio
    import numpy as np

    out8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w = out8.shape[:2]
    out = oiio.ImageOutput.create(png_path)
    if out is None:
        raise RuntimeError(f"OIIO could not create {png_path}: {oiio.geterror()}")
    out.open(png_path, oiio.ImageSpec(w, h, 3, "uint8"))
    out.write_image(out8)
    out.close()
    return png_path


def _exr_to_png(exr_path, png_path, mode, far=1e9, depth_range=None):
    """Convert a single-pass EXR to an 8-bit PNG via OpenImageIO + numpy."""
    return _write_png(_exr_to_rgb(exr_path, mode, far, depth_range), png_path)


def render_passes(context, out_dir, control_source, preview=False,
                  preview_percentage=50, openpose_view_layer="", canny_view_layer=""):
    """
    Render beauty + the native control map for `control_source` into `out_dir`.

    Returns dict: {beauty, control, control_kind, width, height}.
    `control` is None when FUK must derive the map (canny without a canny layer,
    or openpose without a rig layer).
    """
    os.makedirs(out_dir, exist_ok=True)
    scene = context.scene
    view_layer = context.view_layer

    parts = control_parts(control_source)
    pose_vl = _pose_layer(scene, openpose_view_layer)
    canny_vl = _pose_layer(scene, canny_view_layer)
    _require_layers(parts, pose_vl, canny_vl)

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
    layer_use_snap = []     # [(view layer, use)] for the extra layers switched on
    isolated = []

    try:
        # Keep the skeleton out of this layer's beauty and Z pass, whichever control
        # is selected — a rig in the scene must not show up in a depth-only map.
        isolated = _isolate_pose_geometry(view_layer, pose_vl)

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

        want_depth = "depth" in parts
        want_norm = "normals" in parts
        want_pose_layer = "openpose" in parts and pose_vl is not None
        want_canny_layer = "canny" in parts and canny_vl is not None

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
        if want_depth or want_norm or want_pose_layer or want_canny_layer:
            # Drop last run's EXRs first — see _clear_exr.
            _clear_exr(out_dir, "ctl_depth", "ctl_normals", "ctl_pose", "ctl_canny")
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
            # Each extra view layer is rendered as-is and its Image written out: the
            # layer decides what a "pose" or a "canny" render looks like, not us.
            for wanted, vl, name in ((want_pose_layer, pose_vl, "ctl_pose"),
                                     (want_canny_layer, canny_vl, "ctl_canny")):
                if not wanted:
                    continue
                layer_use_snap.append((vl, vl.use))
                vl.use = True
                rl_extra = temp_group.nodes.new("CompositorNodeRLayers")
                rl_extra.scene = scene
                rl_extra.layer = vl.name
                fo = _add_exr_output(temp_group, name, out_dir, "RGBA")
                temp_group.links.new(rl_extra.outputs["Image"], fo.inputs[0])

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
        # Every part is written out under its own name, composite or not, so a map
        # that fails to latch can be traced to the part that is wrong.
        maps = {}
        for part, wanted, exr_name, mode in (
                ("depth", want_depth, "ctl_depth", "depth"),
                ("normals", want_norm, "ctl_normals", "normals"),
                ("canny", want_canny_layer, "ctl_canny", "color"),
                ("openpose", want_pose_layer, "ctl_pose", "color")):
            exr = _find_exr(out_dir, exr_name) if wanted else None
            if exr:
                maps[part] = _exr_to_rgb(exr, mode, far=far_clip)
                control_path = _write_png(maps[part], os.path.join(out_dir, f"{part}.png"))
        if len(parts) > 1:
            # All the parts or nothing — sending some alone would pass for a weak
            # control.
            combined = _composite(parts, maps)
            control_path = (_write_png(combined, os.path.join(out_dir, f"{control_source}.png"))
                            if combined is not None else None)

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
        for vl, use in layer_use_snap:
            vl.use = use
        if temp_group is not None:
            bpy.data.node_groups.remove(temp_group)
        _restore_layer_collections(view_layer, isolated)


# Native controls that can be rendered as a sequence (no per-frame server work).
# "canny" is here for its view-layer form; without a canny layer it would need
# the server to preprocess every frame, and the render refuses it.
SEQUENCE_CONTROLS = {"depth", "normals", "openpose", "canny", *COMBINED_SOURCES}


def render_control_sequence(context, out_dir, control_source, openpose_view_layer="",
                            percentage=100, canny_view_layer=""):
    """Render the control pass over the scene frame range into a folder of PNGs — the
    VACE control 'video'. Returns {control_dir, frames, width, height, fps}.
    `percentage` scales the render resolution (and thus the output video size).

    Depth is normalized over a GLOBAL range across the sequence so it doesn't flicker.
    Supports native controls (depth, normals, openpose rig layer, and depth + openpose
    composited per frame); canny / estimated openpose need per-frame server
    preprocessing and aren't supported here.
    """
    import shutil
    scene = context.scene
    view_layer = context.view_layer
    parts = control_parts(control_source)
    pose_vl = _pose_layer(scene, openpose_view_layer)
    canny_vl = _pose_layer(scene, canny_view_layer)

    # (convert mode, Render Layers socket, File Output socket type)
    main_pass = (("depth", "Depth", "FLOAT") if "depth" in parts else
                 ("normals", "Normal", "RGBA") if "normals" in parts else None)
    want_pose = "openpose" in parts
    want_canny = "canny" in parts
    if (not parts or (want_pose and pose_vl is None) or (want_canny and canny_vl is None)):
        raise RuntimeError(
            "Video control supports depth, normals, openpose (with a Rig Layer) and "
            "canny (with a Canny Layer). Estimated openpose and FUK-derived canny "
            "would need per-frame preprocessing and aren't supported for sequences."
        )
    mode = main_pass[0] if main_pass else "color"

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
    layer_use_snap = []
    isolated = []
    try:
        isolated = _isolate_pose_geometry(view_layer, pose_vl)

        # A sequencer strip would replace the render and skip the compositor entirely —
        # see the note in render_passes. Doubly important here: the FUK_result strip we
        # add after a video generation would otherwise feed the next one its own output.
        scene.render.use_sequencer = False

        scene.render.resolution_percentage = max(10, min(100, int(percentage)))
        if mode == "depth":
            view_layer.use_pass_z = True
        elif mode == "normals":
            view_layer.use_pass_normal = True

        temp_group = bpy.data.node_groups.new("FUK_SEQ_COMP", "CompositorNodeTree")
        if main_pass:
            rl = temp_group.nodes.new("CompositorNodeRLayers")
            rl.scene = scene
            rl.layer = view_layer.name
            fo = _add_exr_output(temp_group, "seq_main", exr_dir, main_pass[2])
            temp_group.links.new(rl.outputs[main_pass[1]], fo.inputs[0])
        for wanted, vl, name in ((want_canny, canny_vl, "seq_canny"),
                                 (want_pose, pose_vl, "seq_pose")):
            if not wanted:
                continue
            layer_use_snap.append((vl, vl.use))
            vl.use = True
            rl_extra = temp_group.nodes.new("CompositorNodeRLayers")
            rl_extra.scene = scene
            rl_extra.layer = vl.name
            fo = _add_exr_output(temp_group, name, exr_dir, "RGBA")
            temp_group.links.new(rl_extra.outputs["Image"], fo.inputs[0])

        scene.use_nodes = True
        scene.render.use_compositing = True
        scene.compositing_node_group = temp_group
        scene.render.image_settings.file_format = "PNG"      # throwaway main output
        scene.render.filepath = os.path.join(exr_dir, "_beauty_")

        bpy.ops.render.render(animation=True)

        # One list of frames per part, keyed like render_passes' `maps`.
        seqs = {}
        if main_pass:
            seqs[mode] = sorted(glob.glob(os.path.join(exr_dir, "seq_main*.exr")))
        if want_canny:
            seqs["canny"] = sorted(glob.glob(os.path.join(exr_dir, "seq_canny*.exr")))
        if want_pose:
            seqs["openpose"] = sorted(glob.glob(os.path.join(exr_dir, "seq_pose*.exr")))
        lengths = {k: len(v) for k, v in seqs.items()}
        if len(set(lengths.values())) > 1:
            # Frames are paired by index below; a short pass would shift every later
            # frame of one map onto the wrong frame of another without erroring.
            raise RuntimeError(f"Control passes disagree on length: {lengths}")
        exrs = next(iter(seqs.values()), [])

        # Wan only runs at 4n+1 frames and rounds UP internally. A control sequence of
        # any other length lands on a different temporal grid than the latents, and
        # VaceWanModel silently zero-pads the control tokens to fit — the control then
        # drifts out of sync instead of erroring. Drop the trailing frames (at most 3)
        # so what we send is exactly what the model will run.
        usable = len(exrs) - ((len(exrs) - 1) % 4) if len(exrs) >= 5 else len(exrs)
        seqs = {k: v[:usable] for k, v in seqs.items()}
        exrs = exrs[:usable]

        cam = scene.camera
        far = cam.data.clip_end if (cam and cam.type == "CAMERA") else 1e9
        depth_range = _global_depth_range(seqs["depth"], far) if "depth" in seqs else None
        for i in range(len(exrs)):
            maps = {part: _exr_to_rgb(frames[i], part if part in ("depth", "normals") else "color",
                                      far=far, depth_range=depth_range)
                    for part, frames in seqs.items()}
            rgb = _composite(parts, maps) if len(parts) > 1 else maps[parts[0]]
            _write_png(rgb, os.path.join(control_dir, f"f_{i:04d}.png"))

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
        for vl, use in layer_use_snap:
            vl.use = use
        if temp_group is not None:
            bpy.data.node_groups.remove(temp_group)
        _restore_layer_collections(view_layer, isolated)
        shutil.rmtree(exr_dir, ignore_errors=True)
