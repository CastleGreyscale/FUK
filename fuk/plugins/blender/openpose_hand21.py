#!/usr/bin/env python3
"""
Convert the 21-keypoint hand rig to the colour convention that controlnet_aux /
DWPose render, which is what FUK's openpose preprocessor emits with
detect_hand=True and therefore what the pose encoder was conditioned on.

The companion to openpose_coco18.py, which does the same job for the body rig.

Usage:
    blender Hand.blend --background --factory-startup \
        --python openpose_hand21.py -- --out Hand_OP21.blend

    # overwrite the source file instead (makes a .bak first)
    blender Hand.blend --background --python openpose_hand21.py -- --inplace

Two things differ from the body conversion, both from draw_handpose:

  * Every one of the 21 keypoints is the SAME colour -- pure blue. The body's
    per-joint rainbow has no hand equivalent, so joint numbering is irrelevant
    here (which is lucky: this rig numbers hand.000-020 in serpentine order,
    not OpenPose order).

  * Bones are NOT attenuated. draw_bodypose fills limbs with fillConvexPoly and
    then addWeighted(0.6); draw_handpose just calls cv2.line at full value. The
    body script's BONE_ALPHA must not be carried over here.
"""

import re
import sys
import shutil
from pathlib import Path

import bpy


# --- colour management -------------------------------------------------------

def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# --- controlnet_aux / DWPose hand tables -------------------------------------
# venv/lib/python3.10/site-packages/controlnet_aux/open_pose/util.py:draw_handpose

# Bone colour: `hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255`, a full
# rainbow sweep over the 20 edges. Reimplemented rather than imported -- Blender
# ships no matplotlib, and for S=V=1 the formula is exact (verified identical to
# matplotlib.colors.hsv_to_rgb for all 20 hues).
def hsv_edge(ie, n_edges=20):
    h = ie / float(n_edges)
    sector = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    q, t = 1.0 - f, f
    return [(1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q)][sector]


# All 21 keypoints: `cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)`. The canvas
# is RGB (the detector's output is handed straight to PIL), so this is blue.
JOINT_COLOR = (0.0, 0.0, 1.0)

# draw_handpose's edge list, in order, as (parent bone -> edge index). The Nth
# edge takes COLOR[N]; the rig's bone curves are parented to the armature bone
# they follow, so parent_bone is the join key -- far more robust than the
# handBone.NNN suffix, which is in neither edge nor anatomical order.
#
#   edges = [[0,1],[1,2],[2,3],[3,4],  [0,5],[5,6],[6,7],[7,8],
#            [0,9],[9,10],[10,11],[11,12],  [0,13],[13,14],[14,15],[15,16],
#            [0,17],[17,18],[18,19],[19,20]]
#
# Keypoints 1-4 are the thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
# In this rig the thumb chain starts at Hand.L (wrist->thumb CMC is edge 0), and
# Finger0..Finger3 are index, middle, ring, pinky outward from the thumb.
BONE_EDGE = {
    "Hand":      0,   # edge 0   wrist -> thumb CMC
    "Thumb1":    1,   # edge 1   thumb CMC -> MCP
    "Thumb2":    2,   # edge 2   thumb MCP -> IP
    "Thumb3":    3,   # edge 3   thumb IP  -> tip
    "Finger0_0": 4,   # edge 4   wrist -> index MCP
    "Finger0_1": 5,
    "Finger0_2": 6,
    "Finger0_3": 7,
    "Finger1_0": 8,   # edge 8   wrist -> middle MCP
    "Finger1_1": 9,
    "Finger1_2": 10,
    "Finger1_3": 11,
    "Finger2_0": 12,  # edge 12  wrist -> ring MCP
    "Finger2_1": 13,
    "Finger2_2": 14,
    "Finger2_3": 15,
    "Finger3_0": 16,  # edge 16  wrist -> pinky MCP
    "Finger3_1": 17,
    "Finger3_2": 18,
    "Finger3_3": 19,
}

# Modelling aids that must never reach the render. HandReference is the photo
# plane the rig was posed against -- left visible it paints a literal photograph
# of a hand into the control map. handBoneBevel is the profile curve the bone
# tubes sweep; it carries no material and is degenerate, but it is still a
# renderable object sitting at the wrist.
NON_POSE_OBJECTS = ["HandReference", "handBoneBevel"]

JOINT_RE = re.compile(r"^hand\.\d+$")
BONE_RE = re.compile(r"^handBone\.\d+$")
# Bones may be side-suffixed (Hand.L / Hand.R) if the rig is mirrored.
SIDE_RE = re.compile(r"\.[LR]$")


def set_color(obj, srgb):
    obj.color = (*(srgb_to_linear(c) for c in srgb), 1.0)


def make_material_opaque(mat):
    """
    Drive the surface straight from object colour, with no alpha anywhere.

    draw_handpose composites nothing -- cv2.line and cv2.circle both write the
    colour in flat. Any Mix/Transparent in the shader would attenuate it, and on
    an unculled tube both walls composite, so the rendered value drifts with
    silhouette thickness. (Stock JointColor is already ObjectInfo -> Surface;
    this is here so a rig that isn't gets the same treatment.)

    Backface culling is deliberately NOT forced on, unlike the body script: that
    was needed there only because the limbs were alpha-blended. These surfaces
    are opaque, so the depth buffer already resolves them -- and the bone tubes
    are swept curves whose end caps depend on the bevel settings, so culling
    could open holes rather than close them.
    """
    if not mat.use_nodes:
        return
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    info = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeObjectInfo"), None)
    if out is None or info is None:
        return

    for node in [n for n in nt.nodes if n.bl_idname in
                 ("ShaderNodeMixShader", "ShaderNodeBsdfTransparent")]:
        nt.nodes.remove(node)

    nt.links.new(info.outputs["Color"], out.inputs["Surface"])
    for value in ("OPAQUE", "NONE"):
        try:
            mat.blend_method = value
            break
        except TypeError:
            continue


def convert():
    changed, missing, hidden = [], [], []

    # --- 21 keypoints, all pure blue ---
    joints = sorted(o for o in bpy.data.objects.keys() if JOINT_RE.match(o))
    for name in joints:
        set_color(bpy.data.objects[name], JOINT_COLOR)
    if joints:
        changed.append(f"  joints {len(joints):2d} objects    -> #0000ff (all keypoints)")
    if len(joints) != 21:
        missing.append(f"expected 21 hand.NNN joints, found {len(joints)}")

    # --- 20 bones, rainbow by edge index ---
    bones = sorted(o for o in bpy.data.objects.keys() if BONE_RE.match(o))
    seen = {}
    for name in bones:
        obj = bpy.data.objects[name]
        if obj.parent_type != "BONE" or not obj.parent_bone:
            missing.append(f"{name} is not bone-parented; cannot identify its edge")
            continue
        key = SIDE_RE.sub("", obj.parent_bone)
        ie = BONE_EDGE.get(key)
        if ie is None:
            missing.append(f"{name} follows unknown bone {obj.parent_bone!r}")
            continue
        if ie in seen:
            missing.append(f"edge {ie} claimed by both {seen[ie]} and {name}")
            continue
        seen[ie] = name
        srgb = hsv_edge(ie)
        set_color(obj, srgb)
        hexcode = "".join(f"{int(round(c * 255)):02x}" for c in srgb)
        changed.append(f"  bone  {name:14s} {obj.parent_bone:13s} edge {ie:2d} -> #{hexcode}")

    for ie in range(20):
        if ie not in seen:
            missing.append(f"no bone curve found for edge {ie}")

    for mat_name in {m for o in bpy.data.objects
                     if (JOINT_RE.match(o.name) or BONE_RE.match(o.name)) and o.data
                     for m in [mm.name for mm in o.data.materials] if m}:
        make_material_opaque(bpy.data.materials[mat_name])
        changed.append(f"  material {mat_name:12s} -> object colour straight to surface, opaque")

    # --- hide the modelling aids ---
    for name in NON_POSE_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.hide_render = True
        obj.hide_viewport = True
        hidden.append(name)

    # --- colour pipeline ---
    # Any view transform other than Standard regrades the emissive primaries into
    # something the pose encoder cannot read regardless of the hex values. Dither
    # matters too: it is 1.0 by default, and ±1/255 of noise smears the flat fills
    # the encoder keys on. (It only bites on 8-bit output, so FUK's EXR path dodges
    # it -- but a PNG rendered straight from Blender would carry it.)
    for scene in bpy.data.scenes:
        vs = scene.view_settings
        vs.view_transform = "Standard"
        vs.look = "None"
        vs.exposure = 0.0
        vs.gamma = 1.0
        scene.display_settings.display_device = "sRGB"
        scene.render.film_transparent = False
        scene.render.dither_intensity = 0.0

    for world in bpy.data.worlds:
        world.use_nodes = True
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                node.inputs["Strength"].default_value = 0.0

    print("\n".join(changed))
    if hidden:
        print(f"  hidden (not part of the pose): {', '.join(hidden)}")
    if missing:
        print(f"  WARNING: {'; '.join(missing)}")
    print(f"\nrecoloured {len(joints)} joints + {len(seen)} bones, hid {len(hidden)}, "
          f"view transform -> Standard, dither -> 0")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    inplace = "--inplace" in argv

    src = Path(bpy.data.filepath)
    if "--out" in argv:
        dst = Path(argv[argv.index("--out") + 1])
        if not dst.is_absolute():
            dst = src.parent / dst
    elif inplace:
        shutil.copy2(src, src.with_suffix(".blend.bak"))
        print(f"backed up -> {src.with_suffix('.blend.bak')}")
        dst = src
    else:
        dst = src.with_name(src.stem + "_OP21.blend")

    convert()
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
