#!/usr/bin/env python3
"""
Convert the io7m BODY_25 OpenPose rig to the COCO-18 colour convention that
controlnet_aux / DWPose render, which is what FUK's openpose preprocessor
(fuk/core/preprocessors/openpose.py) emits and therefore what Wan was
conditioned on.

Usage:
    blender Body25.blend --background --factory-startup \
        --python openpose_coco18.py -- --out Body25_COCO18.blend

    # overwrite the source file instead (makes a .bak first)
    blender Body25.blend --background --python openpose_coco18.py -- --inplace

The rig stores every colour in `object.color` (linear), which both materials
feed straight into the surface as emission.  BoneColor additionally mixes 0.6
with a Transparent BSDF, matching the renderer's 0.6 limb attenuation.
"""

import sys
import shutil
from pathlib import Path

import bpy


# --- colour management -------------------------------------------------------

def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(h):
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(srgb_to_linear(v) for v in (r, g, b))


def hex_to_linear_scaled(h, factor):
    """
    Colour that renders to `factor * hex` in sRGB.

    The CV renderer scales the 8-bit colour (0.6 * 255 = 153) and fills the
    limb opaquely, so the attenuation is baked into the colour here rather
    than done with alpha.
    """
    out = []
    for i in (0, 2, 4):
        s = int(h[i:i + 2], 16) / 255.0
        out.append(srgb_to_linear(s * factor))
    return tuple(out)


# --- controlnet_aux / DWPose COCO-18 tables ----------------------------------
# venv/lib/python3.10/site-packages/controlnet_aux/open_pose/util.py:draw_bodypose

COLORS = [
    "ff0000", "ff5500", "ffaa00", "ffff00", "aaff00", "55ff00",
    "00ff00", "00ff55", "00ffaa", "00ffff", "00aaff", "0055ff",
    "0000ff", "5500ff", "aa00ff", "ff00ff", "ff00aa", "ff0055",
]

# joint index -> colour: `zip(keypoints, colors)`, so COLORS[i] for keypoint i.
JOINT_COLORS = {
    "0_Nose": COLORS[0],
    "1_Neck": COLORS[1],
    "2_RShoulder": COLORS[2],
    "3_RElbow": COLORS[3],
    "4_RWrist": COLORS[4],
    "5_LShoulder": COLORS[5],
    "6_LElbow": COLORS[6],
    "7_LWrist": COLORS[7],
    "9_RHip": COLORS[8],
    "10_RKnee": COLORS[9],
    "11_RAnkle": COLORS[10],
    "12_LHip": COLORS[11],
    "13_LKnee": COLORS[12],
    "14_LAnkle": COLORS[13],
    "15_REye": COLORS[14],
    "16_LEye": COLORS[15],
    "17_REar": COLORS[16],
    "18_LEar": COLORS[17],
}

# limb colour: `zip(limbSeq, colors)`, so the Nth limb in limbSeq order gets
# COLORS[N] -- NOT the colour of the joint it terminates at.  Rig bone meshes
# are named after their distal joint.
LIMB_COLORS = {
    "RShoulder": COLORS[0],   # limbSeq[0]  Neck->RShoulder
    "LShoulder": COLORS[1],   # limbSeq[1]  Neck->LShoulder
    "RElbow": COLORS[2],   # limbSeq[2]  RShoulder->RElbow
    "RWrist": COLORS[3],   # limbSeq[3]  RElbow->RWrist
    "LElbow": COLORS[4],   # limbSeq[4]  LShoulder->LElbow
    "LWrist": COLORS[5],   # limbSeq[5]  LElbow->LWrist
    "RHip": COLORS[6],   # limbSeq[6]  Neck->RHip
    "RKnee": COLORS[7],   # limbSeq[7]  RHip->RKnee
    "RAnkle": COLORS[8],   # limbSeq[8]  RKnee->RAnkle
    "LHip": COLORS[9],   # limbSeq[9]  Neck->LHip
    "LKnee": COLORS[10],  # limbSeq[10] LHip->LKnee
    "LAnkle": COLORS[11],  # limbSeq[11] LKnee->LAnkle
    "Nose": COLORS[12],  # limbSeq[12] Neck->Nose
    "REye": COLORS[13],  # limbSeq[13] Nose->REye
    "REar": COLORS[14],  # limbSeq[14] REye->REar
    "LEye": COLORS[15],  # limbSeq[15] Nose->LEye
    "LEar": COLORS[16],  # limbSeq[16] LEye->LEar
}

# BODY_25-only geometry that COCO-18 has no slot for.  Left visible it paints
# blobs the pose encoder never saw during training.
BODY25_ONLY = [
    "8_MidHip",
    "19_LBigToe", "20_LSmallToe", "21_LHeel",
    "22_RBigToe", "23_RSmallToe", "24_RHeel",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]

# BODY_25 runs the torso as one vertical stick plus two short pelvic stubs.
# COCO-18 has neither: draw_bodypose emits Neck->RHip and Neck->LHip as two
# full-length diagonals.  Replaced by build_coco_torso().
BODY25_TORSO = ["MidHip", "RHip", "LHip"]

COCO_TORSO = [
    # name,          bone anchoring the far end,  limbSeq index
    ("NeckToRHip", "RHip", 6),
    ("NeckToLHip", "LHip", 9),
]

BONE_ALPHA = 0.6
LIMB_RADIUS = 0.01205  # matches the existing stick tubes (0.0241 diameter)


def set_color(obj, hexcode, factor=None):
    lin = hex_to_linear(hexcode) if factor is None else hex_to_linear_scaled(hexcode, factor)
    obj.color = (*lin, 1.0)


def make_bone_material_opaque(mat):
    """
    Stock BoneColor mixes 0.6 with a Transparent BSDF.  With backface culling
    off, both walls of each limb tube composite, so the rendered value lands at
    ~0.64x and varies with silhouette thickness.  draw_bodypose fills limbs
    with an opaque 0.6-scaled colour, so drive the surface directly and let the
    attenuation live in object.color instead.
    """
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    info = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeObjectInfo")

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
    mat.use_backface_culling = True


def build_coco_torso(armature, material):
    """
    Add the two Neck->Hip diagonals COCO-18 expects.

    Each stick is a unit cylinder along +Y with its base at the origin, pinned
    to the Neck bone head and stretched to a hip bone tail.  Constraints rather
    than skinning, so the sticks track any pose without touching the armature.
    """
    import bmesh
    import math
    from mathutils import Matrix

    created = []
    for name, hip_bone, limb_index in COCO_TORSO:
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)

        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=False, segments=16,
            radius1=LIMB_RADIUS, radius2=LIMB_RADIUS, depth=1.0,
        )
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, 0.0, 0.5))
        bm.to_mesh(mesh)
        bm.free()
        mesh.transform(Matrix.Rotation(math.radians(-90.0), 4, "X"))
        mesh.materials.append(material)

        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = armature

        loc = obj.constraints.new("COPY_LOCATION")
        loc.target = armature
        loc.subtarget = "Neck"
        loc.head_tail = 0.0

        stretch = obj.constraints.new("STRETCH_TO")
        stretch.target = armature
        stretch.subtarget = hip_bone
        stretch.head_tail = 1.0
        stretch.rest_length = 1.0
        stretch.volume = "NO_VOLUME"

        set_color(obj, COLORS[limb_index], factor=BONE_ALPHA)
        created.append(f"  limb  {name:14s} -> #{COLORS[limb_index]} @ {BONE_ALPHA} (new)")

    return created


def convert(keep_feet=False, keep_body25_torso=False):
    changed, missing = [], []

    for name, hexcode in JOINT_COLORS.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        set_color(obj, hexcode)
        changed.append(f"  joint {name:14s} -> #{hexcode}")

    bone_mat = bpy.data.materials.get("BoneColor")
    if bone_mat is not None and bone_mat.use_nodes:
        make_bone_material_opaque(bone_mat)

    for name, hexcode in LIMB_COLORS.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        set_color(obj, hexcode, factor=BONE_ALPHA)
        changed.append(f"  limb  {name:14s} -> #{hexcode} @ {BONE_ALPHA}")

    if keep_body25_torso:
        # No COCO-18 equivalent, so the best available approximation is to read
        # the vertical stick as the right-hip limb.
        torso = bpy.data.objects.get("MidHip")
        if torso is not None:
            set_color(torso, COLORS[6], factor=BONE_ALPHA)
            changed.append(f"  limb  {'MidHip':14s} -> #{COLORS[6]} @ {BONE_ALPHA} (approx)")
    else:
        armature = bpy.data.objects.get("Body25")
        if armature is None or bone_mat is None:
            missing.append("Body25 armature / BoneColor (torso rebuild skipped)")
        else:
            changed.extend(build_coco_torso(armature, bone_mat))

    to_hide = list(BODY25_ONLY) if not keep_feet else []
    if not keep_body25_torso:
        to_hide += BODY25_TORSO

    hidden = []
    for name in to_hide:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.hide_render = True
        obj.hide_viewport = True
        hidden.append(name)

    # The shipped file is set to AgX, which desaturates the emissive primaries
    # into something the pose encoder cannot read regardless of hex values.
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
        print(f"  hidden (BODY_25 only): {', '.join(hidden)}")
    if missing:
        print(f"  WARNING missing objects: {', '.join(missing)}")
    print(f"\nrecoloured {len(changed)} objects, hid {len(hidden)}, view transform -> Standard")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    keep_feet = "--keep-feet" in argv
    keep_body25_torso = "--keep-body25-torso" in argv
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
        dst = src.with_name(src.stem + "_COCO18.blend")

    convert(keep_feet=keep_feet, keep_body25_torso=keep_body25_torso)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
