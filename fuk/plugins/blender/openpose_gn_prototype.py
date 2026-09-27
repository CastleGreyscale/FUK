#!/usr/bin/env python3
"""
PROTOTYPE: build an OpenPose control-map generator on top of an Auto-Rig Pro
character, driven entirely by Geometry Nodes.

    blender OPii_Rig.blend --background --factory-startup \
        --python openpose_gn_prototype.py -- --out OPii_OP.blend

The point of the prototype is the ARCHITECTURE, not coverage. Compare it with
the approach used on Hand.blend, where every sphere and tube was a real object,
bone-parented, with its geometry baked in. That works for a static diagram but
cannot animate: changing a bone invalidated every primitive, which is why
hand_rig_anatomy.py needed rebuild_primitives() to re-derive them all by hand.
Here nothing is baked — the skeleton is recomputed from the bones on every frame
evaluation, which is what a control VIDEO needs.


How bone motion reaches Geometry Nodes
--------------------------------------

It can't, directly: there is no node that reads an armature's bone transforms.
The way in is a plain mesh bound to the rig by an ordinary Armature modifier —
one vertex per keypoint, each weighted 1.0 to the bone it should follow. The
modifier deforms those vertices, and the Geometry Nodes modifier downstream
reads the already-deformed positions. That is almost certainly what the OPii
kit's L_Arm / R_Arm curve objects were for; they ship empty, which is why its
generator currently emits nothing.

Topology carries the rest:

  * one lone vertex per keypoint   -> instanced sphere  (the joint dot)
  * one 2-vertex edge per limb     -> curve -> tube     (the limb stick)

Limb vertices are deliberately NOT shared with joint vertices, nor between
limbs. Sharing them would let Mesh to Curve weld separate limbs into one spline
at every joint, and a per-limb colour could not survive that.


Colour
------

The OPii kit spends 34 Set Material nodes and 20 materials to paint this. One
material is enough: a colour attribute per vertex feeding an Emission shader.
Exact values, no palette duplicated into datablocks, and adding a keypoint costs
nothing. Radius travels the same way.

Conventions come from controlnet_aux (see openpose_coco18.py / openpose_hand21.py):

  body joints   COCO-18 palette, full brightness
  body limbs    COCO-18 palette in limbSeq order, x0.6
  hand joints   all pure blue
  hand limbs    HSV rainbow over 20 edges, full brightness

and the widths from the same source: body sticks are ellipse2Poly with a
semi-axis of 4 (so 8px across), hand sticks are cv2.line thickness=2, and every
joint of both is cv2.circle radius 4.
"""

import sys
import math
import shutil
from pathlib import Path

import bpy
from mathutils import Vector


RIG_OBJECT = "OPii_rig"
SOURCE_MESH = "OP_source"
GN_GROUP = "OP_Generator"
MATERIAL = "OP_emission"

COLORS = [
    "ff0000", "ff5500", "ffaa00", "ffff00", "aaff00", "55ff00",
    "00ff00", "00ff55", "00ffaa", "00ffff", "00aaff", "0055ff",
    "0000ff", "5500ff", "aa00ff", "ff00ff", "ff00aa", "ff0055",
]
BODY_LIMB_ALPHA = 0.6

# --- COCO-18 keypoint -> (bone, which end) ---------------------------------
# ARP deform bones. A joint sits at the HEAD of the bone that starts there.
# This character carries no nose/eye/ear bones, so keypoints 0 and 14-17 are
# unmapped and simply skipped — reported rather than faked.
BODY_KP = {
    1:  ("neck.x",    "head"),   # Neck
    2:  ("arm.r",     "head"),   # RShoulder
    3:  ("forearm.r", "head"),   # RElbow
    4:  ("hand.r",    "head"),   # RWrist
    5:  ("arm.l",     "head"),   # LShoulder
    6:  ("forearm.l", "head"),   # LElbow
    7:  ("hand.l",    "head"),   # LWrist
    8:  ("thigh.r",   "head"),   # RHip
    9:  ("leg.r",     "head"),   # RKnee
    10: ("foot.r",    "head"),   # RAnkle
    11: ("thigh.l",   "head"),   # LHip
    12: ("leg.l",     "head"),   # LKnee
    13: ("foot.l",    "head"),   # LAnkle
}
# --- face: derived, because this rig has no facial bones ------------------
# c_pupil / c_iris exist by name but sit at z = -55, out with the picker
# widgets — they are UI, not anatomy. So nose/eyes/ears are built from the Neck
# keypoint plus proportions of the figure's height, and all five ride the head's
# deform bone.
#
# Anchoring on stature rather than on head.x's length is deliberate: it is what
# OpenPose itself effectively encodes, and it self-corrects across characters
# whose head bone spans a different part of the skull. Offsets are fractions of
# figure height, from the Neck keypoint, in (up, forward, lateral).
HEAD_BONE = "head.x"
FACE_OFFSETS = {
    0:  (0.105, 0.045, 0.000),   # Nose   — neck->nose ~0.105 of stature
    14: (0.120, 0.020, -0.020),  # REye   (lateral is signed: + is the figure's left)
    15: (0.120, 0.020, +0.020),  # LEye
    16: (0.121, -0.005, -0.045), # REar
    17: (0.121, -0.005, +0.045), # LEar
}
# The ears' FORWARD offset is the sensitive one. draw_bodypose really does draw
# Nose->Eye->Ear (limbSeq ends [2,1],[1,15],[15,17],[1,16],[16,18]), but in a
# real detection those sticks are short and disappear into the dot cluster.
# Setting the ears well behind the neck axis stretches Eye->Ear into a pair of
# visible spars off the sides of the head — an "antler" silhouette that appears
# in no real OpenPose map. At -0.030 the ear sat 8cm behind the neck on a 1.59m
# figure and the limb ran 1.7x its true length; an ear canal is ~1-2cm back.

BODY_KP_NAME = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle",
    14: "REye", 15: "LEye", 16: "REar", 17: "LEar",
}
# limbSeq order — the Nth limb takes COLORS[N], never the colour of the joint
# it ends at.
BODY_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# --- hand: 21 keypoints -> (bone, end) -------------------------------------
# ARP gives three control bones per digit. Keypoint 0 is the wrist; 1-4 thumb,
# 5-8 index, 9-12 middle, 13-16 ring, 17-20 little, tips on the last tail.
def hand_kp(side):
    kp = {0: (f"hand.{side}", "head")}
    for base, finger in ((1, "thumb"), (5, "index"), (9, "middle"),
                         (13, "ring"), (17, "pinky")):
        for i in range(3):
            kp[base + i] = (f"c_{finger}{i + 1}.{side}", "head")
        kp[base + 3] = (f"c_{finger}3.{side}", "tail")
    return kp


HAND_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
HAND_JOINT_RGB = (0.0, 0.0, 1.0)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_rgb(h, factor=1.0):
    return tuple(srgb_to_linear(int(h[i:i + 2], 16) / 255.0 * factor)
                 for i in (0, 2, 4))


def hand_limb_rgb(ie, n=20):
    """draw_handpose: hsv_to_rgb([ie/20, 1, 1]). Exact for S=V=1."""
    h = ie / float(n)
    sector = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    q, t = 1.0 - f, f
    srgb = [(1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q)][sector]
    return tuple(srgb_to_linear(c) for c in srgb)


# ---------------------------------------------------------------------------

def bone_point(arm_obj, bone_name, end):
    b = arm_obj.data.bones.get(bone_name)
    if b is None:
        return None
    return (b.head_local if end == "head" else b.tail_local).copy()


def resolve_deform_bone(arm_obj, position, tol=1e-4):
    """The DEFORM bone to weight a keypoint to, given where the keypoint sits.

    The semantic tables above name the anatomically meaningful bone (arm.l,
    thigh.l, c_index1.l). On an ARP rig many of those carry use_deform = False —
    they are control//organisational bones, and the actual skinning is done by
    a coincident partner (shoulder.l, thigh_twist.l, c_index1_base.l ...).

    This matters because an Armature modifier SILENTLY IGNORES a vertex group
    whose bone does not deform. Binding to arm.l or thigh.l therefore produced a
    skeleton that half-followed the rig: fingers and the IK hand moved, but
    shoulders, elbows, hips and knees stayed pinned at rest with no error
    anywhere. Resolving by position instead of by a hard-coded partner table
    keeps this working across ARP versions and rig configurations.
    """
    best = None
    for b in arm_obj.data.bones:
        if not b.use_deform:
            continue
        for p in (b.head_local, b.tail_local):
            d = (p - position).length
            if best is None or d < best[0]:
                best = (d, b.name)
    if best is None or best[0] > tol:
        return None, (best[0] if best else float("inf"))
    return best[1], best[0]


def figure_axes(arm_obj):
    """(up, forward, left) for the character, in armature space.

    Forward is read from the toes rather than assumed: this rig faces -Y, and
    guessing wrong mirrors the whole face. Left is forward x up, which on this
    rig puts arm.l at +X — matching the bone positions, so the check is real.
    """
    up = Vector((0.0, 0.0, 1.0))
    toe = arm_obj.data.bones.get("c_toes_end.l") or arm_obj.data.bones.get("foot.l")
    if toe is not None:
        fwd = (toe.tail_local - toe.head_local)
        fwd.z = 0.0
        forward = fwd.normalized() if fwd.length > 1e-6 else Vector((0.0, -1.0, 0.0))
    else:
        forward = Vector((0.0, -1.0, 0.0))
    left = -forward.cross(up)
    return up, forward, left


def face_points(arm_obj, height):
    """{keypoint: position} for nose/eyes/ears, or {} if the neck is unmapped."""
    neck_bone, neck_end = BODY_KP.get(1, (None, None))
    neck = bone_point(arm_obj, neck_bone, neck_end) if neck_bone else None
    if neck is None:
        return {}
    up, forward, left = figure_axes(arm_obj)
    return {
        kp: neck + up * (u * height) + forward * (f * height) + left * (s * height)
        for kp, (u, f, s) in FACE_OFFSETS.items()
    }


def figure_height(arm_obj):
    """Height of the FIGURE, measured only across mapped keypoints.

    Not the armature bounding box: an ARP rig carries its picker and proxy bones
    parked far from the character (on this one, around z = -56 against a figure
    barely 2 units tall), so a naive bbox overstates the height by ~30x and every
    stick comes out fat enough to fill the frame.
    """
    zs = []
    for bone, end in BODY_KP.values():
        p = bone_point(arm_obj, bone, end)
        if p is not None:
            zs.append(p.z)
    if len(zs) < 2:
        raise RuntimeError("too few mapped keypoints to measure the figure")
    # Neck-to-ankle is about 0.82 of standing height.
    return (max(zs) - min(zs)) / 0.82


def stick_radii(height, output_height_px, frame_fraction):
    """(joint, body stick, hand stick) radii in world units.

    OpenPose draws at FIXED PIXEL sizes — cv2.circle radius 4 for every joint of
    both body and hand, an ellipse2Poly semi-axis of 4 for body limbs (8px
    across) and cv2.line thickness 2 for hand limbs. Nothing scales with the
    subject, so the right world size depends on how many world units a pixel
    covers once the figure is framed.

    `output_height_px` is the height the MODEL will read the map at — i.e. the
    generation size, NOT necessarily Blender's render size. FUK resizes the
    control map to the generation dimensions, so sticks sized against a 2048px
    Blender render arrive at 1088 only 4.2px wide, half the convention. Sizing
    against the generation height instead puts them at 8.0px as intended.
    """
    world_per_px = height / max(1.0, frame_fraction * output_height_px)
    return 4.0 * world_per_px, 4.0 * world_per_px, 1.0 * world_per_px


def build_source_mesh(arm_obj, height, output_height_px, frame_fraction):
    """One vertex per keypoint, one 2-vertex edge per limb, each vertex weighted
    to the bone it follows. Returns (object, report)."""
    joint_r, body_r, hand_r = stick_radii(height, output_height_px, frame_fraction)

    verts, edges, groups = [], [], {}
    colors, radii, kinds = [], [], []
    missing, rebound = [], {}
    face = face_points(arm_obj, height)

    def add_vertex(bone, end, rgb, radius, kind, at=None, bind_to=None):
        """`at`/`bind_to` are for derived keypoints (the face), which sit at no
        bone endpoint and so cannot be resolved by position — they ride the head
        bone rigidly instead."""
        p = Vector(at) if at is not None else bone_point(arm_obj, bone, end)
        if p is None:
            return None
        if bind_to is not None:
            bind = bind_to
        else:
            bind, dist = resolve_deform_bone(arm_obj, p)
            if bind is None:
                missing.append(f"{bone}/{end}: no deform bone within tolerance "
                               f"(nearest {dist:.5f}) — this keypoint will not follow")
                bind = bone
            elif bind != bone:
                rebound.setdefault(bone, bind)
        i = len(verts)
        verts.append(p)
        groups.setdefault(bind, []).append(i)
        colors.append(rgb)
        radii.append(radius)
        kinds.append(kind)
        return i

    def add_set(kp_map, limbs, joint_rgb_fn, limb_rgb_fn, limb_radius, limb_kind, label):
        """kp_map values are either (bone, end) or a dict for a derived point."""
        placed = {}
        for idx, spec in sorted(kp_map.items()):
            spec = spec if isinstance(spec, dict) else {"bone": spec[0], "end": spec[1]}
            i = add_vertex(spec.get("bone"), spec.get("end"), joint_rgb_fn(idx),
                           joint_r, 0.0, at=spec.get("at"), bind_to=spec.get("bind_to"))
            if i is None:
                missing.append(f"{label} keypoint {idx} -> {spec.get('bone')!r} not found")
            else:
                placed[idx] = spec
        for n, (a, b) in enumerate(limbs):
            if a not in placed or b not in placed:
                missing.append(f"{label} limb {n} ({a}->{b}) skipped, endpoint unmapped")
                continue
            rgb = limb_rgb_fn(n)
            ends = []
            for kp in (a, b):
                s = placed[kp]
                ends.append(add_vertex(s.get("bone"), s.get("end"), rgb, limb_radius,
                                       limb_kind, at=s.get("at"),
                                       bind_to=s.get("bind_to")))
            edges.append(tuple(ends))

    # Body, with the derived face folded in so the head limbs resolve too.
    body_map = dict(BODY_KP)
    if face and arm_obj.data.bones.get(HEAD_BONE) is not None:
        for kp, pos in face.items():
            body_map[kp] = {"at": pos, "bind_to": HEAD_BONE,
                            "bone": HEAD_BONE, "end": "head"}
    else:
        missing.append(f"no {HEAD_BONE!r} bone — face keypoints 0 and 14-17 skipped")

    add_set(body_map, BODY_LIMBS,
            lambda i: hex_rgb(COLORS[i]),
            lambda n: hex_rgb(COLORS[n], BODY_LIMB_ALPHA),
            body_r, 1.0, "body")
    for side in ("l", "r"):
        add_set(hand_kp(side), HAND_LIMBS,
                lambda i: HAND_JOINT_RGB,
                hand_limb_rgb, hand_r, 2.0, f"hand.{side}")

    old = bpy.data.objects.get(SOURCE_MESH)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new(SOURCE_MESH)
    mesh.from_pydata([tuple(v) for v in verts], edges, [])
    mesh.update()

    obj = bpy.data.objects.new(SOURCE_MESH, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = arm_obj.matrix_world.copy()

    for bone, idxs in groups.items():
        vg = obj.vertex_groups.new(name=bone)
        vg.add(idxs, 1.0, "REPLACE")

    for name, data, kind in (("op_color", colors, "FLOAT_COLOR"),
                             ("radius", radii, "FLOAT"),
                             ("op_kind", kinds, "FLOAT")):
        attr = mesh.attributes.new(name=name, type=kind, domain="POINT")
        if kind == "FLOAT_COLOR":
            for i, c in enumerate(data):
                attr.data[i].color = (*c, 1.0)
        else:
            for i, v in enumerate(data):
                attr.data[i].value = v

    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj

    return obj, {"verts": len(verts), "edges": len(edges), "missing": missing,
                 "rebound": rebound}


def build_material():
    mat = bpy.data.materials.get(MATERIAL) or bpy.data.materials.new(MATERIAL)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "op_color"
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    for value in ("OPAQUE", "NONE"):
        try:
            mat.blend_method = value
            break
        except TypeError:
            continue
    return mat


def build_node_group(material, joint_radius, body_radius, hand_radius):
    old = bpy.data.node_groups.get(GN_GROUP)
    if old:
        bpy.data.node_groups.remove(old)
    ng = bpy.data.node_groups.new(GN_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    n = ng.nodes.new
    gin, gout = n("NodeGroupInput"), n("NodeGroupOutput")

    kind = n("GeometryNodeInputNamedAttribute"); kind.data_type = "FLOAT"
    kind.inputs["Name"].default_value = "op_kind"

    def split(geo, lo, hi):
        """Geometry whose op_kind falls in (lo, hi)."""
        gt = n("FunctionNodeCompare"); gt.data_type, gt.operation = "FLOAT", "GREATER_THAN"
        gt.inputs[1].default_value = lo
        ng.links.new(kind.outputs["Attribute"], gt.inputs[0])
        lt = n("FunctionNodeCompare"); lt.data_type, lt.operation = "FLOAT", "LESS_THAN"
        lt.inputs[1].default_value = hi
        ng.links.new(kind.outputs["Attribute"], lt.inputs[0])
        both = n("FunctionNodeBooleanMath"); both.operation = "AND"
        ng.links.new(gt.outputs["Result"], both.inputs[0])
        ng.links.new(lt.outputs["Result"], both.inputs[1])
        sep = n("GeometryNodeSeparateGeometry"); sep.domain = "POINT"
        ng.links.new(geo, sep.inputs["Geometry"])
        ng.links.new(both.outputs["Boolean"], sep.inputs["Selection"])
        return sep.outputs["Selection"]

    join = n("GeometryNodeJoinGeometry")

    # --- joints: one sphere per lone vertex ---
    # Built at final size rather than instanced-and-scaled. Every OpenPose joint
    # is cv2.circle(..., 4, ...) in BOTH draw_bodypose and draw_handpose, so the
    # size is uniform and a per-point scale buys nothing. It also did not work:
    # feeding the radius attribute into Instance on Points' Scale left every dot
    # at radius 1.0.
    sphere = n("GeometryNodeMeshUVSphere")
    sphere.inputs["Segments"].default_value = 12
    sphere.inputs["Rings"].default_value = 8
    sphere.inputs["Radius"].default_value = joint_radius
    iop = n("GeometryNodeInstanceOnPoints")
    ng.links.new(split(gin.outputs[0], -0.5, 0.5), iop.inputs["Points"])
    ng.links.new(sphere.outputs["Mesh"], iop.inputs["Instance"])
    realize = n("GeometryNodeRealizeInstances")
    ng.links.new(iop.outputs["Instances"], realize.inputs["Geometry"])
    ng.links.new(realize.outputs["Geometry"], join.inputs["Geometry"])

    # --- limbs: one tube per 2-vertex edge, one branch per stick width ---
    # There are exactly two widths (body 4, hand 1), so each branch gets a
    # fixed-radius profile circle. Driving the radius as a field was tried three
    # ways -- Set Curve Radius, the curve's built-in "radius" attribute, and
    # Curve to Mesh's Scale input -- and all three silently left the profile at
    # radius 1.0, i.e. 80x oversized.
    for lo, hi, radius in ((0.5, 1.5, body_radius), (1.5, 2.5, hand_radius)):
        to_curve = n("GeometryNodeMeshToCurve")
        ng.links.new(split(gin.outputs[0], lo, hi), to_curve.inputs[0])
        circle = n("GeometryNodeCurvePrimitiveCircle")
        circle.inputs["Resolution"].default_value = 10
        circle.inputs["Radius"].default_value = radius
        to_mesh = n("GeometryNodeCurveToMesh")
        to_mesh.inputs["Fill Caps"].default_value = True
        ng.links.new(to_curve.outputs["Curve"], to_mesh.inputs["Curve"])
        ng.links.new(circle.outputs["Curve"], to_mesh.inputs["Profile Curve"])
        ng.links.new(to_mesh.outputs["Mesh"], join.inputs["Geometry"])

    setmat = n("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = material
    ng.links.new(join.outputs["Geometry"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], gout.inputs[0])

    for i, node in enumerate(ng.nodes):
        node.location = (i % 6 * 220, -(i // 6) * 220)
    return ng


# ---------------------------------------------------------------------------
# Driving the radii from the camera
# ---------------------------------------------------------------------------
#
# stick_radii() converts OpenPose's fixed PIXEL sizes into world units, but it
# does so once, at bake time, against a `frame_fraction` the caller guesses.
# The result is frozen into the node group, so the sticks keep a constant WORLD
# size while their screen size goes as 1/depth. Sized for a full shot at
# frame_fraction 0.8 they measure ~9px across, near the 8px convention; dolly
# in to a chest-up framing and the same geometry renders ~28px, and at head and
# shoulders ~42px. At that width the map stops reading as a pose diagram and
# ControlNet paints the sticks into the image as objects — most visibly the
# nose/eye/ear cluster, whose five dots merge into one mass and come back as
# hair.
#
# So the radii have to follow the camera instead. Vertical framing at the
# subject's depth is
#
#     world_frame_h = depth * sensor_v / lens
#
# and one output pixel covers world_frame_h / gen_height_px, which is what the
# OpenPose pixel constants below multiply.
#
# `sensor_v` is the fiddly part. Camera.angle_y looks like the answer and is
# not: under the default AUTO sensor fit it is computed from sensor_height and
# ignores the resolution entirely, so on this 576x896 portrait render it
# reports 24mm where the true vertical sensor is 36mm — the 50% error goes
# straight into every stick. AUTO puts sensor_width on the LARGER image
# dimension, which is what the min() below reproduces. A camera explicitly set
# to VERTICAL fit would need sensor_height instead, and is rejected rather than
# silently mis-sized.
#
# Depth is a LOC_DIFF driver variable — a straight distance, not a projection
# onto the view axis. They differ by the cosine of the subject's angle off
# centre, which at this rig's 80mm lens is under 3% at the frame edge and zero
# for a centred figure.
#
# The output height is DERIVED IN THE DRIVER rather than stored. An earlier
# version had the addon precompute it into scene["op_gen_height"] before each
# render, which was correct only for renders that went through that code path:
# an F12 from the UI, or a full-res render after a 25% preview, kept the stale
# number and sized every stick against it — a 280 left over from a quarter-res
# preview puts 32px sticks in a 1120-tall render, four times the convention and
# straight back into the failure this whole mechanism exists to prevent.
# resolution_x/y and resolution_percentage are all readable from the driver, so
# nothing needs to be cached and no render path can be missed.
#
# The one thing the driver cannot derive is the in-context control budget, which
# belongs to the MODEL rather than to the render — hence the one scene property
# that remains. It is stable across renders (it only changes when the model
# does), so it has none of the staleness the height had.

JOINT_PX = 4.0   # draw_bodypose/draw_handpose: cv2.circle(..., 4, ...)
BODY_PX = 4.0    # draw_bodypose: ellipse2Poly semi-axis 4, so 8px across
HAND_PX = 1.0    # draw_handpose: cv2.line(..., thickness=2), so 2px across

CAMERA_OBJECT = "SHOT_CAM"
ANCHOR_KP = 1    # Neck — the head cluster is the most scale-sensitive part

# Scene custom properties earlier versions of this module wrote. Nothing reads
# them now — the driver derives the output height from the render settings and
# inlines the budget — but install_radius_drivers() clears them so a stale one
# cannot be mistaken for a live control.
GEN_HEIGHT_PROP_LEGACY = "op_gen_height"
MAX_PIXELS_PROP_LEGACY = "op_max_pixels"

RADIUS_SOCKETS = (("Joint Radius", JOINT_PX),
                  ("Body Radius", BODY_PX),
                  ("Hand Radius", HAND_PX))


def radius_sockets(ng):
    """[(label, pixels, driver data path)] for the three radius inputs.

    The drivers go on the primitive nodes INSIDE the group rather than on
    modifier inputs. Exposing them as group inputs reads better in the UI, but
    Blender 5.x keeps a geometry nodes modifier's socket values in an
    ID-property group under modifier.properties.inputs, and driver_add() on
    that path fails as "not animatable" — a node socket's default_value is
    animatable and has been for every version this rig has seen.

    The two limb circles are told apart by their current radius rather than by
    node name, since 'Curve Circle' vs 'Curve Circle.001' depends on the order
    build_node_group happened to create them in.
    """
    spheres = [n for n in ng.nodes if n.bl_idname == "GeometryNodeMeshUVSphere"]
    circles = [n for n in ng.nodes
               if n.bl_idname == "GeometryNodeCurvePrimitiveCircle"]
    if len(spheres) != 1 or len(circles) != 2:
        raise RuntimeError(f"{ng.name!r}: expected 1 sphere and 2 circles, "
                           f"found {len(spheres)} and {len(circles)}")
    circles.sort(key=lambda c: c.inputs["Radius"].default_value, reverse=True)

    out = []
    for node, (label, px) in zip((spheres[0], circles[0], circles[1]),
                                 RADIUS_SOCKETS):
        socket = node.inputs["Radius"]
        if socket.is_linked:
            raise RuntimeError(f"{ng.name!r}: {node.name!r} Radius is linked; "
                               "a link overrides default_value, so the driver "
                               "would have no effect")
        idx = list(node.inputs).index(socket)
        out.append((label, px,
                    f'nodes["{node.name}"].inputs[{idx}].default_value'))
    return out


def install_radius_drivers(obj, arm_obj, camera=None, max_pixels=None):
    """Drive the three radii off the camera so sticks hold their PIXEL width.

    Idempotent — re-running replaces the drivers rather than stacking them.
    They re-evaluate per frame, so a camera move across a control VIDEO stays
    correct for free, and they read the render settings live, so no render path
    can be missed.
    """
    scene = bpy.context.scene
    cam = camera or bpy.data.objects.get(CAMERA_OBJECT) or scene.camera
    if cam is None or cam.type != "CAMERA":
        raise RuntimeError(f"no camera to drive from (looked for {CAMERA_OBJECT!r})")
    if cam.data.sensor_fit == "VERTICAL":
        raise RuntimeError(
            f"camera {cam.name!r} uses VERTICAL sensor fit; the driver assumes "
            "AUTO/HORIZONTAL, where sensor_width spans the larger dimension")
    if cam.data.type != "PERSP":
        raise RuntimeError(f"camera {cam.name!r} is {cam.data.type}, not PERSP; "
                           "an orthographic frame does not scale with depth")

    anchor_name, anchor_end = BODY_KP[ANCHOR_KP]
    anchor_pos = bone_point(arm_obj, anchor_name, anchor_end)
    if anchor_pos is None:
        raise RuntimeError(f"anchor bone {anchor_name!r} not on the rig")
    # Same rebinding the source mesh does: the named bone is often a control
    # bone with use_deform off, and a driver on it would not follow the pose.
    anchor_bone, _ = resolve_deform_bone(arm_obj, anchor_pos)
    anchor_bone = anchor_bone or anchor_name

    budget_px = float(max_pixels or 0.0)
    if budget_px < 0:
        budget_px = 0.0
    # Both are leftovers from designs that stored what the driver now derives or
    # inlines. Harmless to leave behind, but confusing to find in a scene, and
    # the second one is the very landmine described below.
    for sc in bpy.data.scenes:
        for stale in (GEN_HEIGHT_PROP_LEGACY, MAX_PIXELS_PROP_LEGACY):
            sc.pop(stale, None)

    mod = obj.modifiers.get("OP_Generator")
    if mod is None or mod.node_group is None:
        raise RuntimeError(f"{obj.name!r} has no OP_Generator nodes modifier")
    ng = mod.node_group
    sockets = radius_sockets(ng)

    if ng.animation_data is None:
        ng.animation_data_create()

    for label, px, data_path in sockets:
        try:
            ng.driver_remove(data_path)
        except (TypeError, RuntimeError):
            pass
        fcurve = ng.driver_add(data_path)
        drv = fcurve.driver
        drv.type = "SCRIPTED"

        def var(name, kind):
            v = drv.variables.new()
            v.name, v.type = name, kind
            return v

        v = var("depth", "LOC_DIFF")
        v.targets[0].id = cam
        v.targets[1].id = arm_obj
        v.targets[1].bone_target = anchor_bone

        for name, ident, dpath in (("sw", cam.data, "sensor_width"),
                                   ("lens", cam.data, "lens"),
                                   ("rx", scene, "render.resolution_x"),
                                   ("ry", scene, "render.resolution_y"),
                                   ("pct", scene, "render.resolution_percentage")):
            v = var(name, "SINGLE_PROP")
            v.targets[0].id_type = "CAMERA" if ident is cam.data else "SCENE"
            v.targets[0].id = ident
            v.targets[0].data_path = dpath

        # Built up in pieces below, then substituted into one expression — a
        # driver holds a single string, but the string is unreadable written
        # flat.
        #
        # Every variable above is a render or camera setting that always exists.
        # An earlier version read the control budget from a scene custom
        # property, and a missing one was not a graceful degradation: the
        # variable evaluates to 0, sqrt(0/A) collapses the divisor onto its
        # floor of 16, and the sticks render at 155px — twenty times the
        # convention, in any scene the addon had not written to. The budget is
        # a literal below instead, so there is nothing left to be absent.
        #
        # rw/rh are what will actually be RENDERED (percentage included), which
        # is also what gets sent as the generation size. `fit` reproduces the
        # AUTO sensor fit: sensor_width spans the larger image dimension.
        rw, rh = "(rx * pct / 100.0)", "(ry * pct / 100.0)"
        fit = "min(1.0, ry / max(1.0, rx))"
        if budget_px:
            # Size for the smaller stage-1 pass the runner denoises with control
            # at when the request is over budget, mirroring _fit_pixel_budget
            # including its deliberate round DOWN to the latent grid. min()
            # covers the under-budget case, where the ratio exceeds 1.
            scale = f"min(1.0, sqrt({budget_px:.1f} / max(1.0, {rw} * {rh})))"
            gen = f"max(16.0, floor({rh} * {scale} / 16.0) * 16.0)"
        else:
            # Size for the rendered canvas: 8px in the map as authored, which is
            # what controlnet_aux itself draws and what FUK's own openpose
            # preprocessor produces.
            gen = f"max(16.0, {rh})"
        # max() guards stop a zeroed resolution from erroring the driver, which
        # in Blender leaves the socket at its last value with no visible failure.
        drv.expression = (f"{px} * depth * sw * {fit} "
                          f"/ (max(1.0, lens) * {gen})")

    return {"camera": cam.name, "anchor_bone": anchor_bone,
            "mode": (f"stage-1 (budget {budget_px/1e6:.2f}MP)" if budget_px
                     else "render canvas")}


# OpenPose stick/dot sizes are absolute pixels, so they depend on how much of
# the frame the figure fills. Override with --frame-fraction when framing tighter
# or wider than a typical full-body shot.
#
# Only the starting value now: install_radius_drivers() takes the radii over
# from the camera, and this is what the sockets read if the drivers are removed.
DEFAULT_FRAME_FRACTION = 0.8

# Height the MODEL reads the control map at — the generation size, not Blender's
# render size. FUK resizes the map before the model sees it, so sizing sticks
# against a large Blender render leaves them too thin by exactly that ratio.
DEFAULT_OUTPUT_HEIGHT = 1024

RENDER_COLLECTION = "OP_render"
VIEW_LAYER = "OpenPose"


def setup_scene(src_obj):
    """Isolate the generated skeleton on its own view layer.

    Setting hide_render on everything else is not enough: this rig DRIVES
    hide_render (it is how the prop switcher hides the unused weapons), so any
    value written here is overwritten on the next evaluation — the machine gun
    and the character mesh both came back and painted themselves into the
    control map. Excluding collections on a dedicated view layer cannot be
    driven out from under us, and it is the shape FUK's bridge already expects,
    since render_passes() takes an openpose_view_layer by name.
    """
    scene = bpy.context.scene

    col = bpy.data.collections.get(RENDER_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(RENDER_COLLECTION)
        scene.collection.children.link(col)
    for c in list(src_obj.users_collection):
        c.objects.unlink(src_obj)
    col.objects.link(src_obj)

    # The armature has to stay IN this view layer. Excluding its collection
    # drops it from that layer's depsgraph, and the Armature modifier then has
    # no evaluated target — the skeleton silently freezes at rest. An armature
    # renders nothing, so keeping it costs nothing.
    rig = bpy.data.objects.get(RIG_OBJECT)
    if rig is not None and col not in list(rig.users_collection):
        col.objects.link(rig)

    vl = scene.view_layers.get(VIEW_LAYER) or scene.view_layers.new(VIEW_LAYER)
    for lc in vl.layer_collection.children:
        lc.exclude = lc.name != RENDER_COLLECTION
    # Objects sitting directly in the scene's master collection have no layer
    # collection to exclude, so those still need hiding individually.
    for obj in scene.collection.objects:
        if obj is not src_obj:
            obj.hide_render = True

    for sc in bpy.data.scenes:
        vs = sc.view_settings
        vs.view_transform, vs.look = "Standard", "None"
        vs.exposure, vs.gamma = 0.0, 1.0
        sc.display_settings.display_device = "sRGB"
        sc.render.film_transparent = False
        sc.render.dither_intensity = 0.0
        sc.render.use_compositing = False
        sc.render.use_sequencer = False
    for world in bpy.data.worlds:
        world.use_nodes = True
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (0, 0, 0, 1)
                node.inputs["Strength"].default_value = 0.0
    return vl


def convert(frame_fraction=DEFAULT_FRAME_FRACTION, output_height_px=None,
            max_pixels=None):
    arm_obj = bpy.data.objects.get(RIG_OBJECT)
    if arm_obj is None:
        raise RuntimeError(f"no armature object named {RIG_OBJECT!r}")

    height = figure_height(arm_obj)
    if output_height_px is None:
        output_height_px = DEFAULT_OUTPUT_HEIGHT

    joint_r, body_r, hand_r = stick_radii(height, output_height_px, frame_fraction)

    material = build_material()
    obj, report = build_source_mesh(arm_obj, height, output_height_px, frame_fraction)
    ng = build_node_group(material, joint_r, body_r, hand_r)
    gn = obj.modifiers.new("OP_Generator", "NODES")
    gn.node_group = ng

    vl = setup_scene(obj)
    print(f"isolated on view layer {vl.name!r} (collection {RENDER_COLLECTION!r})")

    print(f"figure {height:.3f} tall; sticks START sized for a {output_height_px}px "
          f"GENERATION with the figure filling {frame_fraction:.0%} of frame")
    print(f"  -> joint r {joint_r:.5f}, body stick r {body_r:.5f}, "
          f"hand stick r {hand_r:.5f}")
    info = install_radius_drivers(obj, arm_obj, max_pixels=max_pixels)
    print(f"radii now DRIVEN from {info['camera']!r} at bone "
          f"{info['anchor_bone']!r}; sizing mode: {info['mode']}")
    print(f"  frame_fraction is no longer baked, and the output height is read "
          f"live from the render settings — sticks hold {2 * BODY_PX:.0f}px "
          f"across at any camera distance, resolution or preview percentage")
    print(f"source mesh: {report['verts']} verts, {report['edges']} edges, "
          f"{len(obj.vertex_groups)} vertex groups")
    if report["rebound"]:
        print(f"\nrebound {len(report['rebound'])} keypoint(s) onto deform bones "
              f"(the named bone has use_deform=False):")
        for a, b in sorted(report["rebound"].items()):
            print(f"  {a:14s} -> {b}")
    if report["missing"]:
        print(f"\nUNMAPPED ({len(report['missing'])}):")
        for m in report["missing"]:
            print(f"  {m}")
    return obj


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    frame_fraction = DEFAULT_FRAME_FRACTION
    if "--frame-fraction" in argv:
        frame_fraction = float(argv[argv.index("--frame-fraction") + 1])
    render_h = None
    if "--output-height" in argv:
        render_h = int(argv[argv.index("--output-height") + 1])
    # The sizing A/B: omit for the render canvas (what controlnet_aux draws, and
    # what FUK's own preprocessor produces), or pass the model's budget to size
    # for the smaller stage-1 pass the model really reads the map at.
    max_pixels = None
    if "--max-pixels" in argv:
        max_pixels = float(argv[argv.index("--max-pixels") + 1])

    src = Path(bpy.data.filepath)
    if "--out" in argv:
        dst = Path(argv[argv.index("--out") + 1])
        if not dst.is_absolute():
            dst = src.parent / dst
    elif "--drive-only" in argv:
        dst = src
    else:
        dst = src.with_name(src.stem + "_OP.blend")

    if "--drive-only" in argv:
        # Upgrade a blend that was already converted, without rebuilding it.
        # The saved prototype carries scene work the generator does not author
        # (the Canny and Bone view layers, camera placement), so a full rebuild
        # from OPii_Rig.blend would throw that away.
        obj = bpy.data.objects.get(SOURCE_MESH)
        arm_obj = bpy.data.objects.get(RIG_OBJECT)
        if obj is None or arm_obj is None:
            raise RuntimeError(f"--drive-only needs an existing {SOURCE_MESH!r} "
                               f"and {RIG_OBJECT!r} in the file")
        info = install_radius_drivers(obj, arm_obj, max_pixels=max_pixels)
        print(f"radii driven from {info['camera']!r} at bone "
              f"{info['anchor_bone']!r}; sizing mode: {info['mode']}")
    else:
        convert(frame_fraction=frame_fraction, output_height_px=render_h,
                max_pixels=max_pixels)

    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
