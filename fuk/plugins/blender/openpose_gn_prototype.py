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

# --- face: 70 landmarks -----------------------------------------------------
# draw_facepose (identical in controlnet_aux's open_pose and dwpose) is 70 white
# dots of radius 3 and nothing else — no sticks, no per-point colour. So the
# INDEX of a landmark never reaches the map; only where the dots sit does. The
# iBUG-68 numbering is kept anyway because the expressions below are written
# against it: 0-16 jaw, 17-21 / 22-26 brows, 27-35 nose, 36-41 / 42-47 eyes,
# 48-59 outer lips, 60-67 inner lips, then 68 / 69 the pupils.
#
# The rig has no facial bones to hang these on (head.x is the only deform bone
# above the neck), so the face is a fixed template in millimetres, origin at
# the nose tip, axes (lateral: + is the figure's left, up, forward). All 70
# points ride head.x rigidly; expression comes from shape keys on the source
# mesh, which evaluate before the Armature modifier and so compose with any
# head pose.
#
# Only the figure's left half and the midline are written out. The right half
# is mirrored, which keeps the template exactly symmetric.
#
# The front-view proportions (lateral, up) were corrected against a close-up map
# from the control LoRA's own training set, scaled by its pupil spacing: eye line
# 40mm above the nose tip, lids 10mm apart, jaw 136mm wide and nearly parallel
# down to mouth level. The first cut had the eyes 6mm low, the lids 8mm apart
# and a narrower, heart-shaped jaw. Brows sit 23mm over the eye line — the
# sample had them at 30, which one map cannot separate from a raised expression.
# Depth (forward) is still anatomical estimate; a frontal map says nothing
# about it.
FACE_PX = 3.0            # draw_facepose: cv2.circle(..., 3, ...), so 6px across
FACE_KIND = 3.0
FACE_RGB = (1.0, 1.0, 1.0)
FACE_IPD_MM = 63.0       # pupil to pupil in the template below
FACE_WIDTH_MM = 134.0    # jaw landmark 0 to 16
# 0-1 is the natural travel of each expression; the sliders run well past it in
# both directions on purpose. A shape key extrapolates linearly outside 0-1, so
# 2.0 is the same move twice as far and a negative value is its opposite (a
# negative blink is a wide eye, a negative brow_raise a lowered brow). The
# training maps come from a detector that snaps face landmarks to about 1/48 of
# its face crop — roughly 5mm on a close-up — so an anatomically honest 4mm
# move can sit below anything the control model ever saw change.
FACE_SLIDER_RANGE = (-3.0, 3.0)
FACE_LIFT_MM = 40.0      # how far the dots are slid toward the camera; see build_node_group

_FACE_HALF = {
    # jaw, chin (8) up to the ear (16)
    8: (0, -76, -12), 9: (16, -74, -16), 10: (32, -66, -24), 11: (47, -55, -36),
    12: (60, -37, -50), 13: (65, -20, -63), 14: (66, 0, -74), 15: (67, 17, -82),
    16: (67, 36, -88),
    # left brow, inner to outer
    22: (10, 57, -14), 23: (21, 62, -13), 24: (32, 63, -15), 25: (44, 60, -20),
    26: (53, 52, -28),
    # nose: bridge down to the tip, then the base
    27: (0, 40, -22), 28: (0, 27, -15), 29: (0, 13, -7), 30: (0, 0, 0),
    33: (0, -13, -12), 34: (6, -10, -15), 35: (16, -5, -22),
    # left eye: inner corner, upper lid, outer corner, lower lid
    42: (16, 40, -24), 43: (26, 45, -21), 44: (37, 45, -22), 45: (43, 40, -29),
    46: (37, 35, -23), 47: (26, 35, -22),
    # outer lips: upper centre round to the corner (54), then the lower lip
    51: (0, -24, -10), 52: (7, -23, -11), 53: (17, -26, -17), 54: (28, -31, -27),
    55: (18, -38, -19), 56: (8, -42, -14), 57: (0, -43, -13),
    # inner lips
    62: (0, -30, -13), 63: (8, -30, -14), 64: (22, -31, -25),
    65: (8, -33, -14), 66: (0, -33, -13),
    # left pupil
    69: (31.5, 40, -21),
}
# figure's right -> the left-side landmark it mirrors
_FACE_MIRROR = {
    7: 9, 6: 10, 5: 11, 4: 12, 3: 13, 2: 14, 1: 15, 0: 16,
    21: 22, 20: 23, 19: 24, 18: 25, 17: 26,
    32: 34, 31: 35,
    39: 42, 38: 43, 37: 44, 36: 45, 41: 46, 40: 47,
    50: 52, 49: 53, 48: 54, 59: 55, 58: 56,
    61: 63, 60: 64, 67: 65,
    68: 69,
}
FACE_PUPIL_L, FACE_PUPIL_R, FACE_NOSE_TIP = 69, 68, 30


def face_template_mm():
    """{landmark: (lateral, up, forward)} for all 70, in mm from the nose tip."""
    pts = dict(_FACE_HALF)
    for right, left in _FACE_MIRROR.items():
        x, y, z = _FACE_HALF[left]
        pts[right] = (-x, y, z)
    assert sorted(pts) == list(range(70)), "face template must cover 0-69"
    return pts


def face_expressions_mm():
    """{shape key: {landmark: (d_lateral, d_up, d_forward)}}.

    Offsets in mm at slider value 1.0. A shape key is a straight-line blend, so
    these are the END positions of each move rather than arcs — fine over the
    range a face actually travels, and it is what keeps them keyable.
    """
    def sym(entries):
        """Mirror left-side offsets onto the right; midline points pass through."""
        left_to_right = {l: r for r, l in _FACE_MIRROR.items()}
        out = {}
        for idx, (dx, dy, dz) in entries.items():
            out[idx] = (dx, dy, dz)
            if idx in left_to_right:
                out[left_to_right[idx]] = (-dx, dy, dz)
        return out

    def mirrored(entries):
        """The same move on the figure's right side only."""
        left_to_right = {l: r for r, l in _FACE_MIRROR.items()}
        return {left_to_right[i]: (-dx, dy, dz) for i, (dx, dy, dz) in entries.items()}

    # The mandible hinges near the ear, so the drop fades to nothing along the
    # jawline; the lower lip follows the chin and the corners are dragged part way.
    chin = (0.0, -22.0, -8.0)
    def drop(w, dx=0.0):
        return (dx, chin[1] * w, chin[2] * w)
    jaw_open = sym({8: drop(1.0), 9: drop(0.97), 10: drop(0.88), 11: drop(0.72),
                    12: drop(0.5), 13: drop(0.28), 14: drop(0.1),
                    55: drop(0.8), 56: drop(0.8), 57: drop(0.8),
                    65: drop(0.8), 66: drop(0.8),
                    54: drop(0.35, -2.0), 64: drop(0.35, -2.0)})

    brow_l = {i: (0.0, 7.0, 0.5) for i in (22, 23, 24, 25, 26)}
    blink_l = {43: (0.0, -9.5, -0.5), 44: (0.0, -9.5, -0.5)}

    return {
        "jaw_open": jaw_open,
        "smile": sym({54: (7, 5, -5), 64: (6, 4.5, -4.5), 53: (3, 2, -2),
                       55: (3, 2, -2), 52: (1, 0.5, 0), 56: (1, 0.5, 0),
                       63: (2, 1, -1), 65: (2, 1, -1),
                       46: (0, 1.2, 0), 47: (0, 1.2, 0)}),
        "frown": sym({54: (-1, -5, -1), 64: (-1, -4.5, -1),
                       53: (0, -2, 0), 55: (0, -2, 0)}),
        "pucker": sym({54: (-9, 0, 5), 64: (-8, 0, 5), 53: (-5, 0, 5),
                        55: (-5, 0, 5), 52: (-2, 0, 5), 56: (-2, 0, 5),
                        63: (-3, 0, 5), 65: (-3, 0, 5), 51: (0, 0, 5),
                        57: (0, 0, 5), 62: (0, 0, 5), 66: (0, 0, 5)}),
        "brow_raise.L": brow_l,
        "brow_raise.R": mirrored(brow_l),
        "brow_furrow": sym({22: (-3, -4, 0), 23: (-2, -3, 0),
                             24: (-1, -1.5, 0)}),
        "blink.L": blink_l,
        "blink.R": mirrored(blink_l),
        # Both pupils travel together, so these are NOT mirrored. Negative values
        # look the other way: gaze_h +1 is toward the figure's left.
        "gaze_h": {FACE_PUPIL_L: (6, 0, 0), FACE_PUPIL_R: (6, 0, 0)},
        "gaze_v": {FACE_PUPIL_L: (0, 4, 0), FACE_PUPIL_R: (0, 4, 0)},
    }


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


def face_scale(height):
    """World units per template millimetre.

    Matched to the body's own eye spacing rather than to stature, so the face
    template and the COCO eye keypoints agree with each other by construction.
    """
    return (FACE_OFFSETS[15][2] - FACE_OFFSETS[14][2]) * height / FACE_IPD_MM


def face_landmarks(arm_obj, height):
    """({landmark: position}, to_world) for the 70-point face, or ({}, None).

    The template hangs off the body's Nose keypoint, tip on tip. `to_world`
    turns a template-space millimetre OFFSET into an armature-space vector, for
    the expression shape keys.
    """
    body = face_points(arm_obj, height)
    if 0 not in body:
        return {}, None
    up, forward, left = figure_axes(arm_obj)
    s = face_scale(height)

    def to_world(d):
        return (left * d[0] + up * d[1] + forward * d[2]) * s

    return {i: body[0] + to_world(p) for i, p in face_template_mm().items()}, to_world


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


def build_source_mesh(arm_obj, height, output_height_px, frame_fraction,
                      with_face=True):
    """One vertex per keypoint, one 2-vertex edge per limb, each vertex weighted
    to the bone it follows. Returns (object, report)."""
    joint_r, body_r, hand_r = stick_radii(height, output_height_px, frame_fraction)
    face_r = joint_r * FACE_PX / JOINT_PX

    verts, edges, groups = [], [], {}
    colors, radii, kinds = [], [], []
    missing, rebound = [], {}
    face = face_points(arm_obj, height)
    landmarks, to_world = face_landmarks(arm_obj, height) if with_face else ({}, None)
    if landmarks:
        # A real detection puts the COCO eye keypoints on the pupils. The
        # hand-set FACE_OFFSETS eyes sit about 1cm low and 2cm back of where
        # this template's pupils land, which is invisible on its own but reads
        # as two pairs of eyes once the landmarks are drawn beside them.
        face[14], face[15] = landmarks[FACE_PUPIL_R], landmarks[FACE_PUPIL_L]

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

    # Face landmarks: lone vertices like the joints, but their own kind so the
    # node group can give them the smaller white dot.
    face_vert = {}
    if landmarks and arm_obj.data.bones.get(HEAD_BONE) is not None:
        for idx in sorted(landmarks):
            face_vert[idx] = add_vertex(HEAD_BONE, "head", FACE_RGB, face_r,
                                        FACE_KIND, at=landmarks[idx],
                                        bind_to=HEAD_BONE)

    old = bpy.data.objects.get(SOURCE_MESH)
    if old:
        old_mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        # Otherwise the rebuilt mesh comes back as OP_source.001 beside an
        # orphan, and anything looking the datablock up by name finds the corpse.
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
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

    # Expressions. Shape keys sit below the modifier stack, so they deform the
    # landmarks first and head.x then carries the result — an expression holds
    # through any head pose, and each key is an ordinary animatable slider.
    shape_keys = []
    if face_vert:
        obj.shape_key_add(name="Basis", from_mix=False)
        for name, offsets in face_expressions_mm().items():
            key = obj.shape_key_add(name=name, from_mix=False)
            key.slider_min, key.slider_max = FACE_SLIDER_RANGE
            for idx, delta in offsets.items():
                vi = face_vert[idx]
                key.data[vi].co = verts[vi] + to_world(delta)
            shape_keys.append(name)

    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj

    return obj, {"verts": len(verts), "edges": len(edges), "missing": missing,
                 "rebound": rebound, "face_landmarks": len(face_vert),
                 "shape_keys": shape_keys}


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


def build_node_group(material, joint_radius, body_radius, hand_radius,
                     face_radius=None, face_lift=0.0):
    old = bpy.data.node_groups.get(GN_GROUP)
    if old:
        bpy.data.node_groups.remove(old)
    ng = bpy.data.node_groups.new(GN_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    if face_radius is not None:
        # A plain checkbox on the modifier, for the shots where the face should
        # be left to the prompt. Unlike the radii this is a user choice rather
        # than a derived value, so it belongs where a user can see it.
        show_face = ng.interface.new_socket(FACE_TOGGLE, in_out="INPUT",
                                            socket_type="NodeSocketBool")
        show_face.default_value = True

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
    sphere.name = NODE_JOINT
    sphere.inputs["Segments"].default_value = 12
    sphere.inputs["Rings"].default_value = 8
    sphere.inputs["Radius"].default_value = joint_radius
    iop = n("GeometryNodeInstanceOnPoints")
    ng.links.new(split(gin.outputs[0], -0.5, 0.5), iop.inputs["Points"])
    ng.links.new(sphere.outputs["Mesh"], iop.inputs["Instance"])
    realize = n("GeometryNodeRealizeInstances")
    ng.links.new(iop.outputs["Instances"], realize.inputs["Geometry"])
    ng.links.new(realize.outputs["Geometry"], join.inputs["Geometry"])

    # --- face: the same idea at draw_facepose's smaller radius ---
    # A second sphere rather than a scaled instance of the first, for the reason
    # above: per-point scale on Instance on Points did not take.
    if face_radius is not None:
        face_sphere = n("GeometryNodeMeshUVSphere")
        face_sphere.name = NODE_FACE
        face_sphere.inputs["Segments"].default_value = 12
        face_sphere.inputs["Rings"].default_value = 8
        face_sphere.inputs["Radius"].default_value = face_radius
        # draw_facepose runs AFTER the body and hands, so in a real map the
        # white dots sit on top of everything: the pupil dot over the coloured
        # eye joint it shares a position with, the lip dots over the neck stick
        # that crosses them. Rendered as geometry, depth decides instead and the
        # sticks win. Sliding each landmark a few centimetres along its own view
        # ray toward the camera puts it in front without moving where it
        # projects; the cost is a dot 2-3% larger at portrait distance.
        cam = n("GeometryNodeInputActiveCamera")
        cam_info = n("GeometryNodeObjectInfo")
        cam_info.transform_space = "RELATIVE"
        ng.links.new(cam.outputs[0], cam_info.inputs["Object"])
        position = n("GeometryNodeInputPosition")
        to_cam = n("ShaderNodeVectorMath"); to_cam.operation = "SUBTRACT"
        ng.links.new(cam_info.outputs["Location"], to_cam.inputs[0])
        ng.links.new(position.outputs["Position"], to_cam.inputs[1])
        unit = n("ShaderNodeVectorMath"); unit.operation = "NORMALIZE"
        ng.links.new(to_cam.outputs["Vector"], unit.inputs[0])
        lift = n("ShaderNodeVectorMath"); lift.operation = "SCALE"
        lift.inputs["Scale"].default_value = face_lift
        ng.links.new(unit.outputs["Vector"], lift.inputs[0])
        lifted = n("GeometryNodeSetPosition")
        ng.links.new(split(gin.outputs[0], FACE_KIND - 0.5, FACE_KIND + 0.5),
                     lifted.inputs["Geometry"])
        ng.links.new(lift.outputs["Vector"], lifted.inputs["Offset"])

        face_iop = n("GeometryNodeInstanceOnPoints")
        ng.links.new(lifted.outputs["Geometry"], face_iop.inputs["Points"])
        ng.links.new(face_sphere.outputs["Mesh"], face_iop.inputs["Instance"])
        ng.links.new(gin.outputs[FACE_TOGGLE], face_iop.inputs["Selection"])
        face_realize = n("GeometryNodeRealizeInstances")
        ng.links.new(face_iop.outputs["Instances"], face_realize.inputs["Geometry"])
        ng.links.new(face_realize.outputs["Geometry"], join.inputs["Geometry"])

    # --- limbs: one tube per 2-vertex edge, one branch per stick width ---
    # There are exactly two widths (body 4, hand 1), so each branch gets a
    # fixed-radius profile circle. Driving the radius as a field was tried three
    # ways -- Set Curve Radius, the curve's built-in "radius" attribute, and
    # Curve to Mesh's Scale input -- and all three silently left the profile at
    # radius 1.0, i.e. 80x oversized.
    for lo, hi, radius, name in ((0.5, 1.5, body_radius, NODE_BODY),
                                 (1.5, 2.5, hand_radius, NODE_HAND)):
        to_curve = n("GeometryNodeMeshToCurve")
        ng.links.new(split(gin.outputs[0], lo, hi), to_curve.inputs[0])
        circle = n("GeometryNodeCurvePrimitiveCircle")
        circle.name = name
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
# belongs to the MODEL rather than to the render. That is inlined into the
# expression as a literal when stage-1 sizing is asked for; it is not stored in
# the scene either.

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

# Names given to the primitive nodes at build time, so the drivers can find
# them without guessing. FACE_TOGGLE is the modifier checkbox.
NODE_JOINT, NODE_FACE = "OP_JointSphere", "OP_FaceSphere"
NODE_BODY, NODE_HAND = "OP_BodyCircle", "OP_HandCircle"
FACE_TOGGLE = "Face"

# Below this many pixels of face width the landmarks are not drawn. Seventy 6px
# dots only stay separate once the eye landmarks (about 10mm apart on a 130mm
# face) are more than a dot's width from each other, which is 78px of face;
# under that they fuse into a white patch that says nothing about expression.
# Detectors drop face landmarks on small faces for the same reason, so a map
# with a blob where a distant face should be is not one the model was shown.
DEFAULT_FACE_MIN_PX = 80.0


def radius_sockets(ng):
    """[(label, pixels, driver data path)] for each radius input.

    The drivers go on the primitive nodes INSIDE the group rather than on
    modifier inputs. Exposing them as group inputs reads better in the UI, but
    Blender 5.x keeps a geometry nodes modifier's socket values in an
    ID-property group under modifier.properties.inputs, and driver_add() on
    that path fails as "not animatable" — a node socket's default_value is
    animatable and has been for every version this rig has seen.

    Nodes are found by the names build_node_group gives them. Groups built
    before that carry Blender's defaults ('Curve Circle' / 'Curve Circle.001',
    which depend on creation order), so those fall back to telling the two limb
    circles apart by their current radius.
    """
    if NODE_JOINT in ng.nodes:
        nodes = [ng.nodes[NODE_JOINT], ng.nodes[NODE_BODY], ng.nodes[NODE_HAND]]
        labelled = list(RADIUS_SOCKETS)
        if NODE_FACE in ng.nodes:
            nodes.append(ng.nodes[NODE_FACE])
            labelled.append(("Face Radius", FACE_PX))
    else:
        # A group built before the nodes were named: one sphere, two circles,
        # and no face branch.
        spheres = [n for n in ng.nodes if n.bl_idname == "GeometryNodeMeshUVSphere"]
        circles = [n for n in ng.nodes
                   if n.bl_idname == "GeometryNodeCurvePrimitiveCircle"]
        if len(spheres) != 1 or len(circles) != 2:
            raise RuntimeError(f"{ng.name!r}: expected 1 sphere and 2 circles, "
                               f"found {len(spheres)} and {len(circles)}")
        circles.sort(key=lambda c: c.inputs["Radius"].default_value, reverse=True)
        nodes, labelled = [spheres[0], circles[0], circles[1]], list(RADIUS_SOCKETS)

    out = []
    for node, (label, px) in zip(nodes, labelled):
        socket = node.inputs["Radius"]
        if socket.is_linked:
            raise RuntimeError(f"{ng.name!r}: {node.name!r} Radius is linked; "
                               "a link overrides default_value, so the driver "
                               "would have no effect")
        idx = list(node.inputs).index(socket)
        out.append((label, px,
                    f'nodes["{node.name}"].inputs[{idx}].default_value'))
    return out


def install_radius_drivers(obj, arm_obj, camera=None, max_pixels=None,
                           face_min_px=DEFAULT_FACE_MIN_PX):
    """Drive the radii off the camera so sticks and dots hold their PIXEL width.

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
    face_w = FACE_WIDTH_MM * face_scale(figure_height(arm_obj))

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

        v = var("d", "LOC_DIFF")
        v.targets[0].id = cam
        v.targets[1].id = arm_obj
        v.targets[1].bone_target = anchor_bone

        for name, ident, dpath in (("sw", cam.data, "sensor_width"),
                                   ("f", cam.data, "lens"),
                                   ("rx", scene, "render.resolution_x"),
                                   ("ry", scene, "render.resolution_y"),
                                   ("p", scene, "render.resolution_percentage")):
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
        #
        # Variable names are single letters (d depth, f focal length, p
        # percentage) and the spacing is stripped because Blender stores a
        # driver expression in 256 bytes and truncates anything longer without
        # complaint — the face expression below does not fit written out.
        rw, rh = "(rx*p/100)", "(ry*p/100)"
        fit = "min(1,ry/max(1,rx))"
        canvas = f"max(16,{rh})"
        if budget_px:
            # Size for the smaller stage-1 pass the runner denoises with control
            # at when the request is over budget, mirroring _fit_pixel_budget
            # including its deliberate round DOWN to the latent grid. min()
            # covers the under-budget case, where the ratio exceeds 1.
            scale = f"min(1,sqrt({budget_px:.0f}/max(1,{rw}*{rh})))"
            gen = f"max(16,floor({rh}*{scale}/16)*16)"
        else:
            # Size for the rendered canvas: 8px in the map as authored, which is
            # what controlnet_aux itself draws and what FUK's own openpose
            # preprocessor produces.
            gen = canvas
        # max() guards stop a zeroed resolution from erroring the driver, which
        # in Blender leaves the socket at its last value with no visible failure.
        expr = f"{px}*d*sw*{fit}/(max(1,f)*{gen})"
        if label == "Face Radius" and face_min_px and face_min_px > 0:
            # Gate the landmarks on how wide the face is in the map: the dot
            # radius ramps from nothing to full over the single pixel above the
            # threshold. Written with min/max rather than a comparison so it
            # stays inside Blender's built-in expression evaluator and never
            # needs "auto-run Python scripts". Measured on the rendered canvas
            # in both sizing modes — the stage-1 form does not fit in 256 bytes.
            face_px = f"{face_w:.5f}*max(1,f)*{canvas}/max(1e-6,d*sw*{fit})"
            expr += f"*min(1,max(0,{face_px}-{face_min_px:.0f}))"
        if len(expr) > 255:
            raise RuntimeError(f"driver expression for {label} is {len(expr)} "
                               "bytes; Blender truncates at 255")
        drv.expression = expr

    return {"camera": cam.name, "anchor_bone": anchor_bone,
            "mode": (f"stage-1 (budget {budget_px/1e6:.2f}MP)" if budget_px
                     else "render canvas"),
            "face": any(label == "Face Radius" for label, _, _ in sockets),
            "face_min_px": face_min_px}


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
ARMATURE_COLLECTION = "OP_armature"
VIEW_LAYER = "OpenPose"

# The kit's own stick figures, superseded by OP_Generator: the OPii mannequin
# (a posable stick-man mesh — one of the rig's three "characters", next to Chad
# and Stacy) and the Limb_system curve generator. Both are real, render-visible
# geometry, and "Bone" is the view layer the kit built to render them.
LEGACY_COLLECTIONS = ("Opii", "Limb_system")
LEGACY_MESH = "OPii_mesh"
LEGACY_VIEW_LAYER = "Bone"


def _layer_collections(lc):
    yield lc
    for child in lc.children:
        yield from _layer_collections(child)


def isolate_view_layers(scene, src_obj):
    """Give every stick figure in the file exactly one view layer to render in.

    A dedicated view layer only isolates in one direction. The OpenPose layer
    excludes everything else, but nothing excluded the skeletons from the
    OTHER layers — and a depth map rendered from the working layer of a scene
    with this rig in it came back with a stick figure in it. Two sources:

      * The OPii mannequin. It sat in OPii_Rig as well as in its own Opii
        collection, and OPii_Rig cannot be excluded from the working layer
        (the rig controls live there), so the mannequin rendered wherever the
        rig was posable. It is close to our skeleton but not on it, so under a
        depth + openpose composite it read as a second, offset figure. It now
        lives in Opii alone, enabled only in the kit's Bone layer.

      * Our own emission sticks. A collection created after the view layers
        exist is ENABLED in all of them. OP_render shared a collection with
        the armature — which every layer that poses a mesh needs, or the
        Armature modifier has no evaluated target and the mesh freezes at rest
        — so it could not be excluded from Canny or Bone without breaking
        them. The armature now rides in OP_armature, enabled everywhere, and
        OP_render holds geometry only.

    OP_render is deliberately LEFT ON in the working layer (the scene's first):
    with the mannequin gone it is the only thing showing the pose in the
    viewport. FUK's bridge excludes it for the duration of its own renders
    (render._isolate_pose_geometry), so it never reaches a control map; a
    plain F12 from the working layer will still show it.

    Explicit and idempotent — it sets every state rather than toggling, and is
    what --drive-only runs to upgrade an already-converted file.
    """
    col = bpy.data.collections.get(RENDER_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(RENDER_COLLECTION)
        scene.collection.children.link(col)
    for c in list(src_obj.users_collection):
        if c.name != col.name:
            c.objects.unlink(src_obj)
    if src_obj.name not in col.objects:
        col.objects.link(src_obj)

    arm_col = bpy.data.collections.get(ARMATURE_COLLECTION)
    if arm_col is None:
        arm_col = bpy.data.collections.new(ARMATURE_COLLECTION)
        scene.collection.children.link(arm_col)
    rig = bpy.data.objects.get(RIG_OBJECT)
    if rig is not None:
        if rig.name not in arm_col.objects:
            arm_col.objects.link(rig)
        if rig.name in col.objects:
            col.objects.unlink(rig)

    # The mannequin may only live in a collection that can be excluded.
    legacy = bpy.data.objects.get(LEGACY_MESH)
    if legacy is not None:
        homes = [c for c in legacy.users_collection if c.name in LEGACY_COLLECTIONS]
        if homes:
            for c in list(legacy.users_collection):
                if c.name not in LEGACY_COLLECTIONS:
                    c.objects.unlink(legacy)

    vl = scene.view_layers.get(VIEW_LAYER) or scene.view_layers.new(VIEW_LAYER)
    work = scene.view_layers[0].name
    changed = []
    for layer in scene.view_layers:
        is_pose = layer.name == vl.name       # bpy wrappers aren't identity-stable
        for lc in _layer_collections(layer.layer_collection):
            if lc.name == ARMATURE_COLLECTION:
                want = False
            elif lc.name == RENDER_COLLECTION:
                want = not (is_pose or layer.name == work)
            elif lc.name in LEGACY_COLLECTIONS:
                want = layer.name != LEGACY_VIEW_LAYER
            elif is_pose and lc.name != layer.layer_collection.name:
                want = True
            else:
                continue
            if lc.exclude != want:
                lc.exclude = want
                changed.append(f"{lc.name} {'out of' if want else 'into'} {layer.name!r}")
    return vl, changed


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

    # The armature has to stay IN the OpenPose view layer. Excluding its
    # collection drops it from that layer's depsgraph, and the Armature
    # modifier then has no evaluated target — the skeleton silently freezes at
    # rest. An armature renders nothing, so keeping it costs nothing.
    vl, _ = isolate_view_layers(scene, src_obj)
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
            max_pixels=None, with_face=True, face_min_px=DEFAULT_FACE_MIN_PX):
    arm_obj = bpy.data.objects.get(RIG_OBJECT)
    if arm_obj is None:
        raise RuntimeError(f"no armature object named {RIG_OBJECT!r}")

    height = figure_height(arm_obj)
    if output_height_px is None:
        output_height_px = DEFAULT_OUTPUT_HEIGHT

    joint_r, body_r, hand_r = stick_radii(height, output_height_px, frame_fraction)

    material = build_material()
    obj, report = build_source_mesh(arm_obj, height, output_height_px,
                                    frame_fraction, with_face=with_face)
    has_face = bool(report["face_landmarks"])
    ng = build_node_group(material, joint_r, body_r, hand_r,
                          face_radius=joint_r * FACE_PX / JOINT_PX if has_face else None,
                          face_lift=FACE_LIFT_MM * face_scale(height))
    gn = obj.modifiers.new("OP_Generator", "NODES")
    gn.node_group = ng

    vl = setup_scene(obj)
    print(f"isolated on view layer {vl.name!r} (collection {RENDER_COLLECTION!r})")

    print(f"figure {height:.3f} tall; sticks START sized for a {output_height_px}px "
          f"GENERATION with the figure filling {frame_fraction:.0%} of frame")
    print(f"  -> joint r {joint_r:.5f}, body stick r {body_r:.5f}, "
          f"hand stick r {hand_r:.5f}")
    info = install_radius_drivers(obj, arm_obj, max_pixels=max_pixels,
                                  face_min_px=face_min_px)
    if has_face:
        gate = (f"drawn once the face is {face_min_px:.0f}px wide in the map"
                if face_min_px and face_min_px > 0 else "always drawn")
        print(f"face: {report['face_landmarks']} landmarks on {HEAD_BONE!r}, {gate}; "
              f"'{FACE_TOGGLE}' checkbox on the OP_Generator modifier turns it off")
        print(f"  expressions (shape keys on {SOURCE_MESH}): "
              f"{', '.join(report['shape_keys'])}")
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
    # --no-face builds the body and hands only, with the COCO eye keypoints at
    # their original hand-set offsets. --face-min-px 0 draws the landmarks at
    # any size instead of hiding them on small faces.
    with_face = "--no-face" not in argv
    face_min_px = DEFAULT_FACE_MIN_PX
    if "--face-min-px" in argv:
        face_min_px = float(argv[argv.index("--face-min-px") + 1])

    src = Path(bpy.data.filepath)
    if "--out" in argv:
        dst = Path(argv[argv.index("--out") + 1])
        if not dst.is_absolute():
            dst = src.parent / dst
    elif "--drive-only" in argv or "--in-place" in argv:
        # --in-place is a full rebuild of the source mesh, node group and
        # drivers inside an already-converted file. Needed for anything that
        # changes the mesh itself (the face landmarks), which --drive-only
        # deliberately leaves alone.
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
        info = install_radius_drivers(obj, arm_obj, max_pixels=max_pixels,
                                      face_min_px=face_min_px)
        print(f"radii driven from {info['camera']!r} at bone "
              f"{info['anchor_bone']!r}; sizing mode: {info['mode']}")
        _, changed = isolate_view_layers(bpy.context.scene, obj)
        print("view layers: " + ("; ".join(changed) if changed
                                 else "already isolated"))
    else:
        convert(frame_fraction=frame_fraction, output_height_px=render_h,
                max_pixels=max_pixels, with_face=with_face,
                face_min_px=face_min_px)

    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
