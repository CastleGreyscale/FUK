#!/usr/bin/env python3
"""
Give the 21-keypoint hand rig real proportions, and an arm to orient it by.

Third script in the set:
    openpose_hand21.py       colours  (the render contract)
    hand_rig_constraints.py  wrist parent, joint limits, curl controls
    hand_rig_anatomy.py      bone lengths, forearm / upper arm   <- this one

Usage:
    blender Hand_OP21.blend --background --factory-startup \
        --python hand_rig_anatomy.py -- --out Hand_OP21_anat.blend

    --proportions      rescale segments to anthropometric ratios (default on)
    --no-proportions   leave lengths alone
    --arm none         no arm (default: forearm)
    --arm forearm      add elbow + forearm
    --arm full         add shoulder + upper arm as well
    --inplace / --out PATH   as the other scripts


Why the lengths mattered
------------------------

Every phalanx in the shipped rig is roughly the same length, so each distal
segment came out 2.2x to 2.7x longer than it should be relative to its proximal.
A real finger tapers hard at the last joint; this one didn't taper at all, which
is why a pointing index read as an impossibly long digit in generated output.
Measured against the rig:

    finger   rig prox:mid:dist     real prox:mid:dist
    index    1.00 : 0.76 : 0.89    1.00 : 0.56 : 0.40
    middle   1.00 : 1.03 : 1.07    1.00 : 0.59 : 0.39
    ring     1.00 : 0.96 : 1.09    1.00 : 0.62 : 0.42
    little   1.00 : 0.87 : 1.05    1.00 : 0.55 : 0.48

Reference means are from Buryanov & Kotiuk (2010), "Proportions of Hand
Segments" — population averages, so treat the absolute millimetres as
approximate. The ratios are the part that matters and they are stable.

Overall hand size is preserved: everything is scaled so the middle finger still
spans the same wrist-to-tip distance it does now, which keeps existing cameras
and framing valid.


Why an arm
----------

A bare hand gives the pose encoder nothing to orient against — it cannot tell a
palm from a back, or which way the wrist is turned. An arm resolves that, and it
also pulls the hand away from filling the whole frame, which is closer to the
full-body crops the encoder was trained on.

The arm is drawn in the BODY convention, not the hand one, because that is what
a real OpenPose render does: draw_bodypose paints the arm, draw_handpose paints
the hand, and they use different palettes and different opacities. So the arm
segments here take COCO-18 limb colours at 0.6 attenuation while the hand stays
at full — see openpose_coco18.py for the same tables.

This is the LEFT hand (see hand_rig_constraints.py for the derivation), so the
arm is the left arm: body keypoints 5 LShoulder, 6 LElbow, 7 LWrist.
"""

import re
import sys
import math
import shutil
from pathlib import Path

import bpy
from mathutils import Vector


ARMATURE_OBJECT = "HandArmature.L"
WRIST_BONE = "Wrist.L"

JOINT_RE = re.compile(r"^hand\.(\d+)$")
BONE_RE = re.compile(r"^handBone\.(\d+)$")

# --- anthropometric segment lengths, mm ------------------------------------
# (metacarpal, proximal, middle, distal); thumb is (carpal, metacarpal,
# proximal, distal) to match the rig's four thumb-chain bones.
SEGMENTS_MM = {
    "Finger0": (68.1, 39.8, 22.4, 15.8),   # index
    "Finger1": (64.6, 44.6, 26.3, 17.4),   # middle
    "Finger2": (58.0, 41.4, 25.7, 17.3),   # ring
    "Finger3": (53.7, 32.7, 18.1, 15.8),   # little
}
# Thumb: wrist->CMC is a short carpal hop, then metacarpal, proximal, distal.
THUMB_MM = {"Hand": 25.0, "Thumb1": 46.2, "Thumb2": 31.6, "Thumb3": 21.7}

# Arm segments as multiples of hand length (wrist to middle fingertip).
# Adult means: hand ~185mm, forearm ~260mm, upper arm ~310mm.
FOREARM_RATIO = 260.0 / 185.0
UPPERARM_RATIO = 310.0 / 185.0

# --- body (COCO-18) palette for the arm ------------------------------------
# Same tables as openpose_coco18.py. Limbs are attenuated, joints are not.
COLORS = [
    "ff0000", "ff5500", "ffaa00", "ffff00", "aaff00", "55ff00",
    "00ff00", "00ff55", "00ffaa", "00ffff", "00aaff", "0055ff",
    "0000ff", "5500ff", "aa00ff", "ff00ff", "ff00aa", "ff0055",
]
BODY_LIMB_ALPHA = 0.6
ARM_JOINT_COLOR = {          # body keypoint index -> colour
    "LShoulder": COLORS[5],  # keypoint 5
    "LElbow":    COLORS[6],  # keypoint 6
    "LWrist":    COLORS[7],  # keypoint 7
}
ARM_LIMB_COLOR = {           # limbSeq order, not the joint's own colour
    "UpperArm": COLORS[4],   # limbSeq[4]  LShoulder -> LElbow
    "Forearm":  COLORS[5],   # limbSeq[5]  LElbow    -> LWrist
}
# Body sticks are FOUR times the width of hand sticks. draw_bodypose builds each
# limb with ellipse2Poly(..., (length/2, stickwidth), ...) where stickwidth = 4 is
# a SEMI-axis, so the stick is 8px across; draw_handpose uses cv2.line with
# thickness=2. Easy to read as 2x and get half the width.
ARM_LIMB_RADIUS_SCALE = 4.0

# Body and hand joints are the same size — both are cv2.circle(..., 4, ...). So
# the arm dots match an ordinary hand dot, NOT the rig's oversized wrist sphere
# (which is itself 2x the others, a pre-existing deviation from the convention).
ARM_JOINT_MATCHES = "hand.008"


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def set_color(obj, hexcode, factor=1.0):
    obj.color = (*(srgb_to_linear(int(hexcode[i:i + 2], 16) / 255.0 * factor)
                   for i in (0, 2, 4)), 1.0)


# ---------------------------------------------------------------------------
# Snapshot / restore of the rendered primitives
# ---------------------------------------------------------------------------

def snapshot_attachments(arm_obj):
    """Record where each sphere and tube sits, as (bone, 'head'|'tail').

    Both are bone-parented with baked positions, so nothing follows a bone whose
    length changes — the tube keeps its old span and the sphere drifts with the
    tail. Capturing the intent here, before any edit, lets the whole visual layer
    be re-derived from the skeleton afterwards instead of patched.
    """
    arm = arm_obj.data
    out = {"joints": {}, "tubes": {}}

    for obj in bpy.data.objects:
        bone_name = obj.parent_bone
        if obj.parent_type != "BONE" or not bone_name or bone_name not in arm.bones:
            continue
        bone = arm.bones[bone_name]
        if JOINT_RE.match(obj.name):
            # Snap to whichever end it currently sits on.
            pos = Vector(obj.location)
            end = "head" if (pos - bone.head_local).length <= (pos - bone.tail_local).length else "tail"
            out["joints"][obj.name] = (bone_name, end)
        elif BONE_RE.match(obj.name):
            out["tubes"][obj.name] = (bone_name, "span")
    return out


def rebuild_primitives(arm_obj, snap):
    """Put every sphere back on its keypoint and re-span every tube."""
    arm = arm_obj.data
    moved = 0

    for name, (bone_name, end) in snap["joints"].items():
        obj = bpy.data.objects.get(name)
        if obj is None or bone_name not in arm.bones:
            continue
        bone = arm.bones[bone_name]
        obj.location = bone.head_local if end == "head" else bone.tail_local
        moved += 1

    for name, (bone_name, _) in snap["tubes"].items():
        obj = bpy.data.objects.get(name)
        if obj is None or bone_name not in arm.bones or obj.type != "CURVE":
            continue
        bone = arm.bones[bone_name]
        obj.location = bone.head_local
        vec = bone.tail_local - bone.head_local
        for spline in obj.data.splines:
            pts = spline.bezier_points if spline.type == "BEZIER" else spline.points
            if len(pts) < 2:
                continue
            # A POLY point's co is (x, y, z, w) and a freshly added one has
            # w = 0, which collapses it — set the weight explicitly, not just
            # the position.
            for pt, target in ((pts[0], Vector((0.0, 0.0, 0.0))), (pts[-1], vec)):
                if len(pt.co) == 4:
                    pt.co = (target.x, target.y, target.z, 1.0)
                else:
                    pt.co = (target.x, target.y, target.z)
        moved += 1

    return moved


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------

def target_lengths(arm_obj):
    """Anthropometric target length per bone, scaled so the middle finger keeps
    its current wrist-to-tip span (so existing framing still works)."""
    arm = arm_obj.data
    current_middle = sum(arm.bones[f"Finger1_{i}.L"].length for i in range(4))
    real_middle = sum(SEGMENTS_MM["Finger1"])
    scale = current_middle / real_middle

    targets = {}
    for prefix, mm in SEGMENTS_MM.items():
        for i, seg in enumerate(mm):
            targets[f"{prefix}_{i}.L"] = seg * scale
    for name, mm in THUMB_MM.items():
        targets[f"{name}.L"] = mm * scale
    return targets, scale


def rescale_bones(arm_obj, targets):
    """Set each bone's length while preserving its direction, carrying every
    descendant along so the chains stay contiguous.

    Positions are solved in plain vectors first and only then written back, with
    use_connect temporarily cleared. Editing in place does not work: Blender
    keeps a connected child's head glued to its parent's tail, so assigning the
    parent's tail already moves the child, and any explicit shift on top of that
    lands twice — stretching or collapsing the child. (Thumb3 went 0.0959 ->
    0.0141 that way.) Clearing connect makes the writes independent, and the
    solved positions keep the chain contiguous regardless.
    """
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    changed = []
    try:
        ebs = arm_obj.data.edit_bones
        rest = {eb.name: (eb.head.copy(), eb.tail.copy()) for eb in ebs}
        solved = {}

        def solve(eb, delta):
            head, tail = rest[eb.name]
            new_head = head + delta
            vec = tail - head
            length = vec.length
            target = targets.get(eb.name, length)
            if length < 1e-9:
                new_tail = tail + delta
            else:
                new_tail = new_head + vec.normalized() * target
                if eb.name in targets:
                    changed.append((eb.name, length, target))
            solved[eb.name] = (new_head, new_tail)
            child_delta = new_tail - tail
            for child in eb.children:
                solve(child, child_delta)

        for eb in ebs:
            if eb.parent is None:
                solve(eb, Vector((0.0, 0.0, 0.0)))

        connect = {eb.name: eb.use_connect for eb in ebs}
        for eb in ebs:
            eb.use_connect = False
        for eb in ebs:
            if eb.name in solved:
                eb.head, eb.tail = solved[eb.name]
        for eb in ebs:
            eb.use_connect = connect[eb.name]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return changed


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------

def _new_tube(name, parent_obj, bone_name, material, radius):
    """A curve that spans one bone, matching how the hand tubes are built.

    Thickness comes from bevel_depth rather than a scaled bevel object: object
    scale would stretch the tube's LENGTH too, and the length here has to stay
    exactly the bone's.
    """
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    if material is not None:
        curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent_obj
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    return obj


def _new_sphere(name, template, parent_obj, bone_name, diameter):
    """Reuse an existing joint sphere's mesh so the arm dots match the hand's."""
    obj = bpy.data.objects.new(name, template.data.copy())
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent_obj
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    base = max(template.dimensions) or 1.0
    s = diameter / base
    obj.scale = (s, s, s)
    return obj


def add_arm(arm_obj, mode, snap):
    """Add Forearm (+ UpperArm) bones, plus their body-convention primitives."""
    if mode == "none":
        return []

    arm = arm_obj.data
    hand_len = sum(arm.bones[f"Finger1_{i}.L"].length for i in range(4))
    forearm_len = hand_len * FOREARM_RATIO
    upper_len = hand_len * UPPERARM_RATIO

    wrist = arm.bones.get(WRIST_BONE)
    if wrist is None:
        raise RuntimeError(f"{WRIST_BONE} missing — run hand_rig_constraints.py first")
    wrist_pos = wrist.tail_local.copy()          # the hand origin / body LWrist
    back = Vector((0.0, -1.0, 0.0))              # arm extends away from the fingers

    elbow = wrist_pos + back * forearm_len
    shoulder = elbow + back * upper_len

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    created = []
    try:
        ebs = arm.edit_bones

        fore = ebs.get("Forearm.L") or ebs.new("Forearm.L")
        fore.head, fore.tail, fore.roll = elbow, wrist_pos, 0.0
        created.append("Forearm.L")

        if mode == "full":
            upper = ebs.get("UpperArm.L") or ebs.new("UpperArm.L")
            upper.head, upper.tail, upper.roll = shoulder, elbow, 0.0
            fore.parent = upper
            fore.use_connect = False
            created.append("UpperArm.L")
        else:
            fore.parent = None

        # The wrist (and through it the whole hand) now hangs off the forearm.
        ebs[WRIST_BONE].parent = fore
        ebs[WRIST_BONE].use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    # --- primitives, in the BODY palette ---
    template = (bpy.data.objects.get(ARM_JOINT_MATCHES)
                or bpy.data.objects.get("hand.000"))
    material = bpy.data.materials.get("JointColor")
    bevel_src = bpy.data.objects.get("handBoneBevel")
    # handBoneBevel's dimensions are the swept profile, i.e. the tube DIAMETER.
    hand_radius = (max(bevel_src.dimensions[:2]) / 2.0) if bevel_src else 0.00375
    tube_radius = hand_radius * ARM_LIMB_RADIUS_SCALE
    joint_dia = max(template.dimensions) if template else 0.015

    pairs = [("Forearm", "Forearm.L", "LElbow", elbow)]
    if mode == "full":
        pairs.append(("UpperArm", "UpperArm.L", "LShoulder", shoulder))

    for limb, bone_name, joint_name, joint_pos in pairs:
        tname = f"armBone_{limb}"
        old = bpy.data.objects.get(tname)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        tube = _new_tube(tname, arm_obj, bone_name, material, tube_radius)
        set_color(tube, ARM_LIMB_COLOR[limb], BODY_LIMB_ALPHA)
        snap["tubes"][tname] = (bone_name, "span")

        jname = f"armJoint_{joint_name}"
        old = bpy.data.objects.get(jname)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        if template is not None:
            sphere = _new_sphere(jname, template, arm_obj, bone_name, joint_dia)
            set_color(sphere, ARM_JOINT_COLOR[joint_name])
            snap["joints"][jname] = (bone_name, "head")

    return created


# ---------------------------------------------------------------------------

def convert(do_proportions=True, arm_mode="forearm"):
    arm_obj = bpy.data.objects.get(ARMATURE_OBJECT)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise RuntimeError(f"no armature object named {ARMATURE_OBJECT!r}")

    snap = snapshot_attachments(arm_obj)
    print(f"snapshot: {len(snap['joints'])} joint spheres, {len(snap['tubes'])} bone tubes")

    if do_proportions:
        targets, scale = target_lengths(arm_obj)
        changed = rescale_bones(arm_obj, targets)
        print(f"\nproportions (scale {scale:.5f} units/mm, middle-finger span preserved):")
        for name, was, now in sorted(changed):
            print(f"  {name:14s} {was:.4f} -> {now:.4f}  ({now / was:+.2f}x)")

    created = add_arm(arm_obj, arm_mode, snap)
    if created:
        print(f"\narm ({arm_mode}): added {', '.join(created)} "
              f"+ body-convention tubes and joints")

    n = rebuild_primitives(arm_obj, snap)
    print(f"\nrebuilt {n} primitives from the skeleton")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    arm_mode = "forearm"
    if "--arm" in argv:
        arm_mode = argv[argv.index("--arm") + 1]
        if arm_mode not in ("none", "forearm", "full"):
            raise SystemExit("--arm must be none, forearm or full")

    src = Path(bpy.data.filepath)
    if "--out" in argv:
        dst = Path(argv[argv.index("--out") + 1])
        if not dst.is_absolute():
            dst = src.parent / dst
    elif "--inplace" in argv:
        shutil.copy2(src, src.with_suffix(".blend.bak"))
        print(f"backed up -> {src.with_suffix('.blend.bak')}")
        dst = src
    else:
        dst = src.with_name(src.stem + "_anat.blend")

    convert(do_proportions="--no-proportions" not in argv, arm_mode=arm_mode)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
