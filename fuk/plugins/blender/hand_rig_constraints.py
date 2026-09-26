#!/usr/bin/env python3
"""
Make the 21-keypoint hand rig posable: give it a wrist, and stop its hinge
joints bending backwards or twisting.

Companion to openpose_hand21.py, which handles the rig's colours. This one
touches only the armature — no object colours, no materials, no render
settings — so the two can be run in either order.

Usage:
    blender Hand_OP21.blend --background --factory-startup \
        --python hand_rig_constraints.py -- --out Hand_OP21_rigged.blend

    blender Hand_OP21.blend --background --python hand_rig_constraints.py -- --inplace

    --preset NAME  hinge (default) or anatomical — see the preset tables below
    --curl         add one master curl control per digit (Rigify-style)
    --no-wrist     skip the hierarchy fix, add limits only
    --no-limits    skip the limits, add the wrist only

Two presets, because they are for different jobs:

    hinge       Safety rails. Only the four single-DOF joints are limited, so
                nothing can bend backwards or twist, and everything else stays
                completely free. Almost never in your way; almost no help
                either.

    anatomical  Every joint bounded to roughly what a hand can do. Much faster
                to reach a plausible pose, and much harder to reach an
                implausible one. Dial a bone's constraint Influence down when a
                shot genuinely needs to break anatomy.


Two things it fixes
-------------------

1. The rig ships with FIVE root bones. Hand.L parents the thumb chain, but the
   four finger chains (FingerN_0) are parented to nothing, so the bone you'd
   reach for as the wrist rotates only the thumb and there is no bone that moves
   the whole hand. A new Wrist.L is inserted as the shared parent. Hand.L keeps
   driving the thumb exactly as before.

2. The single-DOF joints are unconstrained, so fingers hyperextend backwards and
   twist along their own length. A Limit Rotation per hinge pins the two
   off-axis rotations at zero and allows flexion in one direction only.


Working out the axes
--------------------

The rig is a flat diagram — every bone is coplanar in XY — so which way is
"toward the palm" cannot be read off the rest pose, and a curl-into-a-fist test
is exactly symmetric. Both halves were established separately:

* WHICH axis, by probing the rest matrices: the 16 finger bones all have roll 0
  with local Z = world +Z, so their flexion axis (the one that leaves the flat
  plane) is local X and local Z is in-plane spread. The 4 thumb-chain bones have
  roll -90 degrees, which swaps those two: thumb flexion is about local Z.

* WHICH sign, from handedness. Easiest anchor is a keyboard: both hands rest
  palm DOWN with fingers forward, and the thumbs point toward each other, so a
  left hand in that pose has its thumb to the RIGHT. Mapping that onto the rig —
  fingers +Y, thumb at +X (x = +0.120, ahead of index at +0.096, out to the
  little finger at -0.082) — puts the palm normal at -Z.

  So the flat rig, seen from +Z in Blender's top view, is the BACK of a left
  hand. Flexion curls toward the palm, which is -Z, so for the fingers that is
  NEGATIVE local X (+45 degrees moved the tail to z = +0.099, i.e. away from the
  palm) and for the thumb POSITIVE local Z (+50 degrees moved it to -0.167,
  toward the palm).

Only the flexion axes carry a sign. Spread, thumb fan and twist are either
symmetric or purely in-plane, so they are unaffected by which side the palm is
on. If a right-hand variant is ever mirrored off this file, the two flexion
signs flip back.
"""

import re
import sys
import math
import shutil
from pathlib import Path

import bpy


ARMATURE_OBJECT = "HandArmature.L"
WRIST_BONE = "Wrist.L"

# Length of the wrist stub, as a fraction of the hand's overall span. Purely
# cosmetic — it is a handle to grab, and carries no rendered geometry.
WRIST_FRACTION = 0.28

# --- presets ----------------------------------------------------------------
#
# Ranges are degrees in the bone's own rest frame, per axis, and the axis
# meanings differ between the two roll groups (see the module docstring):
#
#   finger bones (roll 0)      X = flexion (- palmward)   Y = twist   Z = spread
#   thumb chain  (roll -90)    Z = flexion (+ palmward)   Y = twist   X = fan
#
# The palm faces -Z (see the module docstring), so flexion runs negative on the
# fingers and positive on the thumb.
#
# A (0, 0) range pins that axis. First matching pattern wins, so specific
# patterns must precede general ones.

# "hinge": safety only. Locks the four single-DOF joints so they cannot bend
# backwards or twist, and leaves everything else completely free.
HINGE_PRESET = [
    (r"^Finger\d_2$", dict(X=(-110, 0), Y=(0, 0), Z=(0, 0))),   # PIP
    (r"^Finger\d_3$", dict(X=(-80, 0),  Y=(0, 0), Z=(0, 0))),   # DIP
    (r"^Thumb2$",     dict(X=(0, 0),    Y=(0, 0), Z=(0, 60))),  # thumb MCP
    (r"^Thumb3$",     dict(X=(0, 0),    Y=(0, 0), Z=(0, 80))),  # thumb IP
]

# "anatomical": every joint bounded to roughly what a hand can actually do.
#
# The metacarpals matter most here. FingerN_0 rotates about the WRIST, so while
# it is unconstrained each finger's root can swing anywhere and twist along its
# own length — which is what makes the hinge-only rig feel loose to pose. In a
# real hand the index and middle metacarpals are near-rigid and only the ring
# and (especially) the little finger cup across the palm, so they are pinned
# tight and graded outward.
#
# Axial twist is zero at every joint except the thumb's saddle, which really
# does rotate as it opposes. MCPs get a little hyperextension because hands
# genuinely do that when splayed flat against a surface.
ANATOMICAL_PRESET = [
    # metacarpals — cupping across the palm, graded index -> little
    (r"^Finger0_0$", dict(X=(-5, 0),     Y=(0, 0),     Z=(-3, 3))),    # index
    (r"^Finger1_0$", dict(X=(-5, 0),     Y=(0, 0),     Z=(-3, 3))),    # middle
    (r"^Finger2_0$", dict(X=(-10, 0),    Y=(0, 0),     Z=(-4, 4))),    # ring
    (r"^Finger3_0$", dict(X=(-20, 0),    Y=(0, 0),     Z=(-6, 6))),    # little
    # MCP — flexion plus spread; index and little spread furthest.
    # The positive end is hyperextension (hands do that splayed on a surface).
    (r"^Finger0_1$", dict(X=(-90, 25),   Y=(0, 0),     Z=(-25, 25))),
    (r"^Finger1_1$", dict(X=(-90, 25),   Y=(0, 0),     Z=(-20, 20))),
    (r"^Finger2_1$", dict(X=(-90, 25),   Y=(0, 0),     Z=(-20, 20))),
    (r"^Finger3_1$", dict(X=(-90, 25),   Y=(0, 0),     Z=(-30, 30))),
    # PIP / DIP — pure hinges, same as the safety preset
    (r"^Finger\d_2$", dict(X=(-110, 0),  Y=(0, 0),     Z=(0, 0))),
    (r"^Finger\d_3$", dict(X=(-80, 0),   Y=(0, 0),     Z=(0, 0))),
    # thumb: Hand.L sits inside the palm and barely moves; Thumb1 is the saddle.
    # X is the in-plane fan — abduction (away from the index) is the wide side,
    # and being in-plane it is unaffected by which face the palm is on.
    (r"^Hand$",       dict(X=(-10, 10),  Y=(0, 0),     Z=(-10, 10))),
    (r"^Thumb1$",     dict(X=(-45, 15),  Y=(-15, 15),  Z=(-15, 40))),
    (r"^Thumb2$",     dict(X=(0, 0),     Y=(0, 0),     Z=(0, 60))),
    (r"^Thumb3$",     dict(X=(0, 0),     Y=(0, 0),     Z=(-5, 80))),
]

PRESETS = {"hinge": HINGE_PRESET, "anatomical": ANATOMICAL_PRESET}
DEFAULT_PRESET = "hinge"

# Wrist.L is never constrained by either preset: with no forearm in the rig it
# is the handle you place the whole hand with, so anatomical wrist limits would
# only get in the way.

SIDE_RE = re.compile(r"\.[LR]$")
CONSTRAINT_PREFIX = "FUK "
CONSTRAINT_NAME = "FUK Joint Limit"
# Euler order puts the joint's widest axis first, so the N-panel's top field is
# the one you actually reach for. With the others pinned or narrow there is no
# gimbal risk either way.
EULER_ORDER = {"X": "XYZ", "Y": "YXZ", "Z": "ZYX"}


def limits_for(bone_name, preset):
    """{axis: (lo, hi)} for a bone, or None if this preset leaves it free.

    The side suffix is ignored, so a mirrored .R rig matches the same table.
    """
    base = SIDE_RE.sub("", bone_name)
    for pattern, spans in preset:
        if re.match(pattern, base):
            return spans
    return None


def dominant_axis(spans):
    """The axis with the widest travel — the one this joint mostly moves on."""
    return max("XYZ", key=lambda a: spans[a][1] - spans[a][0])


def add_wrist(arm_obj):
    """Insert a shared parent for the thumb chain and the four finger chains.

    Returns (created, [names re-parented]).
    """
    arm = arm_obj.data
    roots = [b.name for b in arm.bones if b.parent is None]
    if WRIST_BONE in roots and len(roots) == 1:
        return False, []

    span = max((b.tail_local.y for b in arm.bones), default=1.0)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        ebs = arm.edit_bones
        wrist = ebs.get(WRIST_BONE)
        if wrist is None:
            wrist = ebs.new(WRIST_BONE)
        # Points along +Y into the wrist, so its tail sits exactly where every
        # chain begins — it reads as a forearm stub rather than a stray bone.
        wrist.head = (0.0, -span * WRIST_FRACTION, 0.0)
        wrist.tail = (0.0, 0.0, 0.0)
        wrist.roll = 0.0
        wrist.parent = None

        reparented = []
        for eb in ebs:
            if eb.name == WRIST_BONE or eb.parent is not None:
                continue
            eb.parent = wrist
            # Left unconnected on purpose: connecting would lock each chain's
            # head to the wrist tail and hide its location channel. Rotation is
            # inherited either way, which is all the fix needs.
            eb.use_connect = False
            reparented.append(eb.name)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    return True, sorted(reparented)


def add_limits(arm_obj, preset):
    """One Limit Rotation per constrained bone. Returns (applied, warnings)."""
    applied, warnings = [], []

    for pb in arm_obj.pose.bones:
        spans = limits_for(pb.name, preset)
        if spans is None:
            continue

        # A pose already saved outside the new limits would be silently snapped
        # by the constraint. Say so rather than quietly changing the file.
        rot = pb.rotation_quaternion
        if pb.rotation_mode == "QUATERNION" and abs(rot.w) < 0.99999:
            warnings.append(f"{pb.name} is posed away from rest; the limit may move it")

        pb.rotation_mode = EULER_ORDER[dominant_axis(spans)]

        # Match on the prefix, not the exact name: re-running with a different
        # preset (or an older build of this script, which called it "FUK Hinge
        # Limit") must replace what is there rather than stack a second limit.
        for existing in [c for c in pb.constraints if c.name.startswith(CONSTRAINT_PREFIX)]:
            pb.constraints.remove(existing)

        c = pb.constraints.new("LIMIT_ROTATION")
        c.name = CONSTRAINT_NAME
        c.owner_space = "LOCAL"
        for a in ("X", "Y", "Z"):
            lo, hi = spans[a]
            setattr(c, f"use_limit_{a.lower()}", True)
            setattr(c, f"min_{a.lower()}", math.radians(lo))
            setattr(c, f"max_{a.lower()}", math.radians(hi))

        shown = "  ".join(
            f"{a}:{'pinned' if spans[a] == (0, 0) else f'{spans[a][0]:+d}..{spans[a][1]:+d}'}"
            for a in ("X", "Y", "Z"))
        applied.append(f"  {pb.name:14s} {shown}")

    return applied, warnings


# --- curl controls ----------------------------------------------------------
#
# One handle per finger that closes the whole chain, the ergonomic half of what
# Rigify's limbs.super_finger gives you. Rigify drives its chain off a master
# control's Y-scale through drivers; this does the same job with COPY_ROTATION
# at a fixed influence per joint, which is far less machinery and survives being
# appended into another file without a generate step.
#
# Weights are how much of the master's rotation each joint takes. A real finger
# does not close uniformly — the PIP leads, the MCP trails — so the middle joint
# is weighted heaviest. The per-joint Limit Rotation still clamps the result, so
# a full-throw master cannot push any joint past its anatomical range.
CURL_WEIGHTS = {"_1": 0.75, "_2": 1.0, "_3": 0.8}
CURL_CONSTRAINT = "FUK Curl"
CURL_SUFFIX = "_curl"

# Thumb chain, keyed on the same idea. Hand.L is palm-internal so it is not
# driven; the master turns the three thumb segments.
THUMB_CURL_WEIGHTS = {"Thumb1": 0.5, "Thumb2": 1.0, "Thumb3": 0.8}


def add_curl_controls(arm_obj):
    """A master bone per digit whose rotation closes that digit.

    Returns a list of (control, [driven bones]).
    """
    arm = arm_obj.data
    chains = [(f"Finger{i}", [f"Finger{i}{s}.L" for s in CURL_WEIGHTS],
               list(CURL_WEIGHTS.values())) for i in range(4)]
    chains.append(("Thumb", [f"{n}.L" for n in THUMB_CURL_WEIGHTS],
                   list(THUMB_CURL_WEIGHTS.values())))

    # --- control bones, offset clear of the rig so they are grabbable ---
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        ebs = arm_obj.data.edit_bones
        wrist = ebs.get(WRIST_BONE)
        for label, bones, _ in chains:
            first = ebs.get(bones[0])
            if first is None:
                continue
            name = label + CURL_SUFFIX
            ctl = ebs.get(name) or ebs.new(name)
            # Sits at the digit's base, pointing the way the digit points, so
            # its local axes match the bones it drives.
            direction = (first.tail - first.head).normalized()
            ctl.head = first.head
            ctl.tail = first.head + direction * first.length * 0.6
            ctl.roll = first.roll
            ctl.parent = wrist if wrist else None
            ctl.use_connect = False
            ctl.use_deform = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    # --- drive each joint off its master ---
    created = []
    for label, bones, weights in chains:
        ctl_name = label + CURL_SUFFIX
        if ctl_name not in arm_obj.pose.bones:
            continue
        driven = []
        for bone_name, weight in zip(bones, weights):
            pb = arm_obj.pose.bones.get(bone_name)
            if pb is None:
                continue
            for old in [c for c in pb.constraints if c.name == CURL_CONSTRAINT]:
                pb.constraints.remove(old)
            c = pb.constraints.new("COPY_ROTATION")
            c.name = CURL_CONSTRAINT
            c.target = arm_obj
            c.subtarget = ctl_name
            c.target_space = "LOCAL"
            c.owner_space = "LOCAL"
            c.mix_mode = "ADD"          # adds to whatever you posed by hand
            c.influence = weight
            # Order matters: curl first, then clamp. A Limit Rotation sitting
            # before the Copy Rotation would be evaluated on the pre-curl value
            # and clamp nothing.
            idx = next((i for i, x in enumerate(pb.constraints)
                        if x.name.startswith(CONSTRAINT_PREFIX) and x != c), None)
            if idx is not None:
                pb.constraints.move(len(pb.constraints) - 1, idx)
            driven.append(f"{bone_name}@{weight}")
        created.append((ctl_name, driven))
    return created


def convert(do_wrist=True, do_limits=True, preset_name=DEFAULT_PRESET,
            do_curl=False):
    arm_obj = bpy.data.objects.get(ARMATURE_OBJECT)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise RuntimeError(f"no armature object named {ARMATURE_OBJECT!r} in this file")

    if do_wrist:
        created, reparented = add_wrist(arm_obj)
        if created:
            print(f"wrist: added {WRIST_BONE}, re-parented {len(reparented)} root "
                  f"chain(s): {', '.join(reparented)}")
        else:
            print(f"wrist: {WRIST_BONE} already present and sole root — unchanged")

    if do_limits:
        preset = PRESETS[preset_name]
        applied, warnings = add_limits(arm_obj, preset)
        print(f"\npreset {preset_name!r} — limits on {len(applied)} bone(s):")
        print("\n".join(applied) if applied else "  none matched")
        if warnings:
            print("\n  WARNING:")
            for w in warnings:
                print(f"    {w}")
        free = [pb.name for pb in arm_obj.pose.bones
                if limits_for(pb.name, preset) is None]
        print(f"\nleft free ({len(free)}): {', '.join(sorted(free))}")

    if do_curl:
        made = add_curl_controls(arm_obj)
        print(f"\ncurl controls ({len(made)}):")
        for ctl, driven in made:
            print(f"  {ctl:16s} -> {', '.join(driven)}")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    inplace = "--inplace" in argv

    preset_name = DEFAULT_PRESET
    if "--preset" in argv:
        preset_name = argv[argv.index("--preset") + 1]
        if preset_name not in PRESETS:
            raise SystemExit(f"--preset must be one of {sorted(PRESETS)}, got {preset_name!r}")

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
        suffix = "_rigged" if preset_name == "hinge" else f"_{preset_name}"
        dst = src.with_name(src.stem + suffix + ".blend")

    convert(do_wrist="--no-wrist" not in argv, do_limits="--no-limits" not in argv,
            preset_name=preset_name, do_curl="--curl" in argv)
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
