"""
Shot description from the camera and the rig — the words the prompt is missing.

The control map says WHERE the figure is; it does not make the model want that
shot. Measured on the stock control-union model (Oct 2026, fixed seed, no LoRAs):
with a prompt that only describes the character and the place, a full-body
skeleton produced a head-and-shoulders portrait every time, with openpose, depth
or both. Raising the control strength did pull the figure onto the skeleton, but
the image was soft by 1.5 and destroyed by 2.0. Adding shot-size words to the
prompt was the one change that latched the pose at strength 1.0 with a clean
image. The prompt and the map have to agree, so this module writes the half of
the prompt that the camera already knows.

Everything here is measured from the scene at render time:

    shot size   which keypoints fall inside the frame (ankles in -> full body,
                knees in -> knees up, ... head only -> close-up)
    angle       elevation of the camera as seen from the chest
    facing      the figure's forward axis against the direction to the camera
    placement   where the chest lands across the frame

and each fact maps to ONE phrase from the fixed vocabulary below. Neutral cases
(eye level, facing the camera, centred) say nothing, so a plain view adds only
how much of the figure is in the image.

Three rules for the wording, all learned the hard way on Qwen:

  * Say what is IN THE IMAGE, never name the shot. "Medium shot" and
    "three-quarter length" are subjective even between people, and the model
    seldom lands them; "the image is from the knees up" leaves nothing to
    interpret.

  * Talk about "the image" and "the camera", not "shot", "shown" or "frame".
    "Framed" draws a literal picture frame. "Shown" appears to do much the same:
    with the subject cut down to "a man", every prompt carrying a "shown from
    ..." clause came back as a photo print on a cream mount (ten of ten), and
    none of the four without it did.

  * Say it on BOTH sides of the user's prompt. A description placed only in
    front was enough from the waist up and for camera angle, but from mid-thigh
    outwards the model still drew a portrait; what latched those was a clause
    AFTER the prompt that names the lowest part of the body in the image.

How well this works depends heavily on the rest of the prompt. With a plain
subject ("a man in a dim bar") every size, both camera angles, profile, from
behind and off-centre placement followed the map. A long character description
drags everything back toward a portrait and blocks the turned views, and one
that mentions shoes puts shoes into a knees-up image. That is the prompt's
business, not this module's, but it is the first thing to suspect.

Standalone on purpose — bpy, mathutils and bpy_extras only, no package-relative
imports — so the phrases can be exercised headlessly against a rig file.
"""

from __future__ import annotations

import math

# --- vocabulary --------------------------------------------------------------
# Before the user's prompt: how much of the figure the image holds.
EXTENT_PREFIX = {
    "extreme_close_up": "close image of only the face",
    "close_up": "image of the head and shoulders",
    "chest_up": "image from the chest up",
    "waist_up": "image from the waist up",
    "thighs_up": "image from mid-thigh up",
    "knees_up": "image from the knees up",
    "full_body": "image of the whole body from head to feet",
    "wide": "image of the whole body from head to feet, small and far from the camera",
}
# After it, for the sizes a prefix alone does not latch: where the image stops.
# Worded around what is OUTSIDE the image for the cropped sizes, and around the
# floor for the whole figure. Two earlier forms dressed the subject instead of
# cropping him: "the image ends at the knees" produced knee-length shorts, and
# "legs and shoes visible" produced bare legs.
EXTENT_SUFFIX = {
    "thighs_up": "the legs below mid-thigh are outside the image",
    "knees_up": "the legs below the knees are outside the image",
    "full_body": "the whole body is in the image down to the feet on the floor",
    "wide": "the whole body is in the image down to the feet on the floor, far from the camera",
}
# Prepended to the suffix when the legs are measured upright. A pose word, and
# so only claimed when it is true: a seated rig must not be told it is standing.
STANDING_WORD = "standing"
UPRIGHT_LEG_FRACTION = 0.85

ANGLE_PHRASES = {
    "below": "the camera is directly below, looking straight up",
    "low": "the camera is low, looking up",
    "eye_level": "",
    "high": "the camera is high, looking down",
    "overhead": "the camera is directly overhead, looking straight down",
}
FACING_PHRASES = {
    "front": "",
    "three_quarter": "the body turned at a three-quarter angle to the camera",
    "profile": "side view, the body in profile",
    "three_quarter_back": "seen from behind at a three-quarter angle",
    "back": "seen from behind, facing away from the camera",
}
PLACEMENT_PHRASES = {
    "left": "positioned on the left side of the image",
    "centre": "",
    "right": "positioned on the right side of the image",
}
# Appended to the negative prompt whenever a description is injected. Wide,
# whole-figure compositions are where this model reaches for a mounted print or
# a strip of panels; these are the things it draws when it does.
NEGATIVE_PHRASES = "border, photo print, white mount, split panels, collage, triptych"

# --- thresholds --------------------------------------------------------------
# Elevation of the camera above (+) or below (-) the chest, in degrees.
ANGLE_BELOW, ANGLE_LOW, ANGLE_HIGH, ANGLE_OVERHEAD = -55.0, -15.0, 20.0, 60.0
# Azimuth between the figure's forward axis and the camera, 0 = facing it.
FACING_FRONT, FACING_3Q, FACING_PROFILE, FACING_3Q_BACK = 25.0, 65.0, 115.0, 155.0
# Chest position across the frame, 0 = left edge.
PLACE_LEFT, PLACE_RIGHT = 0.36, 0.64
# A fully visible figure shorter than this fraction of the frame is a wide shot.
WIDE_FIGURE_FRACTION = 0.5
# With the shoulders in frame and the hips out, the shot is named by how much of
# the torso (shoulders to hips) shows above the bottom edge: at least this much
# is "waist up", at least CHEST is "chest up", and less is head and shoulders.
WAIST_TORSO_FRACTION = 0.6
CHEST_TORSO_FRACTION = 0.25

# --- keypoints ---------------------------------------------------------------
# (bone, end) per keypoint, first match wins. Auto-Rig Pro names first — the rig
# this addon ships against — then Rigify, so a different character rig degrades
# to "no description" rather than to a wrong one.
KEYPOINTS = {
    "head_top": (("head.x", "tail"), ("spine.006", "tail"), ("head", "tail")),
    "neck": (("neck.x", "head"), ("spine.004", "head"), ("neck", "head")),
    "shoulder_l": (("arm.l", "head"), ("upper_arm.L", "head")),
    "shoulder_r": (("arm.r", "head"), ("upper_arm.R", "head")),
    "hip_l": (("thigh.l", "head"), ("thigh.L", "head")),
    "hip_r": (("thigh.r", "head"), ("thigh.R", "head")),
    "knee_l": (("leg.l", "head"), ("shin.L", "head")),
    "knee_r": (("leg.r", "head"), ("shin.R", "head")),
    "ankle_l": (("foot.l", "head"), ("foot.L", "head")),
    "ankle_r": (("foot.r", "head"), ("foot.R", "head")),
}

_RESOLVED = {}   # (armature data name, bone, end) -> (bone, end) that follows the pose


def _following_bone(arm_obj, bone_name, end, tol=1e-4):
    """The bone whose POSED transform tracks this keypoint.

    On an Auto-Rig Pro rig the anatomically named bones (arm.l, thigh.l) are
    organisational: use_deform is off and the pose never moves them. The bone
    that does move is a deform partner sharing the same rest position, so it is
    found by position, as openpose_gn_prototype.resolve_deform_bone does.
    """
    key = (arm_obj.data.name, bone_name, end)
    if key in _RESOLVED:
        return _RESOLVED[key]
    bones = arm_obj.data.bones
    src = bones.get(bone_name)
    result = None
    if src is not None:
        if src.use_deform:
            result = (bone_name, end)
        else:
            at = src.head_local if end == "head" else src.tail_local
            best = None
            for b in bones:
                if not b.use_deform:
                    continue
                for e, p in (("head", b.head_local), ("tail", b.tail_local)):
                    d = (p - at).length
                    if best is None or d < best[0]:
                        best = (d, b.name, e)
            result = (best[1], best[2]) if best and best[0] <= tol else (bone_name, end)
    _RESOLVED[key] = result
    return result


def find_rig(scene, pose_layer_name):
    """The armature driving the skeleton in the rig view layer, or None."""
    vl = scene.view_layers.get(pose_layer_name) if pose_layer_name else None
    if vl is None:
        return None
    first = None
    for obj in vl.objects:
        if obj.type == "MESH":
            for mod in obj.modifiers:
                if mod.type == "ARMATURE" and mod.object is not None:
                    return mod.object
        elif obj.type == "ARMATURE" and first is None:
            first = obj
    return first


def keypoints_world(rig, depsgraph=None):
    """{keypoint: world position} for every keypoint the rig has a bone for."""
    posed = rig.evaluated_get(depsgraph) if depsgraph is not None else rig
    out = {}
    for name, candidates in KEYPOINTS.items():
        for bone_name, end in candidates:
            found = _following_bone(rig, bone_name, end)
            if found is None:
                continue
            pb = posed.pose.bones.get(found[0])
            if pb is None:
                continue
            out[name] = posed.matrix_world @ (pb.head if found[1] == "head" else pb.tail)
            break
    return out


def _mean(points):
    points = [p for p in points if p is not None]
    if not points:
        return None
    total = points[0].copy()
    for p in points[1:]:
        total += p
    return total / len(points)


def measure(scene, camera, rig, depsgraph=None):
    """The four facts for this camera and rig, or None when they cannot be read.

    Returns {"shot", "angle", "facing", "placement"} as vocabulary keys, plus the
    raw numbers they were decided from under "measured" — those go into the
    generation metadata so a surprising phrase can be traced to its cause.
    """
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    if camera is None or camera.type != "CAMERA" or rig is None:
        return None
    kp = keypoints_world(rig, depsgraph)
    neck, head_top = kp.get("neck"), kp.get("head_top")
    if neck is None or head_top is None:
        return None
    cam = camera.evaluated_get(depsgraph) if depsgraph is not None else camera

    view = {k: world_to_camera_view(scene, cam, p) for k, p in kp.items()}

    def in_view(name):
        v = view.get(name)
        return v is not None and v.z > 0.0 and 0.0 <= v.x <= 1.0 and 0.0 <= v.y <= 1.0

    def any_in(*names):
        return any(in_view(n) for n in names)

    def mean_y(*names):
        ys = [view[n].y for n in names if n in view]
        return sum(ys) / len(ys) if ys else None

    shoulders = _mean([kp.get("shoulder_l"), kp.get("shoulder_r")]) or neck
    hips = _mean([kp.get("hip_l"), kp.get("hip_r")])
    y_top, y_neck = view["head_top"].y, view["neck"].y
    y_sh = mean_y("shoulder_l", "shoulder_r")
    y_sh = y_neck if y_sh is None else y_sh
    y_hip, y_ankle = mean_y("hip_l", "hip_r"), mean_y("ankle_l", "ankle_r")

    # --- shot size: the lowest part of the body still in frame ---
    mid_head = world_to_camera_view(scene, cam, (neck + head_top) / 2.0)
    head_in = (in_view("head_top") or in_view("neck")
               or (mid_head.z > 0.0 and 0.0 <= mid_head.x <= 1.0 and 0.0 <= mid_head.y <= 1.0))
    figure_fraction = None
    torso_shown = None
    if any_in("ankle_l", "ankle_r"):
        figure_fraction = abs(y_top - y_ankle) if y_ankle is not None else None
        small = figure_fraction is not None and figure_fraction < WIDE_FIGURE_FRACTION
        shot = "wide" if small else "full_body"
    elif any_in("knee_l", "knee_r"):
        shot = "knees_up"
    elif any_in("hip_l", "hip_r"):
        shot = "thighs_up"
    elif any_in("shoulder_l", "shoulder_r") or (in_view("neck") and y_hip is not None):
        # Bottom edge (y = 0) falls somewhere down the torso: how far?
        span = (y_sh - y_hip) if y_hip is not None else 0.0
        torso_shown = max(0.0, min(1.0, y_sh / span)) if span > 1e-6 else 0.0
        if torso_shown >= WAIST_TORSO_FRACTION:
            shot = "waist_up"
        elif torso_shown >= CHEST_TORSO_FRACTION:
            shot = "chest_up"
        else:
            shot = "close_up"
    elif head_in:
        # Shoulders out of frame: nothing but the head (and neck) is left.
        shot = "extreme_close_up"
    else:
        shot = None   # the head is out of frame; no honest name for that shot

    # --- angle: where the camera sits as seen from what it is looking at ---
    # Measured from the middle of the VISIBLE body, not always the chest: a
    # level close-up sits a head's height above the chest, and judged from
    # there every close-up would read as a high angle.
    visible = [kp[k] for k in view if in_view(k)]
    subject = _mean(visible) if visible else (neck + head_top) / 2.0
    to_cam = cam.matrix_world.translation - subject
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, to_cam.normalized().z)))) \
        if to_cam.length > 1e-6 else 0.0
    if elevation <= ANGLE_BELOW:
        angle = "below"
    elif elevation <= ANGLE_LOW:
        angle = "low"
    elif elevation >= ANGLE_OVERHEAD:
        angle = "overhead"
    elif elevation >= ANGLE_HIGH:
        angle = "high"
    else:
        angle = "eye_level"

    # --- facing: body forward against the horizontal direction to the camera ---
    facing, azimuth = "front", None
    sl, sr = kp.get("shoulder_l"), kp.get("shoulder_r")
    if sl is not None and sr is not None and hips is not None:
        left = sl - sr
        up = shoulders - hips
        forward = left.cross(up)
        forward.z = 0.0
        flat = Vector((to_cam.x, to_cam.y, 0.0))
        if forward.length > 1e-6 and flat.length > 1e-6:
            azimuth = math.degrees(forward.angle(flat))
            if azimuth < FACING_FRONT:
                facing = "front"
            elif azimuth < FACING_3Q:
                facing = "three_quarter"
            elif azimuth < FACING_PROFILE:
                facing = "profile"
            elif azimuth < FACING_3Q_BACK:
                facing = "three_quarter_back"
            else:
                facing = "back"

    # --- placement: the chest across the frame (the head, on a close-up) ---
    anchor = view["neck"] if shot not in ("close_up", "extreme_close_up") else mid_head
    if anchor.x < PLACE_LEFT:
        placement = "left"
    elif anchor.x > PLACE_RIGHT:
        placement = "right"
    else:
        placement = "centre"

    # --- standing: hip-to-ankle drop against the length of the leg itself ---
    standing = None
    drops = []
    for side in ("l", "r"):
        h, k, a = kp.get(f"hip_{side}"), kp.get(f"knee_{side}"), kp.get(f"ankle_{side}")
        if h is not None and k is not None and a is not None:
            length = (h - k).length + (k - a).length
            if length > 1e-6:
                drops.append((h.z - a.z) / length)
    if drops:
        standing = (sum(drops) / len(drops)) >= UPRIGHT_LEG_FRACTION

    facts = {
        "shot": shot, "angle": angle, "facing": facing, "placement": placement,
        "standing": standing,
        "measured": {
            "elevation_deg": round(elevation, 1),
            "azimuth_deg": None if azimuth is None else round(azimuth, 1),
            "anchor_x": round(anchor.x, 3),
            "figure_fraction": None if figure_fraction is None else round(figure_fraction, 3),
            "torso_shown": None if torso_shown is None else round(torso_shown, 3),
            "in_view": sorted(k for k in view if in_view(k)),
        },
    }
    facts["prefix"], facts["suffix"] = prefix(facts), suffix(facts)
    return facts


def prefix(facts):
    """What goes BEFORE the user's prompt: how much is in the image, and the camera's angle."""
    if not facts:
        return ""
    parts = [
        EXTENT_PREFIX.get(facts.get("shot") or "", ""),
        ANGLE_PHRASES.get(facts.get("angle") or "", ""),
    ]
    return ", ".join(p for p in parts if p)


def suffix(facts):
    """What goes AFTER it: where the image stops, which way the figure faces, where it sits."""
    if not facts:
        return ""
    extent = EXTENT_SUFFIX.get(facts.get("shot") or "", "")
    if extent and facts.get("standing"):
        extent = f"{STANDING_WORD}, {extent}"
    parts = [
        extent,
        FACING_PHRASES.get(facts.get("facing") or "", ""),
        PLACEMENT_PHRASES.get(facts.get("placement") or "", ""),
    ]
    return ", ".join(p for p in parts if p)


def label(facts):
    """The injection as one readable line, for the panel and the metadata."""
    if not facts:
        return ""
    before, after = facts.get("prefix") or "", facts.get("suffix") or ""
    return f"{before} \u2026 {after}" if before and after else (before or after)


def compose(facts, prompt):
    """The prompt as sent: shot words, the user's own prompt, then the extent clause."""
    prompt = (prompt or "").strip().rstrip(",").strip()
    if not facts:
        return prompt
    parts = [facts.get("prefix") or "", prompt, facts.get("suffix") or ""]
    return ", ".join(p for p in parts if p)


def compose_negative(facts, negative):
    """The negative prompt as sent: the user's own, plus the print-and-panel terms.

    Only when something was injected — with nothing measured the prompt is the
    user's alone, and so is the negative.
    """
    negative = (negative or "").strip().rstrip(",").strip()
    if not facts or not (facts.get("prefix") or facts.get("suffix")):
        return negative
    return f"{negative}, {NEGATIVE_PHRASES}" if negative else NEGATIVE_PHRASES


def describe(scene, pose_layer_name, depsgraph=None):
    """(label, facts) for the scene's active camera and the rig-layer skeleton.

    `facts` carries the "prefix" and "suffix" strings compose() uses; `label` is
    the same injection as a single line with the prompt's place marked by an
    ellipsis.
    """
    rig = find_rig(scene, pose_layer_name)
    facts = measure(scene, scene.camera, rig, depsgraph)
    return label(facts), facts
