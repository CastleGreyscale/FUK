"""
Motion-adaptive blending between a restored video and its source.

Why this exists
    SeedVR2 is a restoration model, so it removes motion blur — measured on a
    Wan clip, the fast-moving region of frame came back roughly 10x sharper
    (variance of Laplacian 19-28 in the source, 185-266 restored).

    That is the model working correctly, and it is also the problem. Motion blur
    is what makes 24fps read as smooth motion. Strip it and each frame becomes a
    crisp snapshot at a distinct position, so the same displacement now reads as
    stepping; and because the restored detail is hallucinated independently per
    frame, its shape shifts slightly frame to frame, which reads as ghosting.

    Blending globally back toward the source would fix the motion but throw away
    the sharpening everywhere else — which is the entire point of running the
    upscaler. So the blend is weighted per pixel by how much that pixel is
    actually moving: still areas keep all of the restoration, moving areas keep
    their original blur.

The mask is built from the *source*, not the restored output, because the source
is what still carries honest motion blur. Deriving it from the restored frames
would measure the model's hallucination as if it were movement.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np


def build_motion_mask(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray,
    next_gray: Optional[np.ndarray],
    threshold: float,
    dilate_px: int,
    blur_px: int,
) -> np.ndarray:
    """
    Per-pixel 0..1 map of how much this frame is moving.

    Uses the larger of the backward and forward difference so an object is
    covered as it arrives and as it leaves, not just on one side. A plain
    backward difference leaves the trailing edge unprotected, which shows up as
    a hard sharpness change across a moving object.

    The raw difference only marks *edges* of moving objects, so it is dilated to
    cover the body and then blurred: a hard-edged mask produces a visible seam
    between the sharpened and unsharpened regions, which trades one artifact for
    another.
    """
    diffs = []
    if prev_gray is not None:
        diffs.append(cv2.absdiff(curr_gray, prev_gray))
    if next_gray is not None:
        diffs.append(cv2.absdiff(curr_gray, next_gray))
    if not diffs:
        return np.zeros_like(curr_gray, dtype=np.float32)

    motion = diffs[0] if len(diffs) == 1 else np.maximum(diffs[0], diffs[1])

    mask = np.clip(motion.astype(np.float32) / max(threshold, 1e-6), 0.0, 1.0)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k)
    if blur_px > 0:
        sigma = blur_px / 2.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)

    return np.clip(mask, 0.0, 1.0)


def apply_motion_protection(
    source_path: Path,
    restored_path: Path,
    output_path: Path,
    strength: float = 0.7,
    threshold: float = 12.0,
    dilate_px: int = 6,
    blur_px: int = 24,
    progress_callback: Optional[Callable] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Composite `restored` over `source`, reverting moving pixels toward the source.

    strength
        Ceiling on how much of the source is restored in the fastest-moving
        pixels. 0 disables the pass entirely; 1.0 hands the most-moving pixels
        back to the source completely, which keeps all of their motion blur and
        none of the sharpening.
    threshold
        8-bit difference that counts as "fully moving". Lower protects more.

    Returns a summary including the mean mask coverage, which is a useful sanity
    check: a near-zero mean means the clip barely moves and this pass did almost
    nothing, which is the correct behaviour rather than a failure.
    """
    _log = log or (lambda _msg: None)

    source_path, restored_path = Path(source_path), Path(restored_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = cv2.VideoCapture(str(source_path))
    res = cv2.VideoCapture(str(restored_path))
    if not src.isOpened() or not res.isOpened():
        raise RuntimeError("Could not open source or restored video")

    fps = res.get(cv2.CAP_PROP_FPS) or src.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(res.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(res.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(res.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Read the source once. These are the pre-upscale frames, so even a long
    # clip is small next to the restored ones, and we need random-ish access to
    # build a three-frame window for the mask.
    src_frames = []
    while True:
        ok, frame = src.read()
        if not ok:
            break
        src_frames.append(frame)
    src.release()

    if not src_frames:
        res.release()
        raise RuntimeError(f"No frames read from source {source_path}")

    # Grayscale of the *upscaled* source drives the mask, so mask and restored
    # frames share a coordinate space.
    grays = [
        cv2.cvtColor(cv2.resize(f, (width, height), interpolation=cv2.INTER_LANCZOS4),
                     cv2.COLOR_BGR2GRAY)
        for f in src_frames
    ]

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", f"{fps}",
         "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", str(output_path)],
        stdin=subprocess.PIPE,
    )

    mask_sum, written = 0.0, 0
    try:
        idx = 0
        while True:
            ok, restored = res.read()
            if not ok:
                break

            if idx >= len(src_frames):
                # Restored is longer than the source: nothing to blend against,
                # so pass it through rather than dropping frames.
                ffmpeg.stdin.write(np.ascontiguousarray(restored).tobytes())
                written += 1
                idx += 1
                continue

            source_up = cv2.resize(src_frames[idx], (width, height),
                                   interpolation=cv2.INTER_LANCZOS4)
            mask = build_motion_mask(
                grays[idx - 1] if idx > 0 else None,
                grays[idx],
                grays[idx + 1] if idx + 1 < len(grays) else None,
                threshold, dilate_px, blur_px,
            )
            alpha = (mask * strength)[..., None]
            mask_sum += float(mask.mean())

            blended = restored.astype(np.float32) * (1.0 - alpha) \
                + source_up.astype(np.float32) * alpha
            ffmpeg.stdin.write(
                np.ascontiguousarray(np.clip(blended, 0, 255).astype(np.uint8)).tobytes()
            )
            written += 1
            idx += 1

            if progress_callback and total:
                progress_callback(written / total, "Motion protection")
    finally:
        res.release()
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass
        ffmpeg.wait()

    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {output_path} (code {ffmpeg.returncode})")

    coverage = mask_sum / max(written, 1)
    _log(f"Motion protection: strength {strength:.2f}, "
         f"mean mask coverage {coverage:.3f} over {written} frames")

    return {
        "frames": written,
        "strength": strength,
        "mask_coverage": coverage,
        "output_path": str(output_path),
    }
