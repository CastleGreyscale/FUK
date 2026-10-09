"""
Video outpainting helpers for FUK

Outpainting extends a clip past its own frame: the source is placed on a
larger canvas, the model generates only the padding, and the original pixels
are put back afterwards. Everything here is plain PIL/numpy and model-agnostic
— the runner decides how the canvas and mask reach its pipeline (for Wan VACE,
`vace_video` + `vace_video_mask`).

Mask convention: white = generate, black = keep.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1 — x1/y1 exclusive

# Wan's VAE downsamples 8x spatially and VACE folds the mask into 8x8 patches,
# so a source edge off this grid would leave latent cells that are part kept
# footage and part hole.
MASK_GRID = 8


def fit_rect(src_w: int, src_h: int, canvas_w: int, canvas_h: int,
             scale: float = 1.0, align_x: float = 0.5, align_y: float = 0.5,
             snap: int = MASK_GRID) -> Rect:
    """Where the source sits on the canvas.

    The source is contain-fitted (aspect preserved), shrunk by `scale`, and
    positioned by `align_*` (0 = left/top, 0.5 = centred, 1 = right/bottom).
    The rect is snapped to `snap` pixels, so the placed source can be a few
    pixels off its true aspect — invisible at these sizes, and the alternative
    is a ragged mask edge.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Outpaint source has no size: {src_w}x{src_h}")
    scale = min(1.0, max(0.05, float(scale)))
    align_x = min(1.0, max(0.0, float(align_x)))
    align_y = min(1.0, max(0.0, float(align_y)))

    fit = min(canvas_w / src_w, canvas_h / src_h) * scale

    def size(v, limit):
        return int(min(limit, max(snap, round(v / snap) * snap)))

    w, h = size(src_w * fit, canvas_w), size(src_h * fit, canvas_h)
    x0 = int(round((canvas_w - w) * align_x / snap) * snap)
    y0 = int(round((canvas_h - h) * align_y / snap) * snap)
    x0, y0 = min(x0, canvas_w - w), min(y0, canvas_h - h)

    if w >= canvas_w and h >= canvas_h:
        raise ValueError(
            f"Nothing to outpaint: the source already fills the {canvas_w}x{canvas_h} "
            f"canvas. Pick a different output aspect ratio or a source scale below 1.")
    return x0, y0, x0 + w, y0 + h


def build_canvas(frames: List[Image.Image], canvas_w: int, canvas_h: int,
                 rect: Rect) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Place each source frame on the canvas and build the matching mask.

    Returns (canvas_frames, mask_frames). The padding is mid-grey, which is
    what VACE's own `inactive` branch reduces a masked region to anyway.
    """
    x0, y0, x1, y1 = rect
    mask = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    mask.paste((0, 0, 0), rect)

    canvases = []
    for frame in frames:
        canvas = Image.new("RGB", (canvas_w, canvas_h), (127, 127, 127))
        canvas.paste(frame.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
        canvases.append(canvas)
    # One mask object repeated: the hole does not move, and the pipeline only reads it.
    return canvases, [mask] * len(canvases)


def keep_alpha(canvas_w: int, canvas_h: int, rect: Rect, feather: int) -> Image.Image:
    """Alpha for pasting the source back: 255 = original pixel, 0 = generated.

    Ramps over `feather` pixels *inside* the source rect, so the blend eats a
    sliver of original footage rather than exposing generated pixels as if
    they were source. Edges that sit on the canvas border are not feathered —
    there is nothing generated on the other side to blend into.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    alpha = np.ones((h, w), dtype=np.float32)
    feather = int(max(0, min(feather, w // 2, h // 2)))
    if feather > 0:
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)[:, None]
        if x0 > 0:
            alpha = np.minimum(alpha, (xs + 1) / feather)
        if x1 < canvas_w:
            alpha = np.minimum(alpha, (w - xs) / feather)
        if y0 > 0:
            alpha = np.minimum(alpha, (ys + 1) / feather)
        if y1 < canvas_h:
            alpha = np.minimum(alpha, (h - ys) / feather)
    full = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    full[y0:y1, x0:x1] = np.clip(alpha, 0.0, 1.0)
    return Image.fromarray((full * 255).astype(np.uint8), "L")


def composite_source(generated: List[Image.Image], canvases: List[Image.Image],
                     rect: Rect, feather: int = 16) -> List[Image.Image]:
    """Put the original footage back over the model's output.

    VACE regenerates the whole frame — the kept region comes back as a VAE
    round trip of the source, close but not identical. Pasting the source over
    it is what makes the original area pixel-faithful.
    """
    if not generated:
        return generated
    canvas_w, canvas_h = generated[0].size
    alpha = keep_alpha(canvas_w, canvas_h, rect, feather)
    out = []
    for i, frame in enumerate(generated):
        if i >= len(canvases) or canvases[i].size != frame.size:
            out.append(frame)
            continue
        out.append(Image.composite(canvases[i], frame.convert("RGB"), alpha))
    return out
