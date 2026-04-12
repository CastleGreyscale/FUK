# core/exr_utils.py
"""
EXR loading utilities shared across the pipeline.

cv2's EXR codec is often disabled in packaged builds (OPENCV_IO_ENABLE_OPENEXR).
This module uses the OpenEXR + Imath Python libraries instead, which are already
required for EXR export and are reliably available.
"""

from pathlib import Path
from typing import Union
import numpy as np


def load_exr_bgr(path: Union[str, Path]) -> np.ndarray:
    """
    Load an EXR file as a uint8 BGR numpy array suitable for cv2 / PIL pipelines.

    Handles:
    - RGB EXR (beauty, diffuse, etc.) → BGR uint8
    - Single-channel EXR (depth/Z pass, luminance) → 3-channel grey BGR uint8
    - RGBA EXR → alpha channel dropped, RGB retained

    Tone mapping: normalize full float range to [0, 1] then apply gamma 2.2
    (linear → sRGB), matching how ML models and browsers expect images to look.

    Raises:
        RuntimeError: if OpenEXR library is not installed.
        ValueError: if the file cannot be opened or read.
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        raise RuntimeError(
            "OpenEXR Python library is required for EXR input support. "
            "Install with: pip install OpenEXR --break-system-packages"
        )

    path = Path(path)
    try:
        exr = OpenEXR.InputFile(str(path))
    except Exception as e:
        raise ValueError(f"Could not open EXR file '{path}': {e}")

    header = exr.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header['channels'].keys())

    if 'R' in channels and 'G' in channels and 'B' in channels:
        r = np.frombuffer(exr.channel('R', pt), dtype=np.float32).reshape(h, w)
        g = np.frombuffer(exr.channel('G', pt), dtype=np.float32).reshape(h, w)
        b = np.frombuffer(exr.channel('B', pt), dtype=np.float32).reshape(h, w)
        image = np.stack([b, g, r], axis=2)  # BGR order for cv2 compatibility
    else:
        # Single-channel (depth, Z, Y, or whatever is first)
        ch = next((c for c in ('Y', 'Z') if c in channels), channels[0])
        gray = np.frombuffer(exr.channel(ch, pt), dtype=np.float32).reshape(h, w)
        image = np.stack([gray, gray, gray], axis=2)

    image = np.clip(image, 0, None)
    max_val = image.max()
    if max_val > 0:
        image = image / max_val

    # Linear → sRGB gamma so the image looks correct to models trained on sRGB data
    image = np.power(image, 1.0 / 2.2)
    return (image * 255).astype(np.uint8)
