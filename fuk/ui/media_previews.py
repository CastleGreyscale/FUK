"""
Cached, downscaled image previews for the UI.

History thumbnails, hover popups and result views used to load the full
output: a 4x upscale is a 13 MB, 5120x2112 PNG drawn into a 120 px tile, and
EXRs were re-converted to PNG on every request. These helpers write a WebP
copy once, keyed by source path + mtime + size, and reuse it until the source
changes. Callers run them in a worker thread; they block on disk and decode.

No module-level state: the cache directory is passed in by the caller.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Long-edge pixel caps. `thumb` covers the 320 px gallery tiles and hover
# popup; `preview` is a full-screen view, swapped for the original on zoom.
PREVIEW_SIZES = {"thumb": 384, "preview": 1920}

# Formats Pillow (or the EXR path) can downscale. Anything else is served as-is.
PREVIEWABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"}

PREVIEW_CACHE_DIRNAME = ".previews"


def source_etag(src: Path, size: str) -> str:
    """ETag for a preview: changes whenever the source file does."""
    st = src.stat()
    return f'"{st.st_mtime_ns:x}-{st.st_size:x}-{size}"'


def _cache_path(src: Path, size: str, cache_dir: Path, suffix: str) -> Path:
    st = src.stat()
    key = f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{size}"
    return cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}{suffix}"


def _load_exr_rgb(src: Path):
    """Tone-mapped sRGB PIL image via the shared EXR colour path."""
    import cv2
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.exr_utils import load_exr_bgr

    return Image.fromarray(cv2.cvtColor(load_exr_bgr(src), cv2.COLOR_BGR2RGB))


def _write_atomic(path: Path, save) -> None:
    """Write via a temp file + rename so concurrent requests never read half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        save(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def get_preview(src: Path, size: str, cache_dir: Path) -> Optional[Path]:
    """
    Path to a browser-displayable copy of `src`, downscaled to `size`.

    Returns None when the original should be served unchanged: a non-EXR image
    that is already within the size cap. EXRs always get a cached copy, since
    browsers can't display them at all.
    """
    from PIL import Image, ImageOps

    is_exr = src.suffix.lower() == ".exr"
    max_edge = PREVIEW_SIZES[size]
    out = _cache_path(src, size, cache_dir, ".webp")
    if out.exists():
        return out

    if is_exr:
        img = _load_exr_rgb(src)
    else:
        img = Image.open(src)
        if max(img.size) <= max_edge:
            img.close()
            return None
        img.draft("RGB", (max_edge, max_edge))  # JPEG: decode at reduced scale
        img = ImageOps.exif_transpose(img)

    if img.mode.startswith("I;16") or img.mode == "I":
        # 16-bit greyscale (depth passes): scale to 8-bit rather than clip.
        img = img.convert("I").point(lambda v: v * (1 / 256)).convert("RGB")
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB")
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    _write_atomic(out, lambda p: img.save(p, "WEBP", quality=85, method=4))
    return out


def get_exr_png(src: Path, cache_dir: Path) -> Path:
    """Full-resolution tone-mapped PNG of an EXR, converted once and cached."""
    out = _cache_path(src, "full", cache_dir, ".png")
    if not out.exists():
        img = _load_exr_rgb(src)
        _write_atomic(out, lambda p: img.save(p, "PNG", compress_level=1))
    return out
