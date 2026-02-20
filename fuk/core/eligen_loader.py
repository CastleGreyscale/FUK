"""
EliGen Mask Loader for FUK

Loads entity masks + prompts from external sources for EliGen
compositional control. Three modes:

  Directory:  Folder of PNGs — filename becomes entity prompt.
              Numeric prefix controls order (stripped from prompt).
              
              masks/
                01_beautiful woman.png
                02_mirror.png
                03_white dress.png

  PSD file:   Photoshop file — each visible layer is a mask,
              layer name becomes entity prompt. (requires psd-tools)

  ORA file:   OpenRaster file — same as PSD but open standard.
              No dependencies. Supported natively by GIMP, Krita,
              and MyPaint. (File → Export As → .ora)

All modes produce binary masks (white-on-black) at the target
resolution, paired with their prompt strings.

Usage:
    loader = EliGenLoader()
    entities = loader.load("/path/to/masks/")      # directory
    entities = loader.load("/path/to/comp.psd")     # PSD file
    entities = loader.load("/path/to/comp.ora")     # OpenRaster
    # entities = [("beautiful woman", PIL.Image), ("mirror", PIL.Image), ...]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


def _log(category: str, message: str, level: str = "info"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    colors = {
        'info': '\033[96m', 'success': '\033[92m',
        'warning': '\033[93m', 'error': '\033[91m', 'end': '\033[0m',
    }
    symbols = {'info': '', 'success': '✔ ', 'warning': '⚠ ', 'error': '✗ '}
    color = colors.get(level, colors['info'])
    symbol = symbols.get(level, '')
    print(f"{color}[{timestamp}] {symbol}[{category}] {message}{colors['end']}", flush=True)


# Regex: optional numeric prefix + separator, then the actual name
# Matches: "01_beautiful woman", "02-mirror", "3 white dress", "necklace"
_PREFIX_RE = re.compile(r"^(\d+)[_\-\s]+(.+)$")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class EliGenLoader:
    """
    Load EliGen entity masks from a directory or PSD file.
    
    Each entity is a (prompt, mask) pair where:
      - prompt: text description of the entity
      - mask: binary PIL Image (white = entity region, black = background)
    """

    def load(
        self,
        source: str | Path,
        width: int = 1024,
        height: int = 1024,
    ) -> List[Tuple[str, Image.Image]]:
        """
        Load entities from a directory, PSD, or OpenRaster (.ora) file.
        
        Supported formats:
          - Directory of PNG/image files (filename = prompt)
          - PSD (Photoshop) — layer names = prompts (requires psd-tools)
          - ORA (OpenRaster) — layer names = prompts (no dependencies)
            Supported natively by GIMP, Krita, and MyPaint.
        
        Args:
            source: Path to a directory of mask images, .psd, or .ora file
            width: Target mask width (masks are resized to match generation)
            height: Target mask height
            
        Returns:
            List of (prompt_string, binary_mask_pil) tuples, ordered by
            layer order (top-to-bottom = first-to-last).
        """
        source = Path(source)

        if source.is_dir():
            return self._load_directory(source, width, height)
        
        ext = source.suffix.lower()
        if ext == ".psd":
            return self._load_psd(source, width, height)
        elif ext == ".ora":
            return self._load_ora(source, width, height)
        else:
            raise ValueError(
                f"EliGen source must be a directory, .psd, or .ora file, got: {source}\n"
                f"Tip: Krita and GIMP can both export .ora (File → Export As)"
            )

    # ------------------------------------------------------------------
    # Directory mode
    # ------------------------------------------------------------------

    def _load_directory(
        self, dir_path: Path, width: int, height: int
    ) -> List[Tuple[str, Image.Image]]:
        """
        Load masks from a directory of image files.
        
        Filename convention:
          - Optional numeric prefix for ordering: 01_name.png, 02_name.png
          - Stem (minus prefix) becomes the entity prompt
          - Underscores and hyphens in the name are kept as-is
            (artists can use natural spacing in filenames)
          - Files without numeric prefix sort alphabetically after prefixed ones
        """
        entries = []
        for f in sorted(dir_path.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            if f.name.startswith((".", "_")):
                continue  # skip hidden/meta files

            stem = f.stem
            match = _PREFIX_RE.match(stem)
            if match:
                sort_key = int(match.group(1))
                prompt = match.group(2).strip()
            else:
                sort_key = 9999  # unprefixed files sort last
                prompt = stem.strip()

            entries.append((sort_key, prompt, f))

        # Sort by numeric prefix
        entries.sort(key=lambda x: x[0])

        results = []
        for _, prompt, filepath in entries:
            mask = self._to_binary_mask(Image.open(filepath), width, height)
            results.append((prompt, mask))
            _log("ELIGEN", f"  Mask: {filepath.name} → \"{prompt}\"")

        _log("ELIGEN", f"Loaded {len(results)} entities from {dir_path}", "success")
        return results

    # ------------------------------------------------------------------
    # PSD mode
    # ------------------------------------------------------------------

    def _load_psd(
        self, psd_path: Path, width: int, height: int
    ) -> List[Tuple[str, Image.Image]]:
        """
        Load masks from a Photoshop PSD file.
        
        Each visible layer becomes an entity:
          - Layer name = entity prompt
          - Layer content = mask (thresholded to binary)
          - Layer order preserved (top-to-bottom in Photoshop = 
            first-to-last in the entity list)
        
        Skips:
          - Hidden layers (artist can toggle visibility to include/exclude)
          - Layers named "Background" or starting with "_" (utility layers)
          - Group layers (only leaf/pixel layers)
        """
        try:
            from psd_tools import PSDImage
        except ImportError:
            raise ImportError(
                "PSD support requires psd-tools: pip install psd-tools"
            )

        psd = PSDImage.open(psd_path)
        results = []

        for layer in psd:
            # Skip hidden, groups, and utility layers
            if not layer.is_visible():
                continue
            if layer.is_group():
                continue
            if layer.name.lower() in ("background", "bg"):
                continue
            if layer.name.startswith("_"):
                continue

            prompt = layer.name.strip()
            if not prompt:
                continue

            # Composite the layer to a full-canvas image
            layer_image = layer.composite()
            mask = self._to_binary_mask(layer_image, width, height)
            results.append((prompt, mask))
            _log("ELIGEN", f"  Layer: \"{layer.name}\" → mask")

        _log("ELIGEN", f"Loaded {len(results)} entities from {psd_path.name}", "success")
        return results

    # ------------------------------------------------------------------
    # ORA mode (OpenRaster — GIMP, Krita, MyPaint)
    # ------------------------------------------------------------------

    def _load_ora(
        self, ora_path: Path, width: int, height: int
    ) -> List[Tuple[str, Image.Image]]:
        """
        Load masks from an OpenRaster (.ora) file.
        
        ORA is a ZIP containing PNGs + an XML manifest. Supported natively
        by GIMP (File → Export As → .ora), Krita, and MyPaint.
        No additional Python packages required.
        
        Same conventions as PSD mode:
          - Layer name = entity prompt
          - Hidden layers are skipped
          - "Background" / "_" prefix layers are skipped
          - Layer order preserved (top-to-bottom in the app)
        """
        import zipfile
        import xml.etree.ElementTree as ET
        from io import BytesIO

        if not zipfile.is_zipfile(ora_path):
            raise ValueError(f"Not a valid ORA file: {ora_path}")

        results = []

        with zipfile.ZipFile(ora_path, "r") as zf:
            # Parse the layer stack manifest
            stack_xml = zf.read("stack.xml")
            root = ET.fromstring(stack_xml)

            # Canvas dimensions from the ORA file
            canvas_w = int(root.get("w", width))
            canvas_h = int(root.get("h", height))

            # Walk the layer stack — ORA nests layers inside <stack> elements
            for layer_elem in root.iter("layer"):
                name = layer_elem.get("name", "").strip()
                src = layer_elem.get("src", "")
                visibility = layer_elem.get("visibility", "visible")

                # Skip hidden layers
                if visibility != "visible":
                    continue
                # Skip utility layers
                if not name or name.lower() in ("background", "bg"):
                    continue
                if name.startswith("_"):
                    continue
                # Skip if no image data
                if not src:
                    continue

                # Load the layer PNG from the ZIP
                try:
                    layer_data = zf.read(src)
                except KeyError:
                    _log("ELIGEN", f"  Layer \"{name}\": missing {src}, skipping", "warning")
                    continue

                layer_img = Image.open(BytesIO(layer_data)).convert("RGBA")

                # ORA layers have x/y offsets — composite onto full canvas
                x_offset = int(layer_elem.get("x", 0))
                y_offset = int(layer_elem.get("y", 0))
                canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                canvas.paste(layer_img, (x_offset, y_offset))

                mask = self._to_binary_mask(canvas, width, height)
                results.append((name, mask))
                _log("ELIGEN", f"  Layer: \"{name}\" → mask")

        _log("ELIGEN", f"Loaded {len(results)} entities from {ora_path.name}", "success")
        return results

    # ------------------------------------------------------------------
    # Mask processing
    # ------------------------------------------------------------------

    @staticmethod
    def _to_binary_mask(
        img: Image.Image,
        width: int,
        height: int,
        threshold: int = 128,
    ) -> Image.Image:
        """
        Convert any image to a binary white-on-black mask.
        
        Handles:
          - RGBA (uses alpha channel as mask source)
          - RGB (converts to grayscale, thresholds)
          - L/1 (grayscale/binary, thresholds)
          
        Returns RGB image at target resolution: white (255,255,255)
        where entity is, black (0,0,0) elsewhere.
        """
        # Resize first (nearest for binary, bilinear for source)
        img = img.resize((width, height), Image.BILINEAR)

        # Extract the "presence" channel
        if img.mode == "RGBA":
            # Use alpha as the mask source — standard for PSD layers
            # and transparent PNGs
            alpha = img.split()[3]
            gray = alpha
        elif img.mode == "LA":
            gray = img.split()[1]  # alpha channel
        elif img.mode in ("L", "1"):
            gray = img.convert("L")
        else:
            # RGB or other — convert to grayscale
            gray = img.convert("L")

        # Threshold to pure binary
        binary = gray.point(lambda x: 255 if x >= threshold else 0, mode="L")

        # Convert to RGB (EliGen expects RGB masks)
        return Image.merge("RGB", (binary, binary, binary))
