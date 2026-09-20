# core/color_management.py
"""
OCIO-backed output colour management for EXR export.

Implements docs/ACES_COLOR_PIPELINE.md. The short version:

  * The VAE is trained on sRGB-encoded imagery, so the source is always either
    that encoding or — once the decode path has already linearised it — linear
    Rec.709. Which one it is depends on the export path, so the caller says.
  * Wide-gamut targets (ACEScg, ACES2065-1, Linear Rec.2020) go through OCIO,
    using the facility's own config when the seat has one.
  * Rec.709 targets (Linear, sRGB) stay on the exporter's hand-rolled transfer
    functions. An OCIO round trip to linear Rec.709 from a linear Rec.709
    source is a no-op matrix, so routing them through the config would buy
    nothing and risk changing output for every existing project.

OpenColorIO is an optional dependency, in the same pattern as OpenEXR: if the
import fails, only the two Rec.709 targets are offered and export falls back to
the hand-rolled path with a warning rather than failing.
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_CORE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _CORE_DIR.parent / "config"

try:
    import PyOpenColorIO as ocio
    _HAS_OCIO = True
except ImportError:  # optional dependency — see module docstring
    ocio = None
    _HAS_OCIO = False


# ============================================================================
# Source encodings
# ============================================================================

# What state the beauty array is in when it reaches the transform. The latent
# paths linearise inside _fuse_brackets and hand over SCENE_LINEAR; the PNG
# path hands over the 8-bit file contents untouched, which are SRGB_ENCODED.
# Getting this wrong is the double-linearisation bug in §6.1 of the spec.
SRGB_ENCODED = "srgb_encoded"
SCENE_LINEAR = "scene_linear"


# ============================================================================
# Targets
# ============================================================================

@dataclass(frozen=True)
class Target:
    key: str                     # canonical value carried by the API
    label: str                   # dropdown text
    needs_ocio: bool
    aliases: Tuple[str, ...]     # OCIO colour space names, probed in order
    chromaticities: Tuple[float, ...]   # 8 floats: red xy, green xy, blue xy, white xy
    note: str = ""


# Chromaticities per §7. AP0's blue y is negative — AP0 primaries sit outside
# the spectral locus on purpose, so that value is correct and not a typo.
_REC709_CHROMA = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
_REC2020_CHROMA = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290)
_AP1_CHROMA = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
_AP0_CHROMA = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770, 0.32168, 0.33767)


TARGETS: Dict[str, Target] = {t.key: t for t in (
    # "Linear" is the legacy value and stays the default. It is frozen on the
    # hand-rolled path deliberately (spec §13 Q2): an OCIO linear Rec.709
    # transform is numerically the same thing, so remapping it would only risk
    # moving output for projects saved before any of this existed.
    Target("Linear", "Linear Rec.709 (default)", False, (), _REC709_CHROMA),
    Target("Linear Rec.709", "Linear Rec.709 (explicit)", False, (), _REC709_CHROMA),
    Target("sRGB", "sRGB (display-referred, no transform)", False, (), _REC709_CHROMA,
           note="EXR has no transfer-function attribute — the encoding is "
                "recorded in the fuk/colorSpace string attribute only."),
    Target("ACEScg", "ACEScg (AP1)", True,
           ("ACEScg", "ACES - ACEScg", "lin_ap1", "acescg"), _AP1_CHROMA),
    Target("ACES2065-1", "ACES2065-1 (AP0, interchange)", True,
           ("ACES2065-1", "ACES - ACES2065-1", "lin_ap0", "aces2065-1"), _AP0_CHROMA),
    Target("Linear Rec.2020", "Linear Rec.2020", True,
           ("Linear Rec.2020", "lin_rec2020", "Utility - Linear - Rec.2020"),
           _REC2020_CHROMA),
)}

DEFAULT_TARGET = "Linear"

# Colour space names for our two source states. Names vary between config
# generations — the studio built-in calls the texture space "sRGB Encoded
# Rec.709 (sRGB)", the ACES CG config calls it "sRGB - Texture", and older
# facility configs "Utility - sRGB - Texture" — so probe a list rather than
# hardcoding one (spec §13 Q1). getColorSpace() also matches OCIO v2 aliases,
# which covers most of the remaining spellings for free.
_SOURCE_ALIASES: Dict[str, Tuple[str, ...]] = {
    SRGB_ENCODED: (
        "sRGB Encoded Rec.709 (sRGB)", "sRGB - Texture", "srgb_tex",
        "Utility - sRGB - Texture", "sRGB Texture", "sRGB",
    ),
    SCENE_LINEAR: (
        "Linear Rec.709 (sRGB)", "lin_rec709_srgb", "Linear Rec.709",
        "Utility - Linear - Rec.709", "Utility - Linear - sRGB", "lin_rec709",
    ),
}

# Last-resort roles, used only when no alias matched. There is no standard role
# for "the sRGB texture space", so texture_paint is a guess and says so in the
# log; scene_linear is deliberately NOT used for SCENE_LINEAR, because in an
# ACES config that role resolves to ACEScg, not to Rec.709.
_SOURCE_ROLES: Dict[str, Tuple[str, ...]] = {
    SRGB_ENCODED: ("texture_paint", "color_picking"),
    SCENE_LINEAR: (),
}

# The names the built-in config uses, for the cross-config interchange fallback
# (see _resolve_source). These are fixed because the built-in config is fixed.
_BUILTIN_CONFIG_URI = "ocio://studio-config-latest"
_BUILTIN_SOURCE_NAMES = {
    SRGB_ENCODED: "sRGB Encoded Rec.709 (sRGB)",
    SCENE_LINEAR: "Linear Rec.709 (sRGB)",
}


# ============================================================================
# Config resolution
# ============================================================================

@dataclass
class _ConfigState:
    config: object = None
    path: str = ""
    source: str = ""        # env | settings | builtin
    error: str = ""
    loaded: bool = False


_state = _ConfigState()
# Processor cache. getProcessor() is not free and the sequence path would hit
# it once per frame otherwise.
_processors: Dict[Tuple[str, str], object] = {}


def _settings_config_path() -> Optional[str]:
    """`color.ocio_config_path` from defaults.json, if set."""
    try:
        with open(_CONFIG_DIR / "defaults.json") as fh:
            value = json.load(fh).get("color", {}).get("ocio_config_path", "")
        return value.strip() or None
    except (OSError, ValueError, AttributeError):
        return None


def _load_config() -> _ConfigState:
    """Resolve and load the OCIO config once. First hit wins (spec §4)."""
    if _state.loaded:
        return _state

    _state.loaded = True
    if not _HAS_OCIO:
        _state.error = "PyOpenColorIO not installed (pip install opencolorio)"
        return _state

    candidates = []
    env_path = os.environ.get("OCIO", "").strip()
    if env_path:
        candidates.append((env_path, "env"))
    settings_path = _settings_config_path()
    if settings_path:
        candidates.append((settings_path, "settings"))
    candidates.append((_BUILTIN_CONFIG_URI, "builtin"))

    for path, source in candidates:
        try:
            _state.config = ocio.Config.CreateFromFile(path)
            _state.path = path
            _state.source = source
            print(f"[COLOR] OCIO config ({source}): {path}")
            return _state
        except Exception as e:      # noqa: BLE001 — any config failure falls through
            print(f"[COLOR] ⚠ OCIO config from {source} unusable ({path}): "
                  f"{str(e).splitlines()[0][:120]}")
            _state.error = str(e).splitlines()[0][:200]

    _state.config = None
    return _state


_builtin = None


def _builtin_config():
    """The built-in studio config, used for the interchange fallback. Cached on
    the module the same way the main config is."""
    global _builtin
    if _builtin is None and _HAS_OCIO:
        try:
            _builtin = ocio.Config.CreateFromFile(_BUILTIN_CONFIG_URI)
        except Exception:   # noqa: BLE001
            _builtin = False
    return _builtin or None


def is_available() -> bool:
    """True when a usable OCIO config loaded."""
    return _load_config().config is not None


def status() -> Dict[str, object]:
    """Availability block for /api/export/capabilities and export metadata.

    The artist needs to see which config is actually in play — a silent
    fallback to the built-in when the facility config failed to load is exactly
    the kind of thing that surfaces three weeks later as a wrong-looking EXR.
    """
    st = _load_config()
    return {
        "available": st.config is not None,
        "config_path": st.path,
        "config_source": st.source,
        "version": getattr(ocio, "__version__", "") if _HAS_OCIO else "",
        "error": st.error,
    }


# ============================================================================
# Colour space resolution
# ============================================================================

def _find_space(config, names) -> Optional[str]:
    """First of `names` that the config resolves, as its canonical name."""
    for name in names:
        try:
            cs = config.getColorSpace(name)
        except Exception:   # noqa: BLE001 — malformed name, keep probing
            continue
        if cs is not None:
            return cs.getName()
    return None


def _resolve_target(config, target: Target) -> Optional[str]:
    """Canonical name of `target` in this config, or None if it has no such space."""
    return _find_space(config, target.aliases)


def _resolve_source(config, source: str) -> Optional[str]:
    """Canonical name of our source encoding in this config."""
    name = _find_space(config, _SOURCE_ALIASES[source])
    if name:
        return name
    for role in _SOURCE_ROLES[source]:
        try:
            cs = config.getColorSpace(role)
        except Exception:   # noqa: BLE001
            continue
        if cs is not None:
            print(f"[COLOR] ⚠ no known name for the {source} space in this "
                  f"config — falling back to the '{role}' role "
                  f"({cs.getName()}). Check the result before trusting it.")
            return cs.getName()
    return None


def available_targets() -> List[Dict[str, object]]:
    """Targets this seat can actually deliver, for the capabilities endpoint.

    An OCIO target that the loaded config cannot resolve is left out entirely
    rather than offered and failed at export time.
    """
    config = _load_config().config
    out = []
    for target in TARGETS.values():
        if not target.needs_ocio:
            out.append({"value": target.key, "name": target.label,
                        "requires_ocio": False, "note": target.note})
            continue
        if config is None:
            continue
        resolved = _resolve_target(config, target)
        if resolved:
            out.append({"value": target.key, "name": target.label,
                        "requires_ocio": True, "ocio_name": resolved,
                        "note": target.note})
    return out


def resolve(name: Optional[str]) -> Target:
    """Map an API colour_space value onto a Target, case-insensitively.

    Unknown values fall back to the default rather than raising — an export is
    not worth failing over a typo in saved project state.
    """
    if not name:
        return TARGETS[DEFAULT_TARGET]
    if name in TARGETS:
        return TARGETS[name]
    lowered = name.strip().lower()
    for key, target in TARGETS.items():
        if key.lower() == lowered:
            return target
    print(f"[COLOR] ⚠ unknown colour space '{name}' — using {DEFAULT_TARGET}")
    return TARGETS[DEFAULT_TARGET]


# ============================================================================
# Transform
# ============================================================================

def _get_processor(config, src_name: str, dst_name: str, via_builtin: bool):
    """Cached CPU processor for one (source, target) pair.

    `via_builtin` routes the source through the built-in config's definition
    using the `aces_interchange` role, for configs that name their sRGB texture
    space in a way none of the aliases or roles caught.
    """
    key = (("builtin:" if via_builtin else "") + src_name, dst_name)
    if key not in _processors:
        if via_builtin:
            proc = ocio.Config.GetProcessorFromConfigs(
                _builtin_config(), src_name, config, dst_name)
        else:
            proc = config.getProcessor(src_name, dst_name)
        _processors[key] = proc.getDefaultCPUProcessor()
    return _processors[key]


def transform(arr: np.ndarray, target: Target, source: str,
              quiet: bool = False) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply the OCIO transform for `target` to a float32 [H, W, 3] beauty array.

    Returns (array, info). On any failure the array comes back untouched with
    `info['ocio'] = False`, and the caller applies its own hand-rolled
    transform instead — the two paths are mutually exclusive by construction
    (spec §6.1), so a fallback here must not leave a half-applied transform
    behind. Nothing in here clamps: 709 into a wider gamut cannot itself
    produce negatives, but the decode path hands over real values below 0 and
    above 1, and every one of them has to survive to the file (spec §6.3).
    """
    info: Dict[str, object] = {"ocio": False, "target": target.key,
                               "source_encoding": source}

    if not target.needs_ocio:
        return arr, info

    config = _load_config().config
    if config is None:
        info["warning"] = (
            f"OpenColorIO unavailable ({_state.error or 'not installed'}) — "
            f"'{target.key}' fell back to linear Rec.709.")
        return arr, info

    dst_name = _resolve_target(config, target)
    src_name = _resolve_source(config, source)
    via_builtin = False
    if dst_name and not src_name and _builtin_config() is not None:
        # The target resolved but the source did not. Rather than give up,
        # define the source from the built-in config and let OCIO bridge the
        # two through aces_interchange — which is what that role is for.
        src_name = _BUILTIN_SOURCE_NAMES[source]
        via_builtin = True
        print(f"[COLOR] no {source} space in this config — bridging from the "
              f"built-in config's '{src_name}' via aces_interchange")
    if not src_name or not dst_name:
        missing = target.key if not dst_name else f"the {source} source space"
        info["warning"] = (
            f"OCIO config '{_state.path}' does not resolve {missing} — "
            f"'{target.key}' fell back to linear Rec.709.")
        return arr, info

    work = np.ascontiguousarray(arr, dtype=np.float32)
    try:
        started = time.perf_counter()
        # applyRGB is in-place and needs contiguous float32. Measured ~21 ms at
        # 1080p against the decode's seconds-per-frame, so getOptimizedProcessor
        # is not worth the extra surface here (spec §6.2, §11 phase 4).
        _get_processor(config, src_name, dst_name, via_builtin).applyRGB(work)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except Exception as e:  # noqa: BLE001 — never fail an export on a transform
        info["warning"] = (f"OCIO transform {src_name} → {dst_name} failed "
                           f"({str(e).splitlines()[0][:120]}) — "
                           f"'{target.key}' fell back to linear Rec.709.")
        return arr, info

    info.update({"ocio": True, "ocio_source": src_name, "ocio_target": dst_name,
                 "config_path": _state.path, "config_source": _state.source,
                 "transform_ms": round(elapsed_ms, 2)})
    if not quiet:
        print(f"  ✓ OCIO {src_name} → {dst_name} ({elapsed_ms:.1f} ms) "
              f"→ range [{work.min():.4f}, {work.max():.4f}]")
    return work, info


# ============================================================================
# Chromaticities
# ============================================================================

# OpenEXR 3.4's deprecated Python binding accepts a chromaticities attribute
# and then writes eight zeros into it, and the modern OpenEXR.File API rejects
# every documented spelling of the value ("expected a 6-tuple"). Both are
# binding bugs, not EXR ones. So: hand the legacy writer this placeholder,
# which makes it reserve a correctly typed and sized attribute, then overwrite
# the 32 bytes of payload in place. Verified round trip through exrheader and
# both Python readers.
_CHROMA_ATTR_MARKER = b"chromaticities\x00chromaticities\x00"
_CHROMA_ATTR_BYTES = 32


def chromaticities_placeholder():
    """Zeroed Imath.Chromaticities to reserve the header attribute.

    Returns None when Imath is missing, in which case the caller simply writes
    no chromaticities — the export still succeeds.
    """
    try:
        import Imath
    except ImportError:
        return None
    zero = Imath.chromaticity(0.0, 0.0)
    return Imath.Chromaticities(zero, zero, zero, zero)


def patch_chromaticities(path: Path, target: Target, quiet: bool = False) -> bool:
    """Write `target`'s primaries into an already-written EXR's header.

    Only touches the 32 bytes of the reserved attribute's payload; pixels and
    every other attribute are left exactly as OpenEXR wrote them.
    """
    try:
        with open(path, "r+b") as fh:
            # Attributes live at the top of the file; 64 KB is far more than
            # any header we write, cryptomatte manifests included.
            head = fh.read(65536)
            index = head.find(_CHROMA_ATTR_MARKER)
            if index < 0:
                if not quiet:
                    print("  ⚠ chromaticities attribute not found in header — "
                          "skipping (the file is valid, Nuke just won't "
                          "auto-interpret it)")
                return False
            offset = index + len(_CHROMA_ATTR_MARKER)
            size = struct.unpack_from("<i", head, offset)[0]
            if size != _CHROMA_ATTR_BYTES:
                print(f"  ⚠ unexpected chromaticities attribute size {size} — "
                      f"skipping")
                return False
            fh.seek(offset + 4)
            fh.write(struct.pack("<8f", *target.chromaticities))
        return True
    except OSError as e:
        print(f"  ⚠ could not write chromaticities: {e}")
        return False


def header_attributes(target: Target, source: str,
                      info: Dict[str, object]) -> Dict[str, bytes]:
    """String attributes documenting the transform, for the EXR header.

    Resolve does not reliably read chromaticities, so a colourist needs these
    to confirm by hand; and without them a wrong-looking EXR three weeks later
    is unforensicable (spec §9).
    """
    attrs = {
        "fuk/colorSpace": target.key,
        "fuk/sourceColorSpace": ("sRGB (encoded)" if source == SRGB_ENCODED
                                 else "Linear Rec.709"),
        "fuk/colorTransform": ("OCIO" if info.get("ocio") else "built-in"),
    }
    if info.get("ocio"):
        attrs["fuk/ocioConfig"] = str(info.get("config_path", ""))
        attrs["fuk/ocioConfigSource"] = str(info.get("config_source", ""))
        attrs["fuk/ocioTransform"] = (f"{info.get('ocio_source')} → "
                                      f"{info.get('ocio_target')}")
    return {k: v.encode("utf-8") for k, v in attrs.items()}
