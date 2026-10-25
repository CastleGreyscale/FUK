"""
PhotoCraft bridge — control-channel client and pure image helpers.

PhotoCraft (ArtCraft's Rust image editor) has no in-app plugin API and never
calls out, so FUK drives it: the desktop app is started with
`--control <port> --control-token-file <f>` and answers JSON lines on
127.0.0.1. Every connection authenticates first:

    {"id":"auth","method":"auth","params":{"token":"<64 hex>"}}

then each request is {"id","method","params"} → {"id","ok","result"|"error"}.
The protocol is pull-only (no events) and the app gives up on a reply after
60 s.

What the DESKTOP server allows (probed on 0.16; the headless `photocraft-cli
serve` has a richer doc.* API that the desktop app does not):
  - file I/O only through `app.open {path}` / `app.save {path}`, with paths
    RELATIVE to the automation roots — so everything goes through one fixed
    exchange folder (see Exchange). `app.save` re-targets the document it
    saves, so it is only ever called on a scratch duplicate.
  - `engine.execute` for path-free commands. Anything that touches a path or an
    ambient profile (file.saveACopy, layer.exportAs, image.mode.*) is refused.
  - requests run on the UI frame loop: while the window is not drawing
    (occluded under a vsync present mode, display asleep) they queue and time
    out. The launcher uses `--present-mode mailbox` for that reason.

So pulling a document is: image.duplicate → hide utility layers in the copy →
app.save the copy as .ora → close it, then read the ORA here (it carries the
composite plus every layer with its offset). Placing a file is: app.open it as
its own document → layer.copyToDocument into the target → close it.

This module has no FastAPI or project imports (and no module-level mutable
state) so tests can import it without starting the server; the routes live in
photocraft_endpoints.py.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np


class PhotoCraftError(Exception):
    """A bridge failure. `kind` is one of unreachable, auth, protocol, remote, timeout."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class PhotoCraftClient:
    """One authenticated connection to a running PhotoCraft.

    Use as a context manager, one connection per bridge operation, so a
    PhotoCraft restart never leaves a stale socket behind:

        with PhotoCraftClient(port, token_file) as pc:
            doc = pc.call("doc.inspect")
    """

    def __init__(self, port: int, token_file: str | Path, host: str = "127.0.0.1",
                 timeout: float = 65.0, connect_timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.token_file = Path(token_file)
        # Just over PhotoCraft's own 60 s reply limit, so its timeout error
        # (which says whether the command ran) reaches us before ours fires.
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None
        self._reader = None
        self._next_id = 1

    def __enter__(self) -> "PhotoCraftClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        try:
            token = self.token_file.read_text().strip()
        except OSError as e:
            raise PhotoCraftError(
                "unreachable",
                f"No control token at {self.token_file} ({e.strerror}). "
                f"Start PhotoCraft with the FUK launcher.")
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except OSError as e:
            raise PhotoCraftError(
                "unreachable",
                f"PhotoCraft is not listening on {self.host}:{self.port} ({e}). "
                f"Start it with the FUK launcher; a plain `flatpak run` has no "
                f"network share, so its control port is invisible to FUK.")
        sock.settimeout(self.timeout)
        self._sock = sock
        self._reader = sock.makefile("rb")
        reply = self._roundtrip({"id": "auth", "method": "auth", "params": {"token": token}})
        if not reply.get("ok"):
            self.close()
            raise PhotoCraftError("auth", f"PhotoCraft refused the token: {reply.get('error')}")

    def close(self) -> None:
        for closer in (self._reader, self._sock):
            try:
                if closer is not None:
                    closer.close()
            except OSError:
                pass
        self._reader = None
        self._sock = None

    def _roundtrip(self, frame: dict) -> dict:
        if self._sock is None:
            raise PhotoCraftError("protocol", "not connected")
        try:
            self._sock.sendall((json.dumps(frame) + "\n").encode("utf-8"))
            line = self._reader.readline()
        except socket.timeout:
            raise PhotoCraftError(
                "timeout", f"No reply to {frame.get('method')} within {self.timeout:.0f}s")
        except OSError as e:
            raise PhotoCraftError("unreachable", f"Connection to PhotoCraft lost: {e}")
        if not line:
            raise PhotoCraftError("unreachable", "PhotoCraft closed the connection")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError:
            raise PhotoCraftError("protocol", f"Bad reply line: {line[:200]!r}")
        if reply.get("id") != frame["id"]:
            raise PhotoCraftError(
                "protocol", f"Reply id {reply.get('id')!r} does not match request {frame['id']!r}")
        return reply

    def call(self, method: str, **params) -> Any:
        """Run one protocol method and return its result, raising on ok:false."""
        frame = {"id": self._next_id, "method": method, "params": params}
        self._next_id += 1
        reply = self._roundtrip(frame)
        if not reply.get("ok"):
            error = str(reply.get("error"))
            # PhotoCraft's own 60 s deadline: the request sat queued because the
            # window was not drawing frames.
            if error.startswith("timeout"):
                raise PhotoCraftError(
                    "timeout",
                    f"{method}: PhotoCraft did not run the request — its window is "
                    f"probably hidden or the display asleep. Bring it into view and "
                    f"retry. ({error})")
            raise PhotoCraftError("remote", f"{method}: {error}")
        return reply.get("result")

    def execute(self, command: str, wait: bool = True, **params) -> Any:
        """Run an engine command by id (never opens a dialog)."""
        return self.call("engine.execute", command=command, params=params, wait=wait)


@dataclass
class Exchange:
    """The one folder PhotoCraft and FUK share.

    PhotoCraft's automation roots are set to `root/exchange`, and it only
    accepts paths relative to them, so FUK writes inputs to `in/` and asks
    PhotoCraft to write into `out/<op>/`. The root sits under the user's
    Pictures folder by default, which the Flatpak sandbox already sees at the
    same absolute path.
    """

    root: Path

    @property
    def base(self) -> Path:
        return self.root / "exchange"

    @property
    def token_file(self) -> Path:
        return self.root / "control.token"

    @property
    def inbox(self) -> Path:
        return self.base / "in"

    def ensure(self) -> None:
        (self.base / "in").mkdir(parents=True, exist_ok=True)
        (self.base / "out").mkdir(parents=True, exist_ok=True)

    def new_out(self, label: str) -> Path:
        """A fresh out/<op>/ folder for PhotoCraft to write one operation into."""
        self.ensure()
        op = self.base / "out" / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}_{time.time_ns() % 1_000_000:06d}"
        op.mkdir(parents=True)
        return op

    def rel(self, path: Path) -> str:
        """The root-relative path PhotoCraft needs (absolute paths are refused)."""
        return Path(path).resolve().relative_to(self.base.resolve()).as_posix()

    def prune(self, keep: int = 20) -> None:
        """Drop all but the newest `keep` op folders and inbox files.

        Pulled files are copied into the FUK project, so nothing here is the
        only copy of anything.
        """
        for folder in (self.base / "out", self.base / "in"):
            if not folder.is_dir():
                continue
            entries = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in entries[keep:]:
                if old.is_dir():
                    shutil.rmtree(old, ignore_errors=True)
                else:
                    old.unlink(missing_ok=True)


# ============================================================================
# Image helpers (arrays are RGB(A) channel order, uint8 or uint16)
# ============================================================================

# Layers FUK reads but never wants in the composite: @<prompt> EliGen entities
# and _-prefixed utility layers (_mask, the selection layer below).
ENTITY_PREFIX = "@"
UTILITY_PREFIXES = ("@", "_")
MASK_LAYER = "_mask"
SELECTION_LAYER = "_fuk_selection"


def is_utility(name: str) -> bool:
    return name.startswith(UTILITY_PREFIXES)


def _to_rgba(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)


def decode_image(buf: bytes) -> np.ndarray:
    """PNG/TIFF bytes → RGBA at the file's own bit depth."""
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("could not decode image")
    return _to_rgba(arr)


def read_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def write_png(path: str | Path, arr: np.ndarray) -> Path:
    """Write gray / RGB / RGBA (uint8 or uint16) as PNG."""
    path = Path(path)
    if arr.ndim == 3 and arr.shape[2] == 4:
        out = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif arr.ndim == 3:
        out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        out = arr
    if not cv2.imwrite(str(path), out):
        raise OSError(f"could not write {path}")
    return path


def to_8bit(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.round(arr / 257.0).astype(np.uint8)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def flatten_rgb(rgba: np.ndarray, background: int = 255) -> np.ndarray:
    """RGBA → 8-bit RGB, compositing any transparency over a flat background."""
    rgba = to_8bit(rgba)
    if rgba.shape[2] == 3:
        return rgba
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    if alpha.min() >= 1.0:
        return rgba[..., :3].copy()
    rgb = rgba[..., :3].astype(np.float32) * alpha + background * (1.0 - alpha)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def to_canvas(pixels: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Paste a layer trimmed to its bounds back at (x, y) on a transparent canvas."""
    canvas = np.zeros((height, width, pixels.shape[2]), dtype=pixels.dtype)
    h, w = pixels.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = pixels[y0 - y:y1 - y, x0 - x:x1 - x]
    return canvas


def mask_from_layer(rgba: np.ndarray) -> np.ndarray:
    """A canvas-sized layer → 8-bit mask, white = the painted region.

    Paint on a transparent layer is read from its alpha (whatever colour was
    used); a fully opaque layer is read by luminance (white on black).
    Soft edges are kept so FUK's own feathering still has something to work on.
    """
    rgba8 = to_8bit(rgba)
    alpha = rgba8[..., 3]
    if alpha.min() < 255:
        return alpha.copy()
    return cv2.cvtColor(rgba8[..., :3], cv2.COLOR_RGB2GRAY)


_UNSAFE_NAME = re.compile(r"[/\\\x00-\x1f]")


def entity_filename(order: int, prompt: str, max_bytes: int = 180) -> tuple[str, bool]:
    """`NN_<prompt>.png` for core/eligen_loader.py's directory mode.

    The loader parses the order from the numeric prefix and the prompt from the
    rest of the stem. Returns (filename, truncated).
    """
    safe = " ".join(_UNSAFE_NAME.sub(" ", prompt).split())
    raw = safe.encode("utf-8")
    truncated = len(raw) > max_bytes
    if truncated:
        safe = raw[:max_bytes].decode("utf-8", "ignore").rstrip()
    return f"{order:02d}_{safe}.png", truncated


def layer_name(text: str, limit: int = 120) -> str:
    return " ".join(_UNSAFE_NAME.sub(" ", text).split())[:limit] or "FUK"


def fit_back(img: np.ndarray, width: int, height: int, slack: int = 16,
             tolerance: float = 0.01) -> tuple[np.ndarray, bool]:
    """Resize a result back to the document size when only scale and rounding differ.

    FUK rounds output sizes up to multiples of 16 and the bridge may have
    downscaled the pull, so a result that is the document scaled by some s with
    each side then off by under `slack` px (or within `tolerance` of its aspect
    ratio) is resized back so it lands pixel-aligned. Returns (image, resized).
    Anything else is left alone.
    """
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img, False
    rounding_only = (abs(h - w * height / width) < slack or abs(w - h * width / height) < slack)
    same_aspect = abs((w / h) - (width / height)) <= tolerance * (width / height)
    if not (rounding_only or same_aspect):
        return img, False
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4), True


# ---------------------------------------------------------------- depth

def depth_from_raw(z: np.ndarray, sidecar: dict) -> np.ndarray:
    """FUK's raw depth buffer → float32 in [0, 1] with near = 1 (white).

    The raw .npy is true Z (near = small); its sidecar says so. Normalising it
    here (plain min-max, no percentiles) makes the map deterministic and its
    orientation independent of how a model's display PNG was written.
    """
    if sidecar.get("space") != "z":
        raise ValueError(f"raw depth is not Z-space (sidecar: {sidecar})")
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 3:
        z = z[..., 0]
    finite = np.isfinite(z)
    if not finite.any():
        raise ValueError("raw depth has no finite values")
    zmin, zmax = float(z[finite].min()), float(z[finite].max())
    d = (zmax - z) / max(zmax - zmin, 1e-8)
    d[~finite] = 0.0
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def write_depth_png16(path: str | Path, d: np.ndarray) -> Path:
    return write_png(path, np.round(d * 65535.0).astype(np.uint16))


def write_depth_png8_rgb(path: str | Path, d: np.ndarray) -> Path:
    g = np.round(d * 255.0).astype(np.uint8)
    return write_png(path, np.dstack([g, g, g]))


def write_depth_exr(path: str | Path, d: np.ndarray) -> Path:
    """Single-channel 32-bit float EXR (channel Y), near = 1."""
    import OpenEXR

    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"Y": np.ascontiguousarray(d, dtype=np.float32)}) as exr:
        exr.write(str(path))
    return Path(path)


# ---------------------------------------------------------------- ORA

@dataclass
class OraLayer:
    name: str
    visible: bool
    x: int
    y: int
    opacity: float
    pixels: np.ndarray          # RGBA, trimmed to the layer's bounds
    group: str = ""             # "/"-joined parent group names; "" = top level

    def canvas(self, width: int, height: int) -> np.ndarray:
        return to_canvas(self.pixels, self.x, self.y, width, height)


@dataclass
class OraDoc:
    width: int
    height: int
    composite: np.ndarray       # RGBA, the visible-layer merge PhotoCraft wrote
    layers: List[OraLayer] = field(default_factory=list)   # top-first

    def find(self, name: str) -> Optional[OraLayer]:
        return next((l for l in self.layers if l.name == name), None)

    def entities(self, prefix: str = ENTITY_PREFIX) -> List[OraLayer]:
        return [l for l in self.layers if l.name.startswith(prefix) and l.name[len(prefix):].strip()]


def read_ora(path: str | Path) -> OraDoc:
    """Read an OpenRaster file as PhotoCraft writes it (stack.xml lists layers top-first)."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("stack.xml"))
        width, height = int(root.get("w", 0)), int(root.get("h", 0))
        layers: List[OraLayer] = []

        def walk(stack, ox: int, oy: int, group: str, parent_visible: bool) -> None:
            for el in stack:
                x = ox + int(float(el.get("x", 0)))
                y = oy + int(float(el.get("y", 0)))
                visible = parent_visible and el.get("visibility", "visible") != "hidden"
                name = el.get("name", "")
                if el.tag == "stack":
                    walk(el, x, y, f"{group}/{name}" if group else name, visible)
                elif el.tag == "layer" and el.get("src"):
                    layers.append(OraLayer(
                        name=name, visible=visible, x=x, y=y,
                        opacity=float(el.get("opacity", 1.0)),
                        pixels=decode_image(z.read(el.get("src"))), group=group))

        top = root.find("stack")
        if top is not None:
            walk(top, 0, 0, "", True)
        if "mergedimage.png" not in z.namelist():
            raise ValueError(f"{path} has no mergedimage.png")
        composite = decode_image(z.read("mergedimage.png"))
    return OraDoc(width=width, height=height, composite=composite, layers=layers)


# ============================================================================
# Document operations (one PhotoCraftClient connection each)
# ============================================================================

@dataclass
class Export:
    ora: OraDoc
    ora_path: Path
    doc: dict                   # index, name, path, width, height, depth, has_selection
    selection_layer: Optional[str] = None


def _documents(pc: PhotoCraftClient) -> tuple[list, Optional[int]]:
    s = pc.execute("session.inspect") or {}
    return s.get("documents") or [], s.get("active")


def resolve_document(pc: PhotoCraftClient, doc_index: Optional[int] = None) -> dict:
    docs, active = _documents(pc)
    if not docs:
        raise PhotoCraftError("remote", "No document is open in PhotoCraft")
    target = active if doc_index is None else doc_index
    doc = next((d for d in docs if d.get("index") == target), None)
    if doc is None:
        raise PhotoCraftError("remote", f"PhotoCraft has no document {target}")
    return doc


def export_document(pc: PhotoCraftClient, ex: Exchange, doc_index: Optional[int] = None,
                    selection_mask: bool = False) -> Export:
    """Snapshot a document as ORA without touching it.

    Works on an image.duplicate copy (app.save re-targets whatever it saves):
    utility layers are hidden in the copy so the merged composite is clean,
    and with `selection_mask` the active selection is filled white on a new
    `_fuk_selection` layer so it travels as pixels.
    """
    doc = dict(resolve_document(pc, doc_index))
    target = doc["index"]
    pc.execute("document.activate", document=target)
    info = pc.execute("document.inspect", document=target) or {}
    doc["depth"] = int(info.get("depth") or 8)
    doc["has_selection"] = bool(info.get("hasSelection"))

    dup = (pc.execute("image.duplicate", name="fuk_export") or {}).get("document")
    if dup is None or dup == target:
        raise PhotoCraftError("protocol", "image.duplicate did not create a copy")
    sel_layer = None
    out = ex.new_out("pull")
    try:
        rows = (pc.execute("layer.list", document=dup, flat=True) or {}).get("layers") or []
        noisy = [r["id"] for r in rows if r.get("visible") and is_utility(r.get("name", ""))]
        if noisy:
            pc.execute("layer.setProps", layers=noisy, visible=False)
        if selection_mask and doc["has_selection"]:
            pc.execute("layer.new.layer", name=SELECTION_LAYER)
            pc.execute("edit.fill", contents="white")
            pc.execute("layer.setProps", visible=False)
            sel_layer = SELECTION_LAYER
        pc.call("app.save", path=ex.rel(out / "document.ora"))
    finally:
        for cmd, params in (("file.close", {"document": dup}), ("document.activate", {"document": target})):
            try:
                pc.execute(cmd, **params)
            except PhotoCraftError:
                pass
    ora_path = out / "document.ora"
    return Export(ora=read_ora(ora_path), ora_path=ora_path, doc=doc, selection_layer=sel_layer)


def place_array(pc: PhotoCraftClient, ex: Exchange, image: np.ndarray, name: str,
                doc_index: Optional[int] = None, fit: bool = True) -> dict:
    """Add an image to a document as a new layer named `name`.

    The image is staged in the exchange inbox, opened as its own document,
    copied into the target with layer.copyToDocument (which keeps its
    position: canvas-sized images land at 0,0) and the staging document closed.
    """
    doc = resolve_document(pc, doc_index)
    target = doc["index"]
    resized = False
    if fit:
        image, resized = fit_back(image, int(doc["width"]), int(doc["height"]))
    name = layer_name(name)
    ex.ensure()
    staged = write_png(ex.inbox / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}.png", image)

    pc.call("app.open", path=ex.rel(staged))
    _, src = _documents(pc)
    if src is None or src == target:
        raise PhotoCraftError("protocol", "app.open did not open the staged image as a new document")
    try:
        rows = (pc.execute("layer.list", document=src) or {}).get("layers") or []
        if not rows:
            raise PhotoCraftError("protocol", "the staged document has no layers")
        copied = pc.execute("layer.copyToDocument", document=target, source=src,
                            layers=[rows[0]["id"]], name=name) or {}
    finally:
        for cmd, params in (("file.close", {"document": src}), ("document.activate", {"document": target})):
            try:
                pc.execute(cmd, **params)
            except PhotoCraftError:
                pass
    layer_ids = copied.get("layers") or [None]
    h, w = image.shape[:2]
    return {"document": target, "doc_name": doc.get("name"), "layer": layer_ids[0], "name": name,
            "resized": resized, "size": [w, h], "doc_size": [doc["width"], doc["height"]]}
