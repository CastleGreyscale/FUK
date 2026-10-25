"""
PhotoCraft bridge endpoints.

PhotoCraft can't host a FUK panel or call out, so the web UI drives it through
these routes and the FUK server talks to PhotoCraft's control channel itself
(see photocraft_bridge.py for the protocol and what the desktop app allows):

    GET  /api/photocraft/status          connection, open documents, active doc's layers
    POST /api/photocraft/launch          start PhotoCraft with the control port open
    POST /api/photocraft/pull            composite or one layer → import entry (edit/control input)
    POST /api/photocraft/pull-inpaint    composite + mask (selection or `_mask` layer)
    POST /api/photocraft/pull-eligen     `@<prompt>` layers → an EliGen mask folder
    POST /api/photocraft/depth-to-layer  depth of the composite → a new layer (+ 16-bit/EXR files)
    POST /api/photocraft/place           any FUK image → a new layer

Pulled files are copied into an `import_###` history entry (the exchange
folder is reused), so the inputs survive and show up in History. Pulls never
start a generation: they fill the Image tab's inputs and you press Generate.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from blender_endpoints import bump_version
from photocraft_bridge import (
    MASK_LAYER,
    SELECTION_LAYER,
    Exchange,
    PhotoCraftClient,
    PhotoCraftError,
    depth_from_raw,
    entity_filename,
    export_document,
    flatten_rgb,
    mask_from_layer,
    place_array,
    read_image,
    to_8bit,
    write_depth_exr,
    write_depth_png16,
    write_depth_png8_rgb,
    write_png,
)
from project_endpoints import (
    get_cache_root,
    get_generation_output_dir,
    get_project_relative_url,
    save_generation_metadata,
)

router = APIRouter(prefix="/api/photocraft", tags=["photocraft"])

LAUNCHER = Path(__file__).resolve().parents[1] / "plugins" / "photocraft" / "launch_photocraft.sh"

# Depth models that are clean for commercial work (see THIRD_PARTY_LICENSES +
# the DA3 vendor README): DA3-LARGE/GIANT are CC BY-NC.
COMMERCIAL_DEPTH_MODELS = {"da3_mono_large", "da3_metric_large"}

_STATUS_CODES = {"unreachable": 503, "auth": 401, "timeout": 504, "remote": 502, "protocol": 502}

# One PhotoCraft operation at a time: each one opens scratch documents in the
# app, and two interleaved would pick up each other's.
_op_lock = threading.Lock()

_ctx: dict[str, Any] = {}


# ============================================================================
# Plumbing
# ============================================================================

def _cfg() -> dict:
    return ((_ctx.get("defaults") or {}).get("tools") or {}).get("photocraft") or {}


def _pictures_dir() -> Path:
    try:
        out = subprocess.run(["xdg-user-dir", "PICTURES"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.home() / "Pictures"


def _exchange() -> Exchange:
    root = _cfg().get("root") or str(_pictures_dir() / "fuk_photocraft")
    return Exchange(Path(root).expanduser())


def _port() -> int:
    return int(_cfg().get("port") or 47823)


def _client() -> PhotoCraftClient:
    return PhotoCraftClient(_port(), _exchange().token_file)


def _require_project() -> Path:
    cache_root = get_cache_root()
    if not cache_root:
        raise HTTPException(status_code=400, detail="No project loaded")
    return Path(cache_root)


async def _locked(fn: Callable, *args):
    """Run a blocking bridge operation off the event loop, one at a time."""
    def runner():
        if not _op_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="The PhotoCraft bridge is busy with another operation")
        try:
            try:
                _exchange().prune()
            except OSError:
                pass
            return fn(*args)
        finally:
            _op_lock.release()

    try:
        return await asyncio.to_thread(runner)
    except PhotoCraftError as e:
        raise HTTPException(status_code=_STATUS_CODES.get(e.kind, 502), detail=str(e))


def _log(message: str, level: str = "info") -> None:
    log = _ctx.get("log")
    if log is not None:
        getattr(log, level, log.info)("PhotoCraft", message)


def _fit_max_side(img: np.ndarray, max_side: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    if not max_side or max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)


def _max_side(requested: Optional[int]) -> int:
    """0 = full size. The default keeps a 6K document from becoming a 6K generation."""
    if requested is not None:
        return int(requested)
    return int(_cfg().get("max_pull_side", 2048) or 0)


def _thumbnail(src: Path, dest: Path) -> None:
    try:
        from PIL import Image
        with Image.open(src) as im:
            im.thumbnail((400, 400), Image.Resampling.LANCZOS)
            im.convert("RGB").save(dest, "JPEG", quality=85)
    except Exception:
        pass


def _register_import(import_dir: Path, main: Path, display_name: str, photocraft: dict) -> dict:
    """Write register_import-shaped metadata for a pull and tell the UI to refresh."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "source_path": str(main),
        "display_name": display_name,
        "media_type": "image",
        "link_type": "copy",
        "imported": True,
        "is_sequence": False,
        "source_app": "photocraft",
        "photocraft": photocraft,
    }
    (import_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    bump_version()
    cache_root = get_cache_root()
    try:
        import_id = str(import_dir.relative_to(cache_root)) if cache_root else import_dir.name
    except ValueError:
        import_id = import_dir.name
    return {"import_id": import_id}


def _export_meta(export, extra: Optional[dict] = None) -> dict:
    doc = export.doc
    return {
        "doc": doc.get("name"),
        "doc_index": doc.get("index"),
        "doc_size": [doc.get("width"), doc.get("height")],
        "bits": doc.get("depth"),
        **(extra or {}),
    }


def _source_pixels(export, source: str, layer: Optional[str]) -> tuple[np.ndarray, str]:
    """The RGB pixels a pull is about, plus a label for names."""
    ora = export.ora
    if source == "composite":
        return flatten_rgb(ora.composite), "composite"
    found = ora.find(layer or "")
    if found is None:
        raise HTTPException(status_code=404, detail=f"No layer named {layer!r} in {export.doc.get('name')}")
    return flatten_rgb(found.canvas(ora.width, ora.height)), found.name


def _active_layer_name(pc: PhotoCraftClient, doc_index: Optional[int]) -> Optional[str]:
    params = {} if doc_index is None else {"document": doc_index}
    listing = pc.execute("layer.list", flat=True, **params) or {}
    active = listing.get("activeLayer")
    return next((r.get("name") for r in listing.get("layers") or [] if r.get("id") == active), None)


# ============================================================================
# Requests
# ============================================================================

class PullRequest(BaseModel):
    source: str = "composite"            # composite | active | layer
    layer: Optional[str] = None          # layer name, for source=layer
    doc_index: Optional[int] = None      # default: the active document
    max_side: Optional[int] = None       # default: tools.photocraft.max_pull_side (0 = full size)


class PullInpaintRequest(BaseModel):
    mask_from: str = "auto"              # auto (selection, else _mask) | selection | layer | none
    mask_layer: str = MASK_LAYER
    doc_index: Optional[int] = None
    max_side: Optional[int] = None


class PullEligenRequest(BaseModel):
    prefix: str = "@"
    doc_index: Optional[int] = None
    max_side: Optional[int] = None


class DepthRequest(BaseModel):
    source: str = "composite"            # composite | active | layer
    layer: Optional[str] = None
    doc_index: Optional[int] = None
    depth_model: str = "da3_mono_large"
    open_as: str = "none"                # none | png16 | exr — also open the full-precision map as its own document


class PlaceRequest(BaseModel):
    path: str                            # api/project/cache/... URL or absolute path
    layer_name: Optional[str] = None
    doc_index: Optional[int] = None


# ============================================================================
# Routes
# ============================================================================

@router.get("/status")
async def status():
    ex = _exchange()
    base = {
        "port": _port(),
        "root": str(ex.root),
        "token_file_exists": ex.token_file.exists(),
        "reachable": False,
        "busy": _op_lock.locked(),
    }
    if base["busy"]:
        return base

    def probe():
        with _client() as pc:
            s = pc.execute("session.inspect") or {}
            docs = s.get("documents") or []
            active = s.get("active")
            out = {
                "reachable": True,
                "active_doc": active,
                "documents": [{**d, "active": d.get("index") == active} for d in docs],
                "layers": [],
            }
            if active is not None:
                info = pc.execute("document.inspect", document=active) or {}
                out["depth"] = info.get("depth")
                out["has_selection"] = bool(info.get("hasSelection"))
                listing = pc.execute("layer.list", document=active, flat=True) or {}
                out["layers"] = [{
                    "id": r.get("id"),
                    "name": r.get("name", ""),
                    "visible": r.get("visible"),
                    "kind": r.get("kind"),
                    "active": r.get("id") == listing.get("activeLayer"),
                    "is_entity": r.get("name", "").startswith("@"),
                    "is_mask": r.get("name") == MASK_LAYER,
                } for r in listing.get("layers") or []]
            return out

    try:
        base.update(await asyncio.to_thread(probe))
    except PhotoCraftError as e:
        base.update({"error": str(e), "error_kind": e.kind})
    return base


@router.post("/launch")
async def launch():
    def run():
        try:
            with _client() as pc:
                pc.execute("session.inspect")
            return {"started": False, "already_running": True}
        except PhotoCraftError:
            pass
        ex = _exchange()
        ex.ensure()
        env = {**os.environ, "FUK_PC_PORT": str(_port()), "FUK_PC_ROOT": str(ex.root)}
        if _cfg().get("app_id"):
            env["FUK_PC_APP"] = _cfg()["app_id"]
        log_file = open(ex.root / "photocraft.log", "ab")
        proc = subprocess.Popen([str(LAUNCHER)], env=env, stdout=log_file, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        log_file.close()
        _log(f"Launched PhotoCraft (pid {proc.pid}) on port {_port()}")
        return {"started": True, "already_running": False, "pid": proc.pid}

    return await asyncio.to_thread(run)


@router.post("/pull")
async def pull(req: PullRequest):
    _require_project()

    def run():
        with _client() as pc:
            layer = req.layer
            if req.source == "active":
                layer = _active_layer_name(pc, req.doc_index)
            export = export_document(pc, _exchange(), req.doc_index)
        source = "composite" if req.source == "composite" else "layer"
        rgb, label = _source_pixels(export, source, layer)
        full_h, full_w = rgb.shape[:2]
        rgb = _fit_max_side(rgb, _max_side(req.max_side))
        h, w = rgb.shape[:2]

        import_dir = get_generation_output_dir("import")
        path = write_png(import_dir / "photocraft.png", rgb)
        stem = Path(export.doc.get("name") or "document").stem
        reg = _register_import(import_dir, path, f"PhotoCraft: {stem} {label}",
                               _export_meta(export, {"source": source, "layer": label,
                                                     "pulled_size": [w, h]}))
        warnings = []
        if (w, h) != (full_w, full_h):
            warnings.append(f"Downscaled {full_w}x{full_h} → {w}x{h} (max_pull_side); "
                            f"Send to PhotoCraft scales results back up")
        _log(f"Pulled {label} of {export.doc.get('name')} ({w}x{h}) → {reg['import_id']}")
        return {"url": get_project_relative_url(path), "path": str(path), "width": w, "height": h,
                "doc_width": full_w, "doc_height": full_h, "layer": label, **reg, "warnings": warnings}

    return await _locked(run)


@router.post("/pull-inpaint")
async def pull_inpaint(req: PullInpaintRequest):
    _require_project()

    def run():
        want_selection = req.mask_from in ("auto", "selection")
        with _client() as pc:
            export = export_document(pc, _exchange(), req.doc_index, selection_mask=want_selection)
        ora = export.ora
        warnings = []
        mask_layer = None
        if want_selection and export.selection_layer:
            mask_layer = ora.find(export.selection_layer)
        elif req.mask_from == "selection":
            raise HTTPException(status_code=400, detail="There is no active selection in PhotoCraft")
        if mask_layer is None and req.mask_from in ("auto", "layer"):
            mask_layer = ora.find(req.mask_layer)
            if mask_layer is None and req.mask_from == "layer":
                raise HTTPException(status_code=404, detail=f"No {req.mask_layer!r} layer in {export.doc.get('name')}")
        if mask_layer is None and req.mask_from != "none":
            warnings.append(f"No selection and no {req.mask_layer!r} layer: pulled the image only (img2img)")

        rgb = flatten_rgb(ora.composite)
        max_side = _max_side(req.max_side)
        rgb = _fit_max_side(rgb, max_side)
        h, w = rgb.shape[:2]

        import_dir = get_generation_output_dir("import")
        image_path = write_png(import_dir / "photocraft.png", rgb)
        out = {"image": {"url": get_project_relative_url(image_path), "path": str(image_path),
                         "width": w, "height": h}, "mask": None}
        if mask_layer is not None:
            mask = mask_from_layer(mask_layer.canvas(ora.width, ora.height))
            if mask.max() < 128:
                warnings.append(f"The {mask_layer.name!r} mask is empty")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) if mask.shape[:2] != (h, w) else mask
            mask_path = write_png(import_dir / "mask.png", mask)
            out["mask"] = {"url": get_project_relative_url(mask_path), "path": str(mask_path),
                           "from": "selection" if mask_layer.name == SELECTION_LAYER else mask_layer.name}
        stem = Path(export.doc.get("name") or "document").stem
        reg = _register_import(import_dir, image_path, f"PhotoCraft: {stem} inpaint",
                               _export_meta(export, {"source": "inpaint", "pulled_size": [w, h],
                                                     "mask": (out["mask"] or {}).get("from")}))
        return {**out, **reg, "warnings": warnings}

    return await _locked(run)


_ENTITY_COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                   (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


@router.post("/pull-eligen")
async def pull_eligen(req: PullEligenRequest):
    _require_project()

    def run():
        with _client() as pc:
            export = export_document(pc, _exchange(), req.doc_index)
        ora = export.ora
        candidates = ora.entities(req.prefix)
        if not candidates:
            raise HTTPException(status_code=404, detail=(
                f"No layers named '{req.prefix}<prompt>' in {export.doc.get('name')}"))

        plate = _fit_max_side(flatten_rgb(ora.composite), _max_side(req.max_side))
        h, w = plate.shape[:2]
        import_dir = get_generation_output_dir("import")
        eligen_dir = import_dir / "eligen"
        eligen_dir.mkdir()
        preview = (plate.astype(np.float32) * 0.45)
        entities, warnings = [], []
        for layer in candidates:
            prompt = layer.name[len(req.prefix):].strip()
            mask = mask_from_layer(layer.canvas(ora.width, ora.height))
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
            coverage = float((mask >= 128).mean())
            if coverage == 0.0:
                warnings.append(f"Skipped {layer.name!r}: nothing painted")
                continue
            order = len(entities) + 1
            filename, truncated = entity_filename(order, prompt)
            if truncated:
                warnings.append(f"Prompt of {layer.name!r} was too long for a filename and was shortened")
            write_png(eligen_dir / filename, mask)
            colour = np.array(_ENTITY_COLOURS[(order - 1) % len(_ENTITY_COLOURS)], np.float32)
            sel = mask >= 128
            preview[sel] = preview[sel] * 0.4 + colour * 0.6
            entities.append({"order": order, "prompt": prompt, "layer": layer.name, "file": filename,
                             "coverage": round(coverage, 4)})
        if not entities:
            shutil.rmtree(import_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Every entity layer is empty")

        write_png(import_dir / "plate.png", plate)
        preview_path = write_png(import_dir / "eligen_preview.png", np.clip(preview, 0, 255).astype(np.uint8))
        stem = Path(export.doc.get("name") or "document").stem
        reg = _register_import(import_dir, preview_path, f"PhotoCraft: {stem} EliGen ({len(entities)})",
                               _export_meta(export, {"source": "eligen", "entities": entities,
                                                     "pulled_size": [w, h]}))
        _log(f"EliGen: {len(entities)} entities from {export.doc.get('name')} → {eligen_dir}")
        return {"eligen_source": str(eligen_dir), "entities": entities, "width": w, "height": h,
                "preview_url": get_project_relative_url(preview_path),
                "plate_url": get_project_relative_url(import_dir / "plate.png"),
                **reg, "warnings": warnings}

    return await _locked(run)


@router.post("/depth-to-layer")
async def depth_to_layer(req: DepthRequest):
    _require_project()
    if req.depth_model not in COMMERCIAL_DEPTH_MODELS:
        raise HTTPException(status_code=400, detail=(
            f"{req.depth_model} is not cleared for commercial use; use one of "
            f"{sorted(COMMERCIAL_DEPTH_MODELS)}"))
    if req.open_as not in ("none", "png16", "exr"):
        raise HTTPException(status_code=400, detail="open_as must be none, png16 or exr")
    busy = [g for g in (_ctx.get("active_generations") or {}).values()
            if g.get("status") in ("queued", "running")]
    if busy:
        raise HTTPException(status_code=409, detail="A generation is running; depth would compete for VRAM")

    def run():
        ex = _exchange()
        with _client() as pc:
            layer = req.layer
            if req.source == "active":
                layer = _active_layer_name(pc, req.doc_index)
            export = export_document(pc, ex, req.doc_index)
        source = "composite" if req.source == "composite" else "layer"
        rgb, label = _source_pixels(export, source, layer)
        h, w = rgb.shape[:2]

        manager = _ctx["preprocessor_manager"]
        DepthModel = _ctx["DepthModel"]
        gen_dir = get_generation_output_dir("preprocess_depth")
        source_path = write_png(gen_dir / "source.png", rgb)
        try:
            result = manager.depth(image_path=source_path, output_path=gen_dir / "processed.png",
                                   model=DepthModel(req.depth_model), invert=False, normalize=True)
            display = Path(result["output_path"])
            raw = display.parent / f"{display.stem}_raw.npy"
            sidecar = json.loads(raw.with_suffix(".json").read_text())
            d = depth_from_raw(np.load(raw), sidecar)
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            # The display PNG's orientation depends on the model; rewrite it from
            # the raw buffer so History and control maps are near = white too.
            write_depth_png8_rgb(display, d)
            depth16 = write_depth_png16(gen_dir / "depth16.png", d)
            exr = write_depth_exr(gen_dir / "depth.exr", d)
        except Exception:
            shutil.rmtree(gen_dir, ignore_errors=True)
            raise
        finally:
            manager.clear_caches(["depth"])
            _ctx["clear_vram"](full=True)

        save_generation_metadata(
            gen_dir=gen_dir, prompt="Preprocessor: depth", model="depth", seed=None,
            image_size=(w, h), source_image=str(source_path),
            parameters={**result.get("parameters", {}), "orientation": "near=white",
                        "normalization": "min-max of raw Z"},
            source_app="photocraft", photocraft=_export_meta(export, {"source": source, "layer": label}),
        )
        _thumbnail(display, gen_dir / "thumbnail.jpg")
        bump_version()

        name = f"FUK depth {gen_dir.name}"
        warnings = []
        with _client() as pc:
            placed = place_array(pc, ex, np.round(d * 65535.0).astype(np.uint16), name, export.doc["index"])
            opened = None
            if req.open_as != "none":
                src = depth16 if req.open_as == "png16" else exr
                staged = ex.inbox / f"{gen_dir.name}_{src.name}"
                shutil.copy(src, staged)
                opened = pc.call("app.open", path=ex.rel(staged))
                pc.execute("document.activate", document=export.doc["index"])
        if int(export.doc.get("depth") or 8) == 8:
            warnings.append("The document is 8-bit, so the depth layer has 256 levels; "
                            "open the 16-bit or EXR map for smooth falloff")
        _log(f"Depth → layer {name!r} in {export.doc.get('name')}")
        return {**placed, "preprocess_id": str(gen_dir.relative_to(get_cache_root())),
                "control_url": get_project_relative_url(display),
                "depth16_path": str(depth16), "exr_path": str(exr),
                "opened": opened, "doc_bits": export.doc.get("depth"), "warnings": warnings}

    return await _locked(run)


def _layer_name_for(path: Path) -> str:
    """`FUK img_gen_012 s1234` for generations, `FUK <folder>` for other entries."""
    meta_path = path.parent / "metadata.json"
    folder = path.parent.name
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        meta = {}
    if folder.startswith("img_gen_"):
        seed = meta.get("seed")
        return f"FUK {folder}" + (f" s{seed}" if seed is not None else "")
    if folder.startswith(("preprocess_", "import_", "upscale_", "layers_")):
        return f"FUK {folder} {path.stem}"
    return f"FUK {path.stem}"


@router.post("/place")
async def place(req: PlaceRequest):
    path = _ctx["resolve_input_path"](req.path)
    if path is None or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.path}")
    path = Path(path)
    if path.suffix.lower() in (".exr", ".mp4", ".mov", ".webm"):
        png = path.with_suffix(".png")
        if not png.exists():
            raise HTTPException(status_code=400, detail=f"Can only place still images, not {path.suffix}")
        path = png

    def run():
        image = read_image(path)
        if image.shape[2] == 4 and to_8bit(image)[..., 3].min() == 255:
            image = image[..., :3]
        name = req.layer_name or _layer_name_for(path)
        with _client() as pc:
            placed = place_array(pc, _exchange(), image, name, req.doc_index)
        _log(f"Placed {path.name} as {placed['name']!r} in {placed['doc_name']}")
        warnings = []
        if placed["size"] != placed["doc_size"] and not placed["resized"]:
            warnings.append(f"Image is {placed['size'][0]}x{placed['size'][1]} with a different aspect "
                            f"ratio from the document; placed unscaled")
        return {**placed, "warnings": warnings}

    return await _locked(run)


def setup_photocraft_routes(app, *, preprocessor_manager, DepthModel, resolve_input_path,
                            active_generations, clear_vram, defaults, log=None):
    _ctx.update(
        preprocessor_manager=preprocessor_manager,
        DepthModel=DepthModel,
        resolve_input_path=resolve_input_path,
        active_generations=active_generations,
        clear_vram=clear_vram,
        defaults=defaults,
        log=log,
    )
    app.include_router(router)
