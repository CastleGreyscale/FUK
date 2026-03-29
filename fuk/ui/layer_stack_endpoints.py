"""
Layer Stack Endpoints

REST API for the non-destructive latent layer editing system.

Stacks live as img_edit_XXX directories inside the project cache, created
via get_generation_output_dir("img_edit"). This means they appear in
generation history alongside img_gen_XXX entries.

Layer edits run through the normal /api/generate/image endpoint (with
stack_id in the request). After generation, the web server calls
register_layer_from_generation() to add the result to the stack.

Routes:
    POST   /api/layers/init                                     Initialize a new stack from a source image
    GET    /api/layers/{stack_id}                                Get stack manifest
    PATCH  /api/layers/{stack_id}/layer/{layer_id}/toggle        Enable/disable layer
    PATCH  /api/layers/{stack_id}/layer/{layer_id}/version       Switch active version
    POST   /api/layers/{stack_id}/reorder                        Reorder all layers
    DELETE /api/layers/{stack_id}/layer/{layer_id}               Remove layer
    POST   /api/layers/{stack_id}/flatten                        Flatten to preview PNG
    GET    /api/layers/list                                      List all stacks
"""

import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from latent_layer_stack import LatentLayerStack

router = APIRouter(prefix="/api/layers", tags=["layers"])

# Injected at startup by fuk_web_server.py
_backend: Any = None


def initialize_layer_endpoints(backend, cache_root: Path):
    """Called by fuk_web_server.py after backend and project system are ready."""
    global _backend
    _backend = backend
    print("[LAYERS] Endpoint system initialized", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_dir(stack_id: str) -> Path:
    """Resolve stack_id (e.g. 'img_edit_001') to its directory in the project cache."""
    from project_endpoints import get_project_cache_dir
    return get_project_cache_dir() / stack_id


def _load_stack(stack_id: str) -> LatentLayerStack:
    stack_dir = _stack_dir(stack_id)
    if not stack_dir.exists():
        raise HTTPException(status_code=404, detail=f"Stack not found: {stack_id}")
    return LatentLayerStack.load(stack_dir)


def _borrow_vae(device: str = "cuda"):
    """
    Get VAE from the first available cached Qwen pipeline.
    The Qwen image pipeline must already be loaded (it will be — we just ran a gen).
    """
    if not _backend:
        raise RuntimeError("Backend not initialized")

    for cache_key, pipe in _backend.pipelines.items():
        if hasattr(pipe, "vae") and pipe.vae is not None:
            return pipe.vae

    raise HTTPException(
        status_code=503,
        detail="No cached pipeline with VAE found. Run an image generation first."
    )


def _resolve_image_url_to_path(image_url: str) -> Path:
    """Resolve a /api/project/cache/... URL to a filesystem path."""
    from project_endpoints import get_project_cache_dir
    cache_dir = get_project_cache_dir()

    url = image_url.lstrip('/')
    if url.startswith('api/project/cache/'):
        rel = url.removeprefix('api/project/cache/')
        return cache_dir.parent / rel  # cache_dir is project-specific, rel includes project dir
    return Path(image_url)


def _find_latent_in_dir(gen_dir: Path) -> Optional[Path]:
    """Find a .pt latent file in a generation directory.
    
    Checks both top-level and the latents/ subdirectory, since
    the pipeline saves latents to gen_dir/latents/generated.latent.pt
    """
    # Search locations: top-level first, then latents/ subdir
    search_dirs = [gen_dir]
    latents_subdir = gen_dir / "latents"
    if latents_subdir.is_dir():
        search_dirs.append(latents_subdir)
    
    # Prefer .latent.pt, fall back to any .pt
    for search_dir in search_dirs:
        for pattern in ["*.latent.pt", "*.pt"]:
            matches = list(search_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _vae_encode_image(image_path: Path, vae, device: str = "cuda") -> torch.Tensor:
    """
    Encode an arbitrary image through the VAE to get a latent tensor.
    Used when the source image has no pre-saved latent (e.g. imported image).
    """
    from PIL import Image

    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    # Round to 16-multiples — Qwen requires this, and the VAE is happy with
    # any multiple of 8 (16 is always a multiple of 8). Using 8-rounding here
    # and letting the pipeline ShapeChecker ceil-to-16 produces a base latent
    # at a different resolution than subsequent edits, breaking delta math.
    w = ((w + 15) // 16) * 16
    h = ((h + 15) // 16) * 16
    if w == 0 or h == 0:
        raise ValueError(f"Image too small after rounding: {img.size}")
    img = img.resize((w, h), Image.LANCZOS)

    # Convert to tensor: [1, C, H, W] in range [-1, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0

    with torch.inference_mode():
        latent = vae.encode(tensor.to(device))

    return latent.cpu()


# ---------------------------------------------------------------------------
# Public API for web server integration
# ---------------------------------------------------------------------------

def register_layer_from_generation(
    stack_id: str,
    layer_name: str,
    gen_dir: Path,
    latent_path: Optional[str],
    preview_path: Optional[Path],
    params: dict,
) -> dict:
    """
    Called by run_image_generation after a layer edit completes.

    Loads the stack, adds the generation's latent as a new layer,
    and returns the updated stack manifest.

    Args:
        stack_id:      e.g. "img_edit_001"
        layer_name:    Human-readable name for the layer
        gen_dir:       The generation's output directory (already inside the stack dir)
        latent_path:   Path to the .pt latent produced by generation
        preview_path:  Path to generated.png for UI thumbnail
        params:        Generation params for version history

    Returns:
        Updated stack manifest dict
    """
    stack = LatentLayerStack.load(_stack_dir(stack_id))

    if not latent_path or not Path(latent_path).exists():
        raise RuntimeError(f"Layer registration failed: no latent at {latent_path}")

    from latent_layer_stack import unwrap_latent
    edit_latent = unwrap_latent(torch.load(latent_path, map_location="cpu"))

    layer_id = stack.add_layer(
        name=layer_name,
        edit_latent=edit_latent,
        params=params,
        preview_path=str(preview_path) if preview_path else None,
    )

    print(f"[LAYERS] Registered layer '{layer_name}' (id={layer_id}) in stack {stack_id}", flush=True)
    return stack.to_dict()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class InitStackRequest(BaseModel):
    source_image_url: str   # e.g. /api/project/cache/fuk_shot11_260213/img_gen_054/generated.png
    name: Optional[str] = None


class ToggleLayerRequest(BaseModel):
    enabled: bool


class SwitchVersionRequest(BaseModel):
    version: int


class ReorderRequest(BaseModel):
    layer_ids: list


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/list")
async def list_stacks():
    """List all existing layer stacks (img_edit_* dirs with a stack.json)."""
    from project_endpoints import get_project_cache_dir
    import json

    project_cache = get_project_cache_dir()
    stacks = []

    for entry in sorted(project_cache.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("img_edit_"):
            continue
        manifest_path = entry / "stack.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                m = json.load(f)
            stacks.append({
                "stack_id":        entry.name,  # e.g. "img_edit_001"
                "created_at":      m.get("created_at"),
                "updated_at":      m.get("updated_at"),
                "layer_count":     len(m.get("layers", [])),
                "base_image_path": m.get("base_image_path"),
            })

    return {"stacks": stacks}


@router.post("/init")
async def init_stack(req: InitStackRequest):
    """
    Initialize a new layer stack from a source image.

    Creates an img_edit_XXX directory in the project cache, copies or
    VAE-encodes the base latent, and saves the source image as a preview.

    Steps:
        1. Resolve source image URL → filesystem path
        2. Find .pt latent in the source gen dir (or VAE-encode the image)
        3. Create img_edit_XXX dir via get_generation_output_dir
        4. Copy base latent + preview image
        5. Initialize stack.json manifest

    Returns:
        { success, stack_id, stack }
    """
    from project_endpoints import get_generation_output_dir

    # --- Resolve source image ---
    source_path = _resolve_image_url_to_path(req.source_image_url)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Source image not found: {source_path}")

    source_gen_dir = source_path.parent

    # --- Get or create base latent ---
    latent_file = _find_latent_in_dir(source_gen_dir)

    if latent_file:
        print(f"[LAYERS] Found existing latent: {latent_file.name}", flush=True)
        from latent_layer_stack import unwrap_latent
        base_latent = unwrap_latent(torch.load(latent_file, map_location="cpu"))
    else:
        # No latent on disk — VAE encode the image
        print(f"[LAYERS] No latent found in {source_gen_dir.name}, VAE-encoding...", flush=True)
        try:
            vae = _borrow_vae()
            base_latent = _vae_encode_image(source_path, vae)
            print(f"[LAYERS] VAE encode complete: {base_latent.shape}", flush=True)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to VAE-encode source image: {e}"
            )

    # --- Create stack directory ---
    stack_dir = get_generation_output_dir("img_edit")
    stack_id = stack_dir.name   # e.g. "img_edit_001"

    # --- Copy source image as base preview ---
    base_preview = stack_dir / "base_preview.png"
    shutil.copy2(source_path, base_preview)

    # --- Create the stack ---
    stack = LatentLayerStack.create(
        stack_dir=stack_dir,
        base_latent=base_latent,
        base_image_path=base_preview,
    )

    # Store source reference in manifest for traceability
    manifest = stack.to_dict()
    manifest["source_image_url"] = req.source_image_url
    manifest["source_gen_dir"] = str(source_gen_dir)
    stack._manifest.update({
        "source_image_url": req.source_image_url,
        "source_gen_dir": str(source_gen_dir),
    })
    stack._save_manifest()

    print(f"[LAYERS] Stack initialized: {stack_id} (source: {source_gen_dir.name})", flush=True)

    return {
        "success": True,
        "stack_id": stack_id,
        "stack": stack.to_dict(),
        "base_preview_url": f"/api/project/cache/{stack_dir.parent.name}/{stack_id}/base_preview.png",
    }


@router.get("/{stack_id}")
async def get_stack(stack_id: str):
    """Get the full stack manifest."""
    stack = _load_stack(stack_id)
    return {"stack": stack.to_dict()}


@router.patch("/{stack_id}/layer/{layer_id}/toggle")
async def toggle_layer(stack_id: str, layer_id: str, req: ToggleLayerRequest):
    """Enable or disable a layer."""
    stack = _load_stack(stack_id)
    stack.set_enabled(layer_id, req.enabled)
    return {"success": True, "stack": stack.to_dict()}


@router.patch("/{stack_id}/layer/{layer_id}/version")
async def switch_version(stack_id: str, layer_id: str, req: SwitchVersionRequest):
    """Switch a layer to a different version."""
    stack = _load_stack(stack_id)
    stack.set_active_version(layer_id, req.version)
    return {"success": True, "stack": stack.to_dict()}


@router.post("/{stack_id}/reorder")
async def reorder_layers(stack_id: str, req: ReorderRequest):
    """Reorder layers by providing the full ordered list of layer_ids."""
    stack = _load_stack(stack_id)
    stack.reorder_layers(req.layer_ids)
    return {"success": True, "stack": stack.to_dict()}


@router.delete("/{stack_id}/layer/{layer_id}")
async def remove_layer(stack_id: str, layer_id: str):
    """Remove a layer from the stack."""
    stack = _load_stack(stack_id)
    stack.remove_layer(layer_id)
    return {"success": True, "stack": stack.to_dict()}


@router.post("/{stack_id}/flatten")
async def flatten_stack(stack_id: str):
    """
    Flatten all enabled layers and return a preview PNG path.

    Borrows the VAE from the already-cached Qwen pipeline (no extra VRAM cost).
    Preview is saved to stack_dir/flatten_preview.png.
    """
    stack = _load_stack(stack_id)
    stack_dir = _stack_dir(stack_id)

    vae = _borrow_vae()

    preview_path = stack_dir / "flatten_preview.png"
    stack.save_flatten_preview(vae, preview_path)

    # Return a cache-relative URL the frontend can load
    from project_endpoints import get_project_cache_dir
    cache_dir = get_project_cache_dir()
    try:
        rel_path = preview_path.relative_to(cache_dir)
        # Need project dir name in the URL
        project_dir_name = cache_dir.name
        preview_url = f"/api/project/cache/{project_dir_name}/{rel_path}"
    except ValueError:
        preview_url = str(preview_path)

    return {
        "success":     True,
        "preview_url": preview_url,
        "stack":       stack.to_dict(),
    }


def setup_layer_routes(app):
    """Register layer stack routes on the FastAPI app."""
    app.include_router(router)