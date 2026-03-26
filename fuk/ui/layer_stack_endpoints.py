"""
Layer Stack Endpoints

REST API for the non-destructive latent layer editing system.

All stack state is persisted to disk (stack.json) so sessions survive
server restarts. Flatten operations borrow the VAE from the already-cached
Qwen pipeline — no extra model loading.

Routes:
    POST   /api/layers/create              Create a new stack from a generation
    GET    /api/layers/{stack_id}          Get stack manifest
    POST   /api/layers/{stack_id}/add      Run a Qwen edit and add as new layer
    POST   /api/layers/{stack_id}/rerun    Re-run a layer with same (or new) params
    PATCH  /api/layers/{stack_id}/layer/{layer_id}/toggle     Enable/disable layer
    PATCH  /api/layers/{stack_id}/layer/{layer_id}/version    Switch active version
    POST   /api/layers/{stack_id}/layer/{layer_id}/reorder    Reorder all layers
    DELETE /api/layers/{stack_id}/layer/{layer_id}            Remove layer
    POST   /api/layers/{stack_id}/flatten                     Flatten to preview PNG
    GET    /api/layers/list                                   List all stacks
"""

import torch
import uuid
from pathlib import Path
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from latent_layer_stack import LatentLayerStack
from project_endpoints import get_cache_root, get_default_cache_root, get_project_cache_dir

router = APIRouter(prefix="/api/layers", tags=["layers"])

# Injected at startup by fuk_web_server.py
_backend: Any = None
_startup_cache_root: Optional[Path] = None


def initialize_layer_endpoints(backend, cache_root: Path):
    """Called by fuk_web_server.py after backend and project system are ready."""
    global _backend, _startup_cache_root
    _backend = backend
    _startup_cache_root = Path(cache_root)
    print("[LAYERS] Endpoint system initialized", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_cache_root() -> Path:
    """
    Get the current cache root, respecting project switches.

    Priority: project_endpoints current root → startup fallback.
    This mirrors the dual-resolution strategy in the /api/project/cache/
    file-serving endpoint.
    """
    root = get_cache_root()
    if root:
        return root
    if _startup_cache_root:
        return _startup_cache_root
    raise RuntimeError("Layer endpoint system not initialized")


def _resolve_cache_relative(rel_path: str) -> Path:
    """
    Resolve a cache-relative path, trying current project cache then default.

    Matches the fallback logic in project_endpoints' file-serving endpoint:
    current _cache_root first, then _default_cache_root.
    """
    current = get_cache_root()
    default = get_default_cache_root()

    if current:
        candidate = current / rel_path
        if candidate.exists():
            return candidate

    if default and default != current:
        candidate = default / rel_path
        if candidate.exists():
            return candidate

    # Return the best-guess path even if it doesn't exist yet
    # (caller will check .exists() and raise appropriate errors)
    base = current or default or _startup_cache_root
    return base / rel_path


def _stacks_root() -> Path:
    """Layer stacks live in a 'layer_stacks' subdirectory of the project cache."""
    try:
        root = get_project_cache_dir() / "layer_stacks"
    except (RuntimeError, Exception):
        root = _current_cache_root() / "layer_stacks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stack_dir(stack_id: str) -> Path:
    return _stacks_root() / stack_id


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


def _run_qwen_edit(params: dict, output_path: Path) -> dict:
    """Run a Qwen edit generation and return the result dict."""
    from qwen_pipeline import QwenPipelineRunner
    runner = QwenPipelineRunner(_backend)
    return runner.generate(output_path=output_path, **params)

def _resolve_control_image(path_or_url: str) -> Path:
    """Resolve a /api/project/cache/... URL back to a filesystem path."""
    if path_or_url.startswith('/api/project/cache/'):
        rel = path_or_url.removeprefix('/api/project/cache/')
        return _resolve_cache_relative(rel)
    return Path(path_or_url)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateStackRequest(BaseModel):
    image_url: str   # e.g. /api/project/cache/fuk_shot11_260213/img_gen_054/generated.png
    name: Optional[str] = None


class AddLayerRequest(BaseModel):
    name: str
    prompt: str
    model: str = "qwen_edit"
    control_image: Optional[str] = None   # Path to the image to edit (usually base preview)
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    denoising_strength: Optional[float] = None
    negative_prompt: Optional[str] = None
    lora: Optional[str] = None
    lora_multiplier: float = 1.0


class RerunLayerRequest(BaseModel):
    layer_id: str
    version_id: Optional[int] = None     # Which version to pull params from (default: active)
    # Override any params for the new version
    prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    denoising_strength: Optional[float] = None
    lora: Optional[str] = None
    lora_multiplier: float = 1.0


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
    """List all existing layer stacks."""
    root = _stacks_root()
    stacks = []
    for stack_dir in sorted(root.iterdir()):
        manifest_path = stack_dir / "stack.json"
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                m = json.load(f)
            stacks.append({
                "stack_id":   m.get("stack_id"),
                "created_at": m.get("created_at"),
                "updated_at": m.get("updated_at"),
                "layer_count": len(m.get("layers", [])),
                "base_image_path": m.get("base_image_path"),
            })
    return {"stacks": stacks}


@router.post("/create")
async def create_stack(req: CreateStackRequest):
    # Parse the gen_dir directly from the cache URL
    # /api/project/cache/{relative_path} → resolve via project cache roots
    image_url = req.image_url.lstrip('/')
    if not image_url.startswith('api/project/cache/'):
        raise HTTPException(status_code=400, detail="image_url must be a /api/project/cache/... URL")

    rel = image_url.removeprefix('api/project/cache/')
    image_path = _resolve_cache_relative(rel)
    gen_dir = image_path.parent   # strip /generated.png → img_gen_054 dir

    print(f"[LAYERS] create_stack: rel={rel}", flush=True)
    print(f"[LAYERS] create_stack: resolved image_path={image_path}", flush=True)
    print(f"[LAYERS] create_stack: gen_dir={gen_dir} exists={gen_dir.exists()}", flush=True)

    if not gen_dir.exists():
        raise HTTPException(status_code=404, detail=f"Generation directory not found: {gen_dir}")

    # Latents are saved in a latents/ subdirectory by pipeline_base.setup_latent_capture()
    # e.g. img_gen_060/latents/generated.latent.pt
    latent_dir = gen_dir / "latents"
    latent_files = []
    if latent_dir.exists():
        latent_files = list(latent_dir.glob("*.latent.pt")) + list(latent_dir.glob("*.pt"))
    # Fallback: check gen_dir root (legacy or alternate save locations)
    if not latent_files:
        latent_files = list(gen_dir.glob("*.latent.pt")) + list(gen_dir.glob("*.pt"))
    if not latent_files:
        raise HTTPException(status_code=400, detail=f"No latent file found in {gen_dir} or {gen_dir}/latents/. Was save_latent enabled?")
    latent_path = latent_files[0]

    preview_path = image_path if image_path.exists() else None

    base_latent = torch.load(latent_path, map_location="cpu")
    # _capture_latent_hook saves as {'latent': tensor, 'shape': ..., 'dtype': ...}
    if isinstance(base_latent, dict) and 'latent' in base_latent:
        base_latent = base_latent['latent']

    stack_id = str(uuid.uuid4())[:12]
    stack_dir = _stack_dir(stack_id)
    stack = LatentLayerStack.create(stack_dir, base_latent, preview_path)

    return {"success": True, "stack_id": stack_id, "stack": stack.to_dict()}

@router.get("/{stack_id}")
async def get_stack(stack_id: str):
    """Get the full stack manifest."""
    stack = _load_stack(stack_id)
    return {"stack": stack.to_dict()}


@router.post("/{stack_id}/add")
async def add_layer(stack_id: str, req: AddLayerRequest):
    """
    Run a Qwen edit and add the result as a new layer.

    The edit generation runs against the provided control_image (usually the
    current flattened preview). The resulting latent is diffed against the
    base to produce the stored delta.
    """
    stack = _load_stack(stack_id)
    stack_dir = _stack_dir(stack_id)

    # Read base latent dimensions so the edit generation matches exactly.
    # Latent shape is [B, C, H, W]; pixel dims = latent dims × VAE scale (8).
    base_pt = Path(stack.to_dict()["base_latent_path"])
    base_latent_tensor = torch.load(base_pt, map_location="cpu")
    base_h, base_w = base_latent_tensor.shape[-2], base_latent_tensor.shape[-1]
    pixel_h, pixel_w = base_h * 8, base_w * 8

    # Determine output path for this generation
    layer_id_tmp = str(uuid.uuid4())[:8]
    output_path = stack_dir / f"tmp_{layer_id_tmp}_edit.png"

    # Build generation params
    gen_params = {
        "prompt":             req.prompt,
        "model":              req.model,
        "width":              pixel_w,
        "height":             pixel_h,
        "seed":               req.seed,
        "steps":              req.steps,
        "cfg_scale":          req.cfg_scale,
        "denoising_strength": req.denoising_strength,
        "negative_prompt":    req.negative_prompt,
        "lora":               req.lora,
        "lora_multiplier":    req.lora_multiplier,
        "save_latent":        True,
    }
    if req.control_image:
        gen_params["control_image"] = _resolve_control_image(req.control_image)

    # Remove None values so pipeline defaults apply
    gen_params = {k: v for k, v in gen_params.items() if v is not None}

    # Run generation
    result = _run_qwen_edit(gen_params, output_path)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=f"Generation failed: {result}")

    # Load the generated latent
    latent_path = result.get("latent") or result.get("latent_path")
    if not latent_path or not Path(latent_path).exists():
        raise HTTPException(status_code=500, detail="Generation did not produce a latent file")

    edit_latent = torch.load(latent_path, map_location="cpu")
    if isinstance(edit_latent, dict) and 'latent' in edit_latent:
        edit_latent = edit_latent['latent']

    # Store params for version history
    layer_params = {
        "prompt":             req.prompt,
        "seed":               result.get("seed_used") or req.seed,
        "model":              req.model,
        "steps":              req.steps,
        "cfg_scale":          req.cfg_scale,
        "denoising_strength": req.denoising_strength,
        "lora":               req.lora,
    }

    layer_id = stack.add_layer(req.name, edit_latent, layer_params)

    return {
        "success":  True,
        "layer_id": layer_id,
        "stack":    stack.to_dict(),
    }


@router.post("/{stack_id}/rerun")
async def rerun_layer(stack_id: str, req: RerunLayerRequest):
    """
    Re-run a layer's generation (with optional param overrides) and add as a new version.

    Pulls stored params from the specified version (default: active), applies
    any overrides from the request, runs a new generation, and appends the
    result as a new version. The new version becomes active automatically.
    """
    stack = _load_stack(stack_id)
    stack_dir = _stack_dir(stack_id)

    # Read base latent dimensions for resolution matching
    base_pt = Path(stack.to_dict()["base_latent_path"])
    base_latent_tensor = torch.load(base_pt, map_location="cpu")
    base_h, base_w = base_latent_tensor.shape[-2], base_latent_tensor.shape[-1]
    pixel_h, pixel_w = base_h * 8, base_w * 8

    manifest = stack.to_dict()
    layer = next((l for l in manifest["layers"] if l["layer_id"] == req.layer_id), None)
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer not found: {req.layer_id}")

    # Pull params from specified version (or active)
    v_index = req.version_id if req.version_id is not None else layer["active_version"]
    if v_index >= len(layer["versions"]):
        raise HTTPException(status_code=400, detail=f"Version {v_index} does not exist")
    stored = layer["versions"][v_index]

    # Build params: stored values as base, request overrides on top
    gen_params = {
        "prompt":             req.prompt             or stored.get("prompt", ""),
        "model":              stored.get("model", "qwen_edit"),
        "width":              pixel_w,
        "height":             pixel_h,
        "seed":               req.seed               or stored.get("seed"),
        "steps":              req.steps              or stored.get("steps"),
        "cfg_scale":          req.cfg_scale          or stored.get("cfg_scale"),
        "denoising_strength": req.denoising_strength or stored.get("denoising_strength"),
        "lora":               req.lora               or stored.get("lora"),
        "lora_multiplier":    req.lora_multiplier,
        "save_latent":        True,
    }
    gen_params = {k: v for k, v in gen_params.items() if v is not None}

    # Output path
    v_new = len(layer["versions"])
    output_path = stack_dir / f"layer_{req.layer_id}" / f"v{v_new}_edit.png"

    result = _run_qwen_edit(gen_params, output_path)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=f"Re-run failed: {result}")

    latent_path = result.get("latent") or result.get("latent_path")
    if not latent_path or not Path(latent_path).exists():
        raise HTTPException(status_code=500, detail="Re-run did not produce a latent file")

    edit_latent = torch.load(latent_path, map_location="cpu")
    if isinstance(edit_latent, dict) and 'latent' in edit_latent:
        edit_latent = edit_latent['latent']

    new_params = {
        "prompt":             gen_params.get("prompt"),
        "seed":               result.get("seed_used") or gen_params.get("seed"),
        "model":              gen_params.get("model"),
        "steps":              gen_params.get("steps"),
        "cfg_scale":          gen_params.get("cfg_scale"),
        "denoising_strength": gen_params.get("denoising_strength"),
        "lora":               gen_params.get("lora"),
    }

    new_v = stack.add_version(req.layer_id, edit_latent, new_params)

    return {
        "success":     True,
        "new_version": new_v,
        "stack":       stack.to_dict(),
    }


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
    cache_root = _current_cache_root()
    try:
        rel_path = preview_path.relative_to(cache_root)
        preview_url = f"/api/project/cache/{rel_path}"
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