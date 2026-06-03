"""
HD Proxy Conform Endpoint

Two-pass video workflow conform step. Takes a completed i2v "proxy" generation
(low/medium-res), and re-runs Wan VACE at the original source-still resolution
using the proxy video as the motion control and the source still as the
appearance anchor.

See: docs/HD_PROXY_VIDEO_SYSTEM.md
"""

from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime
import json
import uuid


class HDConformRequest(BaseModel):
    source_id: str                          # gen entry id (rel path from cache root)
    # Detail strength = denoising_strength on the latent-init path.
    # 0.20 ≈ texture-only pass, 0.30 default, 0.50 = noticeable motion/detail drift.
    denoising_strength: float = 0.30
    vram_preset: Optional[str] = None       # inherits backend default if None


def setup_hd_proxy_routes(
    app,
    *,
    get_cache_root: Callable[[], Path],
    get_project_cache_dir: Callable[[], Path],
    active_generations: dict,
    video_request_cls: type,                # VideoGenerationRequest class
    run_video_generation: Callable[..., Any],
    log,
):
    """Register the /api/video/hd_conform endpoint."""

    @app.post("/api/video/hd_conform")
    async def hd_conform(request: HDConformRequest, background_tasks: BackgroundTasks):
        # The history endpoint sets each entry's `id` to its relative path from
        # the cache root (e.g. "fuk_shot18_260524/video_012"). Resolve from the
        # cache root for full-path ids; fall back to the active project cache
        # for legacy bare-name ids.
        cache_root = get_cache_root()
        if not cache_root:
            raise HTTPException(status_code=400, detail="Cache root not initialized")

        source_id = request.source_id.strip().lstrip("/")
        candidates = [cache_root / source_id]
        if "/" not in source_id:
            try:
                candidates.append(get_project_cache_dir() / source_id)
            except RuntimeError:
                pass

        source_dir = next((p for p in candidates if p.is_dir()), None)
        if source_dir is None:
            raise HTTPException(
                status_code=404,
                detail=f"Source generation not found: {request.source_id}",
            )

        meta_path = source_dir / "metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Source metadata.json missing")
        with open(meta_path) as f:
            metadata = json.load(f)

        source_still = source_dir / "control" / "start_image.png"
        proxy_video = source_dir / "generated.mp4"
        if not source_still.exists():
            raise HTTPException(
                status_code=400,
                detail="Source still not found (control/start_image.png)",
            )
        if not proxy_video.exists():
            raise HTTPException(
                status_code=400, detail="Proxy video not found (generated.mp4)"
            )

        # Inherit shot parameters from the proxy gen.
        # IMPORTANT: metadata["image_size"] is the *proxy* generation size, not
        # the original source still — those diverge when the user generated at
        # 50%/25%. Read real dimensions from the saved hi-res still instead.
        try:
            from PIL import Image
            with Image.open(source_still) as im:
                width, height = im.size
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to read source still dimensions: {e}"
            )
        video_length = int(metadata.get("video_length", 81))

        prompt_field = metadata.get("prompt")
        if isinstance(prompt_field, dict):
            prompt = prompt_field.get("original", "") or prompt_field.get("enhanced", "")
        else:
            prompt = prompt_field or ""

        seed = metadata.get("seed")
        negative_prompt = metadata.get("negative_prompt") or ""

        # Build a video generation request targeting the VACE conform task.
        # Reuses the existing video generation pipeline end-to-end — no new
        # pipeline code is needed; this is parameter wiring.
        vreq = video_request_cls(
            prompt=prompt,
            task="wan_vace_a14b",
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            video_length=video_length,
            seed=seed,
            image_path=str(source_still),
            control_path=str(proxy_video),
            # The proxy video also goes in as the init latent: pipeline
            # VAE-encodes it and the scheduler partial-noises per
            # denoising_strength, so generation denoises *from* the proxy.
            input_video_path=str(proxy_video),
            denoising_strength=float(request.denoising_strength),
            vram_preset=request.vram_preset,
        )

        generation_id = str(uuid.uuid4())
        active_generations[generation_id] = {
            "id": generation_id,
            "type": "video",
            "subtype": "hd_conform",
            "parent_id": request.source_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "request": vreq.dict(),
            "created_at": datetime.now().isoformat(),
        }
        background_tasks.add_task(run_video_generation, generation_id, vreq)

        log.info(
            "HDConform",
            f"Queued conform from {request.source_id} "
            f"({width}x{height}, {video_length}f, denoise={request.denoising_strength})",
        )
        return {
            "generation_id": generation_id,
            "status": "queued",
            "message": "HD conform queued",
        }
