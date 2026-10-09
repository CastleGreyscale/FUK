"""
FLUX.2 Pipeline Runner for FUK

Handles FLUX.2 image generation:
  - flux2_dev (FLUX.2-dev, best quality)

FLUX.2 natively supports multi-image editing via its edit_image parameter,
mirroring the Qwen edit workflow. Multiple input images are passed directly
to the pipeline; dimension inheritance from the first source image applies
the same way as Qwen edit.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from pipeline_base import PipelineRunner, _log


class Flux2PipelineRunner(PipelineRunner):
    """Runner for all FLUX.2 image generation pipelines."""

    pipeline_family = "flux2"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        model: str = "flux2_dev",
        # Size
        width: int = None,
        height: int = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        denoising_strength: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        embedded_guidance: Optional[float] = None,
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        loras: Optional[List[Dict[str, Any]]] = None,
        # Edit inputs (multiple images, same as Qwen edit)
        control_image: Optional[Union[Path, List[Path]]] = None,
        # Inpainting. Distinct from edit_image: input_image is the picture being
        # repainted and supplies the latents the mask blends against, where
        # edit_image is reference material. Both can be present.
        input_image: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        inpaint_blur_size: Optional[int] = None,
        inpaint_blur_sigma: Optional[float] = None,
        # Soft mask edge in pixels. Drives the latent blur when the two blur
        # values above are not given, and the feather of the pixel composite.
        inpaint_feather: Optional[int] = None,
        # Paste the original pixels back outside the mask after decoding.
        inpaint_preserve: bool = True,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate an image using a FLUX.2 pipeline."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(model)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # Resolve parameters against defaults
        defaults = self.get_family_defaults()
        width = width or defaults.get("width", 1024)
        height = height or defaults.get("height", 1024)
        num_steps = steps or infer_steps or defaults.get("steps", 30)
        # cfg_scale stays at 1.0 (CFG off) — Flux2's guidance is embedded_guidance
        effective_cfg = cfg_scale or defaults.get("cfg_scale", 1.0)
        # guidance_scale from the UI maps to embedded_guidance (the meaningful Flux2 guidance param)
        effective_embedded_guidance = (embedded_guidance if embedded_guidance is not None
                                       else guidance_scale
                                       if guidance_scale is not None
                                       else defaults.get("embedded_guidance", 4.0))
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = (denoising_strength if denoising_strength is not None
                   else defaults.get("denoising_strength", 1.0))

        # Inherit dimensions from first source image (matches Qwen edit behaviour).
        # When inpainting, the picture being repainted wins: its latents have to
        # line up with the noise, where an edit image is only reference material.
        inpainting = "inpaint" in supports and bool(input_image)
        dim_source = input_image if inpainting else control_image
        if dim_source:
            source_path = dim_source[0] if isinstance(dim_source, list) else dim_source
            source_path = Path(str(source_path))
            if source_path.exists() and source_path.is_file():
                try:
                    from PIL import Image as _PILImage
                    with _PILImage.open(str(source_path)) as src:
                        src_w, src_h = src.size
                    src_w = ((src_w + 15) // 16) * 16
                    src_h = ((src_h + 15) // 16) * 16
                    if src_w > 0 and src_h > 0:
                        width, height = src_w, src_h
                        _log(self.log_prefix, f"Inherited dimensions from source: {width}x{height}")
                except Exception as e:
                    _log(self.log_prefix, f"Could not read source image dimensions: {e}", "warning")

        # Logging
        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else "(none)",
            "size": f"{width}x{height}",
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "embedded_guidance": effective_embedded_guidance,
            "denoising_strength": denoise,
            "seed": seed,
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [f"{l.get('name','?')} (α={l.get('alpha', 1.0)})" for l in (loras or [])],
        }
        self.log_generation_header("FLUX2 GENERATION", model_type, entry, log_params)

        # Pipeline + LoRA
        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)

        # Build pipe() kwargs
        pipe_kwargs = dict(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            cfg_scale=effective_cfg,
            embedded_guidance=effective_embedded_guidance,
            denoising_strength=denoise,
        )

        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # Map semantic inputs → edit_image via parameter_map in models.json
        semantic_inputs = {}
        if control_image:
            semantic_inputs["edit_targets"] = control_image

        mapped_inputs = self.map_inputs(model_type, semantic_inputs, width, height)
        pipe_kwargs.update(mapped_inputs)

        # We pre-resize images ourselves, so disable FLUX.2's internal auto-resize
        if "edit_image" in pipe_kwargs:
            pipe_kwargs["edit_image_auto_resize"] = False

        # Inpainting. FLUX.2 resizes the mask to its 16 px latent grid.
        inpaint_base = inpaint_mask = None
        if inpainting:
            inpaint_kwargs, inpaint_base, inpaint_mask = self.build_inpaint_inputs(
                input_image, mask_path, width, height, latent_scale=16,
                feather=inpaint_feather,
                blur_size=inpaint_blur_size, blur_sigma=inpaint_blur_sigma)
            pipe_kwargs.update(inpaint_kwargs)

        # Merge pipeline_kwargs from models.json
        pipe_kwargs.update(pipe_defaults)

        # Latent capture
        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent, model_type)

        try:
            with torch.inference_mode():
                image = pipe(**pipe_kwargs)
            if inpaint_mask is not None and inpaint_preserve:
                image = self.composite_inpaint(
                    inpaint_base, image, inpaint_mask, feather=inpaint_feather or 0)
            image.save(output_path)

            elapsed = time.time() - start_time
            _log(self.log_prefix, f"Image saved: {output_path} ({elapsed:.1f}s)", "success")

            return self.success_result(
                output_path=output_path,
                latent_path=latent_path,
                seed=seed,
                elapsed=elapsed,
                params={
                    "prompt": prompt, "seed": seed,
                    "steps": num_steps, "size": f"{width}x{height}",
                    "cfg_scale": effective_cfg,
                    "embedded_guidance": effective_embedded_guidance,
                    "denoising_strength": denoise,
                },
            )
        except Exception as e:
            _log(self.log_prefix, f"Image generation failed: {e}", "error")
            # pipe.__call__ never reached its own load_models_to_device([]) —
            # without this the promoted weights stay on the GPU for good.
            self.release_pipeline_vram(pipe)
            raise
        finally:
            if cleanup_hook:
                cleanup_hook()
