"""
Krea-2 Pipeline Runner for FUK

Krea-2 is a straight text-to-image model — no edit images, no control inputs,
no denoising_strength. That is why it gets its own runner rather than riding
the Qwen one: QwenPipelineRunner passes a pile of conditioning kwargs that
Krea2Pipeline.__call__ does not accept.

Two variants, registered as separate models because their sampling defaults are
not close to each other:

  krea2_raw     52 steps at cfg 4.5 — the quality option
  krea2_turbo    8 steps at cfg 1.0 with mu=1.15 — distilled, roughly 6x faster

They share the Qwen3-VL-4B text encoder, and the VAE is Qwen-Image's, which is
already on disk for anyone running the Qwen models — so the second variant costs
only its own 24.5GB DiT.

Krea2Pipeline exposes pipe.vae and drives its loop through self.step(), so it
gets latent capture and the live preview/cancel hook for free from
PipelineRunner. At 52 steps the cancel path is worth more here than anywhere
else in FUK.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from pipeline_base import PipelineRunner, _log, GenerationCancelled


class Krea2PipelineRunner(PipelineRunner):
    """Runner for Krea-2 text-to-image generation."""

    pipeline_family = "krea2"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        model: str = "krea2_raw",
        # Size
        width: int = None,
        height: int = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        # Flow-match shift. Krea-2 Turbo needs it; Raw computes one from the
        # resolution when this is None.
        mu: Optional[float] = None,
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        loras: Optional[List[Dict[str, Any]]] = None,
        # Live preview / cancellation
        preview_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate an image using a Krea-2 pipeline."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(model)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        defaults = self.get_family_defaults()
        width = width or defaults.get("width", 1024)
        height = height or defaults.get("height", 1024)
        # Steps and CFG differ sharply between Raw and Turbo, so the per-model
        # pipeline_kwargs win over the family defaults when the caller says
        # nothing — otherwise Turbo would run 52 steps at cfg 4.5 and look wrong.
        num_steps = (steps or infer_steps
                     or pipe_defaults.pop("num_inference_steps", None)
                     or defaults.get("steps", 52))
        effective_cfg = (cfg_scale if cfg_scale is not None else
                         guidance_scale if guidance_scale is not None else
                         pipe_defaults.pop("cfg_scale", None))
        if effective_cfg is None:
            effective_cfg = defaults.get("cfg_scale", 4.5)
        effective_mu = mu if mu is not None else pipe_defaults.pop("mu", None)
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")

        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else "(none)",
            "size": f"{width}x{height}",
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "mu": effective_mu if effective_mu is not None else "auto (from resolution)",
            "seed": seed,
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [f"{l.get('name','?')} (α={l.get('alpha', 1.0)})" for l in (loras or [])],
        }
        self.log_generation_header("KREA-2 GENERATION", model_type, entry, log_params)

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)

        pipe_kwargs = dict(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            cfg_scale=effective_cfg,
        )
        if effective_mu is not None:
            pipe_kwargs["mu"] = effective_mu
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        pipe_kwargs.update(pipe_defaults)

        # Grab the un-hooked decode before latent capture wraps it, so previews
        # decode through the original and capture stays unaffected.
        original_vae_decode = pipe.vae.decode
        preview_cleanup = None
        if preview_callback is not None or cancel_check is not None:
            preview_cleanup = self._install_preview_hook(
                pipe, preview_callback, cancel_check, num_steps, original_vae_decode,
            )

        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent)

        try:
            with torch.inference_mode():
                image = pipe(**pipe_kwargs)
            image.save(output_path)

            elapsed = time.time() - start_time
            _log(self.log_prefix, f"Image saved: {output_path} ({elapsed:.1f}s)", "success")

            return self.success_result(
                output_path=output_path,
                latent_path=latent_path,
                seed=seed,
                elapsed=elapsed,
                params={
                    "prompt": prompt, "seed": seed, "steps": num_steps,
                    "size": f"{width}x{height}", "cfg_scale": effective_cfg,
                    "mu": effective_mu,
                },
            )
        except GenerationCancelled:
            _log(self.log_prefix, "Generation cancelled", "warning")
            # pipe.__call__ never reached its own load_models_to_device([]) —
            # without this the promoted weights stay on the GPU for good.
            self.release_pipeline_vram(pipe)
            raise
        except Exception as e:
            _log(self.log_prefix, f"Image generation failed: {e}", "error")
            self.release_pipeline_vram(pipe)
            raise
        finally:
            if preview_cleanup:
                preview_cleanup()
            if cleanup_hook:
                cleanup_hook()
