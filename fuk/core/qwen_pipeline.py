"""
Qwen Pipeline Runner for FUK

Handles all Qwen-family image generation:
  - qwen_image (base t2i)
  - qwen_image_2512 (improved quality)
  - qwen_edit (multi-image editing)
  - qwen_control_union (in-context structural control)
  - qwen_eligen (entity-level composition via masks + prompts)

Each variant is driven by models.json — adding a new Qwen model
typically requires zero code changes here. Only override when a
variant needs genuinely different generation logic.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from pipeline_base import PipelineRunner, _log


# --- Patch DiffSynth's input-image embedder for "Detail Bias" -------------
# When input_image is None (pure t2i, qwen-edit, qwen-control all hit this
# branch since they route inputs through edit_image / context_image), the
# vendor code returns raw noise regardless of denoising_strength. The
# scheduler, meanwhile, builds a truncated sigma schedule starting at
# sigmas[0] < 1. Scaling the noise by sigmas[0] makes the starting latent
# consistent with the schedule — equivalent to ComfyUI's KSampler behavior
# on an empty latent. At denoising_strength=1.0 this is a no-op (sigmas[0]=1.0).
from diffsynth.pipelines.qwen_image import QwenImageUnit_InputImageEmbedder as _QwenInputEmbedder

_qwen_input_embedder_original = _QwenInputEmbedder.process

def _qwen_input_embedder_patched(self, pipe, input_image, noise, tiled, tile_size, tile_stride):
    if input_image is None:
        sigma_start = pipe.scheduler.sigmas[0]
        return {"latents": noise * sigma_start, "input_latents": None}
    return _qwen_input_embedder_original(self, pipe, input_image, noise, tiled, tile_size, tile_stride)

_QwenInputEmbedder.process = _qwen_input_embedder_patched


class GenerationCancelled(Exception):
    """Raised from the per-step hook to abort a generation mid-denoise."""


class QwenPipelineRunner(PipelineRunner):
    """
    Runner for all Qwen image generation pipelines.
    """

    pipeline_family = "image"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        model: str = "qwen_image",
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
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        loras: Optional[List[Dict[str, Any]]] = None,
        # Control / edit inputs
        control_image: Optional[Union[Path, List[Path]]] = None,
        context_image: Optional[Any] = None,
        # EliGen — entity masks (directory of PNGs or .psd file)
        eligen_source: Optional[Union[str, Path]] = None,
        eligen_alpha: Optional[float] = None,  # Override model LoRA strength
        # VRAM
        vram_preset: Optional[str] = None,
        # Live diffusion preview: callback(step:int, total:int, pil_image) called a
        # few times mid-denoise. Opt-in; None => zero overhead / unchanged path.
        preview_callback: Optional[Any] = None,
        # cancel_check() -> bool, polled each step; True aborts (GenerationCancelled).
        cancel_check: Optional[Any] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate an image using a Qwen pipeline."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(model)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # --- Resolve parameters against defaults ---
        defaults = self.get_family_defaults()
        width = width or defaults.get("width", 1024)
        height = height or defaults.get("height", 1024)
        num_steps = steps or infer_steps or defaults.get("steps", 28)
        effective_cfg = cfg_scale or guidance_scale or defaults.get("cfg_scale", 5.0)
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = (denoising_strength if denoising_strength is not None
                   else defaults.get("denoising_strength", 0.85))

        # --- Inherit dimensions from source image (all non-t2i modes) ---
        # For edit with multiple inputs, first image is the master.
        # Keeps latent dimensions consistent with the source — avoids
        # VAE rounding mismatches in layer stacks and downstream compositing.
        source_path = self._resolve_source_image(control_image, context_image, eligen_source)
        if source_path:
            from PIL import Image as _PILImage
            try:
                with _PILImage.open(str(source_path)) as src:
                    src_w, src_h = src.size
                # Round to nearest 16-multiple — Qwen's ShapeChecker uses ceiling-16
                # rounding internally. Pre-rounding here ensures the edit image
                # (resized by resolve_image_list) matches the noise tensor dimensions
                # exactly. Using 8-rounding here + 16-rounding inside the pipeline
                # causes a 1-token mismatch that produces ghosting / crop artifacts.
                src_w = ((src_w + 15) // 16) * 16
                src_h = ((src_h + 15) // 16) * 16
                if src_w > 0 and src_h > 0:
                    width, height = src_w, src_h
                    _log(self.log_prefix, f"Inherited dimensions from source: {width}x{height}")
            except Exception as e:
                _log(self.log_prefix, f"Could not read source image dimensions: {e}", "warning")

        # --- Logging ---
        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else "(none)",
            "size": f"{width}x{height}",
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "denoising_strength": denoise,
            "seed": seed,
            "exponential_shift_mu": kwargs.get("exponential_shift_mu"),
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [f"{l.get('name','?')} (α={l.get('alpha', 1.0)})" for l in (loras or [])],
        }
        if eligen_source:
            log_params["eligen_source"] = str(eligen_source)
        if eligen_alpha is not None:
            log_params["eligen_alpha"] = eligen_alpha
        self.log_generation_header("IMAGE GENERATION", model_type, entry, log_params)

        # --- Pipeline + LoRA ---
        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)

        # Override model LoRA alpha if requested (e.g. EliGen / control strength slider)
        if eligen_alpha is not None:
            self.backend.override_model_lora_alpha(pipe, cache_key, eligen_alpha)

        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)

        # --- Build pipe() kwargs ---
        pipe_kwargs = dict(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            cfg_scale=effective_cfg,
            denoising_strength=denoise,
        )

        # Qwen-specific: exponential shift mu (timestep control)
        exponential_shift_mu = kwargs.get("exponential_shift_mu")
        if exponential_shift_mu is not None:
            pipe_kwargs["exponential_shift_mu"] = exponential_shift_mu

        # Negative prompt (if model supports it)
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # --- Map semantic inputs → model-specific params ---
        semantic_inputs = {}
        if control_image:
            semantic_inputs["edit_targets"] = control_image
            if isinstance(control_image, list):
                semantic_inputs["control_input"] = control_image[0] if control_image else None
            else:
                semantic_inputs["control_input"] = control_image
        if context_image:
            semantic_inputs["control_input"] = context_image

        mapped_inputs = self.map_inputs(model_type, semantic_inputs, width, height)
        pipe_kwargs.update(mapped_inputs)

        # --- EliGen entity masks ---
        if eligen_source and "eligen" in supports:
            eligen_kwargs = self._load_eligen_entities(eligen_source, width, height)
            pipe_kwargs.update(eligen_kwargs)

        # Merge pipeline_kwargs from models.json
        pipe_kwargs.update(pipe_defaults)

        # --- Per-step hook: live preview and/or cancellation (opt-in) ---
        # Grab the un-hooked VAE decode BEFORE latent capture wraps it, so preview
        # decodes don't trip the "capture first decode" logic and corrupt the latent.
        preview_cleanup = None
        if preview_callback is not None or cancel_check is not None:
            preview_cleanup = self._install_preview_hook(
                pipe, preview_callback, cancel_check, num_steps,
                original_vae_decode=pipe.vae.decode,
            )

        # --- Latent capture ---
        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent)

        # --- Generate ---
        # Log what's actually going to the pipe
        if "negative_prompt" in pipe_kwargs:
            neg = pipe_kwargs["negative_prompt"]
            _log(self.log_prefix, f"  ✓ negative_prompt in pipe_kwargs ({len(neg)} chars): {neg[:60]}...", "success")
        else:
            _log(self.log_prefix, "  ✗ negative_prompt NOT in pipe_kwargs", "warning")

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
                    "prompt": prompt, "seed": seed,
                    "steps": num_steps, "size": f"{width}x{height}",
                    "cfg_scale": effective_cfg,
                    "denoising_strength": denoise,
                },
            )
        except GenerationCancelled:
            _log(self.log_prefix, "Generation cancelled at step boundary", "warning")
            raise
        except Exception as e:
            _log(self.log_prefix, f"Image generation failed: {e}", "error")
            raise
        finally:
            if preview_cleanup:
                preview_cleanup()
            if cleanup_hook:
                cleanup_hook()

    # ------------------------------------------------------------------
    # Per-step hook: live diffusion preview + cancellation
    # ------------------------------------------------------------------

    def _install_preview_hook(self, pipe, callback, cancel_check, total_steps, original_vae_decode):
        """Wrap pipe.step for mid-denoise previews and/or cancellation.

        - cancel_check(): polled each step; True raises GenerationCancelled to abort.
        - callback: a few clean x0 previews decoded via `original_vae_decode` (the
          un-hooked decode, so latent capture is unaffected).
        Restores pipe.step on cleanup. Decode failures are non-fatal.
        """
        original_step = pipe.step
        total = max(1, int(total_steps or 1))
        # Preview at ~quarter points (not the final step — that's the real output).
        marks = {max(1, round(total * f)) for f in (0.25, 0.5, 0.75)} if callback else set()
        marks.discard(total)

        def hooked_step(scheduler, **kw):
            # Check for cancellation BEFORE doing the (expensive) step work.
            if cancel_check is not None and cancel_check():
                raise GenerationCancelled()
            latents_next = original_step(scheduler, **kw)
            step_num = int(kw.get("progress_id", 0)) + 1
            if callback and step_num in marks:
                try:
                    # Decode the x0 PREDICTION (estimated clean latent), not the noisy
                    # sample — flow-match x_t stays near-noise until the end, so decoding
                    # it directly looks like static. `to_final=True` gives sample minus
                    # the velocity scaled by sigma = the current best guess of the result.
                    progress_id = int(kw.get("progress_id", 0))
                    x_t = kw.get("latents")
                    noise_pred = kw.get("noise_pred")
                    if x_t is not None and noise_pred is not None:
                        timestep = scheduler.timesteps[progress_id]
                        preview_latent = scheduler.step(noise_pred, timestep, x_t, to_final=True)
                    else:
                        preview_latent = latents_next
                    pipe.load_models_to_device(["vae"])
                    out = original_vae_decode(
                        preview_latent, device=pipe.device,
                        tiled=kw.get("tiled", False),
                        tile_size=kw.get("tile_size", 128),
                        tile_stride=kw.get("tile_stride", 64),
                    )
                    pil = pipe.vae_output_to_image(out)
                    callback(step_num, total, pil)
                    pipe.load_models_to_device(pipe.in_iteration_models)
                except Exception as e:
                    _log(self.log_prefix, f"Preview decode failed (non-fatal): {e}", "warning")
            return latents_next

        pipe.step = hooked_step
        _log(self.log_prefix,
             f"Step hook installed (preview steps {sorted(marks) or 'off'}, "
             f"cancellable={cancel_check is not None})")

        def cleanup():
            pipe.step = original_step

        return cleanup

    # ------------------------------------------------------------------
    # Source image resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_source_image(
        control_image: Optional[Union[Path, List[Path]]],
        context_image: Optional[Any],
        eligen_source: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        """
        Return the master source image path for dimension inheritance.

        Priority: control_image (first if list) → context_image → None.
        eligen_source may be a directory or .ora/.psd file; resolution
        inheritance for eligen is handled upstream in the web server.
        Returns None for pure t2i (no source images at all).
        """
        if control_image:
            if isinstance(control_image, list):
                candidate = control_image[0] if control_image else None
            else:
                candidate = control_image
            if candidate:
                p = Path(str(candidate))
                if p.exists() and p.is_file():
                    return p

        if context_image:
            p = Path(str(context_image))
            if p.exists() and p.is_file():
                return p

        return None

    # ------------------------------------------------------------------
    # EliGen support
    # ------------------------------------------------------------------

    def _load_eligen_entities(
        self,
        source: Union[str, Path],
        width: int,
        height: int,
    ) -> dict:
        """
        Load EliGen entity masks from a directory or PSD file.
        
        Returns pipe kwargs: eligen_entity_prompts + eligen_entity_masks
        """
        from eligen_loader import EliGenLoader

        source = Path(source)
        if not source.exists():
            _log(self.log_prefix, f"EliGen source not found: {source}", "warning")
            return {}

        loader = EliGenLoader()
        entities = loader.load(source, width=width, height=height)

        if not entities:
            _log(self.log_prefix, "No EliGen entities found", "warning")
            return {}

        prompts = [prompt for prompt, _ in entities]
        masks = [mask for _, mask in entities]

        _log(self.log_prefix,
             f"EliGen: {len(entities)} entities — {', '.join(prompts)}", "success")

        return {
            "eligen_entity_prompts": prompts,
            "eligen_entity_masks": masks,
        }