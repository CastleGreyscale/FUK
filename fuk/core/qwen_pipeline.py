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
        # VRAM
        vram_preset: Optional[str] = None,
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

        # --- Logging ---
        log_params = {
            "prompt": prompt,
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
        self.log_generation_header("IMAGE GENERATION", model_type, entry, log_params)

        # --- Pipeline + LoRA ---
        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
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

        # --- Latent capture ---
        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent)

        # --- Generate ---
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
        except Exception as e:
            _log(self.log_prefix, f"Image generation failed: {e}", "error")
            raise
        finally:
            if cleanup_hook:
                cleanup_hook()

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