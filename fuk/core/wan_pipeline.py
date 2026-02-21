"""
Wan Pipeline Runner for FUK

Handles all Wan-family video generation:
  - wan_i2v_a14b (dual DiT image-to-video)
  - wan_vace_fun_a14b (VACE control video + reference)
  - (future Wan variants go here)

Wan-specific features: sigma_shift, sliding window, motion_bucket_id,
VACE video data loading, tiled inference, dual-DiT boundary switching.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from pipeline_base import PipelineRunner, _log


class WanPipelineRunner(PipelineRunner):
    """
    Runner for all Wan video generation pipelines.
    """

    pipeline_family = "video"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        task: str = "wan_t2v_14b",
        # Size
        width: int = None,
        height: int = None,
        video_length: int = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        denoising_strength: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        # Control inputs
        image_path: Optional[Path] = None,
        end_image_path: Optional[Path] = None,
        control_path: Optional[Path] = None,
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        loras: Optional[List[Dict[str, Any]]] = None,
        # Progress
        progress_callback=None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video using a Wan pipeline."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(task)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # --- Resolve parameters against defaults ---
        defaults = self.get_family_defaults()
        width = width or defaults.get("width") or 832
        height = height or defaults.get("height") or 480
        num_frames = video_length or defaults.get("video_length", 81)
        num_steps = steps or infer_steps or defaults.get("steps", 50)
        effective_cfg = cfg_scale or guidance_scale or defaults.get("cfg_scale", 6.0)
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = (denoising_strength if denoising_strength is not None
                   else defaults.get("denoising_strength", 1.0))

        # Wan-specific params
        sigma_shift = (kwargs.get("sigma_shift") if kwargs.get("sigma_shift") is not None
                       else defaults.get("sigma_shift", 5.0))
        motion_bucket_id = (kwargs.get("motion_bucket_id") if "motion_bucket_id" in kwargs
                            else defaults.get("motion_bucket_id"))
        sliding_window_size = defaults.get("sliding_window_size")
        sliding_window_stride = defaults.get("sliding_window_stride")

        # --- Logging ---
        self.log_generation_header("VIDEO GENERATION", model_type, entry, {
            "prompt": prompt,
            "task": f"{task} → {model_type}",
            "size": f"{width}x{height}",
            "frames": num_frames,
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "denoising_strength": denoise,
            "seed": seed,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "input_image": image_path,
            "control_path": control_path,
            "pipeline_kwargs": pipe_defaults if pipe_defaults else None,
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [f"{l.get('name','?')} (α={l.get('alpha', 1.0)})" for l in (loras or [])],
        })

        # --- Pipeline + LoRA ---
        if progress_callback:
            progress_callback("loading_models", 0, 1)

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)

        if progress_callback:
            progress_callback("generating", 0, num_steps)

        # --- Build pipe() kwargs ---
        pipe_kwargs = dict(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=effective_cfg,
            denoising_strength=denoise,
            sigma_shift=sigma_shift,
        )

        # Optional Wan params — only pass when explicitly set
        if sliding_window_size is not None:
            pipe_kwargs["sliding_window_size"] = sliding_window_size
        if sliding_window_stride is not None:
            pipe_kwargs["sliding_window_stride"] = sliding_window_stride
        if motion_bucket_id is not None:
            pipe_kwargs["motion_bucket_id"] = motion_bucket_id

        # Negative prompt
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # --- Map semantic inputs → model-specific params ---
        semantic_inputs = {}
        if image_path:
            semantic_inputs["reference_image"] = image_path
        if end_image_path:                              # <-- ADD
            semantic_inputs["end_image"] = end_image_path
        if control_path:
            semantic_inputs["control_input"] = control_path

        mapped_inputs = self.map_inputs(model_type, semantic_inputs, width, height)
        pipe_kwargs.update(mapped_inputs)

        # Tiled inference
        if "tiled" in supports and "tiled" not in pipe_defaults:
            pipe_kwargs["tiled"] = True

        # Merge pipeline_kwargs from models.json
        pipe_kwargs.update(pipe_defaults)

        # --- Latent capture ---
        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent)

        # --- Generate ---
        try:
            with torch.inference_mode():
                video = pipe(**pipe_kwargs)

            if progress_callback:
                progress_callback("saving", 0, 1)

            from diffsynth.utils.data import save_video
            save_video(video, str(output_path), fps=24, quality=8)

            elapsed = time.time() - start_time
            _log(self.log_prefix, f"Video saved: {output_path} ({elapsed:.1f}s)", "success")

            if progress_callback:
                progress_callback("complete", 1, 1)

            return self.success_result(
                output_path=output_path,
                latent_path=latent_path,
                seed=seed,
                elapsed=elapsed,
                params={
                    "prompt": prompt, "seed": seed,
                    "steps": num_steps, "size": f"{width}x{height}",
                    "frames": num_frames, "cfg_scale": effective_cfg,
                    "denoising_strength": denoise, "sigma_shift": sigma_shift,
                },
            )
        except Exception as e:
            _log(self.log_prefix, f"Video generation failed: {e}", "error")
            raise
        finally:
            if cleanup_hook:
                cleanup_hook()

    # ------------------------------------------------------------------
    # Wan-specific input mapping overrides
    # ------------------------------------------------------------------

    def _map_single_input(self, semantic_name, model_param, value, width, height):
        """Extend base mapping with Wan-specific types."""

        if model_param == "vace_video":
            vace = self._load_video_data(value, height, width)
            if vace:
                _log(self.log_prefix, f"  Mapped {semantic_name} → vace_video")
                return {"vace_video": vace}
            return None

        if model_param == "vace_reference_image":
            img = self.load_image(value, width, height)
            if img:
                _log(self.log_prefix, f"  Mapped {semantic_name} → vace_reference_image")
                return {"vace_reference_image": img}
            return None
        if model_param == "end_image":
            img = self.load_image(value, width, height)
            if img:
                _log(self.log_prefix, f"  Mapped {semantic_name} → end_image")
                return {"end_image": img}
            return None

        # Fall back to base implementation for common types
        return super()._map_single_input(semantic_name, model_param, value, width, height)

    def _load_video_data(self, source, height: int, width: int):
        """Load a video source into DiffSynth VideoData for VACE."""
        from diffsynth.utils.data import VideoData

        if hasattr(source, "raw_data"):
            return source

        p = Path(str(source))
        if p.exists() and p.suffix in (".mp4", ".avi", ".mov", ".mkv"):
            return VideoData(str(p), height=height, width=width)
        if p.is_dir():
            return VideoData(str(p), height=height, width=width)

        _log(self.log_prefix, f"Could not load video data from: {source}", "warning")
        return None
