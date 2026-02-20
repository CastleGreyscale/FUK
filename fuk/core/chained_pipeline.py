"""
Chained Pipeline Runner — SKELETON / EXAMPLE

Demonstrates how the hub-and-runner architecture supports
multi-model workflows that chain different pipelines together.

Example: Generate image with Qwen → use as input for Wan i2v,
with latent-space operations in between.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from pipeline_base import PipelineRunner, _log


class ChainedPipelineRunner(PipelineRunner):
    """
    Runner that chains multiple pipeline families together.
    
    Because runners hold a reference to the hub, they can grab
    any pipeline — even from different families — within a single
    generate() call. The hub handles VRAM eviction between families
    automatically.
    
    This is a skeleton showing the pattern. Real implementations
    would add proper parameter handling, error recovery, and
    intermediate result management.
    """

    pipeline_family = "chain"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        # Chain-specific params
        image_model: str = "qwen_image",
        video_model: str = "wan_i2v_a14b",
        # Generation params shared across stages
        seed: Optional[int] = None,
        width: int = 832,
        height: int = 480,
        video_length: int = 81,
        # LoRA (applied to image stage)
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        loras: Optional[List[Dict[str, Any]]] = None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Progress
        progress_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chain: text → image → video
        
        Stage 1: Generate a hero image with Qwen
        Stage 2: (Optional) Latent-space transforms
        Stage 3: Animate with Wan i2v
        """
        start_time = time.time()
        output_path = Path(output_path)
        intermediate_dir = output_path.parent / "chain_intermediates"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # =============================================
        # Stage 1: Image generation
        # =============================================
        if progress_callback:
            progress_callback("stage_1_image", 0, 3)

        _log(self.log_prefix, "=" * 60)
        _log(self.log_prefix, "CHAIN Stage 1: Image Generation")
        _log(self.log_prefix, "=" * 60)

        image_path = intermediate_dir / "stage1_image.png"

        # Grab the Qwen pipeline from the hub
        image_pipe = self.get_pipeline(image_model, vram_preset=vram_preset)
        cache_key = self._cache_key(image_model, vram_preset)
        self.apply_loras(image_pipe, cache_key, lora, lora_multiplier, loras)

        # Generate image (simplified — real version would use full param set)
        with torch.inference_mode():
            image = image_pipe(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_inference_steps=28,
                cfg_scale=5.0,
            )
        image.save(image_path)
        _log(self.log_prefix, f"Stage 1 complete: {image_path}", "success")

        # =============================================
        # Stage 2: Latent-space operations (optional)
        # =============================================
        # This is where the mad science goes:
        #   - Load the latent from stage 1
        #   - Apply transforms (camera motion, style transfer, etc.)
        #   - Feed modified latent to stage 3
        #
        # latent = self.backend.load_latent_tensor(latent_path)
        # latent = apply_camera_transform(latent, ...)
        # ...

        # =============================================
        # Stage 3: Video generation
        # =============================================
        if progress_callback:
            progress_callback("stage_3_video", 1, 3)

        _log(self.log_prefix, "=" * 60)
        _log(self.log_prefix, "CHAIN Stage 3: Video Generation")
        _log(self.log_prefix, "=" * 60)

        # Hub auto-evicts the Qwen pipeline to free VRAM for Wan
        video_pipe = self.get_pipeline(video_model, vram_preset=vram_preset)
        video_cache_key = self._cache_key(video_model, vram_preset)

        # Clear any LoRAs on the video pipe (user LoRA was for image stage)
        self.apply_loras(video_pipe, video_cache_key)

        # Load the image we generated as input
        from PIL import Image
        input_image = Image.open(str(image_path)).resize((width, height))

        with torch.inference_mode():
            video = video_pipe(
                prompt=prompt,
                seed=seed,
                input_image=input_image,
                height=height,
                width=width,
                num_frames=video_length,
                num_inference_steps=50,
                cfg_scale=6.0,
                sigma_shift=5.0,
                tiled=True,
            )

        from diffsynth.utils.data import save_video
        save_video(video, str(output_path), fps=24, quality=8)

        elapsed = time.time() - start_time
        _log(self.log_prefix, f"Chain complete: {output_path} ({elapsed:.1f}s)", "success")

        if progress_callback:
            progress_callback("complete", 3, 3)

        return self.success_result(
            output_path=output_path,
            latent_path=None,
            seed=seed,
            elapsed=elapsed,
            params={"prompt": prompt, "seed": seed, "chain": f"{image_model} → {video_model}"},
            intermediate_image=image_path,
        )
