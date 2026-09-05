"""
LTX-2 Pipeline Runner for FUK

LTX-2 is a 19B audio-video model: one denoise produces both the picture and a
matching soundtrack, which is why output always goes through
write_video_audio_ltx2 rather than the plain save_video the Wan runner uses.

FUK drives it image-to-video. Text-to-video works in the underlying pipeline —
omit `image_path` and it denoises from noise — but it is not registered as a
FUK model, because in a VFX pipeline the first frame is nearly always something
you already have.

Function variants are LoRAs on one shared 35GB base, not separate checkpoints:
camera moves (dolly in/out/left/right, jib up/down, static) and the two
in-context LoRAs (Detailer, Union-Control). They are ordinary entries in
defaults_loras.json and load through the same path as every other FUK LoRA, so
adding a ninth camera move costs ~0.3-2.4GB instead of another full model.

The in-context LoRAs additionally take a driving video via `control_path`,
passed to the pipeline as `in_context_videos`:
  - Union-Control: a depth/pose/edge video steers structure.
  - Detailer: a low-detail video is re-rendered with more detail.

Three sampling modes, selected per model entry or per call:
  - one-stage (default): plain denoise at the requested size.
  - distilled: few-step, needs the transformer_distilled weights.
  - two-stage: denoise small, then spatially upscale and refine. Needs the
    stage2 distilled LoRA, declared as "stage2_lora" on the model entry.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from pipeline_base import PipelineRunner, _log

from perf_monitor import record_timing


class LTX2PipelineRunner(PipelineRunner):
    """Runner for LTX-2 audio-video generation."""

    pipeline_family = "ltx2"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        task: str = "ltx2",
        # Size
        width: int = None,
        height: int = None,
        video_length: int = None,
        frame_rate: int = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        denoising_strength: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        # Image-to-video: the first frame
        image_path: Optional[Path] = None,
        input_images_strength: float = 1.0,
        # In-context control video, for the IC-LoRAs
        control_path: Optional[Path] = None,
        in_context_downsample_factor: int = 2,
        # Sampling mode
        use_two_stage_pipeline: Optional[bool] = None,
        use_distilled_pipeline: Optional[bool] = None,
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
        """Generate an audio-video clip using LTX-2."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(task)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # --- Resolve parameters against defaults ---
        defaults = self.get_family_defaults()
        width = width or defaults.get("width") or 1536
        height = height or defaults.get("height") or 1024
        num_frames = video_length or defaults.get("video_length") or 121
        fps = frame_rate or defaults.get("frame_rate") or 24
        num_steps = steps or infer_steps or defaults.get("steps", 30)
        effective_cfg = (cfg_scale if cfg_scale is not None
                         else guidance_scale if guidance_scale is not None
                         else defaults.get("cfg_scale", 3.0))
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = (denoising_strength if denoising_strength is not None
                   else defaults.get("denoising_strength", 1.0))

        # Sampling mode: explicit arg wins, else the model entry's default.
        # Resolved before snapping because two-stage needs a coarser grid.
        two_stage = (use_two_stage_pipeline if use_two_stage_pipeline is not None
                     else pipe_defaults.pop("use_two_stage_pipeline", False))
        distilled = (use_distilled_pipeline if use_distilled_pipeline is not None
                     else pipe_defaults.pop("use_distilled_pipeline", False))
        if two_stage and distilled:
            # Two-stage already uses the distilled LoRA for its refine pass; also
            # asking for the distilled stage-1 schedule is contradictory.
            _log(self.log_prefix,
                 "both two-stage and distilled requested — using two-stage", "warning")
            distilled = False

        # LTX-2's VAE is 8x spatial / 8x temporal with a 32-pixel patch grid, and
        # num_frames must land on 8n+1. Snapping here rather than letting the
        # pipeline do it keeps the control video and first frame loaded at the
        # same dimensions the DiT will actually use.
        #
        # Two-stage halves the size before running its own /32 check, so the
        # request has to be /64 or stage 1 gets rounded up underneath us and the
        # final clip comes back a different size than was logged.
        constraints = self.get_constraints(entry)
        spatial = (constraints.get("spatial_multiple_two_stage", 64) if two_stage
                   else constraints.get("spatial_multiple", 32))
        width, height, num_frames = self.snap_to_grid(
            width, height, num_frames, entry, spatial=spatial)

        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else "(none)",
            "size": f"{width}x{height}x{num_frames} @{fps}fps",
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "denoising_strength": denoise,
            "seed": seed,
            "mode": "two-stage" if two_stage else "distilled" if distilled else "one-stage",
            "first_frame": str(image_path) if image_path else None,
            "in_context_video": str(control_path) if control_path else None,
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [f"{l.get('name','?')} (α={l.get('alpha', 1.0)})" for l in (loras or [])],
        }
        self.log_generation_header("LTX-2 GENERATION", model_type, entry, log_params)

        # Pipeline + LoRA
        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)

        pipe_kwargs = dict(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            cfg_scale=effective_cfg,
            denoising_strength=denoise,
        )
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # First frame (image-to-video)
        if image_path:
            first = self.load_image(image_path, width=width, height=height)
            if first is not None:
                pipe_kwargs["input_images"] = [first]
                pipe_kwargs["input_images_indexes"] = [0]
                pipe_kwargs["input_images_strength"] = input_images_strength
                _log(self.log_prefix, f"  First frame → input_images ({width}x{height})")

        # In-context control video, for the IC-LoRAs. The reference is encoded at
        # a reduced resolution: the LoRA name carries the factor it was trained
        # with (union-control-ref0.5 => half), and the pipeline downsamples by
        # in_context_downsample_factor on top.
        if control_path:
            ctx = self._load_control_video(
                control_path, width, height, num_frames, in_context_downsample_factor)
            if ctx is not None:
                pipe_kwargs["in_context_videos"] = [ctx]
                pipe_kwargs["in_context_downsample_factor"] = in_context_downsample_factor

        if two_stage:
            pipe_kwargs["use_two_stage_pipeline"] = True
            # Stage 2 re-uses the DiT with the distilled LoRA merged in. A user
            # LoRA left loaded would be applied on top of that and compound, so
            # drop it before the refine pass whenever one is active.
            pipe_kwargs["clear_lora_before_state_two"] = bool(lora or loras)
        if distilled:
            pipe_kwargs["use_distilled_pipeline"] = True

        pipe_kwargs.update(pipe_defaults)

        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent, model_type)

        if progress_callback:
            progress_callback("generating", 0, num_steps)

        try:
            _t_pipe = time.perf_counter()
            with torch.inference_mode():
                video, audio = pipe(**pipe_kwargs)
            _pipe_s = time.perf_counter() - _t_pipe
            _log(self.log_prefix, f"[timing] pipe() denoise+decode: {_pipe_s:.1f}s")
            record_timing(f"denoise_per_step:{model_type}:{width}x{height}x{num_frames}",
                          _pipe_s / max(1, num_steps))

            if progress_callback:
                progress_callback("saving", 0, 1)

            # LTX-2 emits picture and sound from one denoise, so they are muxed
            # together here rather than written as separate files.
            from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
            audio_sample_rate = entry.get("audio_sample_rate", 24000)
            write_video_audio_ltx2(
                video=video,
                audio=audio,
                output_path=str(output_path),
                fps=fps,
                audio_sample_rate=audio_sample_rate,
            )
            # The mux is lossy twice over (int16 then AAC), so keep the VAE's
            # float32 audio alongside it.
            wav_path = self.save_audio_sidecar(audio, output_path, audio_sample_rate)

            elapsed = time.time() - start_time
            _log(self.log_prefix,
                 f"Video{' + audio' if audio is not None else ''} saved: "
                 f"{output_path} ({elapsed:.1f}s)", "success")

            if progress_callback:
                progress_callback("complete", 1, 1)

            return self.success_result(
                output_path=output_path,
                latent_path=latent_path,
                seed=seed,
                elapsed=elapsed,
                params={
                    "prompt": prompt, "seed": seed, "steps": num_steps,
                    "size": f"{width}x{height}", "frames": num_frames, "fps": fps,
                    "cfg_scale": effective_cfg, "denoising_strength": denoise,
                    "mode": log_params["mode"],
                },
                has_audio=audio is not None,
                audio_wav=wav_path,
            )
        except Exception as e:
            _log(self.log_prefix, f"Video generation failed: {e}", "error")
            # pipe.__call__ never reached its own load_models_to_device([]) —
            # without this the promoted weights stay on the GPU for good.
            self.release_pipeline_vram(pipe)
            raise
        finally:
            if cleanup_hook:
                cleanup_hook()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_control_video(self, path, width, height, num_frames, downsample_factor):
        """Load an in-context driving video for the IC-LoRAs.

        Loaded at half the target resolution divided by the downsample factor,
        matching what the IC-LoRA examples do — the reference only has to carry
        structure, and encoding it full-size wastes VRAM for no benefit.
        """
        from diffsynth.utils.data import VideoData
        h = max(32, height // downsample_factor // 2)
        w = max(32, width // downsample_factor // 2)
        try:
            data = VideoData(str(path), height=h, width=w).raw_data()
        except Exception as e:
            _log(self.log_prefix, f"Could not load in-context video {path}: {e}", "warning")
            return None
        if len(data) > num_frames:
            data = data[:num_frames]
        _log(self.log_prefix,
             f"  In-context video → {len(data)} frames at {w}x{h} (downsample={downsample_factor})")
        return data
