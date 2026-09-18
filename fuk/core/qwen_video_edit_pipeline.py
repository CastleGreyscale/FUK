"""
Qwen-Video-Edit Pipeline Runner for FUK

Video-to-video editing: a source clip plus a text instruction in, an edited clip
out. Unlike every other video model in the registry this one generates nothing
from scratch — the source video is mandatory conditioning, not an optional
reference, so the tab's video slot is required and the image slots are unused.

Architecture is a hybrid, which is why it gets its own family rather than
riding on the qwen image pipeline: QwenImageDiT as the backbone, the Qwen-Image
text encoder, but the *Wan 2.1* VAE for video encode/decode, joined by a
QwenVideoEditAdapter that projects Wan latents into the DiT's feature space.
The Wan VAE is already on disk for the Wan models, so the marginal cost of this
model is the DiT checkpoint plus the adapter.

Chunking is the thing to understand here. The model works on fixed 45-frame
chunks and takes a *list* of prompts, one per chunk; upstream silently drops
any frames beyond the prompts it was given:

    num_chunks = min(len(prompts), num_video_chunks)   # -> trailing frames lost

FUK's video tab posts a single prompt, so the runner replicates it across every
chunk the source needs. That is what makes "edit this 200-frame clip" behave the
way the tab implies rather than editing the first 45 frames and discarding the
rest. A caller that genuinely wants per-chunk direction can pass `prompts` as a
list and it is used verbatim.

Licence: the weights (`yunpeng1998/Qwen-Video-Edit`) are a personal research
checkpoint published with *no declared licence* — see THIRD_PARTY_LICENSES.md.
Treat as research-only until the author states terms.
"""

from __future__ import annotations

import math
import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from pipeline_base import PipelineRunner, _log
from perf_monitor import record_timing


class QwenVideoEditPipelineRunner(PipelineRunner):
    """Runner for Qwen-Video-Edit prompt-driven video editing."""

    pipeline_family = "qwen_video_edit"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        task: str = "qwen_video_edit",
        # Size
        width: int = None,
        height: int = None,
        video_length: int = None,
        # Source clip — the thing being edited
        control_path: Optional[Path] = None,
        edit_video: Optional[Any] = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        # Qwen-Video-Edit specifics
        chunk_frames: Optional[int] = None,
        prompts: Optional[List[str]] = None,
        zero_cond_t: Optional[bool] = None,
        tiled: Optional[bool] = None,
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        loras: Optional[List] = None,
        # Progress
        progress_callback=None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Edit an existing clip according to a text instruction."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(task)
        entry = self.get_model_entry(model_type)
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # --- Resolve parameters against defaults ---
        defaults = self.get_family_defaults()
        width = width or defaults.get("width") or 640
        height = height or defaults.get("height") or 384
        num_steps = steps or infer_steps or defaults.get("steps", 40)
        # `is not None` rather than `or`: cfg_scale=1.0 is meaningful — it turns
        # the negative branch off entirely — and 0 must not fall through.
        effective_cfg = (cfg_scale if cfg_scale is not None
                         else guidance_scale if guidance_scale is not None
                         else defaults.get("cfg_scale", 4.0))
        # The pipeline treats " " and "" differently: its own default is a
        # single space, and an empty string disables the negative branch.
        negative_prompt = (negative_prompt if negative_prompt is not None
                           else defaults.get("negative_prompt", " ")) or " "

        # The chunk length is the model's trained window, not a user-facing
        # duration. video_length controls how much of the source is edited; this
        # controls the size of the bite it is taken in.
        chunk = int(chunk_frames or defaults.get("chunk_frames", 45))
        width, height, chunk = self.snap_to_grid(width, height, chunk, entry)

        fps = entry.get("fps", defaults.get("fps", 16))

        # --- Source clip ---
        source = edit_video if edit_video is not None else control_path
        if source is None:
            raise ValueError(
                "Qwen-Video-Edit edits an existing clip — it cannot generate one. "
                "Provide a source video in the tab's video slot (control_path)."
            )

        frames = self._load_edit_video(source, width, height)
        if not frames:
            raise ValueError(f"Could not read any frames from source video: {source}")

        # video_length trims the source; unset means edit the whole clip.
        requested = video_length or defaults.get("video_length")
        if requested and len(frames) > int(requested):
            _log(self.log_prefix,
                 f"  Trimming source {len(frames)} -> {int(requested)} frames (video_length)")
            frames = frames[:int(requested)]
        num_frames = len(frames)

        # One prompt per chunk. Anything short of ceil() makes upstream drop the
        # tail of the clip, so replicate rather than let that happen silently.
        num_chunks = max(1, math.ceil(num_frames / chunk))
        if prompts:
            chunk_prompts = list(prompts)
            if len(chunk_prompts) < num_chunks:
                _log(self.log_prefix,
                     f"  {len(chunk_prompts)} prompts for {num_chunks} chunks — "
                     f"repeating the last to cover the clip", "warning")
                chunk_prompts += [chunk_prompts[-1]] * (num_chunks - len(chunk_prompts))
        else:
            chunk_prompts = [prompt] * num_chunks

        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt.strip() or "(blank)",
            "source": str(source),
            "size": f"{width}x{height}x{num_frames} @{fps}fps",
            "chunks": f"{num_chunks} x {chunk} frames",
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "seed": seed,
        }
        self.log_generation_header("QWEN-VIDEO-EDIT GENERATION", model_type, entry, log_params)

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)

        pipe_kwargs = dict(
            edit_video=frames,
            prompts=chunk_prompts,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            num_frames=chunk,
            cfg_scale=effective_cfg,
        )
        if zero_cond_t is not None:
            pipe_kwargs["zero_cond_t"] = bool(zero_cond_t)
        if tiled is not None:
            pipe_kwargs["tiled"] = bool(tiled)

        pipe_kwargs.update(pipe_defaults)

        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent, model_type)

        if progress_callback:
            # Every chunk runs the full denoise schedule, so the honest total is
            # steps * chunks — reporting `steps` would stall the bar at 100% for
            # every chunk after the first.
            progress_callback("generating", 0, num_steps * num_chunks)

        try:
            _t_pipe = time.perf_counter()
            with torch.inference_mode():
                video = pipe(**pipe_kwargs)
            _pipe_s = time.perf_counter() - _t_pipe
            _log(self.log_prefix, f"[timing] pipe() denoise+decode: {_pipe_s:.1f}s")
            record_timing(f"denoise_per_step:{model_type}:{width}x{height}x{chunk}",
                          _pipe_s / max(1, num_steps * num_chunks))

            if progress_callback:
                progress_callback("saving", 0, 1)

            from diffsynth.utils.data import save_video
            save_video(video, str(output_path), fps=fps, quality=8)

            elapsed = time.time() - start_time
            _log(self.log_prefix,
                 f"Video saved: {output_path} ({elapsed:.1f}s)", "success")

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
                    "cfg_scale": effective_cfg, "chunk_frames": chunk,
                    "chunks": num_chunks, "source_video": str(source),
                },
            )
        except Exception as e:
            _log(self.log_prefix, f"Generation failed: {e}", "error")
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

    def _load_edit_video(self, source, width: int, height: int) -> List:
        """Read the source clip into the list of PIL frames the pipeline wants.

        Passed through at the target size so the log, the metadata and the DiT
        agree — the pipeline resizes each frame itself anyway, but only after
        FUK has already reported what it thinks it is running.
        """
        from diffsynth.utils.data import VideoData

        if isinstance(source, (list, tuple)):
            return list(source)
        if hasattr(source, "raw_data"):
            return list(source.raw_data())

        p = Path(str(source))
        if p.is_dir():
            # A folder of frames must go through `image_folder`, not the
            # positional `video_file` arg, which would try to open the
            # directory as a video.
            data = VideoData(image_folder=str(p), height=height, width=width)
        elif p.exists():
            data = VideoData(str(p), height=height, width=width)
        else:
            raise ValueError(f"Source video does not exist: {source}")

        frames = list(data.raw_data())
        _log(self.log_prefix, f"  Source video → {len(frames)} frames at {width}x{height}")
        return frames
