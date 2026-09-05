"""
MiniMax-H3 Pipeline Runner for FUK

MiniMax-H3 is a joint audio-video model: one denoise produces the picture and a
synchronized soundtrack, so output is muxed with write_video_audio rather than
the plain save_video the Wan runner uses. Its audio runs at 32kHz, where LTX-2
uses 24kHz — hence audio_sample_rate being a per-model registry field.

Two DiT checkpoints, registered as separate FUK models because they take
different conditioning and ship their own processors:

  minimax_h3_fl2va   first and/or last frame -> video + audio
                     (`keyframes` with `keyframe_indices` in {0, -1})
  minimax_h3_ref2va  reference media -> video + audio
                     (`references`, a list of image / video / audio / video_audio
                     dicts that the prompt refers to as <Subject 1>, <Video 1>,
                     <Audio 1> and so on)

Both share the text encoder and the two VAEs, so adding the second model costs
only its DiT.

FUK registers the pre-quantized NF4 pruned weights by default. That is the path
Phase 3 pointed at: DiffSynth resolves these checkpoints by hash and applies the
authors' own `quant_config` — bitsandbytes NF4 with load_prequantized and a
calibrated exclusion list — so the quality is nothing like the online
quantization measured in defaults_vram.json. FUK's own quant_* VRAM presets do
not touch these files, and must not: the pattern heuristic in
_should_quantize deliberately does not match them.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from pipeline_base import PipelineRunner, _log
from perf_monitor import record_timing


class MiniMaxH3PipelineRunner(PipelineRunner):
    """Runner for MiniMax-H3 joint audio-video generation."""

    pipeline_family = "minimax_h3"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        task: str = "minimax_h3_fl2va",
        # Size
        width: int = None,
        height: int = None,
        video_length: int = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        flow_shift: Optional[float] = None,
        audio_flow_shift: Optional[float] = None,
        sigma_shift: Optional[float] = None,
        # FL2VA: keyframes
        image_path: Optional[Path] = None,
        end_image_path: Optional[Path] = None,
        # Ref2VA: reference media
        reference_image: Optional[Any] = None,
        reference_video: Optional[Path] = None,
        reference_audio: Optional[Path] = None,
        # Progress
        progress_callback=None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a joint video + audio clip using MiniMax-H3."""
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
        num_frames = video_length or defaults.get("video_length") or 124
        num_steps = steps or infer_steps or defaults.get("steps", 50)
        # `is not None` rather than `or`: cfg_scale=1.0 is meaningful here (it is
        # the upstream default and means CFG off), and 0 must not fall through.
        effective_cfg = (cfg_scale if cfg_scale is not None
                         else guidance_scale if guidance_scale is not None
                         else defaults.get("cfg_scale", 1.0))
        # The video tab posts one generic shift field named sigma_shift, the name
        # Wan uses. Same remap mathematically, different upstream spelling.
        shift = (flow_shift if flow_shift is not None
                 else sigma_shift if sigma_shift is not None
                 else defaults.get("flow_shift", 12.0))
        a_shift = (audio_flow_shift if audio_flow_shift is not None
                   else defaults.get("audio_flow_shift", 3.0))
        # The pipeline treats " " and "" differently — its own default is a single
        # space, and an empty string disables the negative branch entirely.
        negative_prompt = (negative_prompt if negative_prompt is not None
                           else defaults.get("negative_prompt", " ")) or " "

        width, height, num_frames = self.snap_to_grid(width, height, num_frames, entry)
        audio_sample_rate = entry.get("audio_sample_rate", 32000)
        fps = entry.get("fps", 24)

        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt.strip() or "(blank)",
            "size": f"{width}x{height}x{num_frames} @{fps}fps",
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "flow_shift": f"{shift} (audio {a_shift})",
            "seed": seed,
            "first_frame": str(image_path) if image_path else None,
            "last_frame": str(end_image_path) if end_image_path else None,
            "reference_image": str(reference_image) if reference_image else None,
            "reference_video": str(reference_video) if reference_video else None,
            "reference_audio": str(reference_audio) if reference_audio else None,
        }
        self.log_generation_header("MINIMAX-H3 GENERATION", model_type, entry, log_params)

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=effective_cfg,
            flow_shift=shift,
            audio_flow_shift=a_shift,
        )

        if "keyframes" in supports:
            keyframes, indices = self._build_keyframes(image_path, end_image_path, width, height)
            if keyframes:
                pipe_kwargs["keyframes"] = keyframes
                pipe_kwargs["keyframe_indices"] = indices

        if "references" in supports:
            # The video tab has one image slot and posts it as image_path, so on
            # a Ref2VA model that is the subject reference. An explicit
            # reference_image (from the API or a chained call) still wins.
            ref_img = reference_image or image_path
            # control_path is the tab's video slot; for Ref2VA it is a reference
            # clip rather than a control signal.
            ref_vid = reference_video or kwargs.get("control_path")
            references = self._build_references(
                ref_img, ref_vid, reference_audio,
                width, height, num_frames, fps, pipe)
            if references:
                pipe_kwargs["references"] = references

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

            from diffsynth.utils.data.audio_video import write_video_audio
            write_video_audio(
                video=video,
                audio=audio,
                output_path=str(output_path),
                fps=fps,
                audio_sample_rate=audio_sample_rate,
            )

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
                    "cfg_scale": effective_cfg, "flow_shift": shift,
                    "audio_flow_shift": a_shift,
                },
                has_audio=audio is not None,
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

    def _build_keyframes(self, image_path, end_image_path, width, height):
        """FL2VA conditioning: first frame at index 0, last frame at index -1.

        Either may be omitted — a first frame alone is ordinary image-to-video,
        a last frame alone drives toward a target, and both together interpolate.
        """
        frames, indices = [], []
        if image_path:
            img = self.load_image(image_path, width=width, height=height)
            if img is not None:
                frames.append(img)
                indices.append(0)
        if end_image_path:
            img = self.load_image(end_image_path, width=width, height=height)
            if img is not None:
                frames.append(img)
                indices.append(-1)
        if frames:
            _log(self.log_prefix,
                 f"  Keyframes → indices {indices} at {width}x{height}")
        return frames, indices

    def _build_references(self, reference_image, reference_video, reference_audio,
                          width, height, num_frames, fps, pipe):
        """Ref2VA conditioning.

        Order matters: the prompt refers to these positionally as <Subject 1>,
        <Video 1>, <Audio 1>, so they are appended image -> video -> audio and
        that order must match how the prompt was written.
        """
        references = []

        if reference_image:
            paths = reference_image if isinstance(reference_image, list) else [reference_image]
            for p in paths:
                img = self.load_image(p)
                if img is not None:
                    references.append({"type": "image", "image": img})

        if reference_video:
            # A reference video and its own soundtrack are one time-aligned unit,
            # so they go in as a single video_audio entry when the file has sound.
            try:
                from diffsynth.utils.data.audio_video import read_video_audio
                frames, track, rate = read_video_audio(
                    str(reference_video), height=height, width=width,
                    num_frames=num_frames, fps=fps,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                )
                if track is not None:
                    references.append({"type": "video_audio", "video": frames,
                                       "audio": track, "sample_rate": rate})
                else:
                    references.append({"type": "video", "video": frames})
            except Exception as e:
                _log(self.log_prefix,
                     f"Could not read reference video {reference_video}: {e}", "warning")

        if reference_audio:
            try:
                from diffsynth.utils.data.audio import read_audio
                voice, rate = read_audio(
                    str(reference_audio), duration=num_frames / fps,
                    resample=True, resample_rate=pipe.audio_vae.sample_rate,
                )
                references.append({"type": "audio", "audio": voice, "sample_rate": rate})
            except ImportError as e:
                # read_audio only has a torchcodec backend, and torchcodec is not
                # a FUK dependency. Say so rather than dropping the reference and
                # producing a clip that quietly ignores the requested voice.
                raise RuntimeError(
                    f"Audio references need torchcodec, which is not installed ({e}). "
                    f"Install it with: pip install torchcodec"
                ) from e

        if references:
            _log(self.log_prefix,
                 f"  References → {[r['type'] for r in references]}")
        return references
