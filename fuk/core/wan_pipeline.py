"""
Wan Pipeline Runner for FUK

Handles all Wan-family video generation:
  - wan_i2v_a14b (dual DiT image-to-video)
  - wan_vace_a14b (VACE control video + reference)
  - (future Wan variants go here)

Wan-specific features: sigma_shift, sliding window,
VACE video data loading, tiled inference, dual-DiT boundary switching.
"""

from __future__ import annotations

import os
import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from pipeline_base import PipelineRunner, _log, _lora_label
from perf_monitor import record_timing


def attention_backend() -> str:
    """Report which attention kernel the Wan DiT blocks will actually use.

    The Wan DiT's `flash_attention()` picks a backend purely from which
    modules imported successfully, with no env override and no runtime log —
    so a `pip install` alone is not evidence the new kernel is in play. This
    reads the vendored module's own availability flags and mirrors its
    priority chain (see `diffsynth/models/wan_video_dit.py`); if a vendor
    bump reorders that chain, this needs to follow.

    Read live rather than cached: `flash_attention()` re-reads these globals
    on every call, so a benchmark that flips one to force an A/B must see
    the change reflected here too.
    """
    try:
        from diffsynth.models import wan_video_dit as _dit
        if _dit.FLASH_ATTN_3_AVAILABLE:
            return "flash-attn-3"
        if _dit.FLASH_ATTN_2_AVAILABLE:
            return "flash-attn-2"
        if _dit.SAGE_ATTN_AVAILABLE:
            return "sage-attn"
        return "sdpa"
    except Exception as e:  # noqa: BLE001 - diagnostics must never break generation
        return f"unknown ({e})"


def apply_attention_override() -> None:
    """Make DIFFSYNTH_ATTENTION_IMPLEMENTATION reach the Wan DiT.

    The Wan DiT has its own availability flags and its own priority chain,
    separate from `diffsynth/core/attention/attention.py` — and unlike that
    module it reads no env var, so `--no-sage` alone leaves sage-attn running
    here. Forcing the chain means clearing the flags for every kernel that
    outranks the requested one; `flash_attention()` re-reads them per call, so
    this takes effect on the next denoise regardless of import order.

    Only the kernels FUK actually ships are honoured: 'torch'/'sdpa' clears
    all three. A value naming a kernel that is not installed is left alone —
    the chain falls through to whatever is available, and
    `attention_backend()` reports what really ran.
    """
    impl = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "").lower()
    if impl not in ("torch", "sdpa", "sage_attention"):
        return
    try:
        from diffsynth.models import wan_video_dit as _dit
        # Both flash-attn tiers outrank sage, so they clear either way.
        _dit.FLASH_ATTN_3_AVAILABLE = False
        _dit.FLASH_ATTN_2_AVAILABLE = False
        if impl in ("torch", "sdpa"):
            _dit.SAGE_ATTN_AVAILABLE = False
    except Exception as e:  # noqa: BLE001 - never break generation over a perf knob
        _log("wan", f"Could not apply attention override: {e}", "warning")


class WanPipelineRunner(PipelineRunner):
    """
    Runner for all Wan video generation pipelines.
    """

    pipeline_family = "video"

    # The latent grid (/16, 4n+1) now lives in defaults.json under
    # video.constraints and is applied by PipelineRunner.snap_to_grid. It is
    # config rather than a constant because WanVideoPipeline overrides its own
    # division factors from the VAE at load time (wan_video.py:177-179): the
    # Wan 2.1 VAE gives 16, but WanVideoVAE38 (Wan 2.2 TI2V-5B) gives 32, and
    # such an entry can now declare that in its own `constraints` block.

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
        # Animate video inputs
        animate_pose_video: Optional[Path] = None,
        animate_face_video: Optional[Path] = None,
        animate_inpaint_video: Optional[Path] = None,
        animate_mask_video: Optional[Path] = None,
        # Optional init-video (for the HD-proxy conform path). When present,
        # the pipeline VAE-encodes the video and the scheduler adds noise at
        # a level set by `denoising_strength`, so generation denoises *from*
        # the proxy rather than from pure noise. Pass alongside control_path.
        input_video_path: Optional[Path] = None,
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
        num_frames = video_length or defaults.get("video_length", 41)

        # Snap to the grid the pipeline itself enforces, BEFORE anything downstream
        # uses these numbers. WanVideoPipeline rounds height/width up to a multiple of
        # 16 and num_frames to 4n+1 internally — but the control video and reference
        # image are loaded at whatever we pass to map_inputs(). If the two disagree,
        # nothing errors: VaceWanModel.forward truncates or zero-pads the control token
        # sequence to the latent's length, which silently puts the control on a
        # different patch grid than the image and shears it across the frame. The
        # control then reads as noise and the video appears to ignore it entirely.
        # (Blender hits this constantly: render height x percentage is rarely /16 —
        # e.g. 720 x 50% = 360, which rounds to 368.)
        width, height, num_frames = self.snap_to_grid(width, height, num_frames, entry)
        num_steps = steps or infer_steps or defaults.get("steps", 40)
        # `is not None` rather than `or`, so an explicit cfg_scale=0 is honoured
        # instead of silently falling through to guidance_scale.
        effective_cfg = (cfg_scale if cfg_scale is not None
                         else guidance_scale if guidance_scale is not None
                         else defaults.get("cfg_scale", 4.0))
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = (denoising_strength if denoising_strength is not None
                   else defaults.get("denoising_strength", 1.0))

        # Wan-specific params
        sigma_shift = (kwargs.get("sigma_shift") if kwargs.get("sigma_shift") is not None
                       else defaults.get("sigma_shift", 5.0))
        # Dual-DiT boundary: timestep fraction (×1000) below which Wan 2.2 switches
        # from the high-noise DiT to the low-noise DiT. Higher = more steps on the
        # high-noise expert. Only meaningful for dual-DiT (A14B) models.
        switch_dit_boundary = (kwargs.get("switch_dit_boundary") if kwargs.get("switch_dit_boundary") is not None
                               else defaults.get("switch_dit_boundary", 0.875))
        sliding_window_size = (kwargs.get("sliding_window_size") if kwargs.get("sliding_window_size") is not None
                               else defaults.get("sliding_window_size"))
        sliding_window_stride = (kwargs.get("sliding_window_stride") if kwargs.get("sliding_window_stride") is not None
                                 else defaults.get("sliding_window_stride"))

        # TeaCache — training-free step skipping. Opt-in per job; stays None
        # (off) unless a threshold is explicitly passed or set in defaults.
        tea_cache_thresh = (kwargs.get("tea_cache_l1_thresh") if kwargs.get("tea_cache_l1_thresh") is not None
                            else defaults.get("tea_cache_l1_thresh"))
        tea_cache_model_id = kwargs.get("tea_cache_model_id") or defaults.get("tea_cache_model_id")
        if tea_cache_thresh is not None and not tea_cache_model_id:
            tea_cache_model_id = self._tea_cache_model_id(model_type, height)

        # Apply before attention_backend() reads the flags, so the logged
        # backend is the one this run will actually use.
        apply_attention_override()

        # --- Logging ---
        log_params = {
            "prompt": prompt,
            "task": f"{task} → {model_type}",
            "size": f"{width}x{height}",
            "frames": num_frames,
            "steps": num_steps,
            "cfg_scale": effective_cfg,
            "denoising_strength": denoise,
            "seed": seed,
            "sigma_shift": sigma_shift,
            "switch_dit_boundary": switch_dit_boundary,
            "attention": attention_backend(),
            "input_image": image_path,
            "control_path": control_path,
            "pipeline_kwargs": pipe_defaults if pipe_defaults else None,
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [_lora_label(l) for l in (loras or [])],
        }
        if tea_cache_thresh is not None:
            log_params["tea_cache"] = f"l1_thresh={tea_cache_thresh} (coeffs={tea_cache_model_id})"
        # Add animate inputs if present
        if animate_pose_video:
            log_params["animate_pose_video"] = animate_pose_video
        if animate_face_video:
            log_params["animate_face_video"] = animate_face_video
        if animate_inpaint_video:
            log_params["animate_inpaint_video"] = animate_inpaint_video
        if animate_mask_video:
            log_params["animate_mask_video"] = animate_mask_video
        self.log_generation_header("VIDEO GENERATION", model_type, entry, log_params)

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
            switch_DiT_boundary=switch_dit_boundary,
        )

        # Optional Wan params — only pass when explicitly set
        if sliding_window_size is not None:
            pipe_kwargs["sliding_window_size"] = sliding_window_size
        if sliding_window_stride is not None:
            pipe_kwargs["sliding_window_stride"] = sliding_window_stride
        if tea_cache_thresh is not None:
            pipe_kwargs["tea_cache_l1_thresh"] = tea_cache_thresh
            pipe_kwargs["tea_cache_model_id"] = tea_cache_model_id

        # Negative prompt
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # --- Map semantic inputs → model-specific params ---
        semantic_inputs = {}
        if image_path:
            semantic_inputs["reference_image"] = image_path
        if end_image_path:
            semantic_inputs["end_image"] = end_image_path
        if control_path:
            semantic_inputs["control_input"] = control_path
        # Animate video inputs
        if animate_pose_video:
            semantic_inputs["animate_pose_video"] = animate_pose_video
        if animate_face_video:
            semantic_inputs["animate_face_video"] = animate_face_video
        if animate_inpaint_video:
            semantic_inputs["animate_inpaint_video"] = animate_inpaint_video
        if animate_mask_video:
            semantic_inputs["animate_mask_video"] = animate_mask_video

        mapped_inputs = self.map_inputs(model_type, semantic_inputs, width, height)
        pipe_kwargs.update(mapped_inputs)

        # Init-video latent path (HD-proxy conform). Materialize the proxy
        # video to a list of PIL frames at target spatial dimensions; the
        # downstream pipeline VAE-encodes and partial-noises per
        # denoising_strength.
        if input_video_path:
            init_vd = self._load_video_data(input_video_path, height, width)
            if init_vd is not None:
                frames = [init_vd[i] for i in range(min(len(init_vd), num_frames))]
                pipe_kwargs["input_video"] = frames
                _log(self.log_prefix, f"  Init video: {len(frames)} frames @ {width}x{height}")
            else:
                _log(self.log_prefix, f"  Failed to load input_video_path: {input_video_path}", "warning")

        # Tiled inference
        if "tiled" in supports and "tiled" not in pipe_defaults:
            pipe_kwargs["tiled"] = True

        # Merge pipeline_kwargs from models.json
        pipe_kwargs.update(pipe_defaults)

        # --- Latent capture ---
        latent_path, cleanup_hook = self.setup_latent_capture(pipe, output_path, save_latent, model_type)

        # --- Generate ---
        try:
            _t_pipe = time.perf_counter()
            with torch.inference_mode():
                video = pipe(**pipe_kwargs)
            _pipe_s = time.perf_counter() - _t_pipe
            _log(self.log_prefix, f"[timing] pipe() denoise+decode: {_pipe_s:.1f}s")
            record_timing(f"denoise_per_step:{model_type}:{width}x{height}x{num_frames}",
                          _pipe_s / max(1, num_steps))

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
                    "frames": num_frames, "fps": 24, "cfg_scale": effective_cfg,
                    "denoising_strength": denoise, "sigma_shift": sigma_shift,
                    "switch_dit_boundary": switch_dit_boundary,
                    # Speed/quality A/B fields — make the tradeoff readable
                    # straight from generation history, not just benchmarks.
                    "tea_cache_l1_thresh": tea_cache_thresh,
                    "tea_cache_model_id": tea_cache_model_id if tea_cache_thresh is not None else None,
                    "attention_backend": attention_backend(),
                    "denoise_seconds": round(_pipe_s, 1),
                    "sec_per_step": round(_pipe_s / max(1, num_steps), 2),
                },
            )
        except Exception as e:
            _log(self.log_prefix, f"Video generation failed: {e}", "error")
            # pipe.__call__ never reached its own load_models_to_device([]) —
            # without this the promoted weights stay on the GPU for good.
            self.release_pipeline_vram(pipe)
            raise
        finally:
            # VRAM/GC cleanup is handled by the server's clear_vram() after the
            # request completes — a second gc/empty_cache here just adds stalls.
            if cleanup_hook:
                cleanup_hook()

    # ------------------------------------------------------------------
    # TeaCache
    # ------------------------------------------------------------------

    # DiffSynth's TeaCache ships coefficient tables for Wan *2.1* only, and
    # raises ValueError on any id outside that table — including the "" it
    # defaults to. Our models are all Wan 2.2 A14B, so there is no exact
    # match; we pick the nearest 2.1 polynomial as a proxy. This is why
    # thresholds have to be swept empirically rather than lifted from
    # published Wan 2.1 numbers.
    _TEA_CACHE_IDS = {
        "i2v_480": "Wan2.1-I2V-14B-480P",
        "i2v_720": "Wan2.1-I2V-14B-720P",
        "t2v": "Wan2.1-T2V-14B",
    }

    def _tea_cache_model_id(self, model_type: str, height: int) -> str:
        """Pick the closest Wan2.1 TeaCache coefficient set for a 2.2 model."""
        name = (model_type or "").lower()
        if "i2v" in name or "inp" in name or "animate" in name:
            key = "i2v_720" if (height or 0) >= 704 else "i2v_480"
        else:
            key = "t2v"
        return self._TEA_CACHE_IDS[key]

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

        # Animate video inputs — load as raw frame data
        if model_param in ("animate_pose_video", "animate_face_video",
                          "animate_inpaint_video", "animate_mask_video"):
            video_data = self._load_video_data(value, height, width)
            if video_data:
                raw_frames = video_data.raw_data()
                _log(self.log_prefix, f"  Mapped {semantic_name} → {model_param} ({len(raw_frames)} frames)")
                return {model_param: raw_frames}
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
            # A folder of frames must go through `image_folder`, not the positional
            # `video_file` arg (which would try to open the directory as a video).
            return VideoData(image_folder=str(p), height=height, width=width)

        _log(self.log_prefix, f"Could not load video data from: {source}", "warning")
        return None
