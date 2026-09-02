"""
Base Pipeline Runner for FUK

Shared infrastructure for all pipeline-specific runners.
Each runner handles one family of generation (Qwen image, Wan video, etc.)
while the DiffSynthBackend hub manages pipeline lifecycle, VRAM, and LoRA state.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    from diffsynth_backend import DiffSynthBackend


def _log(category: str, message: str, level: str = "info"):
    """Logging helper matching FUK server style."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    colors = {
        'info': '\033[96m', 'success': '\033[92m',
        'warning': '\033[93m', 'error': '\033[91m', 'end': '\033[0m',
    }
    symbols = {'info': '', 'success': '✔ ', 'warning': '⚠ ', 'error': '✗ '}
    color = colors.get(level, colors['info'])
    symbol = symbols.get(level, '')
    print(f"{color}[{timestamp}] {symbol}[{category}] {message}{colors['end']}", flush=True)


def _lora_label(entry: dict) -> str:
    """Render one LoRA spec for the run header.

    Must mirror DiffSynthBackend._resolve_lora_specs' key precedence exactly: the web
    UI sends {key, multiplier} while older callers send {name, alpha}. Reading only
    name/alpha printed every LoRA as "? (α=1.0)" no matter what was really applied —
    which quietly hid the true weights during quality debugging.
    """
    name = entry.get("name") or entry.get("key") or entry.get("path") or "?"
    alpha = entry.get("alpha")
    if alpha is None:
        alpha = entry.get("multiplier", 1.0)
    return f"{name} (α={alpha})"



class GenerationCancelled(Exception):
    """Raised from the per-step hook to abort a generation mid-denoise.

    fuk_web_server distinguishes a user cancellation from a real failure by
    class name, so this type's name is load-bearing — do not rename it without
    updating that check.
    """


class PipelineRunner:
    """
    Base class for pipeline-specific generation runners.

    Subclasses implement:
        - pipeline_family: str property (e.g. "qwen", "wan")
        - generate(): the actual generation method
        - _build_pipe_kwargs(): translate params into pipe() kwargs

    The runner holds a reference to the backend hub for pipeline access,
    LoRA management, latent capture, and shared config. This means chained
    pipelines can grab multiple pipes from the hub within one generate().
    """

    # Subclasses set this — used for log prefixes and defaults.json section
    pipeline_family: str = "base"

    def __init__(self, backend: DiffSynthBackend):
        self.backend = backend

    # ------------------------------------------------------------------
    # Config convenience
    # ------------------------------------------------------------------

    @property
    def defaults_config(self) -> dict:
        return self.backend.defaults_config

    @property
    def models_config(self) -> dict:
        return self.backend.models_config

    def get_family_defaults(self) -> dict:
        """Get defaults.json section for this pipeline family.
        
        e.g. for pipeline_family="image" returns defaults_config["image"]
        Override in subclass if the section name doesn't match the family.
        """
        return self.defaults_config.get(self.pipeline_family, {})

    def get_constraints(self, entry: dict = None) -> dict:
        """Resolve the model's hard parameter constraints.

        The family's `constraints` block in defaults.json, overlaid with any
        per-model `constraints` on the models.json entry — same precedence as
        `pipeline_kwargs`. These are the model's *limits* (latent grid, frame
        lattice, which shift knob exists), not its preferred values, and the
        same block is handed to the UI so both sides validate identically.
        """
        merged = dict(self.get_family_defaults().get("constraints", {}))
        if entry:
            merged.update(entry.get("constraints", {}))
        return merged

    def snap_to_grid(self, width, height, num_frames=None, entry: dict = None,
                     spatial: int = None):
        """Round up to the model's latent grid.

        Rounds *up*, reproducing DiffSynth's own check_resize_height_width. The
        pipelines snap internally anyway, but only after FUK has already resized
        control media and logged the size — so snapping here is what keeps the
        keyframe canvas, the log, and the returned metadata agreeing with what
        the DiT actually ran.

        `spatial` overrides the family value for modes that need a coarser grid
        (LTX-2 two-stage halves the size before its own check, so it needs 64).
        """
        c = self.get_constraints(entry)
        div = int(spatial or c.get("spatial_multiple", 16))
        factor = int(c.get("frame_factor", 4))
        remainder = int(c.get("frame_remainder", 1))
        min_frames = int(c.get("min_frames", factor + remainder))

        def up(v, d):
            return max(d, ((int(v) + d - 1) // d) * d)

        w, h = up(width, div), up(height, div)
        if num_frames is None:
            if (w, h) != (int(width), int(height)):
                _log(self.log_prefix,
                     f"  Snapped to pipeline grid: {width}x{height} -> {w}x{h} (/{div})")
            return w, h

        n = int(num_frames)
        if n % factor != remainder % factor:
            n = ((n - remainder + factor - 1) // factor) * factor + remainder
        n = max(min_frames, n)
        if (w, h, n) != (int(width), int(height), int(num_frames)):
            _log(self.log_prefix,
                 f"  Snapped to pipeline grid: {width}x{height}x{num_frames} -> {w}x{h}x{n} "
                 f"(/{div}, {factor}n+{remainder})")
        return w, h, n

    # ------------------------------------------------------------------
    # Pipeline + LoRA access (delegates to hub)
    # ------------------------------------------------------------------

    def get_pipeline(self, model_type: str, vram_preset: str = None):
        """Get a cached pipeline from the hub."""
        return self.backend.get_pipeline(model_type, vram_preset=vram_preset)

    def resolve_model_type(self, name: str) -> str:
        return self.backend.resolve_model_type(name)

    def get_model_entry(self, model_type: str) -> dict:
        return self.backend.get_model_entry(model_type)

    def _cache_key(self, model_type: str, vram_preset: str = None) -> str:
        """Build the cache key matching the hub's convention."""
        active_preset = vram_preset or self.defaults_config.get("vram", {}).get("preset", "low")
        return f"{model_type}:{active_preset}"

    def apply_loras(self, pipe, cache_key: str, lora: str = None,
                    lora_multiplier: float = 1.0,
                    loras: List[Dict[str, Any]] = None):
        """Resolve and apply user LoRA(s), or clear if none requested."""
        lora_specs = self.backend._resolve_lora_specs(lora, lora_multiplier, loras)
        if lora_specs:
            self.backend._apply_user_loras(pipe, cache_key, lora_specs)
        else:
            self.backend._clear_user_loras(pipe, cache_key)

    # ------------------------------------------------------------------
    # Abort recovery
    # ------------------------------------------------------------------

    def release_pipeline_vram(self, pipe):
        """Demote every VRAM-managed weight back to its offload device.

        Under the offload presets DiffSynth promotes layers from CPU to GPU as
        it denoises (AutoWrappedLinear.forward -> preparing(), gated on
        vram_limit) and that promotion is sticky: the only thing that undoes it
        is load_models_to_device(). Each pipeline calls that at the very END of
        __call__, so an abort — a user cancellation, or an OOM — raises past it
        and leaves the promoted weights resident for the life of the cached
        pipeline. Every later generation then starts that much VRAM down, which
        is self-perpetuating: its OOM leaves its own residency behind.

        gc + torch.cuda.empty_cache() cannot recover this. Those weights are
        live parameters owned by the pipeline, not allocator cache; empty_cache
        only returns blocks nothing references. This call is the only fix short
        of evicting the pipeline entirely.

        No-op when VRAM management is off ("none" preset — nothing was ever
        promoted), and never raises: it runs on paths that are already failing.
        """
        if pipe is None:
            return
        try:
            if not getattr(pipe, "vram_management_enabled", False):
                return
            _t0 = time.perf_counter()
            pipe.load_models_to_device([])
            if torch.cuda.is_available():
                _resv = torch.cuda.memory_reserved() / (1024 ** 3)
                _alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                _vram = f" — VRAM alloc {_alloc:.2f}GB resv {_resv:.2f}GB"
            else:
                _vram = ""
            _log(self.log_prefix,
                 f"Offloaded pipeline weights after abort "
                 f"({time.perf_counter() - _t0:.1f}s){_vram}")
        except Exception as e:
            _log(self.log_prefix,
                 f"Could not offload pipeline weights after abort: {e}", "warning")

    # ------------------------------------------------------------------
    # Per-step hook: live diffusion preview + cancellation
    # ------------------------------------------------------------------

    @staticmethod
    def _vae_can_stream(pipe):
        """True if the VAE can decode without an explicit load_models_to_device().

        Wrapped modules (AutoWrappedModule/AutoWrappedLinear) copy their weights
        to the computation device on-the-fly during forward, so decode works with
        the weights resting on CPU. The exception is disk offload, where offloaded
        weights live on the meta device and MUST be onloaded first. Without vram
        management, load_models_to_device() is a no-op anyway.
        """
        if not getattr(pipe, "vram_management_enabled", False):
            return True
        seen_wrapped = False
        for m in pipe.vae.modules():
            od = getattr(m, "offload_device", None)
            if od is not None:
                seen_wrapped = True
                if str(od) == "disk":
                    return False
        return seen_wrapped

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
        # When VAE weights can stream to GPU during forward, skip the explicit
        # device shuffles — load_models_to_device(["vae"]) demotes the entire
        # GPU-resident DiT to CPU and the follow-up call re-promotes it, a
        # multi-GB PCIe round trip per preview under the CPU-offload presets.
        vae_streams = self._vae_can_stream(pipe) if callback else True

        def hooked_step(scheduler, **kw):
            # Check for cancellation BEFORE doing the (expensive) step work.
            if cancel_check is not None and cancel_check():
                raise GenerationCancelled()
            latents_next = original_step(scheduler, **kw)
            step_num = int(kw.get("progress_id", 0)) + 1
            if callback and step_num in marks:
                try:
                    _t0 = time.perf_counter()
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
                    # Previews don't need full resolution: half-res decode is ~4x
                    # cheaper and its activations fit in the headroom left by the
                    # still-resident DiT.
                    if preview_latent.dim() == 4 and min(preview_latent.shape[-2:]) >= 32:
                        preview_latent = torch.nn.functional.interpolate(
                            preview_latent, scale_factor=0.5, mode="bilinear",
                        )
                    if not vae_streams:
                        pipe.load_models_to_device(["vae"])
                    out = original_vae_decode(
                        preview_latent, device=pipe.device,
                        tiled=kw.get("tiled", False),
                        tile_size=kw.get("tile_size", 128),
                        tile_stride=kw.get("tile_stride", 64),
                    )
                    pil = pipe.vae_output_to_image(out)
                    callback(step_num, total, pil)
                    if not vae_streams:
                        pipe.load_models_to_device(pipe.in_iteration_models)
                    if torch.cuda.is_available():
                        _alloc = torch.cuda.memory_allocated() / (1024**3)
                        _resv = torch.cuda.memory_reserved() / (1024**3)
                        _vram = f", VRAM alloc {_alloc:.1f}GB resv {_resv:.1f}GB"
                    else:
                        _vram = ""
                    _log(self.log_prefix,
                         f"[timing] preview decode @ step {step_num}: "
                         f"{time.perf_counter() - _t0:.2f}s{_vram}")
                except Exception as e:
                    # One failure (e.g. OOM) means the rest would fail too — stop trying.
                    marks.clear()
                    _log(self.log_prefix, f"Preview decode failed (non-fatal, previews disabled): {e}", "warning")
            return latents_next

        pipe.step = hooked_step
        _log(self.log_prefix,
             f"Step hook installed (preview steps {sorted(marks) or 'off'}, "
             f"vae_streams={vae_streams}, cancellable={cancel_check is not None})")

        def cleanup():
            pipe.step = original_step

        return cleanup

    # ------------------------------------------------------------------
    # Latent capture (delegates to hub)
    # ------------------------------------------------------------------

    def setup_latent_capture(self, pipe, output_path: Path, save_latent: bool):
        """
        Conditionally install latent capture hook.
        
        Returns (latent_path, cleanup_fn) — cleanup_fn is None if not capturing.
        Always call cleanup_fn in a finally block.
        """
        if not save_latent:
            return None, None

        latent_dir = output_path.parent / "latents"
        latent_dir.mkdir(exist_ok=True)
        latent_path = latent_dir / f"{output_path.stem}.latent.pt"
        cleanup = self.backend._capture_latent_hook(pipe, latent_path)
        _log(self.log_prefix, f"Latent capture enabled → {latent_path}")
        return latent_path, cleanup

    # ------------------------------------------------------------------
    # Image loading helpers
    # ------------------------------------------------------------------

    def load_image(self, path, width: int = None, height: int = None) -> Optional[Image.Image]:
        """Load a PIL Image, optionally resizing."""
        if path is None:
            return None
        if isinstance(path, list):
            if not path:
                return None
            _log(self.log_prefix, "Warning: load_image received list, using first item", "warning")
            path = path[0]
        p = Path(str(path))
        if not p.exists():
            _log(self.log_prefix, f"Image not found: {p}", "warning")
            return None
        if p.suffix.lower() == '.exr':
            import cv2
            from core.exr_utils import load_exr_bgr
            try:
                bgr = load_exr_bgr(p)
            except Exception as e:
                _log(self.log_prefix, f"Could not read EXR '{p}': {e}", "warning")
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
        else:
            img = Image.open(str(p)).convert("RGB")
        if width and height:
            img = img.resize((width, height))
        return img

    def resolve_image_list(
        self, control_image: Optional[Union[Path, List[Path]]],
        width: int = None, height: int = None,
    ) -> Optional[List[Image.Image]]:
        """Convert control_image path(s) to a list of PIL Images.
        
        When width/height are provided, all images are resized to match
        the generation target — prevents shape mismatches between the
        latent-encoded input and the noise tensor.
        """
        if not control_image:
            return None
        paths = control_image if isinstance(control_image, list) else [control_image]
        images = []
        for p in paths:
            p = Path(str(p))
            if p.exists():
                img = Image.open(str(p)).convert("RGB")
                if width and height and (img.width != width or img.height != height):
                    _log(self.log_prefix,
                         f"  Resizing edit image {p.name}: {img.width}x{img.height} → {width}x{height}")
                    img = img.resize((width, height), Image.LANCZOS)
                images.append(img)
        return images if images else None

    # ------------------------------------------------------------------
    # Input mapping (data-driven from models.json parameter_map)
    # ------------------------------------------------------------------

    def map_inputs(
        self,
        model_type: str,
        semantic_inputs: dict,
        width: int,
        height: int,
    ) -> dict:
        """
        Map semantic inputs to model-specific pipe() kwargs.
        
        Uses the parameter_map from models.json. Override or extend
        _map_single_input() in subclasses for custom input types.
        """
        entry = self.get_model_entry(model_type)
        param_map = entry.get("parameter_map", {})

        pipe_kwargs = {}
        for semantic_name, model_param in param_map.items():
            if semantic_name not in semantic_inputs:
                continue
            value = semantic_inputs[semantic_name]
            if value is None:
                continue
            mapped = self._map_single_input(semantic_name, model_param, value, width, height)
            if mapped is not None:
                pipe_kwargs.update(mapped)

        return pipe_kwargs

    def _map_single_input(
        self,
        semantic_name: str,
        model_param: str,
        value: Any,
        width: int,
        height: int,
    ) -> Optional[dict]:
        """
        Map one semantic input to its pipe kwarg(s).
        
        Base implementation handles common types. Subclasses override
        to add family-specific mappings (e.g. vace_video for Wan).
        Returns dict of pipe kwargs, or None to skip.
        """
        if model_param == "edit_image":
            images = self.resolve_image_list(value, width, height)
            if images:
                _log(self.log_prefix, f"  Mapped {semantic_name} → edit_image: {len(images)} images")
                return {"edit_image": images}

        elif model_param == "context_image":
            img = self.load_image(value, width, height)
            if img:
                _log(self.log_prefix, f"  Mapped {semantic_name} → context_image")
                return {"context_image": img}

        elif model_param == "input_image":
            img = self.load_image(value, width, height)
            if img:
                _log(self.log_prefix, f"  Mapped {semantic_name} → input_image")
                return {"input_image": img}

        else:
            _log(self.log_prefix, f"  Unknown parameter mapping: {semantic_name} → {model_param}", "warning")

        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @property
    def log_prefix(self) -> str:
        return self.pipeline_family.upper()

    def log_generation_header(self, title: str, model_type: str, entry: dict, params: dict):
        """Standard generation header logging."""
        _log(self.log_prefix, "=" * 60)
        _log(self.log_prefix, title)
        _log(self.log_prefix, f"  Model: {model_type} — {entry.get('description', '')}")

        prompt = params.get("prompt", "")
        _log(self.log_prefix, f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        # Log all params that are set
        for key, val in params.items():
            if key == "prompt":
                continue
            if val is not None:
                _log(self.log_prefix, f"  {key}: {val}")

        _log(self.log_prefix, "=" * 60)

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def success_result(self, output_path: Path, latent_path: Path, seed: int,
                       elapsed: float, params: dict, **extra) -> dict:
        """Build a standard success result dict."""
        result = {
            "success": True,
            "latent": latent_path,
            "seed_used": seed,
            "elapsed": round(elapsed, 1),
            "params": params,
        }
        # Add the output key based on suffix
        suffix = output_path.suffix.lower()
        if suffix in ('.mp4', '.avi', '.mov', '.mkv'):
            result["video"] = output_path
        else:
            result["image"] = output_path
        result.update(extra)
        return result