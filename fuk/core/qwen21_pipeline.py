"""
Qwen-Image-2.1 Pipeline Runner for FUK

Qwen-Image-2.1 shares a name with the Qwen-Image family and almost nothing else.
It is a separate DiffSynth pipeline (QwenImage21Pipeline) built on its own
32-layer single-stream block-causal DiT, its own 64-channel RGBA VAE and a
Qwen3-VL text encoder — none of the three is the component the qwen_image models
load, so nothing on disk is shared and QwenPipelineRunner cannot drive it.

What that buys, and why it is worth a runner of its own:

  - Native alpha. The VAE works in a 4-channel RGBA pixel space and the 4th
    decoded channel IS the alpha, so the pipeline hands back an RGBA PIL image
    and PNG output carries transparency with no matting step. There is no alpha
    switch — transparency is produced from the prompt (see PROMPT NOTE below).
  - Unified t2i and editing in one checkpoint, up to 10 reference images, so
    this single entry covers what qwen_image + qwen_image_edit_2511 cover
    separately.

What it does NOT have, all of which the Image tab still sends and this runner
deliberately drops rather than forwarding into a TypeError:

  - guidance_scale — see the CFG note below.
  - denoising_strength / detail_bias — QwenImage21Pipeline takes no input_image
    at inference. Its InputImageEmbedder unit only encodes one while
    scheduler.training is set, i.e. during LoRA/full training. There is no
    img2img path and no Detail-Bias noise scaling to patch.
  - exponential_shift_mu — the scheduler derives its shift from the latent
    sequence length and exposes no override.
  - EliGen entity masks, in-context control, blockwise ControlNets — none of
    them exist for this architecture yet. That is the one real gap versus
    qwen_image_control_union, and why 2.1 supplements the Qwen entries rather
    than replacing them.

CFG NOTE: cfg_scale defaults to 1.0 and the Image tab's guidance_scale is
ignored on purpose. Upstream's inference examples, DiffSynth's default and every
snippet in the model card run this model with no CFG at all; anything above 1.0
turns on the negative branch and doubles the cost per step for an operating
point the model was not shown. The tab's guidance_scale control is a
Qwen-Image-era knob and silently honouring it would mean every default
generation here ran CFG at 2.5. Override with cfg_scale via the API, or by
setting cfg_scale in the qwen21 block of defaults.json.

PROMPT NOTE: for a transparent result the prompt has to ask for one — "isolated
on a fully transparent background", "alpha matte", "die-cut sticker" — and must
not describe an environment or ambient light, or the model fills the canvas.

LICENCE: Qwen Research License — NON-COMMERCIAL. Unlike Qwen-Image and
Qwen-Image-Edit, which are Apache-2.0, nothing this model generates may go into
paid work. See THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from PIL import Image

from pipeline_base import PipelineRunner, _log, _lora_label, GenerationCancelled
from perf_monitor import record_timing


class QwenImage21PipelineRunner(PipelineRunner):
    """Runner for Qwen-Image-2.1 — unified text-to-image and image editing."""

    pipeline_family = "qwen21"

    # The DiT works on a /32 grid (height_division_factor=32), not the /16 the
    # qwen_image models use. Pre-rounding here rather than letting
    # check_resize_height_width do it keeps the size we log, the size we stamp
    # into metadata and the size the model actually produces identical.
    GRID = 32

    def generate(
        self,
        prompt: str,
        output_path: Path,
        model: str = "qwen_image_21",
        # Size
        width: int = None,
        height: int = None,
        # Generation params
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        # Reference / edit images (the tab's control image slots)
        control_image: Optional[Union[Path, List[Path]]] = None,
        # Per-layer KV cache under the block-causal condition. On by default
        # upstream; exposed only so it can be turned off while debugging.
        use_kv_cache: Optional[bool] = None,
        # VAE tiling. No decode guard applies here (see below), so this is the
        # lever for a decode that will not fit.
        tiled: Optional[bool] = None,
        tile_size: Optional[int] = None,
        tile_stride: Optional[int] = None,
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
        """Generate an image (or edit one) with Qwen-Image-2.1."""
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(model)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        defaults = self.get_family_defaults()
        width = width or defaults.get("width", 2048)
        height = height or defaults.get("height", 2048)

        # Pop both registry values UNCONDITIONALLY, before the precedence chains
        # below use them. pipe_defaults is merged into pipe_kwargs last, so a
        # key left in it wins over everything — popping only when the caller
        # said nothing would let pipeline_kwargs silently override an explicit
        # step count or cfg_scale at the final update().
        entry_steps = pipe_defaults.pop("num_inference_steps", None)
        entry_cfg = pipe_defaults.pop("cfg_scale", None)

        num_steps = steps or infer_steps or entry_steps or defaults.get("steps", 40)

        # CFG: explicit argument, then the registry, then the family default.
        # guidance_scale is NOT in this chain — see the CFG note in the module
        # docstring.
        effective_cfg = cfg_scale
        if effective_cfg is None:
            effective_cfg = entry_cfg
        if effective_cfg is None:
            effective_cfg = defaults.get("cfg_scale", 1.0)
        effective_cfg = float(effective_cfg)

        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        use_kv_cache = (use_kv_cache if use_kv_cache is not None
                        else defaults.get("use_kv_cache", True))
        tiled = tiled if tiled is not None else defaults.get("tiled", False)
        tile_size = tile_size or defaults.get("tile_size", 256)
        tile_stride = tile_stride or defaults.get("tile_stride", 192)

        # --- Report the parameters this model has no use for ---------------
        # The Image tab posts one payload shape for every image model, so these
        # arrive on every call. Saying so beats both a silent drop (the result
        # ignores a slider the user moved) and forwarding them (TypeError).
        _ignored = {
            name: value for name, value in (
                ("guidance_scale", kwargs.get("guidance_scale")),
                ("denoising_strength", kwargs.get("denoising_strength")),
                ("exponential_shift_mu", kwargs.get("exponential_shift_mu")),
                ("eligen_source", kwargs.get("eligen_source")),
            ) if value is not None
        }

        # --- Inherit dimensions from the first reference image --------------
        # Same rule as the Qwen edit models: the first image is the master, so
        # an edit comes back at the source's size and composites cleanly. The
        # pipeline resizes the references themselves — each to the target AREA
        # at its own aspect ratio — so only the output size is set here.
        source_path = self._resolve_source_image(control_image)
        if source_path:
            try:
                with Image.open(str(source_path)) as src:
                    src_w, src_h = src.size
                width, height = self._snap(src_w), self._snap(src_h)
                _log(self.log_prefix,
                     f"Inherited dimensions from source: {width}x{height}")
            except Exception as e:
                _log(self.log_prefix,
                     f"Could not read source image dimensions: {e}", "warning")
        width, height = self._snap(width), self._snap(height)

        log_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt else "(none)",
            "size": f"{width}x{height}",
            "steps": num_steps,
            "cfg_scale": f"{effective_cfg} ({'CFG on — 2x cost per step' if effective_cfg > 1 else 'CFG off'})",
            "seed": seed,
            "kv_cache": use_kv_cache,
            "vae_tiling": f"{tile_size}/{tile_stride}" if tiled else False,
            "lora": f"{lora} (α={lora_multiplier})" if lora else None,
            "loras": [_lora_label(spec) for spec in (loras or [])],
            "ignored_params": ", ".join(f"{k}={v}" for k, v in _ignored.items()) or None,
        }
        self.log_generation_header("QWEN-IMAGE-2.1 GENERATION", model_type, entry, log_params)
        if _ignored:
            _log(self.log_prefix,
                 f"Qwen-Image-2.1 has no use for {', '.join(_ignored)} — dropped. "
                 f"See qwen21_pipeline.py for why.", "warning")

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)
        cache_key = self._cache_key(model_type, vram_preset)
        self.apply_loras(pipe, cache_key, lora, lora_multiplier, loras)
        if lora or loras:
            self._warn_if_lora_missed(pipe)

        pipe_kwargs = dict(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            cfg_scale=effective_cfg,
            use_kv_cache=bool(use_kv_cache),
            tiled=bool(tiled),
            tile_size=int(tile_size),
            tile_stride=int(tile_stride),
        )
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # Reference images, via the registry's parameter_map like every other
        # family. _map_single_input below is what keeps their alpha.
        semantic_inputs = {}
        if control_image:
            semantic_inputs["edit_targets"] = control_image
        pipe_kwargs.update(self.map_inputs(model_type, semantic_inputs, width, height))

        pipe_kwargs.update(pipe_defaults)

        # No install_vae_decode_guard() here: its byte-per-pixel estimate is
        # calibrated to QwenImageVAE, and 2.1's is a different decoder over a
        # 64-channel latent, so the guard returns None for it anyway. The
        # pipeline's own `tiled` flag is the fallback for a decode that will not
        # fit — hence it being a registry/defaults knob above.
        original_vae_decode = pipe.vae.decode
        preview_cleanup = None
        if preview_callback is not None or cancel_check is not None:
            preview_cleanup = self._install_preview_hook(
                pipe, preview_callback, cancel_check, num_steps, original_vae_decode,
            )

        latent_path, cleanup_hook = self.setup_latent_capture(
            pipe, output_path, save_latent, model_type)

        try:
            _t_pipe = time.perf_counter()
            with torch.inference_mode():
                image = pipe(**pipe_kwargs)
            _pipe_s = time.perf_counter() - _t_pipe
            _log(self.log_prefix, f"[timing] pipe() denoise+decode: {_pipe_s:.1f}s")
            record_timing(f"denoise_per_step:{model_type}:{width}x{height}",
                          _pipe_s / max(1, num_steps))

            # The pipeline returns RGBA and output_path is a .png, so the alpha
            # survives the save untouched. Report whether the model actually
            # produced transparency — it is prompt-driven, so a fully opaque
            # alpha means the prompt did not ask for a cut-out, not a bug.
            image.save(output_path)
            _log(self.log_prefix, f"Output mode {image.mode}: {self._alpha_summary(image)}")

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
                    "use_kv_cache": bool(use_kv_cache),
                    "edit_images": len(pipe_kwargs.get("edit_image") or []),
                },
            )
        except GenerationCancelled:
            _log(self.log_prefix, "Generation cancelled at step boundary", "warning")
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

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _map_single_input(self, semantic_name, model_param, value, width, height):
        """Load reference images for 2.1, which needs both halves of this
        different from the base implementation.

        resolve_image_list() converts to RGB and resizes every image to the
        generation size. Both are wrong here:

          - .convert("RGB") flattens alpha, and feeding a generated cut-out back
            in as a reference — upstream's own example, and the whole point of
            an RGBA model — then hands the VAE an opaque image. The VAE reads
            all four channels, so the transparency has to survive.
          - Forcing every image to width x height distorts any reference whose
            aspect ratio differs from the output. The pipeline's own
            EditImageEmbedder resizes each one to the target AREA at its own
            ratio, honouring the processor's min_pixels, so pre-resizing here
            only takes that decision away and stretches the result.
        """
        if model_param != "edit_image":
            return super()._map_single_input(semantic_name, model_param, value, width, height)

        paths = value if isinstance(value, list) else [value]
        images = []
        for p in paths:
            p = Path(str(p))
            if not p.exists():
                _log(self.log_prefix, f"  Reference image not found: {p}", "warning")
                continue
            img = Image.open(str(p))
            img.load()
            # RGBA in, RGBA out — the pipeline converts anyway, but doing it here
            # means the log below reports what the VAE will actually see.
            had_alpha = img.mode in ("RGBA", "LA", "P")
            img = img.convert("RGBA")
            images.append(img)
            _log(self.log_prefix,
                 f"  Reference {p.name}: {img.width}x{img.height} "
                 f"{'with alpha' if had_alpha else 'opaque'}, passed through unresized")

        if not images:
            return None
        # Upstream supports up to 10 references; more is not an error worth
        # failing a run over, but it is worth saying out loud.
        if len(images) > 10:
            _log(self.log_prefix,
                 f"  {len(images)} reference images — Qwen-Image-2.1 is documented "
                 f"for up to 10; the extras may be ignored or degrade the result",
                 "warning")
        _log(self.log_prefix, f"  Mapped {semantic_name} → edit_image: {len(images)} images")
        return {"edit_image": images}

    @classmethod
    def _snap(cls, value: int) -> int:
        """Round up to the DiT's /32 latent grid."""
        v = int(value or 0)
        return max(cls.GRID, ((v + cls.GRID - 1) // cls.GRID) * cls.GRID)

    @staticmethod
    def _resolve_source_image(
        control_image: Optional[Union[Path, List[Path]]],
    ) -> Optional[Path]:
        """First existing reference image, the master for dimension inheritance."""
        if not control_image:
            return None
        paths = control_image if isinstance(control_image, list) else [control_image]
        for candidate in paths:
            if not candidate:
                continue
            p = Path(str(candidate))
            if p.exists() and p.is_file():
                return p
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _warn_if_lora_missed(self, pipe):
        """Catch a LoRA that loaded onto nothing.

        DiffSynth's hot-load path matches LoRA tensors to modules by name and
        simply reports how many it patched — a LoRA whose keys match none of
        them is a silent no-op, not an error. Every Qwen LoRA on disk was
        trained against the qwen_image DiT, whose module names this DiT does not
        share, and the LoRA dropdown does not filter scanned (uncurated) files
        by model. So the reachable failure is a user picking one here and
        getting an unchanged image with nothing in the log to explain it.
        """
        dit = getattr(pipe, "dit", None)
        if dit is None:
            return
        patched = sum(
            1 for _, m in dit.named_modules()
            if getattr(m, "lora_A_weights", None)
        )
        if patched == 0:
            _log(self.log_prefix,
                 "LoRA(s) were requested but patched 0 layers of the Qwen-Image-2.1 "
                 "DiT — they are almost certainly trained for the qwen_image DiT, "
                 "which this model does not share. The generation will run as if "
                 "no LoRA were selected.", "error")
        else:
            _log(self.log_prefix, f"LoRA patched {patched} layers", "success")

    @staticmethod
    def _alpha_summary(image: Image.Image) -> str:
        """One line on whether the model actually produced transparency."""
        if image.mode != "RGBA":
            return "no alpha channel"
        alpha = image.getchannel("A")
        lo, hi = alpha.getextrema()
        if lo == hi == 255:
            return "alpha fully opaque (prompt did not ask for a cut-out)"
        if hi == 0:
            return "alpha fully transparent — check the prompt"
        return f"transparency present (alpha range {lo}-{hi})"
