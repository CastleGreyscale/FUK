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
        img = Image.open(str(p))
        if width and height:
            img = img.resize((width, height))
        return img

    def resolve_image_list(
        self, control_image: Optional[Union[Path, List[Path]]]
    ) -> Optional[List[Image.Image]]:
        """Convert control_image path(s) to a list of PIL Images."""
        if not control_image:
            return None
        paths = control_image if isinstance(control_image, list) else [control_image]
        images = []
        for p in paths:
            p = Path(str(p))
            if p.exists():
                images.append(Image.open(str(p)))
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
            images = self.resolve_image_list(value)
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
