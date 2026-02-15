"""
DiffSynth Backend for FUK

Unified generation backend using DiffSynth-Studio 2.0 pipelines.
Fully data-driven — all model definitions live in models.json.
Add new models by editing JSON, no code changes needed.
"""

from __future__ import annotations  # Makes type hints lazy (Python 3.7+)

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Vendor path setup – core/ and vendor/ are siblings under the same root
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # …/fuk/fuk/core
_VENDOR_DIR = _THIS_DIR.parent / "vendor"            # …/fuk/fuk/vendor
_DIFFSYNTH_DIR = _VENDOR_DIR / "DiffSynth-Studio"

if _DIFFSYNTH_DIR.exists():
    if str(_DIFFSYNTH_DIR) not in sys.path:
        sys.path.insert(0, str(_DIFFSYNTH_DIR))
else:
    print(f"⚠  DiffSynth-Studio not found at {_DIFFSYNTH_DIR}")
    print(f"   Expected vendor layout: vendor/DiffSynth-Studio/diffsynth/...")

# ---------------------------------------------------------------------------

import os
import torch
from typing import Optional, Dict, Any, List, Union
import json
import time
from PIL import Image

#from latent_manager import LatentManager


# ---------------------------------------------------------------------------
# Pipeline class registry (populated after DiffSynth import)
# ---------------------------------------------------------------------------
PIPELINE_CLASSES = {}


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


class DiffSynthBackend:
    """
    Unified, data-driven backend for all DiffSynth generation.

    All model definitions come from models.json — adding a new model
    is a JSON edit, not a code change.
    """

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.models_config = self._load_config("models.json")
        self.defaults_config = self._load_config("defaults.json")

        # Setup DiffSynth environment and import pipelines
        self._setup_diffsynth_env()

        # Build alias lookup  (e.g. "i2v-A14B" → "wan_i2v_a14b")
        self._alias_map: Dict[str, str] = {}
        for key, entry in self.models_config.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            for alias in entry.get("aliases", []):
                self._alias_map[alias] = key

        # Cached pipelines keyed by model_type
        self.pipelines: Dict[str, Any] = {}
        self._active_user_lora: Dict[str, str] = {}  # cache_key → user lora_path currently loaded
        self._model_lora_config: Dict[str, dict] = {}  # cache_key → model lora config from models.json
        #self.latent_manager = LatentManager()

        # Scan for available LoRA files
        self._lora_registry: Dict[str, dict] = {}
        self._scan_lora_dirs()

        model_keys = [k for k in self.models_config if not k.startswith("_") and isinstance(self.models_config[k], dict)]
        _log("BACKEND", "DiffSynth Backend initialized", "success")
        _log("BACKEND", f"Config: {config_dir}")
        _log("BACKEND", f"Models: {model_keys}")
        if self._alias_map:
            _log("BACKEND", f"Aliases: {dict(self._alias_map)}")
        if self._lora_registry:
            _log("BACKEND", f"LoRAs: {list(self._lora_registry.keys())}")

    # ------------------------------------------------------------------
    # DiffSynth environment setup
    # ------------------------------------------------------------------

    def _setup_diffsynth_env(self):
        """
        Setup DiffSynth environment variables and import pipelines.
        Called after configs are loaded so we can read models path from defaults.json.
        """
        global PIPELINE_CLASSES
        
        # Read models base path from config (with fallback)
        # Try top-level first (FUK convention), then nested paths.models_root
        models_root = self.defaults_config.get("models_root")
        if not models_root:
            models_root = self.defaults_config.get("paths", {}).get("models_root", "/home/brad/ai/models")
        
        # Store for constructing local model paths
        self.models_root = Path(models_root)
        
        # Setup environment
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = models_root
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "TRUE"
        
        _log("BACKEND", f"Models base path: {models_root}")
        
        # Import DiffSynth pipelines (must happen after env setup)
        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig as WanModelConfig
        
        # Store ModelConfig classes as instance attributes for use in other methods
        self.ModelConfig = ModelConfig
        self.WanModelConfig = WanModelConfig
        
        # Populate pipeline registry
        PIPELINE_CLASSES["qwen"] = QwenImagePipeline
        PIPELINE_CLASSES["wan"] = WanVideoPipeline

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_config(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            _log("BACKEND", f"Config not found: {path}", "warning")
            return {}
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def _scan_lora_dirs(self):
        """Scan configured directories for .safetensors LoRA files."""
        lora_dirs = self.defaults_config.get("lora_dirs", [])
        self._lora_registry = {}
        for dir_path in lora_dirs:
            d = Path(dir_path).expanduser()
            if not d.exists():
                _log("BACKEND", f"LoRA dir not found: {d}", "warning")
                continue
            for f in sorted(d.rglob("*.safetensors")):
                # Key: parent_dir/stem for uniqueness (e.g. "qwen/my_style")
                rel = f.relative_to(d)
                if len(rel.parts) > 1:
                    key = f"{rel.parent}/{f.stem}"
                else:
                    key = f.stem
                self._lora_registry[key] = {
                    "path": str(f),
                    "name": f.stem,
                    "key": key,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                }
        if lora_dirs:
            _log("BACKEND", f"Scanned {len(lora_dirs)} LoRA dir(s) → {len(self._lora_registry)} file(s)")

    def get_available_loras(self) -> list:
        """Return list of discovered LoRA files for the config endpoint."""
        return list(self._lora_registry.values())

    def _resolve_lora_path(self, lora_name: str) -> Optional[str]:
        """
        Resolve a LoRA name/key to a filesystem path.
        
        Lookup order:
          1. Direct match in registry by key
          2. Match by stem name (first match)
          3. Treat as literal file path
        """
        if not lora_name:
            return None

        # 1. Exact key match
        if lora_name in self._lora_registry:
            return self._lora_registry[lora_name]["path"]

        # 2. Match by stem (friendly name)
        for entry in self._lora_registry.values():
            if entry["name"] == lora_name:
                return entry["path"]

        # 3. Literal path fallback
        p = Path(lora_name).expanduser()
        if p.exists() and p.suffix == ".safetensors":
            return str(p)

        _log("BACKEND", f"LoRA not found: {lora_name}", "warning")
        return None

    def _load_model_lora(self, pipeline, lora_cfg: dict):
        """
        Load a model-bundled LoRA (e.g. Control-Union) from local models directory.
        
        Constructs path as: {models_root}/{model_id}/{pattern}
        All models should be pre-downloaded via download_models.py.
        """
        model_id = lora_cfg["model_id"]
        pattern = lora_cfg["pattern"]
        target = lora_cfg.get("target", "dit")
        alpha = lora_cfg.get("alpha", 1.0)

        # Construct local path (models are pre-downloaded)
        lora_path = self.models_root / model_id / pattern
        
        if not lora_path.exists():
            raise FileNotFoundError(
                f"Model LoRA not found: {lora_path}\n"
                f"Run download_models.py to download {model_id}"
            )

        _log("BACKEND", f"Loading model LoRA: {model_id}/{pattern} → pipe.{target} (α={alpha})")

        target_module = getattr(pipeline, target, None)
        if target_module is None:
            raise ValueError(f"Pipeline has no attribute '{target}' for LoRA target")

        # Use standard DiffSynth API (supports alpha parameter)
        pipeline.load_lora(target_module, str(lora_path), alpha=alpha)
        _log("BACKEND", f"Model LoRA loaded: {lora_path.name}", "success")

    def _apply_user_lora(self, pipeline, cache_key: str, lora_path: str, alpha: float = 1.0):
        """
        Load a user-selected LoRA onto a pipeline.
        
        For models with bundled LoRAs: clears all, reloads model LoRA, then adds user LoRA on top.
        For models without bundled LoRAs: just loads the user LoRA.
        
        Tracks state to avoid unnecessary reload if same LoRA is already active.
        """
        current = self._active_user_lora.get(cache_key)
        if current == lora_path:
            _log("BACKEND", f"User LoRA already loaded: {Path(lora_path).stem}")
            return

        # Clear all LoRAs (both model and user)
        try:
            pipeline.clear_lora()
            if current:
                _log("BACKEND", f"Cleared previous user LoRA: {Path(current).stem}")
        except Exception as e:
            _log("BACKEND", f"clear_lora() failed (non-fatal): {e}", "warning")

        # Reload model LoRA if this pipeline has one
        model_lora_cfg = self._model_lora_config.get(cache_key)
        if model_lora_cfg:
            self._load_model_lora(pipeline, model_lora_cfg)

        # Load user LoRA on top
        target_module = getattr(pipeline, "dit", None)
        if target_module is None:
            _log("BACKEND", "Pipeline has no 'dit' — cannot load user LoRA", "error")
            return

        _log("BACKEND", f"Loading user LoRA: {Path(lora_path).stem} (α={alpha})")
        pipeline.load_lora(target_module, lora_path, alpha=alpha)
        self._active_user_lora[cache_key] = lora_path
        _log("BACKEND", f"User LoRA active: {Path(lora_path).stem}", "success")

    def _clear_user_lora(self, pipeline, cache_key: str):
        """
        Clear user LoRA if one is active.
        
        For models with bundled LoRAs: clears all, then reloads model LoRA.
        For models without bundled LoRAs: just clears.
        """
        if cache_key not in self._active_user_lora:
            return
            
        try:
            pipeline.clear_lora()
            _log("BACKEND", f"Cleared user LoRA: {Path(self._active_user_lora[cache_key]).stem}")
        except Exception:
            pass
        
        # Reload model LoRA if this pipeline has one
        model_lora_cfg = self._model_lora_config.get(cache_key)
        if model_lora_cfg:
            self._load_model_lora(pipeline, model_lora_cfg)
        
        del self._active_user_lora[cache_key]

    def _resolve_vram_preset(self, preset_name: str = None) -> tuple:
        """
        Resolve a VRAM preset name to (config_dict_or_None, buffer_gb).
        
        DiffSynth's 4-state offload system:
          offload    → where params live when idle (CPU RAM or disk)
          onload     → intermediate loading state  
          preparing  → pre-computation dtype/device
          computation → active inference (always cuda + bf16)
        
        Returns (None, buffer) for "none" preset (everything stays on GPU).
        """
        vram_section = self.defaults_config.get("vram", {})
        presets = vram_section.get("presets", {})
        
        name = preset_name or vram_section.get("preset", "low")
        preset = presets.get(name)
        if preset is None:
            _log("BACKEND", f"Unknown VRAM preset '{name}', falling back to 'low'", "warning")
            preset = presets.get("low", {})
            name = "low"
        
        config = preset.get("config")
        buffer = preset.get("buffer_gb", 2.0)
        
        if config is None:
            _log("BACKEND", f"VRAM preset: {name} — no offloading")
            return None, buffer
        
        _log("BACKEND", f"VRAM preset: {name} — {preset.get('label', name)}")
        
        # Map string dtype names to torch dtypes
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8": torch.float8_e4m3fn,
        }
        
        def resolve(val):
            """Resolve dtype string. 'disk' stays as string (DiffSynth convention)."""
            if isinstance(val, str) and val != "disk":
                return dtype_map.get(val, torch.bfloat16)
            return val  # "disk" or already a torch dtype
        
        resolved = {
            "offload_dtype":      resolve(config["offload_dtype"]),
            "offload_device":     config["offload_device"],
            "onload_dtype":       resolve(config["onload_dtype"]),
            "onload_device":      config["onload_device"],
            "preparing_dtype":    resolve(config["preparing_dtype"]),
            "preparing_device":   config["preparing_device"],
            "computation_dtype":  resolve(config["computation_dtype"]),
            "computation_device": config["computation_device"],
        }
        return resolved, buffer

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def resolve_model_type(self, name: str) -> str:
        """
        Resolve any name / task / alias to a canonical models.json key.
        Accepts: "wan_i2v_a14b", "i2v-A14B", "qwen_image_2509_edit", etc.
        """
        if name in self.models_config and not name.startswith("_"):
            return name
        if name in self._alias_map:
            return self._alias_map[name]
        raise ValueError(
            f"Unknown model/task: '{name}'. "
            f"Keys: {[k for k in self.models_config if not k.startswith('_') and isinstance(self.models_config[k], dict)]}  "
            f"Aliases: {list(self._alias_map.keys())}"
        )

    def get_model_entry(self, model_type: str) -> dict:
        """Get the models.json entry for a resolved model_type."""
        model_type = self.resolve_model_type(model_type)
        return self.models_config[model_type]

    def list_models(self, pipeline_filter: str = None) -> Dict[str, dict]:
        """List available models, optionally filtered by pipeline type."""
        return {
            k: v for k, v in self.models_config.items()
            if not k.startswith("_") and isinstance(v, dict) and v.get("pipeline")
            and (pipeline_filter is None or v["pipeline"] == pipeline_filter)
        }

    # ------------------------------------------------------------------
    # Pipeline construction — fully data-driven
    # ------------------------------------------------------------------

    def _build_model_configs(self, entry: dict, vram_config: dict = None) -> list:
        """Build ModelConfig list from a model entry's components.
        
        Args:
            entry: The models.json entry
            vram_config: Resolved VRAM offload dict, or None for no offloading
        """
        primary_id = entry["model_id"]

        configs = []
        for comp in entry.get("components", []):
            mid = comp.get("model_id", primary_id)
            kwargs = dict(
                model_id=mid,
                origin_file_pattern=comp["pattern"],
            )
            # Only apply offload config when a preset is active
            if vram_config is not None:
                kwargs.update(vram_config)
            configs.append(self.ModelConfig(**kwargs))
        return configs

    def _build_extra_config(self, entry: dict, key: str) -> Optional[ModelConfig]:
        """Build a tokenizer_config or processor_config from entry."""
        cfg = entry.get(key)
        if cfg is None:
            return None
        mid = cfg.get("model_id", entry["model_id"])
        return self.ModelConfig(model_id=mid, origin_file_pattern=cfg["pattern"])

    def get_pipeline(self, model_type: str, vram_preset: str = None):
        """Get or create a cached pipeline.
        
        Cache key includes VRAM preset — changing preset forces reload.
        """
        model_type = self.resolve_model_type(model_type)
        
        # Resolve preset
        vram_config, buffer_gb = self._resolve_vram_preset(vram_preset)
        active_preset = vram_preset or self.defaults_config.get("vram", {}).get("preset", "low")
        cache_key = f"{model_type}:{active_preset}"

        if cache_key in self.pipelines:
            _log("BACKEND", f"Using cached pipeline: {cache_key}")
            return self.pipelines[cache_key]

        # If same model with different preset is cached, evict it
        stale = [k for k in self.pipelines if k.startswith(f"{model_type}:")]
        for k in stale:
            _log("BACKEND", f"Evicting pipeline (preset changed): {k}")
            del self.pipelines[k]
            self._active_user_lora.pop(k, None)
        if stale and torch.cuda.is_available():
            torch.cuda.empty_cache()

        entry = self.get_model_entry(model_type)
        pipeline_name = entry["pipeline"]
        PipelineCls = PIPELINE_CLASSES.get(pipeline_name)
        if PipelineCls is None:
            raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {list(PIPELINE_CLASSES.keys())}")

        model_configs = self._build_model_configs(entry, vram_config=vram_config)

        _log("BACKEND", f"Loading pipeline: {model_type} (preset: {active_preset})")
        _log("BACKEND", f"  Model ID: {entry['model_id']}")
        _log("BACKEND", f"  Components: {len(model_configs)}")

        # from_pretrained kwargs
        kwargs = dict(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
        )

        # VRAM limit — only set when offloading is active
        if vram_config is not None and torch.cuda.is_available():
            total_vram = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
            vram_limit = total_vram - buffer_gb
            kwargs["vram_limit"] = vram_limit
            _log("BACKEND", f"  VRAM: {total_vram:.1f}GB total − {buffer_gb}GB buffer = {vram_limit:.1f}GB limit")
        else:
            _log("BACKEND", f"  VRAM: no limit (preset={active_preset})")

        # Tokenizer or Processor (mutually exclusive in practice)
        tok = self._build_extra_config(entry, "tokenizer")
        if tok:
            kwargs["tokenizer_config"] = tok
        proc = self._build_extra_config(entry, "processor")
        if proc:
            kwargs["processor_config"] = proc

        pipeline = PipelineCls.from_pretrained(**kwargs)
        _log("BACKEND", f"Pipeline loaded: {cache_key}", "success")

        # Model-bundled LoRA (e.g. Control-Union) — load and track config
        lora_cfg = entry.get("lora")
        if lora_cfg:
            self._load_model_lora(pipeline, lora_cfg)
            self._model_lora_config[cache_key] = lora_cfg  # Track for reloading

        self.pipelines[cache_key] = pipeline
        return pipeline

    def unload_pipeline(self, model_type: str):
        """Unload all cached pipelines for a model type (any preset)."""
        model_type = self.resolve_model_type(model_type)
        stale = [k for k in self.pipelines if k.startswith(f"{model_type}:")]
        for k in stale:
            del self.pipelines[k]
            self._active_user_lora.pop(k, None)
            self._model_lora_config.pop(k, None)
        if stale and torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log("BACKEND", f"Unloaded pipeline(s): {stale}")

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate_image(
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
        # LoRA (standalone / user-selected, not model-bundled)
        lora: Optional[str] = None,
        lora_multiplier: float = 1.0,
        # Control / edit inputs
        control_image: Optional[Union[Path, List[Path]]] = None,
        context_image: Optional[Any] = None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate an image.  Drop-in replacement for QwenImageGenerator.generate().
        """
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(model)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # Merge with defaults.json
        defaults = self.defaults_config.get("image", {})
        width = width or defaults.get("width", 1024)
        height = height or defaults.get("height", 1024)
        num_steps = steps or infer_steps or defaults.get("steps", 28)
        effective_cfg = cfg_scale or guidance_scale or defaults.get("cfg_scale", 5.0)
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = denoising_strength if denoising_strength is not None else defaults.get("denoising_strength", 0.85)

        _log("BACKEND", "=" * 60)
        _log("BACKEND", "IMAGE GENERATION")
        _log("BACKEND", f"  Model: {model_type} — {entry.get('description', '')}")
        _log("BACKEND", f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        _log("BACKEND", f"  Size: {width}x{height}  Steps: {num_steps}  CFG: {effective_cfg}  Denoise: {denoise}  Seed: {seed}")
        exponential_shift_mu = kwargs.get('exponential_shift_mu')
        if exponential_shift_mu is not None:
            _log("BACKEND", f"  Exponential shift mu: {exponential_shift_mu}")
        if lora:
            _log("BACKEND", f"  User LoRA: {lora} (α={lora_multiplier})")
        _log("BACKEND", "=" * 60)

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)

        # --- User LoRA ---
        active_preset = vram_preset or self.defaults_config.get("vram", {}).get("preset", "low")
        cache_key = f"{model_type}:{active_preset}"
        
        if lora:
            lora_path = self._resolve_lora_path(lora)
            if lora_path:
                self._apply_user_lora(pipe, cache_key, lora_path, alpha=lora_multiplier)
            else:
                _log("BACKEND", f"LoRA not found, skipping: {lora}", "warning")
        else:
            # No LoRA requested — clear any previously loaded user LoRA
            self._clear_user_lora(pipe, cache_key)

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

        # Exponential shift mu (Qwen-specific timestep control)
        exponential_shift_mu = kwargs.get('exponential_shift_mu')
        if exponential_shift_mu is not None:
            pipe_kwargs['exponential_shift_mu'] = exponential_shift_mu
            _log("BACKEND", f"  Exponential shift mu: {exponential_shift_mu}")

        # Negative prompt
        if "negative_prompt" in supports and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # Map semantic inputs to model-specific parameters
        semantic_inputs = {}
        if control_image:
            # For edit_targets, pass as-is (can be list or single)
            semantic_inputs["edit_targets"] = control_image
            
            # For control_input (single image), extract first if list
            if isinstance(control_image, list):
                semantic_inputs["control_input"] = control_image[0] if control_image else None
            else:
                semantic_inputs["control_input"] = control_image
                
        if context_image:
            semantic_inputs["control_input"] = context_image
            
        mapped_inputs = self._map_inputs(model_type, semantic_inputs, width, height)
        pipe_kwargs.update(mapped_inputs)

        # Merge pipeline_kwargs from models.json
        pipe_kwargs.update(pipe_defaults)

        # Setup latent capture if requested
        latent_path = None
        cleanup_hook = None
        if save_latent:
            latent_dir = output_path.parent / "latents"
            latent_dir.mkdir(exist_ok=True)
            latent_path = latent_dir / f"{output_path.stem}.latent.pt"
            cleanup_hook = self._capture_latent_hook(pipe, latent_path)
            _log("BACKEND", f"Latent capture enabled → {latent_path}")

        try:
            image = pipe(**pipe_kwargs)
            image.save(output_path)

            elapsed = time.time() - start_time
            _log("BACKEND", f"Image saved: {output_path} ({elapsed:.1f}s)", "success")

            return {
                "success": True,
                "image": output_path,
                "latent": latent_path,
                "seed_used": seed,
                "params": {
                    "prompt": prompt, "seed": seed,
                    "steps": num_steps, "size": f"{width}x{height}",
                    "cfg_scale": effective_cfg,
                    "denoising_strength": denoise,
                },
            }
        except Exception as e:
            _log("BACKEND", f"Image generation failed: {e}", "error")
            raise
        finally:
            # Always cleanup hook if it was installed
            if cleanup_hook:
                cleanup_hook()

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def generate_video(
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
        # Progress
        progress_callback=None,
        # VRAM
        vram_preset: Optional[str] = None,
        # Misc
        save_latent: bool = True,
        infer_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a video.  Drop-in replacement for WanVideoGenerator.generate_video().
        """
        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_type = self.resolve_model_type(task)
        entry = self.get_model_entry(model_type)
        supports = set(entry.get("supports", []))
        pipe_defaults = dict(entry.get("pipeline_kwargs", {}))

        # Merge with defaults.json
        defaults = self.defaults_config.get("video", {})
        width = width or defaults.get("width") or 832
        height = height or defaults.get("height") or 480
        num_frames = video_length or defaults.get("video_length", 81)
        num_steps = steps or infer_steps or defaults.get("steps", 50)
        effective_cfg = cfg_scale or guidance_scale or defaults.get("cfg_scale", 6.0)
        negative_prompt = negative_prompt or defaults.get("negative_prompt", "")
        denoise = denoising_strength if denoising_strength is not None else defaults.get("denoising_strength", 1.0)

        # Wan-specific params from kwargs or defaults
        sigma_shift = kwargs.get('sigma_shift') if kwargs.get('sigma_shift') is not None else defaults.get("sigma_shift", 5.0)
        motion_bucket_id = kwargs.get('motion_bucket_id') if 'motion_bucket_id' in kwargs else defaults.get("motion_bucket_id")
        sliding_window_size = defaults.get("sliding_window_size")
        sliding_window_stride = defaults.get("sliding_window_stride")

        if lora:
            _log("BACKEND", f"  LoRA: {lora} (α={lora_multiplier})")

        _log("BACKEND", "=" * 60)
        _log("BACKEND", "VIDEO GENERATION")
        _log("BACKEND", f"  Task: {task} → {model_type} — {entry.get('description', '')}")
        _log("BACKEND", f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        _log("BACKEND", f"  Size: {width}x{height}  Frames: {num_frames}  Steps: {num_steps}  CFG: {effective_cfg}  Denoise: {denoise}  Seed: {seed}")
        _log("BACKEND", f"  sigma_shift: {sigma_shift}  motion_bucket_id: {motion_bucket_id}")
        if image_path:
            _log("BACKEND", f"  Input image: {image_path}")
        if control_path:
            _log("BACKEND", f"  Control path: {control_path}")
        if pipe_defaults:
            _log("BACKEND", f"  Pipeline kwargs: {pipe_defaults}")
        _log("BACKEND", "=" * 60)

        if progress_callback:
            progress_callback("loading_models", 0, 1)

        pipe = self.get_pipeline(model_type, vram_preset=vram_preset)

        # --- User LoRA ---
        active_preset = vram_preset or self.defaults_config.get("vram", {}).get("preset", "low")
        cache_key = f"{model_type}:{active_preset}"
        
        if lora:
            lora_path = self._resolve_lora_path(lora)
            if lora_path:
                self._apply_user_lora(pipe, cache_key, lora_path, alpha=lora_multiplier)
            else:
                _log("BACKEND", f"LoRA not found, skipping: {lora}", "warning")
        else:
            # No LoRA requested — clear any previously loaded user LoRA
            self._clear_user_lora(pipe, cache_key)

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

        # Map semantic inputs to model-specific parameters
        semantic_inputs = {}
        if image_path:
            semantic_inputs["reference_image"] = image_path
        if control_path:
            semantic_inputs["control_input"] = control_path
            
        mapped_inputs = self._map_inputs(model_type, semantic_inputs, width, height)
        pipe_kwargs.update(mapped_inputs)

        # Tiled (explicit from supports, may also come from pipe_defaults)
        if "tiled" in supports and "tiled" not in pipe_defaults:
            pipe_kwargs["tiled"] = True

        # Merge pipeline_kwargs from models.json (switch_DiT_boundary, tiled, etc.)
        pipe_kwargs.update(pipe_defaults)

        # Setup latent capture if requested
        latent_path = None
        cleanup_hook = None
        if save_latent:
            latent_dir = output_path.parent / "latents"
            latent_dir.mkdir(exist_ok=True)
            latent_path = latent_dir / f"{output_path.stem}.latent.pt"
            cleanup_hook = self._capture_latent_hook(pipe, latent_path)
            _log("BACKEND", f"Latent capture enabled → {latent_path}")

        try:
            video = pipe(**pipe_kwargs)

            if progress_callback:
                progress_callback("saving", 0, 1)

            from diffsynth.utils.data import save_video
            save_video(video, str(output_path), fps=24, quality=5)

            elapsed = time.time() - start_time
            _log("BACKEND", f"Video saved: {output_path} ({elapsed:.1f}s)", "success")

            if progress_callback:
                progress_callback("complete", 1, 1)

            return {
                "success": True,
                "video": output_path,
                "latent": latent_path,
                "seed_used": seed,
                "params": {
                    "prompt": prompt, "seed": seed,
                    "steps": num_steps, "size": f"{width}x{height}",
                    "frames": num_frames, "cfg_scale": effective_cfg,
                    "denoising_strength": denoise, "sigma_shift": sigma_shift,
                },
            }
        except Exception as e:
            _log("BACKEND", f"Video generation failed: {e}", "error")
            raise
        finally:
            # Always cleanup hook if it was installed
            if cleanup_hook:
                cleanup_hook()

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _load_image(self, path, width: int = None, height: int = None) -> Optional[Image.Image]:
        """Load a PIL Image, optionally resizing."""
        if path is None:
            return None
            
        # Handle unexpected list (should be caught earlier, but be defensive)
        if isinstance(path, list):
            if not path:
                return None
            _log("BACKEND", f"Warning: _load_image received list, using first item", "warning")
            path = path[0]
            
        p = Path(str(path))
        if not p.exists():
            _log("BACKEND", f"Image not found: {p}", "warning")
            return None
        img = Image.open(str(p))
        if width and height:
            img = img.resize((width, height))
        return img

    def _resolve_image_list(
        self, control_image: Optional[Union[Path, List[Path]]]
    ) -> Optional[List[Image.Image]]:
        """Convert control_image path(s) to a list of PIL Images for edit mode."""
        if not control_image:
            return None
        paths = control_image if isinstance(control_image, list) else [control_image]
        images = []
        for p in paths:
            p = Path(str(p))
            if p.exists():
                images.append(Image.open(str(p)))
        return images if images else None

    def _load_video_data(self, source, height: int, width: int):
        """Load a video source into DiffSynth VideoData for VACE."""
        from diffsynth.utils.data import VideoData

        if hasattr(source, 'raw_data'):
            return source

        p = Path(str(source))
        if p.exists() and p.suffix in ('.mp4', '.avi', '.mov', '.mkv'):
            return VideoData(str(p), height=height, width=width)
        if p.is_dir():
            return VideoData(str(p), height=height, width=width)

        _log("BACKEND", f"Could not load video data from: {source}", "warning")
        return None

    def _map_inputs(
        self,
        model_type: str,
        semantic_inputs: dict,
        width: int,
        height: int,
    ) -> dict:
        """
        Map semantic inputs to model-specific parameters using parameter_map.
        
        Args:
            model_type: Model identifier (e.g. "qwen_image_edit_2511")
            semantic_inputs: Dict with semantic keys like:
                - reference_image: Path to style/reference image
                - control_input: Path to control video/image
                - edit_targets: Path or list of paths to images to edit
            width, height: Target dimensions for loading
            
        Returns:
            Dict of model-specific pipe() kwargs ready to pass to pipeline
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
            
            # Map to model-specific parameter and format
            if model_param == "edit_image":
                # Qwen-Edit wants list of PIL Images
                images = self._resolve_image_list(value)
                if images:
                    pipe_kwargs["edit_image"] = images
                    _log("BACKEND", f"  Mapped edit_targets → edit_image: {len(images)} images")
                    
            elif model_param == "context_image":
                # Control-Union wants single PIL Image
                img = self._load_image(value, width, height)
                if img:
                    pipe_kwargs["context_image"] = img
                    _log("BACKEND", f"  Mapped control_input → context_image")
                    
            elif model_param == "input_image":
                # Wan i2v wants single PIL Image
                img = self._load_image(value, width, height)
                if img:
                    pipe_kwargs["input_image"] = img
                    _log("BACKEND", f"  Mapped reference_image → input_image")
                    
            elif model_param == "vace_video":
                # Wan VACE wants VideoData for control
                vace = self._load_video_data(value, height, width)
                if vace:
                    pipe_kwargs["vace_video"] = vace
                    _log("BACKEND", f"  Mapped control_input → vace_video")
                    
            elif model_param == "vace_reference_image":
                # Wan VACE wants PIL Image for reference
                img = self._load_image(value, width, height)
                if img:
                    pipe_kwargs["vace_reference_image"] = img
                    _log("BACKEND", f"  Mapped reference_image → vace_reference_image")
            
            else:
                _log("BACKEND", f"  Unknown parameter mapping: {semantic_name} → {model_param}", "warning")
        
        return pipe_kwargs

    # ------------------------------------------------------------------
    # Latent capture and management
    # ------------------------------------------------------------------

    def _save_latent_tensor(self, latent: torch.Tensor, output_path: Path) -> Path:
        """
        Save a raw latent tensor to disk using torch.save().
        
        Args:
            latent: The latent tensor from diffusion (before VAE decode)
            output_path: Where to save the .pt file
            
        Returns:
            Path to saved latent file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with metadata for later reconstruction
        latent_data = {
            'latent': latent.cpu(),  # Move to CPU for storage
            'shape': list(latent.shape),
            'dtype': str(latent.dtype),
        }
        
        torch.save(latent_data, str(output_path))
        _log("BACKEND", f"Latent tensor saved: {output_path} (shape: {latent.shape})")
        return output_path

    def _capture_latent_hook(self, pipe, save_path: Path):
        """
        Install a hook to capture latent before VAE decode.
        
        Captures a CPU copy of the latent with minimal GPU disruption.
        The actual torch.save() happens in cleanup, AFTER decode completes,
        so disk I/O doesn't block the GPU pipeline during generation.
        
        Args:
            pipe: The pipeline instance
            save_path: Where to save the captured latent
            
        Returns:
            Function to remove the hook and save latent
        """
        original_decode = pipe.vae.decode
        captured = {}  # Will hold {'latent': cpu_tensor, ...} after first decode
        
        def hooked_decode(latent, *args, **kwargs):
            if not captured:  # Only capture once
                # detach().cpu() copies to CPU without keeping a GPU clone
                captured['latent'] = latent.detach().cpu()
                captured['shape'] = list(latent.shape)
                captured['dtype'] = str(latent.dtype)
            return original_decode(latent, *args, **kwargs)
        
        # Install hook
        pipe.vae.decode = hooked_decode
        
        # Return cleanup function — saves to disk AFTER generation
        def cleanup():
            pipe.vae.decode = original_decode
            if captured:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(captured, str(save_path))
                _log("BACKEND", f"Latent tensor saved: {save_path} (shape: {captured['shape']})")
                captured.clear()  # Free the CPU tensor
            
        return cleanup

    def load_latent_tensor(self, latent_path: Path) -> torch.Tensor:
        """
        Load a latent tensor saved with _save_latent_tensor().
        
        Args:
            latent_path: Path to the .pt file
            
        Returns:
            The latent tensor (on CPU)
        """
        if not Path(latent_path).exists():
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        latent_data = torch.load(str(latent_path), map_location='cpu')
        latent = latent_data['latent']
        _log("BACKEND", f"Latent loaded: {latent_path} (shape: {latent.shape})")
        return latent

    def decode_latent_direct(
        self,
        latent_path: Path,
        output_path: Path,
        model_type: str = "qwen_image",
    ) -> Path:
        """
        Decode a saved latent tensor directly to an image/video file.
        
        This bypasses the full generation pipeline and just runs VAE decode
        on a pre-saved latent, useful for re-rendering at different quality
        settings or formats without regenerating.
        
        Args:
            latent_path: Path to saved .pt latent file
            output_path: Where to save the decoded output
            model_type: Which model's VAE to use for decoding
            
        Returns:
            Path to the decoded output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load latent
        latent = self.load_latent_tensor(latent_path)
        
        # Get pipeline for its VAE
        pipe = self.get_pipeline(model_type)
        
        # Move to GPU and decode
        device = next(pipe.vae.parameters()).device
        latent = latent.to(device)
        
        _log("BACKEND", f"Decoding latent with {model_type} VAE...")
        
        with torch.no_grad():
            decoded = pipe.vae.decode(latent)
        
        # Save based on output format
        if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # Image output
            from diffsynth.utils.data import save_image
            save_image(decoded, str(output_path))
        elif output_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Video output
            from diffsynth.utils.data import save_video
            save_video(decoded, str(output_path), fps=24, quality=5)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        _log("BACKEND", f"Decoded output saved: {output_path}", "success")
        return output_path

    # ------------------------------------------------------------------
    # Latent → EXR decode
    # ------------------------------------------------------------------

    def decode_latent_to_exr(
        self,
        latent_path: Path,
        output_exr: Path,
        model_type: str = "qwen_image",
        layer_name: str = "beauty",
    ) -> Path:
        """Decode a saved latent directly to EXR (lossless path)."""
        pipe = self.get_pipeline(model_type)
        return self.latent_manager.decode_to_exr(
            latent_path=latent_path,
            vae=pipe.vae,
            output_exr=output_exr,
            layer_name=layer_name,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release GPU memory and clear cached pipelines."""
        self.pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log("BACKEND", "Backend cleanup complete", "success")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing DiffSynth Backend...\n")
    config_dir = Path(__file__).resolve().parent.parent / "config"

    try:
        backend = DiffSynthBackend(config_dir)
        print(f"\n  Vendor dir:  {_VENDOR_DIR}")
        print(f"  DiffSynth:   {_DIFFSYNTH_DIR} ({'found' if _DIFFSYNTH_DIR.exists() else 'NOT FOUND'})")

        # Test model resolution
        print("\n  Model resolution:")
        test_names = [
            "qwen_image", "qwen_image_2512", "qwen_edit",
            "qwen_control_union", "qwen_image_2509_edit",
            "wan_t2v_14b", "t2v-14B", "i2v-14B", "i2v-A14B",
            "wan_vace_fun_a14b", "t2v-14B-FC", "i2v-14B-FC",
        ]
        for name in test_names:
            try:
                resolved = backend.resolve_model_type(name)
                entry = backend.get_model_entry(resolved)
                supports = entry.get("supports", [])
                print(f"    {name:30s} → {resolved:25s} [{entry['pipeline']}] supports: {supports}")
            except ValueError as e:
                print(f"    {name:30s} → ERROR: {e}")

        # List by pipeline type
        print(f"\n  Qwen models: {list(backend.list_models('qwen').keys())}")
        print(f"  Wan models:  {list(backend.list_models('wan').keys())}")

    except Exception as e:
        print(f"✗ Init failed: {e}")
        import traceback
        traceback.print_exc()