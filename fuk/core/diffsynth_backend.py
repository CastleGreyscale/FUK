"""
DiffSynth Backend for FUK

Unified generation backend using DiffSynth-Studio 2.0 pipelines.
Fully data-driven — all model definitions live in models.json.
Add new models by editing JSON, no code changes needed.
"""

from __future__ import annotations  # Makes type hints lazy (Python 3.7+)

import sys
from pathlib import Path
# Ensure NVRTC can find JIT compilation builtins (pip-installed CUDA libs)
import os, site
_site_packages = site.getsitepackages()[0] if site.getsitepackages() else ""
for _nvrtc_candidate in [
    os.path.join(_site_packages, "nvidia", "cu13", "lib"),
    os.path.join(_site_packages, "nvidia", "cuda_nvrtc", "lib"),
]:
    if os.path.isdir(_nvrtc_candidate):
        os.environ["LD_LIBRARY_PATH"] = (
            _nvrtc_candidate + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        break

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

# core/ itself must be importable for bare sibling imports (perf_monitor,
# pipeline runners) even when this module is imported as core.diffsynth_backend.
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ---------------------------------------------------------------------------

import os
import time
import torch
from typing import Optional, Dict, Any, List
import json

from perf_monitor import record_timing
from encode_cache import install_encode_cache


#from latent_manager import LatentManager


# ---------------------------------------------------------------------------
# Pipeline class registry (populated after DiffSynth import)
# ---------------------------------------------------------------------------
PIPELINE_CLASSES = {}


# Both spellings of the CUDA allocator config variable. torch 2.9 warns that
# the first is deprecated but is still the only one it actually parses; see the
# evidence in _setup_diffsynth_env. Set both, trust neither warning.
ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


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
        self.defaults_config = self._load_defaults()

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
        self._active_user_loras: Dict[str, List[tuple]] = {}  # cache_key → [(path, alpha), ...]
        self._model_lora_config: Dict[str, dict] = {}  # cache_key → model lora config from models.json
        self._model_lora_alpha: Dict[str, float] = {}   # cache_key → current model LoRA alpha
        #self.latent_manager = LatentManager()

        # Scan for available LoRA files
        self._lora_registry: Dict[str, dict] = {}
        self._scan_lora_dirs()

        # --- Pipeline runners ---
        # Each runner handles one generation family (image, video, chain, etc.)
        # Runners hold a back-reference to this hub for pipeline/LoRA access.
        # Import here to avoid circular imports at module level.
        self.runners: Dict[str, Any] = {}
        self._register_runners()

        model_keys = [k for k in self.models_config if not k.startswith("_") and isinstance(self.models_config[k], dict)]
        _log("BACKEND", "DiffSynth Backend initialized", "success")
        _log("BACKEND", f"Config: {config_dir}")
        _log("BACKEND", f"Models: {model_keys}")
        if self._alias_map:
            _log("BACKEND", f"Aliases: {dict(self._alias_map)}")
        if self._lora_registry:
            _log("BACKEND", f"LoRAs: {list(self._lora_registry.keys())}")

    # ------------------------------------------------------------------
    # Runner registry
    # ------------------------------------------------------------------

    def _register_runners(self):
        import importlib, sys
        
        # Ensure core/ is importable (runner files live alongside this file)
        core_dir = str(Path(__file__).resolve().parent)
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        
        try:
            from qwen_pipeline import QwenPipelineRunner
            self.runners["image"] = QwenPipelineRunner(self)
        except ImportError as e:
            _log("BACKEND", f"QwenPipelineRunner not available: {e}", "warning")

        try:
            from wan_pipeline import WanPipelineRunner
            self.runners["video"] = WanPipelineRunner(self)
        except ImportError:
            _log("BACKEND", "WanPipelineRunner not found — using inline generate_video()", "warning")

        try:
            from flux2_pipeline import Flux2PipelineRunner
            self.runners["flux2"] = Flux2PipelineRunner(self)
        except ImportError as e:
            _log("BACKEND", f"Flux2PipelineRunner not available: {e}", "warning")

        try:
            from ltx2_pipeline import LTX2PipelineRunner
            self.runners["ltx2"] = LTX2PipelineRunner(self)
        except ImportError as e:
            _log("BACKEND", f"LTX2PipelineRunner not available: {e}", "warning")

        try:
            from minimax_h3_pipeline import MiniMaxH3PipelineRunner
            self.runners["minimax_h3"] = MiniMaxH3PipelineRunner(self)
        except ImportError as e:
            _log("BACKEND", f"MiniMaxH3PipelineRunner not available: {e}", "warning")

        try:
            from krea2_pipeline import Krea2PipelineRunner
            self.runners["krea2"] = Krea2PipelineRunner(self)
        except ImportError as e:
            _log("BACKEND", f"Krea2PipelineRunner not available: {e}", "warning")

        try:
            from threed_pipeline import ThreeDPipelineRunner
            self.runners["threed"] = ThreeDPipelineRunner(self)
        except ImportError as e:
            _log("BACKEND", f"ThreeDPipelineRunner not available: {e}", "warning")

        # Future runners:
        # from chained_pipeline import ChainedPipelineRunner
        # self.runners["chain"] = ChainedPipelineRunner(self)

        if self.runners:
            _log("BACKEND", f"Runners: {list(self.runners.keys())}")

    def run(self, family: str, **kwargs) -> Dict[str, Any]:
        """
        Dispatch generation to the appropriate runner.

        When a model is specified, auto-detects the runner from the model's
        pipeline type (e.g. flux2 models route to Flux2PipelineRunner even
        when called with family="image"). Falls back to family-based routing.

        Usage:
            backend.run("image", prompt="...", model="qwen_image", ...)
            backend.run("image", prompt="...", model="flux2_dev", ...)
            backend.run("video", prompt="...", task="wan_i2v_a14b", ...)
        """
        # Auto-detect runner from model pipeline type when possible
        model_name = kwargs.get("model") or kwargs.get("task")
        if model_name:
            try:
                model_type = self.resolve_model_type(model_name)
                entry = self.get_model_entry(model_type)
                pipeline_type = entry.get("pipeline")
                if pipeline_type and pipeline_type in self.runners:
                    return self.runners[pipeline_type].generate(**kwargs)
            except (ValueError, KeyError):
                pass

        runner = self.runners.get(family)
        if runner:
            return runner.generate(**kwargs)
        raise ValueError(
            f"No runner for family '{family}'. "
            f"Available: {list(self.runners.keys())}. "
            f"Ensure pipeline runner files are in the core/ directory."
        )

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
        # Reduce CUDA allocator fragmentation across repeated generations.
        # This is what keeps a Wan 2.2 expert swap from leaving the pool too
        # fragmented to serve the next large activation.
        #
        # BOTH names are set on purpose. Do not "clean this up" to just the new
        # one on the strength of torch's deprecation warning — the warning is
        # misleading on torch 2.9.1+cu130. Verified empirically:
        #
        #   PYTORCH_CUDA_ALLOC_CONF=bogus:True  -> RuntimeError: Unrecognized
        #                                          CachingAllocator option
        #                                          (parsed and honoured)
        #   PYTORCH_ALLOC_CONF=bogus:True       -> no error at all
        #                                          (not parsed by this build)
        #
        # and via torch.cuda.memory_snapshot()[i]["is_expandable"]:
        #   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> True
        #   PYTORCH_ALLOC_CONF=expandable_segments:True      -> False
        #
        # So the deprecated spelling is still the one that works here; the new
        # spelling is set alongside it so this keeps working when a future torch
        # really does drop the old name. The only cost is the deprecation
        # warning on startup, which is cosmetic.
        for _var in ALLOC_CONF_VARS:
            os.environ.setdefault(_var, "expandable_segments:True")
        
        _log("BACKEND", f"Models base path: {models_root}")
        
        # Import DiffSynth pipelines (must happen after env setup)
        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig as WanModelConfig
        from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig as Flux2ModelConfig
        from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig as LTX2ModelConfig
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig as MiniMaxH3ModelConfig
        from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig as Krea2ModelConfig

        # Store ModelConfig classes as instance attributes for use in other methods
        self.ModelConfig = ModelConfig
        self.WanModelConfig = WanModelConfig
        self.Flux2ModelConfig = Flux2ModelConfig
        self.LTX2ModelConfig = LTX2ModelConfig
        self.MiniMaxH3ModelConfig = MiniMaxH3ModelConfig
        self.Krea2ModelConfig = Krea2ModelConfig

        # Populate pipeline registry
        PIPELINE_CLASSES["qwen"] = QwenImagePipeline
        PIPELINE_CLASSES["wan"] = WanVideoPipeline
        PIPELINE_CLASSES["flux2"] = Flux2ImagePipeline
        PIPELINE_CLASSES["ltx2"] = LTX2AudioVideoPipeline
        PIPELINE_CLASSES["minimax_h3"] = MiniMaxH3Pipeline
        PIPELINE_CLASSES["krea2"] = Krea2Pipeline

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def reload_config(self) -> dict:
        """Re-read models.json and the defaults fragments from disk.

        The Models panel writes `enabled` into models.json and needs the change
        reflected in the generation dropdowns immediately. Only the config and
        the derived lookups are rebuilt — loaded pipelines are left alone, since
        toggling a model's visibility says nothing about the weights already
        resident in RAM, and dropping them would cost a multi-minute reload.

        Returns a small summary so callers can log what changed.
        """
        self.models_config = self._load_config("models.json")
        self.defaults_config = self._load_defaults()

        self._alias_map = {}
        for key, entry in self.models_config.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            for alias in entry.get("aliases", []):
                self._alias_map[alias] = key

        self._scan_lora_dirs()

        keys = [k for k in self.models_config
                if not k.startswith("_") and isinstance(self.models_config[k], dict)]
        enabled = [k for k in keys if self.models_config[k].get("enabled", True)]
        _log("BACKEND", f"Config reloaded: {len(enabled)}/{len(keys)} models enabled")
        return {"models": len(keys), "enabled": len(enabled),
                "loras": len(self._lora_registry)}

    def _load_config(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            _log("BACKEND", f"Config not found: {path}", "warning")
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _load_defaults(self) -> dict:
        merged = self._load_config("defaults.json")
        for fragment in ("defaults_loras.json", "defaults_vram.json", "defaults_spec_tool.json", "defaults_dataset.json"):
            merged.update(self._load_config(fragment))
        return merged

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def _scan_lora_dirs(self):
        """Build LoRA registry from scanned dir + curated defined entries."""
        self._lora_registry = {}

        # Scanned: every .safetensors under scanned_loras_path (no model filter)
        scanned_path = self.defaults_config.get("scanned_loras_path")
        if scanned_path:
            d = Path(scanned_path).expanduser()
            if d.exists():
                for f in sorted(d.rglob("*.safetensors")):
                    rel = f.relative_to(d)
                    key = f"{rel.parent}/{f.stem}" if len(rel.parts) > 1 else f.stem
                    self._lora_registry[key] = {
                        "path": str(f),
                        "name": f.stem,
                        "key": key,
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                    }
            else:
                _log("BACKEND", f"Scanned LoRA dir not found: {d}", "warning")

        # Defined: curated entries from defaults_loras.json (carry model + metadata)
        defined_base = self.defaults_config.get("defined_loras_path")
        defined_base = Path(defined_base).expanduser() if defined_base else None
        for entry in self.defaults_config.get("loras", []):
            raw_path = entry.get("path")
            if not raw_path:
                continue
            p = Path(raw_path).expanduser()
            if not p.is_absolute() and defined_base is not None:
                p = defined_base / p
            if not p.exists():
                _log("BACKEND", f"Defined LoRA not found: {p}", "warning")
                continue
            name = entry.get("name") or p.stem
            model = entry.get("model")
            model_key = model[0] if isinstance(model, list) else model
            key = f"{model_key}/{name}" if model_key else name
            self._lora_registry[key] = {
                "path": str(p),
                "name": name,
                "key": key,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                "model": model,
                "default_strength": entry.get("default_strength"),
                "trigger_word": entry.get("trigger_word"),
                "inject_text": entry.get("inject_text"),
            }

        _log("BACKEND", f"LoRA registry built → {len(self._lora_registry)} entries")

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

    def _resolve_lora_specs(
        self,
        lora: Optional[str],
        lora_multiplier: float,
        loras: Optional[List[Dict[str, Any]]],
    ) -> List[tuple]:
        """
        Normalize LoRA inputs into a list of (resolved_path, alpha) tuples.
        
        Handles both the legacy single-LoRA params (lora + lora_multiplier)
        and the new multi-LoRA list (loras). They combine — the legacy param
        is prepended to the list if both are provided.
        
        Returns empty list if no LoRAs requested or none could be resolved.
        """
        specs = []

        # Legacy single-LoRA param
        if lora:
            path = self._resolve_lora_path(lora)
            if path:
                specs.append((path, lora_multiplier))
            else:
                _log("BACKEND", f"LoRA not found, skipping: {lora}", "warning")

        # Multi-LoRA list
        if loras:
            for entry in loras:
                if entry.get("bypass"):
                    continue
                name = entry.get("name") or entry.get("key") or entry.get("path", "")
                alpha = entry.get("alpha") if entry.get("alpha") is not None else entry.get("multiplier", 1.0)
                if not name:
                    continue
                path = self._resolve_lora_path(name)
                if path:
                    # Avoid duplicates (same path already added via legacy param)
                    if not any(p == path for p, _ in specs):
                        specs.append((path, alpha))
                else:
                    _log("BACKEND", f"LoRA not found, skipping: {name}", "warning")

        return specs

    def _load_model_lora(self, pipeline, lora_cfg: dict, alpha_override: float = None):
        """
        Load a model-bundled LoRA (e.g. Control-Union, EliGen) from local models directory.
        
        Constructs path as: {models_root}/{model_id}/{pattern}
        All models should be pre-downloaded via download_models.py.
        
        Args:
            alpha_override: If set, use this alpha instead of models.json default.
        """
        model_id = lora_cfg["model_id"]
        pattern = lora_cfg["pattern"]
        target = lora_cfg.get("target", "dit")
        alpha = alpha_override if alpha_override is not None else lora_cfg.get("alpha", 1.0)

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

    def _apply_user_loras(self, pipeline, cache_key: str, lora_specs: List[tuple]):
        """
        Load one or more user-selected LoRAs onto a pipeline.
        
        Args:
            pipeline: The DiffSynth pipeline instance
            cache_key: Pipeline cache key (model_type:preset)
            lora_specs: List of (resolved_path, alpha) tuples
        
        Compares the full spec list (paths + alphas) against what's currently
        loaded. Only clears and rebuilds when something actually changed.
        
        For models with bundled LoRAs: clears all, reloads model LoRA,
        then stacks user LoRAs on top.
        """
        current = self._active_user_loras.get(cache_key, [])
        if current == lora_specs:
            names = ", ".join(f"{Path(p).stem}(α={a})" for p, a in lora_specs)
            _log("BACKEND", f"User LoRA(s) already loaded: {names}")
            return

        # Something changed — need to rebuild the LoRA stack
        _t_lora = time.perf_counter()
        has_model_lora = cache_key in self._model_lora_config
        
        if current or has_model_lora:
            # Must clear and rebuild when:
            #   - swapping user LoRAs (current is non-empty), OR
            #   - pipeline has a model LoRA (e.g. control-union) since
            #     DiffSynth stacks LoRA deltas additively and the model
            #     LoRA needs to be reloaded cleanly before user LoRA(s)
            try:
                pipeline.clear_lora()
                if current:
                    names = ", ".join(Path(p).stem for p, _ in current)
                    _log("BACKEND", f"Cleared previous user LoRA(s): {names}")
                elif has_model_lora:
                    _log("BACKEND", "Cleared model LoRA for clean rebuild")
            except Exception as e:
                _log("BACKEND", f"clear_lora() failed (non-fatal): {e}", "warning")

            # Reload model LoRA since clear_lora() nukes everything (at current alpha)
            model_lora_cfg = self._model_lora_config.get(cache_key)
            if model_lora_cfg:
                active_alpha = self._model_lora_alpha.get(cache_key, model_lora_cfg.get("alpha", 1.0))
                self._load_model_lora(pipeline, model_lora_cfg, alpha_override=active_alpha)
        # else: no model LoRA, no user LoRAs — just load directly

        # Load user LoRA(s) on top
        target_module = getattr(pipeline, "dit", None)
        if target_module is None:
            _log("BACKEND", "Pipeline has no 'dit' — cannot load user LoRA", "error")
            return

        for lora_path, alpha in lora_specs:
            _log("BACKEND", f"Loading user LoRA: {Path(lora_path).stem} (α={alpha})")
            pipeline.load_lora(target_module, lora_path, alpha=alpha)

        self._active_user_loras[cache_key] = lora_specs
        names = ", ".join(f"{Path(p).stem}(α={a})" for p, a in lora_specs)
        _log("BACKEND", f"[timing] LoRA rebuild: {time.perf_counter() - _t_lora:.1f}s — active: {names}", "success")

    def _clear_user_loras(self, pipeline, cache_key: str):
        """
        Clear all user LoRAs if any are active.
        
        For models with bundled LoRAs: clears all, then reloads model LoRA
        at the currently active alpha (which may differ from models.json).
        For models without bundled LoRAs: just clears.
        """
        if cache_key not in self._active_user_loras:
            return
        
        current = self._active_user_loras[cache_key]
        try:
            pipeline.clear_lora()
            names = ", ".join(Path(p).stem for p, _ in current)
            _log("BACKEND", f"Cleared user LoRA(s): {names}")
        except Exception:
            pass
        
        # Reload model LoRA if this pipeline has one (at current alpha)
        model_lora_cfg = self._model_lora_config.get(cache_key)
        if model_lora_cfg:
            active_alpha = self._model_lora_alpha.get(cache_key, model_lora_cfg.get("alpha", 1.0))
            self._load_model_lora(pipeline, model_lora_cfg, alpha_override=active_alpha)
        
        del self._active_user_loras[cache_key]

    def override_model_lora_alpha(self, pipeline, cache_key: str, alpha: float):
        """
        Change the model-bundled LoRA alpha at runtime (e.g. EliGen strength slider).
        
        If the alpha differs from what's currently loaded, clears ALL LoRAs
        and reloads the model LoRA at the new alpha. User LoRAs are marked
        as cleared so the subsequent apply_loras() call will reload them.
        
        No-op if this pipeline has no model LoRA or alpha hasn't changed.
        """
        model_lora_cfg = self._model_lora_config.get(cache_key)
        if not model_lora_cfg:
            return  # No model LoRA on this pipeline
        
        current_alpha = self._model_lora_alpha.get(cache_key, model_lora_cfg.get("alpha", 1.0))
        if abs(current_alpha - alpha) < 1e-4:
            _log("BACKEND", f"Model LoRA alpha unchanged ({alpha:.2f})")
            return
        
        _log("BACKEND", f"Model LoRA alpha: {current_alpha:.2f} → {alpha:.2f}")
        
        # Clear everything and rebuild at new alpha
        try:
            pipeline.clear_lora()
        except Exception as e:
            _log("BACKEND", f"clear_lora() during alpha override: {e}", "warning")
        
        self._load_model_lora(pipeline, model_lora_cfg, alpha_override=alpha)
        self._model_lora_alpha[cache_key] = alpha
        
        # Mark user LoRAs as cleared so apply_loras() knows to reload them
        self._active_user_loras.pop(cache_key, None)

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
            # A preset may quantize without offloading — shrinking the weights
            # can be enough on its own. Return a config carrying only that, so
            # the caller still treats the preset as active.
            quantize = self._build_quantize_config(preset, name)
            return ({"quantize": quantize} if quantize is not None else None), buffer

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
        quantize = self._build_quantize_config(preset, name)
        if quantize is not None:
            resolved["quantize"] = quantize
        return resolved, buffer

    def _build_quantize_config(self, preset: dict, preset_name: str):
        """Build a DiffSynth QuantizeConfig from a preset's optional `quantize` block.

        Quantization is orthogonal to the 4-state offload system: it shrinks the
        weights themselves, where offloading only moves them. The two compose,
        so any preset may carry a `quantize` block alongside its `config`.

        Returns None when the preset does not ask for quantization. Raises with
        an actionable message when it does but the backend package is missing —
        silently falling back to full precision would look like a mysterious
        OOM rather than a missing dependency.
        """
        spec = preset.get("quantize")
        if not spec:
            return None

        from diffsynth.core.quant import QuantizeConfig

        kwargs = {k: v for k, v in spec.items() if k != "_comment"}
        method = kwargs.get("method")
        try:
            quantize = QuantizeConfig(**kwargs)
        except ImportError as e:
            # QuantizeConfig imports its backend lazily, so a missing torchao /
            # comfy-kitchen / bitsandbytes surfaces here rather than at startup.
            pkg = {"torchao": "torchao", "comfy_kitchen": "comfy-kitchen",
                   "bitsandbytes": "bitsandbytes"}.get(str(method).split("_")[0], method)
            raise RuntimeError(
                f"VRAM preset '{preset_name}' asks for quantization method '{method}', "
                f"but its backend is not installed ({e}). Install it with: "
                f"pip install {pkg}"
            ) from e

        _log("BACKEND", f"  Quantization: {method} (mode={kwargs.get('mode', 'dynamic')})")
        return quantize

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

    def _get_model_config_class(self, entry: dict):
        """Return the correct ModelConfig class for a pipeline type.
        
        Wan and Qwen pipelines may have different ModelConfig classes
        with different fields/validation. Using the wrong one forces
        unnecessary re-processing during from_pretrained().
        """
        pipeline_type = entry.get("pipeline", "qwen")
        if pipeline_type == "wan":
            return self.WanModelConfig
        if pipeline_type == "flux2":
            return self.Flux2ModelConfig
        if pipeline_type == "ltx2":
            return self.LTX2ModelConfig
        if pipeline_type == "minimax_h3":
            return self.MiniMaxH3ModelConfig
        if pipeline_type == "krea2":
            return self.Krea2ModelConfig
        return self.ModelConfig

    def _build_model_configs(self, entry: dict, vram_config: dict = None) -> list:
        """Build ModelConfig list from a model entry's components.
        
        Args:
            entry: The models.json entry
            vram_config: Resolved VRAM offload dict, or None for no offloading
        """
        primary_id = entry["model_id"]
        ConfigCls = self._get_model_config_class(entry)

        # `quantize` rides in on vram_config but is not an offload field and is
        # not applied to every component — split it out. Copy rather than pop,
        # since the caller reuses the dict.
        quantize = None
        offload = None
        if vram_config is not None:
            quantize = vram_config.get("quantize")
            offload = {k: v for k, v in vram_config.items() if k != "quantize"}

        configs = []
        for comp in entry.get("components", []):
            mid = comp.get("model_id", primary_id)
            kwargs = dict(
                model_id=mid,
                origin_file_pattern=comp["pattern"],
            )
            # Only apply offload config when a preset is active
            if offload:
                kwargs.update(offload)
            if quantize is not None and self._should_quantize(comp):
                kwargs["quantize"] = quantize
            configs.append(ConfigCls(**kwargs))
        return configs

    # Pattern fragments that identify a denoiser (DiT) component. Quantization
    # is applied to these only. VAEs and text encoders are far more sensitive to
    # weight quantization — a 4-bit VAE decoder speckles every frame at pixel
    # level, which is visually much worse than anything the DiT does — and
    # upstream's own examples quantize the transformer alone.
    _DENOISER_PATTERN_MARKERS = (
        "transformer",        # Qwen-Image, FLUX.2
        "high_noise_model",   # Wan 2.2 dual-DiT
        "low_noise_model",
        "dit",
    )

    def _should_quantize(self, comp: dict) -> bool:
        """True if this component is a denoiser and should be quantized.

        A registry entry can override the pattern heuristic explicitly with
        "quantize": true/false on the component.
        """
        override = comp.get("quantize")
        if override is not None:
            return bool(override)
        pattern = comp.get("pattern", "").lower()
        return any(m in pattern for m in self._DENOISER_PATTERN_MARKERS)

    def _build_extra_config(self, entry: dict, key: str):
        """Build a tokenizer_config or processor_config from entry."""
        cfg = entry.get(key)
        if cfg is None:
            return None
        ConfigCls = self._get_model_config_class(entry)
        mid = cfg.get("model_id", entry["model_id"])
        return ConfigCls(model_id=mid, origin_file_pattern=cfg["pattern"])

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
            # Re-insert to mark most-recently-used (dicts preserve insertion order)
            self.pipelines[cache_key] = self.pipelines.pop(cache_key)
            _log("BACKEND", f"Using cached pipeline: {cache_key}")
            return self.pipelines[cache_key]
        entry = self.get_model_entry(model_type)
        pipeline_name = entry["pipeline"]

        # --- Eviction: LRU with a small slot budget ---
        # Idle pipelines under CPU-offload presets hold ~0 VRAM (their weights
        # rest in system RAM), so keeping e.g. one image + one video pipeline
        # cached avoids a full from_pretrained disk reload on every model
        # switch. Fully-resident pipelines ("none" preset) can't share the GPU,
        # so any involvement of "none" collapses the budget to a single slot.
        import gc
        evicted = []

        # The same model under a different preset is pure waste — always evict.
        for k in [k for k in self.pipelines if k.startswith(f"{model_type}:")]:
            _log("BACKEND", f"Evicting pipeline (same model, different preset): {k}")
            self._evict_pipeline(k)
            evicted.append(k)

        slots = int(self.defaults_config.get("vram", {}).get("pipeline_cache_slots", 2))
        if active_preset == "none" or any(k.endswith(":none") for k in self.pipelines):
            slots = 1
        # Leave room for the pipeline about to be loaded.
        while len(self.pipelines) > max(0, slots - 1):
            lru = next(iter(self.pipelines))
            _log("BACKEND", f"Evicting pipeline (LRU, budget {slots}): {lru}")
            self._evict_pipeline(lru)
            evicted.append(lru)

        if evicted:
            # GC first so Python releases refs, THEN clear CUDA cache
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log("BACKEND", f"Evicted {len(evicted)} pipeline(s), RAM freed", "success")


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
            redirect_common_files=False,
        )

        # VRAM limit — only set when offloading is active.
        #
        # This budget is what DiffSynth fills with resident weights; activations
        # land on top of it, so buffer_gb has to cover the largest single
        # activation the model produces (Wan 2.2's rope_apply wants >4GB at
        # 720p) plus whatever other processes hold.
        #
        # Memory held by *other* processes is subtracted, but this process's own
        # reserved pool is added back: it is cached weights we are about to
        # evict or reuse, not a constraint. Reading mem_get_info()[0] alone
        # would make the budget depend on whatever happens to be cached at call
        # time — with a Wan pipeline resident, free is under 1GB and the limit
        # would come out near zero.
        if vram_config is not None and torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info("cuda")
            gib = 1024 ** 3
            total_vram = total_b / gib
            reclaimable = torch.cuda.memory_reserved() / gib
            other_procs = max(total_vram - free_b / gib - reclaimable, 0.0)

            # Floor at 4GB: a hostile desktop should degrade into heavy
            # offloading, not hand DiffSynth a negative or absurd budget.
            vram_limit = max(total_vram - other_procs - buffer_gb, 4.0)
            kwargs["vram_limit"] = vram_limit
            _log("BACKEND",
                 f"  VRAM: {total_vram:.1f}GB total − {other_procs:.1f}GB other procs "
                 f"− {buffer_gb}GB buffer = {vram_limit:.1f}GB limit")
        else:
            _log("BACKEND", f"  VRAM: no limit (preset={active_preset})")

        # Tokenizer or Processor (mutually exclusive in practice)
        tok = self._build_extra_config(entry, "tokenizer")
        if tok:
            kwargs["tokenizer_config"] = tok
        proc = self._build_extra_config(entry, "processor")
        if proc:
            kwargs["processor_config"] = proc
        # LTX-2's two-stage sampling needs a distilled LoRA merged at construction
        # time — it is a from_pretrained argument, not a runtime LoRA. Pipelines
        # that do not accept it drop it in the signature filter below.
        stage2 = self._build_extra_config(entry, "stage2_lora")
        if stage2:
            kwargs["stage2_lora_config"] = stage2

        import inspect as _inspect
        valid_params = _inspect.signature(PipelineCls.from_pretrained).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        _t0 = time.perf_counter()
        pipeline = PipelineCls.from_pretrained(**filtered_kwargs)
        _load_s = time.perf_counter() - _t0
        _log("BACKEND", f"[timing] pipeline load {cache_key}: {_load_s:.1f}s", "success")
        record_timing(f"pipeline_load:{cache_key}", _load_s)

        # Skip re-encoding unchanged control/reference/input media across gens
        if self.defaults_config.get("vram", {}).get("vae_encode_cache", True):
            if install_encode_cache(pipeline, log=lambda m: _log("BACKEND", m)):
                _log("BACKEND", "VAE encode cache installed")

        # Model-bundled LoRA (e.g. Control-Union, EliGen) — load and track config
        lora_cfg = entry.get("lora")
        if lora_cfg:
            self._load_model_lora(pipeline, lora_cfg)
            self._model_lora_config[cache_key] = lora_cfg
            self._model_lora_alpha[cache_key] = lora_cfg.get("alpha", 1.0)

        self.pipelines[cache_key] = pipeline
        return pipeline

    def _evict_pipeline(self, cache_key: str):
        """Drop a cached pipeline and its LoRA bookkeeping (caller handles GC)."""
        self.pipelines.pop(cache_key, None)
        self._active_user_loras.pop(cache_key, None)
        self._model_lora_config.pop(cache_key, None)
        self._model_lora_alpha.pop(cache_key, None)

    def unload_pipeline(self, model_type: str):
        """Unload all cached pipelines for a model type (any preset)."""
        model_type = self.resolve_model_type(model_type)
        stale = [k for k in self.pipelines if k.startswith(f"{model_type}:")]
        for k in stale:
            del self.pipelines[k]
            self._active_user_loras.pop(k, None)
            self._model_lora_config.pop(k, None)
            self._model_lora_alpha.pop(k, None)
        if stale and torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log("BACKEND", f"Unloaded pipeline(s): {stale}")

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

    # Where the video latent gets decoded, per pipeline family. Both halves
    # matter: the attribute differs (vae / video_vae / video_vae_decoder) and so
    # does the method — MiniMax's pipeline calls decode_video(), not decode(),
    # so hooking `decode` there would install cleanly and then never fire.
    # Ordered most-specific first; `vae` is last because it is the generic name.
    _VIDEO_DECODER_CANDIDATES = (
        ("video_vae", "decode_video"),        # MiniMax-H3
        ("video_vae_decoder", "decode"),      # LTX-2
        ("vae", "decode"),                    # Qwen, Wan, FLUX.2
    )

    def _resolve_video_decoder(self, pipe):
        """Find (decoder_module, method_name) for the pipeline's video decode.

        Returns None when the pipeline exposes none of them — audio-only or
        future families — so latent capture can be skipped rather than raising
        mid-generation. getattr is guarded because these are nn.Modules, whose
        __getattr__ raises AttributeError rather than returning None.
        """
        for attr, method in self._VIDEO_DECODER_CANDIDATES:
            try:
                decoder = getattr(pipe, attr, None)
            except AttributeError:
                continue
            if decoder is not None and callable(getattr(decoder, method, None)):
                return decoder, method
        return None

    def _capture_latent_hook(self, pipe, save_path: Path):
        """
        Install a hook to capture latent before VAE decode.
        
        Uses non-blocking copy to avoid stalling the GPU pipeline during
        the PCIe transfer. The actual torch.save() happens in cleanup,
        AFTER decode completes, so disk I/O doesn't block generation.
        
        Args:
            pipe: The pipeline instance
            save_path: Where to save the captured latent
            
        Returns:
            Function to remove the hook and save latent
        """
        target = self._resolve_video_decoder(pipe)
        if target is None:
            _log("BACKEND",
                 f"No video decoder found on {type(pipe).__name__} — "
                 f"skipping latent capture", "warning")
            return lambda: None
        decoder, method_name = target

        original_decode = getattr(decoder, method_name)
        # Whether the decoder already had its own attribute, or was inheriting
        # the method from its class. Restoring by assignment either way would
        # leave a bound method sitting on the instance, shadowing the class
        # method and holding a reference cycle back to the instance.
        had_own_attr = method_name in decoder.__dict__
        captured = {}  # Will hold {'latent': cpu_tensor, ...} after first decode

        def hooked_decode(latent, *args, **kwargs):
            if not captured:  # Only capture once
                # non_blocking=True overlaps the PCIe transfer with decode
                # computation instead of stalling the GPU pipeline.
                # pin_memory on the destination enables async DMA.
                cpu_latent = torch.empty_like(latent, device='cpu').pin_memory()
                cpu_latent.copy_(latent.detach(), non_blocking=True)
                captured['latent'] = cpu_latent
                captured['shape'] = list(latent.shape)
                captured['dtype'] = str(latent.dtype)
            return original_decode(latent, *args, **kwargs)

        # Install hook
        setattr(decoder, method_name, hooked_decode)

        # Return cleanup function — saves to disk AFTER generation
        def cleanup():
            if had_own_attr:
                setattr(decoder, method_name, original_decode)
            else:
                try:
                    delattr(decoder, method_name)   # fall back to the class method
                except AttributeError:
                    setattr(decoder, method_name, original_decode)
            if captured:
                _t0 = time.perf_counter()
                # Synchronize to ensure non-blocking copy completed
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(captured, str(save_path))
                _log("BACKEND", f"[timing] latent save: {time.perf_counter() - _t0:.2f}s — {save_path} (shape: {captured['shape']})")
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
        
        # Move to GPU and decode. Resolved rather than assuming `pipe.vae`,
        # because the audio-video families name theirs differently — and say so
        # plainly instead of surfacing a bare AttributeError from deep in the
        # pipeline object.
        target = self._resolve_video_decoder(pipe)
        if target is None:
            raise RuntimeError(
                f"'{model_type}' exposes no video decoder this path recognises, "
                f"so its latents cannot be decoded here."
            )
        decoder, method_name = target
        device = next(decoder.parameters()).device
        latent = latent.to(device)

        _log("BACKEND", f"Decoding latent with {model_type} VAE ({method_name})...")

        with torch.no_grad():
            decoded = getattr(decoder, method_name)(latent)
        
        # Save based on output format
        if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # Image output. This used to call a `save_image` from
            # diffsynth.utils.data that has never existed in any DiffSynth we
            # have vendored, so this branch always raised ImportError. Use the
            # pipeline's own tensor->PIL helper, the same one the live preview
            # path uses in qwen_pipeline.py.
            pipe.vae_output_to_image(decoded).save(str(output_path))
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