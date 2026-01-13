# core/wan_video_wrapper.py
"""
Wan Video Generation Wrapper

Flow:
1. UI reads defaults.json → video section for initial values
2. User adjusts settings, clicks Generate
3. Server receives request with user params
4. This wrapper:
   - Loads tool config from configs/tools/musubi-wan.json
   - Maps generic param names to musubi CLI args
   - Applies performance flags from tool config (fp8, etc.)
   - Builds and executes command
5. Returns results to server for UI update
"""

import sys
import os

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import subprocess
from typing import Optional, Tuple, Callable, Dict, Any
from enum import Enum
import json
import re
import time


def _log(category: str, message: str, level: str = "info"):
    """Simple logging helper"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    colors = {
        'info': '\033[96m',    # cyan
        'success': '\033[92m', # green
        'warning': '\033[93m', # yellow
        'error': '\033[91m',   # red
        'end': '\033[0m'
    }
    
    symbols = {
        'info': '',
        'success': '✓ ',
        'warning': '⚠ ',
        'error': '✗ '
    }
    
    color = colors.get(level, colors['info'])
    symbol = symbols.get(level, '')
    print(f"{color}[{timestamp}] {symbol}[{category}] {message}{colors['end']}", flush=True)


class WanTask(Enum):
    """Wan model tasks"""
    T2V_1_3B = "t2v-1.3B"
    T2V_14B = "t2v-14B"
    I2V_14B = "i2v-14B"
    T2I_14B = "t2i-14B"
    # Fun Control variants
    T2V_1_3B_FC = "t2v-1.3B-FC"
    T2V_14B_FC = "t2v-14B-FC"
    I2V_14B_FC = "i2v-14B-FC"
    # FLF2V (First and Last Frame to Video)
    FLF2V_14B = "flf2v-14B"
    # Wan2.2
    T2V_A14B = "t2v-A14B"
    I2V_A14B = "i2v-A14B"


class WanVideoGenerator:
    """
    Wrapper for Wan video generation via musubi-tuner backend.
    
    Reads tool-specific settings from configs/tools/musubi-wan.json
    """
    
    def __init__(
        self,
        models_config_path: Path,
        tool_config_path: Path,
        defaults_config_path: Path,
        musubi_vendor_path: Path
    ):
        """
        Initialize the generator with config paths.
        
        Args:
            models_config_path: Path to models.json (model paths, LoRAs)
            tool_config_path: Path to configs/tools/musubi-wan.json
            defaults_config_path: Path to defaults.json (user defaults)
            musubi_vendor_path: Path to musubi-tuner vendor directory
        """
        self.musubi_path = musubi_vendor_path
        self.models_config = self._load_json(models_config_path)
        self.tool_config = self._load_json(tool_config_path)
        self.defaults_config = self._load_json(defaults_config_path)
        
        # Cache commonly accessed config values
        self.arg_mapping = self.tool_config.get("arg_mapping", {})
        self.cli_flags = self.tool_config.get("cli_flags", {})
        self.performance = self.tool_config.get("performance", {})
        
        _log("WAN", f"Initialized with musubi path: {musubi_vendor_path}")
        _log("WAN", f"Tool config loaded: {len(self.arg_mapping)} arg mappings")
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON config file"""
        with open(path) as f:
            return json.load(f)
    
    def _get_default(self, key: str, fallback: Any = None) -> Any:
        """Get default value from defaults.json video section"""
        return self.defaults_config.get("video", {}).get(key, fallback)
    
    def _get_lora_info(self, lora_name: str) -> dict:
        """Get LoRA info including path, trigger, and multiplier"""
        loras = self.models_config.get("loras", {})
        if lora_name in loras:
            lora_info = loras[lora_name]
            # Handle both old (string path) and new (dict) formats
            if isinstance(lora_info, str):
                return {"path": lora_info, "trigger": "", "multiplier": 1.0}
            return lora_info
        # If not found in config, treat as direct path
        return {"path": lora_name, "trigger": "", "multiplier": 1.0}
    
    def generate_video(
        self,
        prompt: str,
        output_path: Path,
        task: WanTask,
        # Size params
        width: int = None,
        height: int = None,
        video_length: int = None,
        # Generation params
        seed: Optional[int] = None,
        infer_steps: int = None,
        guidance_scale: float = None,
        flow_shift: float = None,
        blocks_to_swap: int = None,
        negative_prompt: Optional[str] = None,
        # Control inputs
        image_path: Optional[Path] = None,
        end_image_path: Optional[Path] = None,
        control_path: Optional[Path] = None,
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = None,
        # Progress callback
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using musubi-tuner backend.
        
        All parameters fall back to defaults.json if not specified.
        Performance flags (fp8, etc.) come from configs/tools/musubi-wan.json.
        
        Returns:
            Dict with output_path, seed_used, and metadata
        """
        start_time = time.time()
        
        # Apply defaults for any unspecified params
        width = width or self._get_default("size", [832, 480])[0]
        height = height or self._get_default("size", [832, 480])[1]
        video_length = video_length or self._get_default("length", 81)
        infer_steps = infer_steps if infer_steps is not None else self._get_default("infer_steps", 20)
        guidance_scale = guidance_scale if guidance_scale is not None else self._get_default("guidance_scale", 5.0)
        flow_shift = flow_shift if flow_shift is not None else self._get_default("flow_shift", 5.0)
        blocks_to_swap = blocks_to_swap if blocks_to_swap is not None else self._get_default("blocks_to_swap", 0)
        negative_prompt = negative_prompt or self._get_default("negative_prompt", "")
        lora_multiplier = lora_multiplier if lora_multiplier is not None else self._get_default("lora_multiplier", 1.0)
        
        # Validate video_length is 4n+1
        if (video_length - 1) % 4 != 0:
            raise ValueError(f"video_length must be 4n+1, got {video_length}")
        
        # Build generation params dict
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "video_length": video_length,
            "infer_steps": infer_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "blocks_to_swap": blocks_to_swap,
            "seed": seed,
            "image_path": image_path,
            "end_image_path": end_image_path,
            "control_path": control_path,
        }
        
        # Add LoRA if specified
        if lora:
            lora_info = self._get_lora_info(lora)
            params["lora_path"] = lora_info["path"]
            params["lora_multiplier"] = lora_multiplier if lora_multiplier != 1.0 else lora_info.get("multiplier", 1.0)
            
            # Auto-append trigger word if present
            trigger = lora_info.get("trigger", "")
            if trigger and trigger.lower() not in prompt.lower():
                params["prompt"] = f"{trigger}, {prompt}"
                _log("WAN", f"Auto-appended LoRA trigger: {trigger}")
        
        # Get performance settings from tool config
        perf = self.performance
        params["fp8"] = perf.get("fp8", True)
        params["fp8_scaled"] = perf.get("fp8_scaled", True)
        params["fp8_t5"] = perf.get("fp8_t5", True)
        params["vae_cache_cpu"] = perf.get("vae_cache_cpu", True)
        params["cpu_noise"] = perf.get("cpu_noise", True)
        params["lazy_loading"] = perf.get("lazy_loading", True)
        params["attn_mode"] = perf.get("attn_mode", "torch")
        
        # ====== LOGGING ======
        self._log_generation_start(task, params)
        
        # Build command using tool config
        cmd = self._build_command(task, output_path, params)
        
        # Log the command
        self._log_command(cmd)
        
        # Execute with progress tracking
        _log("WAN", f"Working directory: {self.musubi_path}")
        _log("WAN", f"Save directory: {output_path.parent}")
        self._execute_with_progress(cmd, infer_steps, progress_callback)
        
        # Find and rename output files
        final_output = self._finalize_output(output_path)
        
        elapsed = time.time() - start_time
        _log("WAN", f"Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)", "success")
        
        return {
            "output_path": str(final_output),
            "seed_used": seed,
            "elapsed_seconds": elapsed,
            "params": params,
        }
    
    def _build_command(self, task: WanTask, output_path: Path, params: dict) -> list:
        """
        Build musubi command using tool config mappings.
        """
        # Get model paths from models.json
        task_config = self.models_config["wan_models"][task.value]
        
        _log("WAN", f"Model config for {task.value}:")
        _log("WAN", f"  DIT: {task_config['dit']}")
        _log("WAN", f"  VAE: {task_config['vae']}")
        _log("WAN", f"  T5: {task_config['t5']}")
        
        # Start with script path
        script = self.tool_config.get("script", "wan_generate_video.py")
        save_dir = output_path.parent
        
        cmd = [
            "python",
            str(self.musubi_path / script),
            "--task", task.value,
            "--dit", task_config["dit"],
            "--vae", task_config["vae"],
            "--t5", task_config["t5"],
            "--save_path", str(save_dir),
        ]
        
        # CLIP model for Wan2.1 I2V tasks (not needed for Wan2.2)
        if task.value in ["i2v-14B", "i2v-14B-FC", "flf2v-14B"] and "clip" in task_config:
            cmd.extend(["--clip", task_config["clip"]])
            _log("WAN", f"  CLIP: {task_config['clip']}")
        
        # DIT high noise for Wan2.2 models (dual DIT architecture)
        if "dit_high_noise" in task_config:
            cmd.extend(["--dit_high_noise", task_config["dit_high_noise"]])
            _log("WAN", f"  DIT High Noise: {task_config['dit_high_noise']}")
        
        # Map params using arg_mapping from tool config
        mapping = self.arg_mapping
        
        # Prompt
        if "prompt" in params and "positive_prompt" in mapping:
            cmd.extend([mapping["positive_prompt"], params["prompt"]])
        
        # Negative prompt
        if params.get("negative_prompt") and "negative_prompt" in mapping:
            cmd.extend([mapping["negative_prompt"], params["negative_prompt"]])
        
        # Video size - musubi uses --video_size W H
        if "size" in mapping:
            cmd.extend([mapping["size"], str(params["width"]), str(params["height"])])
        
        # Video length
        if "length" in mapping:
            cmd.extend([mapping["length"], str(params["video_length"])])
        
        # Standard generation params
        param_keys = ["infer_steps", "guidance_scale", "flow_shift", "attn_mode"]
        for key in param_keys:
            if key in params and params[key] is not None and key in mapping:
                cmd.extend([mapping[key], str(params[key])])
        
        # Blocks to swap
        if params.get("blocks_to_swap", 0) > 0 and "blocks_to_swap" in mapping:
            cmd.extend([mapping["blocks_to_swap"], str(params["blocks_to_swap"])])
        
        # Seed
        if params.get("seed") is not None and "seed" in mapping:
            cmd.extend([mapping["seed"], str(params["seed"])])
        
        # Control inputs
        if params.get("image_path") and "image_path" in mapping:
            cmd.extend([mapping["image_path"], str(params["image_path"])])
            _log("WAN", f"Start image: {params['image_path']}")
        
        if params.get("end_image_path") and "end_image_path" in mapping:
            cmd.extend([mapping["end_image_path"], str(params["end_image_path"])])
            _log("WAN", f"End image: {params['end_image_path']}")
        
        if params.get("control_path") and "control_path" in mapping:
            cmd.extend([mapping["control_path"], str(params["control_path"])])
            _log("WAN", f"Control path: {params['control_path']}")
        
        # LoRA
        if params.get("lora_path") and "lora_path" in mapping:
            cmd.extend([mapping["lora_path"], params["lora_path"]])
            cmd.extend([mapping["lora_multiplier"], str(params.get("lora_multiplier", 1.0))])
            _log("WAN", f"LoRA: {params['lora_path']} @ {params.get('lora_multiplier', 1.0)}x")
        
        # Performance flags from tool config (conditional flags)
        conditional = self.cli_flags.get("conditional", {})
        
        if params.get("fp8") and "fp8" in conditional:
            cmd.extend(conditional["fp8"])
            if params.get("fp8_scaled") and "fp8_scaled" in conditional:
                cmd.extend(conditional["fp8_scaled"])
        
        if params.get("fp8_t5") and "fp8_t5" in conditional:
            cmd.extend(conditional["fp8_t5"])
        
        if params.get("vae_cache_cpu") and "vae_cache_cpu" in conditional:
            cmd.extend(conditional["vae_cache_cpu"])
        
        if params.get("cpu_noise") and "cpu_noise" in conditional:
            cmd.extend(conditional["cpu_noise"])
        
        if params.get("lazy_loading") and "lazy_loading" in conditional:
            cmd.extend(conditional["lazy_loading"])
        
        # Output type
        defaults_override = self.tool_config.get("defaults_override", {})
        if "output_type" in defaults_override:
            cmd.extend(["--output_type", defaults_override["output_type"]])
        
        return cmd
    
    def _execute_with_progress(self, cmd: list, total_steps: int, progress_callback: Optional[Callable]) -> int:
        """Execute command with real-time progress tracking"""
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.musubi_path)
        )
        
        step_pattern = re.compile(r'(\d+)/(\d+)')
        
        print("\n--- MUSUBI OUTPUT ---")
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line = line.rstrip()
            
            # Colorize output
            if 'error' in line.lower() or 'exception' in line.lower():
                print(f"\033[91m  [MUSUBI] {line}\033[0m")
            elif 'warning' in line.lower():
                print(f"\033[93m  [MUSUBI] {line}\033[0m")
            elif 'loading' in line.lower() or 'loaded' in line.lower():
                print(f"\033[96m  [MUSUBI] {line}\033[0m")
            else:
                print(f"  [MUSUBI] {line}")
            
            # Parse progress
            if progress_callback:
                if "step" in line.lower():
                    match = step_pattern.search(line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        progress_callback("denoising", current, total)
                
                # Phase detection
                if "loading" in line.lower():
                    progress_callback("loading_models", 0, 1)
                elif "encoding" in line.lower():
                    progress_callback("encoding", 0, 1)
                elif "vae" in line.lower() and "decod" in line.lower():
                    progress_callback("vae_decode", 0, 1)
                elif "saving" in line.lower():
                    progress_callback("saving", 0, 1)
        
        print("--- END MUSUBI OUTPUT ---\n")
        
        process.wait()
        
        if process.returncode != 0:
            _log("WAN", f"Generation failed with code {process.returncode}", "error")
            raise RuntimeError(f"Wan video generation failed with code {process.returncode}")
        
        if progress_callback:
            progress_callback("complete", 1, 1)
        
        return process.returncode
    
    def _finalize_output(self, output_path: Path) -> Path:
        """Find generated files and rename to expected paths"""
        save_dir = output_path.parent
        
        # Find and rename video
        video_files = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if video_files:
            latest_video = video_files[-1]
            _log("WAN", f"Found video: {latest_video}")
            
            if latest_video != output_path:
                if output_path.exists():
                    output_path.unlink()
                latest_video.rename(output_path)
                _log("WAN", f"Renamed to: {output_path}")
        else:
            _log("WAN", "WARNING: No video file found after generation!", "warning")
        
        # Handle latent file
        latent_files = list(save_dir.glob("*.safetensors"))
        if latent_files:
            latest_latent = sorted(latent_files, key=lambda p: p.stat().st_mtime)[-1]
            expected_latent = save_dir / "latent.safetensors"
            if latest_latent != expected_latent:
                if expected_latent.exists():
                    expected_latent.unlink()
                latest_latent.rename(expected_latent)
                _log("WAN", f"Renamed latent to: {expected_latent}")
        
        return output_path
    
    def _log_generation_start(self, task: WanTask, params: dict):
        """Log generation parameters"""
        print("\n" + "=" * 70)
        print(f"  WAN VIDEO GENERATION")
        print("=" * 70, flush=True)
        
        print(f"\n  Parameters:")
        print(f"    task: {task.value}")
        
        prompt = params.get("prompt", "")
        print(f"    prompt: {prompt[:80]}..." if len(prompt) > 80 else f"    prompt: {prompt}")
        
        print(f"    size: {params.get('width')}x{params.get('height')}")
        print(f"    video_length: {params.get('video_length')} frames")
        print(f"    seed: {params.get('seed')}")
        print(f"    infer_steps: {params.get('infer_steps')}")
        print(f"    guidance_scale: {params.get('guidance_scale')}")
        print(f"    flow_shift: {params.get('flow_shift')}")
        print(f"    blocks_to_swap: {params.get('blocks_to_swap')}")
        
        print(f"\n  Control Inputs:")
        print(f"    image_path: {params.get('image_path')}")
        print(f"    end_image_path: {params.get('end_image_path')}")
        print(f"    control_path: {params.get('control_path')}")
        
        print(f"\n  Performance (from tool config):")
        print(f"    fp8: {params.get('fp8')}")
        print(f"    fp8_scaled: {params.get('fp8_scaled')}")
        print(f"    fp8_t5: {params.get('fp8_t5')}")
        print(f"    vae_cache_cpu: {params.get('vae_cache_cpu')}")
        print(f"    cpu_noise: {params.get('cpu_noise')}")
        print(f"    lazy_loading: {params.get('lazy_loading')}")
        print(f"    attn_mode: {params.get('attn_mode')}")
        
        if params.get("lora_path"):
            print(f"\n  LoRA:")
            print(f"    path: {params['lora_path']}")
            print(f"    multiplier: {params.get('lora_multiplier', 1.0)}")
    
    def _log_command(self, cmd: list):
        """Log the built command"""
        print("\n  Command:")
        for i, part in enumerate(cmd):
            if part.startswith('--'):
                print(f"    {part}", end='')
            elif i > 0 and cmd[i-1].startswith('--'):
                print(f" {part}")
            else:
                print(f"    {part}")
        print("\n" + "=" * 70 + "\n")


# Factory function for easy initialization
def create_video_generator(
    config_dir: Path,
    vendor_dir: Path
) -> WanVideoGenerator:
    """
    Create WanVideoGenerator with standard config paths.
    
    Args:
        config_dir: Directory containing models.json, defaults.json, and tools/
        vendor_dir: Directory containing musubi-tuner
        
    Returns:
        Configured WanVideoGenerator instance
    """
    return WanVideoGenerator(
        models_config_path=config_dir / "models.json",
        tool_config_path=config_dir / "tools" / "musubi-wan.json",
        defaults_config_path=config_dir / "defaults.json",
        musubi_vendor_path=vendor_dir / "musubi-tuner"
    )