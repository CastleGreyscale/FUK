# core/qwen_image_wrapper.py
"""
Qwen Image Generation Wrapper

Flow:
1. UI reads defaults.json for initial form values
2. User adjusts settings, clicks Generate
3. Server receives request with user params
4. This wrapper:
   - Loads tool config from configs/tools/musubi-qwen.json
   - Maps generic param names to musubi CLI args
   - Applies always-on flags (fp8_scaled, vae_enable_tiling)
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
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
import json
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


class QwenModel(Enum):
    IMAGE = "qwen_image"
    EDIT_2509 = "qwen_image_2509_edit"


class QwenImageGenerator:
    """
    Wrapper for Qwen image generation via musubi-tuner backend.
    
    Reads tool-specific settings from configs/tools/musubi-qwen.json
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
            tool_config_path: Path to configs/tools/musubi-qwen.json
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
        
        _log("QWEN", f"Initialized with musubi path: {musubi_vendor_path}")
        _log("QWEN", f"Tool config loaded: {len(self.arg_mapping)} arg mappings")
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON config file"""
        with open(path) as f:
            return json.load(f)
    
    def _get_default(self, key: str, fallback: Any = None) -> Any:
        """Get default value from defaults.json image section"""
        return self.defaults_config.get("image", {}).get(key, fallback)
    
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
    
    def generate(
        self,
        prompt: str,
        output_path: Path,
        model: QwenModel = QwenModel.IMAGE,
        # Size params (UI sends width + aspect_ratio, server calculates height)
        width: int = None,
        height: int = None,
        # Generation params
        seed: Optional[int] = None,
        infer_steps: int = None,
        guidance_scale: float = None,
        flow_shift: Optional[float] = None,
        blocks_to_swap: int = None,
        negative_prompt: Optional[str] = None,
        # LoRA
        lora: Optional[str] = None,
        lora_multiplier: float = None,
        # Control images for edit mode (single Path or list of Paths)
        control_image: Optional[Union[Path, List[Path]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image using musubi-tuner backend.
        
        All parameters fall back to defaults.json if not specified.
        Tool-specific flags come from configs/tools/musubi-qwen.json.
        
        Returns:
            Dict with output_path, seed_used, and metadata
        """
        start_time = time.time()
        
        # Apply defaults for any unspecified params
        width = width or self._get_default("width", 1344)
        height = height or width  # Square if not specified
        infer_steps = infer_steps if infer_steps is not None else self._get_default("infer_steps", 20)
        guidance_scale = guidance_scale if guidance_scale is not None else self._get_default("guidance_scale", 2.1)
        flow_shift = flow_shift if flow_shift is not None else self._get_default("flow_shift", 2.1)
        blocks_to_swap = blocks_to_swap if blocks_to_swap is not None else self._get_default("blocks_to_swap", 0)
        negative_prompt = negative_prompt or self._get_default("negative_prompt", "")
        lora_multiplier = lora_multiplier if lora_multiplier is not None else self._get_default("lora_multiplier", 1.0)
        
        # Build generation params dict
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": height,
            "height": width,
            "infer_steps": infer_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "blocks_to_swap": blocks_to_swap,
            "seed": seed,
        }
        
        # Add LoRA if specified
        if lora:
            lora_info = self._get_lora_info(lora)
            params["lora_path"] = lora_info["path"]
            # Use provided multiplier or LoRA's default
            params["lora_multiplier"] = lora_multiplier if lora_multiplier != 1.0 else lora_info.get("multiplier", 1.0)
            
            # Auto-append trigger word if present and not already in prompt
            trigger = lora_info.get("trigger", "")
            if trigger and trigger.lower() not in prompt.lower():
                params["prompt"] = f"{trigger}, {prompt}"
                _log("QWEN", f"Auto-appended LoRA trigger: {trigger}")
        
        # Add control images for edit mode (can be single or list)
        # musubi accepts multiple images as space-separated paths
        if control_image:
            if isinstance(control_image, list):
                params["control_images"] = [str(p) for p in control_image]
            else:
                params["control_images"] = [str(control_image)]
        
        # ====== LOGGING ======
        self._log_generation_start(model, params)
        
        # Build command using tool config
        cmd = self._build_command(model, output_path, params)
        
        # Log the command
        self._log_command(cmd)
        
        # Execute
        _log("QWEN", f"Working directory: {self.musubi_path}")
        result = self._execute(cmd)
        
        # Find and rename output file
        final_output = self._finalize_output(output_path)
        
        elapsed = time.time() - start_time
        _log("QWEN", f"Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)", "success")
        
        return {
            "output_path": str(final_output),
            "seed_used": seed,
            "elapsed_seconds": elapsed,
            "params": params,
        }
    
    def _build_command(self, model: QwenModel, output_path: Path, params: dict) -> list:
        """
        Build musubi command using tool config mappings.
        
        Reads arg_mapping from musubi-qwen.json to translate generic
        param names to musubi-specific CLI args.
        """
        # Get model paths from models.json
        model_config = self.models_config["models"][model.value]
        
        _log("QWEN", f"Model config for {model.value}:")
        _log("QWEN", f"  DIT: {model_config['dit']}")
        _log("QWEN", f"  VAE: {model_config['vae']}")
        _log("QWEN", f"  Text Encoder: {model_config['text_encoder']}")
        
        # Start with script path
        script = self.tool_config.get("script", "qwen_image_generate_image.py")
        cmd = ["python", str(self.musubi_path / script)]
        
        # Add model paths (these use direct mapping, not arg_mapping)
        cmd.extend(["--dit", model_config["dit"]])
        cmd.extend(["--vae", model_config["vae"]])
        cmd.extend(["--text_encoder", model_config["text_encoder"]])
        
        # Add output directory
        cmd.extend(["--save_path", str(output_path.parent)])
        
        # Map params using arg_mapping from tool config
        mapping = self.arg_mapping
        
        # Prompt
        if "prompt" in params and "positive_prompt" in mapping:
            cmd.extend([mapping["positive_prompt"], params["prompt"]])
        
        # Negative prompt  
        if params.get("negative_prompt") and "negative_prompt" in mapping:
            cmd.extend([mapping["negative_prompt"], params["negative_prompt"]])
        
        # Image size - musubi uses --image_size W H
        if "size" in mapping:
            cmd.extend([mapping["size"], str(params["width"]), str(params["height"])])
        
        # Standard generation params
        param_keys = ["infer_steps", "guidance_scale", "blocks_to_swap"]
        for key in param_keys:
            if key in params and params[key] is not None and key in mapping:
                cmd.extend([mapping[key], str(params[key])])
        
        # Flow shift (optional - musubi calculates dynamically if not set)
        if params.get("flow_shift") is not None and "flow_shift" in mapping:
            cmd.extend([mapping["flow_shift"], str(params["flow_shift"])])
        
        # Seed
        if params.get("seed") is not None and "seed" in mapping:
            cmd.extend([mapping["seed"], str(params["seed"])])
        
        # LoRA
        if params.get("lora_path") and "lora_path" in mapping:
            cmd.extend([mapping["lora_path"], params["lora_path"]])
            cmd.extend([mapping["lora_multiplier"], str(params.get("lora_multiplier", 1.0))])
        
        # Control images (edit mode) - musubi uses space-separated paths
        if params.get("control_images"):
            # Add conditional flags for edit mode
            conditional = self.cli_flags.get("conditional", {})
            if "edit_mode" in conditional:
                cmd.extend(conditional["edit_mode"])
            if "control_image" in mapping:
                # Add the flag once, followed by ALL image paths as separate args
                cmd.append(mapping["control_image"])
                cmd.extend(params["control_images"])
        
        # Add always-on flags from tool config
        always_flags = self.cli_flags.get("always", [])
        cmd.extend(always_flags)
        
        # Add output type
        defaults_override = self.tool_config.get("defaults_override", {})
        if "output_type" in defaults_override:
            cmd.extend(["--output_type", defaults_override["output_type"]])
        
        return cmd
    
    def _execute(self, cmd: list) -> int:
        """Execute command with real-time output streaming"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.musubi_path)
        )
        
        print("--- MUSUBI OUTPUT ---")
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line_stripped = line.rstrip()
            
            # Colorize output based on content
            if 'error' in line_stripped.lower() or 'exception' in line_stripped.lower():
                print(f"\033[91m  [MUSUBI] {line_stripped}\033[0m")
            elif 'warning' in line_stripped.lower():
                print(f"\033[93m  [MUSUBI] {line_stripped}\033[0m")
            elif 'loading' in line_stripped.lower() or 'loaded' in line_stripped.lower():
                print(f"\033[96m  [MUSUBI] {line_stripped}\033[0m")
            else:
                print(f"  [MUSUBI] {line_stripped}")
        
        process.wait()
        print("--- END MUSUBI OUTPUT ---\n")
        
        if process.returncode != 0:
            _log("QWEN", f"Generation failed with code {process.returncode}", "error")
            raise RuntimeError(f"Musubi generation failed with code {process.returncode}")
        
        return process.returncode
    
    def _finalize_output(self, output_path: Path) -> Path:
        """Find generated file and rename to expected output path"""
        save_dir = output_path.parent
        generated_files = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        
        if not generated_files:
            _log("QWEN", "No output file found after generation!", "error")
            raise RuntimeError("No output file found after generation")
        
        latest_file = generated_files[-1]
        _log("QWEN", f"Found generated file: {latest_file}")
        
        if latest_file != output_path:
            latest_file.rename(output_path)
            _log("QWEN", f"Renamed to: {output_path}")
        
        return output_path
    
    def _log_generation_start(self, model: QwenModel, params: dict):
        """Log generation parameters"""
        print("\n" + "=" * 70)
        print(f"  QWEN IMAGE GENERATION")
        print("=" * 70, flush=True)
        
        print(f"\n  Parameters:")
        print(f"    model: {model.value}")
        
        prompt = params.get("prompt", "")
        print(f"    prompt: {prompt[:80]}..." if len(prompt) > 80 else f"    prompt: {prompt}")
        
        neg = params.get("negative_prompt", "")
        print(f"    negative_prompt: {neg[:50]}..." if len(neg) > 50 else f"    negative_prompt: {neg}")
        
        print(f"    size: {params.get('width')}x{params.get('height')}")
        print(f"    seed: {params.get('seed')}")
        print(f"    infer_steps: {params.get('infer_steps')}")
        print(f"    guidance_scale: {params.get('guidance_scale')}")
        print(f"    flow_shift: {params.get('flow_shift', 'auto')}")
        print(f"    blocks_to_swap: {params.get('blocks_to_swap')}")
        
        if params.get("control_images"):
            print(f"\n  Control Images ({len(params['control_images'])}):")
            for i, img in enumerate(params["control_images"], 1):
                print(f"    [{i}] {img}")
        
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
def create_generator(
    config_dir: Path,
    vendor_dir: Path
) -> QwenImageGenerator:
    """
    Create QwenImageGenerator with standard config paths.
    
    Args:
        config_dir: Directory containing models.json, defaults.json, and tools/
        vendor_dir: Directory containing musubi-tuner
        
    Returns:
        Configured QwenImageGenerator instance
    """
    return QwenImageGenerator(
        models_config_path=config_dir / "models.json",
        tool_config_path=config_dir / "tools" / "musubi-qwen.json",
        defaults_config_path=config_dir / "defaults.json",
        musubi_vendor_path=vendor_dir / "musubi-tuner"
    )