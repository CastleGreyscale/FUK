# core/qwen_image_wrapper.py
import sys
import os

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import subprocess
from typing import Optional, Tuple
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
    """Wrapper for Qwen image generation via musubi-tuner backend"""
    
    def __init__(self, config_path: Path, musubi_vendor_path: Path, defaults_path: Optional[Path] = None):
        self.musubi_path = musubi_vendor_path
        self.config = self._load_config(config_path)
        self.defaults = self._load_config(defaults_path) if defaults_path else {}
        _log("QWEN", f"Initialized with musubi path: {musubi_vendor_path}")
        
    def _load_config(self, config_path: Path):
        with open(config_path) as f:
            return json.load(f)
    
    def generate(self,
                prompt: str,
                output_path: Path,
                model: QwenModel = QwenModel.IMAGE,
                control_image: Optional[Path] = None,
                image_size: Tuple[int, int] = (1024, 1024),
                seed: Optional[int] = None,
                lora: Optional[str] = None,
                lora_multiplier: float = 1.0,
                infer_steps: int = 20,
                guidance_scale: float = 4.0,
                blocks_to_swap: int = 10,
                negative_prompt: Optional[str] = None,
                flow_shift: Optional[float] = None,
                **kwargs) -> Path:
        """
        Generate image using musubi-tuner backend.
        
        If negative_prompt is None, will use default from config.
        If flow_shift is None, musubi will use its dynamic calculation based on resolution.
        Lower flow_shift (e.g., 2.1) gives more natural results, higher (3.0+) gives more detail.
        """
        start_time = time.time()
        
        # Use default negative prompt if not specified
        if negative_prompt is None:
            negative_prompt = self.defaults.get("negative_prompt", "")
        
        # ====== LOGGING ======
        print("\n" + "=" * 70)
        print(f"  QWEN IMAGE GENERATION")
        print("=" * 70, flush=True)
        
        print(f"\n  Parameters:")
        print(f"    model: {model.value}")
        print(f"    prompt: {prompt[:80]}..." if len(prompt) > 80 else f"    prompt: {prompt}")
        print(f"    negative_prompt: {negative_prompt[:50]}..." if len(negative_prompt) > 50 else f"    negative_prompt: {negative_prompt}")
        print(f"    image_size: {image_size[0]}x{image_size[1]}")
        print(f"    seed: {seed}")
        print(f"    infer_steps: {infer_steps}")
        print(f"    guidance_scale: {guidance_scale}")
        print(f"    flow_shift: {flow_shift if flow_shift else 'auto'}")
        print(f"    blocks_to_swap: {blocks_to_swap}")
        
        if control_image:
            print(f"\n  Control Image:")
            print(f"    path: {control_image}")
            print(f"    exists: {control_image.exists() if isinstance(control_image, Path) else 'unknown'}")
        
        if lora:
            print(f"\n  LoRA:")
            print(f"    lora: {lora}")
            print(f"    multiplier: {lora_multiplier}")
        
        print(f"\n  Output:")
        print(f"    path: {output_path}")
        
        cmd = self._build_command(
            prompt=prompt,
            output_path=output_path,
            model=model,
            control_image=control_image,
            image_size=image_size,
            seed=seed,
            lora=lora,
            lora_multiplier=lora_multiplier,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            blocks_to_swap=blocks_to_swap,
            negative_prompt=negative_prompt,
            flow_shift=flow_shift,
            **kwargs
        )
        
        # Log the command
        print("\n  Command:")
        for i, part in enumerate(cmd):
            if part.startswith('--'):
                print(f"    {part}", end='')
            elif i > 0 and cmd[i-1].startswith('--'):
                print(f" {part}")
            else:
                print(f"    {part}")
        print("\n" + "=" * 70 + "\n")
        
        _log("QWEN", f"Working directory: {self.musubi_path}")
        
        # Stream output in real-time
        # Run from musubi-tuner directory so module imports work correctly
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
        
        # Find the generated file (musubi auto-names it)
        save_dir = output_path.parent
        generated_files = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        
        if not generated_files:
            _log("QWEN", "No output file found after generation!", "error")
            raise RuntimeError("No output file found after generation")
        
        latest_file = generated_files[-1]  # most recent
        _log("QWEN", f"Found generated file: {latest_file}")
        
        # Rename to our desired output_path if different
        if latest_file != output_path:
            latest_file.rename(output_path)
            _log("QWEN", f"Renamed to: {output_path}")
        
        elapsed = time.time() - start_time
        _log("QWEN", f"Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)", "success")
        
        return output_path

    def _build_command(self, prompt, output_path, model, control_image, 
                       image_size, seed, lora, lora_multiplier, infer_steps, 
                       guidance_scale, blocks_to_swap, negative_prompt, flow_shift, **kwargs):
        
        # Get model-specific config
        model_config = self.config["models"][model.value]
        
        _log("QWEN", f"Model config for {model.value}:")
        _log("QWEN", f"  DIT: {model_config['dit']}")
        _log("QWEN", f"  VAE: {model_config['vae']}")
        _log("QWEN", f"  Text Encoder: {model_config['text_encoder']}")
        
        # Musubi saves to directory, we'll need to find the file after
        save_dir = output_path.parent
        
        cmd = [
            "python", 
            str(self.musubi_path / "qwen_image_generate_image.py"),
            "--dit", model_config["dit"],
            "--vae", model_config["vae"],
            "--text_encoder", model_config["text_encoder"],
            "--prompt", prompt,
            "--negative_prompt", negative_prompt,
            "--output_type", "images",
            "--save_path", str(save_dir),
            "--image_size", str(image_size[0]), str(image_size[1]),
            "--infer_steps", str(infer_steps),
            "--guidance_scale", str(guidance_scale),
            "--blocks_to_swap", str(blocks_to_swap),
            "--fp8_scaled",
            "--vae_enable_tiling"
        ]
        
        # Add flow_shift if specified (otherwise musubi uses dynamic calculation)
        if flow_shift is not None:
            cmd.extend(["--flow_shift", str(flow_shift)])
        
        # Edit mode for control images
        if control_image:
            cmd.extend(["--edit", "--control_image_path", str(control_image)])
            _log("QWEN", f"Edit mode enabled with control image: {control_image}")
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        if lora:
            lora_path = self.config["loras"].get(lora, lora)
            cmd.extend([
                "--lora_weight", lora_path,
                "--lora_multiplier", str(lora_multiplier)
            ])
            _log("QWEN", f"LoRA: {lora_path} @ {lora_multiplier}x")
        
        return cmd