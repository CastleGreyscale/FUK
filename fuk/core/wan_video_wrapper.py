# core/wan_video_wrapper.py
import sys
import os

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import subprocess
from typing import Optional, Tuple, Callable
from enum import Enum
import json
import re
import time

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


class WanVideoGenerator:
    """Wrapper for Wan video generation with focus on controllable workflows"""
    
    def __init__(self, config_path: Path, musubi_vendor_path: Path, defaults_path: Optional[Path] = None):
        self.musubi_path = musubi_vendor_path
        self.config = self._load_config(config_path)
        self.defaults = self._load_config(defaults_path) if defaults_path else {}
        _log("WAN", f"Initialized with musubi path: {musubi_vendor_path}")
        
    def _load_config(self, config_path: Path):
        with open(config_path) as f:
            return json.load(f)
    
    def generate_video(self,
                      prompt: str,
                      output_path: Path,
                      task: WanTask,
                      video_size: Tuple[int, int] = (832, 480),
                      video_length: int = 81,
                      seed: Optional[int] = None,
                      # Control options
                      image_path: Optional[Path] = None,
                      end_image_path: Optional[Path] = None,
                      control_path: Optional[Path] = None,
                      # Generation parameters
                      infer_steps: int = 20,
                      guidance_scale: float = 5.0,
                      flow_shift: float = 5.0,
                      negative_prompt: Optional[str] = None,
                      # Performance options
                      blocks_to_swap: int = 0,
                      fp8: bool = True,
                      fp8_scaled: bool = True,
                      fp8_t5: bool = True,
                      vae_cache_cpu: bool = False,  # Only enable if needed
                      attn_mode: str = "torch",
                      # LoRA
                      lora: Optional[str] = None,
                      lora_multiplier: float = 1.0,
                      # Progress callback
                      progress_callback: Optional[Callable[[str, int, int], None]] = None,
                      **kwargs) -> Path:
        """
        Generate video using Wan models.
        """
        start_time = time.time()
        
        # Validate video_length is 4n+1
        if (video_length - 1) % 4 != 0:
            raise ValueError(f"video_length must be 4n+1, got {video_length}")
        
        # Use default negative prompt if not specified
        if negative_prompt is None:
            negative_prompt = self.defaults.get("negative_prompt", "")
        
        # ====== LOGGING ======
        print("\n" + "=" * 70)
        print(f"  WAN VIDEO GENERATION")
        print("=" * 70, flush=True)
        
        print(f"\n  Parameters:")
        print(f"    task: {task.value}")
        print(f"    prompt: {prompt[:80]}..." if len(prompt) > 80 else f"    prompt: {prompt}")
        print(f"    video_size: {video_size[0]}x{video_size[1]}")
        print(f"    video_length: {video_length} frames")
        print(f"    seed: {seed}")
        print(f"    infer_steps: {infer_steps}")
        print(f"    guidance_scale: {guidance_scale}")
        print(f"    flow_shift: {flow_shift}")
        
        print(f"\n  Control Images:")
        print(f"    image_path: {image_path}")
        print(f"    end_image_path: {end_image_path}")
        print(f"    control_path: {control_path}")
        
        print(f"\n  Performance:")
        print(f"    blocks_to_swap: {blocks_to_swap}")
        print(f"    fp8: {fp8}")
        print(f"    fp8_scaled: {fp8_scaled}")
        print(f"    fp8_t5: {fp8_t5}")
        print(f"    vae_cache_cpu: {vae_cache_cpu}")
        print(f"    attn_mode: {attn_mode}")
        
        if lora:
            print(f"\n  LoRA:")
            print(f"    lora: {lora}")
            print(f"    multiplier: {lora_multiplier}")
        
        print()
        
        cmd = self._build_command(
            prompt=prompt,
            output_path=output_path,
            task=task,
            video_size=video_size,
            video_length=video_length,
            seed=seed,
            image_path=image_path,
            end_image_path=end_image_path,
            control_path=control_path,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            negative_prompt=negative_prompt,
            blocks_to_swap=blocks_to_swap,
            fp8=fp8,
            fp8_scaled=fp8_scaled,
            fp8_t5=fp8_t5,
            vae_cache_cpu=vae_cache_cpu,
            attn_mode=attn_mode,
            lora=lora,
            lora_multiplier=lora_multiplier,
            **kwargs
        )
        
        # Log the command
        print("  Command:")
        for i, part in enumerate(cmd):
            if part.startswith('--'):
                print(f"    {part}", end='')
            elif i > 0 and cmd[i-1].startswith('--'):
                print(f" {part}")
            else:
                print(f"    {part}")
        print("\n" + "=" * 70 + "\n")
        
        _log("WAN", f"Working directory: {self.musubi_path}")
        _log("WAN", f"Save directory: {output_path.parent}")
        _log("WAN", f"Expected output: {output_path}")
        
        # Run with progress tracking
        self._run_with_progress(cmd, infer_steps, progress_callback)
        
        # Find generated video file and rename to expected output path
        save_dir = output_path.parent
        _log("WAN", f"Looking for generated video in: {save_dir}")
        
        # Musubi creates files with timestamps or specific names
        # Look for most recent .mp4 file
        video_files = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        
        if video_files:
            latest_video = video_files[-1]
            _log("WAN", f"Found video: {latest_video}")
            
            # Rename to expected output path if different
            if latest_video != output_path:
                if output_path.exists():
                    output_path.unlink()  # Remove existing
                latest_video.rename(output_path)
                _log("WAN", f"Renamed to: {output_path}")
        else:
            _log("WAN", "WARNING: No video file found after generation!", "warning")
        
        # Also handle latent file if present
        latent_files = list(save_dir.glob("*.safetensors"))
        if latent_files:
            latest_latent = sorted(latent_files, key=lambda p: p.stat().st_mtime)[-1]
            expected_latent = save_dir / "latent.safetensors"
            if latest_latent != expected_latent:
                if expected_latent.exists():
                    expected_latent.unlink()
                latest_latent.rename(expected_latent)
                _log("WAN", f"Renamed latent to: {expected_latent}")
        
        elapsed = time.time() - start_time
        _log("WAN", f"Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)", "success")
        
        return output_path
    
    def _run_with_progress(self, cmd, total_steps, progress_callback):
        """Run subprocess with real-time progress tracking"""
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.musubi_path)  # Run from musubi directory
        )
        
        current_phase = "initialization"
        step_pattern = re.compile(r'(\d+)/(\d+)')
        
        print("\n--- MUSUBI OUTPUT ---")
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            line = line.rstrip()
            
            # Colorize output based on content
            if 'error' in line.lower() or 'exception' in line.lower():
                print(f"\033[91m  [MUSUBI] {line}\033[0m")
            elif 'warning' in line.lower():
                print(f"\033[93m  [MUSUBI] {line}\033[0m")
            elif 'loading' in line.lower() or 'loaded' in line.lower():
                print(f"\033[96m  [MUSUBI] {line}\033[0m")
            else:
                print(f"  [MUSUBI] {line}")
            
            # Parse progress from output
            if "step" in line.lower():
                match = step_pattern.search(line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    
                    if progress_callback:
                        progress_callback("denoising", current, total)
            
            # Detect phase changes
            if "loading" in line.lower():
                current_phase = "loading_models"
                if progress_callback:
                    progress_callback(current_phase, 0, 1)
            elif "encoding" in line.lower():
                current_phase = "encoding"
                if progress_callback:
                    progress_callback(current_phase, 0, 1)
            elif "vae" in line.lower() and "decod" in line.lower():
                current_phase = "vae_decode"
                if progress_callback:
                    progress_callback(current_phase, 0, 1)
            elif "saving" in line.lower():
                current_phase = "saving"
                if progress_callback:
                    progress_callback(current_phase, 0, 1)
        
        print("--- END MUSUBI OUTPUT ---\n")
        
        process.wait()
        
        if process.returncode != 0:
            _log("WAN", f"Generation failed with code {process.returncode}", "error")
            raise RuntimeError(f"Wan video generation failed with code {process.returncode}")
        
        # Final callback
        if progress_callback:
            progress_callback("complete", 1, 1)
    
    def _build_command(self, prompt, output_path, task, video_size, video_length,
                      seed, image_path, end_image_path, control_path,
                      infer_steps, guidance_scale, flow_shift, negative_prompt,
                      blocks_to_swap, fp8, fp8_scaled, fp8_t5, vae_cache_cpu, 
                      attn_mode, lora, lora_multiplier, **kwargs):
        
        # Get model configs based on task
        task_config = self.config["wan_models"][task.value]
        
        _log("WAN", f"Model config for {task.value}:")
        _log("WAN", f"  DIT: {task_config['dit']}")
        _log("WAN", f"  VAE: {task_config['vae']}")
        _log("WAN", f"  T5: {task_config['t5']}")
        
        # IMPORTANT: save_path should be directory, not file
        # Musubi will create files inside this directory
        save_dir = output_path.parent
        
        cmd = [
            "python",
            str(self.musubi_path / "wan_generate_video.py"),
            "--task", task.value,
            "--dit", task_config["dit"],
            "--vae", task_config["vae"],
            "--t5", task_config["t5"],
            "--prompt", prompt,
            "--save_path", str(save_dir),  # Pass directory, not file
            "--video_size", str(video_size[0]), str(video_size[1]),
            "--video_length", str(video_length),
            "--infer_steps", str(infer_steps),
            "--guidance_scale", str(guidance_scale),
            "--flow_shift", str(flow_shift),
            "--negative_prompt", negative_prompt,
            "--attn_mode", attn_mode,
            "--output_type", "both"  # Save both latent and video
        ]
        
        # CLIP model for Wan2.1 I2V tasks
        if task.value in ["i2v-14B", "i2v-14B-FC", "flf2v-14B"] and "clip" in task_config:
            cmd.extend(["--clip", task_config["clip"]])
            _log("WAN", f"  CLIP: {task_config['clip']}")
        
        # FP8 options
        if fp8:
            cmd.append("--fp8")
            if fp8_scaled:
                cmd.append("--fp8_scaled")
        
        # FP8 T5 - reduces T5 VRAM usage
        if fp8_t5:
            cmd.append("--fp8_t5")
        
        # VAE CPU cache - only use if hitting OOM, slows generation
        if vae_cache_cpu:
            cmd.append("--vae_cache_cpu")
        
        # Block swapping for VRAM management
        if blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
        
        # Seed
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # I2V: Starting image
        if image_path:
            cmd.extend(["--image_path", str(image_path)])
            _log("WAN", f"Start image: {image_path}")
        
        # FLF2V: Ending image
        if end_image_path:
            cmd.extend(["--end_image_path", str(end_image_path)])
            _log("WAN", f"End image: {end_image_path}")
        
        # Fun Control: Control video/images
        if control_path:
            cmd.extend(["--control_path", str(control_path)])
            _log("WAN", f"Control path: {control_path}")
        
        # LoRA
        if lora:
            lora_path = self.config["loras"].get(lora, lora)
            cmd.extend([
                "--lora_weight", lora_path,
                "--lora_multiplier", str(lora_multiplier)
            ])
            _log("WAN", f"LoRA: {lora_path} @ {lora_multiplier}x")
        
        # Additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd