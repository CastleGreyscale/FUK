# core/wan_video_wrapper.py
from pathlib import Path
import subprocess
from typing import Optional, Tuple, Callable
from enum import Enum
import json
import re

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
    """Wrapper for Wan video generation with focus on controllable workflows"""
    
    def __init__(self, config_path: Path, musubi_vendor_path: Path, defaults_path: Optional[Path] = None):
        self.musubi_path = musubi_vendor_path
        self.config = self._load_config(config_path)
        self.defaults = self._load_config(defaults_path) if defaults_path else {}
        
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
        
        Args:
            prompt: Text description of desired video
            output_path: Where to save the video
            task: Which Wan model/task to use
            video_size: (width, height) tuple
            video_length: Number of frames (must be 4n+1)
            seed: Random seed for reproducibility
            
            Control options:
            image_path: Starting image for I2V or FLF2V
            end_image_path: Ending image for FLF2V
            control_path: Control video/images for Fun Control models
            
            Generation parameters:
            infer_steps: Number of denoising steps
            guidance_scale: CFG scale
            flow_shift: Flow shift value (3.0 for I2V 480p, 5.0 for others)
            negative_prompt: What to avoid in generation
            
            Performance options:
            blocks_to_swap: Number of blocks to offload to CPU (max 39 for 14B)
                           Recommended: 0-10 for 24GB VRAM, 10-20 for 16GB, 20+ for <16GB
            fp8: Use fp8 precision
            fp8_scaled: Use fp8 weight optimization
            fp8_t5: Use fp8 for T5 text encoder
            vae_cache_cpu: Use CPU for VAE internal cache (slower, only if OOM)
            attn_mode: Attention implementation (torch/sdpa/xformers/sageattn/flash2)
            
            LoRA:
            lora: LoRA name or path
            lora_multiplier: LoRA strength
            
            Progress:
            progress_callback: Function(phase_name, current_step, total_steps)
        """
        
        # Validate video_length is 4n+1
        if (video_length - 1) % 4 != 0:
            raise ValueError(f"video_length must be 4n+1, got {video_length}")
        
        # Use default negative prompt if not specified
        if negative_prompt is None:
            negative_prompt = self.defaults.get("negative_prompt", "")
        
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
        
        print("\n=== Running Wan Video Generation ===")
        print(" ".join(cmd))
        print("=" * 50 + "\n")
        
        # Run with progress tracking
        self._run_with_progress(cmd, infer_steps, progress_callback)
        
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
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            print(line, end='')  # Still print to console
            
            # Parse progress from output
            # Musubi typically outputs: "Step 5/20" or similar
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
        
        process.wait()
        
        if process.returncode != 0:
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
        
        cmd = [
            "python",
            str(self.musubi_path / "wan_generate_video.py"),
            "--task", task.value,
            "--dit", task_config["dit"],
            "--vae", task_config["vae"],
            "--t5", task_config["t5"],
            "--prompt", prompt,
            "--save_path", str(output_path),
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
        
        # FLF2V: Ending image
        if end_image_path:
            cmd.extend(["--end_image_path", str(end_image_path)])
        
        # Fun Control: Control video/images
        if control_path:
            cmd.extend(["--control_path", str(control_path)])
        
        # LoRA
        if lora:
            lora_path = self.config["loras"].get(lora, lora)
            cmd.extend([
                "--lora_weight", lora_path,
                "--lora_multiplier", str(lora_multiplier)
            ])
        
        # Additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd