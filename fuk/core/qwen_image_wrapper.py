# core/qwen_image_wrapper.py
from pathlib import Path
import subprocess
from typing import Optional, Tuple
from enum import Enum
import json

class QwenModel(Enum):
    IMAGE = "qwen_image"
    EDIT_2509 = "qwen_image_2509_edit"
        
class QwenImageGenerator:
    """Wrapper for Qwen image generation via musubi-tuner backend"""
    
    def __init__(self, config_path: Path, musubi_vendor_path: Path, defaults_path: Optional[Path] = None):
        self.musubi_path = musubi_vendor_path
        self.config = self._load_config(config_path)
        self.defaults = self._load_config(defaults_path) if defaults_path else {}
        
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
        
        # Use default negative prompt if not specified
        if negative_prompt is None:
            negative_prompt = self.defaults.get("negative_prompt", "")
        
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
        
        print("\n=== Running Musubi ===")
        print(" ".join(cmd))
        print("=" * 50 + "\n")
        
        # Stream output in real-time
        # Run from musubi-tuner directory so module imports work correctly
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            cwd=str(self.musubi_path)  # Run from musubi directory
        )
        
        # Print all output
        print(result.stdout)
        
        if result.returncode != 0:
            raise RuntimeError(f"Musubi generation failed")
        
        # Find the generated file (musubi auto-names it)
        save_dir = output_path.parent
        generated_files = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        
        if not generated_files:
            raise RuntimeError("No output file found after generation")
        
        latest_file = generated_files[-1]  # most recent
        
        # Rename to our desired output_path if different
        if latest_file != output_path:
            latest_file.rename(output_path)
        
        return output_path

    def _build_command(self, prompt, output_path, model, control_image, 
                       image_size, seed, lora, lora_multiplier, infer_steps, 
                       guidance_scale, blocks_to_swap, negative_prompt, flow_shift, **kwargs):
        
        # Get model-specific config
        model_config = self.config["models"][model.value]
        print(f"DEBUG: Using model {model.value}")
        print(f"DEBUG: DIT path: {model_config['dit']}")
        
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
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        if lora:
            cmd.extend([
                "--lora_weight", self.config["loras"].get(lora, lora),
                "--lora_multiplier", str(lora_multiplier)
            ])
        
        return cmd
