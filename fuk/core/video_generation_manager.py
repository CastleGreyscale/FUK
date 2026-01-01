# core/video_generation_manager.py
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Optional, Dict, Any, List

class VideoGenerationManager:
    """Manages video generation outputs and metadata"""
    
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(exist_ok=True)
        
    def create_generation_dir(self, workflow_type: str = "video") -> Path:
        """Create timestamped generation directory"""
        date_dir = self.output_root / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)
        
        existing = [d for d in date_dir.iterdir() 
                   if d.is_dir() and d.name.startswith(f"{workflow_type}_")]
        next_num = len(existing) + 1
        
        gen_dir = date_dir / f"{workflow_type}_{next_num:03d}"
        gen_dir.mkdir()
        
        return gen_dir
    
    def save_metadata(self,
                     gen_dir: Path,
                     prompt: str,
                     enhanced_prompt: str,
                     task: str,
                     video_size: tuple,
                     video_length: int,
                     seed: Optional[int],
                     image_path: Optional[Path] = None,
                     end_image_path: Optional[Path] = None,
                     control_path: Optional[Path] = None,
                     lora: Optional[str] = None,
                     lora_multiplier: float = 1.0,
                     infer_steps: int = 20,
                     guidance_scale: float = 5.0,
                     flow_shift: float = 5.0,
                     negative_prompt: str = "",
                     **kwargs) -> Path:
        """Save generation metadata"""
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "workflow_type": self._infer_workflow_type(task, image_path, end_image_path, control_path),
            "prompt": {
                "original": prompt,
                "enhanced": enhanced_prompt
            },
            "task": task,
            "video_size": list(video_size),
            "video_length": video_length,
            "seed": seed,
            "control_inputs": {
                "start_image": str(image_path) if image_path else None,
                "end_image": str(end_image_path) if end_image_path else None,
                "control_path": str(control_path) if control_path else None,
            },
            "lora": lora,
            "lora_multiplier": lora_multiplier,
            "infer_steps": infer_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "negative_prompt": negative_prompt,
            **kwargs
        }
        
        metadata_path = gen_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, indent=2, fp=f)
        
        return metadata_path
    
    def _infer_workflow_type(self, task, image_path, end_image_path, control_path):
        """Determine workflow type from inputs"""
        if "flf2v" in task.lower():
            return "flf2v"
        elif control_path:
            return "fun_control"
        elif image_path:
            return "i2v"
        else:
            return "t2v"
    
    def get_output_paths(self, gen_dir: Path) -> Dict[str, Path]:
        """Return standard output paths for a video generation"""
        return {
            "video_mp4": gen_dir / "generated.mp4",
            "latent": gen_dir / "latent.safetensors",
            "start_image": gen_dir / "start_image.png",
            "end_image": gen_dir / "end_image.png",
            "control_reference": gen_dir / "control",
            "exr_sequence": gen_dir / "exr_frames",
            "png_sequence": gen_dir / "png_frames",
            "metadata": gen_dir / "metadata.json"
        }
    
    def export_to_exr_sequence(self,
                               gen_dir: Path,
                               video_path: Path,
                               linear: bool = True,
                               cleanup_png: bool = True) -> Path:
        """
        Export generated video to EXR frame sequence
        
        NOTE: This converts MP4 â†’ EXR which preserves compression artifacts.
        For lossless output, use export_latent_to_exr() instead.
        
        Args:
            gen_dir: Generation directory
            video_path: Source video file
            linear: Convert to linear color space
            cleanup_png: Remove intermediate PNG files after conversion
            
        Returns:
            Path to EXR sequence directory
        """
        from utils.format_convert import FormatConverter
        
        # Validate video file exists
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_path.is_dir():
            raise ValueError(f"Expected video file, got directory: {video_path}")
        
        paths = self.get_output_paths(gen_dir)
        exr_dir = paths["exr_sequence"]
        exr_dir.mkdir(exist_ok=True)
        
        print(f"\n=== Exporting MP4 to EXR sequence ===")
        print(f"WARNING: Converting from compressed MP4 - artifacts will be preserved")
        print(f"For lossless output, use export_latent_to_exr() instead\n")
        print(f"Video: {video_path}")
        print(f"Video size: {video_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"Output: {exr_dir}")
        
        # Convert video to EXR sequence
        exr_frames = FormatConverter.video_to_exr_sequence(
            video_path=video_path,
            output_dir=exr_dir,
            frame_pattern="frame_%04d.exr",
            linear=linear
        )
        
        print(f"âœ“ Created {len(exr_frames)} EXR frames")
        
        return exr_dir
    
    def export_latent_to_exr(self,
                            gen_dir: Path,
                            task: str,
                            config_path: Path,
                            musubi_path: Path,
                            linear: bool = True) -> Path:
        """
        Export latent tensor directly to EXR frames (LOSSLESS)
        
        This is the PROPER workflow for professional output:
          Generation â†’ Latent â†’ VAE decode â†’ EXR
        
        Instead of the lossy path:
          Generation â†’ Latent â†’ VAE decode â†’ MP4 â†’ Extract â†’ EXR
        
        Args:
            gen_dir: Generation directory
            task: Wan task type (e.g., "i2v-14B")
            config_path: Path to models.json
            musubi_path: Path to musubi-tuner
            linear: Save in linear color space
            
        Returns:
            Path to EXR sequence directory
        """
        from utils.latent_to_exr import LatentToEXRDecoder
        
        gen_dir = Path(gen_dir)
        latent_path = gen_dir / "latent.safetensors"
        
        if not latent_path.exists():
            raise FileNotFoundError(
                f"No latent file found in {gen_dir}. "
                "Make sure generation used output_type='both'"
            )
        
        print(f"\n=== Decoding Latent to EXR (LOSSLESS) ===")
        print(f"Latent: {latent_path}")
        print(f"Latent size: {latent_path.stat().st_size / (1024*1024):.2f} MB\n")
        
        decoder = LatentToEXRDecoder(musubi_path, config_path)
        
        exr_dir = decoder.decode_latent_to_exr_sequence(
            latent_path=latent_path,
            output_dir=gen_dir,
            task=task,
            linear=linear
        )
        
        print(f"âœ“ Decoded latent to {exr_dir}")
        
        return exr_dir
    
    def copy_control_inputs(self, 
                           gen_dir: Path,
                           image_path: Optional[Path] = None,
                           end_image_path: Optional[Path] = None,
                           control_path: Optional[Path] = None):
        """Copy control inputs to generation directory for reference"""
        
        paths = self.get_output_paths(gen_dir)
        
        if image_path and image_path.exists():
            shutil.copy(image_path, paths["start_image"])
        
        if end_image_path and end_image_path.exists():
            shutil.copy(end_image_path, paths["end_image"])
        
        if control_path and Path(control_path).exists():
            control_ref = paths["control_reference"]
            control_ref.mkdir(exist_ok=True)
            
            control_path = Path(control_path)
            if control_path.is_file():
                # Single video file
                shutil.copy(control_path, control_ref / control_path.name)
            elif control_path.is_dir():
                # Directory of images
                for img in sorted(control_path.glob("*.png")) + sorted(control_path.glob("*.jpg")):
                    shutil.copy(img, control_ref / img.name)