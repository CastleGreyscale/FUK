# core/image_generation_manager.py
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, Any

class ImageGenerationManager:
    """Manages image generation outputs and metadata"""
    
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(exist_ok=True)
        
    def create_generation_dir(self) -> Path:
        """Create timestamped generation directory"""
        date_dir = self.output_root / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)
        
        # Find next generation number for today
        existing = [d for d in date_dir.iterdir() if d.is_dir() and d.name.startswith("generation_")]
        next_num = len(existing) + 1
        
        gen_dir = date_dir / f"generation_{next_num:03d}"
        gen_dir.mkdir()
        
        return gen_dir
    
    def save_metadata(self, 
                     gen_dir: Path,
                     prompt: str,
                     enhanced_prompt: str,
                     model: str,
                     seed: Optional[int],
                     image_size: tuple,
                     control_image: Optional[Path] = None,
                     lora: Optional[str] = None,
                     lora_multiplier: float = 1.0,
                     infer_steps: int = 20,
                     guidance_scale: float = 4.0,
                     negative_prompt: str = "",
                     flow_shift: Optional[float] = None,
                     **kwargs) -> Path:
        """Save generation metadata"""
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "prompt": {
                "original": prompt,
                "enhanced": enhanced_prompt
            },
            "model": model,
            "seed": seed,
            "image_size": list(image_size),
            "control_image": str(control_image) if control_image else None,
            "lora": lora,
            "lora_multiplier": lora_multiplier,
            "infer_steps": infer_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "flow_shift": flow_shift,
            **kwargs  # capture any additional params
        }
        
        metadata_path = gen_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, indent=2, fp=f)
        
        return metadata_path
    
    def get_output_paths(self, gen_dir: Path) -> Dict[str, Path]:
        """Return standard output paths for a generation"""
        return {
            "generated_png": gen_dir / "generated.png",
            "generated_exr": gen_dir / "generated.exr",
            "source": gen_dir / "source.png",
            "metadata": gen_dir / "metadata.json"
        }