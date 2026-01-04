# core/preprocessors/crypto.py
"""
Cryptomatte / Instance Segmentation Preprocessor

Uses SAM2 (Segment Anything Model 2) for instance segmentation,
then converts to cryptomatte-style ID mattes for compositing.

Good for:
- Per-object selection in comp
- Rotoscoping automation
- Object-aware effects
- VFX isolation passes
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import torch
import colorsys

from .base import BasePreprocessor


class SAMModel(str, Enum):
    """Available SAM model sizes"""
    TINY = "sam2_hiera_tiny"      # Fast (~39MB)
    SMALL = "sam2_hiera_small"    # Balanced (~46MB)
    BASE = "sam2_hiera_base_plus" # Good quality (~81MB)
    LARGE = "sam2_hiera_large"    # Best quality (~224MB)


# Mapping for both sam2 and sam2.1 checkpoint names
SAM_CHECKPOINT_NAMES = {
    SAMModel.TINY: ["sam2.1_hiera_tiny.pt", "sam2_hiera_tiny.pt"],
    SAMModel.SMALL: ["sam2.1_hiera_small.pt", "sam2_hiera_small.pt"],
    SAMModel.BASE: ["sam2.1_hiera_base_plus.pt", "sam2_hiera_base_plus.pt"],
    SAMModel.LARGE: ["sam2.1_hiera_large.pt", "sam2_hiera_large.pt"],
}


class CryptoPreprocessor(BasePreprocessor):
    """
    Instance segmentation using SAM2, output as cryptomatte-style ID mattes
    
    Generates:
    - ID matte image (colored visualization)
    - Individual mask layers (optional)
    - Cryptomatte-compatible EXR (future)
    
    Requires vendor: git clone https://github.com/facebookresearch/segment-anything-2 vendor/segment-anything-2
    """
    
    def __init__(
        self,
        model_size: SAMModel = SAMModel.LARGE,
        config_path: Optional[Path] = None
    ):
        super().__init__(config_path)
        self.model_size = model_size
        self.predictor = None
        self.mask_generator = None
        print(f"Crypto preprocessor initialized: {model_size.value} on {self.device}")
    
    def _initialize(self):
        """Load SAM2 model"""
        try:
            vendor_path = self._get_vendor_path("segment-anything-2")
            
            if vendor_path.exists():
                import sys
                sys.path.insert(0, str(vendor_path))
            
            # Try importing SAM2
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            print(f"Loading SAM2 {self.model_size.value}...")
            
            # Find checkpoint
            checkpoint_path = self._find_sam_checkpoint()
            config_path = self._find_sam_config()
            
            if not checkpoint_path:
                raise FileNotFoundError("SAM2 checkpoint not found")
            
            # Build model
            sam2 = build_sam2(
                config_file=config_path,  # Hydra config name, not a file path
                ckpt_path=str(checkpoint_path),
                device=self.device
            )
            
            # Create automatic mask generator for full-image segmentation
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=32,           # Density of point grid
                points_per_batch=64,          # Points processed at once
                pred_iou_thresh=0.8,          # Filter low-confidence masks
                stability_score_thresh=0.92,  # Filter unstable masks
                crop_n_layers=1,              # Multi-scale crops
                min_mask_region_area=100,     # Minimum mask size
            )
            
            print("✓ SAM2 loaded")
            
        except ImportError as e:
            raise ImportError(
                f"SAM2 not installed. Setup:\n"
                f"  git clone https://github.com/facebookresearch/segment-anything-2 vendor/segment-anything-2\n"
                f"  cd vendor/segment-anything-2\n"
                f"  pip install -e . --break-system-packages\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2: {e}")
    
    def _find_sam_checkpoint(self) -> Optional[Path]:
        """Find SAM2 checkpoint file (supports both sam2 and sam2.1 naming)"""
        vendor_path = self._get_vendor_path("segment-anything-2")
        
        # Get possible checkpoint names for this model size
        checkpoint_names = SAM_CHECKPOINT_NAMES.get(self.model_size, [])
        
        search_dirs = [
            vendor_path / "checkpoints",
            vendor_path,
            Path.home() / "ai" / "models" / "sam2",
        ]
        
        # Search for any matching checkpoint
        for search_dir in search_dirs:
            for checkpoint_name in checkpoint_names:
                path = search_dir / checkpoint_name
                if path.exists():
                    print(f"  Found checkpoint: {path}")
                    return path
        
        print(f"✗ SAM2 checkpoint not found for {self.model_size.value}")
        print("  Download from: https://github.com/facebookresearch/segment-anything-2#download-checkpoints")
        print("  Searched for:", checkpoint_names)
        print("  In directories:")
        for d in search_dirs:
            print(f"    - {d}")
        
        return None
    
    def _find_sam_config(self) -> Optional[str]:
        """Get SAM2 config name for Hydra (not a file path)"""
        # SAM2 uses Hydra - we need to pass the config NAME, not a path
        # The config is resolved relative to sam2/configs/
        config_names = {
            SAMModel.TINY: "configs/sam2.1/sam2.1_hiera_t.yaml",
            SAMModel.SMALL: "configs/sam2.1/sam2.1_hiera_s.yaml",
            SAMModel.BASE: "configs/sam2.1/sam2.1_hiera_b+.yaml",
            SAMModel.LARGE: "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        
        config_name = config_names.get(self.model_size)
        print(f"  Using config: {config_name}")
        return config_name
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        max_objects: int = 50,
        min_area: int = 500,
        output_mode: str = "id_matte",  # 'id_matte', 'layers', 'both'
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate instance segmentation / cryptomatte
        
        Args:
            image_path: Input image
            output_path: Where to save result
            max_objects: Maximum number of objects to segment
            min_area: Minimum mask area in pixels
            output_mode: What to output
                - 'id_matte': Single image with colored IDs
                - 'layers': Individual mask files per object
                - 'both': Both outputs
            
        Returns:
            Dict with output_path(s) and metadata
        """
        self._ensure_initialized()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Generate masks
        print(f"Segmenting image ({w}x{h})...")
        masks = self.mask_generator.generate(image_rgb)
        
        # Filter and sort masks
        masks = [m for m in masks if m['area'] >= min_area]
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:max_objects]
        
        print(f"Found {len(masks)} objects")
        
        outputs = {}
        
        # Generate ID matte visualization
        if output_mode in ['id_matte', 'both']:
            id_matte = self._create_id_matte(masks, (h, w))
            
            params = {
                'method': 'crypto',
                'mode': 'id_matte',
                'num_objects': len(masks),
                'max_objects': max_objects,
            }
            unique_output = self._make_unique_path(output_path, params)
            cv2.imwrite(str(unique_output), cv2.cvtColor(id_matte, cv2.COLOR_RGB2BGR))
            outputs['id_matte'] = str(unique_output)
        
        # Generate individual layer masks
        if output_mode in ['layers', 'both']:
            layers_dir = output_path.parent / f"{output_path.stem}_layers"
            layers_dir.mkdir(parents=True, exist_ok=True)
            
            layer_paths = []
            for i, mask in enumerate(masks):
                mask_image = (mask['segmentation'].astype(np.uint8) * 255)
                layer_path = layers_dir / f"object_{i:03d}.png"
                cv2.imwrite(str(layer_path), mask_image)
                layer_paths.append(str(layer_path))
            
            outputs['layers'] = layer_paths
            outputs['layers_dir'] = str(layers_dir)
        
        return {
            "output_path": outputs.get('id_matte', outputs.get('layers', [None])[0]),
            "method": "crypto",
            "num_objects": len(masks),
            "outputs": outputs,
            "parameters": {
                'max_objects': max_objects,
                'min_area': min_area,
                'output_mode': output_mode,
            },
        }
    
    def _create_id_matte(self, masks: List[Dict], shape: Tuple[int, int]) -> np.ndarray:
        """
        Create cryptomatte-style ID visualization
        
        Each object gets a unique, visually distinct color.
        Colors are generated using golden ratio to maximize distinction.
        
        Args:
            masks: List of mask dictionaries from SAM2
            shape: Output shape (height, width)
            
        Returns:
            RGB image with colored object IDs
        """
        h, w = shape
        id_matte = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate distinct colors using golden ratio
        golden_ratio = 0.618033988749895
        
        for i, mask in enumerate(masks):
            # Generate color using golden ratio for maximum distinction
            hue = (i * golden_ratio) % 1.0
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 - (i % 4) * 0.05       # Vary brightness slightly
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            color = (int(r * 255), int(g * 255), int(b * 255))
            
            # Apply mask
            mask_bool = mask['segmentation']
            id_matte[mask_bool] = color
        
        return id_matte
    
    def get_masks(self, image_path: Path, max_objects: int = 50, min_area: int = 500) -> List[Dict]:
        """
        Get raw mask data (for programmatic use, EXR export, etc.)
        
        Returns:
            List of mask dictionaries with 'segmentation', 'area', 'bbox', etc.
        """
        self._ensure_initialized()
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = self.mask_generator.generate(image_rgb)
        masks = [m for m in masks if m['area'] >= min_area]
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:max_objects]
        
        return masks
    
    def segment_point(
        self,
        image_path: Path,
        point: Tuple[int, int],
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Segment object at a specific point (interactive mode)
        
        Args:
            image_path: Input image
            point: (x, y) coordinate to segment
            output_path: Where to save result
            
        Returns:
            Dict with mask output
        """
        # This would use SAM2's point-based prompting
        # For now, fall back to automatic segmentation
        # TODO: Implement point-based segmentation using SAM2Predictor
        
        return self.process(image_path, output_path, **kwargs)