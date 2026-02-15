# core/preprocessors/normals.py
"""
Surface Normal Map Preprocessor

Methods:
1. Depth-derived - Fast, computed from depth gradient (no extra model)
2. DSINE - Dedicated normal estimation model (better quality)

Good for:
- Relighting in comp
- Material/surface detail
- 3D reconstruction
- ControlNet normal conditioning
"""

from pathlib import Path
from typing import Dict, Any, Optional, Literal
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import torch

from .base import BasePreprocessor
from .depth import DepthPreprocessor, DepthModel


class NormalsMethod(str, Enum):
    """Available normal estimation methods"""
    FROM_DEPTH = "from_depth"    # Derive from depth map (fast, no extra model)
    DSINE = "dsine"              # Dedicated normal estimator (better quality)


class NormalsPreprocessor(BasePreprocessor):
    """
    Surface normal map estimation
    
    Can derive normals from depth (fast) or use dedicated DSINE model (quality).
    """
    
    def __init__(
        self,
        method: NormalsMethod = NormalsMethod.FROM_DEPTH,
        depth_model: DepthModel = DepthModel.DEPTH_ANYTHING_V2,
        config_path: Optional[Path] = None
    ):
        super().__init__(config_path)
        self.method = method
        self.depth_model = depth_model
        self.model = None
        self._depth_processor = None
        print(f"Normals preprocessor initialized: {method.value} on {self.device}")
    
    def _initialize(self):
        """Load model based on method"""
        if self.method == NormalsMethod.FROM_DEPTH:
            # Use depth processor for depth-derived normals
            self._depth_processor = DepthPreprocessor(
                model_type=self.depth_model,
                config_path=self.config_path
            )
        elif self.method == NormalsMethod.DSINE:
            self._load_dsine()
    
    def _load_dsine(self):
        """
        Load DSINE normal estimator
        
        Requires vendor: git clone https://github.com/baegwangbin/DSINE vendor/DSINE
        """
        try:
            vendor_path = self._get_vendor_path("DSINE")
            
            if not vendor_path.exists():
                raise ImportError(
                    f"DSINE not found at {vendor_path}\n"
                    f"Install with: git clone https://github.com/baegwangbin/DSINE {vendor_path}"
                )
            
            import sys
            # Add vendor path to sys.path if not already there
            vendor_str = str(vendor_path)
            if vendor_str not in sys.path:
                sys.path.insert(0, vendor_str)
            
            print("Loading DSINE normal estimator...")
            
            # Try different import patterns depending on DSINE structure
            DSINE = None
            import_error = None
            
            # Pattern 1: Direct import from models.dsine
            try:
                from models.dsine import DSINE as DSINEModel
                DSINE = DSINEModel
            except ImportError as e1:
                import_error = str(e1)
                
                # Pattern 2: Try importing from projects.dsine 
                try:
                    from projects.dsine.models.dsine import DSINE as DSINEModel
                    DSINE = DSINEModel
                except ImportError as e2:
                    
                    # Pattern 3: Try main DSINE module
                    try:
                        from DSINE.models.dsine import DSINE as DSINEModel
                        DSINE = DSINEModel
                    except ImportError as e3:
                        raise ImportError(
                            f"Could not import DSINE model. Tried:\n"
                            f"  1. from models.dsine import DSINE - {e1}\n"
                            f"  2. from projects.dsine.models.dsine import DSINE - {e2}\n"
                            f"  3. from DSINE.models.dsine import DSINE - {e3}\n"
                            f"Check your DSINE installation at: {vendor_path}"
                        )
            
            # Find checkpoint
            checkpoint_path = None
            possible_checkpoint_paths = [
                vendor_path / "checkpoints" / "dsine.pt",
                vendor_path / "weights" / "dsine.pt",
                vendor_path / "pretrained" / "dsine.pt",
                vendor_path / "checkpoints" / "dsine_v00.pt",
            ]
            
            for path in possible_checkpoint_paths:
                if path.exists():
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"DSINE checkpoint not found. Tried:\n" +
                    "\n".join(f"  - {p}" for p in possible_checkpoint_paths) +
                    f"\n\nDownload from: https://github.com/baegwangbin/DSINE/releases"
                )
            
            # Load model
            self.model = DSINE()
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ DSINE loaded from {checkpoint_path.name}")
                
        except Exception as e:
            print(f"⚠ Could not load DSINE: {e}")
            print("  Falling back to depth-derived normals...")
            self.method = NormalsMethod.FROM_DEPTH
            self._depth_processor = DepthPreprocessor(
                model_type=self.depth_model,
                config_path=self.config_path
            )
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        space: Literal["tangent", "world", "object"] = "tangent",
        flip_y: bool = False,
        flip_x: bool = False,
        intensity: float = 1.0,
        exact_output: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate surface normals
        
        Args:
            image_path: Input image
            output_path: Where to save result
            space: Normal space ('tangent', 'world', 'object')
            flip_y: Flip Y component (for different engine conventions)
            flip_x: Flip X component
            intensity: Normal intensity multiplier (affects depth-derived)
            exact_output: If True, write to exact output_path (for video frames)
            
        Returns:
            Dict with output_path and metadata
        """
        self._ensure_initialized()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Generate normals based on method
        if self.method == NormalsMethod.FROM_DEPTH:
            normals = self._normals_from_depth(image_path, intensity)
        else:
            normals = self._normals_from_dsine(image)
        
        # Apply flips if needed
        if flip_x:
            normals[:, :, 0] = -normals[:, :, 0]
        if flip_y:
            normals[:, :, 1] = -normals[:, :, 1]
        
        # Convert from [-1, 1] to [0, 255] for standard normal map encoding
        # Normal map convention: R=X, G=Y, B=Z
        normals_uint8 = ((normals + 1) * 0.5 * 255).astype(np.uint8)
        
        # OpenCV uses BGR, so swap R and B
        normals_bgr = cv2.cvtColor(normals_uint8, cv2.COLOR_RGB2BGR)
        
        # Save - use exact path for video frames, unique path for single images
        params = {
            'method': 'normals',
            'estimation': self.method.value,
            'space': space,
            'flip_y': flip_y,
            'flip_x': flip_x,
            'intensity': intensity,
        }
        final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
        cv2.imwrite(str(final_output), normals_bgr)
        
        return {
            "output_path": str(final_output),
            "method": "normals",
            "estimation": self.method.value,
            "parameters": params,
            "raw_normals": normals,  # float32 [-1, 1] for lossless EXR export
        }
    
    def _normals_from_depth(self, image_path: Path, intensity: float = 1.0) -> np.ndarray:
        """
        Compute normals from depth gradient
        
        Uses Sobel operators to find surface gradients, then computes
        normal vectors from the gradient field.
        
        Args:
            image_path: Input image path
            intensity: Multiplier for gradient (higher = more pronounced normals)
            
        Returns:
            Normal map as float32 array in [-1, 1] range, shape (H, W, 3)
        """
        # Get depth map
        depth = self._depth_processor.get_raw_depth(image_path)
        
        # Compute gradients using Sobel
        # Scale depth for better gradient calculation
        depth_scaled = depth * intensity
        
        # Sobel gradients
        grad_x = cv2.Sobel(depth_scaled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_scaled, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normal vector: n = normalize([-dz/dx, -dz/dy, 1])
        # The Z component is 1 (pointing toward camera)
        h, w = depth.shape
        normals = np.zeros((h, w, 3), dtype=np.float32)
        
        normals[:, :, 0] = -grad_x  # X component (red)
        normals[:, :, 1] = -grad_y  # Y component (green)  
        normals[:, :, 2] = 1.0      # Z component (blue)
        
        # Normalize each vector
        norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
        normals = normals / (norm + 1e-8)
        
        return normals
    
    def _normals_from_dsine(self, image: np.ndarray) -> np.ndarray:
        """
        Compute normals using DSINE model
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            Normal map as float32 array in [-1, 1] range, shape (H, W, 3)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Prepare input tensor
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            normals = self.model(image_tensor)
        
        # Convert to numpy
        normals = normals.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Resize if needed
        if normals.shape[:2] != (h, w):
            normals = cv2.resize(normals, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return normals
    
    def get_raw_normals(self, image_path: Path, intensity: float = 1.0) -> np.ndarray:
        """
        Get raw normal vectors (for EXR export, etc.)
        
        Returns float32 normal array in [-1, 1] range
        """
        self._ensure_initialized()
        
        image = cv2.imread(str(image_path))
        
        if self.method == NormalsMethod.FROM_DEPTH:
            return self._normals_from_depth(image_path, intensity)
        else:
            return self._normals_from_dsine(image)
    
    def unload(self):
        """
        Unload normals models from VRAM
        
        Handles both DSINE model and depth processor.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self._depth_processor is not None:
            self._depth_processor.unload()
            self._depth_processor = None
        
        # Call base cleanup
        super().unload()