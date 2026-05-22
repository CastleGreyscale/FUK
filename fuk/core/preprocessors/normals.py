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
        depth_model: DepthModel = DepthModel.DA3_MONO_LARGE,
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
        Load DSINE normal estimator (v02, dsine.pt checkpoint).
        Manually manages sys.path so DSINE's local 'utils/' is found before
        any installed PyPI 'utils' package (pip install utils installs the
        wrong thing — uninstall it if present).
        """
        try:
            vendor_path = self._get_vendor_path("DSINE")

            if not vendor_path.exists():
                raise ImportError(
                    f"DSINE not found at {vendor_path}\n"
                    f"Install: git clone https://github.com/baegwangbin/DSINE {vendor_path}"
                )

            checkpoint_path = vendor_path / "checkpoints" / "dsine.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"DSINE checkpoint not found: {checkpoint_path}\n"
                    f"Download: https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
                )

            print("Loading DSINE normal estimator...")

            import sys
            vendor_str = str(vendor_path)

            # Ensure DSINE's local dirs come first — beats any installed PyPI 'utils'
            if vendor_str in sys.path:
                sys.path.remove(vendor_str)
            sys.path.insert(0, vendor_str)

            # Evict any cached PyPI 'utils' and any stale DSINE model imports so
            # everything re-resolves fresh from vendor_str at the front of sys.path.
            stale = [
                k for k in list(sys.modules)
                if k == 'utils' or k.startswith('utils.')
                or k == 'models' or k.startswith('models.')
            ]
            for k in stale:
                del sys.modules[k]

            from models.dsine import DSINE as DSINEFactory

            ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            model = DSINEFactory()
            model.load_state_dict(ckpt['model'], strict=True)
            model.eval()
            model = model.to(self.device)
            model.pixel_coords = model.pixel_coords.to(self.device)
            self.model = model

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
        
        # Load image (handles EXR via ANYDEPTH + gamma correction)
        image = self.load_image_bgr(image_path)

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
    
    @staticmethod
    def _dsine_padding(H: int, W: int):
        """Compute (l, r, t, b) padding to make H and W multiples of 32."""
        def _side(n):
            if n % 32 == 0:
                return 0, 0
            new = 32 * ((n // 32) + 1)
            a = (new - n) // 2
            return a, (new - n) - a
        l, r = _side(W)
        t, b = _side(H)
        return l, r, t, b

    def _dsine_intrins(self, H: int, W: int, fov_deg: float = 60.0):
        """Pinhole intrinsic matrix from FOV, principal point at image centre."""
        import math
        f = (max(H, W) / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        intrins = torch.tensor(
            [[f, 0, W / 2.0 - 0.5],
             [0, f, H / 2.0 - 0.5],
             [0, 0, 1.0]],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # (1, 3, 3)
        return intrins

    def _normals_from_dsine(self, image: np.ndarray) -> np.ndarray:
        """
        Compute normals using the DSINE v02 model.
        Preprocessing is inlined (ImageNet norm, 32px padding, pinhole intrinsics)
        so there is no runtime dependency on DSINE's local utils/ package.

        Args:
            image: BGR uint8 numpy array

        Returns:
            float32 normal map in [-1, 1], shape (H, W, 3)
        """
        import torch.nn.functional as F
        from torchvision.transforms.functional import normalize as tvf_normalize

        h, w = image.shape[:2]

        # BGR -> RGB float32 tensor [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad to multiples of 32
        l, r, t, b = self._dsine_padding(h, w)
        img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)

        # ImageNet normalisation
        img = tvf_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Pinhole intrinsics; shift principal point to account for padding
        intrins = self._dsine_intrins(h, w)
        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        with torch.no_grad():
            pred = self.model(img, intrins=intrins)[-1]  # (1, 3, H_pad, W_pad)
            pred = pred[:, :, t:t + h, l:l + w]         # unpad

        normals = pred[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        return normals.astype(np.float32)
    
    def get_raw_normals(self, image_path: Path, intensity: float = 1.0) -> np.ndarray:
        """
        Get raw normal vectors (for EXR export, etc.)
        
        Returns float32 normal array in [-1, 1] range
        """
        self._ensure_initialized()
        
        image = self.load_image_bgr(image_path)

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