# core/preprocessors/depth.py
"""
Depth Estimation Preprocessor

Models (in order of quality):
1. Depth Anything V3 - Latest SOTA with multi-view support
2. Depth Anything V2 - Excellent quality, local checkpoint
3. ZoeDepth - Metric depth estimation
4. MiDaS Large - Good quality, slower
5. MiDaS Small - Fast, lower quality

Good for:
- 3D scene control
- Parallax effects
- Bokeh/DOF effects
- Spatial composition
- EXR depth channel export
"""

from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import torch

from .base import BasePreprocessor


class DepthModel(str, Enum):
    """Available depth estimation models"""
    MIDAS_SMALL = "midas_small"           # Fast, lower quality
    MIDAS_LARGE = "midas_large"           # Balanced (default fallback)
    DEPTH_ANYTHING_V2 = "depth_anything_v2"  # Great quality, local
    DEPTH_ANYTHING_V3 = "depth_anything_v3"  # DA3 default (alias for DA3_MONO_LARGE)
    # DA3 models (use HuggingFace API)
    DA3_MONO_LARGE = "da3_mono_large"     # Monocular depth, best for single images
    DA3_METRIC_LARGE = "da3_metric_large" # Metric depth in meters
    DA3_LARGE = "da3_large"               # Multi-view capable
    DA3_GIANT = "da3_giant"               # Largest model
    ZOEDEPTH = "zoedepth"                 # Metric depth estimation


# Map our enum to HuggingFace model IDs
DA3_MODEL_IDS = {
    DepthModel.DEPTH_ANYTHING_V3: "depth-anything/DA3MONO-LARGE",  # Default V3 alias
    DepthModel.DA3_MONO_LARGE: "depth-anything/DA3MONO-LARGE",
    DepthModel.DA3_METRIC_LARGE: "depth-anything/DA3METRIC-LARGE",
    DepthModel.DA3_LARGE: "depth-anything/DA3-LARGE-1.1",
    DepthModel.DA3_GIANT: "depth-anything/DA3-GIANT-1.1",
}


class DepthPreprocessor(BasePreprocessor):
    """
    Depth map estimation with multiple model options
    
    Handles model loading, fallbacks, and auto-download of checkpoints.
    """
    
    def __init__(
        self,
        model_type: DepthModel = DepthModel.DA3_MONO_LARGE,
        config_path: Optional[Path] = None
    ):
        super().__init__(config_path)
        self.model_type = model_type
        self.requested_model = model_type  # Track what was requested vs what loaded
        self.model = None
        self.transform = None
        self._is_da3 = model_type in DA3_MODEL_IDS
        print(f"Depth preprocessor initialized: {model_type.value} on {self.device}")
    
    def _initialize(self):
        """Load model based on selected type"""
        if self.model_type == DepthModel.MIDAS_SMALL:
            self._load_midas("MiDaS_small")
        elif self.model_type == DepthModel.MIDAS_LARGE:
            self._load_midas("DPT_Large")
        elif self.model_type == DepthModel.DEPTH_ANYTHING_V2:
            self._load_depth_anything_v2()
        elif self.model_type in DA3_MODEL_IDS:
            self._load_depth_anything_v3()
        elif self.model_type == DepthModel.ZOEDEPTH:
            self._load_zoedepth()
    
    def _load_midas(self, model_name: str = "DPT_Large"):
        """
        Load MiDaS depth estimation model
        
        Args:
            model_name: "MiDaS_small" (fast) or "DPT_Large" (quality)
        """
        print(f"Loading MiDaS {model_name}...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Load appropriate transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name == "DPT_Large":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print("✓ MiDaS loaded")
    
    def _load_depth_anything_v3(self):
        """
        Load Depth Anything V3 using HuggingFace API
        
        DA3 features:
        - Uses from_pretrained() pattern
        - Downloads models automatically from HuggingFace
        - Can also use local cache in /home/brad/ai/models/depth/
        - Supports monocular, metric, and multi-view modes
        
        Falls back to V2 if not available.
        """
        try:
            # Try vendor directory first for local installation
            vendor_path = self._get_vendor_path("Depth-Anything-3")
            if not vendor_path.exists():
                for name in ["depth-anything-3", "DepthAnything3", "DA3"]:
                    alt_path = self._get_vendor_path(name)
                    if alt_path.exists():
                        vendor_path = alt_path
                        break
            
            if vendor_path.exists():
                import sys
                sys.path.insert(0, str(vendor_path))
                print(f"Using Depth Anything 3 from vendor: {vendor_path}")
            
            # Import DA3 API
            from depth_anything_3.api import DepthAnything3
            
            model_id = DA3_MODEL_IDS.get(self.model_type, "depth-anything/DA3MONO-LARGE")
            print(f"Loading Depth Anything 3: {model_id}...")
            
            # Check for local model cache
            local_model_path = Path.home() / "ai" / "models" / "depth" / model_id.split("/")[-1]
            
            if local_model_path.exists():
                print(f"  Using local model: {local_model_path}")
                self.model = DepthAnything3.from_pretrained(str(local_model_path))
            else:
                # Download from HuggingFace
                print(f"  Downloading from HuggingFace: {model_id}")
                self.model = DepthAnything3.from_pretrained(model_id)
            
            self.model = self.model.to(device=self.device)
            self._is_da3 = True
            
            print(f"✓ Depth Anything 3 loaded: {model_id}")
            
        except ImportError as e:
            print(f"⚠ Depth Anything 3 not available: {e}")
            print("  Install with: pip install -e . (from Depth-Anything-3 repo)")
            print("  Falling back to V2...")
            self.model_type = DepthModel.DEPTH_ANYTHING_V2
            self._is_da3 = False
            self._load_depth_anything_v2()
        except Exception as e:
            print(f"⚠ Could not load Depth Anything 3: {e}")
            print("  Falling back to V2...")
            self.model_type = DepthModel.DEPTH_ANYTHING_V2
            self._is_da3 = False
            self._load_depth_anything_v2()

    def _load_depth_anything_v2(self):
        """
        Load Depth Anything V2
        
        Auto-downloads checkpoint from HuggingFace if missing.
        Supports both pip install and vendor directory approaches.
        """
        try:
            # Try importing - pip install or vendor
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
                print("Using installed depth-anything-v2 package")
            except ImportError:
                # Try vendor directory
                vendor_path = self._get_vendor_path("Depth-Anything-V2")
                if not vendor_path.exists():
                    raise ImportError(
                        f"depth-anything-v2 not found. Either:\n"
                        f"  1. pip install depth-anything-v2\n"
                        f"  2. Clone to vendor: git clone https://github.com/DepthAnything/Depth-Anything-V2 {vendor_path}"
                    )
                
                import sys
                sys.path.insert(0, str(vendor_path))
                from depth_anything_v2.dpt import DepthAnythingV2
                print(f"Using depth-anything-v2 from vendor: {vendor_path}")
            
            print("Loading Depth Anything V2...")
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            encoder = 'vitl'  # Large for best quality
            self.model = DepthAnythingV2(**model_configs[encoder])
            
            # Find checkpoint
            checkpoint_path = self._find_depth_anything_v2_checkpoint()
            
            if not checkpoint_path:
                raise FileNotFoundError("Checkpoint not found")
            
            # Load checkpoint
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            
            self.model.to(self.device)
            self.model.eval()
            self._is_da3 = False
            
            print("✓ Depth Anything V2 loaded successfully")
            
        except (ImportError, FileNotFoundError) as e:
            print(f"✗ Depth Anything V2 not available: {e}")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE
            self._is_da3 = False
            self._load_midas("DPT_Large")
    
    def _find_depth_anything_v2_checkpoint(self) -> Optional[Path]:
        """Search for Depth Anything V2 checkpoint in known locations"""
        
        # Try auto-download first if config available
        if self.config_path and self.config_path.exists():
            try:
                import importlib.util
                downloader_path = Path(__file__).parent.parent / "model_downloader.py"
                if downloader_path.exists():
                    spec = importlib.util.spec_from_file_location("model_downloader", downloader_path)
                    downloader_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(downloader_module)
                    
                    checkpoint_path = downloader_module.ensure_model_downloaded(
                        "depth_anything_v2", self.config_path
                    )
                    if checkpoint_path:
                        return Path(checkpoint_path)
            except Exception as e:
                print(f"  Auto-download attempt failed: {e}")
        
        # Manual search in known locations
        search_paths = [
            # User's model directory
            Path.home() / "ai" / "models" / "depth" / "depth_anything_v2_vitl.pth",
            Path.home() / "ai" / "models" / "checkpoints" / "depth_anything_v2_vitl.pth",
            # Vendor directory
            self._get_vendor_path("Depth-Anything-V2") / "checkpoints" / "depth_anything_v2_vitl.pth",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        print("✗ Checkpoint not found. Searched:")
        for path in search_paths:
            print(f"    - {path}")
        print("  Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Large")
        
        return None
    
    def _load_zoedepth(self):
        """Load ZoeDepth (metric depth estimation)"""
        try:
            print("Loading ZoeDepth...")
            self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self._is_da3 = False
            print("✓ ZoeDepth loaded")
            
        except Exception as e:
            print(f"⚠ Could not load ZoeDepth: {e}")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE
            self._load_midas("DPT_Large")
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate depth map
        
        Args:
            image_path: Input image
            output_path: Where to save result
            **kwargs:
                invert: Invert depth (far = white, near = black)
                normalize: Normalize depth to [0, 1]
                colormap: Apply colormap (None, 'inferno', 'viridis', 'magma', 'plasma')
                exact_output: If True, write to exact output_path (for video frames)
                process_res: DA3 processing resolution (default 756)
                process_res_method: DA3 resize method (default 'lower_bound_resize')
                
        Returns:
            Dict with output_path and metadata
        """
        self._ensure_initialized()
        
        # Extract parameters from kwargs
        invert = kwargs.get('invert', False)
        normalize = kwargs.get('normalize', True)
        colormap = kwargs.get('colormap', 'inferno')
        exact_output = kwargs.get('exact_output', False)
        process_res = kwargs.get('process_res', 1344)
        process_res_method = kwargs.get('process_res_method', 'lower_bound_resize')
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process based on ACTUAL model type (handles fallbacks)
        depth = self._infer_depth(image_rgb, str(image_path), process_res, process_res_method)
        
        # Normalize
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Invert if requested
        if invert:
            depth = 1.0 - depth
        
        # Apply colormap or grayscale
        if colormap:
            depth_uint8 = (depth * 255).astype(np.uint8)
            colormap_func = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
            output_image = cv2.applyColorMap(depth_uint8, colormap_func)
        else:
            depth_uint8 = (depth * 255).astype(np.uint8)
            output_image = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        
        # Save - use exact path for video frames, unique path for single images
        params = {
            'method': 'depth',
            'model': self.model_type.value,
            'invert': invert,
            'normalize': normalize,
            'colormap': colormap,
        }
        final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
        cv2.imwrite(str(final_output), output_image)
        
        return {
            "output_path": str(final_output),
            "method": "depth",
            "model": self.model_type.value,
            "parameters": params,
        }
    
    def _infer_depth(self, image_rgb: np.ndarray, image_path: str = None, 
                     process_res: int = 1344, process_res_method: str = "lower_bound_resize") -> np.ndarray:
        """Run inference based on loaded model type"""
        
        if self.model_type in [DepthModel.MIDAS_SMALL, DepthModel.MIDAS_LARGE]:
            input_batch = self.transform(image_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            return prediction.cpu().numpy()
        
        elif self._is_da3 and self.model_type in DA3_MODEL_IDS:
            # DA3 uses different inference API
            if image_path:
                prediction = self.model.inference(
                    [image_path],
                    process_res=process_res,
                    process_res_method=process_res_method,
                )
            else:
                # Save temp file if we only have array
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_path = f.name
                    Image.fromarray(image_rgb).save(temp_path)
                    prediction = self.model.inference(
                        [temp_path],
                        process_res=process_res,
                        process_res_method=process_res_method,
                    )
                    Path(temp_path).unlink()
            
            # prediction.depth is [N, H, W] array
            depth = prediction.depth[0]
            
            # Resize to match input if needed
            if depth.shape[:2] != image_rgb.shape[:2]:
                depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]))
            
            return depth
        
        elif self.model_type == DepthModel.DEPTH_ANYTHING_V2:
            return self.model.infer_image(image_rgb)
        
        elif self.model_type == DepthModel.ZOEDEPTH:
            pil_image = Image.fromarray(image_rgb)
            return self.model.infer_pil(pil_image)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_raw_depth(self, image_path: Path) -> np.ndarray:
        """
        Get raw depth values (for EXR export, normal calculation, etc.)
        
        Returns normalized float32 depth array [0, 1]
        """
        self._ensure_initialized()
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = self._infer_depth(image_rgb, str(image_path))
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth.astype(np.float32)