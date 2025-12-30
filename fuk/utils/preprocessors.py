# utils/preprocessors.py
"""
Image Preprocessing Tools for FUK Pipeline

Provides:
- Canny edge detection
- OpenPose pose estimation  
- Depth estimation (MiDaS, Depth Anything V2/V3, ZoeDepth)
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import torch
import json
import hashlib
import time

# ============================================================================
# Depth Model Selection
# ============================================================================

class DepthModel(str, Enum):
    """Available depth estimation models"""
    MIDAS_SMALL = "midas_small"      # Fast, lower quality
    MIDAS_LARGE = "midas_large"      # Balanced (default fallback)
    DEPTH_ANYTHING_V2 = "depth_anything_v2"  # SOTA quality
    DEPTH_ANYTHING_V3 = "depth_anything_v3"  # Latest (if available)
    ZOEDEPTH = "zoedepth"            # Metric depth estimation


# ============================================================================
# Canny Edge Detection
# ============================================================================

class CannyPreprocessor:
    """
    Canny edge detection preprocessor
    
    Good for:
    - Architectural/structural control
    - Line art generation
    - Preserving sharp edges
    """
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        low_threshold: int = 100,
        high_threshold: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply Canny edge detection
        
        Args:
            image_path: Input image path
            output_path: Where to save result
            low_threshold: Lower threshold for edge detection (0-255)
            high_threshold: Upper threshold for edge detection (0-255)
            
        Returns:
            Dict with output_path and metadata
        """
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Convert to 3-channel for consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Save result with unique filename based on parameters
        params = {
            'method': 'canny',
            'low_threshold': low_threshold,
            'high_threshold': high_threshold
        }
        unique_output = self._make_unique_path(output_path, params)
        cv2.imwrite(str(unique_output), edges_rgb)
        
        return {
            "output_path": str(unique_output),
            "method": "canny",
            "parameters": {
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
            }
        }
    
    def _make_unique_path(self, base_path: Path, params: dict) -> Path:
        """Generate unique filename to prevent caching"""
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 10000
        
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        return parent / f"{stem}_{param_hash}_{timestamp}{suffix}"


# ============================================================================
# OpenPose Estimation
# ============================================================================

class OpenPosePreprocessor:
    """
    OpenPose pose estimation using controlnet_aux
    
    Good for:
    - Character pose control
    - Animation/motion transfer
    - Human figure consistency
    """
    
    def __init__(self):
        self.processor = None
    
    def _initialize(self):
        """Lazy load processor"""
        if self.processor is not None:
            return
        
        try:
            from controlnet_aux import OpenposeDetector
            print("Loading OpenPose detector...")
            self.processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            print("✓ OpenPose loaded")
        except ImportError:
            raise ImportError(
                "controlnet_aux not installed. Install with:\n"
                "pip install controlnet-aux"
            )
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        detect_hand: bool = False,
        detect_face: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect pose using OpenPose
        
        Args:
            image_path: Input image
            output_path: Where to save result
            detect_hand: Detect hand keypoints
            detect_face: Detect face keypoints
            
        Returns:
            Dict with output_path and metadata
        """
        
        if self.processor is None:
            self._initialize()
        
        # Load image
        image = Image.open(image_path)
        
        # Process with OpenPose
        pose_image = self.processor(
            image,
            hand_and_face=detect_hand or detect_face,
            include_hand=detect_hand,
            include_face=detect_face,
        )
        
        # Save result with unique filename
        params = {
            'method': 'openpose',
            'detect_hand': detect_hand,
            'detect_face': detect_face
        }
        unique_output = self._make_unique_path(output_path, params)
        pose_image.save(unique_output)
        
        return {
            "output_path": str(unique_output),
            "method": "openpose",
            "parameters": {
                "detect_hand": detect_hand,
                "detect_face": detect_face,
            }
        }
    
    def _make_unique_path(self, base_path: Path, params: dict) -> Path:
        """Generate unique filename to prevent caching"""
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 10000
        
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        return parent / f"{stem}_{param_hash}_{timestamp}{suffix}"


# ============================================================================
# Depth Estimation
# ============================================================================

class DepthPreprocessor:
    """
    Depth map estimation with multiple model options
    
    Models (in order of quality):
    1. Depth Anything V3 - Latest SOTA (if available)
    2. Depth Anything V2 - Excellent quality, requires checkpoint
    3. ZoeDepth - Metric depth estimation
    4. MiDaS Large - Good quality, slower
    5. MiDaS Small - Fast, lower quality
    
    Good for:
    - 3D scene control
    - Parallax effects
    - Bokeh/DOF effects
    - Spatial composition
    """
    
    def __init__(self, model_type: DepthModel = DepthModel.DEPTH_ANYTHING_V2):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Depth preprocessor using device: {self.device}")
    
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
    
    def _load_depth_anything_v2(self):
        """
        Load Depth Anything V2
        
        Requirements:
        1. Install package: pip install depth-anything-v2
           OR clone: git clone https://github.com/DepthAnything/Depth-Anything-V2
        2. Download checkpoint to: ~/ai/models/checkpoints/depth_anything_v2_vitl.pth
           From: https://huggingface.co/depth-anything/Depth-Anything-V2-Large
        """
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            print("Loading Depth Anything V2...")
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            # Use vitl (large) for best quality
            encoder = 'vitl'
            
            self.model = DepthAnythingV2(**model_configs[encoder])
            
            # Load checkpoint (user needs to download this)
            checkpoint_path = Path.home() / "ai" / "models" / "checkpoints" / "depth_anything_v2_vitl.pth"
            
            if checkpoint_path.exists():
                print(f"  Found checkpoint: {checkpoint_path}")
                self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
                print(f"✓ Loaded checkpoint successfully")
            else:
                print(f"✗ Checkpoint not found: {checkpoint_path}")
                print("  Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Large")
                print("  File: depth_anything_v2_vitl.pth")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✓ Depth Anything V2 loaded successfully")
            
        except ImportError as e:
            print(f"✗ Depth Anything V2 package not installed")
            print(f"  Import error: {e}")
            print(f"  Install with: pip install depth-anything-v2 --break-system-packages")
            print(f"  OR clone: git clone https://github.com/DepthAnything/Depth-Anything-V2")
            print(f"           cd Depth-Anything-V2")
            print(f"           pip install -e . --break-system-packages")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE  # FIX: Update type
            self._load_midas("DPT_Large")
            
        except FileNotFoundError as e:
            print(f"✗ Depth Anything V2 checkpoint not found: {e}")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE  # FIX: Update type
            self._load_midas("DPT_Large")
    
    def _load_depth_anything_v3(self):
        """
        Load Depth Anything V3 (latest SOTA)
        
        V3 improvements:
        - Better edge preservation
        - Improved small object detail
        - Faster inference
        """
        try:
            # V3 might use HuggingFace transformers
            from transformers import pipeline
            
            print("Loading Depth Anything V3...")
            
            # Try loading from HuggingFace
            self.model = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",  # Update when V3 released
                device=0 if self.device == "cuda" else -1
            )
            
            print("✓ Depth Anything V3 loaded")
            
        except Exception as e:
            print(f"⚠ Could not load Depth Anything V3: {e}")
            print("  Falling back to V2...")
            self.model_type = DepthModel.DEPTH_ANYTHING_V2  # FIX: Update type
            self._load_depth_anything_v2()
    
    def _load_zoedepth(self):
        """Load ZoeDepth (metric depth estimation)"""
        try:
            print("Loading ZoeDepth...")
            self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print("✓ ZoeDepth loaded")
            
        except Exception as e:
            print(f"⚠ Could not load ZoeDepth: {e}")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE  # FIX: Update type
            self._load_midas("DPT_Large")
    
    def _initialize(self):
        """Load model based on selected type"""
        if self.model_type == DepthModel.MIDAS_SMALL:
            self._load_midas("MiDaS_small")
        elif self.model_type == DepthModel.MIDAS_LARGE:
            self._load_midas("DPT_Large")
        elif self.model_type == DepthModel.DEPTH_ANYTHING_V2:
            self._load_depth_anything_v2()
        elif self.model_type == DepthModel.DEPTH_ANYTHING_V3:
            self._load_depth_anything_v3()
        elif self.model_type == DepthModel.ZOEDEPTH:
            self._load_zoedepth()
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        invert: bool = False,
        normalize: bool = True,
        colormap: Optional[str] = "inferno",
    ) -> Dict[str, Any]:
        """
        Estimate depth map
        
        Args:
            image_path: Input image
            output_path: Where to save result
            invert: Invert depth (far = white, near = black)
            normalize: Normalize depth to [0, 1]
            colormap: Apply colormap (None, 'inferno', 'viridis', 'magma')
            
        Returns:
            Dict with output_path and metadata
        """
        
        if self.model is None:
            self._initialize()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process based on ACTUAL model type (not requested, in case of fallback)
        if self.model_type in [DepthModel.MIDAS_SMALL, DepthModel.MIDAS_LARGE]:
            # MiDaS processing
            input_batch = self.transform(image_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth = prediction.cpu().numpy()
        
        elif self.model_type in [DepthModel.DEPTH_ANYTHING_V2, DepthModel.DEPTH_ANYTHING_V3]:
            # Depth Anything processing
            depth = self.model.infer_image(image_rgb)
        
        elif self.model_type == DepthModel.ZOEDEPTH:
            # ZoeDepth processing
            pil_image = Image.fromarray(image_rgb)
            depth = self.model.infer_pil(pil_image)
        
        # Normalize depth
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Invert if requested (near = white, far = black)
        if invert:
            depth = 1.0 - depth
        
        # Apply colormap or save as grayscale
        if colormap:
            # Convert to 0-255 range
            depth_uint8 = (depth * 255).astype(np.uint8)
            
            # Apply colormap
            colormap_func = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
            depth_colored = cv2.applyColorMap(depth_uint8, colormap_func)
            output_image = depth_colored
        else:
            # Grayscale output
            depth_uint8 = (depth * 255).astype(np.uint8)
            output_image = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        
        # Save result with unique filename
        params = {
            'method': 'depth',
            'model': self.model_type.value,
            'invert': invert,
            'normalize': normalize,
            'colormap': colormap
        }
        unique_output = self._make_unique_path(output_path, params)
        cv2.imwrite(str(unique_output), output_image)
        
        return {
            "output_path": str(unique_output),
            "method": "depth",
            "model": self.model_type.value,
            "parameters": {
                "invert": invert,
                "normalize": normalize,
                "colormap": colormap,
            }
        }
    
    def _make_unique_path(self, base_path: Path, params: dict) -> Path:
        """Generate unique filename to prevent caching"""
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 10000
        
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        return parent / f"{stem}_{param_hash}_{timestamp}{suffix}"


# ============================================================================
# Preprocessor Manager
# ============================================================================

class PreprocessorManager:
    """
    Central manager for all preprocessing operations
    
    Usage:
        manager = PreprocessorManager(output_dir)
        result = manager.canny(image_path, output_path, low_threshold=100)
        result = manager.openpose(image_path, output_path, detect_hand=True)
        result = manager.depth(image_path, output_path, model=DepthModel.DEPTH_ANYTHING_V2)
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/preprocessed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._canny = CannyPreprocessor()
        self._openpose = None  # Lazy load
        self._depth_processors = {}  # Cache depth models
    
    def canny(
        self,
        image_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply Canny edge detection"""
        return self._canny.process(image_path, output_path, **kwargs)
    
    def openpose(
        self,
        image_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply OpenPose pose estimation"""
        if self._openpose is None:
            self._openpose = OpenPosePreprocessor()
        return self._openpose.process(image_path, output_path, **kwargs)
    
    def depth(
        self,
        image_path: Path,
        output_path: Path,
        model: DepthModel = DepthModel.DEPTH_ANYTHING_V2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply depth estimation
        
        Args:
            image_path: Input image
            output_path: Where to save result
            model: Which depth model to use
            **kwargs: Additional parameters (invert, normalize, colormap)
        """
        # Cache depth processors by model type
        if model not in self._depth_processors:
            self._depth_processors[model] = DepthPreprocessor(model)
        
        return self._depth_processors[model].process(image_path, output_path, **kwargs)