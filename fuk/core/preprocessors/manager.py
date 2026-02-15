# core/preprocessors/manager.py
"""
Preprocessor Manager

Central orchestrator for all preprocessing operations.
Handles lazy loading, caching, config-based defaults, and unified interface.

Refactored to use centralized config system.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

from .canny import CannyPreprocessor, create_canny_preprocessor
from .openpose import OpenPosePreprocessor
from .depth import DepthPreprocessor, DepthModel
from .normals import NormalsPreprocessor, NormalsMethod
from .crypto import CryptoPreprocessor, SAMModel


class PreprocessorManager:
    """
    Central manager for all preprocessing operations
    
    Provides a unified interface and handles lazy loading of models.
    Each preprocessor is only initialized when first used.
    
    Usage:
        manager = PreprocessorManager(output_dir, config_dir=Path("config"))
        
        # Simple methods - defaults from config
        result = manager.canny(image_path, output_path)
        result = manager.depth(image_path, output_path)
        result = manager.normals(image_path, output_path)
        result = manager.crypto(image_path, output_path)
        
        # OpenPose (slower, lazy loaded)
        result = manager.openpose(image_path, output_path, detect_hand=True)
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        config_path: Optional[Path] = None  # Legacy support
    ):
        """
        Initialize the preprocessor manager
        
        Args:
            output_dir: Default output directory for preprocessed images
            config_dir: Directory containing defaults.json and other configs
            config_path: Legacy - path to config file (deprecated, use config_dir)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/preprocessed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Config directory for defaults
        self.config_dir = Path(config_dir) if config_dir else None
        self.config_path = Path(config_path) if config_path else None
        
        # Load all defaults once
        self._defaults = self._load_defaults()
        
        # Lazy-loaded preprocessors (None until first use)
        self._canny: Optional[CannyPreprocessor] = None
        self._openpose: Optional[OpenPosePreprocessor] = None
        self._depth_cache: Dict[DepthModel, DepthPreprocessor] = {}
        self._normals_cache: Dict[str, NormalsPreprocessor] = {}
        self._crypto_cache: Dict[SAMModel, CryptoPreprocessor] = {}
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load preprocessor defaults from config"""
        defaults = {
            "canny": {},
            "openpose": {},
            "depth": {},
            "normals": {},
            "crypto": {},
        }
        
        if self.config_dir:
            defaults_path = self.config_dir / "defaults.json"
            if defaults_path.exists():
                try:
                    with open(defaults_path) as f:
                        all_defaults = json.load(f)
                    preprocess_defaults = all_defaults.get("preprocess", {})
                    defaults.update(preprocess_defaults)
                    print(f"[PreprocessorManager] Loaded defaults from {defaults_path}")
                except Exception as e:
                    print(f"[PreprocessorManager] Warning: Could not load defaults: {e}")
        
        return defaults
    
    # =========================================================================
    # Canny Edge Detection
    # =========================================================================
    
    def canny(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        blur_kernel: Optional[int] = None,
        invert: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply Canny edge detection
        
        Fast, no model loading required. Defaults from config/defaults.json.
        
        Args:
            image_path: Input image path
            output_path: Output path (defaults to output_dir)
            low_threshold: Lower edge threshold (0-255)
            high_threshold: Upper edge threshold (0-255)
            blur_kernel: Gaussian blur kernel size
            invert: Invert output (white background)
        """
        # Lazy init with config defaults
        if self._canny is None:
            self._canny = CannyPreprocessor(defaults=self._defaults.get("canny", {}))
        
        output_path = output_path or self.output_dir / f"{Path(image_path).stem}_canny.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self._canny.process(
            image_path=Path(image_path),
            output_path=output_path,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur_kernel=blur_kernel,
            invert=invert,
            **kwargs
        )
    
    def canny_video(
        self,
        video_path: Path,
        output_path: Path,
        output_mode: str = "mp4",
        progress_callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply Canny edge detection to video
        
        Args:
            video_path: Input video path
            output_path: Output path
            output_mode: "mp4" or "sequence"
            progress_callback: Optional progress callback
            **kwargs: Canny parameters (low_threshold, high_threshold, etc.)
        """
        if self._canny is None:
            self._canny = CannyPreprocessor(defaults=self._defaults.get("canny", {}))
        
        return self._canny.process_video(
            video_path=Path(video_path),
            output_path=output_path,
            output_mode=output_mode,
            progress_callback=progress_callback,
            **kwargs
        )
    
    # =========================================================================
    # OpenPose Pose Estimation
    # =========================================================================
    
    def openpose(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        detect_body: Optional[bool] = None,
        detect_hand: Optional[bool] = None,
        detect_face: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply OpenPose pose estimation
        
        Lazy loads controlnet_aux on first use.
        
        Args:
            image_path: Input image
            output_path: Output path
            detect_body: Detect body keypoints
            detect_hand: Detect hand keypoints (slower)
            detect_face: Detect face keypoints (slower)
        """
        if self._openpose is None:
            self._openpose = OpenPosePreprocessor(config_path=self.config_path)
        
        # Apply defaults
        openpose_defaults = self._defaults.get("openpose", {})
        detect_body = detect_body if detect_body is not None else openpose_defaults.get("include_body", True)
        detect_hand = detect_hand if detect_hand is not None else openpose_defaults.get("include_hands", False)
        detect_face = detect_face if detect_face is not None else openpose_defaults.get("include_face", False)
        
        output_path = output_path or self.output_dir / f"{Path(image_path).stem}_pose.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self._openpose.process(
            image_path=Path(image_path),
            output_path=output_path,
            detect_body=detect_body,
            detect_hand=detect_hand,
            detect_face=detect_face,
            **kwargs
        )
    
    # =========================================================================
    # Depth Estimation
    # =========================================================================
    
    def depth(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        model: Optional[DepthModel] = None,
        invert: Optional[bool] = None,
        normalize: Optional[bool] = None,
        range_min: float = 0.0,
        range_max: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply depth estimation
        
        Caches depth models by type for reuse. Defaults from config.
        
        Args:
            image_path: Input image
            output_path: Output path
            model: Depth model to use
            invert: Invert depth (far = white)
            normalize: Normalize to [0, 1]
            range_min: Low end of depth range remap (0.0-1.0)
            range_max: High end of depth range remap (0.0-1.0)
        """
        # Apply defaults
        depth_defaults = self._defaults.get("depth", {})
        
        if model is None:
            model_name = depth_defaults.get("model", "depth_anything_v2")
            # Map config names to enum
            model_map = {
                "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                "da2_vit_large": DepthModel.DEPTH_ANYTHING_V2,
                "da3_mono_large": DepthModel.DA3_MONO_LARGE,
                "midas": DepthModel.MIDAS,
            }
            model = model_map.get(model_name, DepthModel.DEPTH_ANYTHING_V2)
        
        invert = invert if invert is not None else depth_defaults.get("invert", False)
        normalize = normalize if normalize is not None else depth_defaults.get("normalize", True)
        
        # Cache depth processors by model type
        if model not in self._depth_cache:
            self._depth_cache[model] = DepthPreprocessor(
                model_type=model,
                config_path=self.config_path
            )
        
        output_path = output_path or self.output_dir / f"{Path(image_path).stem}_depth.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self._depth_cache[model].process(
            image_path=Path(image_path),
            output_path=output_path,
            invert=invert,
            normalize=normalize,
            range_min=range_min,
            range_max=range_max,
            **kwargs
        )
    
    def get_raw_depth(
        self,
        image_path: Path,
        model: DepthModel = DepthModel.DEPTH_ANYTHING_V2
    ):
        """Get raw depth values as float32 array [0, 1]"""
        if model not in self._depth_cache:
            self._depth_cache[model] = DepthPreprocessor(
                model_type=model,
                config_path=self.config_path
            )
        
        return self._depth_cache[model].get_raw_depth(Path(image_path))
    
    # =========================================================================
    # Normal Map Estimation
    # =========================================================================
    
    def normals(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        method: Optional[NormalsMethod] = None,
        depth_model: Optional[DepthModel] = None,
        space: Optional[str] = None,
        flip_y: Optional[bool] = None,
        flip_x: Optional[bool] = None,
        intensity: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate surface normals
        
        Args:
            image_path: Input image
            output_path: Output path
            method: 'from_depth' (fast) or 'dsine' (quality)
            depth_model: Which depth model to use (for from_depth method)
            space: Normal space ('tangent', 'world', 'object')
            flip_y: Flip Y component
            flip_x: Flip X component
            intensity: Normal intensity (affects depth-derived)
        """
        # Apply defaults
        normals_defaults = self._defaults.get("normals", {})
        
        if method is None:
            method_name = normals_defaults.get("method", "depth_gradient")
            method_map = {
                "depth_gradient": NormalsMethod.FROM_DEPTH,
                "from_depth": NormalsMethod.FROM_DEPTH,
                "dsine": NormalsMethod.DSINE,
            }
            method = method_map.get(method_name, NormalsMethod.FROM_DEPTH)
        
        if depth_model is None:
            dm_name = normals_defaults.get("depth_model", "da3_mono_large")
            dm_map = {
                "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                "da3_mono_large": DepthModel.DA3_MONO_LARGE,
            }
            depth_model = dm_map.get(dm_name, DepthModel.DA3_MONO_LARGE)
        
        space = space if space is not None else normals_defaults.get("space", "tangent")
        flip_y = flip_y if flip_y is not None else normals_defaults.get("flip_y", False)
        flip_x = flip_x if flip_x is not None else normals_defaults.get("flip_x", False)
        intensity = intensity if intensity is not None else normals_defaults.get("intensity", 1.0)
        
        cache_key = f"{method.value}_{depth_model.value}"
        
        if cache_key not in self._normals_cache:
            self._normals_cache[cache_key] = NormalsPreprocessor(
                method=method,
                depth_model=depth_model,
                config_path=self.config_path
            )
        
        output_path = output_path or self.output_dir / f"{Path(image_path).stem}_normals.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self._normals_cache[cache_key].process(
            image_path=Path(image_path),
            output_path=output_path,
            space=space,
            flip_y=flip_y,
            flip_x=flip_x,
            intensity=intensity,
            **kwargs
        )
    
    def get_raw_normals(
        self,
        image_path: Path,
        method: NormalsMethod = NormalsMethod.FROM_DEPTH,
        depth_model: DepthModel = DepthModel.DEPTH_ANYTHING_V2,
        intensity: float = 1.0
    ):
        """Get raw normal vectors as float32 array [-1, 1]"""
        cache_key = f"{method.value}_{depth_model.value}"
        
        if cache_key not in self._normals_cache:
            self._normals_cache[cache_key] = NormalsPreprocessor(
                method=method,
                depth_model=depth_model,
                config_path=self.config_path
            )
        
        return self._normals_cache[cache_key].get_raw_normals(Path(image_path), intensity)
    
    # =========================================================================
    # Cryptomatte / Instance Segmentation
    # =========================================================================
    
    def crypto(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        model: Optional[SAMModel] = None,
        max_objects: Optional[int] = None,
        min_area: Optional[int] = None,
        output_mode: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate instance segmentation / cryptomatte
        
        Uses SAM2 for segmentation, outputs cryptomatte-style ID mattes.
        
        Args:
            image_path: Input image
            output_path: Output path
            model: SAM2 model size
            max_objects: Maximum objects to segment
            min_area: Minimum mask area in pixels
            output_mode: 'id_matte', 'layers', or 'both'
        """
        # Apply defaults
        crypto_defaults = self._defaults.get("crypto", {})
        
        if model is None:
            model_name = crypto_defaults.get("model", "sam2_large")
            model_map = {
                "sam2": SAMModel.LARGE,
                "sam2_large": SAMModel.LARGE,
                "sam2_base": SAMModel.BASE,
                "sam2_small": SAMModel.SMALL,
                "sam2_tiny": SAMModel.TINY,
            }
            model = model_map.get(model_name, SAMModel.LARGE)
        
        max_objects = max_objects if max_objects is not None else crypto_defaults.get("max_objects", 50)
        min_area = min_area if min_area is not None else crypto_defaults.get("min_area", 500)
        output_mode = output_mode if output_mode is not None else crypto_defaults.get("output_mode", "id_matte")
        
        if model not in self._crypto_cache:
            self._crypto_cache[model] = CryptoPreprocessor(
                model_size=model,
                config_path=self.config_path
            )
        
        output_path = output_path or self.output_dir / f"{Path(image_path).stem}_crypto.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self._crypto_cache[model].process(
            image_path=Path(image_path),
            output_path=output_path,
            max_objects=max_objects,
            min_area=min_area,
            output_mode=output_mode,
            **kwargs
        )
    
    def get_masks(
        self,
        image_path: Path,
        model: SAMModel = SAMModel.LARGE,
        max_objects: int = 50,
        min_area: int = 500
    ):
        """Get raw mask data from SAM2"""
        if model not in self._crypto_cache:
            self._crypto_cache[model] = CryptoPreprocessor(
                model_size=model,
                config_path=self.config_path
            )
        
        return self._crypto_cache[model].get_masks(
            Path(image_path),
            max_objects=max_objects,
            min_area=min_area
        )
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def process_all_layers(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        layers: Optional[Dict[str, bool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate all enabled AOV layers for an image
        
        Args:
            image_path: Input image
            output_dir: Directory for outputs
            layers: Dict of layer_name -> enabled
                {
                    'depth': True,
                    'normals': True,
                    'crypto': False,
                }
        
        Returns:
            Dict with all generated layer paths
        """
        layers = layers or {
            'depth': True,
            'normals': True,
            'crypto': False,
        }
        
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = Path(image_path)
        stem = image_path.stem
        
        results = {}
        
        if layers.get('depth'):
            try:
                results['depth'] = self.depth(
                    image_path,
                    output_dir / f"{stem}_depth.png",
                    **kwargs.get('depth', {})
                )
            except Exception as e:
                results['depth'] = {'error': str(e)}
        
        if layers.get('normals'):
            try:
                results['normals'] = self.normals(
                    image_path,
                    output_dir / f"{stem}_normals.png",
                    **kwargs.get('normals', {})
                )
            except Exception as e:
                results['normals'] = {'error': str(e)}
        
        if layers.get('crypto'):
            try:
                results['crypto'] = self.crypto(
                    image_path,
                    output_dir / f"{stem}_crypto.png",
                    **kwargs.get('crypto', {})
                )
            except Exception as e:
                results['crypto'] = {'error': str(e)}
        
        return results
    
    # =========================================================================
    # VRAM Management
    # =========================================================================
    
    def clear_caches(self, models: Optional[list] = None):
        """
        Unload preprocessor models from VRAM
        
        Call this after video processing to free VRAM for other tasks.
        
        Args:
            models: List of model types to clear ('depth', 'normals', 'crypto', 'openpose', 'canny')
                   If None, clears ALL cached models.
        
        Example:
            manager.clear_caches()  # Clear all
            manager.clear_caches(['depth'])  # Clear only depth models
        """
        import gc
        import torch
        
        cleared = []
        
        # Clear specific or all models
        clear_all = models is None
        models = models or ['depth', 'normals', 'crypto', 'openpose', 'canny']
        
        if ('depth' in models or clear_all) and self._depth_cache:
            for model_type, preprocessor in self._depth_cache.items():
                preprocessor.unload()
                cleared.append(f"depth:{model_type.value}")
            self._depth_cache.clear()
        
        if ('normals' in models or clear_all) and self._normals_cache:
            for cache_key, preprocessor in self._normals_cache.items():
                preprocessor.unload()
                cleared.append(f"normals:{cache_key}")
            self._normals_cache.clear()
        
        if ('crypto' in models or clear_all) and self._crypto_cache:
            for model_type, preprocessor in self._crypto_cache.items():
                preprocessor.unload()
                cleared.append(f"crypto:{model_type.value}")
            self._crypto_cache.clear()
        
        if ('openpose' in models or clear_all) and self._openpose is not None:
            self._openpose.unload()
            self._openpose = None
            cleared.append("openpose")
        
        if ('canny' in models or clear_all) and self._canny is not None:
            # Canny doesn't use GPU, but clear for consistency
            self._canny = None
            cleared.append("canny")
        
        # Force garbage collection and CUDA cache clear
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[PreprocessorManager] VRAM after clear - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        if cleared:
            print(f"[PreprocessorManager] Cleared caches: {', '.join(cleared)}")
        else:
            print(f"[PreprocessorManager] No models to clear")


# ============================================================================
# Factory Function
# ============================================================================

def create_preprocessor_manager(
    output_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> PreprocessorManager:
    """
    Factory function to create a PreprocessorManager with config
    
    Args:
        output_dir: Default output directory
        config_dir: Directory containing defaults.json
        
    Returns:
        Configured PreprocessorManager instance
    """
    return PreprocessorManager(
        output_dir=output_dir,
        config_dir=config_dir,
    )