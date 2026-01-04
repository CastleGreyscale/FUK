# core/preprocessors/manager.py
"""
Preprocessor Manager

Central orchestrator for all preprocessing operations.
Handles lazy loading, caching, and unified interface.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .canny import CannyPreprocessor
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
        manager = PreprocessorManager(output_dir)
        
        # Simple methods
        result = manager.canny(image_path, output_path, low_threshold=100)
        result = manager.depth(image_path, output_path, model=DepthModel.DEPTH_ANYTHING_V2)
        result = manager.normals(image_path, output_path)
        result = manager.crypto(image_path, output_path)
        
        # OpenPose (slower, lazy loaded)
        result = manager.openpose(image_path, output_path, detect_hand=True)
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize the preprocessor manager
        
        Args:
            output_dir: Default output directory for preprocessed images
            config_path: Path to config file (for model auto-download)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/preprocessed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path) if config_path else None
        
        # Lazy-loaded preprocessors
        self._canny = CannyPreprocessor()
        self._openpose: Optional[OpenPosePreprocessor] = None
        self._depth_cache: Dict[DepthModel, DepthPreprocessor] = {}
        self._normals_cache: Dict[str, NormalsPreprocessor] = {}
        self._crypto_cache: Dict[SAMModel, CryptoPreprocessor] = {}
    
    # =========================================================================
    # Canny Edge Detection
    # =========================================================================
    
    def canny(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        low_threshold: int = 100,
        high_threshold: int = 200,
        blur_kernel: int = 5,
        invert: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply Canny edge detection
        
        Fast, no model loading required.
        
        Args:
            image_path: Input image path
            output_path: Output path (defaults to output_dir)
            low_threshold: Lower edge threshold (0-255)
            high_threshold: Upper edge threshold (0-255)
            blur_kernel: Gaussian blur kernel size
            invert: Invert output (white background)
        """
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
    
    # =========================================================================
    # OpenPose Pose Estimation
    # =========================================================================
    
    def openpose(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        detect_body: bool = True,
        detect_hand: bool = False,
        detect_face: bool = False,
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
        model: DepthModel = DepthModel.DEPTH_ANYTHING_V2,
        invert: bool = False,
        normalize: bool = True,
        colormap: Optional[str] = "inferno",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply depth estimation
        
        Caches depth models by type for reuse.
        
        Args:
            image_path: Input image
            output_path: Output path
            model: Depth model to use
            invert: Invert depth (far = white)
            normalize: Normalize to [0, 1]
            colormap: Apply colormap (None for grayscale)
        """
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
            colormap=colormap,
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
        method: NormalsMethod = NormalsMethod.FROM_DEPTH,
        depth_model: DepthModel = DepthModel.DEPTH_ANYTHING_V2,
        space: str = "tangent",
        flip_y: bool = False,
        flip_x: bool = False,
        intensity: float = 1.0,
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
        model: SAMModel = SAMModel.LARGE,
        max_objects: int = 50,
        min_area: int = 500,
        output_mode: str = "id_matte",
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
