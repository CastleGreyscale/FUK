# core/preprocessors/canny.py
"""
Canny Edge Detection Preprocessor

Good for:
- Architectural/structural control
- Line art generation
- Preserving sharp edges
- ControlNet edge conditioning

Refactored to use centralized config system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
import cv2

from .base import SimplePreprocessor


class CannyPreprocessor(SimplePreprocessor):
    """
    Canny edge detection - pure OpenCV, no model loading required
    
    Supports both single images and video processing.
    """
    
    def __init__(
        self,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Canny preprocessor
        
        Args:
            defaults: Default parameters from config (low_threshold, high_threshold, etc.)
        """
        super().__init__()
        self.defaults = defaults or {}
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        blur_kernel: Optional[int] = None,
        invert: Optional[bool] = None,
        exact_output: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply Canny edge detection
        
        Args:
            image_path: Input image path
            output_path: Where to save result
            low_threshold: Lower threshold for edge detection (0-255)
            high_threshold: Upper threshold for edge detection (0-255)
            blur_kernel: Gaussian blur kernel size (odd number)
            invert: Invert output (white background, black lines)
            exact_output: If True, write to exact output_path (for video frames)
            
        Returns:
            Dict with output_path and metadata
        """
        # Apply defaults from config
        low_threshold = low_threshold if low_threshold is not None else self.defaults.get('low_threshold', 100)
        high_threshold = high_threshold if high_threshold is not None else self.defaults.get('high_threshold', 200)
        blur_kernel = blur_kernel if blur_kernel is not None else self.defaults.get('blur_kernel', 5)
        invert = invert if invert is not None else self.defaults.get('invert', False)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Invert if requested (for white background)
        if invert:
            edges = 255 - edges
        
        # Convert to 3-channel for consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Build parameters dict for unique path and metadata
        params = {
            'method': 'canny',
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'blur_kernel': blur_kernel,
            'invert': invert,
        }
        
        # Save result - use exact path for video frames, unique path for single images
        final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
        cv2.imwrite(str(final_output), edges_rgb)
        
        return {
            "output_path": str(final_output),
            "method": "canny",
            "parameters": params,
            "input_path": str(image_path),
            "dimensions": {
                "width": image.shape[1],
                "height": image.shape[0],
            }
        }
    
    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        output_mode: str = "mp4",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        blur_kernel: Optional[int] = None,
        invert: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process video frame-by-frame with Canny edge detection
        
        Args:
            video_path: Input video path
            output_path: Output path (video or directory for sequences)
            output_mode: "mp4" or "sequence"
            progress_callback: Optional (progress, message) callback
            low_threshold, high_threshold, blur_kernel, invert: Canny parameters
            
        Returns:
            Dict with output info and processing stats
        """
        # Apply defaults
        low_threshold = low_threshold if low_threshold is not None else self.defaults.get('low_threshold', 100)
        high_threshold = high_threshold if high_threshold is not None else self.defaults.get('high_threshold', 200)
        blur_kernel = blur_kernel if blur_kernel is not None else self.defaults.get('blur_kernel', 5)
        invert = invert if invert is not None else self.defaults.get('invert', False)
        
        # Use parent class video processing with our parameters
        return super().process_video(
            video_path=video_path,
            output_path=output_path,
            output_mode=output_mode,
            progress_callback=progress_callback,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur_kernel=blur_kernel,
            invert=invert,
            **kwargs
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_canny_preprocessor(
    config_dir: Optional[Path] = None,
) -> CannyPreprocessor:
    """
    Factory function to create a Canny preprocessor with config-based defaults
    
    Args:
        config_dir: Directory containing defaults.json
        
    Returns:
        Configured CannyPreprocessor instance
    """
    defaults = {}
    
    if config_dir:
        defaults_path = Path(config_dir) / "defaults.json"
        if defaults_path.exists():
            try:
                with open(defaults_path) as f:
                    all_defaults = json.load(f)
                defaults = all_defaults.get("preprocess", {}).get("canny", {})
                print(f"[Canny] Loaded defaults: {defaults}")
            except Exception as e:
                print(f"[Canny] Warning: Could not load defaults: {e}")
    
    return CannyPreprocessor(defaults=defaults)


# ============================================================================
# Convenience function for direct use
# ============================================================================

def canny_edge_detection(
    image_path: Path,
    output_path: Path,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_kernel: int = 5,
    invert: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for one-off Canny edge detection
    
    For repeated use, create a CannyPreprocessor instance instead.
    """
    preprocessor = CannyPreprocessor()
    return preprocessor.process(
        image_path=image_path,
        output_path=output_path,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        blur_kernel=blur_kernel,
        invert=invert,
    )