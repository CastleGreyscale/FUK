# core/preprocessors/canny.py
"""
Canny Edge Detection Preprocessor

Good for:
- Architectural/structural control
- Line art generation
- Preserving sharp edges
- ControlNet edge conditioning
"""

from pathlib import Path
from typing import Dict, Any
import cv2

from .base import SimplePreprocessor


class CannyPreprocessor(SimplePreprocessor):
    """
    Canny edge detection - pure OpenCV, no model loading required
    """
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        low_threshold: int = 100,
        high_threshold: int = 200,
        blur_kernel: int = 5,
        invert: bool = False,
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
        
        # Save result - use exact path for video frames, unique path for single images
        params = {
            'method': 'canny',
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'blur_kernel': blur_kernel,
            'invert': invert,
        }
        final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
        cv2.imwrite(str(final_output), edges_rgb)
        
        return {
            "output_path": str(final_output),
            "method": "canny",
            "parameters": params,
        }