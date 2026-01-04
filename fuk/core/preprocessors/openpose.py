# core/preprocessors/openpose.py
"""
OpenPose Pose Estimation Preprocessor

Good for:
- Character pose control
- Animation/motion transfer
- Human figure consistency
- ControlNet pose conditioning
"""

from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from .base import BasePreprocessor


class OpenPosePreprocessor(BasePreprocessor):
    """
    OpenPose pose estimation using controlnet_aux
    
    Requires: pip install controlnet-aux
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__(config_path)
        self.processor = None
    
    def _initialize(self):
        """Lazy load the OpenPose detector"""
        try:
            from controlnet_aux import OpenposeDetector
            
            print("Loading OpenPose detector...")
            self.processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            print("✓ OpenPose loaded")
            
        except ImportError:
            raise ImportError(
                "controlnet_aux not installed. Install with:\n"
                "pip install controlnet-aux --break-system-packages"
            )
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        detect_body: bool = True,
        detect_hand: bool = False,
        detect_face: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect pose using OpenPose
        
        Args:
            image_path: Input image
            output_path: Where to save result
            detect_body: Detect body keypoints (always True)
            detect_hand: Detect hand keypoints (slower)
            detect_face: Detect face keypoints (slower)
            
        Returns:
            Dict with output_path and metadata
        """
        self._ensure_initialized()
        
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
            'detect_body': detect_body,
            'detect_hand': detect_hand,
            'detect_face': detect_face,
        }
        unique_output = self._make_unique_path(output_path, params)
        pose_image.save(unique_output)
        
        return {
            "output_path": str(unique_output),
            "method": "openpose",
            "parameters": params,
        }
