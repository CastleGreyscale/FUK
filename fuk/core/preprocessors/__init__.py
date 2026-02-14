# core/preprocessors/__init__.py
"""
FUK Image Preprocessors

AOV / Control Image Generation:
- Canny edge detection
- OpenPose pose estimation
- Depth estimation (MiDaS, Depth Anything V2/V3, ZoeDepth)
- Normal map estimation (depth-derived or DSINE)
- Cryptomatte / Instance segmentation (SAM2)

Video Batch Processing:
- VideoDepthBatchProcessor - Temporally consistent depth maps
- VideoCryptoBatchProcessor - Temporally consistent crypto mattes (object tracking)

Usage:
    from core.preprocessors import PreprocessorManager, DepthModel
    
    manager = PreprocessorManager(output_dir="outputs/preprocessed")
    
    # Edge detection (fast, no model)
    result = manager.canny(image_path, output_path, low_threshold=100)
    
    # Depth estimation
    result = manager.depth(image_path, output_path, model=DepthModel.DEPTH_ANYTHING_V2)
    
    # Normal maps
    result = manager.normals(image_path, output_path)
    
    # Instance segmentation / cryptomatte
    result = manager.crypto(image_path, output_path)
    
    # Pose estimation (slower)
    result = manager.openpose(image_path, output_path, detect_hand=True)
    
    # Generate all AOVs at once
    results = manager.process_all_layers(image_path, output_dir, layers={
        'depth': True,
        'normals': True,
        'crypto': True,
    })
    
    # Video batch processing (for temporal consistency)
    from core.preprocessors import VideoCryptoBatchProcessor
    
    processor = VideoCryptoBatchProcessor()
    result = processor.process_video_batch(
        video_path=video,
        output_path=output,
        model_size="large",
        max_objects=50,
    )
"""

# Core manager
from .manager import PreprocessorManager

# Individual preprocessors (for direct use if needed)
from .canny import CannyPreprocessor
from .openpose import OpenPosePreprocessor
from .depth import DepthPreprocessor, DepthModel
from .normals import NormalsPreprocessor, NormalsMethod
from .crypto import CryptoPreprocessor, SAMModel



# Base class (for custom preprocessors)
from .base import BasePreprocessor, SimplePreprocessor

__all__ = [
    # Manager
    "PreprocessorManager",
    
    # Enums
    "DepthModel",
    "NormalsMethod",
    "SAMModel",
    
    # Preprocessors
    "CannyPreprocessor",
    "OpenPosePreprocessor",
    "DepthPreprocessor",
    "NormalsPreprocessor",
    "CryptoPreprocessor",

    # Base classes
    "BasePreprocessor",
    "SimplePreprocessor",
]
