# core/preprocessors/base.py
"""
Base class for all FUK preprocessors

Provides shared utilities:
- Unique filename generation (prevents caching issues)
- Common output patterns
- Device detection
- Video processing support
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import json
import hashlib
import time
import torch


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors
    
    Subclasses must implement:
    - process(): Main processing method for single images
    - _initialize(): Lazy model loading
    
    Video processing is handled automatically via process_video()
    which calls process() on each frame.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
    
    @abstractmethod
    def process(
        self,
        image_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image
        
        Args:
            image_path: Input image path
            output_path: Where to save result
            **kwargs: Processor-specific parameters
            
        Returns:
            Dict with:
                - output_path: Final output location
                - method: Processor name
                - parameters: Dict of parameters used
        """
        pass
    
    @abstractmethod
    def _initialize(self):
        """Lazy-load models and resources"""
        pass
    
    def _ensure_initialized(self):
        """Call before processing to ensure model is loaded"""
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    def _make_unique_path(
        self, 
        base_path: Path, 
        params: dict,
        exact_output: bool = False
    ) -> Path:
        """
        Generate unique filename to prevent caching issues
        
        Appends a hash of parameters + timestamp to the filename.
        This ensures different parameter combinations produce different files
        and browser/UI caching doesn't serve stale results.
        
        Args:
            base_path: Original output path
            params: Dictionary of parameters to hash
            exact_output: If True, skip unique naming and use base_path as-is.
                         Use this for video frame processing where the caller
                         expects output at the exact specified path.
            
        Returns:
            New path with unique suffix, or base_path if exact_output=True
        """
        # Skip unique naming for video frame processing
        if exact_output:
            return base_path
        
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 10000
        
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        return parent / f"{stem}_{param_hash}_{timestamp}{suffix}"
    
    def _get_vendor_path(self, vendor_name: str) -> Path:
        """
        Get path to a vendor directory
        
        Searches relative to this file's location:
        core/preprocessors/base.py -> core/ -> vendor/
        
        Args:
            vendor_name: Name of vendor directory
            
        Returns:
            Path to vendor directory
        """
        # Go up: preprocessors/ -> core/ -> fuk/ -> vendor/
        return Path(__file__).parent.parent.parent / "vendor" / vendor_name
    
    def _try_import_from_vendor(self, vendor_name: str, module_path: str):
        """
        Try importing a module from vendor directory
        
        Args:
            vendor_name: Vendor directory name
            module_path: Dot-separated module path (e.g., 'depth_anything_v2.dpt')
            
        Returns:
            Imported module or raises ImportError
        """
        import sys
        
        vendor_path = self._get_vendor_path(vendor_name)
        if not vendor_path.exists():
            raise ImportError(f"Vendor not found: {vendor_path}")
        
        # Add to path temporarily
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))
        
        # Import the module
        import importlib
        return importlib.import_module(module_path)
    
    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        output_mode: str = "mp4",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a video frame-by-frame
        
        Uses VideoProcessor to extract frames, applies self.process() to each,
        then reassembles to output format.
        
        Args:
            video_path: Input video path
            output_path: Output path (video or directory for sequences)
            output_mode: "mp4" or "sequence"
            progress_callback: Optional (progress, message) callback
            **kwargs: Processor-specific parameters passed to process()
            
        Returns:
            Dict with output info and per-frame results
        """
        # Import here to avoid circular imports
        from core.video_processor import VideoProcessor, OutputMode
        
        self._ensure_initialized()
        
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        processor = VideoProcessor()
        
        # Create a frame processor that wraps self.process()
        # Pass exact_output=True to ensure frames are written to exact paths
        def frame_processor(input_frame: Path, output_frame: Path) -> Dict[str, Any]:
            return self.process(input_frame, output_frame, exact_output=True, **kwargs)
        
        mode = OutputMode.SEQUENCE if output_mode == "sequence" else OutputMode.MP4
        
        result = processor.process_video(
            input_video=video_path,
            output_path=output_path,
            frame_processor=frame_processor,
            output_mode=mode,
            progress_callback=progress_callback,
        )
        
        # Add processor-specific info
        result["method"] = self.__class__.__name__
        result["parameters"] = kwargs
        
        return result
    
    @staticmethod
    def is_video(path: Union[str, Path]) -> bool:
        """Check if path is a video file"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        return Path(path).suffix.lower() in video_extensions
    
    @staticmethod
    def is_image(path: Union[str, Path]) -> bool:
        """Check if path is an image file"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
        return Path(path).suffix.lower() in image_extensions


class SimplePreprocessor(BasePreprocessor):
    """
    Base for preprocessors that don't need model loading (e.g., Canny)
    """
    
    def _initialize(self):
        """No-op for simple preprocessors"""
        pass