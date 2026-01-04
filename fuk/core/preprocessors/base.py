# core/preprocessors/base.py
"""
Base class for all FUK preprocessors

Provides shared utilities:
- Unique filename generation (prevents caching issues)
- Common output patterns
- Device detection
"""

from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import json
import hashlib
import time
import torch


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors
    
    Subclasses must implement:
    - process(): Main processing method
    - _initialize(): Lazy model loading
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
    
    def _make_unique_path(self, base_path: Path, params: dict) -> Path:
        """
        Generate unique filename to prevent caching issues
        
        Appends a hash of parameters + timestamp to the filename.
        This ensures different parameter combinations produce different files
        and browser/UI caching doesn't serve stale results.
        
        Args:
            base_path: Original output path
            params: Dictionary of parameters to hash
            
        Returns:
            New path with unique suffix
        """
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


class SimplePreprocessor(BasePreprocessor):
    """
    Base for preprocessors that don't need model loading (e.g., Canny)
    """
    
    def _initialize(self):
        """No-op for simple preprocessors"""
        pass
