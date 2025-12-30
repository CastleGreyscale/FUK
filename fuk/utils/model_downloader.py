# utils/model_downloader.py
"""
Automatic model downloader for FUK preprocessor models
Handles downloading missing checkpoints from HuggingFace
"""

from pathlib import Path
import json
import urllib.request
import sys
from typing import Optional, Dict, Any


class ModelDownloader:
    """Auto-download missing preprocessor model checkpoints"""
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load models.json config"""
        with open(self.config_path) as f:
            return json.load(f)
    
    def get_preprocessor_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model info from config"""
        return self.config.get("preprocessor_models", {}).get(model_name)
    
    def check_and_download(self, model_name: str) -> Optional[Path]:
        """
        Check if model exists, download if missing
        
        Args:
            model_name: Model identifier (e.g., "depth_anything_v2")
            
        Returns:
            Path to checkpoint if successful, None if auto-download not available
        """
        
        model_info = self.get_preprocessor_model_info(model_name)
        
        if not model_info:
            print(f"⚠  Model '{model_name}' not found in config")
            return None
        
        # Check if this model has auto-download info
        if "checkpoint" not in model_info or "url" not in model_info:
            # Model downloads automatically (like MiDaS via torch.hub)
            return None
        
        checkpoint_path = Path(model_info["checkpoint"])
        
        # Already exists
        if checkpoint_path.exists():
            print(f"✓ Found checkpoint: {checkpoint_path}")
            return checkpoint_path
        
        # Download
        print(f"\n{'='*60}")
        print(f"Downloading {model_name}")
        print(f"{'='*60}")
        print(f"URL: {model_info['url']}")
        print(f"Size: ~{model_info.get('size_mb', '?')} MB")
        print(f"Destination: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        try:
            # Create parent directory
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            self._download_with_progress(
                model_info["url"],
                checkpoint_path,
                model_info.get("size_mb", 0)
            )
            
            print(f"\n✓ Download complete: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            # Clean up partial download
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return None
    
    def _download_with_progress(self, url: str, output_path: Path, expected_size_mb: int):
        """Download file with progress bar"""
        
        def progress_hook(block_num, block_size, total_size):
            """Show download progress"""
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                
                # Progress bar
                bar_length = 40
                filled = int(bar_length * downloaded / total_size)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(f"\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
            else:
                # No total size, just show downloaded
                downloaded_mb = (block_num * block_size) / (1024 * 1024)
                print(f"\rDownloaded: {downloaded_mb:.1f} MB", end='', flush=True)
        
        urllib.request.urlretrieve(url, str(output_path), reporthook=progress_hook)
        print()  # New line after progress


# Convenience function
def ensure_model_downloaded(model_name: str, config_path: Path) -> Optional[Path]:
    """
    Ensure a preprocessor model is downloaded and return its path
    
    Usage:
        checkpoint_path = ensure_model_downloaded("depth_anything_v2", config_path)
        if checkpoint_path:
            # Use the checkpoint
            model.load_state_dict(torch.load(checkpoint_path))
    """
    downloader = ModelDownloader(config_path)
    return downloader.check_and_download(model_name)


if __name__ == "__main__":
    # Test downloader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_downloader.py <model_name>")
        print("\nAvailable models:")
        print("  - depth_anything_v2")
        sys.exit(1)
    
    model_name = sys.argv[1]
    config_path = Path(__file__).parent.parent / "config" / "models.json"
    
    result = ensure_model_downloaded(model_name, config_path)
    
    if result:
        print(f"✓ Model ready: {result}")
    else:
        print(f"✗ Model not available for auto-download")
        sys.exit(1)