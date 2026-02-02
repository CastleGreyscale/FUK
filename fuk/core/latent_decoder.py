# core/latent_decoder.py
"""
Latent Decoder

Decodes safetensors latent files to float32 pixel arrays for true EXR export.
Bypasses the lossy MP4 compression step entirely.

Flow:
    latent.safetensors → VAE decode → float32 pixels → EXR (lossless)
    
Instead of:
    latent → VAE decode → MP4 (lossy!) → PNG → EXR (quality already lost)

Uses musubi-tuner's VAE loading infrastructure for compatibility.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Generator, Callable
import numpy as np
import json

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'


def _log(category: str, message: str, level: str = "info"):
    """Simple logging helper"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    colors = {
        'info': '\033[96m',
        'success': '\033[92m',
        'warning': '\033[93m',
        'error': '\033[91m',
        'end': '\033[0m'
    }
    
    color = colors.get(level, colors['info'])
    print(f"{color}[{timestamp}] [{category}] {message}{colors['end']}", flush=True)


class LatentDecoder:
    """
    Decode latent safetensors files using the Wan/Qwen VAE.
    
    Outputs float32 numpy arrays in [0, 1] range, ready for EXR export.
    
    Requires musubi-tuner to be in the Python path for VAE loading.
    """
    
    def __init__(
        self,
        vae_path: str,
        musubi_path: Optional[Path] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        """
        Initialize the decoder with VAE model.
        
        Args:
            vae_path: Path to VAE checkpoint
            musubi_path: Path to musubi-tuner (added to sys.path if provided)
            device: torch device
            dtype: Model dtype (bfloat16 recommended for Wan VAE)
        """
        self.vae_path = Path(vae_path)
        self.device = device
        self.dtype = dtype
        self.vae = None
        self.vae_type = None
        
        # Add musubi to path if provided
        if musubi_path:
            musubi_path = Path(musubi_path)
            
            # Check for src directory (musubi-tuner structure)
            src_path = musubi_path / "src"
            if src_path.exists():
                musubi_str = str(src_path)
                _log("LATENT", f"Using musubi src path: {musubi_str}")
            else:
                musubi_str = str(musubi_path)
                _log("LATENT", f"Using musubi root path: {musubi_str}")
            
            if musubi_str not in sys.path:
                sys.path.insert(0, musubi_str)
                _log("LATENT", f"Added to sys.path: {musubi_str}")
            
            # Check if musubi_tuner module is accessible
            musubi_tuner_path = Path(musubi_str) / "musubi_tuner"
            if musubi_tuner_path.exists():
                _log("LATENT", f"Found musubi_tuner at: {musubi_tuner_path}")
                wan_vae_path = musubi_tuner_path / "wan" / "modules" / "vae.py"
                if wan_vae_path.exists():
                    _log("LATENT", f"Found wan VAE module: {wan_vae_path}")
            else:
                _log("LATENT", f"musubi_tuner not found at: {musubi_tuner_path}", "warning")
        else:
            _log("LATENT", "No musubi_path provided - custom VAE loading may fail", "warning")
        
        _log("LATENT", f"Decoder initialized with VAE: {vae_path}")
    
    def _ensure_vae_loaded(self):
        """Lazy-load VAE on first use"""
        if self.vae is not None:
            return
        
        import torch
        
        _log("LATENT", "Loading VAE...")
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)
        
        # Handle file path vs directory path
        # models.json might point to the .safetensors file or the directory
        vae_path = Path(self.vae_path)
        if vae_path.is_file():
            # If it's a file, use parent directory for diffusers loading
            vae_dir = vae_path.parent
            vae_file = vae_path
        else:
            vae_dir = vae_path
            vae_file = vae_path / "diffusion_pytorch_model.safetensors"
        
        _log("LATENT", f"VAE directory: {vae_dir}")
        _log("LATENT", f"VAE file: {vae_file}")
        
        # Check if config.json exists (needed for diffusers)
        config_path = vae_dir / "config.json"
        has_config = config_path.exists()
        _log("LATENT", f"config.json exists: {has_config}")
        
        # Try loading as Wan/Qwen 3D causal VAE via musubi-tuner
        # These VAEs are NOT standard diffusers AutoencoderKL - they're custom 3D causal VAEs
        try:
            vae_loaded = False
            
            # Approach 1: Try musubi's WanVAE (for both image and video, they share architecture)
            try:
                from musubi_tuner.wan.modules.vae import WanVAE
                _log("LATENT", "Found WanVAE class in musubi_tuner.wan.modules.vae")
                
                # WanVAE needs config to instantiate
                if has_config:
                    import json
                    with open(config_path) as f:
                        config = json.load(f)
                    
                    self.vae = WanVAE(**config)
                    
                    # Load weights
                    from safetensors.torch import load_file
                    state_dict = load_file(str(vae_file))
                    self.vae.load_state_dict(state_dict)
                    vae_loaded = True
                    _log("LATENT", "Loaded WanVAE with config", "success")
                else:
                    _log("LATENT", "WanVAE requires config.json", "warning")
                    
            except ImportError as e:
                _log("LATENT", f"Could not import WanVAE: {e}", "warning")
                
                # List what IS available in the module
                try:
                    from musubi_tuner.wan.modules import vae as vae_module
                    available = [name for name in dir(vae_module) if not name.startswith('_')]
                    _log("LATENT", f"Available in vae module: {available}")
                except:
                    pass
            except Exception as e:
                _log("LATENT", f"Failed to load WanVAE: {e}", "warning")
            
            # Approach 2: Try qwen image utils (fallback for image VAE)
            if not vae_loaded:
                try:
                    from musubi_tuner.qwen_image import qwen_image_utils
                    _log("LATENT", "Found qwen_image_utils module")
                    
                    # Check for VAE loading function
                    if hasattr(qwen_image_utils, 'load_vae'):
                        self.vae = qwen_image_utils.load_vae(str(vae_file))
                        vae_loaded = True
                        _log("LATENT", "Loaded VAE via qwen_image_utils.load_vae", "success")
                    else:
                        available = [name for name in dir(qwen_image_utils) if 'vae' in name.lower() or 'load' in name.lower()]
                        _log("LATENT", f"VAE-related functions in qwen_image_utils: {available}")
                        
                except ImportError as e:
                    _log("LATENT", f"Could not import qwen_image_utils: {e}", "warning")
            
            # Approach 3: Try wan utils
            if not vae_loaded:
                try:
                    from musubi_tuner.wan import wan_utils
                    _log("LATENT", "Found wan_utils module")
                    
                    if hasattr(wan_utils, 'load_vae'):
                        self.vae = wan_utils.load_vae(str(vae_file))
                        vae_loaded = True
                        _log("LATENT", "Loaded VAE via wan_utils.load_vae", "success")
                    else:
                        available = [name for name in dir(wan_utils) if 'vae' in name.lower() or 'load' in name.lower()]
                        _log("LATENT", f"VAE-related functions in wan_utils: {available}")
                        
                except ImportError as e:
                    _log("LATENT", f"Could not import wan_utils: {e}", "warning")
            
            if vae_loaded:
                self.vae = self.vae.to(device=self.device, dtype=torch_dtype)
                self.vae.eval()
                self.vae_type = "musubi"
                return
            else:
                _log("LATENT", "Could not find a working VAE loader in musubi", "warning")
                
        except Exception as e:
            import traceback
            _log("LATENT", f"Failed to load via musubi: {e}", "warning")
            _log("LATENT", f"Traceback: {traceback.format_exc()}", "warning")
        
        # Try loading as diffusers AutoencoderKL (only works for standard SD-style VAEs)
        # The Qwen/Wan VAE is NOT compatible with diffusers - skip if we detect custom config
        if has_config:
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            # Check for non-diffusers config keys (indicates custom VAE)
            custom_keys = {'attn_scales', 'base_dim', 'dim_mult', 'temperal_downsample', 'z_dim'}
            if custom_keys & set(config.keys()):
                _log("LATENT", "Detected custom 3D causal VAE config (not diffusers-compatible)", "warning")
                _log("LATENT", "This VAE requires musubi-tuner for loading", "warning")
            else:
                # Standard diffusers VAE - try loading
                try:
                    from diffusers import AutoencoderKL
                    
                    _log("LATENT", f"Loading as diffusers AutoencoderKL from {vae_dir}...")
                    self.vae = AutoencoderKL.from_pretrained(
                        str(vae_dir),
                        torch_dtype=torch_dtype
                    )
                    self.vae = self.vae.to(self.device)
                    self.vae.eval()
                    self.vae_type = "diffusers"
                    _log("LATENT", "Loaded diffusers AutoencoderKL", "success")
                    return
                    
                except Exception as e:
                    _log("LATENT", f"Failed to load as diffusers VAE: {e}", "warning")
        else:
            _log("LATENT", f"No config.json in {vae_dir} - cannot determine VAE type", "warning")
        
        # Provide helpful error message
        error_msg = f"""Could not load VAE from {self.vae_path}

The Qwen/Wan VAE is a custom 3D causal VAE that requires musubi-tuner for loading.
It is NOT compatible with standard diffusers AutoencoderKL.

To fix this:
1. Make sure musubi-tuner is properly installed in vendor/musubi-tuner
2. The src/musubi_tuner directory should exist with wan/ and qwen_image/ subdirectories
3. Check that the VAE loading code in musubi_tuner exports a usable VAE class

Current paths:
- VAE directory: {vae_dir}
- VAE file: {vae_file}
- Config exists: {has_config}
"""
        raise RuntimeError(error_msg)
    
    def decode_image_latent(
        self,
        latent_path: Path,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Decode a single image latent to pixel array.
        
        Args:
            latent_path: Path to latent.safetensors
            normalize: Normalize output to [0, 1] range
            
        Returns:
            Float32 numpy array of shape (H, W, C) in [0, 1] range
        """
        import torch
        from safetensors.torch import load_file
        
        self._ensure_vae_loaded()
        
        _log("LATENT", f"Decoding image latent: {latent_path}")
        
        # Load latent tensor
        latent_dict = load_file(str(latent_path))
        
        # Find the latent tensor (key varies by generator)
        latent = self._find_latent_tensor(latent_dict)
        _log("LATENT", f"Latent shape: {latent.shape}, VAE type: {self.vae_type}")
        
        # Move to device with correct dtype
        vae_dtype = getattr(self.vae, 'dtype', torch.bfloat16)
        latent = latent.to(device=self.device, dtype=vae_dtype)
        
        # Decode based on VAE type
        with torch.no_grad():
            if self.vae_type == "musubi":
                # Musubi VAE (Qwen/Wan) expects 5D: (B, C, T, H, W)
                # The latent from Qwen is already 5D with T=1
                if latent.dim() == 4:
                    # (B, C, H, W) -> (B, C, 1, H, W)
                    latent = latent.unsqueeze(2)
                elif latent.dim() == 3:
                    # (C, H, W) -> (1, C, 1, H, W)
                    latent = latent.unsqueeze(0).unsqueeze(2)
                
                _log("LATENT", f"Calling musubi VAE decode with shape: {latent.shape}")
                decoded = self.vae.decode(latent)
                
                # Handle dict return type (musubi VAE returns dict)
                if isinstance(decoded, dict):
                    _log("LATENT", f"Decode returned dict with keys: {list(decoded.keys())}")
                    # Log types of values
                    for k, v in decoded.items():
                        _log("LATENT", f"  Key '{k}': type={type(v).__name__}, shape={getattr(v, 'shape', 'N/A')}")
                    
                    # Try common keys first
                    tensor_found = False
                    for key in ['sample', 'x', 'output', 'decoded', 'video', 'image', 'samples', 'recon', 'reconstruction']:
                        if key in decoded and isinstance(decoded[key], torch.Tensor):
                            decoded = decoded[key]
                            _log("LATENT", f"Extracted tensor from key '{key}'")
                            tensor_found = True
                            break
                    
                    # If no known key, find first tensor
                    if not tensor_found:
                        for k, v in decoded.items():
                            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                                decoded = v
                                _log("LATENT", f"Extracted tensor from key '{k}'")
                                tensor_found = True
                                break
                    
                    if not tensor_found:
                        raise ValueError(f"Could not find tensor in VAE output dict. Keys: {list(decoded.keys())}")
                
                _log("LATENT", f"Decoded shape: {decoded.shape}")
                
                # Output is (B, C, T, H, W) - extract single frame
                if decoded.dim() == 5:
                    decoded = decoded[:, :, 0, :, :]  # (B, C, H, W)
                    
            elif self.vae_type == "wan_causal":
                # Wan VAE expects (B, C, T, H, W)
                if latent.dim() == 4:
                    latent = latent.unsqueeze(2)
                decoded = self.vae.decode(latent)
                if isinstance(decoded, dict):
                    decoded = decoded.get('sample', decoded.get('x', list(decoded.values())[0]))
                if decoded.dim() == 5:
                    decoded = decoded[:, :, 0, :, :]
                    
            elif self.vae_type in ["diffusers", "safetensors"]:
                # Standard diffusers VAE expects (B, C, H, W)
                if latent.dim() == 5:
                    latent = latent[:, :, 0, :, :]
                decoded = self.vae.decode(latent).sample
                
            else:
                # Generic fallback
                _log("LATENT", f"Using generic decode for VAE type: {self.vae_type}", "warning")
                decoded = self.vae.decode(latent)
                if isinstance(decoded, dict):
                    decoded = decoded.get('sample', decoded.get('x', list(decoded.values())[0]))
                if decoded.dim() == 5:
                    decoded = decoded[:, :, 0, :, :]
        
        # Convert to numpy (B, C, H, W) -> (H, W, C)
        _log("LATENT", f"Final decoded shape before permute: {decoded.shape}")
        pixels = decoded[0].permute(1, 2, 0).cpu().float().numpy()
        
        if normalize:
            # VAE typically outputs in [-1, 1], normalize to [0, 1]
            pixels = (pixels + 1.0) / 2.0
            pixels = np.clip(pixels, 0.0, 1.0)
        
        _log("LATENT", f"Decoded to {pixels.shape}, range [{pixels.min():.3f}, {pixels.max():.3f}]", "success")
        
        return pixels.astype(np.float32)
    
    def decode_video_latent(
        self,
        latent_path: Path,
        normalize: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Decode video latent to pixel arrays, yielding frame by frame.
        
        Memory efficient: decodes all at once but yields one frame at a time.
        
        Args:
            latent_path: Path to latent.safetensors (video latent)
            normalize: Normalize output to [0, 1] range
            progress_callback: Optional callback(current, total)
            
        Yields:
            Tuple of (frame_index, frame_array) where frame_array is (H, W, C) float32
        """
        import torch
        from safetensors.torch import load_file
        
        self._ensure_vae_loaded()
        
        _log("LATENT", f"Decoding video latent: {latent_path}")
        
        # Load latent tensor
        latent_dict = load_file(str(latent_path))
        latent = self._find_latent_tensor(latent_dict)
        _log("LATENT", f"Video latent shape: {latent.shape}, VAE type: {self.vae_type}")
        
        # Video latent shape should be (B, C, T, H, W) or (C, T, H, W)
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)  # Add batch dim
        
        if latent.dim() != 5:
            raise ValueError(f"Expected 5D video latent (B,C,T,H,W), got shape {latent.shape}")
        
        # Move to device
        vae_dtype = getattr(self.vae, 'dtype', torch.bfloat16)
        latent = latent.to(device=self.device, dtype=vae_dtype)
        
        # Decode full video
        _log("LATENT", f"Running VAE decode with shape: {latent.shape}")
        with torch.no_grad():
            if self.vae_type in ["wan_causal", "musubi"]:
                # Musubi/Wan 3D CausalVAE handles 5D input directly
                decoded = self.vae.decode(latent)
                
                # Handle dict return type
                if isinstance(decoded, dict):
                    _log("LATENT", f"Decode returned dict with keys: {list(decoded.keys())}")
                    # Log types of values
                    for k, v in decoded.items():
                        _log("LATENT", f"  Key '{k}': type={type(v).__name__}, shape={getattr(v, 'shape', 'N/A')}")
                    
                    # Try common keys first
                    tensor_found = False
                    for key in ['sample', 'x', 'output', 'decoded', 'video', 'image', 'samples', 'recon', 'reconstruction']:
                        if key in decoded and isinstance(decoded[key], torch.Tensor):
                            decoded = decoded[key]
                            _log("LATENT", f"Extracted tensor from key '{key}'")
                            tensor_found = True
                            break
                    
                    # If no known key, find first tensor
                    if not tensor_found:
                        for k, v in decoded.items():
                            if isinstance(v, torch.Tensor) and v.dim() >= 4:
                                decoded = v
                                _log("LATENT", f"Extracted tensor from key '{k}'")
                                tensor_found = True
                                break
                    
                    if not tensor_found:
                        raise ValueError(f"Could not find tensor in VAE output dict. Keys: {list(decoded.keys())}")
                
                _log("LATENT", f"Decoded shape: {decoded.shape}")
            elif self.vae_type in ["diffusers", "safetensors"]:
                # For 2D VAEs, decode frame by frame
                decoded_frames = []
                for t in range(latent.shape[2]):
                    frame_latent = latent[:, :, t, :, :]
                    frame_decoded = self.vae.decode(frame_latent).sample
                    decoded_frames.append(frame_decoded)
                decoded = torch.stack(decoded_frames, dim=2)
            else:
                # Generic - try direct decode
                decoded = self.vae.decode(latent)
                if isinstance(decoded, dict):
                    for key in ['sample', 'x', 'output', 'decoded', 'video']:
                        if key in decoded and isinstance(decoded[key], torch.Tensor):
                            decoded = decoded[key]
                            break
                    else:
                        for k, v in decoded.items():
                            if isinstance(v, torch.Tensor):
                                decoded = v
                                break
        
        # decoded shape: (B, C, T, H, W)
        decoded = decoded[0]  # Remove batch dim -> (C, T, H, W)
        num_frames = decoded.shape[1]
        
        _log("LATENT", f"Decoded {num_frames} frames")
        
        # Yield frames one at a time
        for frame_idx in range(num_frames):
            frame = decoded[:, frame_idx, :, :]  # (C, H, W)
            frame = frame.permute(1, 2, 0).cpu().float().numpy()  # (H, W, C)
            
            if normalize:
                frame = (frame + 1.0) / 2.0
                frame = np.clip(frame, 0.0, 1.0)
            
            if progress_callback:
                progress_callback(frame_idx + 1, num_frames)
            
            yield frame_idx, frame.astype(np.float32)
        
        _log("LATENT", f"Yielded {num_frames} frames", "success")
    
    def decode_video_latent_all(
        self,
        latent_path: Path,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """
        Decode entire video latent at once.
        
        Warning: Uses more memory than the generator version.
        """
        return [frame for _, frame in self.decode_video_latent(latent_path, normalize)]
    
    def _find_latent_tensor(self, latent_dict: dict):
        """Find the latent tensor in a safetensors dict"""
        import torch
        
        # Common keys used by different generators
        for key in ['latent', 'latents', 'x', 'sample', 'z']:
            if key in latent_dict:
                return latent_dict[key]
        
        # Fall back to first tensor
        if latent_dict:
            first_key = list(latent_dict.keys())[0]
            _log("LATENT", f"Using tensor key: {first_key}", "warning")
            return latent_dict[first_key]
        
        raise ValueError("No latent tensor found in safetensors file")
    
    def get_latent_info(self, latent_path: Path) -> Dict[str, Any]:
        """
        Get information about a latent file without full decoding.
        """
        from safetensors import safe_open
        
        latent_path = Path(latent_path)
        info = {
            "path": str(latent_path),
            "file_size": latent_path.stat().st_size,
            "file_size_mb": latent_path.stat().st_size / (1024 * 1024),
        }
        
        with safe_open(str(latent_path), framework="pt") as f:
            keys = list(f.keys())
            info["tensor_keys"] = keys
            
            # Find main latent tensor
            for key in ['latent', 'latents', 'x', 'sample', 'z'] + keys[:1]:
                if key in keys:
                    tensor = f.get_tensor(key)
                    info["shape"] = list(tensor.shape)
                    info["dtype"] = str(tensor.dtype)
                    info["latent_key"] = key
                    break
        
        # Estimate output dimensions (assuming 8x spatial compression)
        if "shape" in info:
            shape = info["shape"]
            spatial_ratio = 8
            temporal_ratio = 4  # Wan uses 4x temporal compression
            
            if len(shape) == 4:  # (B, C, H, W) - Image
                info["type"] = "image"
                info["estimated_output"] = {
                    "height": shape[2] * spatial_ratio,
                    "width": shape[3] * spatial_ratio,
                    "channels": 3,
                }
            elif len(shape) == 5:  # (B, C, T, H, W) - Video
                info["type"] = "video"
                info["estimated_output"] = {
                    "frames": (shape[2] - 1) * temporal_ratio + 1,
                    "height": shape[3] * spatial_ratio,
                    "width": shape[4] * spatial_ratio,
                    "channels": 3,
                }
        
        return info
    
    def unload(self):
        """Unload VAE to free VRAM"""
        if self.vae is not None:
            import torch
            del self.vae
            self.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log("LATENT", "VAE unloaded", "success")


def create_decoder_from_config(
    models_config_path: Path,
    musubi_path: Optional[Path] = None,
    model_type: str = "wan",
    device: str = "cuda",
) -> LatentDecoder:
    """
    Factory function to create decoder with VAE path from models.json
    
    Args:
        models_config_path: Path to models.json
        musubi_path: Path to musubi-tuner vendor directory
        model_type: "wan" or "qwen" to select VAE
        device: torch device
        
    Returns:
        Configured LatentDecoder
    """
    with open(models_config_path) as f:
        config = json.load(f)
    
    # Get VAE path based on model type
    if model_type == "wan":
        # Try various Wan model configs
        for model_key in ["wan_i2v_14b", "wan_t2v_14b", "wan_i2v_480p", "wan_t2v_480p"]:
            if model_key in config.get("models", {}):
                model_config = config["models"][model_key]
                break
        else:
            raise ValueError("No Wan model config found")
    else:
        model_config = config["models"].get("qwen_image")
        if not model_config:
            raise ValueError("No Qwen model config found")
    
    vae_path = model_config.get("vae")
    if not vae_path:
        raise ValueError(f"No VAE path in model config")
    
    return LatentDecoder(
        vae_path=vae_path,
        musubi_path=musubi_path,
        device=device,
    )
