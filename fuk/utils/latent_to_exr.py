# utils/latent_to_exr.py
"""
Decode saved latent tensors directly to EXR frames

This bypasses MP4 compression entirely, preserving maximum quality
from the generation process.

WORKFLOW:
  Generation → Latent (safetensors) → VAE decode → EXR frames
  
vs the lossy path:
  Generation → Latent → VAE decode → MP4 encode → Extract → EXR (pointless)
"""

from pathlib import Path
from typing import Optional, List, Tuple
import subprocess
import json


class LatentToEXRDecoder:
    """Decode latent tensors directly to EXR frames without MP4 compression"""
    
    def __init__(self, musubi_vendor_path: Path, config_path: Path):
        self.musubi_path = Path(musubi_vendor_path)
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Path):
        with open(config_path) as f:
            return json.load(f)
    
    def decode_latent_to_frames(self,
                                latent_path: Path,
                                output_dir: Path,
                                task: str,
                                frame_format: str = "png",
                                vae_tiling: bool = True,
                                fp8: bool = True) -> List[Path]:
        """
        Decode latent tensor to individual frames
        
        Args:
            latent_path: Path to .safetensors latent file
            output_dir: Directory to save decoded frames
            task: Wan task type (e.g., "i2v-14B")
            frame_format: Output format ("png" or "exr" if supported)
            vae_tiling: Use VAE tiling for memory efficiency
            fp8: Use fp8 for VAE
            
        Returns:
            List of decoded frame paths
        """
        
        latent_path = Path(latent_path)
        output_dir = Path(output_dir)
        
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get VAE config for this task
        task_config = self.config["wan_models"][task]
        vae_path = task_config["vae"]
        
        print(f"\n=== Decoding Latent to Frames ===")
        print(f"Latent: {latent_path}")
        print(f"Output: {output_dir}")
        print(f"VAE: {vae_path}")
        print(f"Format: {frame_format.upper()}\n")
        
        # Build decode command
        cmd = [
            "python",
            str(self.musubi_path / "wan_decode_latent.py"),  # May need to create this
            "--latent_path", str(latent_path),
            "--vae", vae_path,
            "--output_dir", str(output_dir),
            "--output_format", frame_format,
        ]
        
        if vae_tiling:
            cmd.append("--vae_tiling")
        
        if fp8:
            cmd.append("--fp8")
        
        print(f"Command: {' '.join(cmd)}")
        
        # Note: This assumes musubi-tuner has a latent decoder script
        # If not, we need to create one using their VAE directly
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # If the decode script doesn't exist, use fallback
            print("Direct decode script not found, using Python VAE decode...")
            return self._decode_with_python_vae(
                latent_path, output_dir, vae_path, frame_format, vae_tiling, fp8
            )
        
        # Collect decoded frames
        if frame_format == "png":
            frames = sorted(output_dir.glob("frame_*.png"))
        elif frame_format == "exr":
            frames = sorted(output_dir.glob("frame_*.exr"))
        else:
            frames = sorted(output_dir.glob(f"frame_*.{frame_format}"))
        
        print(f"✓ Decoded {len(frames)} frames")
        return frames
    
    def _decode_with_python_vae(self,
                                latent_path: Path,
                                output_dir: Path,
                                vae_path: str,
                                frame_format: str,
                                vae_tiling: bool,
                                fp8: bool) -> List[Path]:
        """
        Fallback: Decode using Python directly with VAE model
        
        This is the implementation if musubi doesn't have a standalone decoder
        """
        import torch
        from safetensors.torch import load_file
        from diffusers import AutoencoderKL
        from PIL import Image
        import numpy as np
        
        print("Loading latent tensor...")
        latent_dict = load_file(str(latent_path))
        
        # Extract latent tensor (key may vary)
        if "latent" in latent_dict:
            latent = latent_dict["latent"]
        elif "samples" in latent_dict:
            latent = latent_dict["samples"]
        else:
            # Take the first tensor
            latent = next(iter(latent_dict.values()))
        
        print(f"Latent shape: {latent.shape}")
        
        # Load VAE
        print("Loading VAE...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=torch.float16 if fp8 else torch.float32
        ).to(device)
        
        if vae_tiling:
            vae.enable_tiling()
        
        # Decode frame by frame
        latent = latent.to(device)
        num_frames = latent.shape[2]  # [B, C, F, H, W]
        
        print(f"Decoding {num_frames} frames...")
        
        frames = []
        for i in range(num_frames):
            # Extract single frame latent
            frame_latent = latent[:, :, i:i+1, :, :]
            
            # Decode
            with torch.no_grad():
                decoded = vae.decode(frame_latent).sample
            
            # Convert to numpy [H, W, C]
            frame_np = decoded[0].permute(1, 2, 0).cpu().float().numpy()
            
            # Denormalize from [-1, 1] to [0, 1]
            frame_np = (frame_np + 1.0) / 2.0
            frame_np = np.clip(frame_np, 0, 1)
            
            # Save based on format
            frame_path = output_dir / f"frame_{i+1:04d}.{frame_format}"
            
            if frame_format == "png":
                # Convert to 8-bit PNG
                frame_8bit = (frame_np * 255).astype(np.uint8)
                Image.fromarray(frame_8bit).save(frame_path)
            
            elif frame_format == "exr":
                # Save as 32-bit EXR - this is the whole point!
                self._save_exr(frame_np, frame_path)
            
            frames.append(frame_path)
            
            if (i + 1) % 10 == 0:
                print(f"  Decoded {i+1}/{num_frames}")
        
        print(f"✓ Decoded {len(frames)} frames to {frame_format.upper()}")
        
        return frames
    
    def _save_exr(self, frame_np: np.ndarray, output_path: Path):
        """Save numpy array as 32-bit EXR"""
        import OpenEXR
        import Imath
        
        height, width = frame_np.shape[:2]
        
        # Prepare EXR header
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        
        # Write EXR (frame_np is already float32 in [0, 1])
        exr = OpenEXR.OutputFile(str(output_path), header)
        
        channels = {
            'R': frame_np[:, :, 0].astype(np.float32).tobytes(),
            'G': frame_np[:, :, 1].astype(np.float32).tobytes(),
            'B': frame_np[:, :, 2].astype(np.float32).tobytes()
        }
        
        exr.writePixels(channels)
        exr.close()
    
    def decode_latent_to_exr_sequence(self,
                                      latent_path: Path,
                                      output_dir: Path,
                                      task: str,
                                      linear: bool = True) -> Path:
        """
        Convenience method: Decode latent directly to EXR sequence
        
        This is the PROPER workflow for professional output:
          Generation → Latent → EXR (no MP4 compression)
        
        Args:
            latent_path: Saved latent tensor
            output_dir: Where to save EXR frames
            task: Wan task type
            linear: Save in linear color space (True for compositing)
            
        Returns:
            Path to EXR sequence directory
        """
        
        # First decode to PNG (intermediate)
        temp_dir = output_dir / "temp_decode"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        png_frames = self.decode_latent_to_frames(
            latent_path=latent_path,
            output_dir=temp_dir,
            task=task,
            frame_format="png"
        )
        
        # Convert PNG to EXR with proper color space
        from utils.format_convert import FormatConverter
        
        exr_dir = output_dir / "exr_frames"
        exr_dir.mkdir(exist_ok=True, parents=True)
        
        exr_frames = []
        for i, png_path in enumerate(png_frames, start=1):
            exr_path = exr_dir / f"frame_{i:04d}.exr"
            FormatConverter.png_to_exr_32bit(png_path, exr_path, linear=linear)
            exr_frames.append(exr_path)
            
            if i % 10 == 0:
                print(f"  Converted {i}/{len(png_frames)} to EXR")
        
        # Cleanup temp PNGs
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"\n✓ Created {len(exr_frames)} EXR frames (direct from latent)")
        print(f"  Location: {exr_dir}")
        
        return exr_dir


def decode_generation_to_exr(gen_dir: Path,
                            config_path: Path,
                            musubi_path: Path,
                            task: str) -> Path:
    """
    High-level function: Find latent in generation dir and decode to EXR
    
    Usage:
        exr_dir = decode_generation_to_exr(
            gen_dir=Path("outputs/video/2025-12-26/i2v_001"),
            config_path=Path("config/models.json"),
            musubi_path=Path("vendor/musubi-tuner"),
            task="i2v-14B"
        )
    """
    
    gen_dir = Path(gen_dir)
    latent_path = gen_dir / "latent.safetensors"
    
    if not latent_path.exists():
        raise FileNotFoundError(
            f"No latent found in {gen_dir}. "
            "Make sure you generated with output_type='both'"
        )
    
    decoder = LatentToEXRDecoder(musubi_path, config_path)
    
    exr_dir = decoder.decode_latent_to_exr_sequence(
        latent_path=latent_path,
        output_dir=gen_dir,
        task=task,
        linear=True
    )
    
    return exr_dir