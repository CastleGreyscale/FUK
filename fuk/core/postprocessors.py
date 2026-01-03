# core/postprocessors.py
"""
Post-Processing Tools for FUK Pipeline

Provides:
- Image upscaling (Real-ESRGAN, SwinIR)
- Video frame interpolation (RIFE)
- Video upscaling (batch processing)
"""
import sys
import os

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import torch
import subprocess
import shutil
import tempfile
import time
import json

# ============================================================================
# Upscaler Models
# ============================================================================

class UpscaleModel(str, Enum):
    """Available upscaling models"""
    REALESRGAN_X2 = "realesrgan_x2"
    REALESRGAN_X4 = "realesrgan_x4"
    REALESRGAN_ANIME = "realesrgan_anime"
    SWINIR = "swinir"
    LANCZOS = "lanczos"  # Fast fallback


class InterpolationModel(str, Enum):
    """Available frame interpolation models"""
    RIFE = "rife"
    RIFE_V4 = "rife_v4"


# ============================================================================
# Real-ESRGAN Upscaler
# ============================================================================

class RealESRGANUpscaler:
    """
    Real-ESRGAN image upscaler
    
    Uses the realesrgan-ncnn-vulkan binary for GPU-accelerated upscaling
    Falls back to PIL/Lanczos if binary not available
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path.home() / ".cache" / "realesrgan"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for ncnn binary
        self.ncnn_binary = self._find_ncnn_binary()
        self.torch_model = None
        
        print(f"[Upscaler] Initialized")
        print(f"[Upscaler] NCNN binary: {self.ncnn_binary or 'Not found, will use torch/fallback'}")
    
    def _find_ncnn_binary(self) -> Optional[Path]:
        """Find realesrgan-ncnn-vulkan binary"""
        # Get the directory where this script lives
        script_dir = Path(__file__).resolve().parent  # fuk/fuk/core
        fuk_root = script_dir.parent  # fuk/fuk
        project_root = fuk_root.parent  # fuk
        
        print(f"[Upscaler] Looking for NCNN binary...")
        print(f"[Upscaler]   Script dir: {script_dir}")
        print(f"[Upscaler]   FUK root: {fuk_root}")
        
        # Check vendors directory (try both singular and plural)
        vendors_paths = [
            fuk_root / "vendor" / "realesrgan-ncnn" / "realesrgan-ncnn-vulkan",
            fuk_root / "vendors" / "realesrgan-ncnn" / "realesrgan-ncnn-vulkan",
            project_root / "vendor" / "realesrgan-ncnn" / "realesrgan-ncnn-vulkan",
            project_root / "vendors" / "realesrgan-ncnn" / "realesrgan-ncnn-vulkan",
        ]
        
        for path in vendors_paths:
            print(f"[Upscaler]   Checking: {path}")
            if path.exists():
                print(f"[Upscaler] Found NCNN in vendors: {path}")
                return path
        
        # Check common system locations
        candidates = [
            Path("/usr/local/bin/realesrgan-ncnn-vulkan"),
            Path("/usr/bin/realesrgan-ncnn-vulkan"),
            Path.home() / "bin" / "realesrgan-ncnn-vulkan",
            Path.home() / ".local" / "bin" / "realesrgan-ncnn-vulkan",
        ]
        
        for path in candidates:
            if path.exists():
                print(f"[Upscaler] Found NCNN at: {path}")
                return path
        
        # Check PATH
        result = shutil.which("realesrgan-ncnn-vulkan")
        if result:
            print(f"[Upscaler] Found NCNN in PATH: {result}")
            return Path(result)
        
        print(f"[Upscaler] NCNN binary not found")
        return None
    
    def _load_torch_model(self, scale: int = 4):
        """Load Real-ESRGAN via torch (fallback)"""
        if self.torch_model is not None:
            return
        
        try:
            # Try the standard realesrgan package first
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                           num_block=23, num_grow_ch=32, scale=scale)
            
            model_path = self.models_dir / f"RealESRGAN_x{scale}plus.pth"
            
            if not model_path.exists():
                print(f"[Upscaler] Model not found at {model_path}")
                print(f"[Upscaler] Download from: https://github.com/xinntao/Real-ESRGAN/releases")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.torch_model = RealESRGANer(
                scale=scale,
                model_path=str(model_path),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if torch.cuda.is_available() else False
            )
            print(f"[Upscaler] Loaded torch Real-ESRGAN model")
            
        except ImportError as e:
            print(f"[Upscaler] Could not load Real-ESRGAN package: {e}")
            print(f"[Upscaler] This is usually a torchvision compatibility issue.")
            print(f"[Upscaler] Solutions:")
            print(f"[Upscaler]   1. Install NCNN binary: realesrgan-ncnn-vulkan")
            print(f"[Upscaler]   2. Fix packages: pip install basicsr==1.4.2 realesrgan==0.3.0")
            print(f"[Upscaler]   3. Use 'lanczos' model (no AI, but works)")
            self.torch_model = None
        except Exception as e:
            print(f"[Upscaler] Error loading torch model: {e}")
            self.torch_model = None
    
    def upscale_ncnn(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 4,
        model_name: str = "realesrgan-x4plus",
        denoise: float = 0.5,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Upscale using ncnn binary"""
        
        if not self.ncnn_binary:
            raise RuntimeError("NCNN binary not found")
        
        if progress_callback:
            progress_callback(0.1, "Starting NCNN upscaler")
        
        # Map scale to model
        model_map = {
            2: "realesrgan-x4plus",  # Will resize down after
            4: "realesrgan-x4plus",
            8: "realesrgan-x4plus",  # Run twice
        }
        
        # Find models directory (alongside the binary in vendors, or default location)
        models_dir = self.ncnn_binary.parent / "models"
        if not models_dir.exists():
            models_dir = self.ncnn_binary.parent  # Some setups have models next to binary
        
        cmd = [
            str(self.ncnn_binary),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", model_map.get(scale, "realesrgan-x4plus"),
            "-s", str(min(scale, 4)),  # NCNN only supports up to 4x
        ]
        
        # Add model path if using vendors directory
        if models_dir.exists() and (models_dir / "realesrgan-x4plus.bin").exists():
            cmd.extend(["-m", str(models_dir)])
        
        if progress_callback:
            progress_callback(0.2, "Running Real-ESRGAN")
        
        print(f"[Upscaler] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"NCNN upscale failed: {result.stderr}")
        
        # If scale > 4, run again
        if scale == 8:
            if progress_callback:
                progress_callback(0.6, "Running second pass for 8x")
            
            temp_path = output_path.with_suffix('.temp.png')
            shutil.move(output_path, temp_path)
            
            cmd[2] = str(temp_path)  # Input is now the 4x result
            result = subprocess.run(cmd, capture_output=True, text=True)
            temp_path.unlink()
            
            if result.returncode != 0:
                raise RuntimeError(f"NCNN upscale (2nd pass) failed: {result.stderr}")
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return {"output_path": str(output_path), "scale": scale, "method": "ncnn"}
    
    def upscale_torch(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 4,
        denoise: float = 0.5,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Upscale using torch model"""
        
        self._load_torch_model(scale)
        
        if self.torch_model is None:
            raise RuntimeError("Could not load torch model")
        
        if progress_callback:
            progress_callback(0.1, "Loading image")
        
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        
        if progress_callback:
            progress_callback(0.3, "Upscaling with Real-ESRGAN")
        
        output, _ = self.torch_model.enhance(img, outscale=scale)
        
        if progress_callback:
            progress_callback(0.9, "Saving result")
        
        cv2.imwrite(str(output_path), output)
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return {"output_path": str(output_path), "scale": scale, "method": "torch"}
    
    def upscale_lanczos(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 4,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Fast Lanczos upscale fallback"""
        
        if progress_callback:
            progress_callback(0.2, "Loading image")
        
        img = Image.open(input_path)
        new_size = (img.width * scale, img.height * scale)
        
        if progress_callback:
            progress_callback(0.5, f"Resizing to {new_size[0]}x{new_size[1]}")
        
        upscaled = img.resize(new_size, Image.LANCZOS)
        
        if progress_callback:
            progress_callback(0.9, "Saving result")
        
        upscaled.save(output_path, quality=95)
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return {"output_path": str(output_path), "scale": scale, "method": "lanczos"}
    
    def upscale(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 4,
        model: str = "realesrgan",
        denoise: float = 0.5,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Main upscale method - automatically selects best available backend
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[Upscaler] Upscaling {input_path.name} by {scale}x using {model}")
        
        # Route to appropriate backend
        if model == "lanczos":
            return self.upscale_lanczos(input_path, output_path, scale, progress_callback)
        
        # Try NCNN first (fastest, best quality)
        if self.ncnn_binary:
            try:
                return self.upscale_ncnn(input_path, output_path, scale, 
                                         denoise=denoise, progress_callback=progress_callback)
            except Exception as e:
                print(f"[Upscaler] NCNN failed: {e}")
        
        # Try torch model
        try:
            return self.upscale_torch(input_path, output_path, scale,
                                      denoise=denoise, progress_callback=progress_callback)
        except Exception as e:
            print(f"[Upscaler] Torch Real-ESRGAN not available: {e}")
        
        # Final fallback to Lanczos
        print(f"[Upscaler] Using Lanczos fallback (install realesrgan-ncnn-vulkan for AI upscaling)")
        return self.upscale_lanczos(input_path, output_path, scale, progress_callback)


# ============================================================================
# RIFE Frame Interpolator
# ============================================================================

class RIFEInterpolator:
    """
    RIFE (Real-Time Intermediate Flow Estimation) frame interpolator
    
    Uses rife-ncnn-vulkan binary for GPU-accelerated interpolation
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path.home() / ".cache" / "rife"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for ncnn binary
        self.ncnn_binary = self._find_ncnn_binary()
        
        print(f"[Interpolator] Initialized")
        print(f"[Interpolator] NCNN binary: {self.ncnn_binary or 'Not found'}")
    
    def _find_ncnn_binary(self) -> Optional[Path]:
        """Find rife-ncnn-vulkan binary"""
        # Get the directory where this script lives
        script_dir = Path(__file__).resolve().parent  # fuk/fuk/core
        fuk_root = script_dir.parent  # fuk/fuk
        project_root = fuk_root.parent  # fuk
        
        print(f"[Interpolator] Looking for RIFE NCNN binary...")
        
        # Check vendors directory (try both singular and plural)
        vendors_paths = [
            fuk_root / "vendor" / "rife-ncnn" / "rife-ncnn-vulkan",
            fuk_root / "vendors" / "rife-ncnn" / "rife-ncnn-vulkan",
            project_root / "vendor" / "rife-ncnn" / "rife-ncnn-vulkan",
            project_root / "vendors" / "rife-ncnn" / "rife-ncnn-vulkan",
        ]
        
        for path in vendors_paths:
            print(f"[Interpolator]   Checking: {path}")
            if path.exists():
                print(f"[Interpolator] Found RIFE NCNN in vendors: {path}")
                return path
        
        # Check common system locations
        candidates = [
            Path("/usr/local/bin/rife-ncnn-vulkan"),
            Path("/usr/bin/rife-ncnn-vulkan"),
            Path.home() / "bin" / "rife-ncnn-vulkan",
            Path.home() / ".local" / "bin" / "rife-ncnn-vulkan",
        ]
        
        for path in candidates:
            if path.exists():
                print(f"[Interpolator] Found RIFE NCNN at: {path}")
                return path
        
        result = shutil.which("rife-ncnn-vulkan")
        if result:
            print(f"[Interpolator] Found RIFE NCNN in PATH: {result}")
            return Path(result)
        
        print(f"[Interpolator] RIFE NCNN binary not found")
        return None
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> List[Path]:
        """Extract frames from video using ffmpeg"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback(0.1, "Extracting frames from video")
        
        # Get video info
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,r_frame_rate",
            "-of", "json",
            str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        # Extract frames - RIFE expects just numbered files: 00000.png, 00001.png, etc.
        # NOT frame_00000.png
        frame_pattern = output_dir / "%08d.png"
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vsync", "0",
            "-start_number", "0",
            str(frame_pattern)
        ]
        
        print(f"[Interpolator] Extracting frames: {' '.join(extract_cmd)}")
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[Interpolator] FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
        
        frames = sorted(output_dir.glob("*.png"))
        print(f"[Interpolator] Extracted {len(frames)} frames")
        if frames:
            print(f"[Interpolator] First: {frames[0].name}, Last: {frames[-1].name}")
        
        return frames
    
    def interpolate_frames_ncnn(
        self,
        input_dir: Path,
        output_dir: Path,
        rife_n: int = 2,
        model: str = "rife-v4.6",
        progress_callback: Optional[Callable] = None
    ) -> List[Path]:
        """
        Interpolate frames using RIFE NCNN binary.
        
        Args:
            rife_n: Multiplier exponent (number of doubling passes)
                    rife_n=1 = 2x frames
                    rife_n=2 = 4x frames  
                    rife_n=3 = 8x frames
        
        Note: rife-ncnn-vulkan -n parameter = TARGET TOTAL FRAME COUNT
              NOT the number of passes! We must calculate the target.
        """
        
        if not self.ncnn_binary:
            raise RuntimeError("RIFE NCNN binary not found")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        multiplier = 2 ** rife_n  # rife_n=2 = 4x, etc.
        if progress_callback:
            progress_callback(0.3, f"Interpolating frames ({multiplier}x)")
        
        # Debug: show input frames
        input_frames = sorted(input_dir.glob("*.png"))
        num_input_frames = len(input_frames)
        print(f"[Interpolator] Input dir: {input_dir}")
        print(f"[Interpolator] Input frames: {num_input_frames}")
        if input_frames:
            print(f"[Interpolator] First input: {input_frames[0].name}, Last: {input_frames[-1].name}")
        
        if num_input_frames < 2:
            raise ValueError(f"Need at least 2 frames to interpolate, got {num_input_frames}")
        
        # Calculate target frame count for RIFE
        # RIFE -n parameter = TOTAL output frames (not passes!)
        # For N input frames with multiplier M: output = (N-1) * M + 1
        # This creates M-1 intermediate frames between each pair
        target_frame_count = (num_input_frames - 1) * multiplier + 1
        
        print(f"[Interpolator] Multiplier: {multiplier}x")
        print(f"[Interpolator] Target frame count: {target_frame_count}")
        
        # Find models directory (alongside the binary in vendors)
        models_dir = self.ncnn_binary.parent / model
        if not models_dir.exists():
            # Try in models subdirectory
            models_dir = self.ncnn_binary.parent / "models" / model
        
        # Build command
        # rife-ncnn-vulkan -n = target total frame count (NOT doubling passes!)
        cmd = [
            str(self.ncnn_binary),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-n", str(target_frame_count),
        ]
        
        # Add model path if found
        if models_dir.exists():
            cmd.extend(["-m", str(models_dir)])
            print(f"[Interpolator] Using model: {models_dir}")
        else:
            print(f"[Interpolator] Model dir not found: {models_dir}")
            # List available models
            available = list(self.ncnn_binary.parent.glob("rife*"))
            print(f"[Interpolator] Available models: {[m.name for m in available]}")
        
        print(f"[Interpolator] RIFE -n {target_frame_count} (target frames from {num_input_frames} inputs)")
        
        print(f"[Interpolator] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"[Interpolator] RIFE stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
        print(f"[Interpolator] RIFE stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
        print(f"[Interpolator] RIFE return code: {result.returncode}")
        
        if result.returncode != 0:
            raise RuntimeError(f"RIFE interpolation failed: {result.stderr}")
        
        # RIFE outputs frames with various naming conventions
        # Check for any image files
        all_pngs = sorted(output_dir.glob("*.png"))
        all_jpgs = sorted(output_dir.glob("*.jpg"))
        all_images = all_pngs + all_jpgs
        
        print(f"[Interpolator] Output dir: {output_dir}")
        print(f"[Interpolator] Found {len(all_pngs)} PNG, {len(all_jpgs)} JPG files")
        
        if not all_images:
            # List what's actually in the directory
            all_files = list(output_dir.iterdir())
            print(f"[Interpolator] Output dir contents ({len(all_files)} items): {[f.name for f in all_files[:20]]}")
            
            # Maybe RIFE wrote to a subdirectory?
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    sub_pngs = list(subdir.glob("*.png"))
                    print(f"[Interpolator] Subdir {subdir.name}: {len(sub_pngs)} PNGs")
            
            raise RuntimeError("RIFE produced no output frames")
        
        # Sort by name (handles both numeric and named files)
        all_images = sorted(all_images, key=lambda p: p.stem)
        
        # Rename to sequential frame_XXXXX.png format for ffmpeg
        print(f"[Interpolator] Renaming {len(all_images)} frames to frame_XXXXX.png format...")
        renamed_frames = []
        for idx, old_path in enumerate(all_images):
            new_path = output_dir / f"frame_{idx:05d}.png"
            if old_path != new_path:
                # If it's a jpg, convert to png
                if old_path.suffix.lower() == '.jpg':
                    img = cv2.imread(str(old_path))
                    cv2.imwrite(str(new_path), img)
                    old_path.unlink()
                else:
                    old_path.rename(new_path)
            renamed_frames.append(new_path)
        
        print(f"[Interpolator] Renamed {len(renamed_frames)} frames (first: {renamed_frames[0].name})")
        return renamed_frames
    
    def interpolate_frames_opencv(
        self,
        input_dir: Path,
        output_dir: Path,
        multiplier: int = 2,
        progress_callback: Optional[Callable] = None
    ) -> List[Path]:
        """Fallback: Simple frame blending interpolation using OpenCV"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frames = sorted(input_dir.glob("*.png"))
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to interpolate")
        
        if progress_callback:
            progress_callback(0.3, "Interpolating frames (OpenCV blend)")
        
        output_frames = []
        frame_idx = 0
        
        for i in range(len(frames) - 1):
            # Copy original frame
            frame1 = cv2.imread(str(frames[i]))
            frame2 = cv2.imread(str(frames[i + 1]))
            
            # Save first frame
            out_path = output_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(out_path), frame1)
            output_frames.append(out_path)
            frame_idx += 1
            
            # Generate intermediate frames
            for j in range(1, multiplier):
                alpha = j / multiplier
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                out_path = output_dir / f"frame_{frame_idx:05d}.png"
                cv2.imwrite(str(out_path), blended)
                output_frames.append(out_path)
                frame_idx += 1
            
            if progress_callback:
                progress_callback(0.3 + 0.5 * (i / len(frames)), 
                                f"Blending frame {i+1}/{len(frames)-1}")
        
        # Save last frame
        out_path = output_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), frame2)
        output_frames.append(out_path)
        
        return output_frames
    
    def frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        input_fps: float,
        output_fps: int = 24,
        progress_callback: Optional[Callable] = None
    ) -> Path:
        """
        Combine frames back into video.
        
        Args:
            frames_dir: Directory containing numbered png files
            output_path: Where to save the video
            input_fps: The effective framerate of the input frames (preserves duration)
            output_fps: The target output framerate
        """
        
        if progress_callback:
            progress_callback(0.9, "Encoding output video")
        
        # Find all png frames
        frame_files = sorted(frames_dir.glob("*.png"))
        num_frames = len(frame_files)
        
        if num_frames == 0:
            raise RuntimeError("No frames to encode")
        
        # Detect the naming pattern from first file
        first_name = frame_files[0].stem
        if first_name.startswith("frame_"):
            frame_pattern = frames_dir / "frame_%05d.png"
        else:
            # Numeric naming like 00000000.png - detect number of digits
            num_digits = len(first_name)
            frame_pattern = frames_dir / f"%0{num_digits}d.png"
        
        expected_duration = num_frames / input_fps
        print(f"[Interpolator] Encoding {num_frames} frames")
        print(f"[Interpolator]   Pattern: {frame_pattern.name}")
        print(f"[Interpolator]   Input: {input_fps}fps, Output: {output_fps}fps")
        print(f"[Interpolator]   Expected duration: {expected_duration:.2f}s")
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_fps),
            "-i", str(frame_pattern),
            "-r", str(output_fps),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        
        print(f"[Interpolator] FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[Interpolator] FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"Video encoding failed: {result.stderr}")
        
        print(f"[Interpolator] Output video: {output_path}")
        return output_path
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video duration and frame count"""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}
        
        try:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            
            # Parse frame rate (e.g., "16/1" or "24000/1001")
            fps_str = stream.get("r_frame_rate", "16/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            # Get frame count or estimate from duration
            nb_frames = stream.get("nb_frames")
            if nb_frames:
                frame_count = int(nb_frames)
            else:
                duration = float(stream.get("duration", 0))
                frame_count = int(duration * fps)
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": frame_count / fps if fps > 0 else 0
            }
        except:
            return {}
    
    def interpolate_video(
        self,
        input_path: Path,
        output_path: Path,
        source_fps: int = 16,
        target_fps: int = 24,
        model: str = "rife",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Main interpolation method - handles full video pipeline.
        Preserves original video duration while changing frame rate.
        
        Note on RIFE multipliers:
          rife_n=1 = 2x frames (1 intermediate between each pair)
          rife_n=2 = 4x frames (3 intermediates between each pair)
          rife_n=3 = 8x frames (7 intermediates between each pair)
        
        The actual -n parameter passed to rife-ncnn-vulkan is the
        TARGET TOTAL FRAME COUNT, calculated from the multiplier.
        
        For 16fps -> 24fps (need 1.5x frames):
          Use multiplier 4x, encode at 64fps -> 24fps
          Duration preserved: (frames * 4) / (fps * 4) = original_duration
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get source video info for logging
        video_info = self.get_video_info(input_path)
        source_duration = video_info.get("duration", 0)
        detected_fps = video_info.get("fps", source_fps)
        
        # Use detected fps if available, otherwise use provided source_fps
        actual_source_fps = detected_fps if detected_fps > 0 else source_fps
        
        # Calculate how many doublings we need
        # For 16->24: need 1.5x, so use 2 doublings (4x) and let ffmpeg drop frames
        # For 16->30: need 1.875x, so use 2 doublings (4x)
        # For 16->60: need 3.75x, so use 2 doublings (4x)
        fps_ratio = target_fps / actual_source_fps
        
        # rife_n = multiplier exponent (2^rife_n = frame multiplier)
        # rife_n=1 = 2x, rife_n=2 = 4x, rife_n=3 = 8x
        if fps_ratio <= 2:
            rife_n = 2  # 4x frames - good for 1.5x or 2x
        elif fps_ratio <= 4:
            rife_n = 2  # 4x frames
        else:
            rife_n = 3  # 8x frames for very high ratios
        
        # Actual frame multiplier from RIFE
        rife_multiplier = 2 ** rife_n  # rife_n=2 = 4x, rife_n=3 = 8x
        
        print(f"[Interpolator] ===== FRAME INTERPOLATION =====")
        print(f"[Interpolator] Input: {input_path.name}")
        print(f"[Interpolator] Source: {actual_source_fps}fps, duration: {source_duration:.2f}s")
        print(f"[Interpolator] Target: {target_fps}fps (same duration)")
        print(f"[Interpolator] Frame ratio needed: {fps_ratio:.2f}x")
        print(f"[Interpolator] RIFE multiplier: {rife_multiplier}x frames")
        print(f"[Interpolator] Encode: {actual_source_fps * rife_multiplier}fps -> {target_fps}fps")
        
        # Create temp directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            output_frames_dir = temp_path / "output_frames"
            
            # Extract frames
            self.extract_frames(input_path, input_frames_dir, progress_callback)
            
            # Count actual extracted frames
            extracted_frames = len(list(input_frames_dir.glob("*.png")))
            print(f"[Interpolator] Extracted {extracted_frames} frames")
            
            # Expected frames after RIFE: multiplier^n * (N-1) + 1
            expected_rife_frames = (2 ** rife_n) * (extracted_frames - 1) + 1
            print(f"[Interpolator] Expected after RIFE: ~{expected_rife_frames} frames")
            
            # Interpolate
            try:
                if self.ncnn_binary:
                    self.interpolate_frames_ncnn(
                        input_frames_dir, output_frames_dir,
                        rife_n,  # Pass multiplier exponent (2^rife_n = multiplier)
                        progress_callback=progress_callback
                    )
                else:
                    self.interpolate_frames_opencv(
                        input_frames_dir, output_frames_dir,
                        rife_multiplier,
                        progress_callback=progress_callback
                    )
            except Exception as e:
                print(f"[Interpolator] NCNN failed, using OpenCV fallback: {e}")
                self.interpolate_frames_opencv(
                    input_frames_dir, output_frames_dir,
                    rife_multiplier,
                    progress_callback=progress_callback
                )
            
            # Count interpolated frames
            interp_frames = len(list(output_frames_dir.glob("*.png")))
            print(f"[Interpolator] After RIFE: {interp_frames} frames")
            
            # Encode video with correct timing
            # input_fps = source_fps * multiplier (preserves original duration)
            # output_fps = target_fps
            effective_input_fps = actual_source_fps * rife_multiplier
            expected_duration = interp_frames / effective_input_fps
            expected_output_frames = int(interp_frames * target_fps / effective_input_fps)
            
            print(f"[Interpolator] Encode: {interp_frames} @ {effective_input_fps}fps = {expected_duration:.2f}s")
            print(f"[Interpolator] Output: ~{expected_output_frames} frames @ {target_fps}fps")
            
            self.frames_to_video(
                output_frames_dir, output_path,
                input_fps=effective_input_fps,
                output_fps=target_fps,
                progress_callback=progress_callback
            )
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return {
            "output_path": str(output_path),
            "source_fps": actual_source_fps,
            "target_fps": target_fps,
            "source_frames": extracted_frames,
            "interpolated_frames": interp_frames,
            "rife_n": rife_n,
            "multiplier": rife_multiplier,
            "duration": source_duration,
            "method": "ncnn" if self.ncnn_binary else "opencv"
        }


# ============================================================================
# PostProcessor Manager
# ============================================================================

class PostProcessorManager:
    """
    Unified manager for all post-processing operations
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./outputs/postprocessed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors lazily
        self._upscaler = None
        self._interpolator = None
        
        print(f"[PostProcessor] Manager initialized, output: {self.output_dir}")
    
    @property
    def upscaler(self) -> RealESRGANUpscaler:
        if self._upscaler is None:
            self._upscaler = RealESRGANUpscaler()
        return self._upscaler
    
    @property
    def interpolator(self) -> RIFEInterpolator:
        if self._interpolator is None:
            self._interpolator = RIFEInterpolator()
        return self._interpolator
    
    def upscale_image(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 4,
        model: str = "realesrgan",
        denoise: float = 0.5,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Upscale a single image"""
        return self.upscaler.upscale(
            input_path, output_path, scale, model, denoise, progress_callback
        )
    
    def interpolate_video(
        self,
        input_path: Path,
        output_path: Path,
        source_fps: int = 16,
        target_fps: int = 24,
        model: str = "rife",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Interpolate video frames"""
        return self.interpolator.interpolate_video(
            input_path, output_path, source_fps, target_fps, model, progress_callback
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Report available capabilities"""
        return {
            "upscaling": {
                "available": True,
                "ncnn_available": self.upscaler.ncnn_binary is not None,
                "models": ["realesrgan", "lanczos"],
                "scales": [2, 4, 8],
            },
            "interpolation": {
                "available": True,
                "ncnn_available": self.interpolator.ncnn_binary is not None,
                "models": ["rife"],
                "target_fps": [24, 30, 60],
            }
        }


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    manager = PostProcessorManager()
    print("\nCapabilities:")
    import json
    print(json.dumps(manager.get_capabilities(), indent=2))