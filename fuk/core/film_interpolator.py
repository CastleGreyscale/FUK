# core/film_interpolator.py
"""
FILM Frame Interpolator for FUK Pipeline

Google's FILM (Frame Interpolation for Large Motion) via the PyTorch port.
Replaces RIFE with native float32 PyTorch inference — no external binaries,
no uint8 bottleneck.

Model: dajes/frame-interpolation-pytorch (Apache 2.0)
Paper: Reda et al., FILM: Frame Interpolation for Large Motion, ECCV 2022

Usage:
    interpolator = FILMInterpolator()
    result = interpolator.interpolate_video(
        input_path, output_path,
        source_fps=16, target_fps=24,
    )
"""

import sys
import os
import subprocess
import tempfile
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ============================================================================
# Constants
# ============================================================================

# TorchScript model from: https://github.com/dajes/frame-interpolation-pytorch/releases
FILM_MODEL_URL = "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt"
FILM_MODEL_FILENAME = "film_net_fp32.pt"

# Default cache location for the model
DEFAULT_MODEL_DIR = Path.home() / "ai" / "models" / "film"


# ============================================================================
# FILM Interpolator
# ============================================================================

class FILMInterpolator:
    """
    FILM frame interpolation using native PyTorch.
    
    Key advantages over RIFE-ncnn:
    - Float32 tensors throughout (no uint8 quantization)
    - Native PyTorch (no external binary dependency)
    - Better quality for large motion (FILM's core innovation)
    - Recursive binary interpolation for optimal quality at any multiplier
    
    The model operates on normalized [0, 1] float tensors with shape (B, 3, H, W).
    Frames stay in float32 until final output.
    """
    
    def __init__(self, models_dir: Optional[Path] = None, device: Optional[str] = None):
        self.models_dir = models_dir or DEFAULT_MODEL_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Use fp16 on CUDA for speed, fp32 on CPU
        self.precision = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.model = None
        self._model_path = self.models_dir / FILM_MODEL_FILENAME
        
        print(f"[FILM] Initialized")
        print(f"[FILM] Device: {self.device}, Precision: {self.precision}")
        print(f"[FILM] Model path: {self._model_path}")
        print(f"[FILM] Model available: {self._model_path.exists()}")
    
    # ========================================================================
    # Model Management
    # ========================================================================
    
    def _ensure_model(self):
        """Load model if not already loaded, download if needed."""
        if self.model is not None:
            return
        
        if not self._model_path.exists():
            self._download_model()
        
        print(f"[FILM] Loading TorchScript model from {self._model_path}...")
        start = time.time()
        self.model = torch.jit.load(str(self._model_path), map_location="cpu")
        self.model.eval().to(device=self.device, dtype=self.precision)
        print(f"[FILM] Model loaded in {time.time() - start:.1f}s")
    
    def _download_model(self):
        """Download FILM TorchScript model from GitHub releases."""
        print(f"[FILM] Model not found at {self._model_path}")
        print(f"[FILM] Downloading from {FILM_MODEL_URL}...")
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import urllib.request
            temp_path = self._model_path.with_suffix(".tmp")
            urllib.request.urlretrieve(FILM_MODEL_URL, str(temp_path))
            temp_path.rename(self._model_path)
            print(f"[FILM] Downloaded to {self._model_path}")
        except Exception as e:
            print(f"[FILM] Auto-download failed: {e}")
            print(f"[FILM] Please download manually:")
            print(f"[FILM]   wget {FILM_MODEL_URL} -O {self._model_path}")
            raise RuntimeError(
                f"FILM model not found. Download from:\n"
                f"  {FILM_MODEL_URL}\n"
                f"Place at: {self._model_path}"
            )
    
    def cleanup(self):
        """Unload model and free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[FILM] Model unloaded, VRAM freed")
    
    # ========================================================================
    # Frame I/O — float32 native
    # ========================================================================
    
    def _load_frame(self, path: Path) -> torch.Tensor:
        """Load an image as a float32 tensor [1, 3, H, W] in [0, 1] range."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load frame: {path}")
        
        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(device=self.device, dtype=self.precision)
    
    def _save_frame(self, tensor: torch.Tensor, path: Path):
        """Save a [1, 3, H, W] float tensor as PNG."""
        # Clamp, convert to uint8 for saving
        img = tensor.squeeze(0).clamp(0, 1).float().cpu()
        img = (img * 255.0).byte().permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)
    
    def _save_frame_float32(self, tensor: torch.Tensor, path: Path):
        """Save a [1, 3, H, W] float tensor as 16-bit PNG for max quality."""
        img = tensor.squeeze(0).clamp(0, 1).float().cpu()
        # Scale to uint16 range for 16-bit PNG
        img = (img * 65535.0).to(torch.int32).permute(1, 2, 0).numpy().astype(np.uint16)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)
    
    # ========================================================================
    # Core Interpolation
    # ========================================================================
    
    @torch.no_grad()
    def interpolate_pair(
        self, frame1: torch.Tensor, frame2: torch.Tensor, dt: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate a single frame between two input frames.
        
        Args:
            frame1: [1, 3, H, W] tensor in [0, 1]
            frame2: [1, 3, H, W] tensor in [0, 1]  
            dt: Time position between frames (0.0 = frame1, 1.0 = frame2)
        
        Returns:
            Interpolated frame [1, 3, H, W] in [0, 1]
        """
        self._ensure_model()
        
        # FILM needs height and width divisible by 32
        _, _, h, w = frame1.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            frame1 = torch.nn.functional.pad(frame1, (0, pad_w, 0, pad_h), mode="replicate")
            frame2 = torch.nn.functional.pad(frame2, (0, pad_w, 0, pad_h), mode="replicate")
        
        dt_tensor = frame1.new_full((1, 1), dt)
        result = self.model(frame1, frame2, dt_tensor)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :h, :w]
        
        return result
    
    def interpolate_recursive(
        self,
        frames: List[torch.Tensor],
        times_to_interpolate: int,
        progress_callback: Optional[Callable] = None,
    ) -> List[torch.Tensor]:
        """
        Recursively interpolate between frames — FILM's native mode.
        
        Each level doubles the frame count by inserting midpoints between
        every adjacent pair. This is higher quality than linear dt spacing
        because each interpolation is always at dt=0.5 (the sweet spot).
        
        Args:
            frames: List of [1, 3, H, W] tensors
            times_to_interpolate: Number of recursive passes
                1 = 2x frames, 2 = 4x frames, 3 = 8x frames
            progress_callback: Optional (progress, message) callback
        
        Returns:
            List of interpolated [1, 3, H, W] tensors
        """
        self._ensure_model()
        
        result = list(frames)
        total_passes = times_to_interpolate
        
        for level in range(times_to_interpolate):
            new_result = [result[0]]
            pairs_total = len(result) - 1
            
            for i in range(pairs_total):
                mid = self.interpolate_pair(result[i], result[i + 1], dt=0.5)
                new_result.append(mid)
                new_result.append(result[i + 1])
                
                if progress_callback:
                    overall = (level + (i + 1) / pairs_total) / total_passes
                    progress_callback(
                        0.2 + 0.6 * overall,
                        f"FILM pass {level + 1}/{total_passes}, "
                        f"pair {i + 1}/{pairs_total}"
                    )
            
            result = new_result
            print(f"[FILM] Level {level + 1}/{total_passes}: {len(result)} frames")
        
        return result
    
    # ========================================================================
    # Video Pipeline
    # ========================================================================
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> List[Path]:
        """Extract frames from video using ffmpeg."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback(0.05, "Extracting frames from video")
        
        # Use PNG for lossless extraction
        frame_pattern = output_dir / "%08d.png"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vsync", "0",
            "-start_number", "0",
            str(frame_pattern),
        ]
        
        print(f"[FILM] Extracting frames: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
        
        frames = sorted(output_dir.glob("*.png"))
        print(f"[FILM] Extracted {len(frames)} frames")
        return frames
    
    def frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        input_fps: float,
        output_fps: int = 24,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """Encode frames back to video with ffmpeg."""
        if progress_callback:
            progress_callback(0.9, "Encoding output video")
        
        frame_files = sorted(frames_dir.glob("*.png"))
        num_frames = len(frame_files)
        
        if num_frames == 0:
            raise RuntimeError("No frames to encode")
        
        # Detect naming pattern
        first_name = frame_files[0].stem
        if first_name.startswith("frame_"):
            frame_pattern = frames_dir / "frame_%05d.png"
        else:
            num_digits = len(first_name)
            frame_pattern = frames_dir / f"%0{num_digits}d.png"
        
        expected_duration = num_frames / input_fps
        print(f"[FILM] Encoding {num_frames} frames")
        print(f"[FILM]   Input: {input_fps}fps, Output: {output_fps}fps")
        print(f"[FILM]   Expected duration: {expected_duration:.2f}s")
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(input_fps),
            "-i", str(frame_pattern),
            "-r", str(output_fps),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "14",
            "-pix_fmt", "yuv444p",
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Video encoding failed: {result.stderr}")
        
        print(f"[FILM] Output: {output_path}")
        return output_path
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video FPS, frame count, and duration via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(video_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}
        
        try:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            
            fps_str = stream.get("r_frame_rate", "16/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            nb_frames = stream.get("nb_frames")
            if nb_frames and nb_frames != "N/A":
                frame_count = int(nb_frames)
            else:
                duration = float(stream.get("duration", 0))
                frame_count = int(duration * fps)
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": frame_count / fps if fps > 0 else 0,
            }
        except Exception:
            return {}
    
    def interpolate_video(
        self,
        input_path: Path,
        output_path: Path,
        source_fps: int = 16,
        target_fps: int = 24,
        model: str = "film",
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point — full video interpolation pipeline.
        
        Same interface as RIFEInterpolator.interpolate_video() for
        drop-in compatibility with PostProcessorManager.
        
        Pipeline:
        1. Extract frames from video (PNG, lossless)
        2. Load as float32 tensors
        3. Recursive binary interpolation via FILM
        4. Save interpolated frames
        5. Encode to output video
        
        Duration is preserved: more frames at higher FPS = same duration.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Get source info
        video_info = self.get_video_info(input_path)
        detected_fps = video_info.get("fps", source_fps)
        actual_source_fps = detected_fps if detected_fps > 0 else source_fps
        source_duration = video_info.get("duration", 0)
        
        # Calculate recursive levels needed
        # Each level doubles frames. We want enough to hit target_fps.
        # For 16->24: need 1.5x, use 2 levels (4x), ffmpeg drops extras
        # For 16->30: need ~2x, use 2 levels (4x)
        # For 16->60: need 3.75x, use 2 levels (4x)
        fps_ratio = target_fps / actual_source_fps
        
        if fps_ratio <= 2:
            times_to_interpolate = 1  # 2x frames
        elif fps_ratio <= 4:
            times_to_interpolate = 2  # 4x frames
        else:
            times_to_interpolate = 3  # 8x frames
        
        multiplier = 2 ** times_to_interpolate
        
        print(f"[FILM] ===== FRAME INTERPOLATION (FILM) =====")
        print(f"[FILM] Input: {input_path.name}")
        print(f"[FILM] Source: {actual_source_fps}fps, Duration: {source_duration:.2f}s")
        print(f"[FILM] Target: {target_fps}fps")
        print(f"[FILM] Ratio: {fps_ratio:.2f}x, Levels: {times_to_interpolate} ({multiplier}x)")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            output_frames_dir = temp_path / "output_frames"
            output_frames_dir.mkdir()
            
            # 1. Extract frames
            frame_paths = self.extract_frames(input_path, input_frames_dir, progress_callback)
            num_source = len(frame_paths)
            
            if num_source < 2:
                raise ValueError(f"Need at least 2 frames, got {num_source}")
            
            # 2. Load as float32 tensors
            if progress_callback:
                progress_callback(0.15, f"Loading {num_source} frames as float32 tensors")
            
            print(f"[FILM] Loading {num_source} frames to {self.device}...")
            tensors = [self._load_frame(p) for p in frame_paths]
            
            # 3. Recursive FILM interpolation
            print(f"[FILM] Running {times_to_interpolate} recursive interpolation passes...")
            interpolated = self.interpolate_recursive(
                tensors, times_to_interpolate, progress_callback
            )
            
            num_output = len(interpolated)
            print(f"[FILM] Result: {num_source} -> {num_output} frames")
            
            # 4. Save interpolated frames
            if progress_callback:
                progress_callback(0.85, f"Saving {num_output} frames")
            
            print(f"[FILM] Saving {num_output} frames...")
            for idx, tensor in enumerate(interpolated):
                out_path = output_frames_dir / f"frame_{idx:05d}.png"
                self._save_frame(tensor, out_path)
            
            # Free tensors and VRAM
            del tensors, interpolated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 5. Encode video — preserve original duration
            effective_input_fps = actual_source_fps * multiplier
            self.frames_to_video(
                output_frames_dir, output_path,
                input_fps=effective_input_fps,
                output_fps=target_fps,
                progress_callback=progress_callback,
            )
        
        elapsed = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        print(f"[FILM] Done in {elapsed:.1f}s")
        
        return {
            "output_path": str(output_path),
            "source_fps": actual_source_fps,
            "target_fps": target_fps,
            "source_frames": num_source,
            "interpolated_frames": num_output,
            "times_to_interpolate": times_to_interpolate,
            "multiplier": multiplier,
            "duration": source_duration,
            "elapsed_seconds": elapsed,
            "method": "film",
        }


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FILM frame interpolation")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--source-fps", type=int, default=16)
    parser.add_argument("--target-fps", type=int, default=24)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    interp = FILMInterpolator(device=args.device)
    result = interp.interpolate_video(
        Path(args.input), Path(args.output),
        source_fps=args.source_fps,
        target_fps=args.target_fps,
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")
    interp.cleanup()