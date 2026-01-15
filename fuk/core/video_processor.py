# core/video_processor.py
"""
Video Processing Utilities for FUK Pipeline

Provides frame-by-frame video processing for:
- Preprocessors (Canny, Depth, Normals, etc.)
- Postprocessors (Upscaling)
- Layers (batch AOV generation)

Outputs:
- MP4 video (current demo)
- Image sequences (future EXR workflow)

The design keeps processors unchanged - they still work on single frames.
This module handles extraction, coordination, and reassembly.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'


class OutputMode(str, Enum):
    """Video output format"""
    MP4 = "mp4"          # Reassemble to MP4 (demo mode)
    SEQUENCE = "sequence"  # Keep as image sequence (EXR workflow)


class VideoProcessor:
    """
    Frame-by-frame video processor
    
    Extracts frames, applies a processing function to each,
    then reassembles to video or leaves as sequence.
    
    Usage:
        processor = VideoProcessor()
        
        # Process with any function that takes (input_path, output_path) -> result
        result = processor.process_video(
            input_video=Path("input.mp4"),
            output_path=Path("output.mp4"),
            frame_processor=lambda inp, out: my_processor.process(inp, out),
            output_mode=OutputMode.MP4,
            progress_callback=lambda p, msg: print(f"{p*100:.0f}% {msg}")
        )
    """
    
    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        self.ffprobe_path = self._find_ffprobe()
        
        print(f"[VideoProcessor] Initialized")
        print(f"[VideoProcessor]   FFmpeg: {self.ffmpeg_path or 'Not found'}")
        print(f"[VideoProcessor]   FFprobe: {self.ffprobe_path or 'Not found'}")
    
    def _find_ffmpeg(self) -> Optional[Path]:
        """Find ffmpeg binary"""
        result = shutil.which("ffmpeg")
        return Path(result) if result else None
    
    def _find_ffprobe(self) -> Optional[Path]:
        """Find ffprobe binary"""
        result = shutil.which("ffprobe")
        return Path(result) if result else None
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get video metadata
        
        Returns:
            Dict with fps, duration, width, height, frame_count
        """
        if not self.ffprobe_path:
            # Fallback to OpenCV
            cap = cv2.VideoCapture(str(video_path))
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            }
            cap.release()
            return info
        
        cmd = [
            str(self.ffprobe_path),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(video_path)
        ]
        
        import json
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), {})
        
        # Parse fps from r_frame_rate (e.g., "24/1" or "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "24/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        return {
            "fps": fps,
            "frame_count": int(video_stream.get("nb_frames", 0)) or int(float(data.get("format", {}).get("duration", 0)) * fps),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
        }
    
    def extract_thumbnail(
        self,
        video_path: Path,
        output_path: Path = None,
        quality: int = 5
    ) -> Optional[Path]:
        """
        Extract first frame as thumbnail image
        
        Args:
            video_path: Input video
            output_path: Output thumbnail path (default: video_path.thumb.jpg)
            quality: JPEG quality (2-31, lower is better, 5 is good balance)
            
        Returns:
            Path to thumbnail if successful, None otherwise
        """
        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.with_suffix('.thumb.jpg')
        else:
            output_path = Path(output_path)
        
        if not self.ffmpeg_path:
            print(f"[VideoProcessor] Cannot extract thumbnail - FFmpeg not found")
            return None
        
        cmd = [
            str(self.ffmpeg_path),
            "-y",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", str(quality),
            str(output_path)
        ]
        
        print(f"[VideoProcessor] Extracting thumbnail: {output_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[VideoProcessor] Thumbnail extraction failed: {result.stderr}")
            return None
        
        if output_path.exists():
            print(f"[VideoProcessor] Thumbnail created: {output_path} ({output_path.stat().st_size} bytes)")
            return output_path
        
        return None
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> int:
        """
        Extract all frames from video
        
        Args:
            video_path: Input video
            output_dir: Directory to save frames as frame_NNNNNN.png
            progress_callback: Optional (progress, message) callback
            
        Returns:
            Number of frames extracted
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_info = self.get_video_info(video_path)
        total_frames = video_info.get("frame_count", 0)
        
        print(f"[VideoProcessor] Extracting frames from {video_path}")
        print(f"[VideoProcessor]   Output dir: {output_dir}")
        print(f"[VideoProcessor]   Expected frames: {total_frames}")
        
        if progress_callback:
            progress_callback(0.0, f"Extracting {total_frames} frames...")
        
        if self.ffmpeg_path:
            # Use FFmpeg for extraction (faster, more reliable)
            cmd = [
                str(self.ffmpeg_path),
                "-i", str(video_path),
                "-vsync", "0",
                "-q:v", "2",  # High quality
                str(output_dir / "frame_%06d.png")
            ]
            
            print(f"[VideoProcessor] Running FFmpeg: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # FFmpeg outputs progress to stderr
            for line in process.stderr:
                if "frame=" in line:
                    try:
                        frame_str = line.split("frame=")[1].split()[0].strip()
                        frame_num = int(frame_str)
                        if progress_callback and total_frames > 0:
                            progress_callback(
                                0.05 + 0.25 * (frame_num / total_frames),
                                f"Extracting frame {frame_num}/{total_frames}"
                            )
                    except (ValueError, IndexError):
                        pass
            
            process.wait()
            
            if process.returncode != 0:
                print(f"[VideoProcessor] FFmpeg extraction returned non-zero: {process.returncode}")
        else:
            # Fallback to OpenCV
            print(f"[VideoProcessor] Using OpenCV fallback for extraction")
            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_path = output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                
                frame_idx += 1
                if progress_callback and total_frames > 0:
                    progress_callback(
                        0.05 + 0.25 * (frame_idx / total_frames),
                        f"Extracting frame {frame_idx}/{total_frames}"
                    )
            
            cap.release()
        
        # Count extracted frames
        extracted = len(list(output_dir.glob("frame_*.png")))
        print(f"[VideoProcessor] Extracted {extracted} frames to {output_dir}")
        
        # List first few frames for debugging
        frames = sorted(output_dir.glob("frame_*.png"))[:5]
        for f in frames:
            print(f"[VideoProcessor]   - {f.name}")
        if len(list(output_dir.glob("frame_*.png"))) > 5:
            print(f"[VideoProcessor]   ... and {extracted - 5} more")
        
        return extracted
    
    def assemble_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float,
        frame_pattern: str = "frame_%06d.png",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Assemble frames back into video
        
        Args:
            frames_dir: Directory containing frames
            output_path: Output video path
            fps: Target framerate
            frame_pattern: Frame filename pattern
            progress_callback: Optional callback
            
        Returns:
            Dict with output info
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        frame_count = len(list(frames_dir.glob("frame_*.png")))
        
        print(f"[VideoProcessor] Assembling video from {frame_count} frames")
        print(f"[VideoProcessor]   Frames dir: {frames_dir}")
        print(f"[VideoProcessor]   Output: {output_path}")
        print(f"[VideoProcessor]   FPS: {fps}")
        
        if progress_callback:
            progress_callback(0.9, f"Assembling {frame_count} frames to video...")
        
        if self.ffmpeg_path:
            cmd = [
                str(self.ffmpeg_path),
                "-y",  # Overwrite
                "-framerate", str(fps),
                "-i", str(frames_dir / frame_pattern),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",  # High quality
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            
            print(f"[VideoProcessor] Running FFmpeg: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[VideoProcessor] FFmpeg assembly error: {result.stderr}")
            else:
                print(f"[VideoProcessor] FFmpeg assembly complete")
        else:
            # Fallback to OpenCV
            print(f"[VideoProcessor] Using OpenCV fallback for assembly")
            frames = sorted(frames_dir.glob("frame_*.png"))
            if not frames:
                raise ValueError("No frames to assemble")
            
            first_frame = cv2.imread(str(frames[0]))
            h, w = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                writer.write(frame)
            
            writer.release()
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        # Verify output
        if output_path.exists():
            print(f"[VideoProcessor] Output video created: {output_path} ({output_path.stat().st_size} bytes)")
        else:
            print(f"[VideoProcessor] WARNING: Output video not found at {output_path}")
        
        return {
            "output_path": str(output_path),
            "frame_count": frame_count,
            "fps": fps,
            "duration": frame_count / fps,
        }
    
    def process_video(
        self,
        input_video: Path,
        output_path: Path,
        frame_processor: Callable[[Path, Path], Dict[str, Any]],
        output_mode: OutputMode = OutputMode.MP4,
        parallel: bool = False,
        max_workers: int = 4,
        preserve_fps: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process video frame-by-frame
        
        Args:
            input_video: Input video path
            output_path: Output path (video file or directory for sequences)
            frame_processor: Function that processes (input_frame, output_frame) -> result_dict
            output_mode: MP4 or SEQUENCE
            parallel: Process frames in parallel (use for GPU-bound processors)
            max_workers: Number of parallel workers
            preserve_fps: Keep original FPS
            progress_callback: (progress, message) callback
            processor_kwargs: Additional kwargs to pass to processor
            
        Returns:
            Dict with processing results and output info
        """
        input_video = Path(input_video)
        output_path = Path(output_path)
        processor_kwargs = processor_kwargs or {}
        
        # Get video info
        video_info = self.get_video_info(input_video)
        fps = video_info["fps"]
        total_frames = video_info["frame_count"]
        
        print(f"\n[VideoProcessor] ===== PROCESSING VIDEO =====")
        print(f"[VideoProcessor] Input: {input_video.name}")
        print(f"[VideoProcessor] Frames: {total_frames} @ {fps:.2f}fps")
        print(f"[VideoProcessor] Mode: {output_mode.value}")
        print(f"[VideoProcessor] Output path: {output_path}")
        
        if progress_callback:
            progress_callback(0.0, "Starting video processing...")
        
        # Create temp directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            output_frames_dir = temp_path / "output_frames"
            
            input_frames_dir.mkdir()
            output_frames_dir.mkdir()
            
            print(f"[VideoProcessor] Temp input dir: {input_frames_dir}")
            print(f"[VideoProcessor] Temp output dir: {output_frames_dir}")
            
            # Step 1: Extract frames (0% - 25%)
            extracted = self.extract_frames(
                input_video, 
                input_frames_dir,
                progress_callback=lambda p, m: progress_callback(p * 0.25, m) if progress_callback else None
            )
            
            # Get list of input frames
            input_frames = sorted(input_frames_dir.glob("frame_*.png"))
            
            if not input_frames:
                raise ValueError("No frames extracted from video")
            
            print(f"[VideoProcessor] Found {len(input_frames)} input frames to process")
            
            # Step 2: Process frames (25% - 90%)
            results = []
            errors = []
            
            def process_single_frame(idx: int, frame_path: Path) -> Dict[str, Any]:
                """Process a single frame"""
                output_frame = output_frames_dir / frame_path.name
                try:
                    result = frame_processor(frame_path, output_frame, **processor_kwargs)
                    
                    # Verify output was created
                    if output_frame.exists():
                        print(f"[VideoProcessor] Frame {idx}: OK -> {output_frame.name}")
                    else:
                        print(f"[VideoProcessor] Frame {idx}: WARNING - processor didn't create output")
                        # Copy original as fallback
                        shutil.copy(frame_path, output_frame)
                    
                    return {"success": True, "frame": idx, "result": result}
                except Exception as e:
                    print(f"[VideoProcessor] Frame {idx} failed: {e}")
                    # Copy original as fallback
                    shutil.copy(frame_path, output_frame)
                    return {"success": False, "frame": idx, "error": str(e)}
            
            if parallel and max_workers > 1:
                # Parallel processing (for CPU-bound processors)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(process_single_frame, idx, frame): idx 
                        for idx, frame in enumerate(input_frames)
                    }
                    
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        
                        if not result["success"]:
                            errors.append(result)
                        
                        if progress_callback:
                            completed = len(results)
                            progress_callback(
                                0.25 + 0.65 * (completed / len(input_frames)),
                                f"Processing frame {completed}/{len(input_frames)}"
                            )
            else:
                # Sequential processing (for GPU-bound processors)
                for idx, frame_path in enumerate(input_frames):
                    result = process_single_frame(idx, frame_path)
                    results.append(result)
                    
                    if not result["success"]:
                        errors.append(result)
                    
                    if progress_callback:
                        progress_callback(
                            0.25 + 0.65 * ((idx + 1) / len(input_frames)),
                            f"Processing frame {idx + 1}/{len(input_frames)}"
                        )
            
            processed_frames = len(list(output_frames_dir.glob("frame_*.png")))
            print(f"[VideoProcessor] Processed {processed_frames} frames ({len(errors)} errors)")
            
            # List output frames for debugging
            output_frame_list = sorted(output_frames_dir.glob("frame_*.png"))
            print(f"[VideoProcessor] Output frames in temp dir: {len(output_frame_list)}")
            for f in output_frame_list[:3]:
                print(f"[VideoProcessor]   - {f.name} ({f.stat().st_size} bytes)")
            if len(output_frame_list) > 3:
                print(f"[VideoProcessor]   ... and {len(output_frame_list) - 3} more")
            
            # Step 3: Output (90% - 100%)
            if output_mode == OutputMode.SEQUENCE:
                # Copy frames to output directory
                print(f"[VideoProcessor] Copying {processed_frames} frames to {output_path}")
                output_path.mkdir(parents=True, exist_ok=True)
                
                copied_frames = []
                for frame in sorted(output_frames_dir.glob("frame_*.png")):
                    dest = output_path / frame.name
                    shutil.copy(frame, dest)
                    copied_frames.append(frame.name)
                
                print(f"[VideoProcessor] Copied {len(copied_frames)} frames to output")
                
                # Verify output
                final_frames = sorted(output_path.glob("frame_*.png"))
                print(f"[VideoProcessor] Final output contains {len(final_frames)} frames")
                
                # Get first frame for preview
                first_frame = final_frames[0] if final_frames else None
                first_frame_name = first_frame.name if first_frame else None
                
                if progress_callback:
                    progress_callback(1.0, "Complete")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "output_mode": "sequence",
                    "frame_count": len(final_frames),
                    "fps": fps,
                    "errors": [e["error"] for e in errors],
                    # NEW: Include frame information for sequences
                    "frames": [f.name for f in final_frames],
                    "first_frame": first_frame_name,
                    "preview_path": str(first_frame) if first_frame else None,
                }
            
            else:  # MP4
                assembly_result = self.assemble_video(
                    output_frames_dir,
                    output_path,
                    fps=fps,
                    progress_callback=lambda p, m: progress_callback(0.9 + p * 0.1, m) if progress_callback else None
                )
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "output_mode": "mp4",
                    "frame_count": processed_frames,
                    "fps": fps,
                    "duration": assembly_result["duration"],
                    "errors": [e["error"] for e in errors],
                }
    
    def process_video_layers(
        self,
        input_video: Path,
        output_dir: Path,
        layer_processors: Dict[str, Callable[[Path, Path], Dict[str, Any]]],
        output_mode: OutputMode = OutputMode.MP4,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process video to generate multiple layer outputs (depth, normals, etc.)
        
        Each layer processor runs on all frames, producing separate outputs.
        
        Args:
            input_video: Input video
            output_dir: Directory for outputs (one per layer)
            layer_processors: Dict of layer_name -> processor_function
            output_mode: MP4 or SEQUENCE per layer
            progress_callback: Progress callback
            
        Returns:
            Dict with results per layer
        """
        input_video = Path(input_video)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_info = self.get_video_info(input_video)
        total_frames = video_info["frame_count"]
        fps = video_info["fps"]
        
        num_layers = len(layer_processors)
        
        print(f"\n[VideoProcessor] ===== PROCESSING VIDEO LAYERS =====")
        print(f"[VideoProcessor] Layers: {list(layer_processors.keys())}")
        print(f"[VideoProcessor] Frames: {total_frames} @ {fps:.2f}fps")
        print(f"[VideoProcessor] Output dir: {output_dir}")
        print(f"[VideoProcessor] Output mode: {output_mode.value}")
        
        if progress_callback:
            progress_callback(0.0, f"Processing {num_layers} layers...")
        
        # Extract frames once
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            input_frames_dir.mkdir()
            
            # Extract (10% of total progress)
            extracted = self.extract_frames(
                input_video,
                input_frames_dir,
                progress_callback=lambda p, m: progress_callback(p * 0.1, m) if progress_callback else None
            )
            
            input_frames = sorted(input_frames_dir.glob("frame_*.png"))
            
            results = {
                "success": True,
                "layers": {},
                "errors": {},
            }
            
            # Process each layer (90% of progress split among layers)
            layer_progress_weight = 0.9 / max(num_layers, 1)
            
            for layer_idx, (layer_name, processor) in enumerate(layer_processors.items()):
                layer_start_progress = 0.1 + layer_idx * layer_progress_weight
                
                print(f"\n[VideoProcessor] Processing layer: {layer_name}")
                
                layer_output_dir = temp_path / f"output_{layer_name}"
                layer_output_dir.mkdir()
                
                layer_errors = []
                
                # Process all frames for this layer
                for frame_idx, frame_path in enumerate(input_frames):
                    output_frame = layer_output_dir / frame_path.name
                    
                    try:
                        processor(frame_path, output_frame)
                        
                        if not output_frame.exists():
                            print(f"[VideoProcessor] {layer_name} frame {frame_idx}: WARNING - no output")
                            shutil.copy(frame_path, output_frame)
                            
                    except Exception as e:
                        print(f"[VideoProcessor] {layer_name} frame {frame_idx} failed: {e}")
                        layer_errors.append({"frame": frame_idx, "error": str(e)})
                        # Copy original as fallback
                        shutil.copy(frame_path, output_frame)
                    
                    if progress_callback:
                        frame_progress = (frame_idx + 1) / len(input_frames)
                        progress_callback(
                            layer_start_progress + frame_progress * layer_progress_weight,
                            f"{layer_name}: frame {frame_idx + 1}/{len(input_frames)}"
                        )
                
                # Assemble or copy output
                if output_mode == OutputMode.MP4:
                    layer_output = output_dir / f"{layer_name}.mp4"
                    self.assemble_video(layer_output_dir, layer_output, fps)
                    
                    results["layers"][layer_name] = {
                        "output_path": str(layer_output),
                        "frame_count": len(input_frames),
                        "errors": len(layer_errors),
                    }
                else:
                    layer_seq_dir = output_dir / layer_name
                    layer_seq_dir.mkdir(exist_ok=True)
                    
                    copied_frames = []
                    for frame in sorted(layer_output_dir.glob("frame_*.png")):
                        shutil.copy(frame, layer_seq_dir / frame.name)
                        copied_frames.append(frame.name)
                    
                    print(f"[VideoProcessor] {layer_name}: Copied {len(copied_frames)} frames to {layer_seq_dir}")
                    
                    # Get first frame for preview
                    final_frames = sorted(layer_seq_dir.glob("frame_*.png"))
                    first_frame = final_frames[0] if final_frames else None
                    
                    results["layers"][layer_name] = {
                        "output_path": str(layer_seq_dir),
                        "frame_count": len(final_frames),
                        "errors": len(layer_errors),
                        # NEW: Include frame info for sequences
                        "frames": [f.name for f in final_frames],
                        "first_frame": first_frame.name if first_frame else None,
                        "preview_path": str(first_frame) if first_frame else None,
                    }
                
                if layer_errors:
                    results["errors"][layer_name] = layer_errors
            
            if progress_callback:
                progress_callback(1.0, "Complete")
            
            results["fps"] = fps
            results["frame_count"] = len(input_frames)
            
            return results


# ============================================================================
# Convenience Functions
# ============================================================================

def is_video_file(path: Union[str, Path]) -> bool:
    """Check if path is a video file based on extension"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    return Path(path).suffix.lower() in video_extensions


def is_image_file(path: Union[str, Path]) -> bool:
    """Check if path is an image file based on extension"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
    return Path(path).suffix.lower() in image_extensions


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video processor")
    parser.add_argument("input", type=Path, help="Input video")
    parser.add_argument("--output", type=Path, default=Path("test_output.mp4"))
    parser.add_argument("--mode", choices=["mp4", "sequence"], default="mp4")
    args = parser.parse_args()
    
    processor = VideoProcessor()
    
    # Test with a simple grayscale processor
    def grayscale_processor(input_path: Path, output_path: Path) -> Dict[str, Any]:
        img = cv2.imread(str(input_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(output_path), gray_bgr)
        return {"method": "grayscale"}
    
    output_mode = OutputMode.SEQUENCE if args.mode == "sequence" else OutputMode.MP4
    
    result = processor.process_video(
        args.input,
        args.output,
        grayscale_processor,
        output_mode=output_mode,
        progress_callback=lambda p, m: print(f"{p*100:.0f}% {m}")
    )
    
    print("\nResult:")
    import json
    print(json.dumps(result, indent=2))