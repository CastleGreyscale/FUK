# core/video_utils.py
"""
Shared Video Utilities

Single source of truth for:
- FFmpeg/FFprobe discovery
- Video metadata extraction
- Frame extraction and assembly
- File type detection
- Color palette generation (golden ratio for crypto mattes)

Every module that touches video frames imports from here.
No more per-class reimplementations.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import cv2
import numpy as np
import colorsys


# ============================================================================
# File Type Detection
# ============================================================================

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


def is_video_file(path: Union[str, Path]) -> bool:
    """Check if path is a video file"""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: Union[str, Path]) -> bool:
    """Check if path is an image file"""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


# ============================================================================
# FFmpeg Discovery
# ============================================================================

# Module-level cache so we only search PATH once
_ffmpeg_path: Optional[Path] = None
_ffprobe_path: Optional[Path] = None
_searched = False


def find_ffmpeg() -> Optional[Path]:
    """Find ffmpeg binary (cached after first call)"""
    global _ffmpeg_path, _searched
    if not _searched:
        _discover_tools()
    return _ffmpeg_path


def find_ffprobe() -> Optional[Path]:
    """Find ffprobe binary (cached after first call)"""
    global _ffprobe_path, _searched
    if not _searched:
        _discover_tools()
    return _ffprobe_path


def _discover_tools():
    """One-time discovery of ffmpeg/ffprobe"""
    global _ffmpeg_path, _ffprobe_path, _searched
    
    result = shutil.which("ffmpeg")
    _ffmpeg_path = Path(result) if result else None
    
    result = shutil.which("ffprobe")
    _ffprobe_path = Path(result) if result else None
    
    _searched = True
    
    print(f"[video_utils] FFmpeg:  {_ffmpeg_path or 'Not found'}")
    print(f"[video_utils] FFprobe: {_ffprobe_path or 'Not found'}")


# ============================================================================
# Video Metadata
# ============================================================================

def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    Get video metadata via ffprobe with OpenCV fallback.
    
    Returns:
        Dict with: fps, frame_count, width, height, duration, codec (ffprobe only)
    """
    video_path = Path(video_path)
    ffprobe = find_ffprobe()
    
    if ffprobe:
        return _get_video_info_ffprobe(video_path, ffprobe)
    else:
        return _get_video_info_opencv(video_path)


def _get_video_info_ffprobe(video_path: Path, ffprobe: Path) -> Dict[str, Any]:
    """Get video info via ffprobe (preferred - more accurate)"""
    import json as json_mod
    
    cmd = [
        str(ffprobe),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json_mod.loads(result.stdout)
        
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {}
        )
        
        # Parse fps from r_frame_rate (e.g., "24/1" or "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "24/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        
        duration = float(data.get("format", {}).get("duration", 0))
        frame_count = int(video_stream.get("nb_frames", 0))
        if not frame_count and duration and fps:
            frame_count = int(duration * fps)
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": duration,
            "codec": video_stream.get("codec_name", "unknown"),
        }
    except Exception as e:
        print(f"[video_utils] ffprobe failed, falling back to OpenCV: {e}")
        return _get_video_info_opencv(video_path)


def _get_video_info_opencv(video_path: Path) -> Dict[str, Any]:
    """Get video info via OpenCV (fallback)"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": frame_count / max(fps, 1),
    }
    cap.release()
    return info


# ============================================================================
# Frame Extraction
# ============================================================================

def extract_frames(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Path]:
    """
    Extract all frames from video as PNGs.
    
    Uses FFmpeg when available (faster, more reliable), OpenCV as fallback.
    Frame naming: frame_000001.png, frame_000002.png, ...
    
    Args:
        video_path: Input video file
        output_dir: Directory to write frames into (created if needed)
        progress_callback: Optional (progress 0-1, message) callback
        
    Returns:
        Sorted list of extracted frame paths
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ffmpeg = find_ffmpeg()
    
    if ffmpeg:
        _extract_frames_ffmpeg(video_path, output_dir, ffmpeg, progress_callback)
    else:
        _extract_frames_opencv(video_path, output_dir, progress_callback)
    
    frames = sorted(output_dir.glob("frame_*.png"))
    print(f"[video_utils] Extracted {len(frames)} frames to {output_dir}")
    return frames


def _extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    ffmpeg: Path,
    progress_callback: Optional[Callable] = None,
):
    """Extract frames using FFmpeg"""
    info = get_video_info(video_path)
    total_frames = info.get("frame_count", 0)
    
    cmd = [
        str(ffmpeg),
        "-i", str(video_path),
        "-vsync", "0",
        "-q:v", "2",
        str(output_dir / "frame_%06d.png")
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    for line in process.stderr:
        if "frame=" in line and progress_callback and total_frames > 0:
            try:
                frame_str = line.split("frame=")[1].split()[0].strip()
                frame_num = int(frame_str)
                progress_callback(
                    frame_num / total_frames,
                    f"Extracting frame {frame_num}/{total_frames}"
                )
            except (ValueError, IndexError):
                pass
    
    process.wait()
    if process.returncode != 0:
        print(f"[video_utils] FFmpeg extraction returned non-zero: {process.returncode}")


def _extract_frames_opencv(
    video_path: Path,
    output_dir: Path,
    progress_callback: Optional[Callable] = None,
):
    """Extract frames using OpenCV (fallback)"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imwrite(str(output_dir / f"frame_{frame_idx + 1:06d}.png"), frame)
        frame_idx += 1
        
        if progress_callback and total_frames > 0:
            progress_callback(
                frame_idx / total_frames,
                f"Extracting frame {frame_idx}/{total_frames}"
            )
    
    cap.release()


# ============================================================================
# Video Assembly
# ============================================================================

def assemble_video(
    frames_dir: Path,
    output_path: Path,
    fps: float,
    frame_pattern: str = "frame_%06d.png",
    crf: int = 18,
) -> Dict[str, Any]:
    """
    Assemble frames into MP4 video.
    
    Uses FFmpeg when available (libx264, yuv420p), OpenCV as fallback.
    
    Args:
        frames_dir: Directory containing frame PNGs
        output_path: Output .mp4 path
        fps: Target framerate
        frame_pattern: Frame filename pattern for FFmpeg
        crf: Constant Rate Factor (lower = higher quality, 18 is visually lossless)
        
    Returns:
        Dict with: output_path, frame_count, fps, duration
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frame_count = len(list(frames_dir.glob("frame_*.png")))
    ffmpeg = find_ffmpeg()
    
    if ffmpeg:
        _assemble_video_ffmpeg(frames_dir, output_path, fps, ffmpeg, frame_pattern, crf)
    else:
        _assemble_video_opencv(frames_dir, output_path, fps)
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[video_utils] Assembled {frame_count} frames -> {output_path.name} ({size_mb:.1f}MB)")
    else:
        print(f"[video_utils] WARNING: Output video not found at {output_path}")
    
    return {
        "output_path": str(output_path),
        "frame_count": frame_count,
        "fps": fps,
        "duration": frame_count / fps if fps else 0,
    }


def _assemble_video_ffmpeg(
    frames_dir: Path,
    output_path: Path,
    fps: float,
    ffmpeg: Path,
    frame_pattern: str,
    crf: int,
):
    """Assemble using FFmpeg"""
    cmd = [
        str(ffmpeg),
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / frame_pattern),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[video_utils] FFmpeg assembly error: {result.stderr[-500:]}")


def _assemble_video_opencv(frames_dir: Path, output_path: Path, fps: float):
    """Assemble using OpenCV (fallback)"""
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise ValueError(f"No frame_*.png files in {frames_dir}")
    
    first_frame = cv2.imread(str(frames[0]))
    h, w = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)
    
    writer.release()


# ============================================================================
# Thumbnail Extraction
# ============================================================================

def extract_thumbnail(
    video_path: Path,
    output_path: Optional[Path] = None,
    quality: int = 5,
) -> Optional[Path]:
    """
    Extract first frame as thumbnail image.
    
    Args:
        video_path: Input video
        output_path: Output thumbnail path (default: video_path.thumb.jpg)
        quality: JPEG quality (2-31, lower is better)
        
    Returns:
        Path to thumbnail if successful, None otherwise
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix('.thumb.jpg')
    else:
        output_path = Path(output_path)
    
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(f"[video_utils] Cannot extract thumbnail - FFmpeg not found")
        return None
    
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", str(quality),
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[video_utils] Thumbnail extraction failed: {result.stderr}")
        return None
    
    if output_path.exists():
        return output_path
    
    return None


# ============================================================================
# Color Palette (Golden Ratio)
# ============================================================================

def generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct RGB colors using golden ratio spacing.
    
    Used by crypto matte processors for consistent per-object coloring.
    Saturation and value are varied slightly to maximize visual distinction.
    
    Args:
        num_colors: Number of distinct colors to generate
        
    Returns:
        List of (R, G, B) tuples in 0-255 range
    """
    GOLDEN_RATIO = 0.618033988749895
    colors = []
    
    for i in range(num_colors):
        hue = (i * GOLDEN_RATIO) % 1.0
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.9 - (i % 4) * 0.05
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors


# ============================================================================
# Colormap Application (shared by depth processors)
# ============================================================================

def apply_depth_greyscale(
    depth: np.ndarray,
    range_min: float = 0.0,
    range_max: float = 1.0,
) -> np.ndarray:
    """
    Convert normalized [0,1] depth map to greyscale with optional range remapping.
    
    Always outputs greyscale — colormaps distort results when used as
    control inputs for generation models.
    
    Range remapping clips and rescales:
      - Values below range_min become black (0)
      - Values above range_max become white (255)
      - Values between are linearly stretched to fill 0-255
    
    Args:
        depth: Normalized float depth array [0, 1]
        range_min: Low end of range remap (default 0.0)
        range_max: High end of range remap (default 1.0)
                  
    Returns:
        BGR uint8 image suitable for cv2.imwrite()
    """
    # Apply range remapping
    if range_min != 0.0 or range_max != 1.0:
        # Clamp range_max > range_min to avoid division by zero
        effective_range = max(range_max - range_min, 1e-6)
        depth = (np.clip(depth, range_min, range_max) - range_min) / effective_range
    
    depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)


# Keep old name as alias for backward compatibility during transition
def apply_depth_colormap(
    depth: np.ndarray,
    colormap: Optional[str] = None,
    range_min: float = 0.0,
    range_max: float = 1.0,
) -> np.ndarray:
    """Backward-compatible wrapper — ignores colormap, always greyscale."""
    return apply_depth_greyscale(depth, range_min=range_min, range_max=range_max)