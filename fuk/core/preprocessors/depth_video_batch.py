# core/preprocessors/depth_video_batch.py
"""
Batch Video Depth Processing for Depth Anything V3

FIXED: Now uses GLOBAL normalization across all frames to prevent jitter.
The original code normalized each frame independently, causing temporal flicker.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from enum import Enum

# Import the depth preprocessor models
from fuk.core.preprocessors.depth import DepthModel, DA3_MODEL_IDS


# Default DA3 settings for video (optimized for temporal consistency)
DA3_VIDEO_DEFAULTS = {
    "process_res": 1344,  # Match config recommendation
    "process_res_method": "lower_bound_resize",  # Preserves aspect ratio
    "ref_view_strategy": "middle",  # Temporal consistency
}

# Chunking settings (based on DA3-Streaming recommendations)
# Process videos in chunks to avoid memory corruption
CHUNK_SETTINGS = {
    "max_chunk_size": 60,  # Max frames per chunk (DA3-Streaming uses 60-120)
    "overlap": 10,         # Frames overlap between chunks for blending
    "blend_frames": 5,     # Frames to blend at chunk boundaries
}


class VideoDepthBatchProcessor:
    """
    Batch processor for video depth estimation with DA3
    
    Maintains temporal consistency by:
    1. Processing all frames together (batch inference)
    2. Using GLOBAL normalization across all frames (not per-frame)
    3. Optional temporal smoothing
    """
    
    def __init__(self, ffmpeg_path: Optional[Path] = None):
        """Initialize with optional FFmpeg path"""
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
    
    def _find_ffmpeg(self) -> Optional[Path]:
        """Find ffmpeg binary"""
        import shutil as sh
        result = sh.which("ffmpeg")
        return Path(result) if result else None
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata"""
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
    
    def process_video_batch(
        self,
        video_path: Path,
        output_path: Path,
        depth_model,  # Already initialized DepthPreprocessor instance
        model_type: DepthModel,
        invert: bool = False,
        normalize: bool = True,
        colormap: Optional[str] = "inferno",
        process_res: int = 756,
        process_res_method: str = "lower_bound_resize",
        output_mode: str = "mp4",
        temporal_smooth: int = 3,  # NEW: temporal smoothing window (0=off, try 3)
    ) -> Dict[str, Any]:
        """
        Process entire video with DA3 batch inference
        
        Args:
            video_path: Input video file
            output_path: Output video file or directory
            depth_model: Initialized depth model instance
            model_type: Type of depth model being used
            invert: Invert depth values
            normalize: Normalize to 0-1 (now uses GLOBAL min/max)
            colormap: Colormap to apply
            process_res: DA3 processing resolution
            process_res_method: DA3 resize method
            output_mode: 'mp4' or 'sequence'
            temporal_smooth: Temporal median filter window (0=off)
            
        Returns:
            Processing result dict
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        print(f"\n[DepthBatch] ===== BATCH VIDEO DEPTH PROCESSING =====")
        print(f"[DepthBatch] Model: {model_type.value}")
        print(f"[DepthBatch] Input: {video_path}")
        print(f"[DepthBatch] Output: {output_path}")
        print(f"[DepthBatch] Temporal smoothing: {temporal_smooth if temporal_smooth > 0 else 'off'}")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["frame_count"]
        original_size = (video_info["width"], video_info["height"])
        
        print(f"[DepthBatch] Frames: {total_frames} @ {fps:.2f}fps")
        print(f"[DepthBatch] Original size: {original_size[0]}x{original_size[1]}")
        
        # Create temp directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            output_frames_dir = temp_path / "output_frames"
            
            input_frames_dir.mkdir()
            output_frames_dir.mkdir()
            
            # Step 1: Extract frames from video
            print(f"[DepthBatch] Extracting frames...")
            self._extract_frames(video_path, input_frames_dir)
            
            # Get all frame paths
            frame_paths = sorted(input_frames_dir.glob("frame_*.png"))
            print(f"[DepthBatch] Extracted {len(frame_paths)} frames")
            
            if not frame_paths:
                raise ValueError("No frames extracted from video")
            
            # Step 2: BATCH INFERENCE
            print(f"[DepthBatch] Running batch inference...")
            
            # Ensure model is initialized (lazy loading)
            print(f"[DepthBatch] Initializing depth model...")
            depth_model._ensure_initialized()
            print(f"[DepthBatch] Model initialized: {depth_model.model is not None}")
            
            # Convert frame paths to strings for DA3 API
            frame_path_strs = [str(p) for p in frame_paths]
            
            try:
                # DA3 batch inference
                prediction = depth_model.model.inference(
                    image=frame_path_strs,
                    process_res=process_res,
                    process_res_method=process_res_method,
                )
                
                # Get all depth maps as numpy array [N, H, W]
                all_depths = prediction.depth
                print(f"[DepthBatch] Inference complete - shape: {all_depths.shape}")
                
                # Step 3: GLOBAL NORMALIZATION (key fix for jitter)
                if normalize:
                    global_min = all_depths.min()
                    global_max = all_depths.max()
                    print(f"[DepthBatch] Global depth range: [{global_min:.4f}, {global_max:.4f}]")
                    
                    # Normalize ALL frames using the SAME min/max
                    all_depths = (all_depths - global_min) / (global_max - global_min + 1e-8)
                
                # Step 4: Optional temporal smoothing
                if temporal_smooth > 1:
                    print(f"[DepthBatch] Applying temporal smoothing (window={temporal_smooth})...")
                    all_depths = self._temporal_smooth(all_depths, window=temporal_smooth)
                
                # Step 5: Invert if requested
                if invert:
                    all_depths = 1.0 - all_depths
                
                # Step 6: Resize and colorize each frame
                print(f"[DepthBatch] Post-processing depth maps...")
                
                for i, (frame_path, depth_map) in enumerate(zip(frame_paths, all_depths)):
                    output_frame_path = output_frames_dir / frame_path.name
                    
                    # Resize to original video dimensions if needed
                    if depth_map.shape[:2] != (original_size[1], original_size[0]):
                        depth_map = cv2.resize(
                            depth_map, 
                            original_size,  # (width, height)
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Apply colormap (depth is already normalized)
                    output_image = self._apply_colormap(depth_map, colormap)
                    
                    # Save
                    cv2.imwrite(str(output_frame_path), output_image)
                    
                    if (i + 1) % 10 == 0:
                        print(f"[DepthBatch]   Processed {i+1}/{len(frame_paths)} frames")
                
                print(f"[DepthBatch] Post-processing complete")
                
            except Exception as e:
                print(f"[DepthBatch] ERROR during batch inference: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Step 7: Assemble output
            if output_mode == "sequence":
                print(f"[DepthBatch] Saving as image sequence to {output_path}")
                output_path.mkdir(parents=True, exist_ok=True)
                
                for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                    shutil.copy(frame_path, output_path / frame_path.name)
                
                # SAVE RAW DEPTH DATA for lossless EXR export
                raw_depth_path = output_path / "depth_raw.npy"
                np.save(str(raw_depth_path), all_depths.astype(np.float32))
                print(f"[DepthBatch] Saved raw depth data: {raw_depth_path}")
                
                frames = sorted([f.name for f in output_path.glob("*.png")])
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": True,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "first_frame": frames[0] if frames else None,
                    "frames": frames,
                    "raw_data_path": str(raw_depth_path),
                }
            else:
                # Assemble to MP4
                print(f"[DepthBatch] Assembling MP4 to {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self._assemble_video(output_frames_dir, output_path, fps)
                
                # SAVE RAW DEPTH DATA alongside MP4 for lossless EXR export
                raw_depth_path = output_path.parent / f"{output_path.stem}_raw.npy"
                np.save(str(raw_depth_path), all_depths.astype(np.float32))
                print(f"[DepthBatch] Saved raw depth data: {raw_depth_path}")
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": False,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "raw_data_path": str(raw_depth_path),
                }
    
    def _temporal_smooth(self, depths: np.ndarray, window: int = 3) -> np.ndarray:
        """
        Apply temporal median filter to reduce frame-to-frame noise
        
        Args:
            depths: [N, H, W] depth array
            window: Window size (must be odd, e.g., 3, 5)
        
        Returns:
            Smoothed depth array
        """
        if window <= 1:
            return depths
        
        n_frames = len(depths)
        half_window = window // 2
        smoothed = np.zeros_like(depths)
        
        for i in range(n_frames):
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window + 1)
            smoothed[i] = np.median(depths[start:end], axis=0)
        
        return smoothed
    
    def _apply_colormap(self, depth: np.ndarray, colormap: Optional[str] = "inferno") -> np.ndarray:
        """Apply colormap to normalized depth [0,1]"""
        depth_uint8 = (depth * 255).astype(np.uint8)
        
        if colormap:
            colormap_func = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
            return cv2.applyColorMap(depth_uint8, colormap_func)
        else:
            return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
    
    def _extract_frames(self, video_path: Path, output_dir: Path):
        """Extract frames from video using FFmpeg or OpenCV"""
        if self.ffmpeg_path:
            import subprocess
            cmd = [
                str(self.ffmpeg_path),
                "-i", str(video_path),
                "-vsync", "0",
                "-q:v", "2",
                str(output_dir / "frame_%06d.png")
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # OpenCV fallback
            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.png"), frame)
                frame_idx += 1
            cap.release()
    
    def _postprocess_depth(
        self,
        depth: np.ndarray,
        normalize: bool = True,
        invert: bool = False,
        colormap: Optional[str] = "inferno",
    ) -> np.ndarray:
        """
        Post-process depth map (invert, colormap)
        
        NOTE: Normalization is done GLOBALLY before this is called,
        so depth is already in [0, 1] range.
        """
        # Invert if requested
        if invert:
            depth = 1.0 - depth
        
        # Convert to uint8
        depth_uint8 = (depth * 255).astype(np.uint8)
        
        # Apply colormap or grayscale
        if colormap:
            colormap_func = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
            output_image = cv2.applyColorMap(depth_uint8, colormap_func)
        else:
            output_image = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        
        return output_image
    
    def _assemble_video(self, frames_dir: Path, output_path: Path, fps: float):
        """Assemble frames into MP4 video"""
        if self.ffmpeg_path:
            import subprocess
            cmd = [
                str(self.ffmpeg_path),
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-y",
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # OpenCV fallback
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