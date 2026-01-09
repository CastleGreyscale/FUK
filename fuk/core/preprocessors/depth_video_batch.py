# core/preprocessors/depth_video_batch.py
"""
Batch Video Depth Processing for Depth Anything V3

This module provides batch-aware depth processing for videos to maintain
temporal consistency. Unlike frame-by-frame processing, this passes ALL
frames to DA3 at once with ref_view_strategy="middle" for coherence.

Usage in video_endpoints.py:
    if depth_model in DA3_MODELS and use_batch_processing:
        result = process_depth_video_batch(...)
    else:
        result = video_processor.process_video(...)  # Frame-by-frame
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


class VideoDepthBatchProcessor:
    """
    Batch processor for video depth estimation with DA3
    
    Maintains temporal consistency by processing all frames together
    instead of one-at-a-time.
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
    ) -> Dict[str, Any]:
        """
        Process entire video with DA3 batch inference
        
        Args:
            video_path: Input video file
            output_path: Output video file or directory
            depth_model: Initialized depth model instance
            model_type: Type of depth model being used
            invert: Invert depth values
            normalize: Normalize to 0-1
            colormap: Colormap to apply
            process_res: DA3 processing resolution
            process_res_method: DA3 resize method
            output_mode: 'mp4' or 'sequence'
            
        Returns:
            Processing result dict
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        print(f"\n[DepthBatch] ===== BATCH VIDEO DEPTH PROCESSING =====")
        print(f"[DepthBatch] Model: {model_type.value}")
        print(f"[DepthBatch] Input: {video_path}")
        print(f"[DepthBatch] Output: {output_path}")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["frame_count"]
        
        print(f"[DepthBatch] Frames: {total_frames} @ {fps:.2f}fps")
        
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
            
            # Step 2: BATCH INFERENCE - Pass all frames to DA3 at once
            print(f"[DepthBatch] Running batch inference with ref_view_strategy='middle'...")
            
            # Ensure model is initialized (lazy loading)
            print(f"[DepthBatch] Initializing depth model...")
            depth_model._ensure_initialized()
            print(f"[DepthBatch] Model initialized: {depth_model.model is not None}")
            
            # Convert frame paths to strings for DA3 API
            frame_path_strs = [str(p) for p in frame_paths]
            
            # Call DA3 with all frames at once for temporal consistency
            try:
                from depth_anything_3.api import DepthAnything3
                
                # DA3 batch inference with temporal coherence
                prediction = depth_model.model.inference(
                    image=frame_path_strs,
                    process_res=process_res,
                    process_res_method=process_res_method,
                    ref_view_strategy="middle",  # KEY: Temporal consistency
                )
                
                print(f"[DepthBatch] Batch inference complete - got {prediction.depth.shape[0]} depth maps")
                
                # Step 3: Post-process and save each depth map
                print(f"[DepthBatch] Post-processing depth maps...")
                
                for i, (frame_path, depth_map) in enumerate(zip(frame_paths, prediction.depth)):
                    output_frame_path = output_frames_dir / frame_path.name
                    
                    # Post-process depth map
                    depth_processed = self._postprocess_depth(
                        depth_map,
                        normalize=normalize,
                        invert=invert,
                        colormap=colormap,
                    )
                    
                    # Save
                    cv2.imwrite(str(output_frame_path), depth_processed)
                    
                    if (i + 1) % 10 == 0:
                        print(f"[DepthBatch]   Processed {i+1}/{len(frame_paths)} frames")
                
                print(f"[DepthBatch] Post-processing complete")
                
            except Exception as e:
                print(f"[DepthBatch] ERROR during batch inference: {e}")
                raise
            
            # Step 4: Assemble output
            if output_mode == "sequence":
                # Copy frames to output directory
                print(f"[DepthBatch] Saving as image sequence to {output_path}")
                output_path.mkdir(parents=True, exist_ok=True)
                
                for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                    shutil.copy(frame_path, output_path / frame_path.name)
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": True,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "first_frame": sorted(output_path.glob("*.png"))[0].name if output_path.glob("*.png") else None,
                }
            else:
                # Assemble to MP4
                print(f"[DepthBatch] Assembling MP4 to {output_path}")
                self._assemble_video(output_frames_dir, output_path, fps)
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": False,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                }
    
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
        """Post-process depth map (normalize, invert, colormap)"""
        
        # Normalize
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Invert
        if invert:
            depth = 1.0 - depth
        
        # Apply colormap or grayscale
        if colormap:
            depth_uint8 = (depth * 255).astype(np.uint8)
            colormap_func = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
            output_image = cv2.applyColorMap(depth_uint8, colormap_func)
        else:
            depth_uint8 = (depth * 255).astype(np.uint8)
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