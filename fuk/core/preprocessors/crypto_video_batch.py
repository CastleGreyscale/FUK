# core/preprocessors/crypto_video_batch.py
"""
Video Crypto Matte Batch Processing using SAM2 Video Predictor

This module provides temporally-consistent crypto matte generation for videos.
Unlike frame-by-frame processing which causes color flickering (objects get
reassigned IDs each frame), this uses SAM2's video predictor to TRACK objects
across frames with consistent IDs.

Workflow:
1. Extract video frames
2. Detect all objects on frame 0 using SAM2AutomaticMaskGenerator
3. Initialize SAM2VideoPredictor with the video
4. Add detected objects as prompts (bounding boxes) on frame 0
5. Propagate through entire video - IDs stay consistent!
6. Render ID mattes with consistent colors per object
7. Assemble output

Usage in video_endpoints.py:
    if request.method == "crypto":
        batch_processor = VideoCryptoBatchProcessor()
        result = batch_processor.process_video_batch(...)
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import colorsys
import torch


class VideoCryptoBatchProcessor:
    """
    Batch processor for video crypto matte generation with temporal consistency.
    
    Uses SAM2VideoPredictor to track objects across frames instead of
    re-detecting each frame (which causes color flickering).
    """
    
    def __init__(self, ffmpeg_path: Optional[Path] = None):
        """Initialize with optional FFmpeg path"""
        self.ffmpeg_path = ffmpeg_path or self._find_ffmpeg()
        self.sam2_model = None
        self.video_predictor = None
        self.mask_generator = None
        self._sam_version = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    def _initialize_sam2(self, model_size: str = "large"):
        """Initialize SAM2 models (video predictor + mask generator)"""
        if self.video_predictor is not None:
            return  # Already initialized
        
        print(f"[CryptoBatch] Initializing SAM2 video predictor...")
        
        # Add vendor path
        import sys
        vendor_path = Path(__file__).parent.parent.parent / "vendor" / "segment-anything-2"
        if vendor_path.exists():
            sys.path.insert(0, str(vendor_path))
        
        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            # Find checkpoint and config
            checkpoint_path, config_name = self._find_sam2_checkpoint(model_size)
            
            if not checkpoint_path:
                raise FileNotFoundError(f"SAM2 checkpoint not found for {model_size}")
            
            # Detect version
            if "sam2.1" in checkpoint_path.name:
                self._sam_version = "sam2.1"
            else:
                self._sam_version = "sam2"
            
            print(f"[CryptoBatch] Version: {self._sam_version}")
            print(f"[CryptoBatch] Checkpoint: {checkpoint_path}")
            print(f"[CryptoBatch] Config: {config_name}")
            
            # Build the video predictor
            self.video_predictor = build_sam2_video_predictor(
                config_file=config_name,
                ckpt_path=str(checkpoint_path),
                device=self._device,
            )
            
            # Also build the base model for automatic mask generation
            sam2_model = build_sam2(
                config_file=config_name,
                ckpt_path=str(checkpoint_path),
                device=self._device,
            )
            
            # Create automatic mask generator for initial object detection
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                min_mask_region_area=100,
            )
            
            print(f"[CryptoBatch] ✓ SAM2 video predictor initialized")
            
        except ImportError as e:
            raise ImportError(
                f"SAM2 not installed. Setup:\n"
                f"  git clone https://github.com/facebookresearch/segment-anything-2 vendor/segment-anything-2\n"
                f"  cd vendor/segment-anything-2\n"
                f"  pip install -e . --break-system-packages\n"
                f"Error: {e}"
            )
    
    def _find_sam2_checkpoint(self, model_size: str) -> Tuple[Optional[Path], Optional[str]]:
        """Find SAM2 checkpoint and config for given model size"""
        
        # Mapping for checkpoint names (prefer 2.1)
        checkpoint_names = {
            "tiny": ["sam2.1_hiera_tiny.pt", "sam2_hiera_tiny.pt"],
            "small": ["sam2.1_hiera_small.pt", "sam2_hiera_small.pt"],
            "base": ["sam2.1_hiera_base_plus.pt", "sam2_hiera_base_plus.pt"],
            "large": ["sam2.1_hiera_large.pt", "sam2_hiera_large.pt"],
        }
        
        # Config names for Hydra
        config_names = {
            "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        
        # Search directories
        search_dirs = [
            Path.home() / "ai" / "models" / "sam",
            Path(__file__).parent.parent.parent / "vendor" / "segment-anything-2" / "checkpoints",
            Path.home() / "ai" / "models" / "sam2",
        ]
        
        names_to_find = checkpoint_names.get(model_size, checkpoint_names["large"])
        config_name = config_names.get(model_size, config_names["large"])
        
        # Search for checkpoint
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ckpt_name in names_to_find:
                path = search_dir / ckpt_name
                if path.exists():
                    return path, config_name
        
        return None, None
    
    def process_video_batch(
        self,
        video_path: Path,
        output_path: Path,
        model_size: str = "large",
        max_objects: int = 50,
        min_area: int = 500,
        output_mode: str = "mp4",
    ) -> Dict[str, Any]:
        """
        Process entire video with SAM2 video predictor for consistent crypto mattes.
        
        Args:
            video_path: Input video file
            output_path: Output video file or directory
            model_size: SAM2 model size (tiny/small/base/large)
            max_objects: Maximum number of objects to track
            min_area: Minimum mask area in pixels
            output_mode: 'mp4' or 'sequence'
            
        Returns:
            Processing result dict
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        print(f"\n[CryptoBatch] ===== VIDEO CRYPTO MATTE PROCESSING =====")
        print(f"[CryptoBatch] Model: SAM2 {model_size}")
        print(f"[CryptoBatch] Input: {video_path}")
        print(f"[CryptoBatch] Output: {output_path}")
        print(f"[CryptoBatch] Max objects: {max_objects}")
        
        # Initialize SAM2
        self._initialize_sam2(model_size)
        
        # Get video info
        video_info = self.get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["frame_count"]
        width = video_info["width"]
        height = video_info["height"]
        
        print(f"[CryptoBatch] Video: {total_frames} frames @ {fps:.2f}fps, {width}x{height}")
        
        # Create temp directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            output_frames_dir = temp_path / "output_frames"
            
            input_frames_dir.mkdir()
            output_frames_dir.mkdir()
            
            # Step 1: Extract frames from video
            print(f"[CryptoBatch] Extracting frames...")
            self._extract_frames(video_path, input_frames_dir)
            
            frame_paths = sorted(input_frames_dir.glob("frame_*.png"))
            print(f"[CryptoBatch] Extracted {len(frame_paths)} frames")
            
            if not frame_paths:
                raise ValueError("No frames extracted from video")
            
            # Step 2: Detect objects on first frame using AutomaticMaskGenerator
            print(f"[CryptoBatch] Detecting objects on frame 0...")
            first_frame = cv2.imread(str(frame_paths[0]))
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            
            masks = self.mask_generator.generate(first_frame_rgb)
            
            # Filter and sort masks
            masks = [m for m in masks if m['area'] >= min_area]
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:max_objects]
            
            num_objects = len(masks)
            print(f"[CryptoBatch] Detected {num_objects} objects on frame 0")
            
            if num_objects == 0:
                print(f"[CryptoBatch] WARNING: No objects detected!")
                # Create empty output
                return self._create_empty_output(
                    frame_paths, output_frames_dir, output_path, fps, output_mode, (height, width)
                )
            
            # Step 3: Initialize video predictor with the video frames directory
            print(f"[CryptoBatch] Initializing video tracking state...")
            
            # SAM2 video predictor expects a directory of JPEG frames
            # Convert our PNGs to JPEG naming convention
            jpeg_frames_dir = temp_path / "jpeg_frames"
            jpeg_frames_dir.mkdir()
            
            for i, frame_path in enumerate(frame_paths):
                # SAM2 expects frames named as 00000.jpg, 00001.jpg, etc.
                frame = cv2.imread(str(frame_path))
                jpeg_path = jpeg_frames_dir / f"{i:05d}.jpg"
                cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Initialize inference state
            with torch.inference_mode(), torch.autocast(self._device, dtype=torch.bfloat16):
                inference_state = self.video_predictor.init_state(video_path=str(jpeg_frames_dir))
                
                # Step 4: Add object prompts from detected masks on frame 0
                print(f"[CryptoBatch] Adding {num_objects} object prompts...")
                
                for obj_id, mask_data in enumerate(masks):
                    # Get bounding box for this mask
                    bbox = mask_data['bbox']  # [x, y, w, h] format from SAM2
                    # Convert to [x1, y1, x2, y2] format
                    box = np.array([
                        bbox[0],
                        bbox[1],
                        bbox[0] + bbox[2],
                        bbox[1] + bbox[3]
                    ])
                    
                    # Add bounding box prompt for this object
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=box,
                    )
                
                # Step 5: Propagate through entire video
                print(f"[CryptoBatch] Propagating masks through video...")
                
                # Store all masks per frame
                video_segments = {}
                
                for frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                    # out_mask_logits: [num_objects, 1, H, W]
                    video_segments[frame_idx] = {
                        obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                        for i, obj_id in enumerate(out_obj_ids)
                    }
                    
                    if (frame_idx + 1) % 20 == 0:
                        print(f"[CryptoBatch]   Propagated to frame {frame_idx + 1}/{len(frame_paths)}")
                
                print(f"[CryptoBatch] Propagation complete - {len(video_segments)} frames processed")
            
            # Step 6: Generate consistent ID mattes for each frame
            print(f"[CryptoBatch] Generating ID matte frames...")
            
            # Generate consistent color palette for all objects
            colors = self._generate_color_palette(num_objects)
            
            # Also build raw ID matte array for lossless EXR export
            # Use uint16 to support up to 65535 objects
            raw_id_mattes = np.zeros((len(frame_paths), height, width), dtype=np.uint16)
            
            for frame_idx, frame_path in enumerate(frame_paths):
                if frame_idx not in video_segments:
                    # Frame not processed (shouldn't happen), create empty
                    id_matte = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    # Create ID matte from tracked masks
                    id_matte = self._create_id_matte_from_segments(
                        video_segments[frame_idx],
                        colors,
                        (height, width)
                    )
                    
                    # Build raw ID matte (object ID per pixel)
                    for obj_id, mask in video_segments[frame_idx].items():
                        # Resize mask if needed
                        if mask.shape[:2] != (height, width):
                            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                            mask = mask.astype(bool)
                        # Object IDs are 1-indexed (0 = background)
                        raw_id_mattes[frame_idx][mask] = obj_id + 1
                
                # Save frame
                output_frame_path = output_frames_dir / frame_path.name
                cv2.imwrite(str(output_frame_path), cv2.cvtColor(id_matte, cv2.COLOR_RGB2BGR))
                
                if (frame_idx + 1) % 20 == 0:
                    print(f"[CryptoBatch]   Rendered {frame_idx + 1}/{len(frame_paths)} frames")
            
            print(f"[CryptoBatch] ID matte generation complete")
            
            # Step 7: Assemble output
            if output_mode == "sequence":
                print(f"[CryptoBatch] Saving as image sequence to {output_path}")
                output_path.mkdir(parents=True, exist_ok=True)
                
                for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                    shutil.copy(frame_path, output_path / frame_path.name)
                
                # SAVE RAW CRYPTO DATA for lossless EXR export
                raw_crypto_path = output_path / "crypto_raw.npy"
                np.save(str(raw_crypto_path), raw_id_mattes)
                print(f"[CryptoBatch] Saved raw crypto data: {raw_crypto_path} (shape: {raw_id_mattes.shape})")
                
                frames_list = [f.name for f in sorted(output_path.glob("*.png"))]
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": True,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "num_objects": num_objects,
                    "sam_version": self._sam_version,
                    "first_frame": frames_list[0] if frames_list else None,
                    "frames": frames_list,
                    "raw_data_path": str(raw_crypto_path),
                }
            else:
                print(f"[CryptoBatch] Assembling MP4 to {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self._assemble_video(output_frames_dir, output_path, fps)
                
                # SAVE RAW CRYPTO DATA alongside MP4 for lossless EXR export
                raw_crypto_path = output_path.parent / f"{output_path.stem}_raw.npy"
                np.save(str(raw_crypto_path), raw_id_mattes)
                print(f"[CryptoBatch] Saved raw crypto data: {raw_crypto_path} (shape: {raw_id_mattes.shape})")
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": False,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "num_objects": num_objects,
                    "sam_version": self._sam_version,
                    "raw_data_path": str(raw_crypto_path),
                }
    
    def _generate_color_palette(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate visually distinct colors using golden ratio"""
        colors = []
        golden_ratio = 0.618033988749895
        
        for i in range(num_colors):
            hue = (i * golden_ratio) % 1.0
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.9 - (i % 4) * 0.05
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
        return colors
    
    def _create_id_matte_from_segments(
        self,
        segments: Dict[int, np.ndarray],
        colors: List[Tuple[int, int, int]],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create ID matte from tracked segments with consistent colors"""
        h, w = shape
        id_matte = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply masks in order (larger objects first, but object ID determines color)
        for obj_id, mask in segments.items():
            if obj_id < len(colors):
                color = colors[obj_id]
                # Resize mask if needed (SAM2 might output at different resolution)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(bool)
                id_matte[mask] = color
        
        return id_matte
    
    def _create_empty_output(
        self,
        frame_paths: List[Path],
        output_frames_dir: Path,
        output_path: Path,
        fps: float,
        output_mode: str,
        shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Create empty output when no objects detected"""
        h, w = shape
        empty_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        for frame_path in frame_paths:
            output_frame_path = output_frames_dir / frame_path.name
            cv2.imwrite(str(output_frame_path), empty_frame)
        
        if output_mode == "sequence":
            output_path.mkdir(parents=True, exist_ok=True)
            for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                shutil.copy(frame_path, output_path / frame_path.name)
            
            return {
                "output_path": str(output_path),
                "is_sequence": True,
                "frame_count": len(frame_paths),
                "fps": fps,
                "num_objects": 0,
                "first_frame": sorted(output_path.glob("*.png"))[0].name if list(output_path.glob("*.png")) else None,
            }
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._assemble_video(output_frames_dir, output_path, fps)
            
            return {
                "output_path": str(output_path),
                "is_sequence": False,
                "frame_count": len(frame_paths),
                "fps": fps,
                "num_objects": 0,
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
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.video_predictor is not None:
            del self.video_predictor
            self.video_predictor = None
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None
        if self.sam2_model is not None:
            del self.sam2_model
            self.sam2_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[CryptoBatch] Cleaned up GPU memory")