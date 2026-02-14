# core/preprocessors/crypto.py
"""
Cryptomatte / Instance Segmentation Preprocessor

Uses SAM2/SAM2.1 (Segment Anything Model 2) for instance segmentation,
then converts to cryptomatte-style ID mattes for compositing.

Handles both single images and video:
  - process()       -> single image segmentation via AutomaticMaskGenerator
  - process_video() -> temporally-consistent tracking via SAM2VideoPredictor

Good for:
- Per-object selection in comp
- Rotoscoping automation
- Object-aware effects
- VFX isolation passes
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
import cv2
import numpy as np
import torch

from .base import BasePreprocessor
from core.video_utils import (
    get_video_info, extract_frames, assemble_video,
    generate_color_palette, is_video_file,
)


class SAMModel(str, Enum):
    """Available SAM model sizes"""
    TINY = "sam2_hiera_tiny"
    SMALL = "sam2_hiera_small"
    BASE = "sam2_hiera_base_plus"
    LARGE = "sam2_hiera_large"


# ============================================================================
# SAM2 Model Discovery (single source of truth)
# ============================================================================

# Checkpoint filenames — prefer 2.1, fall back to 2.0
SAM_CHECKPOINT_NAMES = {
    SAMModel.TINY:  ["sam2.1_hiera_tiny.pt", "sam2_hiera_tiny.pt"],
    SAMModel.SMALL: ["sam2.1_hiera_small.pt", "sam2_hiera_small.pt"],
    SAMModel.BASE:  ["sam2.1_hiera_base_plus.pt", "sam2_hiera_base_plus.pt"],
    SAMModel.LARGE: ["sam2.1_hiera_large.pt", "sam2_hiera_large.pt"],
}

# Hydra config names (passed to SAM2 builder, not file paths)
SAM_CONFIG_NAMES = {
    SAMModel.TINY:  "configs/sam2.1/sam2.1_hiera_t.yaml",
    SAMModel.SMALL: "configs/sam2.1/sam2.1_hiera_s.yaml",
    SAMModel.BASE:  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    SAMModel.LARGE: "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# Standard mask generator settings (shared between image and video paths)
MASK_GENERATOR_DEFAULTS = {
    "points_per_side": 32,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "min_mask_region_area": 100,
}

# Map string sizes to enum (for API convenience)
SIZE_STRING_MAP = {
    "tiny": SAMModel.TINY,
    "small": SAMModel.SMALL,
    "base": SAMModel.BASE,
    "large": SAMModel.LARGE,
    "sam2": SAMModel.LARGE,
    "sam2_large": SAMModel.LARGE,
    "sam2_base": SAMModel.BASE,
    "sam2_small": SAMModel.SMALL,
    "sam2_tiny": SAMModel.TINY,
}


def resolve_model_size(size) -> SAMModel:
    """Convert string or SAMModel to SAMModel enum"""
    if isinstance(size, SAMModel):
        return size
    return SIZE_STRING_MAP.get(str(size).lower(), SAMModel.LARGE)


def find_sam_checkpoint(
    model_size: SAMModel,
    vendor_path: Optional[Path] = None,
) -> Tuple[Optional[Path], str]:
    """
    Find SAM2 checkpoint file and config name.
    
    Searches in order:
    1. User's ~/ai/models/sam/ directory
    2. Vendor segment-anything-2/checkpoints/
    3. General model directories
    
    Returns:
        (checkpoint_path, config_name) — checkpoint_path is None if not found
    """
    checkpoint_names = SAM_CHECKPOINT_NAMES.get(model_size, [])
    config_name = SAM_CONFIG_NAMES.get(model_size, SAM_CONFIG_NAMES[SAMModel.LARGE])
    
    # Build search directories
    search_dirs = [
        Path.home() / "ai" / "models" / "sam",
    ]
    
    if vendor_path and vendor_path.exists():
        search_dirs.append(vendor_path / "checkpoints")
        search_dirs.append(vendor_path)
    else:
        # Default vendor location
        default_vendor = Path(__file__).parent.parent.parent / "vendor" / "segment-anything-2"
        if default_vendor.exists():
            search_dirs.append(default_vendor / "checkpoints")
            search_dirs.append(default_vendor)
    
    search_dirs.extend([
        Path.home() / "ai" / "models" / "sam2",
        Path.home() / "ai" / "models" / "checkpoints",
    ])
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ckpt_name in checkpoint_names:
            path = search_dir / ckpt_name
            if path.exists():
                return path, config_name
    
    print(f"✗ SAM2 checkpoint not found for {model_size.value}")
    print("  Download from: https://github.com/facebookresearch/segment-anything-2#download-checkpoints")
    print("  Searched for:", checkpoint_names)
    return None, config_name


def detect_sam_version(checkpoint_path: Path) -> str:
    """Detect SAM version from checkpoint filename"""
    return "sam2.1" if "sam2.1" in checkpoint_path.name else "sam2"


# ============================================================================
# CryptoPreprocessor — unified image + video
# ============================================================================

class CryptoPreprocessor(BasePreprocessor):
    """
    Instance segmentation using SAM2/SAM2.1, output as cryptomatte-style ID mattes.
    
    Handles both single images and video:
      - process()       -> single image, uses AutomaticMaskGenerator
      - process_video() -> temporal tracking via SAM2VideoPredictor
    
    Generates:
    - ID matte image (colored visualization)
    - Individual mask layers (optional)
    - Raw ID data for lossless EXR export (video)
    
    Requires vendor:
        git clone https://github.com/facebookresearch/segment-anything-2 vendor/segment-anything-2
    """
    
    def __init__(
        self,
        model_size: SAMModel = SAMModel.LARGE,
        config_path: Optional[Path] = None
    ):
        super().__init__(config_path)
        self.model_size = model_size
        
        # Image path models
        self.mask_generator = None
        
        # Video path models (lazy loaded only when needed)
        self.video_predictor = None
        
        self._sam_version = None
        self._vendor_path = None
        
        print(f"Crypto preprocessor initialized: {model_size.value} on {self.device}")
    
    def _initialize(self):
        """Load SAM2 model for image segmentation"""
        self._ensure_vendor_path()
        self._build_mask_generator()
    
    def _ensure_vendor_path(self):
        """Add SAM2 vendor to sys.path"""
        if self._vendor_path is not None:
            return
        
        self._vendor_path = self._get_vendor_path("segment-anything-2")
        if self._vendor_path.exists():
            import sys
            if str(self._vendor_path) not in sys.path:
                sys.path.insert(0, str(self._vendor_path))
    
    def _build_mask_generator(self):
        """Build SAM2AutomaticMaskGenerator for image segmentation"""
        if self.mask_generator is not None:
            return
        
        try:
            self._ensure_vendor_path()
            
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            checkpoint_path, config_name = find_sam_checkpoint(
                self.model_size, self._vendor_path
            )
            
            if not checkpoint_path:
                raise FileNotFoundError("SAM2 checkpoint not found")
            
            self._sam_version = detect_sam_version(checkpoint_path)
            
            print(f"[Crypto] Loading SAM2 {self.model_size.value}...")
            print(f"[Crypto]   Version: {self._sam_version}")
            print(f"[Crypto]   Checkpoint: {checkpoint_path}")
            
            sam2_model = build_sam2(
                config_file=config_name,
                ckpt_path=str(checkpoint_path),
                device=self.device,
            )
            
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                **MASK_GENERATOR_DEFAULTS,
            )
            
            print(f"✓ SAM2 ({self._sam_version}) mask generator loaded")
            
        except ImportError as e:
            raise ImportError(
                f"SAM2 not installed. Setup:\n"
                f"  git clone https://github.com/facebookresearch/segment-anything-2 vendor/segment-anything-2\n"
                f"  cd vendor/segment-anything-2\n"
                f"  pip install -e . --break-system-packages\n"
                f"Error: {e}"
            )
    
    def _build_video_predictor(self):
        """Build SAM2VideoPredictor for temporal tracking (lazy, only for video)"""
        if self.video_predictor is not None:
            return
        
        self._ensure_vendor_path()
        
        from sam2.build_sam import build_sam2_video_predictor
        
        checkpoint_path, config_name = find_sam_checkpoint(
            self.model_size, self._vendor_path
        )
        
        if not checkpoint_path:
            raise FileNotFoundError("SAM2 checkpoint not found for video predictor")
        
        self._sam_version = detect_sam_version(checkpoint_path)
        
        print(f"[Crypto] Loading SAM2 video predictor...")
        print(f"[Crypto]   Checkpoint: {checkpoint_path}")
        
        self.video_predictor = build_sam2_video_predictor(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=self.device,
        )
        
        print(f"✓ SAM2 video predictor loaded")
    
    # ========================================================================
    # Single Image Processing
    # ========================================================================
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        max_objects: int = 50,
        min_area: int = 500,
        output_mode: str = "id_matte",
        exact_output: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate instance segmentation / cryptomatte for a single image.
        
        Args:
            image_path: Input image
            output_path: Where to save result
            max_objects: Maximum number of objects to segment
            min_area: Minimum mask area in pixels
            output_mode: 'id_matte', 'layers', or 'both'
            exact_output: If True, write to exact output_path (for video frames)
        """
        self._ensure_initialized()
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        print(f"Segmenting image ({w}x{h}) with {self._sam_version or 'SAM2'}...")
        masks = self.mask_generator.generate(image_rgb)
        
        masks = [m for m in masks if m['area'] >= min_area]
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:max_objects]
        
        print(f"Found {len(masks)} objects")
        
        outputs = {}
        
        if output_mode in ['id_matte', 'both']:
            colors = generate_color_palette(len(masks))
            id_matte = self._create_id_matte(masks, (h, w), colors)
            
            params = {
                'method': 'crypto',
                'mode': 'id_matte',
                'num_objects': len(masks),
                'max_objects': max_objects,
                'sam_version': self._sam_version,
            }
            final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
            cv2.imwrite(str(final_output), cv2.cvtColor(id_matte, cv2.COLOR_RGB2BGR))
            outputs['id_matte'] = str(final_output)
        
        if output_mode in ['layers', 'both']:
            layers_dir = output_path.parent / f"{output_path.stem}_layers"
            layers_dir.mkdir(parents=True, exist_ok=True)
            
            layer_paths = []
            for i, mask in enumerate(masks):
                mask_image = (mask['segmentation'].astype(np.uint8) * 255)
                layer_path = layers_dir / f"object_{i:03d}.png"
                cv2.imwrite(str(layer_path), mask_image)
                layer_paths.append(str(layer_path))
            
            outputs['layers'] = layer_paths
            outputs['layers_dir'] = str(layers_dir)
        
        return {
            "output_path": outputs.get('id_matte', outputs.get('layers', [None])[0]),
            "method": "crypto",
            "num_objects": len(masks),
            "sam_version": self._sam_version,
            "outputs": outputs,
            "parameters": {
                'max_objects': max_objects,
                'min_area': min_area,
                'output_mode': output_mode,
            },
        }
    
    # ========================================================================
    # Video Processing (temporal tracking via SAM2VideoPredictor)
    # ========================================================================
    
    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        output_mode: str = "mp4",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process video with SAM2 video predictor for temporally-consistent crypto mattes.
        
        Workflow:
        1. Extract video frames
        2. Detect all objects on frame 0 using AutomaticMaskGenerator
        3. Initialize SAM2VideoPredictor with the video
        4. Add detected objects as prompts (bounding boxes) on frame 0
        5. Propagate through entire video — IDs stay consistent
        6. Render ID mattes with consistent colors per object
        7. Assemble output
        
        Args:
            video_path: Input video file
            output_path: Output video file or directory
            output_mode: 'mp4' or 'sequence'
            progress_callback: Optional (progress, message) callback
            **kwargs:
                max_objects: Maximum number of objects to track (default 50)
                min_area: Minimum mask area in pixels (default 500)
        """
        self._ensure_initialized()
        self._build_video_predictor()
        
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        max_objects = kwargs.get('max_objects', 50)
        min_area = kwargs.get('min_area', 500)
        
        print(f"\n[Crypto] ===== VIDEO CRYPTO MATTE PROCESSING =====")
        print(f"[Crypto] Model: SAM2 {self.model_size.value}")
        print(f"[Crypto] Input: {video_path}")
        print(f"[Crypto] Output: {output_path}")
        print(f"[Crypto] Max objects: {max_objects}")
        
        video_info = get_video_info(video_path)
        fps = video_info["fps"]
        width = video_info["width"]
        height = video_info["height"]
        
        print(f"[Crypto] Video: {video_info['frame_count']} frames @ {fps:.2f}fps, {width}x{height}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_frames_dir = temp_path / "input_frames"
            output_frames_dir = temp_path / "output_frames"
            input_frames_dir.mkdir()
            output_frames_dir.mkdir()
            
            # Step 1: Extract frames
            if progress_callback:
                progress_callback(0.0, "Extracting frames...")
            
            frame_paths = extract_frames(video_path, input_frames_dir)
            
            if not frame_paths:
                raise ValueError("No frames extracted from video")
            
            # Step 2: Detect objects on first frame
            if progress_callback:
                progress_callback(0.1, "Detecting objects on frame 0...")
            
            print(f"[Crypto] Detecting objects on frame 0...")
            first_frame = cv2.imread(str(frame_paths[0]))
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            
            masks = self.mask_generator.generate(first_frame_rgb)
            masks = [m for m in masks if m['area'] >= min_area]
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:max_objects]
            
            num_objects = len(masks)
            print(f"[Crypto] Detected {num_objects} objects on frame 0")
            
            if num_objects == 0:
                print(f"[Crypto] WARNING: No objects detected!")
                return self._create_empty_output(
                    frame_paths, output_frames_dir, output_path, fps, output_mode, (height, width)
                )
            
            # Step 3: Prepare JPEG frames for SAM2 video predictor
            if progress_callback:
                progress_callback(0.15, "Preparing video tracking...")
            
            jpeg_frames_dir = temp_path / "jpeg_frames"
            jpeg_frames_dir.mkdir()
            
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                jpeg_path = jpeg_frames_dir / f"{i:05d}.jpg"
                cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Step 4: Track objects through video
            if progress_callback:
                progress_callback(0.2, f"Tracking {num_objects} objects through video...")
            
            print(f"[Crypto] Initializing video tracking state...")
            
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                inference_state = self.video_predictor.init_state(
                    video_path=str(jpeg_frames_dir)
                )
                
                # Add object prompts from detected masks on frame 0
                print(f"[Crypto] Adding {num_objects} object prompts...")
                for obj_id, mask_data in enumerate(masks):
                    bbox = mask_data['bbox']  # [x, y, w, h]
                    box = np.array([
                        bbox[0], bbox[1],
                        bbox[0] + bbox[2], bbox[1] + bbox[3]
                    ])
                    
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=box,
                    )
                
                # Propagate through entire video
                print(f"[Crypto] Propagating masks through video...")
                video_segments = {}
                
                for frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                    video_segments[frame_idx] = {
                        obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                        for i, obj_id in enumerate(out_obj_ids)
                    }
                    
                    if (frame_idx + 1) % 20 == 0:
                        print(f"[Crypto]   Propagated to frame {frame_idx + 1}/{len(frame_paths)}")
                        if progress_callback:
                            progress_callback(
                                0.2 + 0.5 * ((frame_idx + 1) / len(frame_paths)),
                                f"Tracking frame {frame_idx + 1}/{len(frame_paths)}"
                            )
                
                print(f"[Crypto] Propagation complete - {len(video_segments)} frames processed")
            
            # Step 5: Render ID mattes
            if progress_callback:
                progress_callback(0.7, "Rendering ID mattes...")
            
            print(f"[Crypto] Generating ID matte frames...")
            colors = generate_color_palette(num_objects)
            
            # Raw ID array for lossless EXR export (uint16 supports up to 65535 objects)
            raw_id_mattes = np.zeros((len(frame_paths), height, width), dtype=np.uint16)
            
            for frame_idx, frame_path in enumerate(frame_paths):
                if frame_idx in video_segments:
                    id_matte = self._create_id_matte_from_segments(
                        video_segments[frame_idx], colors, (height, width)
                    )
                    
                    # Build raw ID data (1-indexed, 0=background)
                    for obj_id, mask in video_segments[frame_idx].items():
                        if mask.shape[:2] != (height, width):
                            mask = cv2.resize(
                                mask.astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        raw_id_mattes[frame_idx][mask] = obj_id + 1
                else:
                    id_matte = np.zeros((height, width, 3), dtype=np.uint8)
                
                output_frame_path = output_frames_dir / frame_path.name
                cv2.imwrite(str(output_frame_path), cv2.cvtColor(id_matte, cv2.COLOR_RGB2BGR))
                
                if (frame_idx + 1) % 20 == 0:
                    print(f"[Crypto]   Rendered {frame_idx + 1}/{len(frame_paths)} frames")
            
            # Step 6: Assemble output
            if progress_callback:
                progress_callback(0.9, "Assembling output...")
            
            if output_mode == "sequence":
                output_path.mkdir(parents=True, exist_ok=True)
                for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                    shutil.copy(frame_path, output_path / frame_path.name)
                
                raw_crypto_path = output_path / "crypto_raw.npy"
                np.save(str(raw_crypto_path), raw_id_mattes)
                print(f"[Crypto] Saved raw crypto data: {raw_crypto_path} (shape: {raw_id_mattes.shape})")
                
                frames_list = [f.name for f in sorted(output_path.glob("*.png"))]
                
                if progress_callback:
                    progress_callback(1.0, "Complete")
                
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
                output_path.parent.mkdir(parents=True, exist_ok=True)
                assemble_video(output_frames_dir, output_path, fps)
                
                raw_crypto_path = output_path.parent / f"{output_path.stem}_raw.npy"
                np.save(str(raw_crypto_path), raw_id_mattes)
                print(f"[Crypto] Saved raw crypto data: {raw_crypto_path} (shape: {raw_id_mattes.shape})")
                
                if progress_callback:
                    progress_callback(1.0, "Complete")
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": False,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "num_objects": num_objects,
                    "sam_version": self._sam_version,
                    "raw_data_path": str(raw_crypto_path),
                }
    
    # ========================================================================
    # Raw Data Access
    # ========================================================================
    
    def get_masks(
        self,
        image_path: Path,
        max_objects: int = 50,
        min_area: int = 500,
    ) -> List[Dict]:
        """
        Get raw mask data (for programmatic use, EXR export, etc.)
        
        Returns list of mask dicts with 'segmentation', 'area', 'bbox', etc.
        """
        self._ensure_initialized()
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = self.mask_generator.generate(image_rgb)
        masks = [m for m in masks if m['area'] >= min_area]
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:max_objects]
        
        return masks
    
    def segment_point(
        self,
        image_path: Path,
        point: Tuple[int, int],
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Segment object at a specific point (interactive mode).
        TODO: Implement point-based segmentation using SAM2Predictor
        """
        return self.process(image_path, output_path, **kwargs)
    
    def get_version(self) -> str:
        return self._sam_version or "unknown"
    
    # ========================================================================
    # ID Matte Rendering
    # ========================================================================
    
    @staticmethod
    def _create_id_matte(
        masks: List[Dict],
        shape: Tuple[int, int],
        colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Create ID matte from AutomaticMaskGenerator output"""
        h, w = shape
        id_matte = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            if i < len(colors):
                mask_bool = mask['segmentation']
                id_matte[mask_bool] = colors[i]
        
        return id_matte
    
    @staticmethod
    def _create_id_matte_from_segments(
        segments: Dict[int, np.ndarray],
        colors: List[Tuple[int, int, int]],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create ID matte from VideoPredictor tracked segments"""
        h, w = shape
        id_matte = np.zeros((h, w, 3), dtype=np.uint8)
        
        for obj_id, mask in segments.items():
            if obj_id < len(colors):
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                id_matte[mask] = colors[obj_id]
        
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
            cv2.imwrite(str(output_frames_dir / frame_path.name), empty_frame)
        
        if output_mode == "sequence":
            output_path.mkdir(parents=True, exist_ok=True)
            for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                shutil.copy(frame_path, output_path / frame_path.name)
            
            frames = sorted(output_path.glob("*.png"))
            return {
                "output_path": str(output_path),
                "is_sequence": True,
                "frame_count": len(frame_paths),
                "fps": fps,
                "num_objects": 0,
                "first_frame": frames[0].name if frames else None,
            }
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            assemble_video(output_frames_dir, output_path, fps)
            
            return {
                "output_path": str(output_path),
                "is_sequence": False,
                "frame_count": len(frame_paths),
                "fps": fps,
                "num_objects": 0,
            }
    
    # ========================================================================
    # VRAM Management
    # ========================================================================
    
    def unload(self):
        """Unload SAM2 models from VRAM"""
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None
        
        if self.video_predictor is not None:
            del self.video_predictor
            self.video_predictor = None
        
        super().unload()