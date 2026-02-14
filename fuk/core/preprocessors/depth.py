# core/preprocessors/depth.py
"""
Depth Estimation Preprocessor

Handles both single images and video with a unified interface.
Video uses batch inference with GLOBAL normalization for temporal consistency.

Models (in order of quality):
1. Depth Anything V3 - Latest SOTA with multi-view support
2. Depth Anything V2 - Excellent quality, local checkpoint
3. ZoeDepth - Metric depth estimation
4. MiDaS Large - Good quality, slower
5. MiDaS Small - Fast, lower quality

Good for:
- 3D scene control
- Parallax effects
- Bokeh/DOF effects
- Spatial composition
- EXR depth channel export
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import tempfile
import shutil
import cv2
import numpy as np
from PIL import Image
import torch
import json

from .base import BasePreprocessor
from core.video_utils import (
    get_video_info, extract_frames, assemble_video,
    apply_depth_colormap, is_video_file,
)


class DepthModel(str, Enum):
    """Available depth estimation models"""
    MIDAS_SMALL = "midas_small"
    MIDAS_LARGE = "midas_large"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    DEPTH_ANYTHING_V3 = "depth_anything_v3"  # Alias for DA3_MONO_LARGE
    DA3_MONO_LARGE = "da3_mono_large"
    DA3_METRIC_LARGE = "da3_metric_large"
    DA3_LARGE = "da3_large"
    DA3_GIANT = "da3_giant"
    ZOEDEPTH = "zoedepth"


# HuggingFace model IDs for DA3 variants
DA3_MODEL_IDS = {
    DepthModel.DEPTH_ANYTHING_V3: "depth-anything/DA3MONO-LARGE",
    DepthModel.DA3_MONO_LARGE: "depth-anything/DA3MONO-LARGE",
    DepthModel.DA3_METRIC_LARGE: "depth-anything/DA3METRIC-LARGE",
    DepthModel.DA3_LARGE: "depth-anything/DA3-LARGE-1.1",
    DepthModel.DA3_GIANT: "depth-anything/DA3-GIANT-1.1",
}

# Hardcoded fallback defaults (used if config not found)
_FALLBACK_DEFAULTS = {
    "process_res": 1344,
    "process_res_method": "lower_bound_resize",
}


class DepthPreprocessor(BasePreprocessor):
    """
    Depth map estimation with multiple model options.
    
    Handles both single images and video:
      - process()       -> single image depth
      - process_video() -> batch inference with global normalization + temporal smoothing
    """
    
    # Class-level config cache (loaded once, shared across instances)
    _da3_config: Optional[Dict] = None
    _config_loaded: bool = False
    
    def __init__(
        self,
        model_type: DepthModel = DepthModel.DA3_MONO_LARGE,
        config_path: Optional[Path] = None
    ):
        super().__init__(config_path)
        self.model_type = model_type
        self.requested_model = model_type
        self.model = None
        self.transform = None
        self._is_da3 = model_type in DA3_MODEL_IDS
        
        self._load_da3_config()
        
        print(f"Depth preprocessor initialized: {model_type.value} on {self.device}")
        if self._is_da3:
            print(f"  DA3 config: process_res={self.da3_process_res}, method={self.da3_process_res_method}")
    
    # ========================================================================
    # Config
    # ========================================================================
    
    @classmethod
    def _load_da3_config(cls):
        """Load DA3 config from JSON file (class method, loads once)"""
        if cls._config_loaded:
            return
        cls._config_loaded = True
        
        config_locations = [
            Path(__file__).parent.parent.parent / "config" / "tools" / "depth-anything-v3.json",
            Path("/home/brad/fuk/config/tools/depth-anything-v3.json"),
            Path("/home/brad/fuk/depth-anything-v3.json"),
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        cls._da3_config = json.load(f)
                    print(f"[Depth] Loaded DA3 config from: {config_path}")
                    return
                except Exception as e:
                    print(f"[Depth] Warning: Failed to load config from {config_path}: {e}")
        
        print(f"[Depth] Warning: DA3 config not found, using fallback defaults")
        cls._da3_config = {}
    
    @property
    def da3_process_res(self) -> int:
        if self._da3_config and "inference_defaults" in self._da3_config:
            return self._da3_config["inference_defaults"].get("process_res", _FALLBACK_DEFAULTS["process_res"])
        return _FALLBACK_DEFAULTS["process_res"]
    
    @property
    def da3_process_res_method(self) -> str:
        if self._da3_config and "inference_defaults" in self._da3_config:
            return self._da3_config["inference_defaults"].get("process_res_method", _FALLBACK_DEFAULTS["process_res_method"])
        return _FALLBACK_DEFAULTS["process_res_method"]
    
    # ========================================================================
    # Model Loading
    # ========================================================================
    
    def _initialize(self):
        """Load model based on selected type"""
        if self.model_type == DepthModel.MIDAS_SMALL:
            self._load_midas("MiDaS_small")
        elif self.model_type == DepthModel.MIDAS_LARGE:
            self._load_midas("DPT_Large")
        elif self.model_type == DepthModel.DEPTH_ANYTHING_V2:
            self._load_depth_anything_v2()
        elif self.model_type in DA3_MODEL_IDS:
            self._load_depth_anything_v3()
        elif self.model_type == DepthModel.ZOEDEPTH:
            self._load_zoedepth()
    
    def _load_midas(self, model_name: str = "DPT_Large"):
        print(f"Loading MiDaS {model_name}...")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name == "DPT_Large":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print("✓ MiDaS loaded")
    
    def _load_depth_anything_v3(self):
        try:
            vendor_path = self._get_vendor_path("Depth-Anything-3")
            if not vendor_path.exists():
                for name in ["depth-anything-3", "DepthAnything3", "DA3"]:
                    alt_path = self._get_vendor_path(name)
                    if alt_path.exists():
                        vendor_path = alt_path
                        break
            
            if vendor_path.exists():
                import sys
                sys.path.insert(0, str(vendor_path))
                print(f"Using Depth Anything 3 from vendor: {vendor_path}")
            
            from depth_anything_3.api import DepthAnything3
            
            model_id = DA3_MODEL_IDS.get(self.model_type, "depth-anything/DA3MONO-LARGE")
            print(f"Loading Depth Anything 3: {model_id}...")
            
            local_model_path = Path.home() / "ai" / "models" / "depth" / model_id.split("/")[-1]
            
            if local_model_path.exists():
                print(f"  Using local model: {local_model_path}")
                self.model = DepthAnything3.from_pretrained(str(local_model_path))
            else:
                print(f"  Downloading from HuggingFace: {model_id}")
                self.model = DepthAnything3.from_pretrained(model_id)
            
            self.model = self.model.to(device=self.device)
            self._is_da3 = True
            print(f"✓ Depth Anything 3 loaded: {model_id}")
            
        except ImportError as e:
            print(f"⚠ Depth Anything 3 not available: {e}")
            print("  Falling back to V2...")
            self.model_type = DepthModel.DEPTH_ANYTHING_V2
            self._is_da3 = False
            self._load_depth_anything_v2()
        except Exception as e:
            print(f"⚠ Could not load Depth Anything 3: {e}")
            print("  Falling back to V2...")
            self.model_type = DepthModel.DEPTH_ANYTHING_V2
            self._is_da3 = False
            self._load_depth_anything_v2()
    
    def _load_depth_anything_v2(self):
        try:
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
                print("Using installed depth-anything-v2 package")
            except ImportError:
                vendor_path = self._get_vendor_path("Depth-Anything-V2")
                if not vendor_path.exists():
                    raise ImportError(
                        f"depth-anything-v2 not found. Either:\n"
                        f"  1. pip install depth-anything-v2\n"
                        f"  2. Clone to vendor: git clone https://github.com/DepthAnything/Depth-Anything-V2 {vendor_path}"
                    )
                import sys
                sys.path.insert(0, str(vendor_path))
                from depth_anything_v2.dpt import DepthAnythingV2
                print(f"Using depth-anything-v2 from vendor: {vendor_path}")
            
            print("Loading Depth Anything V2...")
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            encoder = 'vitl'
            self.model = DepthAnythingV2(**model_configs[encoder])
            
            checkpoint_path = self._find_v2_checkpoint()
            if checkpoint_path and checkpoint_path.exists():
                print(f"  Loading checkpoint: {checkpoint_path}")
                self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            else:
                raise ValueError("Depth Anything V2 checkpoint not found")
            
            self.model.to(self.device)
            self.model.eval()
            self._is_da3 = False
            print("✓ Depth Anything V2 loaded")
            
        except Exception as e:
            print(f"⚠ Could not load Depth Anything V2: {e}")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE
            self._is_da3 = False
            self._load_midas("DPT_Large")
    
    def _find_v2_checkpoint(self) -> Optional[Path]:
        """Find Depth Anything V2 checkpoint"""
        if self.config_path:
            try:
                import importlib.util
                downloader_path = self.config_path.parent / "model_downloader.py"
                if downloader_path.exists():
                    spec = importlib.util.spec_from_file_location("model_downloader", downloader_path)
                    downloader_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(downloader_module)
                    checkpoint_path = downloader_module.ensure_model_downloaded(
                        "depth_anything_v2", self.config_path
                    )
                    if checkpoint_path:
                        return Path(checkpoint_path)
            except Exception as e:
                print(f"  Auto-download attempt failed: {e}")
        
        search_paths = [
            Path.home() / "ai" / "models" / "depth" / "depth_anything_v2_vitl.pth",
            Path.home() / "ai" / "models" / "checkpoints" / "depth_anything_v2_vitl.pth",
            self._get_vendor_path("Depth-Anything-V2") / "checkpoints" / "depth_anything_v2_vitl.pth",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        print("✗ Checkpoint not found. Searched:")
        for path in search_paths:
            print(f"    - {path}")
        print("  Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Large")
        return None
    
    def _load_zoedepth(self):
        try:
            print("Loading ZoeDepth...")
            self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self._is_da3 = False
            print("✓ ZoeDepth loaded")
        except Exception as e:
            print(f"⚠ Could not load ZoeDepth: {e}")
            print("  Falling back to MiDaS...")
            self.model_type = DepthModel.MIDAS_LARGE
            self._load_midas("DPT_Large")
    
    # ========================================================================
    # Single Image Processing
    # ========================================================================
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate depth map for a single image.
        
        Args:
            image_path: Input image
            output_path: Where to save result
            **kwargs:
                invert: Invert depth (far=white, near=black)
                normalize: Normalize depth to [0, 1]
                colormap: Apply colormap (None, 'inferno', 'viridis', 'magma', 'plasma')
                exact_output: If True, write to exact output_path (for video frames)
                process_res: DA3 processing resolution (default from config)
                process_res_method: DA3 resize method (default from config)
        """
        self._ensure_initialized()
        
        invert = kwargs.get('invert', False)
        normalize = kwargs.get('normalize', True)
        colormap = kwargs.get('colormap', 'inferno')
        exact_output = kwargs.get('exact_output', False)
        process_res = kwargs.get('process_res', self.da3_process_res)
        process_res_method = kwargs.get('process_res_method', self.da3_process_res_method)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = self._infer_depth(image_rgb, str(image_path), process_res, process_res_method)
        
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        if invert:
            depth = 1.0 - depth
        
        output_image = apply_depth_colormap(depth, colormap)
        
        params = {
            'method': 'depth',
            'model': self.model_type.value,
            'invert': invert,
            'normalize': normalize,
            'colormap': colormap,
        }
        final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
        cv2.imwrite(str(final_output), output_image)
        
        return {
            "output_path": str(final_output),
            "method": "depth",
            "model": self.model_type.value,
            "parameters": params,
        }
    
    # ========================================================================
    # Video Batch Processing (overrides BasePreprocessor.process_video)
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
        Process video with batch inference and global normalization.
        
        For DA3 models, all frames are inferred in a single batch call,
        then normalized using the GLOBAL min/max across all frames.
        This prevents the temporal jitter caused by per-frame normalization.
        
        Non-DA3 models fall back to frame-by-frame processing via the base class.
        
        Args:
            video_path: Input video file
            output_path: Output video file or directory
            output_mode: 'mp4' or 'sequence'
            progress_callback: Optional (progress, message) callback
            **kwargs:
                invert: Invert depth values
                normalize: Normalize to 0-1 (uses GLOBAL min/max)
                colormap: Colormap to apply
                process_res: DA3 processing resolution
                process_res_method: DA3 resize method
                temporal_smooth: Temporal median filter window (0=off, default 3)
        """
        self._ensure_initialized()
        
        # Non-DA3 models: fall back to frame-by-frame (base class)
        if not self._is_da3:
            return super().process_video(
                video_path=video_path,
                output_path=output_path,
                output_mode=output_mode,
                progress_callback=progress_callback,
                **kwargs
            )
        
        # DA3 batch path
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        invert = kwargs.get('invert', False)
        normalize = kwargs.get('normalize', True)
        colormap = kwargs.get('colormap', 'inferno')
        process_res = kwargs.get('process_res', self.da3_process_res)
        process_res_method = kwargs.get('process_res_method', self.da3_process_res_method)
        temporal_smooth = kwargs.get('temporal_smooth', 3)
        
        print(f"\n[Depth] ===== BATCH VIDEO DEPTH PROCESSING =====")
        print(f"[Depth] Model: {self.model_type.value}")
        print(f"[Depth] Input: {video_path}")
        print(f"[Depth] Output: {output_path}")
        print(f"[Depth] Temporal smoothing: {temporal_smooth if temporal_smooth > 0 else 'off'}")
        
        video_info = get_video_info(video_path)
        fps = video_info["fps"]
        original_size = (video_info["width"], video_info["height"])
        
        print(f"[Depth] Frames: {video_info['frame_count']} @ {fps:.2f}fps, {original_size[0]}x{original_size[1]}")
        
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
            
            # Step 2: Batch inference
            if progress_callback:
                progress_callback(0.1, f"Running batch inference on {len(frame_paths)} frames...")
            
            print(f"[Depth] Running batch inference...")
            frame_path_strs = [str(p) for p in frame_paths]
            
            try:
                prediction = self.model.inference(
                    image=frame_path_strs,
                    process_res=process_res,
                    process_res_method=process_res_method,
                )
                
                all_depths = prediction.depth  # [N, H, W]
                print(f"[Depth] Inference complete - shape: {all_depths.shape}")
                
            except Exception as e:
                print(f"[Depth] ERROR during batch inference: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Step 3: Global normalization
            if normalize:
                global_min = all_depths.min()
                global_max = all_depths.max()
                print(f"[Depth] Global depth range: [{global_min:.4f}, {global_max:.4f}]")
                all_depths = (all_depths - global_min) / (global_max - global_min + 1e-8)
            
            # Step 4: Temporal smoothing
            if temporal_smooth > 1:
                print(f"[Depth] Applying temporal smoothing (window={temporal_smooth})...")
                all_depths = self._temporal_smooth(all_depths, window=temporal_smooth)
            
            # Step 5: Invert
            if invert:
                all_depths = 1.0 - all_depths
            
            if progress_callback:
                progress_callback(0.6, "Post-processing depth maps...")
            
            # Step 6: Colorize each frame
            print(f"[Depth] Post-processing depth maps...")
            for i, (frame_path, depth_map) in enumerate(zip(frame_paths, all_depths)):
                output_frame_path = output_frames_dir / frame_path.name
                
                # Resize to original dimensions if needed
                if depth_map.shape[:2] != (original_size[1], original_size[0]):
                    depth_map = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_LINEAR)
                
                output_image = apply_depth_colormap(depth_map, colormap)
                cv2.imwrite(str(output_frame_path), output_image)
                
                if (i + 1) % 10 == 0:
                    print(f"[Depth]   Processed {i+1}/{len(frame_paths)} frames")
                    if progress_callback:
                        progress_callback(
                            0.6 + 0.3 * ((i + 1) / len(frame_paths)),
                            f"Colorizing frame {i+1}/{len(frame_paths)}"
                        )
            
            # Step 7: Assemble output
            if progress_callback:
                progress_callback(0.9, "Assembling output...")
            
            if output_mode == "sequence":
                output_path.mkdir(parents=True, exist_ok=True)
                for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                    shutil.copy(frame_path, output_path / frame_path.name)
                
                # Save raw depth data for lossless EXR export
                raw_depth_path = output_path / "depth_raw.npy"
                np.save(str(raw_depth_path), all_depths.astype(np.float32))
                print(f"[Depth] Saved raw depth data: {raw_depth_path}")
                
                frames = sorted([f.name for f in output_path.glob("*.png")])
                
                if progress_callback:
                    progress_callback(1.0, "Complete")
                
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
                output_path.parent.mkdir(parents=True, exist_ok=True)
                assemble_video(output_frames_dir, output_path, fps)
                
                # Save raw depth data alongside MP4
                raw_depth_path = output_path.parent / f"{output_path.stem}_raw.npy"
                np.save(str(raw_depth_path), all_depths.astype(np.float32))
                print(f"[Depth] Saved raw depth data: {raw_depth_path}")
                
                if progress_callback:
                    progress_callback(1.0, "Complete")
                
                return {
                    "output_path": str(output_path),
                    "is_sequence": False,
                    "frame_count": len(frame_paths),
                    "fps": fps,
                    "raw_data_path": str(raw_depth_path),
                }
    
    # ========================================================================
    # Inference (shared by single image and video)
    # ========================================================================
    
    def _infer_depth(
        self,
        image_rgb: np.ndarray,
        image_path: str = None,
        process_res: int = None,
        process_res_method: str = None,
    ) -> np.ndarray:
        """Run inference based on loaded model type"""
        if process_res is None:
            process_res = self.da3_process_res
        if process_res_method is None:
            process_res_method = self.da3_process_res_method
        
        if self.model_type in [DepthModel.MIDAS_SMALL, DepthModel.MIDAS_LARGE]:
            input_batch = self.transform(image_rgb).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            return prediction.cpu().numpy()
        
        elif self._is_da3 and self.model_type in DA3_MODEL_IDS:
            print(f"[DA3] Inference with process_res={process_res}, method={process_res_method}")
            
            if image_path:
                prediction = self.model.inference(
                    [image_path],
                    process_res=process_res,
                    process_res_method=process_res_method,
                )
            else:
                import tempfile as _tempfile
                with _tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_path = f.name
                    Image.fromarray(image_rgb).save(temp_path)
                    prediction = self.model.inference(
                        [temp_path],
                        process_res=process_res,
                        process_res_method=process_res_method,
                    )
                    Path(temp_path).unlink()
            
            depth = prediction.depth[0]
            if depth.shape[:2] != image_rgb.shape[:2]:
                depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]))
            return depth
        
        elif self.model_type == DepthModel.DEPTH_ANYTHING_V2:
            return self.model.infer_image(image_rgb)
        
        elif self.model_type == DepthModel.ZOEDEPTH:
            pil_image = Image.fromarray(image_rgb)
            return self.model.infer_pil(pil_image)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    # ========================================================================
    # Raw Data Access (for EXR export, normals, etc.)
    # ========================================================================
    
    def get_raw_depth(self, image_path: Path) -> np.ndarray:
        """
        Get raw depth values (for EXR export, normal calculation, etc.)
        Returns normalized float32 depth array [0, 1]
        """
        self._ensure_initialized()
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = self._infer_depth(image_rgb, str(image_path))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    @staticmethod
    def _temporal_smooth(depths: np.ndarray, window: int = 3) -> np.ndarray:
        """
        Temporal median filter to reduce frame-to-frame noise.
        
        Args:
            depths: [N, H, W] depth array
            window: Window size (must be odd, e.g., 3, 5)
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
    
    def unload(self):
        """Unload depth model from VRAM"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.transform is not None:
            del self.transform
            self.transform = None
        super().unload()