# core/preprocessors/depth.py
"""
Depth Estimation Preprocessor

Handles both single images and video with a unified interface.
Video uses chunked batch inference with GLOBAL normalization for temporal
consistency, streaming raw depth to an on-disk .npy memmap so peak RAM stays
proportional to the chunk size rather than the frame count.

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
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import deque
from enum import Enum
import tempfile
import shutil
import gc
import cv2
import numpy as np
from PIL import Image
import torch
import json

from .base import BasePreprocessor
from core.video_utils import (
    get_video_info, extract_frames, assemble_video,
    apply_depth_greyscale, apply_depth_colormap, is_video_file,
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


# Models whose raw inference output is inverse depth (disparity): larger value
# = NEARER. DA3 and ZoeDepth return true depth, where larger = farther.
# Anything that reasons about surface geometry has to know which it is holding.
DISPARITY_MODELS = {
    DepthModel.MIDAS_SMALL,
    DepthModel.MIDAS_LARGE,
    DepthModel.DEPTH_ANYTHING_V2,
}

# Models that return absolute depth in metres, usable as Z with no further
# assumption. Everything else — including the DA3 mono/large/giant series — is
# relative, and needs a near-plane offset before it means anything geometrically.
#
# Do NOT extend this set from documentation alone. DA3's README says the mono
# series "directly predicts depth" rather than disparity, which is true and yet
# does not make the output usable as Z: measured on real footage its near plane
# lands at ~0 (a 960x576 theatre frame returned -0.515 .. 4.162, with the
# foreground subject spanning -0.068 .. 0.167). Z~0 on the nearest subject is
# the worst case for the normal formula — the stabilising +Z term vanishes,
# the perspective term takes over, and the closest thing in frame, the one the
# eye goes to first, is exactly what falls apart.
METRIC_MODELS = {
    DepthModel.ZOEDEPTH,
}

# Relative disparity carries an unknown shift, so disparity -> depth is only
# recoverable up to a choice of how far the background sits. This is the ratio
# of near-plane to far-plane distance used when reciprocating: 0.1 puts the
# farthest surface at 10x the distance of the nearest. Only affects
# DISPARITY_MODELS — prefer a DA3 variant, which returns true depth directly.
DISPARITY_FAR_RATIO = 0.1

# HuggingFace model IDs for DA3 variants
DA3_MODEL_IDS = {
    DepthModel.DEPTH_ANYTHING_V3: "depth-anything/DA3MONO-LARGE",
    DepthModel.DA3_MONO_LARGE: "depth-anything/DA3MONO-LARGE",
    DepthModel.DA3_METRIC_LARGE: "depth-anything/DA3METRIC-LARGE",
    DepthModel.DA3_LARGE: "depth-anything/DA3-LARGE-1.1",
    DepthModel.DA3_GIANT: "depth-anything/DA3-GIANT-1.1",
}

# DA3 variants that use cross-view (global) attention. Their presets set
# alt_start >= 0, so every view in a batch attends to every other view and
# chunking genuinely changes the result — those chunks need overlap + affine
# alignment. The mono/metric presets use alt_start = -1 (local attention
# only), so each view is computed independently of the others and chunk size
# has no mathematical effect there.
DA3_MULTIVIEW_MODELS = {
    DepthModel.DA3_LARGE,
    DepthModel.DA3_GIANT,
}

# DA3METRIC derives a single least-squares metric scale per inference call, so
# it also needs overlap alignment to stay consistent across chunk boundaries.
DA3_GLOBAL_SCALE_MODELS = {
    DepthModel.DA3_METRIC_LARGE,
}

# Hardcoded fallback defaults (used if config not found)
_FALLBACK_DEFAULTS = {
    "process_res": 1344,
    "process_res_method": "lower_bound_resize",
    "chunk_size": 0,        # 0 = auto-size from free VRAM
    "chunk_overlap": 2,     # frames re-inferred per chunk (multi-view models only)
}

# Chunk auto-sizing. Rough per-frame VRAM cost of a DA3 forward pass, in bytes
# per processed pixel: fp32 input + the four stashed backbone feature layers +
# DPT head intermediates. Deliberately conservative; an OOM still halves the
# chunk and retries, so over-estimating only costs a little throughput.
_VRAM_BYTES_PER_PIXEL = 420
_VRAM_BUDGET_FRACTION = 0.55
_MIN_CHUNK = 1
_MAX_CHUNK = 64


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

    @property
    def da3_chunk_size(self) -> int:
        """Frames per inference call for video. 0 = auto-size from free VRAM."""
        if self._da3_config and "inference_defaults" in self._da3_config:
            return int(self._da3_config["inference_defaults"].get("chunk_size", _FALLBACK_DEFAULTS["chunk_size"]))
        return _FALLBACK_DEFAULTS["chunk_size"]

    @property
    def da3_chunk_overlap(self) -> int:
        """Frames re-inferred between chunks to align scale (multi-view models)."""
        if self._da3_config and "inference_defaults" in self._da3_config:
            return int(self._da3_config["inference_defaults"].get("chunk_overlap", _FALLBACK_DEFAULTS["chunk_overlap"]))
        return _FALLBACK_DEFAULTS["chunk_overlap"]
    
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
            self._suppress_processed_images()
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
    
    def _suppress_processed_images(self):
        """
        Stop DA3 from building Prediction.processed_images.

        The vendor's DepthAnything3._add_processed_images() denormalises the
        input batch for visualisation. It multiplies the float32 (N,H,W,3)
        batch by float64 ImageNet constants, which promotes the whole thing to
        float64 and then makes two more full-size float64 copies for the clip
        and the *255. That is ~4x the size of the depth output, three times
        over, and FUK never reads processed_images.

        Overriding it with an instance attribute shadows the bound method
        without touching vendor code.
        """
        if self.model is None:
            return
        try:
            self.model._add_processed_images = lambda prediction, imgs_cpu: prediction
        except Exception as e:
            print(f"[Depth] Note: could not disable processed_images allocation: {e}")

    @staticmethod
    def _release_prediction(prediction) -> None:
        """
        Drop the large fields FUK never reads (conf, sky, aux, processed images).

        Without this the whole Prediction stays alive for as long as the caller
        holds a reference to prediction.depth, pinning a second float32
        (N,H,W) confidence map plus whatever else the model emitted.
        """
        for field in ("conf", "sky", "aux", "processed_images", "gaussians"):
            try:
                setattr(prediction, field, None)
            except Exception:
                pass

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
                range_min: Low end of depth range remap (0.0-1.0)
                range_max: High end of depth range remap (0.0-1.0)
                guided_filter: Apply guided edge refinement (default False)
                exact_output: If True, write to exact output_path (for video frames)
                process_res: DA3 processing resolution (default from config)
                process_res_method: DA3 resize method (default from config)
        """
        self._ensure_initialized()
        
        invert = kwargs.get('invert', False)
        normalize = kwargs.get('normalize', True)
        range_min = kwargs.get('range_min', 0.0)
        range_max = kwargs.get('range_max', 1.0)
        guided_filter = kwargs.get('guided_filter', False)
        exact_output = kwargs.get('exact_output', False)
        process_res = kwargs.get('process_res', self.da3_process_res)
        process_res_method = kwargs.get('process_res_method', self.da3_process_res_method)
        
        image = self.load_image_bgr(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = self._infer_depth(image_rgb, str(image_path), process_res, process_res_method)
        
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        if invert:
            depth = 1.0 - depth
        
        # Optional guided edge refinement (off by default - degrades DA3 quality)
        if guided_filter:
            depth = self._guided_upsample(depth, image)
        
        output_image = apply_depth_greyscale(depth, range_min=range_min, range_max=range_max)
        
        params = {
            'method': 'depth',
            'model': self.model_type.value,
            'invert': invert,
            'normalize': normalize,
            'range_min': range_min,
            'range_max': range_max,
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
    # Chunked Inference (bounds peak memory for long videos)
    # ========================================================================

    @staticmethod
    def _estimate_processed_size(
        src_w: int,
        src_h: int,
        process_res: int,
        process_res_method: str,
        patch: int = 14,
    ) -> Tuple[int, int]:
        """
        Predict the resolution DA3's InputProcessor will produce.

        Mirrors InputProcessor._resize_image + _make_divisible_by_resize so we
        can size chunks before the first frame is loaded.
        """
        if process_res_method.startswith("lower_bound"):
            scale = process_res / float(min(src_w, src_h))
        else:
            scale = process_res / float(max(src_w, src_h))

        w = max(1, int(round(src_w * scale)))
        h = max(1, int(round(src_h * scale)))

        # Round each dimension to the nearest multiple of the patch size
        w = max(patch, int(round(w / patch)) * patch)
        h = max(patch, int(round(h / patch)) * patch)
        return w, h

    def _auto_chunk_size(self, processed_w: int, processed_h: int) -> int:
        """Pick a frames-per-inference count that fits in free VRAM."""
        pixels = processed_w * processed_h

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, _total = torch.cuda.mem_get_info()
            except Exception:
                free_bytes = 8 * 1024 ** 3
        else:
            # CPU inference: bound by system RAM instead
            try:
                import psutil
                free_bytes = psutil.virtual_memory().available
            except Exception:
                free_bytes = 8 * 1024 ** 3

        budget = free_bytes * _VRAM_BUDGET_FRACTION
        per_frame = max(1.0, pixels * _VRAM_BYTES_PER_PIXEL)
        chunk = int(budget // per_frame)
        return max(_MIN_CHUNK, min(_MAX_CHUNK, chunk))

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    def _infer_chunk(
        self,
        frame_paths: List[str],
        process_res: int,
        process_res_method: str,
    ) -> np.ndarray:
        """
        Run DA3 on one chunk of frames and return float32 depth (K, H, W).

        The returned array is detached from the Prediction so the confidence
        map and friends can be collected immediately.
        """
        prediction = self.model.inference(
            image=frame_paths,
            process_res=process_res,
            process_res_method=process_res_method,
        )

        depths = np.ascontiguousarray(prediction.depth, dtype=np.float32)
        self._release_prediction(prediction)
        prediction.depth = None
        del prediction

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return depths

    @staticmethod
    def _fit_affine(current: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
        """
        Least-squares scale/shift mapping `current` onto `reference`.

        Used on the overlapping frames between chunks so multi-view and metric
        models don't drift in scale across chunk boundaries. Returns (a, b) for
        a * current + b; falls back to identity if the fit is degenerate.
        """
        x = current.reshape(-1).astype(np.float64)
        y = reference.reshape(-1).astype(np.float64)

        # Subsample - a few hundred thousand pixels is plenty for two unknowns
        max_samples = 200_000
        if x.size > max_samples:
            step = x.size // max_samples
            x = x[::step]
            y = y[::step]

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 16:
            return 1.0, 0.0
        x = x[finite]
        y = y[finite]

        var = float(np.var(x))
        if not np.isfinite(var) or var < 1e-12:
            return 1.0, 0.0

        a = float(np.cov(x, y, bias=True)[0, 1] / var)
        if not np.isfinite(a) or abs(a) < 1e-8:
            return 1.0, 0.0
        b = float(np.mean(y) - a * np.mean(x))
        if not np.isfinite(b):
            return 1.0, 0.0
        return a, b

    def _stream_depth_to_memmap(
        self,
        frame_path_strs: List[str],
        raw_depth_path: Path,
        process_res: int,
        process_res_method: str,
        chunk_size: int,
        overlap: int,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        progress_span: Tuple[float, float] = (0.1, 0.6),
    ) -> Tuple[np.memmap, float, float]:
        """
        Infer depth in chunks, writing each chunk straight to an on-disk .npy.

        Returns (memmap, global_min, global_max). Peak RAM is one chunk of
        depth plus one chunk of preprocessed input, not the whole video.
        """
        n_frames = len(frame_path_strs)
        mm: Optional[np.memmap] = None
        global_min = np.inf
        global_max = -np.inf

        p_start, p_end = progress_span
        idx = 0
        written_upto = 0   # frames [0, written_upto) already committed to mm
        cur_chunk = max(_MIN_CHUNK, chunk_size)

        while idx < n_frames:
            # Overlap only makes sense if the chunk is strictly bigger than it
            cur_overlap = min(overlap, max(0, cur_chunk - 1))
            end = min(n_frames, idx + cur_chunk)

            try:
                depths = self._infer_chunk(
                    frame_path_strs[idx:end], process_res, process_res_method
                )
            except Exception as e:
                if not self._is_oom_error(e) or cur_chunk <= _MIN_CHUNK:
                    raise
                cur_chunk = max(_MIN_CHUNK, cur_chunk // 2)
                print(f"[Depth] OOM on chunk at frame {idx}; retrying with chunk_size={cur_chunk}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            if mm is None:
                h, w = depths.shape[1], depths.shape[2]
                print(f"[Depth] Raw depth buffer: {n_frames}x{h}x{w} float32 "
                      f"({n_frames * h * w * 4 / 1024 ** 3:.2f} GB) -> {raw_depth_path}")
                mm = np.lib.format.open_memmap(
                    str(raw_depth_path), mode="w+", dtype=np.float32, shape=(n_frames, h, w)
                )
            elif depths.shape[1:] != mm.shape[1:]:
                # Would otherwise surface as an opaque broadcast error
                raise ValueError(
                    f"Chunk at frame {idx} returned {depths.shape[1:]} but the buffer "
                    f"is {mm.shape[1:]}; frames must share a resolution"
                )

            # Frames before write_from were already committed by the previous
            # chunk; they exist only to align this chunk's scale to it. Derive
            # this from the write cursor, not from cur_overlap, which an OOM
            # retry may have shrunk since the previous chunk.
            write_from = min(max(idx, written_upto), end)

            if write_from > idx:
                a, b = self._fit_affine(
                    depths[: write_from - idx], np.asarray(mm[idx:write_from])
                )
                if a != 1.0 or b != 0.0:
                    depths *= a
                    depths += b

            new = depths[write_from - idx:]
            if new.size:
                mm[write_from:end] = new
                chunk_min = float(np.nanmin(new))
                chunk_max = float(np.nanmax(new))
                global_min = min(global_min, chunk_min)
                global_max = max(global_max, chunk_max)
                written_upto = end

            del depths, new

            if progress_callback:
                frac = end / float(n_frames)
                progress_callback(
                    p_start + (p_end - p_start) * frac,
                    f"Depth inference {end}/{n_frames} frames",
                )
            print(f"[Depth]   Inferred {end}/{n_frames} frames (chunk={cur_chunk})")

            if end >= n_frames:
                break
            idx = end - cur_overlap

        if mm is None:
            raise ValueError("No frames were inferred")

        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise ValueError("Depth inference produced no finite values")

        return mm, global_min, global_max

    @staticmethod
    def _temporal_smooth_memmap(mm: np.memmap, window: int) -> None:
        """
        In-place temporal median filter over an on-disk depth array.

        Keeps a rolling buffer of the *original* values for the frames still
        inside the filter window, so writing frame i doesn't corrupt the input
        for frames i+1 .. i+half. RAM cost is `window` frames.
        """
        if window <= 1:
            return

        n_frames = mm.shape[0]
        half = window // 2
        buf: deque = deque()  # (frame_index, original array)

        for i in range(n_frames):
            hi = min(n_frames - 1, i + half)
            lo = max(0, i - half)

            # Extend the buffer forward to cover the window
            while not buf or buf[-1][0] < hi:
                j = buf[-1][0] + 1 if buf else lo
                buf.append((j, np.array(mm[j], dtype=np.float32)))

            # Drop frames that have fallen out behind the window
            while buf and buf[0][0] < lo:
                buf.popleft()

            if len(buf) == 1:
                continue

            stack = np.stack([arr for _, arr in buf])
            mm[i] = np.median(stack, axis=0)
            del stack

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
        Process video with chunked inference and global normalization.

        For DA3 models, frames are inferred in chunks and streamed to an
        on-disk float32 .npy memmap, then normalized using the GLOBAL min/max
        across all frames. Global normalization prevents the temporal jitter
        caused by per-frame normalization; chunking keeps peak memory
        proportional to chunk_size instead of the frame count.

        Chunk size has no mathematical effect on the mono presets: their
        backbone uses local (per-view) attention only, so each frame is
        computed independently. It is not bit-exact, because the model runs
        under bfloat16 autocast and is not reproducible run to run even at a
        fixed batch size — measured drift between chunk sizes is ~1e-3 of the
        depth range on ~0.4% of pixels, concentrated on depth edges, i.e.
        below one 8-bit level.

        Multi-view presets (da3_large, da3_giant) and the metric preset do
        depend on batch composition, so they re-infer `chunk_overlap` frames
        per chunk and affine-align each chunk to the previous one to keep
        depth scale continuous across boundaries.

        Non-DA3 models fall back to frame-by-frame processing via the base class.

        Args:
            video_path: Input video file
            output_path: Output video file or directory
            output_mode: 'mp4' or 'sequence'
            progress_callback: Optional (progress, message) callback
            **kwargs:
                invert: Invert depth values
                normalize: Normalize to 0-1 (uses GLOBAL min/max)
                range_min: Low end of depth range remap (0.0-1.0)
                range_max: High end of depth range remap (0.0-1.0)
                process_res: DA3 processing resolution
                process_res_method: DA3 resize method
                guided_filter: Apply guided edge refinement per-frame (default False)
                temporal_smooth: Temporal median filter window (0=off, default 0)
                chunk_size: Frames per inference call (0/None = auto from free VRAM)
                chunk_overlap: Frames re-inferred per chunk for scale alignment
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
        range_min = kwargs.get('range_min', 0.0)
        range_max = kwargs.get('range_max', 1.0)
        process_res = kwargs.get('process_res', self.da3_process_res)
        process_res_method = kwargs.get('process_res_method', self.da3_process_res_method)
        temporal_smooth = kwargs.get('temporal_smooth', 0)
        guided_filter = kwargs.get('guided_filter', False)
        chunk_size = kwargs.get('chunk_size') or self.da3_chunk_size
        overlap = kwargs.get('chunk_overlap')
        if overlap is None:
            overlap = self.da3_chunk_overlap

        # Chunk composition only matters when the backbone mixes views or a
        # batch-global scale is fitted. Neither is true for the mono presets,
        # so they skip the overlap entirely and pay no re-inference cost.
        needs_alignment = (
            self.model_type in DA3_MULTIVIEW_MODELS
            or self.model_type in DA3_GLOBAL_SCALE_MODELS
        )
        if not needs_alignment:
            overlap = 0

        video_info = get_video_info(video_path)
        fps = video_info["fps"]
        original_size = (video_info["width"], video_info["height"])

        if chunk_size <= 0:
            proc_w, proc_h = self._estimate_processed_size(
                original_size[0], original_size[1], process_res, process_res_method
            )
            chunk_size = self._auto_chunk_size(proc_w, proc_h)
            auto_note = f" (auto, est. {proc_w}x{proc_h} processed)"
        else:
            auto_note = " (configured)"

        print(f"\n[Depth] ===== CHUNKED VIDEO DEPTH PROCESSING =====")
        print(f"[Depth] Model: {self.model_type.value}")
        print(f"[Depth] Input: {video_path}")
        print(f"[Depth] Output: {output_path}")
        print(f"[Depth] Chunk size: {chunk_size} frames{auto_note}")
        print(f"[Depth] Chunk overlap: {overlap} frames"
              f"{' (scale alignment)' if overlap else ' (not needed - local attention)'}")
        print(f"[Depth] Temporal smoothing: {temporal_smooth if temporal_smooth > 0 else 'off'}")
        print(f"[Depth] Guided filter: {'on' if guided_filter else 'off'}")

        print(f"[Depth] Frames: {video_info['frame_count']} @ {fps:.2f}fps, {original_size[0]}x{original_size[1]}")

        # The raw depth buffer is written directly to its final destination so
        # it never has to live in RAM (and never lands on a tmpfs scratch dir).
        if output_mode == "sequence":
            output_path.mkdir(parents=True, exist_ok=True)
            raw_depth_path = output_path / "depth_raw.npy"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_depth_path = output_path.parent / f"{output_path.stem}_raw.npy"

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

            # Step 2: Chunked inference, streamed to the on-disk raw buffer
            if progress_callback:
                progress_callback(0.1, f"Running inference on {len(frame_paths)} frames...")

            frame_path_strs = [str(p) for p in frame_paths]
            mm = None

            try:
                mm, global_min, global_max = self._stream_depth_to_memmap(
                    frame_path_strs=frame_path_strs,
                    raw_depth_path=raw_depth_path,
                    process_res=process_res,
                    process_res_method=process_res_method,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    progress_callback=progress_callback,
                    progress_span=(0.1, 0.55),
                )
                print(f"[Depth] Inference complete - shape: {mm.shape}")

                # Step 3: Temporal smoothing (in place, on disk)
                #
                # Applied before normalization rather than after. A median
                # commutes with the increasing affine map used for
                # normalization, so the result is identical to the old
                # normalize-then-smooth order, but this way the global range is
                # already known and each frame is touched exactly once.
                if temporal_smooth > 1:
                    print(f"[Depth] Applying temporal smoothing (window={temporal_smooth})...")
                    if progress_callback:
                        progress_callback(0.55, "Temporal smoothing...")
                    self._temporal_smooth_memmap(mm, window=temporal_smooth)

                if normalize:
                    print(f"[Depth] Global depth range: [{global_min:.4f}, {global_max:.4f}]")
                    inv_range = 1.0 / (global_max - global_min + 1e-8)

                if progress_callback:
                    progress_callback(0.6, "Post-processing depth maps...")

                # Step 4: Normalize + invert per frame, then write greyscale.
                # Each frame is read from disk, finished in place, and written
                # back, so only one frame is resident at a time.
                print(f"[Depth] Post-processing depth maps...")
                for i, frame_path in enumerate(frame_paths):
                    output_frame_path = output_frames_dir / frame_path.name

                    depth_map = np.array(mm[i], dtype=np.float32)

                    if normalize:
                        np.subtract(depth_map, global_min, out=depth_map)
                        np.multiply(depth_map, inv_range, out=depth_map)

                    if invert:
                        np.subtract(1.0, depth_map, out=depth_map)

                    # Write the finished values back so the .npy matches the
                    # frames (normalized + smoothed + inverted, as before)
                    mm[i] = depth_map

                    # Resize to original dimensions if needed
                    if depth_map.shape[:2] != (original_size[1], original_size[0]):
                        depth_map = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_LANCZOS4)

                    # Optional guided edge refinement (off by default)
                    if guided_filter:
                        try:
                            guide_bgr = self.load_image_bgr(frame_path)
                            depth_map = self._guided_upsample(depth_map, guide_bgr)
                        except ValueError:
                            pass

                    output_image = apply_depth_greyscale(depth_map, range_min=range_min, range_max=range_max)
                    cv2.imwrite(str(output_frame_path), output_image)

                    del depth_map

                    if (i + 1) % 10 == 0:
                        print(f"[Depth]   Processed {i+1}/{len(frame_paths)} frames")
                        if progress_callback:
                            progress_callback(
                                0.6 + 0.3 * ((i + 1) / len(frame_paths)),
                                f"Processing frame {i+1}/{len(frame_paths)}"
                            )

                mm.flush()
                print(f"[Depth] Saved raw depth data: {raw_depth_path}")

            except Exception:
                # Don't leave a half-written buffer behind for EXR export to find
                if mm is not None:
                    del mm
                    mm = None
                raw_depth_path.unlink(missing_ok=True)
                raise
            finally:
                if mm is not None:
                    del mm
                gc.collect()

            # Step 5: Assemble output
            if progress_callback:
                progress_callback(0.9, "Assembling output...")

            if output_mode == "sequence":
                for frame_path in sorted(output_frames_dir.glob("frame_*.png")):
                    shutil.copy(frame_path, output_path / frame_path.name)

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
                assemble_video(output_frames_dir, output_path, fps)

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
                fd, temp_path = _tempfile.mkstemp(suffix='.png')
                import os as _os
                _os.close(fd)
                try:
                    Image.fromarray(image_rgb).save(temp_path)
                    prediction = self.model.inference(
                        [temp_path],
                        process_res=process_res,
                        process_res_method=process_res_method,
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            depth = np.asarray(prediction.depth[0], dtype=np.float32)
            self._release_prediction(prediction)
            del prediction

            if depth.shape[:2] != image_rgb.shape[:2]:
                depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
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
        
        image = self.load_image_bgr(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = self._infer_depth(image_rgb, str(image_path))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)

    def get_depth_z(self, image_path: Path) -> Tuple[np.ndarray, bool]:
        """
        Depth oriented as true Z — near = small, far = large.

        For anything that reconstructs surface geometry (normals, point clouds,
        meshing) rather than just displaying a depth image. get_raw_depth()
        min-max normalises and hands back whatever orientation the model
        happened to use, so DA3 comes out near=0 while MiDaS/DAv2 come out
        near=1; consuming that as if it were Z silently inverts the geometry on
        half the model list.

        Relative models are rescaled to [0, 1] against ROBUST percentiles rather
        than min/max. A single blown highlight or a sliver of negative depth on
        a silhouette would otherwise set the scale for the whole frame, and the
        subject that matters is usually the one crushed by it.

        Returns:
            (depth, is_metric)
            is_metric=True  - absolute metres, usable as Z directly
            is_metric=False - relative, rescaled to [0, 1] with 0 = nearest; the
                              caller must add a near-plane offset before using
                              it as Z, because 0 means "nearest thing in frame",
                              not "at the camera"
        """
        self._ensure_initialized()

        image = self.load_image_bgr(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw = self._infer_depth(image_rgb, str(image_path)).astype(np.float32)

        if self.model_type in METRIC_MODELS and raw.min() > 0:
            return raw, True

        if self.model_type in DISPARITY_MODELS:
            # Scale-and-shift-invariant disparity: the shift is genuinely
            # unrecoverable, so reciprocating across an assumed near/far ratio
            # is the best available reconstruction.
            disp = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
            raw = 1.0 / (disp * (1.0 - DISPARITY_FAR_RATIO) + DISPARITY_FAR_RATIO)

        lo, hi = np.percentile(raw, 0.5), np.percentile(raw, 99.5)
        z = (raw - lo) / (hi - lo + 1e-8)
        return np.clip(z, -0.1, 1.1).astype(np.float32), False

    # ========================================================================
    # Utilities
    # ========================================================================
    

    @staticmethod
    def _guided_upsample(
        depth: np.ndarray,
        guide_bgr: np.ndarray,
        r: int = 10,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        Guided filter: snap coarse depth edges to RGB boundaries.
        
        WARNING: This smooths depth detail. DA3 already produces sharp edges,
        so this is typically NOT needed and will degrade quality.
        Only enable via guided_filter=True kwarg when using older/coarser models
        (MiDaS, V2) that produce blocky ViT patch artifacts.

        Args:
            depth:     float32 [0,1] depth map at target resolution
            guide_bgr: uint8 BGR source image at same resolution
            r:         filter radius (10-16 works well for 1280-wide)
            eps:       regularisation - lower = sharper (1e-3 to 1e-4)

        Returns:
            float32 [0,1] depth with edge-aligned boundaries
        """
        h, w = depth.shape[:2]

        # Use luminance as guide (single channel, fast)
        guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # O(N) box-filter based guided filter (He et al. 2013)
        ksize = (2 * r + 1, 2 * r + 1)
        N = cv2.boxFilter(np.ones((h, w), np.float32), -1, ksize)

        mean_I  = cv2.boxFilter(guide,         -1, ksize) / N
        mean_p  = cv2.boxFilter(depth,         -1, ksize) / N
        mean_Ip = cv2.boxFilter(guide * depth, -1, ksize) / N
        mean_II = cv2.boxFilter(guide * guide, -1, ksize) / N

        cov_Ip = mean_Ip - mean_I * mean_p
        var_I  = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, ksize) / N
        mean_b = cv2.boxFilter(b, -1, ksize) / N

        return np.clip(mean_a * guide + mean_b, 0.0, 1.0)

    def unload(self):
        """Unload depth model from VRAM"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.transform is not None:
            del self.transform
            self.transform = None
        super().unload()