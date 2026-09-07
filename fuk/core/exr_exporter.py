# core/exr_exporter.py
"""
Multi-Layer EXR Exporter

Combines AOV layers (Beauty, Depth, Normals, Cryptomatte) into 
industry-standard multi-layer EXR files for compositing.

Supports:
- Multi-layer EXR with all AOVs in one file (export_multilayer)
- Individual single-layer EXRs per AOV (export_single_layers)
- Video/sequence export from latent (export_video_sequence - latent-only)
- 16-bit half or 32-bit float
- Various compression methods (ZIP, PIZ, DWAA, etc.)
- Linear/sRGB color space handling

LATENT WORKFLOW:
- export_video_sequence() requires beauty latent (true lossless path)
- export_multilayer() works from PNG/image files (standard path)
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Callable
import numpy as np
from PIL import Image
from enum import Enum
import subprocess
import tempfile
import shutil
import json


# Scale-free depth models return Z in [0, 1], where 0 is the nearest thing in
# frame rather than the camera. Comp depth tools work in scene units — a focus
# distance, a blur falloff over metres — so handing them a 0..1 buffer bunches
# every control against one end of its range and the result reads as a uniform
# blur over the whole frame. Map onto a plausible scene instead. Metric models
# (ZoeDepth) bypass this entirely and export real metres.
RELATIVE_Z_NEAR_M = 1.0
RELATIVE_Z_FAR_M = 100.0


class EXRCompression(str, Enum):
    """Available EXR compression methods"""
    NONE = "NONE"
    ZIP = "ZIP"           # Lossless, good compression
    PIZ = "PIZ"           # Lossless, wavelet-based
    PXR24 = "PXR24"       # Lossy, 24-bit
    B44 = "B44"           # Lossy, fast decode
    B44A = "B44A"         # Lossy, fast decode with alpha
    DWAA = "DWAA"         # Lossy, small files
    DWAB = "DWAB"         # Lossy, small files (tiled)


class EXRExporter:
    """
    Export AOV layers to multi-layer EXR files
    
    LATENT-ONLY VERSION:
    - Beauty pass MUST be provided as latent file
    - No MP4/PNG fallbacks (fail fast if latent missing)
    - Professional lossless pipeline only
    """
    
    def __init__(self):
        # Check for OpenEXR
        try:
            import OpenEXR
            import Imath
            self.OpenEXR = OpenEXR
            self.Imath = Imath
            self._has_openexr = True
        except ImportError:
            print("⚠ OpenEXR not installed. Install with: pip install OpenEXR --break-system-packages")
            self._has_openexr = False
    
    # ========================================================================
    # Video/Sequence Support - AOVs Only (not beauty)
    # ========================================================================
    
    def _is_video_file(self, path: Path) -> bool:
        """Check if path is a video file"""
        video_exts = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
        return path.suffix.lower() in video_exts
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams', '-show_format',
            str(video_path)
        ]
        
        try:
            import json
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Parse frame rate
            fps_str = video_stream.get('r_frame_rate', '24/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': fps,
                'frame_count': int(video_stream.get('nb_frames', 0)),
                'duration': float(data.get('format', {}).get('duration', 0)),
            }
        except Exception as e:
            print(f"⚠ Failed to get video info: {e}")
            return {'width': 0, 'height': 0, 'fps': 24, 'frame_count': 0, 'duration': 0}
    
    def _extract_frames(
        self, 
        video_path: Path, 
        output_dir: Path,
        frame_pattern: str = "frame_%04d.png"
    ) -> List[Path]:
        """Extract all frames from a video file (for AOVs only, not beauty)"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = output_dir / frame_pattern
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-fps_mode', 'passthrough',
            str(output_pattern)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Return sorted list of extracted frames
        frames = sorted(output_dir.glob("frame_*.png"))
        return frames
    
    def _find_raw_data_path(self, layer_path: Path, layer_name: str) -> Optional[Path]:
        """
        Look for raw .npy file alongside an MP4, still, or sequence directory.

        The batch preprocessors save these automatically:
        depth.mp4     -> depth_raw.npy  (same directory)
        depth_seq/    -> depth_seq/depth_raw.npy
        depth_xx.png  -> depth_xx_raw.npy  (stills)
        """
        if self._is_video_file(layer_path):
            raw_path = layer_path.parent / f"{layer_path.stem}_raw.npy"
            if raw_path.exists():
                return raw_path
            raw_path = layer_path.parent / f"{layer_name}_raw.npy"
            if raw_path.exists():
                return raw_path
        elif layer_path.is_dir():
            for name in [f'{layer_name}_raw.npy', 'depth_raw.npy', 'crypto_raw.npy', 'normals_raw.npy']:
                raw_path = layer_path / name
                if raw_path.exists():
                    return raw_path
        else:
            # Still image: float buffer written beside the 8-bit preview
            raw_path = layer_path.parent / f"{layer_path.stem}_raw.npy"
            if raw_path.exists():
                return raw_path
        return None
    
    def _load_depth_meta(self, raw_npy_path: Optional[Path]) -> Optional[dict]:
        """
        Read the sidecar that says a raw depth buffer holds true Z.

        Returns None for a pass made before the sidecar existed. Those buffers
        hold a display map (near = white, invert toggle baked in) — the
        inverse of a Z channel — and there is no way to tell from the array
        which orientation the model produced, so callers warn rather than
        guess and silently flip someone's geometry.
        """
        if raw_npy_path is None:
            return None
        try:
            meta = json.loads(raw_npy_path.with_suffix(".json").read_text())
        except (OSError, ValueError):
            return None
        return meta if meta.get("space") == "z" else None

    def _depth_to_scene_z(self, arr: np.ndarray, depth_meta: Optional[dict]) -> np.ndarray:
        """
        Put a raw depth buffer into the units an EXR Z channel is read in.

        Metric passes are already metres. Relative passes are [0,1] with 0 at
        the nearest surface in frame, which is an orientation but not a scale,
        so they are mapped onto an assumed scene depth.
        """
        arr = arr.astype(np.float32)
        if depth_meta is None:
            return arr
        if depth_meta.get("is_metric"):
            return arr
        return RELATIVE_Z_NEAR_M + arr * (RELATIVE_Z_FAR_M - RELATIVE_Z_NEAR_M)

    def _find_beauty_latent_path(self, beauty_path: Path) -> Optional[Path]:
        """
        Find the corresponding .latent.pt file for a beauty pass video/image.
        
        Search order:
            1. latents/{stem}.latent.pt  (exact match)
            2. latents/{base_name}.latent.pt  (without extension suffix)
            3. latents/generated.latent.pt  (standard generation name)
            4. First .latent.pt found  (fallback - there's usually only one)
        
        Returns:
            Path to .latent.pt file if found, None otherwise
        """
        latent_dir = beauty_path.parent / "latents"
        if not latent_dir.exists():
            return None
        
        # 1. Exact stem match
        latent_file = latent_dir / f"{beauty_path.stem}.latent.pt"
        if latent_file.exists():
            return latent_file
        
        # 2. Base name match
        base_name = beauty_path.stem.split('.')[0]
        latent_file = latent_dir / f"{base_name}.latent.pt"
        if latent_file.exists():
            return latent_file
        
        # 3. Standard generation name
        latent_file = latent_dir / "generated.latent.pt"
        if latent_file.exists():
            return latent_file
        
        # 4. Fallback: any .latent.pt
        latent_files = list(latent_dir.glob("*.latent.pt"))
        if latent_files:
            return latent_files[0]
        
        return None
    
    # How to reach the picture decoder of each model family, and what it
    # decodes to.  These differ in every respect that matters here:
    #
    #   attr / method  where the decoder lives on the pipeline and what its
    #                  decode entry point is called.  MiniMax-H3 keeps a raw
    #                  `decode` alongside `decode_video`, but only the latter
    #                  un-normalises the latent, so the method is named
    #                  explicitly rather than probed for.
    #   component      substring that picks the decoder's weights out of the
    #                  models.json component list.  LTX-2 ships its encoder and
    #                  decoder as separate files, and both audio-video families
    #                  ship an audio VAE that must not be picked up here.
    #   value_range    MiniMax-H3's VAE reverts an ImageNet normalisation and
    #                  clamps, so it hands back [0, 1]; every other family
    #                  decodes to [-1, 1].
    _VAE_FAMILIES = {
        "qwen":       {"attr": "vae", "method": "decode", "component": "vae", "value_range": (-1.0, 1.0)},
        "wan":        {"attr": "vae", "method": "decode", "component": "vae", "value_range": (-1.0, 1.0)},
        "flux2":      {"attr": "vae", "method": "decode", "component": "vae", "value_range": (-1.0, 1.0)},
        "krea2":      {"attr": "vae", "method": "decode", "component": "vae", "value_range": (-1.0, 1.0)},
        "minimax_h3": {"attr": "video_vae", "method": "decode_video",
                       "component": "video_vae", "value_range": (0.0, 1.0)},
        "ltx2":       {"attr": "video_vae_decoder", "method": "decode",
                       "component": "video_vae_decoder", "value_range": (-1.0, 1.0)},
    }

    # Latent channel count → pipeline family, for latents captured before the
    # hook started recording which model produced them.  Qwen-Image, Wan and
    # Krea-2 all use 16-channel VAEs, so tensor rank breaks the tie: a 5-D
    # latent is video (Wan), a 4-D one is a still (Qwen).
    _LATENT_CHANNEL_FAMILIES = {
        (16, False): "qwen",
        (32, False): "flux2",
        (16, True): "wan",
        (24, True): "minimax_h3",
        (128, True): "ltx2",
    }

    def _vae_family(self, backend, model_type: str) -> dict:
        """Family spec for a model type, or a clear error naming the pipeline."""
        pipeline_type = backend.get_model_entry(model_type).get("pipeline", "qwen")
        spec = self._VAE_FAMILIES.get(pipeline_type)
        if spec is None:
            raise ValueError(
                f"'{model_type}' uses the '{pipeline_type}' pipeline, which the EXR "
                f"exporter has no VAE decode path for yet."
            )
        return spec

    def _resolve_latent_model(self, backend, latent_data, latent, is_video: bool) -> str:
        """Work out which model's VAE decodes this latent.

        Prefers the model_type stamped in by the capture hook.  Latents saved
        before that existed carry no provenance, so fall back to the channel
        count — decoding a 24-channel MiniMax latent with Wan's 16-channel VAE
        fails on a tensor size mismatch, which is what this avoids.
        """
        recorded = latent_data.get('model_type') if isinstance(latent_data, dict) else None
        if recorded:
            try:
                resolved = backend.resolve_model_type(recorded)
                self._vae_family(backend, resolved)   # reject unsupported families here
                return resolved
            except ValueError as e:
                print(f"  ⚠️  Recorded model '{recorded}' unusable ({e}) — inferring from latent")

        channels = latent.shape[1]
        family = self._LATENT_CHANNEL_FAMILIES.get((channels, is_video))
        if family is None:
            raise ValueError(
                f"Cannot tell which VAE decodes this latent: {channels} channels, "
                f"{'video' if is_video else 'image'}, and the file records no model. "
                f"Pass model_type explicitly."
            )

        candidates = [k for k, v in backend.models_config.items()
                      if isinstance(v, dict) and v.get("pipeline") == family]
        if not candidates:
            raise ValueError(
                f"Latent looks like a '{family}' latent ({channels} channels) but no "
                f"'{family}' model is registered in models.json."
            )
        print(f"  📄 No model recorded in latent — inferred '{candidates[0]}' "
              f"from {channels} channels")
        return candidates[0]

    def _load_vae_only(self, backend, model_type: str):
        """
        Load ONLY the picture VAE for a model — no DiT, no text encoder.

        Loads the single decoder component through DiffSynth's model pool
        rather than the pipeline's from_pretrained, which would also pull in
        tokenizers and processors this path never uses.

        Returns:
            (vae, decode_fn, value_range) — the module, a callable taking
            (latent, device) that returns a decoded tensor, and the [min, max]
            range that tensor comes back in.
        """
        import inspect
        import torch

        spec = self._vae_family(backend, model_type)
        attr, method_name = spec["attr"], spec["method"]

        def _wrap(module):
            decode = getattr(module, method_name)
            params = inspect.signature(decode).parameters

            def decode_fn(latent, device):
                kwargs = {}
                if 'device' in params:
                    kwargs['device'] = device
                # Wan and Qwen tile only on request and are faster untiled; the
                # audio-video VAEs default to tiling and need it at these sizes.
                if 'tiled' in params and params['tiled'].default is False:
                    kwargs['tiled'] = False
                return decode(latent, **kwargs)

            return module, decode_fn, spec["value_range"]

        # 1. Check if a pipeline is already cached — grab its VAE for free.
        # getattr is guarded because pipelines are nn.Modules, whose __getattr__
        # raises AttributeError rather than returning the default.
        for key, pipe in backend.pipelines.items():
            if not key.startswith(f"{model_type}:"):
                continue
            try:
                module = getattr(pipe, attr, None)
            except AttributeError:
                module = None
            if module is not None:
                print(f"  📄 Reusing {attr} from cached pipeline: {key}")
                return _wrap(module)

        # 2. No cached pipeline — load the decoder component on its own
        entry = backend.get_model_entry(model_type)
        primary_id = entry["model_id"]
        pipeline_type = entry["pipeline"]

        want = spec["component"]
        vae_comps = [c for c in entry.get("components", [])
                     if want in c["pattern"].lower()]
        if not vae_comps:
            raise ValueError(
                f"No '{want}' component in models.json for '{model_type}'")

        comp = vae_comps[0]
        mid = comp.get("model_id", primary_id)

        print(f"  📄 Loading VAE-only: {mid} / {comp['pattern']}")

        # Asked of the backend instance rather than imported: importing the
        # registry by name re-executes diffsynth_backend under a second module
        # name and yields an empty one.
        PipelineCls = backend.get_pipeline_class(pipeline_type)
        if PipelineCls is None:
            raise RuntimeError(
                f"Pipeline class for '{pipeline_type}' not registered — the backend "
                f"has not finished initialising.")
        MC = backend._get_model_config_class(entry)

        # Pre-quantized checkpoints (the MiniMax-H3 NF4 weights) carry their own
        # quant config and cannot be loaded or cast to float32 — bitsandbytes
        # 4-bit tensors only dequantize on the GPU. Load those at bf16 and leave
        # the cast to the caller.
        prequantized = "nf4" in comp["pattern"].lower() or "int8" in comp["pattern"].lower()
        dtype = torch.bfloat16 if prequantized else torch.float32

        shell = PipelineCls(device="cpu", torch_dtype=dtype)
        pool = shell.download_and_load_models(
            [MC(model_id=mid, origin_file_pattern=comp["pattern"])])
        if not pool.model:
            raise RuntimeError(f"VAE failed to load from {mid}/{comp['pattern']}")

        vae = pool.model[0]
        print(f"  📄 VAE loaded to CPU ({type(vae).__name__}, {dtype})")
        return _wrap(vae)

    @staticmethod
    def _is_quantized(module) -> bool:
        """True if the module holds bitsandbytes 4-bit weights.

        Those cannot be cast to float32 or run on the CPU, so the decode paths
        have to keep them in their native dtype and stay on the GPU.
        """
        return any(type(p).__name__ == "Params4bit" for p in module.parameters())

    def _decode_beauty_latents(
        self,
        latent_path: Path,
        backend,
        model_type: str = "auto",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        bracketed: bool = False,
        scales: Optional[List[float]] = None,
        noise_bracketed: bool = False,
        sigmas: Optional[List[float]] = None,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """
        Decode beauty pass latents to float32 linear numpy arrays.

        When bracketed or noise_bracketed is True, decodes the latent multiple
        times with scale or noise perturbations and fuses each frame with Mertens
        before linearising.  The VAE is loaded once and reused for all passes.
        """
        import torch

        if scales is None:
            scales = [0.85, 1.0, 1.15]
        if sigmas is None:
            sigmas = [0.0, 0.025, 0.05]

        print(f"  📄 Decoding beauty latents: {latent_path.name}")

        latent_data = torch.load(str(latent_path), map_location='cpu', weights_only=False)
        latent = latent_data['latent'] if isinstance(latent_data, dict) else latent_data
        is_video = latent.ndim == 5

        if model_type == "auto":
            model_type = self._resolve_latent_model(backend, latent_data, latent, is_video)
        else:
            model_type = backend.resolve_model_type(model_type)

        decode_mode = ("noise" if noise_bracketed else "scale" if bracketed else "standard")
        print(f"  📄 Latent shape: {latent.shape}, model: {model_type}, decode: {decode_mode}")

        vae, decode_fn, (v_min, v_max) = self._load_vae_only(backend, model_type)
        vae = vae.eval()
        # Pre-quantized (NF4) VAEs cannot be cast or run on the CPU — leave them
        # in their native dtype and match the latent to it.
        quantized = self._is_quantized(vae)
        work_dtype = next(vae.parameters()).dtype if quantized else torch.float32
        if not quantized:
            vae = vae.to(dtype=torch.float32)
        latent = latent.to(dtype=work_dtype)

        def _vae_decode(lat):
            """GPU-first, CPU fallback. Returns float32 tensor."""
            def _run(device):
                vae.to(device)
                l = lat.to(device)
                with torch.no_grad():
                    return decode_fn(l, device)

            result = None
            if torch.cuda.is_available():
                try:
                    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                    print(f"  📄 GPU decode ({free_gb:.1f} GB free)...")
                    result = _run(torch.device('cuda'))
                    print(f"  📄 GPU decode complete")
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if quantized:
                        raise
                    print(f"  ⚠️  GPU failed: {str(e).split(chr(10))[0][:80]}, falling back to CPU...")
                    vae.cpu()
                    torch.cuda.empty_cache()
            if result is None:
                if quantized:
                    raise RuntimeError(
                        f"'{model_type}' ships a 4-bit quantized VAE, which only "
                        f"dequantizes on the GPU — no CUDA device is available.")
                import os
                torch.set_num_threads(os.cpu_count() or 4)
                print(f"  📄 CPU decode...")
                result = _run(torch.device('cpu'))
                print(f"  📄 CPU decode complete")
            if result.dtype != torch.float32:
                result = result.to(torch.float32)
            return result

        def _tensor_to_frames(decoded):
            """Convert decoded tensor → list of HWC float32 frames in sRGB [0, 1].

            Most DiffSynth VAEs decode to [-1, 1], but MiniMax-H3's reverts an
            ImageNet normalisation and hands back [0, 1]. Remap from whichever
            this family uses before clipping, or half the tonal range collapses
            to black and contrast doubles.
            """
            pixels = (decoded.cpu().numpy() - v_min) / (v_max - v_min)
            frames = []
            if is_video and pixels.ndim == 5:
                for fi in range(pixels.shape[2]):
                    f = np.transpose(pixels[0, :, fi, :, :], (1, 2, 0))
                    frames.append(np.clip(f[:, :, :3], 0.0, 1.0).astype(np.float32))
            elif pixels.ndim == 4:
                f = pixels[0]
                if f.shape[0] in (1, 3, 4):
                    f = np.transpose(f, (1, 2, 0))
                frames.append(np.clip(f[:, :, :3], 0.0, 1.0).astype(np.float32))
            else:
                raise ValueError(f"Unexpected decoded shape: {pixels.shape}")
            return frames

        # Build bracket passes
        if noise_bracketed:
            rng = torch.Generator()
            rng.manual_seed(seed)
            passes = []
            for sigma in sigmas:
                label = f"σ={sigma:.3f}"
                print(f"  [BRACKET] noise {label}")
                lat = latent if sigma == 0.0 else latent + torch.randn(latent.shape, dtype=latent.dtype, generator=rng) * sigma
                passes.append(_tensor_to_frames(_vae_decode(lat)))
        elif bracketed:
            passes = []
            for s in scales:
                print(f"  [BRACKET] scale {s:.2f}×")
                passes.append(_tensor_to_frames(_vae_decode(latent * s)))
        else:
            passes = [_tensor_to_frames(_vae_decode(latent))]

        # Release VAE
        vae.cpu()
        del vae, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fuse brackets and linearise
        num_frames = len(passes[0])
        if len(passes) > 1:
            import cv2
            merge_mertens = cv2.createMergeMertens()
            decoded_frames = []
            print(f"  [BRACKET] Mertens fusing {num_frames} frames × {len(passes)} brackets...")
            for fi in range(num_frames):
                imgs_u8 = [(p[fi] * 255).astype(np.uint8) for p in passes]
                fused = np.clip(merge_mertens.process(imgs_u8), 0.0, 1.0).astype(np.float32)
                decoded_frames.append(self._srgb_to_linear(fused))
                if progress_callback and (fi + 1) % 10 == 0:
                    progress_callback(fi + 1, num_frames)
        else:
            decoded_frames = []
            for fi, frame in enumerate(passes[0]):
                decoded_frames.append(self._srgb_to_linear(frame))
                if progress_callback and (fi + 1) % 10 == 0:
                    progress_callback(fi + 1, num_frames)

        print(f"  ✅ {len(decoded_frames)} beauty frames decoded ({decode_mode})")
        return decoded_frames

    
    def export_video_sequence(
        self,
        beauty_latent: Path,
        aov_layers: Dict[str, str],
        backend,
        output_dir: Path,
        filename_pattern: str = "frame.{frame:04d}.exr",
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        start_frame: int = 1,
        model_type: str = "auto",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        bracketed: bool = False,
        scales: Optional[List[float]] = None,
        noise_bracketed: bool = False,
        sigmas: Optional[List[float]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Export video sequence to multilayer EXR (LATENT-ONLY VERSION)
        
        Args:
            beauty_latent: Path to .latent.pt file (REQUIRED)
            aov_layers: Dict of AOV layer paths (depth, normals, crypto)
                Example: {'depth': 'depth.mp4', 'normals': 'normals.mp4'}
            backend: DiffSynthBackend instance (REQUIRED)
            output_dir: Output directory for EXR sequence
            filename_pattern: Pattern with {frame:04d} placeholder
            bit_depth: 16 or 32
            compression: Compression method
            start_frame: Starting frame number
            model_type: Model type for decoding ("auto", or explicit pipeline key)
            progress_callback: Optional callback(current_frame, total_frames)
            
        Returns:
            Dict with sequence info
            
        Raises:
            ValueError: If beauty_latent doesn't exist or backend is None
        """
        if not self._has_openexr:
            raise RuntimeError("OpenEXR not installed")
        
        # Validate required parameters
        beauty_latent = Path(beauty_latent)
        if not beauty_latent.exists():
            raise ValueError(f"Beauty latent not found: {beauty_latent}")
        
        if backend is None:
            raise ValueError("Backend is required for latent decoding")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Exporting EXR Sequence (LATENT-ONLY)")
        print(f"{'='*60}")
        print(f"Beauty Latent: {beauty_latent}")
        print(f"AOV Layers: {list(aov_layers.keys())}")
        print(f"Output: {output_dir}")
        print(f"Pattern: {filename_pattern}")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"{'='*60}\n")
        
        # Decode beauty latents
        beauty_decoded_frames = self._decode_beauty_latents(
            latent_path=beauty_latent,
            backend=backend,
            model_type=model_type,
            progress_callback=None,
            bracketed=bracketed,
            scales=scales,
            noise_bracketed=noise_bracketed,
            sigmas=sigmas,
            seed=seed,
        )
        
        total_frames = len(beauty_decoded_frames)
        print(f"\n  ✅ Beauty: {total_frames} frames from latent (lossless)")
        
        # Process AOV layers
        frame_counts = {}
        layer_paths = {}
        raw_data = {}
        depth_meta = None

        for layer_name, layer_path in aov_layers.items():
            if layer_path is None:
                continue
            
            layer_path = Path(layer_path)
            if not layer_path.exists():
                print(f"  ⚠ Skipping {layer_name}: path not found")
                continue
            
            layer_paths[layer_name] = layer_path
            
            # Check for raw .npy data (lossless AOVs)
            raw_npy_path = self._find_raw_data_path(layer_path, layer_name)
            if raw_npy_path:
                try:
                    raw_array = np.load(str(raw_npy_path))
                    raw_data[layer_name] = raw_array
                    frame_counts[layer_name] = len(raw_array)
                    print(f"  ✅ {layer_name}: {len(raw_array)} frames from RAW .npy (lossless)")
                    if layer_name == 'depth':
                        depth_meta = self._load_depth_meta(raw_npy_path)
                        if depth_meta is None:
                            print("     ⚠ no Z sidecar — this pass predates Z-correct "
                                  "export and holds a display map (near=white, invert "
                                  "baked in). Re-run the depth pass for a usable Z.")
                        else:
                            units = 'metres' if depth_meta.get('is_metric') else 'relative'
                            print(f"     true Z ({units})")
                    continue
                except Exception as e:
                    print(f"  ⚠ {layer_name}: raw .npy failed ({e}), falling back to MP4")
            
            # Fallback to video extraction for AOVs (not ideal but acceptable)
            if self._is_video_file(layer_path):
                info = self._get_video_info(layer_path)
                frame_counts[layer_name] = info['frame_count']
                print(f"  ⚠ {layer_name}: {info['frame_count']} frames from MP4 (lossy!)")
            elif layer_path.is_dir():
                frames = sorted(layer_path.glob("*.png")) + sorted(layer_path.glob("*.jpg"))
                frame_counts[layer_name] = len(frames)
                print(f"  ✅ {layer_name}: {len(frames)} frames (sequence)")
        
        # Verify frame counts match
        if frame_counts:
            min_frames = min(frame_counts.values())
            if min_frames != total_frames:
                print(f"\n  ⚠ Warning: Frame count mismatch!")
                print(f"     Beauty: {total_frames} frames")
                for name, count in frame_counts.items():
                    print(f"     {name}: {count} frames")
                total_frames = min(total_frames, min_frames)
                print(f"     Using: {total_frames} frames (minimum)")
        
        print(f"\n  Processing {total_frames} frames...")
        
        # Create temp directory for AOV frame extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="exr_export_"))
        
        try:
            # Extract AOV frames from videos (only if no raw .npy)
            extracted_frames = {}
            
            for layer_name, layer_path in layer_paths.items():
                if layer_name in raw_data:
                    continue  # We have raw numpy data
                if self._is_video_file(layer_path):
                    print(f"  Extracting {layer_name} frames...")
                    layer_temp = temp_dir / layer_name
                    frames = self._extract_frames(layer_path, layer_temp)
                    extracted_frames[layer_name] = frames
                elif layer_path.is_dir():
                    frames = sorted(layer_path.glob("*.png")) + sorted(layer_path.glob("*.jpg"))
                    extracted_frames[layer_name] = frames
            
            # Process frame by frame
            exported_frames = []
            total_size = 0
            
            for frame_idx in range(total_frames):
                frame_num = start_frame + frame_idx
                
                # Build AOV layer dict for this frame
                frame_layers = {}
                for layer_name, frames in extracted_frames.items():
                    if frame_idx < len(frames):
                        frame_layers[layer_name] = str(frames[frame_idx])
                
                # Build raw array dict for this frame (including beauty!)
                frame_raw = {}
                
                # Add decoded beauty frame
                if frame_idx < len(beauty_decoded_frames):
                    frame_raw['beauty'] = beauty_decoded_frames[frame_idx]
                
                # Add AOV raw data
                for layer_name, arr in raw_data.items():
                    if frame_idx < len(arr):
                        frame_raw[layer_name] = arr[frame_idx]
                
                # Generate output filename
                output_filename = filename_pattern.format(frame=frame_num)
                output_path = output_dir / output_filename
                
                # Export this frame
                try:
                    result = self._export_frame_multilayer(
                        layers=frame_layers,  # AOVs from extracted files
                        output_path=output_path,
                        bit_depth=bit_depth,
                        compression=compression,
                        quiet=True,
                        raw_arrays=frame_raw,  # Beauty + AOVs from raw data
                        depth_meta=depth_meta,
                    )
                    
                    exported_frames.append(output_path)
                    total_size += result.get('file_size', 0)
                    
                except Exception as e:
                    print(f"  ⚠ Frame {frame_num} failed: {e}")
                
                # Progress callback
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)
                
                # Progress indicator
                if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
                    print(f"  Exported {frame_idx + 1}/{total_frames} frames...")
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"\n✅ Exported EXR sequence")
        print(f"  Frames: {len(exported_frames)}")
        print(f"  Total Size: {total_size_mb:.2f} MB")
        print(f"  Output: {output_dir}")
        
        return {
            "output_dir": str(output_dir),
            "frame_count": len(exported_frames),
            "start_frame": start_frame,
            "end_frame": start_frame + len(exported_frames) - 1,
            "filename_pattern": filename_pattern,
            "total_size": total_size,
            "total_size_mb": total_size_mb,
            "bit_depth": bit_depth,
            "compression": compression,
            "layers_included": ['beauty'] + list(aov_layers.keys()),
            "frames": [str(f) for f in exported_frames],
        }
    
    # ==================================================================
    # Cryptomatte Helpers (Psyop Cryptomatte Spec 1.2)
    # ==================================================================
    
    @staticmethod
    def _mm3_hash(name: str) -> int:
        """
        MurmurHash3_32 — the standard hash used by the Cryptomatte spec.
        Pure Python implementation, no external deps.
        """
        import struct as _struct
        
        key = name.encode('utf-8')
        length = len(key)
        seed = 0
        c1 = 0xcc9e2d51
        c2 = 0x1b873593
        h1 = seed
        
        # Body (process 4-byte chunks)
        n_blocks = length // 4
        for i in range(n_blocks):
            k1 = _struct.unpack_from('<I', key, i * 4)[0]
            k1 = (k1 * c1) & 0xFFFFFFFF
            k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
            k1 = (k1 * c2) & 0xFFFFFFFF
            h1 ^= k1
            h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
            h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF
        
        # Tail
        tail_start = n_blocks * 4
        k1 = 0
        tail_len = length & 3
        if tail_len >= 3:
            k1 ^= key[tail_start + 2] << 16
        if tail_len >= 2:
            k1 ^= key[tail_start + 1] << 8
        if tail_len >= 1:
            k1 ^= key[tail_start]
            k1 = (k1 * c1) & 0xFFFFFFFF
            k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
            k1 = (k1 * c2) & 0xFFFFFFFF
            h1 ^= k1
        
        # Finalization
        h1 ^= length
        h1 ^= (h1 >> 16)
        h1 = (h1 * 0x85ebca6b) & 0xFFFFFFFF
        h1 ^= (h1 >> 13)
        h1 = (h1 * 0xc2b2ae35) & 0xFFFFFFFF
        h1 ^= (h1 >> 16)
        
        return h1
    
    @staticmethod
    def _uint32_to_float32(val: int) -> float:
        """Bit-cast uint32 -> float32 (NOT value conversion)."""
        import struct as _struct
        return _struct.unpack('f', _struct.pack('I', val & 0xFFFFFFFF))[0]
    
    def _build_cryptomatte_channels(
        self,
        id_matte: np.ndarray,
        num_objects: int = 0,
        quiet: bool = False,
    ) -> tuple:
        """
        Convert raw object-ID matte into spec-compliant Cryptomatte channels.
        
        Cryptomatte spec (Psyop 1.2):
          - RGBA layers: R=ID1(float), G=cov1, B=ID2(float), A=cov2
          - IDs are MurmurHash3_32 of object name, bit-cast to float32
          - Channel names: CryptoObject00.R/G/B/A
          - MUST be 32-bit float (half precision corrupts hashes)
          - Manifest: JSON mapping name -> hex hash
        
        Args:
            id_matte: uint16 [H, W] with 0=background, 1..N=objects
            num_objects: max object count (0 = auto-detect from data)
        
        Returns:
            (channels_dict, metadata_dict)
        """
        h, w = id_matte.shape[:2]
        if id_matte.ndim > 2:
            id_matte = id_matte[:, :, 0]
        
        # Determine objects present
        unique_ids = np.unique(id_matte)
        unique_ids = unique_ids[unique_ids > 0]  # exclude background
        max_id = int(unique_ids.max()) if len(unique_ids) > 0 else num_objects
        
        # Build manifest: object names -> MurmurHash3 -> hex + float
        id_to_hash_float = {}
        manifest = {}
        
        for obj_id in range(1, max_id + 1):
            name = f"object_{obj_id:03d}"
            hash_uint = self._mm3_hash(name)
            hash_float = self._uint32_to_float32(hash_uint)
            hex_hash = f"{hash_uint:08x}"
            
            id_to_hash_float[obj_id] = hash_float
            manifest[name] = hex_hash
        
        # Build per-pixel ID and coverage arrays
        # Single-winner matte: each pixel has 1 object with coverage=1.0
        id_float_map = np.zeros(max_id + 1, dtype=np.float32)
        for obj_id, hf in id_to_hash_float.items():
            id_float_map[obj_id] = hf
        
        safe_ids = np.clip(id_matte.astype(np.int32), 0, max_id)
        
        # CryptoObject00: R=ID, G=coverage, B=0, A=0 (2nd pair empty)
        channels = {
            'CryptoObject00.R': id_float_map[safe_ids],
            'CryptoObject00.G': np.where(safe_ids > 0, 1.0, 0.0).astype(np.float32),
            'CryptoObject00.B': np.zeros((h, w), dtype=np.float32),
            'CryptoObject00.A': np.zeros((h, w), dtype=np.float32),
        }
        
        # Metadata for EXR header
        layer_hash = f"{self._mm3_hash('CryptoObject'):08x}"
        metadata = {
            f'cryptomatte/{layer_hash}/name': 'CryptoObject',
            f'cryptomatte/{layer_hash}/hash': 'MurmurHash3_32',
            f'cryptomatte/{layer_hash}/conversion': 'uint32_to_float32',
            f'cryptomatte/{layer_hash}/manifest': json.dumps(manifest),
        }
        
        if not quiet:
            print(f"  [Crypto] Cryptomatte: {len(unique_ids)} unique objects, "
                  f"manifest: {len(manifest)} entries")
        
        return channels, metadata

    def _add_depth_channels(
        self,
        channels_dict: dict,
        channel_info: dict,
        arr: np.ndarray,
        bit_depth: int,
        pixel_type,
    ) -> None:
        """
        Write depth under both a bare 'Z' and a layered 'depth.Z'.

        A channel's layer is everything before the last dot, so a bare 'Z' has
        no layer and lands in the base layer next to R/G/B. Nuke and Fusion map
        that onto their depth aux buffer automatically, but Resolve enumerates
        only prefixed names — so with 'Z' alone, normals and crypto show up in
        its layer picker and depth silently does not, appearing instead as an
        extra channel of the beauty layer. 'depth.Z' (the Nuke/Arnold spelling)
        gives Resolve a layer to select; the bare 'Z' stays for everyone else.
        """
        payload = self._to_bytes(arr, bit_depth)
        for name in ('Z', 'depth.Z'):
            channels_dict[name] = payload
            channel_info[name] = self.Imath.Channel(pixel_type)

    # Codecs that quantise FLOAT channels. Fine for beauty, fatal for
    # Cryptomatte: the ID channels carry MurmurHash3 bit patterns reinterpreted
    # as floats, so any lossy pass turns a handful of exact object IDs into
    # thousands of near-miss values and the mattes stop matching anything.
    # (B44/B44A are absent deliberately — they only touch HALF channels, and
    # crypto is always written FLOAT.)
    _FLOAT_LOSSY = {'PXR24', 'DWAA', 'DWAB'}

    def _compression_attr(self, compression: str, has_cryptomatte: bool = False):
        """
        Map a compression name onto an Imath.Compression header attribute.

        Unknown names fall back to ZIP rather than raising — a bad preset
        should not cost someone a long sequence export. Same for a lossy pick
        on a file carrying Cryptomatte, which would destroy the ID channels.
        """
        table = {
            'NONE': 'NO_COMPRESSION',
            'RLE': 'RLE_COMPRESSION',
            'ZIPS': 'ZIPS_COMPRESSION',
            'ZIP': 'ZIP_COMPRESSION',
            'PIZ': 'PIZ_COMPRESSION',
            'PXR24': 'PXR24_COMPRESSION',
            'B44': 'B44_COMPRESSION',
            'B44A': 'B44A_COMPRESSION',
            'DWAA': 'DWAA_COMPRESSION',
            'DWAB': 'DWAB_COMPRESSION',
        }
        key = str(compression).upper().replace('_COMPRESSION', '')
        attr = table.get(key)
        if attr is None:
            print(f"  ⚠ Unknown EXR compression '{compression}', using ZIP")
            attr = 'ZIP_COMPRESSION'
        elif has_cryptomatte and key in self._FLOAT_LOSSY:
            print(f"  ⚠ {key} would corrupt the Cryptomatte ID channels, using ZIP")
            attr = 'ZIP_COMPRESSION'
        return self.Imath.Compression(getattr(self.Imath.Compression, attr))

    def _export_frame_multilayer(
        self,
        layers: Dict[str, str],
        output_path: Path,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        linear: bool = True,
        quiet: bool = False,
        raw_arrays: Optional[Dict[str, np.ndarray]] = None,
        depth_meta: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Internal method to export a single frame (used by sequence export).
        Same as export_multilayer but with quiet option for batch processing.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load all layers
        loaded_layers = {}
        width, height = None, None

        # Process raw arrays first (lossless, no file I/O)
        if raw_arrays:
            for layer_name, arr in raw_arrays.items():
                if width is None:
                    height, width = arr.shape[:2] if arr.ndim >= 2 else (0, 0)
                
                if layer_name == 'beauty':
                    # Decoded latent: float32 [H,W,3] range [0,1], already in linear space
                    if arr.ndim == 3 and arr.shape[2] in [3, 4]:
                        loaded_layers['beauty'] = arr[:,:,:3].astype(np.float32)
                        if arr.shape[2] == 4:
                            loaded_layers['alpha'] = arr[:, :, 3].astype(np.float32)
                    elif arr.ndim == 2:
                        loaded_layers['beauty'] = np.stack([arr, arr, arr], axis=-1).astype(np.float32)

                elif layer_name == 'depth':
                    # Already float32 [H,W] range [0,1]
                    loaded_layers['depth'] = arr.astype(np.float32) if arr.ndim == 2 else arr[:,:,0].astype(np.float32)
                    
                elif layer_name == 'normals':
                    # Already float32 [H,W,3] range [-1,1] — no conversion needed
                    loaded_layers['normals_raw'] = arr[:,:,:3].astype(np.float32)
                    
                elif layer_name == 'crypto':
                    # uint16 [H,W] object IDs — store as float for EXR
                    loaded_layers['crypto_raw'] = arr.astype(np.float32)
            
            # Resize any layers that don't match the established dimensions
            # (beauty from latent decode may differ slightly from preprocessor .npy)
            if width is not None and height is not None:
                for key, arr in loaded_layers.items():
                    arr_h, arr_w = arr.shape[:2]
                    if arr_h != height or arr_w != width:
                        from PIL import Image as _PILImage
                        # Crypto/ID mattes MUST use nearest-neighbor (no interpolation)
                        is_id_data = 'crypto' in key
                        resample = _PILImage.NEAREST if is_id_data else _PILImage.LANCZOS
                        if arr.ndim == 2:
                            img = _PILImage.fromarray(arr, mode='F')
                            img = img.resize((width, height), resample)
                            loaded_layers[key] = np.array(img).astype(np.float32)
                        else:
                            # Multi-channel: resize each channel
                            channels = []
                            for c in range(arr.shape[2]):
                                img = _PILImage.fromarray(arr[:,:,c], mode='F')
                                img = img.resize((width, height), resample)
                                channels.append(np.array(img).astype(np.float32))
                            loaded_layers[key] = np.stack(channels, axis=-1)

        for layer_name, layer_path in layers.items():
            if layer_name in loaded_layers:
                continue  # Already have from raw_arrays
            if layer_path is None:
                continue
                
            layer_path = Path(layer_path)
            if not layer_path.exists():
                continue
            
            img = Image.open(layer_path)
            arr = np.array(img).astype(np.float32) / 255.0
            
            if width is None:
                height, width = arr.shape[:2]

            if layer_name == 'beauty':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[2] == 4:
                    # Alpha is linear coverage, not colour — never gamma it
                    loaded_layers['alpha'] = arr[:, :, 3].copy()
                    arr = arr[:, :, :3]
                if linear:
                    arr = self._srgb_to_linear(arr)
                loaded_layers['beauty'] = arr

            elif layer_name == 'depth':
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                loaded_layers['depth'] = arr

            elif layer_name == 'normals':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                loaded_layers['normals'] = arr

            elif layer_name == 'crypto':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                loaded_layers['crypto'] = arr

        if not loaded_layers:
            raise ValueError("No valid layers to export")

        # Build EXR channels
        channels_dict = {}
        channel_info = {}

        pixel_type = (
            self.Imath.PixelType(self.Imath.PixelType.HALF)
            if bit_depth == 16
            else self.Imath.PixelType(self.Imath.PixelType.FLOAT)
        )
        
        if 'beauty' in loaded_layers:
            arr = loaded_layers['beauty']
            channels_dict['R'] = self._to_bytes(arr[:, :, 0], bit_depth)
            channels_dict['G'] = self._to_bytes(arr[:, :, 1], bit_depth)
            channels_dict['B'] = self._to_bytes(arr[:, :, 2], bit_depth)
            channel_info['R'] = self.Imath.Channel(pixel_type)
            channel_info['G'] = self.Imath.Channel(pixel_type)
            channel_info['B'] = self.Imath.Channel(pixel_type)

            # Always write A. Without it comps read the base layer as RGB and
            # synthesise their own alpha, and 'Z' next to three channels is
            # what made depth look like a stray extra colour channel.
            alpha = loaded_layers.get('alpha')
            if alpha is None:
                alpha = np.ones(arr.shape[:2], dtype=np.float32)
            channels_dict['A'] = self._to_bytes(alpha, bit_depth)
            channel_info['A'] = self.Imath.Channel(pixel_type)

        if 'depth' in loaded_layers:
            self._add_depth_channels(
                channels_dict, channel_info,
                self._depth_to_scene_z(loaded_layers['depth'], depth_meta),
                bit_depth, pixel_type,
            )

        # Normals: prefer raw (already [-1,1]) over PNG (needs conversion)
        if 'normals_raw' in loaded_layers:
            arr = loaded_layers['normals_raw']
            channels_dict['N.X'] = self._to_bytes(arr[:, :, 0], bit_depth)
            channels_dict['N.Y'] = self._to_bytes(arr[:, :, 1], bit_depth)
            channels_dict['N.Z'] = self._to_bytes(arr[:, :, 2], bit_depth)
            channel_info['N.X'] = self.Imath.Channel(pixel_type)
            channel_info['N.Y'] = self.Imath.Channel(pixel_type)
            channel_info['N.Z'] = self.Imath.Channel(pixel_type)
        elif 'normals' in loaded_layers:
            arr = loaded_layers['normals']
            arr_decoded = arr * 2.0 - 1.0
            channels_dict['N.X'] = self._to_bytes(arr_decoded[:, :, 0], bit_depth)
            channels_dict['N.Y'] = self._to_bytes(arr_decoded[:, :, 1], bit_depth)
            channels_dict['N.Z'] = self._to_bytes(arr_decoded[:, :, 2], bit_depth)
            channel_info['N.X'] = self.Imath.Channel(pixel_type)
            channel_info['N.Y'] = self.Imath.Channel(pixel_type)
            channel_info['N.Z'] = self.Imath.Channel(pixel_type)

        if 'crypto_raw' in loaded_layers:
            # Use spec-compliant Cryptomatte via _build_cryptomatte_channels
            # Raw data is uint16 object IDs — need MurmurHash3 + manifest
            id_matte = loaded_layers['crypto_raw']
            if id_matte.ndim > 2:
                id_matte = id_matte[:, :, 0]
            crypto_channels, crypto_metadata = self._build_cryptomatte_channels(
                id_matte.astype(np.uint16),
                quiet=quiet,
            )
            # Cryptomatte MUST be 32-bit float (half corrupts hash bit-patterns)
            float_type = self.Imath.PixelType(self.Imath.PixelType.FLOAT)
            for ch_name, ch_data in crypto_channels.items():
                channels_dict[ch_name] = ch_data.astype(np.float32).tobytes()
                channel_info[ch_name] = self.Imath.Channel(float_type)
        elif 'crypto' in loaded_layers:
            arr = loaded_layers['crypto']
            r_channel = arr[:, :, 0] if arr.ndim == 3 else arr
            id_matte = (r_channel * 255.0).round().astype(np.uint16)
            crypto_channels, crypto_metadata = self._build_cryptomatte_channels(id_matte, quiet=quiet)
            float_type = self.Imath.PixelType(self.Imath.PixelType.FLOAT)
            for ch_name, ch_data in crypto_channels.items():
                channels_dict[ch_name] = ch_data.astype(np.float32).tobytes()
                channel_info[ch_name] = self.Imath.Channel(float_type)
        else:
            crypto_metadata = {}
        
        # Create and write EXR
        header = self.OpenEXR.Header(width, height)
        header['channels'] = channel_info
        header['compression'] = self._compression_attr(
            compression, has_cryptomatte=bool(crypto_metadata)
        )

        # Write Cryptomatte manifest metadata into header (required by spec)
        for meta_key, meta_val in crypto_metadata.items():
            header[meta_key] = meta_val.encode('utf-8') if isinstance(meta_val, str) else meta_val
        
        exr_file = self.OpenEXR.OutputFile(str(output_path), header)
        exr_file.writePixels(channels_dict)
        exr_file.close()
        
        file_size = output_path.stat().st_size
        
        if not quiet:
            print(f"✅ Exported: {output_path.name} ({file_size / (1024*1024):.2f} MB)")
            print(f"   Layers: {list(channels_dict.keys())}")
        
        return {
            "output_path": str(output_path),
            "file_size": file_size,
            "width": width,
            "height": height,
            "channels": list(channels_dict.keys()),
            "bit_depth": bit_depth,
            "layers_included": list(loaded_layers.keys()),
        }
    
    # ========================================================================
    # Single Image Export (from PNG/image files)
    # ========================================================================
    
    def export_multilayer(
        self,
        layers: Dict[str, str],
        output_path: Path,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        linear: bool = True,
    ) -> Dict[str, Any]:
        """
        Export multiple AOV layers to a single multi-layer EXR
        
        Args:
            layers: Dict mapping layer names to image paths
                {
                    'beauty': '/path/to/beauty.png',
                    'depth': '/path/to/depth.png',
                    'normals': '/path/to/normals.png',
                    'crypto': '/path/to/crypto.png',
                }
            output_path: Where to save the EXR
            bit_depth: 16 (half float) or 32 (full float)
            compression: Compression method
            linear: Convert beauty to linear color space
            
        Returns:
            Dict with output info
        """
        if not self._has_openexr:
            raise RuntimeError("OpenEXR not installed")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Exporting Multi-Layer EXR")
        print(f"{'='*60}")
        print(f"Layers: {list(layers.keys())}")
        print(f"Output: {output_path}")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"Compression: {compression}")
        print(f"{'='*60}\n")
        
        # Load all layers and determine dimensions
        loaded_layers = {}
        width, height = None, None
        depth_meta = None

        for layer_name, layer_path in layers.items():
            if layer_path is None:
                continue

            layer_path = Path(layer_path)
            if not layer_path.exists():
                print(f"  ⚠ Skipping {layer_name}: file not found")
                continue

            # Depth: prefer the float buffer over the 8-bit preview. 256 levels
            # of quantisation is not something a focus pull can be built on.
            if layer_name == 'depth':
                raw_npy = self._find_raw_data_path(layer_path, layer_name)
                depth_meta = self._load_depth_meta(raw_npy)
                if raw_npy is not None and depth_meta is not None:
                    arr = np.load(str(raw_npy)).astype(np.float32)
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    if width is None:
                        height, width = arr.shape[:2]
                    loaded_layers['depth'] = arr
                    units = 'metres' if depth_meta.get('is_metric') else 'relative'
                    print(f"  ✓ Loaded depth from raw .npy (true Z, {units})")
                    continue
                print("  ⚠ Depth has no Z sidecar — writing the 8-bit display map "
                      "as-is. Its orientation is near=white, the inverse of a Z "
                      "channel. Re-run the depth pass for a usable Z.")

            img = Image.open(layer_path)
            arr = np.array(img).astype(np.float32) / 255.0

            if width is None:
                height, width = arr.shape[:2]

            if layer_name == 'beauty':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[2] == 4:
                    # Alpha is linear coverage, not colour — never gamma it
                    loaded_layers['alpha'] = arr[:, :, 3].copy()
                    arr = arr[:, :, :3]
                if linear:
                    arr = self._srgb_to_linear(arr)
                loaded_layers['beauty'] = arr
                print(f"  ✓ Loaded beauty ({width}x{height}, {'linear' if linear else 'sRGB'})")

            elif layer_name == 'depth':
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                loaded_layers['depth'] = arr
                print(f"  ✓ Loaded depth")

            elif layer_name == 'normals':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                loaded_layers['normals'] = arr
                print(f"  ✓ Loaded normals")
                
            elif layer_name == 'crypto':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                loaded_layers['crypto'] = arr
                print(f"  ✓ Loaded cryptomatte")
        
        if not loaded_layers:
            raise ValueError("No valid layers to export")

        # Beauty sets the frame size when it's there; every channel in one EXR
        # part must match it. The raw depth buffer is written at the depth
        # model's process resolution, which is often not the render size.
        if 'beauty' in loaded_layers:
            height, width = loaded_layers['beauty'].shape[:2]
        for key, arr in loaded_layers.items():
            if arr.shape[:2] == (height, width):
                continue
            print(f"  ↳ resizing {key} {arr.shape[1]}x{arr.shape[0]} → {width}x{height}")
            # Crypto/ID mattes MUST use nearest-neighbor (no interpolation)
            resample = Image.NEAREST if 'crypto' in key else Image.LANCZOS
            if arr.ndim == 2:
                loaded_layers[key] = np.array(
                    Image.fromarray(arr, mode='F').resize((width, height), resample)
                ).astype(np.float32)
            else:
                loaded_layers[key] = np.stack([
                    np.array(
                        Image.fromarray(arr[:, :, c], mode='F').resize((width, height), resample)
                    ).astype(np.float32)
                    for c in range(arr.shape[2])
                ], axis=-1)

        # Build EXR channels
        channels_dict = {}
        channel_info = {}

        pixel_type = (
            self.Imath.PixelType(self.Imath.PixelType.HALF) 
            if bit_depth == 16 
            else self.Imath.PixelType(self.Imath.PixelType.FLOAT)
        )
        
        if 'beauty' in loaded_layers:
            arr = loaded_layers['beauty']
            channels_dict['R'] = self._to_bytes(arr[:, :, 0], bit_depth)
            channels_dict['G'] = self._to_bytes(arr[:, :, 1], bit_depth)
            channels_dict['B'] = self._to_bytes(arr[:, :, 2], bit_depth)
            channel_info['R'] = self.Imath.Channel(pixel_type)
            channel_info['G'] = self.Imath.Channel(pixel_type)
            channel_info['B'] = self.Imath.Channel(pixel_type)

            # Always write A — see _export_frame_multilayer for why
            alpha = loaded_layers.get('alpha')
            if alpha is None:
                alpha = np.ones(arr.shape[:2], dtype=np.float32)
            channels_dict['A'] = self._to_bytes(alpha, bit_depth)
            channel_info['A'] = self.Imath.Channel(pixel_type)

        if 'depth' in loaded_layers:
            self._add_depth_channels(
                channels_dict, channel_info,
                self._depth_to_scene_z(loaded_layers['depth'], depth_meta),
                bit_depth, pixel_type,
            )

        if 'normals' in loaded_layers:
            arr = loaded_layers['normals']
            arr_decoded = arr * 2.0 - 1.0
            channels_dict['N.X'] = self._to_bytes(arr_decoded[:, :, 0], bit_depth)
            channels_dict['N.Y'] = self._to_bytes(arr_decoded[:, :, 1], bit_depth)
            channels_dict['N.Z'] = self._to_bytes(arr_decoded[:, :, 2], bit_depth)
            channel_info['N.X'] = self.Imath.Channel(pixel_type)
            channel_info['N.Y'] = self.Imath.Channel(pixel_type)
            channel_info['N.Z'] = self.Imath.Channel(pixel_type)
        
        crypto_metadata = {}
        if 'crypto' in loaded_layers:
            arr = loaded_layers['crypto']
            r_channel = arr[:, :, 0] if arr.ndim == 3 else arr
            id_matte = (r_channel * 255.0).round().astype(np.uint16)
            crypto_channels, crypto_metadata = self._build_cryptomatte_channels(id_matte)
            float_type = self.Imath.PixelType(self.Imath.PixelType.FLOAT)
            for ch_name, ch_data in crypto_channels.items():
                channels_dict[ch_name] = ch_data.astype(np.float32).tobytes()
                channel_info[ch_name] = self.Imath.Channel(float_type)

        # Create EXR header
        header = self.OpenEXR.Header(width, height)
        header['channels'] = channel_info
        header['compression'] = self._compression_attr(
            compression, has_cryptomatte=bool(crypto_metadata)
        )
        for meta_key, meta_val in crypto_metadata.items():
            header[meta_key] = meta_val.encode('utf-8') if isinstance(meta_val, str) else meta_val

        # Write EXR
        exr_file = self.OpenEXR.OutputFile(str(output_path), header)
        exr_file.writePixels(channels_dict)
        exr_file.close()
        
        file_size = output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n✓ Exported multi-layer EXR")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Channels: {list(channels_dict.keys())}")
        
        return {
            "output_path": str(output_path),
            "width": width,
            "height": height,
            "channels": list(channels_dict.keys()),
            "bit_depth": bit_depth,
            "compression": compression,
            "file_size": file_size,
            "layers_included": list(loaded_layers.keys()),
        }
    
    def export_from_latent(
        self,
        latent_path: Path,
        output_path: Path,
        backend,
        model_type: str = "auto",
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        bracketed: bool = False,
        scales: Optional[List[float]] = None,
        noise_bracketed: bool = False,
        sigmas: Optional[List[float]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Decode a .latent.pt file directly to an EXR using the DiffSynth VAE.

        Uses _load_vae_only — no musubi-tuner, no full pipeline load.

        Decode modes:
          - default: single clean decode
          - bracketed=True: scale bracketing (0.7×/1.0×/1.3×) + Mertens fusion
          - noise_bracketed=True: noise perturbation (σ=0/0.05/0.10) + Mertens fusion
        """
        import torch

        if not self._has_openexr:
            raise RuntimeError("OpenEXR not installed")

        if scales is None:
            scales = [0.85, 1.0, 1.15]
        if sigmas is None:
            sigmas = [0.0, 0.025, 0.05]

        latent_path = Path(latent_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load latent (.pt dict or raw tensor)
        latent_data = torch.load(str(latent_path), map_location='cpu', weights_only=False)
        if isinstance(latent_data, dict) and 'latent' in latent_data:
            latent = latent_data['latent']
        elif isinstance(latent_data, torch.Tensor):
            latent = latent_data
        else:
            raise ValueError(f"Unrecognised latent format in {latent_path}")

        if model_type == "auto":
            model_type = self._resolve_latent_model(
                backend, latent_data, latent, latent.ndim == 5)
        else:
            model_type = backend.resolve_model_type(model_type)

        # Load VAE via DiffSynth (reuses cached pipeline if already loaded)
        vae, decode_fn, (v_min, v_max) = self._load_vae_only(backend, model_type)
        vae = vae.eval()
        # NF4 VAEs cannot be cast or run on the CPU — see _decode_beauty_latents.
        quantized = self._is_quantized(vae)
        work_dtype = next(vae.parameters()).dtype if quantized else torch.float32
        if not quantized:
            vae = vae.to(dtype=torch.float32)
        latent = latent.to(dtype=work_dtype)

        def _decode(lat):
            """GPU-first decode, CPU fallback. Returns float32 numpy (H, W, C)."""
            def _run(device):
                vae.to(device)
                l = lat.to(device)
                with torch.no_grad():
                    out = decode_fn(l, device)
                if isinstance(out, torch.Tensor):
                    return out
                # Some VAEs return a dict
                if isinstance(out, dict):
                    for k in ('sample', 'x', 'output', 'decoded'):
                        if k in out and isinstance(out[k], torch.Tensor):
                            return out[k]
                    return list(out.values())[0]
                raise ValueError(f"Unexpected VAE output type: {type(out)}")

            t = None
            if torch.cuda.is_available():
                try:
                    t = _run(torch.device('cuda'))
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    if quantized:
                        raise
                    vae.cpu()
                    torch.cuda.empty_cache()
            if t is None:
                if quantized:
                    raise RuntimeError(
                        f"'{model_type}' ships a 4-bit quantized VAE, which only "
                        f"dequantizes on the GPU — no CUDA device is available.")
                t = _run(torch.device('cpu'))

            # (1, C, H, W) or (1, C, 1, H, W) → (H, W, C)
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            arr = t.cpu().numpy()
            if arr.ndim == 5:
                arr = arr[0, :, 0, :, :]   # drop batch + temporal
            elif arr.ndim == 4:
                arr = arr[0]               # drop batch
            if arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            arr = arr[:, :, :3].astype(np.float32)
            # Map the family's own output range to [0, 1] sRGB before returning
            arr = (arr - v_min) / (v_max - v_min)
            return arr

        # Build brackets
        if noise_bracketed:
            print(f"  [BRACKET] MODE: noise — sigmas={sigmas} seed={seed}")
            rng = torch.Generator()
            rng.manual_seed(seed)
            brackets = []
            for sigma in sigmas:
                if sigma == 0.0:
                    b = _decode(latent)
                else:
                    noise = torch.randn(latent.shape, dtype=latent.dtype, generator=rng)
                    b = _decode(latent + noise * sigma)
                print(f"  [BRACKET]   σ={sigma:.3f} → raw range [{b.min():.4f}, {b.max():.4f}]")
                brackets.append(b)
        elif bracketed:
            print(f"  [BRACKET] MODE: scale — scales={scales}")
            brackets = []
            for s in scales:
                b = _decode(latent * s)
                print(f"  [BRACKET]   scale={s:.2f}x → raw range [{b.min():.4f}, {b.max():.4f}]")
                brackets.append(b)
        else:
            print(f"  [BRACKET] MODE: standard (single decode)")
            b = _decode(latent)
            print(f"  [BRACKET]   raw range [{b.min():.4f}, {b.max():.4f}]")
            brackets = [b]

        # Release VAE VRAM
        vae.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clip each bracket to [0, 1] (Mertens expects perceptual-space [0,1])
        brackets = [np.clip(b, 0.0, 1.0) for b in brackets]

        if len(brackets) > 1:
            import cv2
            # Mertens expects uint8 — float32 [0,1] inputs get divided by 255 internally
            imgs_u8 = [(b * 255).astype(np.uint8) for b in brackets]
            fused = cv2.createMergeMertens().process(imgs_u8)
            fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
            print(f"  [BRACKET] Mertens fused range [{fused.min():.4f}, {fused.max():.4f}]")
        else:
            fused = brackets[0]

        # sRGB → scene-linear for EXR
        arr = self._srgb_to_linear(fused)
        print(f"  [BRACKET] Post-linearise range [{arr.min():.4f}, {arr.max():.4f}] → writing EXR")
        height, width = arr.shape[:2]

        pixel_type = (
            self.Imath.PixelType(self.Imath.PixelType.HALF)
            if bit_depth == 16
            else self.Imath.PixelType(self.Imath.PixelType.FLOAT)
        )
        channels_dict = {
            'R': self._to_bytes(arr[:, :, 0], bit_depth),
            'G': self._to_bytes(arr[:, :, 1], bit_depth),
            'B': self._to_bytes(arr[:, :, 2], bit_depth),
        }
        channel_info = {ch: self.Imath.Channel(pixel_type) for ch in channels_dict}

        header = self.OpenEXR.Header(width, height)
        header['channels'] = channel_info
        header['compression'] = self._compression_attr(compression)
        exr_file = self.OpenEXR.OutputFile(str(output_path), header)
        exr_file.writePixels(channels_dict)
        exr_file.close()

        file_size = output_path.stat().st_size
        return {
            "output_path": str(output_path),
            "width": width,
            "height": height,
            "channels": list(channels_dict.keys()),
            "bit_depth": bit_depth,
            "compression": compression,
            "file_size": file_size,
            "layers_included": ["beauty"],
            "bracketed": bracketed,
            "noise_bracketed": noise_bracketed,
        }

    def export_single_layers(
        self,
        layers: Dict[str, str],
        output_dir: Path,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        linear: bool = True,
        filename_prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Export each AOV layer as a separate EXR file
        
        Args:
            layers: Dict mapping layer names to image paths
            output_dir: Directory to save EXR files
            bit_depth: 16 or 32
            compression: Compression method
            linear: Convert beauty to linear
            filename_prefix: Optional prefix for filenames
            
        Returns:
            Dict with output info per layer
        """
        if not self._has_openexr:
            raise RuntimeError("OpenEXR not installed")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for layer_name, layer_path in layers.items():
            if layer_path is None:
                continue
            
            output_path = output_dir / f"{filename_prefix}{layer_name}.exr"
            
            result = self.export_multilayer(
                layers={layer_name: layer_path},
                output_path=output_path,
                bit_depth=bit_depth,
                compression=compression,
                linear=linear if layer_name == 'beauty' else False,
            )
            
            results[layer_name] = result
        
        return results
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def _to_bytes(self, arr: np.ndarray, bit_depth: int) -> bytes:
        """Convert numpy array to bytes for EXR"""
        if bit_depth == 16:
            return arr.astype(np.float16).tobytes()
        else:
            return arr.astype(np.float32).tobytes()
    
    @staticmethod
    def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear color space"""
        return np.where(
            img <= 0.04045,
            img / 12.92,
            np.power((img + 0.055) / 1.055, 2.4)
        )
    
    @staticmethod
    def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
        """Convert linear to sRGB color space"""
        return np.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * np.power(img, 1/2.4) - 0.055
        )