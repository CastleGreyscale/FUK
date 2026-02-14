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
            '-vsync', '0',
            str(output_pattern)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Return sorted list of extracted frames
        frames = sorted(output_dir.glob("frame_*.png"))
        return frames
    
    def _find_raw_data_path(self, layer_path: Path, layer_name: str) -> Optional[Path]:
        """
        Look for raw .npy file alongside an MP4 or inside a sequence directory.
        
        The batch preprocessors save these automatically:
        depth.mp4     -> depth_raw.npy  (same directory)
        depth_seq/    -> depth_seq/depth_raw.npy
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
        return None
    
    def _find_beauty_latent_path(self, beauty_path: Path) -> Optional[Path]:
        """
        Find the corresponding .latent.pt file for a beauty pass video/image.
        
        Latents are stored in a 'latents/' subdirectory next to the beauty file:
            generated.mp4 -> latents/generated.latent.pt
            render/frame_001.png -> latents/frame_001.latent.pt
        
        Returns:
            Path to .latent.pt file if found, None otherwise
        """
        latent_dir = beauty_path.parent / "latents"
        if not latent_dir.exists():
            return None
        
        latent_file = latent_dir / f"{beauty_path.stem}.latent.pt"
        if latent_file.exists():
            return latent_file
        
        base_name = beauty_path.stem.split('.')[0]
        latent_file = latent_dir / f"{base_name}.latent.pt"
        if latent_file.exists():
            return latent_file
        
        return None
    
    def _load_vae_only(self, backend, model_type: str):
        """
        Load ONLY the VAE from a model config — no DiT, no text encoder.
        
        Uses DiffSynth's from_pretrained with a single ModelConfig containing
        just the VAE weights.  Loads ~1.3 GB to CPU instead of 20+ GB for the
        full pipeline.
        
        Returns:
            (vae, needs_device_arg) — the VAE module and whether decode() needs a device kwarg
        """
        import inspect
        
        # 1. Check if a pipeline is already cached — grab its VAE for free
        for key, pipe in backend.pipelines.items():
            if key.startswith(f"{model_type}:") and hasattr(pipe, 'vae') and pipe.vae is not None:
                print(f"  📄 Reusing VAE from cached pipeline: {key}")
                vae = pipe.vae
                needs_device = 'device' in inspect.signature(vae.decode).parameters
                return vae, needs_device
        
        # 2. No cached pipeline — load VAE-only via DiffSynth
        entry = backend.get_model_entry(model_type)
        primary_id = entry["model_id"]
        pipeline_type = entry["pipeline"]
        
        # Find the VAE component in model config
        vae_comps = [c for c in entry.get("components", [])
                     if "VAE" in c["pattern"] or "vae" in c["pattern"]]
        if not vae_comps:
            raise ValueError(f"No VAE component in models.json for '{model_type}'")
        
        comp = vae_comps[0]
        mid = comp.get("model_id", primary_id)
        
        print(f"  📄 Loading VAE-only: {mid} / {comp['pattern']}")
        
        # Build minimal pipeline with just the VAE component
        import torch
        if pipeline_type == "wan":
            from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig as MC
            PipelineCls = WanVideoPipeline
        else:
            from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig as MC
            PipelineCls = QwenImagePipeline
        
        vae_config = MC(model_id=mid, origin_file_pattern=comp["pattern"])
        
        pipe = PipelineCls.from_pretrained(
            torch_dtype=torch.float32,
            device="cpu",
            model_configs=[vae_config],
        )
        
        vae = pipe.vae
        if vae is None:
            raise RuntimeError(f"VAE failed to load from {mid}/{comp['pattern']}")
        
        needs_device = 'device' in inspect.signature(vae.decode).parameters
        print(f"  📄 VAE loaded to CPU ({type(vae).__name__})")
        return vae, needs_device

    def _decode_beauty_latents(
        self,
        latent_path: Path,
        backend,  # DiffSynthBackend instance
        model_type: str = "auto",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """
        Decode beauty pass latents to numpy arrays.
        
        Loads only the VAE (not the full pipeline) and decodes on CPU.
        CPU decode avoids both VRAM pressure and the conv3d CUDA kernel
        issue present in many PyTorch builds.
        
        Args:
            latent_path: Path to .latent.pt file
            backend: DiffSynthBackend instance
            model_type: Model type ("auto" to detect, or explicit key)
            progress_callback: Optional progress callback
        
        Returns:
            List of decoded frames as float32 numpy arrays [H, W, C] in linear [0, 1]
        """
        import torch
        import inspect
        
        print(f"  📄 Decoding beauty latents: {latent_path.name}")
        
        # Load latent
        latent_data = torch.load(str(latent_path), map_location='cpu')
        latent = latent_data['latent']
        is_video = latent.ndim == 5
        
        # Auto-detect model type
        if model_type == "auto":
            if is_video:
                wan_models = [k for k, v in backend.models_config.items() 
                             if isinstance(v, dict) and v.get("pipeline") == "wan"]
                if wan_models:
                    model_type = wan_models[0]
                else:
                    raise ValueError("No Wan pipeline in models.json for video decode")
            else:
                model_type = "qwen_image"
        
        print(f"  📄 Latent shape: {latent.shape}, type: {'video' if is_video else 'image'}")
        print(f"  📄 Using model: {model_type}")
        
        # ------------------------------------------------------------------
        # Load VAE only — no DiT, no text encoder, no VRAM usage
        # ------------------------------------------------------------------
        vae, needs_device_arg = self._load_vae_only(backend, model_type)
        
        # ------------------------------------------------------------------
        # Decode: GPU (float32) first, CPU fallback
        # float32 avoids the bf16 conv3d kernel issue (cuDNN handles f32).
        # VAE-only load (~1.3GB) leaves ~22GB free on a 24GB card.
        # ------------------------------------------------------------------
        vae = vae.to(dtype=torch.float32).eval()
        latent = latent.to(dtype=torch.float32)
        
        def _vae_decode(vae, latent, device, needs_device_arg):
            vae.to(device)
            lat = latent.to(device)
            with torch.no_grad():
                if needs_device_arg:
                    return vae.decode(lat, device=device, tiled=False)
                else:
                    return vae.decode(lat)
        
        decoded = None
        
        if torch.cuda.is_available():
            try:
                free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                print(f"  📄 Attempting GPU decode (float32, tiled=False, {free_gb:.1f} GB free)...")
                decoded = _vae_decode(vae, latent, torch.device('cuda'), needs_device_arg)
                print(f"  📄 GPU decode complete")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                err_short = str(e).split('\n')[0][:120]
                print(f"  ⚠️  GPU decode failed: {err_short}")
                print(f"  📄 Falling back to CPU...")
                vae.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                decoded = None
        
        if decoded is None:
            import os
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            print(f"  📄 Decoding on CPU (float32, {num_threads} threads, tiled=False)...")
            decoded = _vae_decode(vae, latent, torch.device('cpu'), needs_device_arg)
            print(f"  📄 CPU decode complete")
        
        # Clean up VAE
        vae.cpu()
        del vae, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ------------------------------------------------------------------
        # Convert to numpy frames with sRGB → Linear
        # ------------------------------------------------------------------
        decoded_frames = []
        
        if isinstance(decoded, torch.Tensor):
            if decoded.dtype == torch.bfloat16:
                decoded = decoded.to(torch.float32)
            
            pixels = decoded.cpu().numpy()
            del decoded
            
            if is_video and pixels.ndim == 5:
                num_frames = pixels.shape[2]
                print(f"  📄 Extracting {num_frames} frames...")
                
                for frame_idx in range(num_frames):
                    frame = pixels[0, :, frame_idx, :, :]  # (C, H, W)
                    
                    # [-1, 1] → [0, 1]
                    if frame.min() < 0:
                        frame = (frame + 1.0) / 2.0
                    
                    # CHW → HWC
                    frame = np.transpose(frame, (1, 2, 0))
                    
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    
                    # sRGB → Linear (VAE trained on sRGB, EXR needs linear)
                    frame = self._srgb_to_linear(frame)
                    
                    decoded_frames.append(frame.astype(np.float32))
                    
                    if progress_callback and (frame_idx + 1) % 10 == 0:
                        progress_callback(frame_idx + 1, num_frames)
            
            elif pixels.ndim == 4:
                frame = pixels[0]
                
                if frame.min() < 0:
                    frame = (frame + 1.0) / 2.0
                
                if frame.shape[0] in [1, 3, 4]:
                    frame = np.transpose(frame, (1, 2, 0))
                
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                frame = self._srgb_to_linear(frame)
                decoded_frames.append(frame.astype(np.float32))
            
            else:
                raise ValueError(f"Unexpected decoded shape: {pixels.shape}")
        
        print(f"  ✅ Decoded {len(decoded_frames)} beauty frames (linear color space)")
        return decoded_frames

    
    def export_video_sequence(
        self,
        beauty_latent: Path,  # REQUIRED - no fallback!
        aov_layers: Dict[str, str],  # depth, normals, crypto (optional)
        backend,  # DiffSynthBackend instance - REQUIRED
        output_dir: Path,
        filename_pattern: str = "frame.{frame:04d}.exr",
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        start_frame: int = 1,
        model_type: str = "auto",
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
        )
        
        total_frames = len(beauty_decoded_frames)
        print(f"\n  ✅ Beauty: {total_frames} frames from latent (lossless)")
        
        # Process AOV layers
        frame_counts = {}
        layer_paths = {}
        raw_data = {}
        
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
    
    def _export_frame_multilayer(
        self,
        layers: Dict[str, str],
        output_path: Path,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        linear: bool = True,
        quiet: bool = False,
        raw_arrays: Optional[Dict[str, np.ndarray]] = None,
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
        
        if 'depth' in loaded_layers:
            arr = loaded_layers['depth']
            channels_dict['Z'] = self._to_bytes(arr, bit_depth)
            channel_info['Z'] = self.Imath.Channel(pixel_type)
        
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
            arr = loaded_layers['crypto_raw']
            channels_dict['crypto.ID'] = self._to_bytes(arr, bit_depth)
            channel_info['crypto.ID'] = self.Imath.Channel(pixel_type)
        elif 'crypto' in loaded_layers:
            arr = loaded_layers['crypto']
            channels_dict['crypto.R'] = self._to_bytes(arr[:, :, 0], bit_depth)
            channels_dict['crypto.G'] = self._to_bytes(arr[:, :, 1], bit_depth)
            channels_dict['crypto.B'] = self._to_bytes(arr[:, :, 2], bit_depth)
            channel_info['crypto.R'] = self.Imath.Channel(pixel_type)
            channel_info['crypto.G'] = self.Imath.Channel(pixel_type)
            channel_info['crypto.B'] = self.Imath.Channel(pixel_type)
        
        # Create and write EXR
        header = self.OpenEXR.Header(width, height)
        header['channels'] = channel_info
        
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
        
        for layer_name, layer_path in layers.items():
            if layer_path is None:
                continue
                
            layer_path = Path(layer_path)
            if not layer_path.exists():
                print(f"  ⚠ Skipping {layer_name}: file not found")
                continue
            
            img = Image.open(layer_path)
            arr = np.array(img).astype(np.float32) / 255.0
            
            if width is None:
                height, width = arr.shape[:2]
            
            if layer_name == 'beauty':
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[2] == 4:
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
        
        if 'depth' in loaded_layers:
            arr = loaded_layers['depth']
            channels_dict['Z'] = self._to_bytes(arr, bit_depth)
            channel_info['Z'] = self.Imath.Channel(pixel_type)
        
        if 'normals' in loaded_layers:
            arr = loaded_layers['normals']
            arr_decoded = arr * 2.0 - 1.0
            channels_dict['N.X'] = self._to_bytes(arr_decoded[:, :, 0], bit_depth)
            channels_dict['N.Y'] = self._to_bytes(arr_decoded[:, :, 1], bit_depth)
            channels_dict['N.Z'] = self._to_bytes(arr_decoded[:, :, 2], bit_depth)
            channel_info['N.X'] = self.Imath.Channel(pixel_type)
            channel_info['N.Y'] = self.Imath.Channel(pixel_type)
            channel_info['N.Z'] = self.Imath.Channel(pixel_type)
        
        if 'crypto' in loaded_layers:
            arr = loaded_layers['crypto']
            channels_dict['crypto.R'] = self._to_bytes(arr[:, :, 0], bit_depth)
            channels_dict['crypto.G'] = self._to_bytes(arr[:, :, 1], bit_depth)
            channels_dict['crypto.B'] = self._to_bytes(arr[:, :, 2], bit_depth)
            channel_info['crypto.R'] = self.Imath.Channel(pixel_type)
            channel_info['crypto.G'] = self.Imath.Channel(pixel_type)
            channel_info['crypto.B'] = self.Imath.Channel(pixel_type)
        
        # Create EXR header
        header = self.OpenEXR.Header(width, height)
        header['channels'] = channel_info
        
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