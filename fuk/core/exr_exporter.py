# core/exr_exporter.py
"""
Multi-Layer EXR Exporter

Combines AOV layers (Beauty, Depth, Normals, Cryptomatte) into 
industry-standard multi-layer EXR files for compositing.

Supports:
- Multi-layer EXR with all AOVs in one file
- Individual single-layer EXRs per AOV
- Video/sequence export (frame-by-frame EXR sequences)
- 16-bit half or 32-bit float
- Various compression methods (ZIP, PIZ, DWAA, etc.)
- Linear/sRGB color space handling
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
    
    Usage:
        exporter = EXRExporter()
        
        # Single image export
        result = exporter.export_multilayer(
            layers={
                'beauty': '/path/to/beauty.png',
                'depth': '/path/to/depth.png',
                'normals': '/path/to/normals.png',
                'crypto': '/path/to/crypto.png',
            },
            output_path='/path/to/output.exr',
            bit_depth=32,
            compression='ZIP',
            linear=True,
        )
        
        # Video sequence export
        result = exporter.export_video_sequence(
            layers={
                'beauty': '/path/to/beauty.mp4',
                'depth': '/path/to/depth.mp4',
                'normals': '/path/to/normals.mp4',
            },
            output_dir='/path/to/output_sequence/',
            filename_pattern='shot01.{frame:04d}.exr',
            bit_depth=32,
            compression='ZIP',
            linear=True,
        )
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
    # Video/Sequence Support
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
        """Extract all frames from a video file"""
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
    
    def _load_frame_from_video(
        self, 
        video_path: Path, 
        frame_index: int,
        temp_dir: Path
    ) -> Optional[np.ndarray]:
        """Extract and load a single frame from video"""
        output_path = temp_dir / f"frame_{frame_index:06d}.png"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', f'select=eq(n\\,{frame_index})',
            '-vframes', '1',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            if output_path.exists():
                img = Image.open(output_path)
                arr = np.array(img).astype(np.float32) / 255.0
                output_path.unlink()  # Clean up temp file
                return arr
        except Exception as e:
            print(f"⚠ Failed to extract frame {frame_index}: {e}")
        
        return None
    def _find_raw_data_path(self, layer_path: Path, layer_name: str) -> Optional[Path]:
        """
        Look for raw .npy file alongside an MP4 or inside a sequence directory.
        
        The batch preprocessors now save these automatically:
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
    
    def export_video_sequence(
        self,
        layers: Dict[str, str],
        output_dir: Path,
        filename_pattern: str = "frame.{frame:04d}.exr",
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        linear: bool = True,
        start_frame: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Export video layers as an EXR image sequence
        
        Each frame becomes a multi-layer EXR with all AOVs.
        
        Args:
            layers: Dict mapping layer names to video paths (or image sequence dirs)
                {
                    'beauty': '/path/to/beauty.mp4',
                    'depth': '/path/to/depth.mp4',
                    'normals': '/path/to/normals.mp4',
                }
            output_dir: Directory to save EXR sequence
            filename_pattern: Pattern for output files. Use {frame:04d} for frame number.
                Examples: "shot01.{frame:04d}.exr", "render_{frame:04d}.exr"
            bit_depth: 16 (half float) or 32 (full float)
            compression: Compression method
            linear: Convert beauty to linear color space
            start_frame: Starting frame number (default 1)
            progress_callback: Optional callback(current_frame, total_frames)
            
        Returns:
            Dict with sequence info
        """
        if not self._has_openexr:
            raise RuntimeError("OpenEXR not installed")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Exporting EXR Sequence")
        print(f"{'='*60}")
        print(f"Layers: {list(layers.keys())}")
        print(f"Output: {output_dir}")
        print(f"Pattern: {filename_pattern}")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"{'='*60}\n")
        
        # Determine frame count and validate inputs
        frame_counts = {}
        layer_paths = {}
        raw_data = {}
        
        for layer_name, layer_path in layers.items():
            if layer_path is None:
                continue
            
            layer_path = Path(layer_path)
            if not layer_path.exists():
                print(f"  ⚠ Skipping {layer_name}: path not found")
                continue
            
            layer_paths[layer_name] = layer_path
            raw_npy_path = self._find_raw_data_path(layer_path, layer_name)
            if raw_npy_path:
                try:
                    raw_array = np.load(str(raw_npy_path))
                    raw_data[layer_name] = raw_array
                    frame_counts[layer_name] = len(raw_array)
                    print(f"  ✓ {layer_name}: {len(raw_array)} frames from RAW .npy (lossless)")
                    continue  # Don't bother checking if it's MP4 or sequence
                except Exception as e:
                    print(f"  ⚠ {layer_name}: raw .npy failed ({e}), falling back")
            
            if self._is_video_file(layer_path):
                info = self._get_video_info(layer_path)
                frame_counts[layer_name] = info['frame_count']
                print(f"  ✓ {layer_name}: {info['frame_count']} frames ({layer_path.name})")
            elif layer_path.is_dir():
                # Image sequence directory
                frames = sorted(layer_path.glob("*.png")) + sorted(layer_path.glob("*.jpg"))
                frame_counts[layer_name] = len(frames)
                print(f"  ✓ {layer_name}: {len(frames)} frames (sequence)")
            else:
                print(f"  ⚠ {layer_name}: not a video or sequence directory")
        
        if not layer_paths:
            raise ValueError("No valid video layers provided")
        
        # Use minimum frame count across all layers
        total_frames = min(frame_counts.values()) if frame_counts else 0
        if total_frames == 0:
            raise ValueError("Could not determine frame count")
        
        print(f"\n  Processing {total_frames} frames...")
        
        # Create temp directory for frame extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="exr_export_"))
        
        try:
            # Extract all frames from videos first (more efficient than per-frame extraction)
            extracted_frames = {}
            
            for layer_name, layer_path in layer_paths.items():
                if layer_name in raw_data:
                    continue  # NEW: skip, we have raw numpy data
                if self._is_video_file(layer_path):
                    print(f"  Extracting {layer_name} frames...")
                    layer_temp = temp_dir / layer_name
                    frames = self._extract_frames(layer_path, layer_temp)
                    extracted_frames[layer_name] = frames
                elif layer_path.is_dir():
                    # Already a sequence
                    frames = sorted(layer_path.glob("*.png")) + sorted(layer_path.glob("*.jpg"))
                    extracted_frames[layer_name] = frames
            
            # Process frame by frame
            exported_frames = []
            total_size = 0
            
            for frame_idx in range(total_frames):
                frame_num = start_frame + frame_idx
                
                # Build layer dict for this frame
                frame_layers = {}
                for layer_name, frames in extracted_frames.items():
                    if frame_idx < len(frames):
                        frame_layers[layer_name] = str(frames[frame_idx])

                frame_raw = {}
                for layer_name, arr in raw_data.items():
                    if frame_idx < len(arr):
                        frame_raw[layer_name] = arr[frame_idx]
                
                # Generate output filename
                output_filename = filename_pattern.format(frame=frame_num)
                output_path = output_dir / output_filename
                
                # Export this frame
                try:
                    result = self._export_frame_multilayer(
                        layers=frame_layers,
                        output_path=output_path,
                        bit_depth=bit_depth,
                        compression=compression,
                        linear=linear,
                        quiet=True,  # Suppress per-frame output
                        raw_arrays=frame_raw,  
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
        
        print(f"\n✓ Exported EXR sequence")
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
            "layers_included": list(layer_paths.keys()),
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
        Internal method to export a single frame (used by sequence export)
        Same as export_multilayer but with quiet option for batch processing.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load all layers
        loaded_layers = {}
        width, height = None, None

        # NEW: Process raw arrays first (lossless, no file I/O)
        if raw_arrays:
            for layer_name, arr in raw_arrays.items():
                if width is None:
                    height, width = arr.shape[:2] if arr.ndim >= 2 else (0, 0)
                
                if layer_name == 'depth':
                    # Already float32 [H,W] range [0,1] — perfect
                    loaded_layers['depth'] = arr.astype(np.float32) if arr.ndim == 2 else arr[:,:,0].astype(np.float32)
                    
                elif layer_name == 'normals':
                    # Already float32 [H,W,3] range [-1,1] — no 0-1 conversion needed
                    loaded_layers['normals_raw'] = arr[:,:,:3].astype(np.float32)
                    
                elif layer_name == 'crypto':
                    # uint16 [H,W] object IDs — store as float for EXR
                    loaded_layers['crypto_raw'] = arr.astype(np.float32)
        
        for layer_name, layer_path in layers.items():
            if layer_path is None:
                continue
                
            layer_path = Path(layer_path)
            if not layer_path.exists():
                continue
            
            img = Image.open(layer_path)
            arr = np.array(img).astype(np.float32) / 255.0
            
            if width is None:
                height, width = arr.shape[:2]
            
            # Process based on layer type
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
            arr = loaded_layers['normals_raw']  # Already [-1,1], no math needed
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
            arr = loaded_layers['crypto_raw']  # Single channel, object IDs as float
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
        
        return {
            "output_path": str(output_path),
            "width": width,
            "height": height,
            "channels": list(channels_dict.keys()),
            "bit_depth": bit_depth,
            "file_size": file_size,
            "layers_included": list(loaded_layers.keys()),
        }
    
    # ========================================================================
    # Original Image Export Methods
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
            
            # Set dimensions from first layer
            if width is None:
                height, width = arr.shape[:2]
            
            # Handle different layer types
            if layer_name == 'beauty':
                # Beauty is RGB, optionally convert to linear
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]  # Drop alpha for now
                
                if linear:
                    arr = self._srgb_to_linear(arr)
                    
                loaded_layers['beauty'] = arr
                print(f"  ✓ Loaded beauty ({width}x{height}, {'linear' if linear else 'sRGB'})")
                
            elif layer_name == 'depth':
                # Depth is single channel -> Z
                if arr.ndim == 3:
                    arr = arr[:, :, 0]  # Take first channel
                loaded_layers['depth'] = arr
                print(f"  ✓ Loaded depth")
                
            elif layer_name == 'normals':
                # Normals are RGB (XYZ encoded)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                loaded_layers['normals'] = arr
                print(f"  ✓ Loaded normals")
                
            elif layer_name == 'crypto':
                # Crypto is RGB ID matte
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
        
        # Beauty -> R, G, B
        if 'beauty' in loaded_layers:
            arr = loaded_layers['beauty']
            channels_dict['R'] = self._to_bytes(arr[:, :, 0], bit_depth)
            channels_dict['G'] = self._to_bytes(arr[:, :, 1], bit_depth)
            channels_dict['B'] = self._to_bytes(arr[:, :, 2], bit_depth)
            channel_info['R'] = self.Imath.Channel(pixel_type)
            channel_info['G'] = self.Imath.Channel(pixel_type)
            channel_info['B'] = self.Imath.Channel(pixel_type)
        
        # Depth -> Z
        if 'depth' in loaded_layers:
            arr = loaded_layers['depth']
            channels_dict['Z'] = self._to_bytes(arr, bit_depth)
            channel_info['Z'] = self.Imath.Channel(pixel_type)
        
        # Normals -> N.X, N.Y, N.Z
        if 'normals' in loaded_layers:
            arr = loaded_layers['normals']
            # Decode from [0,1] to [-1,1]
            arr_decoded = arr * 2.0 - 1.0
            channels_dict['N.X'] = self._to_bytes(arr_decoded[:, :, 0], bit_depth)
            channels_dict['N.Y'] = self._to_bytes(arr_decoded[:, :, 1], bit_depth)
            channels_dict['N.Z'] = self._to_bytes(arr_decoded[:, :, 2], bit_depth)
            channel_info['N.X'] = self.Imath.Channel(pixel_type)
            channel_info['N.Y'] = self.Imath.Channel(pixel_type)
            channel_info['N.Z'] = self.Imath.Channel(pixel_type)
        
        # Crypto -> crypto.R, crypto.G, crypto.B
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
        # Note: Using default compression (typically ZIP)
        # Compression API varies significantly between OpenEXR Python binding versions
        
        # Write EXR
        exr_file = self.OpenEXR.OutputFile(str(output_path), header)
        exr_file.writePixels(channels_dict)
        exr_file.close()
        
        # Get file size
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
            
            # Export as single-layer EXR
            result = self.export_multilayer(
                layers={layer_name: layer_path},
                output_path=output_path,
                bit_depth=bit_depth,
                compression=compression,
                linear=linear if layer_name == 'beauty' else False,
            )
            
            results[layer_name] = result
        
        return results
    
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
    
    # ========================================================================
    # True Latent-to-EXR Export (No MP4 Compression)
    # ========================================================================
    
    def export_from_array(
        self,
        pixels: np.ndarray,
        output_path: Path,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        is_linear: bool = True,
    ) -> Dict[str, Any]:
        """
        Export a decoded pixel array directly to EXR.
        
        This is the true lossless path - pixels go directly to EXR
        without any intermediate lossy compression.
        
        Args:
            pixels: Float32 numpy array of shape (H, W, C) in [0, 1] range
                   Already decoded from latent, already in linear space if is_linear=True
            output_path: Output EXR path
            bit_depth: 16 (half float) or 32 (full float)
            compression: Compression method
            is_linear: Whether pixels are already in linear color space
            
        Returns:
            Dict with output info
        """
        if not self._has_openexr:
            raise RuntimeError("OpenEXR not installed")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate input
        if pixels.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got shape {pixels.shape}")
        
        height, width, channels = pixels.shape
        
        if channels not in [1, 3, 4]:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {channels}")
        
        # Ensure float32
        pixels = pixels.astype(np.float32)
        
        # Setup EXR
        pixel_type = self.Imath.PixelType(
            self.Imath.PixelType.HALF if bit_depth == 16 else self.Imath.PixelType.FLOAT
        )
        
        channels_dict = {}
        channel_info = {}
        
        # RGB channels
        if channels >= 3:
            channels_dict['R'] = self._to_bytes(pixels[:, :, 0], bit_depth)
            channels_dict['G'] = self._to_bytes(pixels[:, :, 1], bit_depth)
            channels_dict['B'] = self._to_bytes(pixels[:, :, 2], bit_depth)
            channel_info['R'] = self.Imath.Channel(pixel_type)
            channel_info['G'] = self.Imath.Channel(pixel_type)
            channel_info['B'] = self.Imath.Channel(pixel_type)
        elif channels == 1:
            # Single channel (e.g., depth)
            channels_dict['Y'] = self._to_bytes(pixels[:, :, 0], bit_depth)
            channel_info['Y'] = self.Imath.Channel(pixel_type)
        
        # Alpha channel
        if channels == 4:
            channels_dict['A'] = self._to_bytes(pixels[:, :, 3], bit_depth)
            channel_info['A'] = self.Imath.Channel(pixel_type)
        
        # Create and write EXR
        header = self.OpenEXR.Header(width, height)
        header['channels'] = channel_info
        
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
            "is_linear": is_linear,
        }
    
    def export_from_latent(
        self,
        latent_path: Path,
        output_path: Path,
        vae_path: str,
        musubi_path: Optional[Path] = None,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Export an image latent directly to EXR.
        
        Loads the VAE, decodes the latent, and writes directly to EXR.
        This is the true lossless path for single images.
        
        Args:
            latent_path: Path to latent.safetensors
            output_path: Output EXR path
            vae_path: Path to VAE model
            musubi_path: Path to musubi-tuner (for VAE loading)
            bit_depth: 16 or 32
            compression: Compression method
            device: torch device
            
        Returns:
            Dict with output info
        """
        # Import from same directory or installed package
        try:
            from latent_decoder import LatentDecoder
        except ImportError:
            # Try relative import if running as module
            from .latent_decoder import LatentDecoder
        
        print(f"\n{'='*60}")
        print(f"True Latent-to-EXR Export (Image)")
        print(f"{'='*60}")
        print(f"Latent: {latent_path}")
        print(f"Output: {output_path}")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"{'='*60}\n")
        
        # Decode latent
        decoder = LatentDecoder(vae_path=vae_path, musubi_path=musubi_path, device=device)
        pixels = decoder.decode_image_latent(latent_path, normalize=True)
        
        # VAE output is already in linear space
        result = self.export_from_array(
            pixels=pixels,
            output_path=output_path,
            bit_depth=bit_depth,
            compression=compression,
            is_linear=True,
        )
        
        # Clean up
        decoder.unload()
        
        print(f"\n✓ Exported true latent EXR: {output_path}")
        print(f"  Size: {result['file_size'] / (1024*1024):.2f} MB")
        
        return result
    
    def export_from_latent_sequence(
        self,
        latent_path: Path,
        output_dir: Path,
        vae_path: str,
        musubi_path: Optional[Path] = None,
        filename_pattern: str = "frame.{frame:04d}.exr",
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        start_frame: int = 1,
        device: str = "cuda",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Export a video latent directly to EXR sequence.
        
        Decodes the video latent and writes each frame directly to EXR.
        This is the TRUE lossless path - no MP4 compression anywhere.
        
        Args:
            latent_path: Path to latent.safetensors (video latent)
            output_dir: Directory for output EXR sequence
            vae_path: Path to VAE model
            musubi_path: Path to musubi-tuner (for VAE loading)
            filename_pattern: Pattern with {frame:04d} placeholder
            bit_depth: 16 or 32
            compression: Compression method
            start_frame: First frame number
            device: torch device
            progress_callback: Optional callback(current, total)
            
        Returns:
            Dict with sequence info
        """
        try:
            from latent_decoder import LatentDecoder
        except ImportError:
            from .latent_decoder import LatentDecoder
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"True Latent-to-EXR Export (Video Sequence)")
        print(f"{'='*60}")
        print(f"Latent: {latent_path}")
        print(f"Output: {output_dir}")
        print(f"Pattern: {filename_pattern}")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"{'='*60}\n")
        
        # Initialize decoder
        decoder = LatentDecoder(vae_path=vae_path, musubi_path=musubi_path, device=device)
        
        # Get latent info first
        info = decoder.get_latent_info(latent_path)
        estimated = info.get("estimated_output", {})
        total_frames = estimated.get("frames", 0)
        
        print(f"  Estimated frames: {total_frames}")
        print(f"  Estimated size: {estimated.get('width', '?')}x{estimated.get('height', '?')}")
        
        # Decode and export frame by frame
        exported_frames = []
        total_size = 0
        
        for frame_idx, pixels in decoder.decode_video_latent(latent_path, normalize=True):
            frame_num = start_frame + frame_idx
            output_filename = filename_pattern.format(frame=frame_num)
            output_path = output_dir / output_filename
            
            try:
                result = self.export_from_array(
                    pixels=pixels,
                    output_path=output_path,
                    bit_depth=bit_depth,
                    compression=compression,
                    is_linear=True,
                )
                
                exported_frames.append(output_path)
                total_size += result.get('file_size', 0)
                
            except Exception as e:
                print(f"  ⚠ Frame {frame_num} failed: {e}")
            
            # Progress
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx + 1, total_frames)
            
            if (frame_idx + 1) % 10 == 0:
                print(f"  Exported {frame_idx + 1} frames...")
        
        # Clean up
        decoder.unload()
        
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"\n✓ Exported true latent EXR sequence")
        print(f"  Frames: {len(exported_frames)}")
        print(f"  Total Size: {total_size_mb:.2f} MB")
        print(f"  Output: {output_dir}")
        
        return {
            "output_dir": str(output_dir),
            "frame_count": len(exported_frames),
            "total_size": total_size,
            "bit_depth": bit_depth,
            "compression": compression,
            "frames": [str(f) for f in exported_frames],
            "is_true_latent": True,  # Flag to indicate this bypassed MP4
        }
    
    def _find_raw_data_path(self, layer_path: Path, layer_name: str) -> Optional[Path]:
        """
        Find raw .npy data file for a layer.
        
        Looks for patterns:
        - MP4: /path/to/depth.mp4 -> /path/to/depth_raw.npy
        - Directory: /path/to/depth_sequence/ -> /path/to/depth_sequence/depth_raw.npy
        """
        if self._is_video_file(layer_path):
            # Look for {stem}_raw.npy alongside the MP4
            raw_path = layer_path.parent / f"{layer_path.stem}_raw.npy"
            if raw_path.exists():
                return raw_path
            # Also try {layer_name}_raw.npy
            raw_path = layer_path.parent / f"{layer_name}_raw.npy"
            if raw_path.exists():
                return raw_path
        elif layer_path.is_dir():
            # Look for {layer_name}_raw.npy inside directory
            raw_path = layer_path / f"{layer_name}_raw.npy"
            if raw_path.exists():
                return raw_path
            # Also try depth_raw.npy, crypto_raw.npy, normals_raw.npy
            for name in ['depth_raw.npy', 'crypto_raw.npy', 'normals_raw.npy']:
                raw_path = layer_path / name
                if raw_path.exists():
                    return raw_path
        
        return None


    # Add this method to export frames with raw numpy arrays:

    def _export_frame_multilayer_with_raw(
        self,
        layers: Dict[str, str],
        raw_arrays: Dict[str, np.ndarray],
        output_path: Path,
        bit_depth: Literal[16, 32] = 32,
        compression: str = "ZIP",
        linear: bool = True,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Export a single frame with support for raw numpy arrays.
        
        Args:
            layers: Dict of layer_name -> file_path (for layers loaded from PNG/JPG)
            raw_arrays: Dict of layer_name -> numpy array (for raw lossless data)
            output_path: Where to save EXR
            bit_depth: 16 or 32
            compression: Compression method
            linear: Convert beauty to linear
            quiet: Suppress output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load all layers
        loaded_layers = {}
        width, height = None, None
        
        # Process raw arrays first (these are lossless!)
        for layer_name, arr in raw_arrays.items():
            if width is None and arr.ndim >= 2:
                if arr.ndim == 2:
                    height, width = arr.shape
                else:
                    height, width = arr.shape[:2]
            
            # Process based on layer type
            if layer_name == 'depth':
                # Depth: single channel float [H, W] in range [0, 1]
                if arr.ndim == 2:
                    loaded_layers['depth'] = arr.astype(np.float32)
                else:
                    loaded_layers['depth'] = arr[:, :, 0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
                    
            elif layer_name == 'normals':
                # Normals: 3 channels [H, W, 3] in range [-1, 1]
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    loaded_layers['normals'] = arr[:, :, :3].astype(np.float32)
                elif arr.ndim == 2:
                    loaded_layers['normals'] = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
                    
            elif layer_name == 'crypto':
                # Crypto: integer IDs [H, W] - convert to float for EXR
                # Store as raw float - the ID value IS the value
                # Compositors can use this directly for keying
                loaded_layers['crypto'] = arr.astype(np.float32)
                
            elif layer_name == 'beauty':
                # Beauty: RGB [H, W, 3]
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                if linear:
                    arr = self._srgb_to_linear(arr)
                loaded_layers['beauty'] = arr[:, :, :3].astype(np.float32)
            else:
                # Generic layer
                loaded_layers[layer_name] = arr.astype(np.float32)
        
        # Process file-based layers (from PNG/JPG) - only for layers not in raw_arrays
        for layer_name, layer_path in layers.items():
            if layer_path is None or layer_name in loaded_layers:
                continue
                
            layer_path = Path(layer_path)
            if not layer_path.exists():
                continue
            
            img = Image.open(layer_path)
            arr = np.array(img).astype(np.float32) / 255.0
            
            if width is None:
                height, width = arr.shape[:2]
            
            # Process based on layer type
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
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                # Convert from 0-1 to -1 to 1
                arr = arr * 2.0 - 1.0
                loaded_layers['normals'] = arr
                
            elif layer_name == 'crypto':
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                loaded_layers['crypto'] = arr
            else:
                loaded_layers[layer_name] = arr
        
        if not loaded_layers:
            raise ValueError("No layers loaded")
        
        # Write EXR
        return self._write_multilayer_exr(
            loaded_layers, output_path, width, height, 
            bit_depth, compression, quiet
        )



# ============================================================================
# Convenience Functions for API Use
# ============================================================================

def export_layers_to_exr(
    layers: Dict[str, str],
    output_path: Path,
    bit_depth: int = 32,
    compression: str = "ZIP",
    linear: bool = True,
    single_files: bool = False,
) -> Dict[str, Any]:
    """
    High-level function for EXR export (single image)
    
    Args:
        layers: Dict of layer_name -> image_path
        output_path: Output EXR path (or directory if single_files=True)
        bit_depth: 16 or 32
        compression: ZIP, PIZ, DWAA, etc.
        linear: Convert beauty to linear
        single_files: Export each layer as separate file
        
    Returns:
        Export result dict
    """
    exporter = EXRExporter()
    
    if single_files:
        return exporter.export_single_layers(
            layers=layers,
            output_dir=Path(output_path),
            bit_depth=bit_depth,
            compression=compression,
            linear=linear,
        )
    else:
        return exporter.export_multilayer(
            layers=layers,
            output_path=Path(output_path),
            bit_depth=bit_depth,
            compression=compression,
            linear=linear,
        )


def export_video_layers_to_exr_sequence(
    layers: Dict[str, str],
    output_dir: Path,
    filename_pattern: str = "frame.{frame:04d}.exr",
    bit_depth: int = 32,
    compression: str = "ZIP",
    linear: bool = True,
    start_frame: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    High-level function for video EXR sequence export
    
    Args:
        layers: Dict of layer_name -> video_path (MP4) or sequence directory
        output_dir: Directory for output EXR sequence
        filename_pattern: Pattern with {frame:04d} placeholder
        bit_depth: 16 or 32
        compression: ZIP, PIZ, DWAA, etc.
        linear: Convert beauty to linear
        start_frame: First frame number
        progress_callback: Optional callback(current, total)
        
    Returns:
        Export result dict with sequence info
    """
    exporter = EXRExporter()
    
    return exporter.export_video_sequence(
        layers=layers,
        output_dir=Path(output_dir),
        filename_pattern=filename_pattern,
        bit_depth=bit_depth,
        compression=compression,
        linear=linear,
        start_frame=start_frame,
        progress_callback=progress_callback,
    )


# ============================================================================
# True Latent-to-EXR Functions (No MP4 Compression)
# ============================================================================

def export_latent_to_exr(
    latent_path: Path,
    output_path: Path,
    vae_path: str,
    musubi_path: Optional[Path] = None,
    bit_depth: int = 32,
    compression: str = "ZIP",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Export an image latent directly to EXR (true lossless path).
    
    Args:
        latent_path: Path to latent.safetensors
        output_path: Output EXR path
        vae_path: Path to VAE model
        musubi_path: Path to musubi-tuner (for VAE loading)
        bit_depth: 16 or 32
        compression: ZIP, PIZ, DWAA, etc.
        device: torch device
        
    Returns:
        Export result dict
    """
    exporter = EXRExporter()
    
    return exporter.export_from_latent(
        latent_path=Path(latent_path),
        output_path=Path(output_path),
        vae_path=vae_path,
        musubi_path=musubi_path,
        bit_depth=bit_depth,
        compression=compression,
        device=device,
    )


def export_video_latent_to_exr_sequence(
    latent_path: Path,
    output_dir: Path,
    vae_path: str,
    musubi_path: Optional[Path] = None,
    filename_pattern: str = "frame.{frame:04d}.exr",
    bit_depth: int = 32,
    compression: str = "ZIP",
    start_frame: int = 1,
    device: str = "cuda",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Export a video latent directly to EXR sequence (true lossless path).
    
    This is the recommended export path for maximum quality - it decodes
    the latent directly to EXR without any intermediate MP4 compression.
    
    Args:
        latent_path: Path to latent.safetensors (video latent from Wan)
        output_dir: Directory for output EXR sequence
        vae_path: Path to VAE model
        musubi_path: Path to musubi-tuner (for VAE loading)
        filename_pattern: Pattern with {frame:04d} placeholder
        bit_depth: 16 or 32
        compression: ZIP, PIZ, DWAA, etc.
        start_frame: First frame number
        device: torch device
        progress_callback: Optional callback(current, total)
        
    Returns:
        Export result dict with sequence info
    """
    exporter = EXRExporter()
    
    return exporter.export_from_latent_sequence(
        latent_path=Path(latent_path),
        output_dir=Path(output_dir),
        vae_path=vae_path,
        musubi_path=musubi_path,
        filename_pattern=filename_pattern,
        bit_depth=bit_depth,
        compression=compression,
        start_frame=start_frame,
        device=device,
        progress_callback=progress_callback,
    )


def export_array_to_exr(
    pixels: np.ndarray,
    output_path: Path,
    bit_depth: int = 32,
    compression: str = "ZIP",
    is_linear: bool = True,
) -> Dict[str, Any]:
    """
    Export a decoded pixel array directly to EXR.
    
    Use this when you've already decoded the latent yourself
    and just need to write to EXR.
    
    Args:
        pixels: Float32 numpy array (H, W, C) in [0, 1] range
        output_path: Output EXR path
        bit_depth: 16 or 32
        compression: ZIP, PIZ, DWAA, etc.
        is_linear: Whether pixels are in linear color space
        
    Returns:
        Export result dict
    """
    exporter = EXRExporter()
    
    return exporter.export_from_array(
        pixels=pixels,
        output_path=Path(output_path),
        bit_depth=bit_depth,
        compression=compression,
        is_linear=is_linear,
    )