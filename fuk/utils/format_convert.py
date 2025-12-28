# utils/format_convert.py
from pathlib import Path
import numpy as np
from PIL import Image
import OpenEXR
import Imath
import subprocess
from typing import Optional, List

class FormatConverter:
    
    @staticmethod
    def png_to_exr_32bit(input_path: Path, output_path: Path, linear: bool = True):
        """Convert PNG to 32-bit linear EXR"""
        
        # Load PNG
        img = Image.open(input_path)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # If not already linear, convert from sRGB
        if not linear:
            img_array = FormatConverter._srgb_to_linear(img_array)
        
        height, width = img_array.shape[:2]
        
        # Prepare EXR header
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        
        # Handle RGB vs RGBA
        if img_array.shape[2] == 4:
            header['channels']['A'] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        
        # Write EXR
        exr = OpenEXR.OutputFile(str(output_path), header)
        
        channels = {
            'R': img_array[:, :, 0].tobytes(),
            'G': img_array[:, :, 1].tobytes(),
            'B': img_array[:, :, 2].tobytes()
        }
        
        if img_array.shape[2] == 4:
            channels['A'] = img_array[:, :, 3].tobytes()
        
        exr.writePixels(channels)
        exr.close()
        
        return output_path
    
    @staticmethod
    def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear color space"""
        return np.where(
            img <= 0.04045,
            img / 12.92,
            np.power((img + 0.055) / 1.055, 2.4)
        )
    
    @staticmethod
    def png_16bit(input_path: Path, output_path: Path):
        """Convert to 16-bit PNG"""
        img = Image.open(input_path)
        # Convert to 16-bit
        img_16 = img.convert('I;16')
        img_16.save(output_path)
        return output_path
    
    # ===== EXR Reading =====
    
    @staticmethod
    def exr_to_png(input_path: Path, output_path: Path, linear: bool = True) -> Path:
        """
        Convert 32-bit linear EXR to PNG
        
        Args:
            input_path: EXR file to read
            output_path: PNG file to write
            linear: If True, convert from linear to sRGB
        """
        exr_file = OpenEXR.InputFile(str(input_path))
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Read channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = exr_file.header()['channels']
        
        # Read RGB
        r_str = exr_file.channel('R', FLOAT)
        g_str = exr_file.channel('G', FLOAT)
        b_str = exr_file.channel('B', FLOAT)
        
        r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
        g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width))
        b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width))
        
        # Stack channels
        if 'A' in channels:
            a_str = exr_file.channel('A', FLOAT)
            a = np.frombuffer(a_str, dtype=np.float32).reshape((height, width))
            img_array = np.stack([r, g, b, a], axis=2)
        else:
            img_array = np.stack([r, g, b], axis=2)
        
        # Convert to sRGB if needed
        if linear:
            img_array = FormatConverter._linear_to_srgb(img_array)
        
        # Clamp and convert to 8-bit
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(img_array)
        img.save(output_path)
        
        return output_path
    
    @staticmethod
    def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
        """Convert linear to sRGB color space"""
        return np.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * np.power(img, 1/2.4) - 0.055
        )
    
    # ===== Video Sequence Handling =====
    
    @staticmethod
    def video_to_png_sequence(video_path: Path, 
                             output_dir: Path,
                             frame_pattern: str = "frame_%04d.png",
                             fps: Optional[int] = None) -> List[Path]:
        """
        Extract video frames to PNG sequence using ffmpeg
        
        Args:
            video_path: Input video file (mp4, mov, etc)
            output_dir: Directory to save frames
            frame_pattern: Naming pattern (must include %d format specifier)
            fps: Optional FPS for extraction (None = use source fps)
            
        Returns:
            List of generated frame paths
        """
        video_path = Path(video_path)
        
        # Validate input file exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_path.is_dir():
            raise ValueError(f"Expected file, got directory: {video_path}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_pattern = output_dir / frame_pattern
        
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]  # Added -y to overwrite
        
        if fps:
            cmd.extend(["-r", str(fps)])
        
        cmd.extend([
            "-qscale:v", "1",  # Highest quality
            str(output_pattern)
        ])
        
        print(f"Extracting frames: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg stdout: {result.stdout}")
            print(f"ffmpeg stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        # Return list of generated frames
        frames = sorted(output_dir.glob("frame_*.png"))
        print(f"Extracted {len(frames)} frames to {output_dir}")
        
        return frames
    
    @staticmethod
    def video_to_exr_sequence(video_path: Path,
                             output_dir: Path,
                             frame_pattern: str = "frame_%04d.exr",
                             linear: bool = True,
                             fps: Optional[int] = None) -> List[Path]:
        """
        Convert video to EXR frame sequence
        
        Args:
            video_path: Input video
            output_dir: Directory for EXR frames
            frame_pattern: Naming pattern for EXR files
            linear: Convert to linear color space
            fps: Optional FPS override
            
        Returns:
            List of EXR frame paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Validate input
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_path.is_dir():
            raise ValueError(f"Expected video file, got directory: {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # First extract to PNG
        temp_dir = output_dir / "temp_png"
        temp_dir.mkdir(exist_ok=True)
        
        png_frames = FormatConverter.video_to_png_sequence(
            video_path, temp_dir, "temp_%04d.png", fps
        )
        
        # Convert each PNG to EXR
        exr_frames = []
        for i, png_path in enumerate(png_frames, start=1):
            exr_path = output_dir / (frame_pattern % i)
            FormatConverter.png_to_exr_32bit(png_path, exr_path, linear=linear)
            exr_frames.append(exr_path)
            print(f"Converted {i}/{len(png_frames)}: {exr_path.name}")
        
        # Clean up temp PNGs
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"Created {len(exr_frames)} EXR frames in {output_dir}")
        return exr_frames
    
    @staticmethod
    def exr_sequence_to_png_sequence(exr_dir: Path,
                                    output_dir: Path,
                                    exr_pattern: str = "*.exr",
                                    frame_pattern: str = "frame_%04d.png",
                                    linear: bool = True) -> List[Path]:
        """
        Convert EXR sequence to PNG sequence
        
        Args:
            exr_dir: Directory containing EXR files
            output_dir: Output directory for PNGs
            exr_pattern: Glob pattern for EXR files
            frame_pattern: Output naming pattern
            linear: Convert from linear to sRGB
            
        Returns:
            List of PNG frame paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exr_files = sorted(Path(exr_dir).glob(exr_pattern))
        
        if not exr_files:
            raise ValueError(f"No EXR files found in {exr_dir} matching {exr_pattern}")
        
        png_frames = []
        for i, exr_path in enumerate(exr_files, start=1):
            png_path = output_dir / (frame_pattern % i)
            FormatConverter.exr_to_png(exr_path, png_path, linear=linear)
            png_frames.append(png_path)
            print(f"Converted {i}/{len(exr_files)}: {png_path.name}")
        
        print(f"Created {len(png_frames)} PNG frames in {output_dir}")
        return png_frames
    
    @staticmethod
    def png_sequence_to_video(png_dir: Path,
                             output_path: Path,
                             fps: int = 24,
                             frame_pattern: str = "frame_%04d.png",
                             codec: str = "libx264",
                             crf: int = 18) -> Path:
        """
        Encode PNG sequence to video using ffmpeg
        
        Args:
            png_dir: Directory containing PNG frames
            output_path: Output video path
            fps: Frames per second
            frame_pattern: Input frame pattern
            codec: Video codec (libx264, libx265, prores, etc)
            crf: Quality (0-51, lower is better quality, 18 is visually lossless)
            
        Returns:
            Path to created video
        """
        input_pattern = png_dir / frame_pattern
        
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", str(input_pattern),
            "-c:v", codec,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",  # Compatibility
            str(output_path)
        ]
        
        print(f"Encoding video: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg stdout: {result.stdout}")
            print(f"ffmpeg stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg encoding failed: {result.stderr}")
        
        print(f"Created video: {output_path}")
        return output_path
    
    @staticmethod
    def exr_sequence_to_video(exr_dir: Path,
                             output_path: Path,
                             fps: int = 24,
                             linear: bool = True,
                             **kwargs) -> Path:
        """
        Convert EXR sequence directly to video
        
        Args:
            exr_dir: Directory with EXR frames
            output_path: Output video path
            fps: Frames per second
            linear: Convert from linear to sRGB
            **kwargs: Additional args for png_sequence_to_video
            
        Returns:
            Path to created video
        """
        # Convert EXRs to temp PNG sequence
        temp_dir = output_path.parent / "temp_png_for_encode"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            FormatConverter.exr_sequence_to_png_sequence(
                exr_dir, temp_dir, linear=linear
            )
            
            # Encode to video
            FormatConverter.png_sequence_to_video(
                temp_dir, output_path, fps=fps, **kwargs
            )
            
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir)
        
        return output_path