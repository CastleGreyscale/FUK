# utils/depth_export.py
"""
Depth Export Pipeline
Demonstrates using depth preprocessor for EXR sequence with depth channel

This shows how the depth preprocessor integrates with the output/export section
"""

from pathlib import Path
import numpy as np
from typing import Optional, List
import OpenEXR
import Imath
from utils.preprocessors import PreprocessorManager, DepthModel
from utils.format_convert import FormatConverter


class DepthExporter:
    """
    Export video sequences with embedded depth channels
    
    Workflow:
    1. Generate video with Wan
    2. Extract frames to PNG
    3. Process each frame with depth estimation
    4. Combine RGB + Depth into multi-channel EXR
    5. Ready for compositing in Nuke/Fusion/AE
    """
    
    def __init__(self, preprocessor: PreprocessorManager):
        self.preprocessor = preprocessor
    
    def export_video_with_depth(
        self,
        video_path: Path,
        output_dir: Path,
        depth_model: DepthModel = DepthModel.DEPTH_ANYTHING_V2,
        linear_rgb: bool = True,
    ) -> Path:
        """
        Export video as EXR sequence with embedded depth
        
        Args:
            video_path: Input video file
            output_dir: Where to save EXR sequence
            depth_model: Which depth model to use
            linear_rgb: Convert RGB to linear color space
            
        Returns:
            Path to EXR sequence directory
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Exporting Video with Depth Channel")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Output: {output_dir}")
        print(f"Depth Model: {depth_model.value}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract video to PNG frames
        print("[1/4] Extracting video frames...")
        temp_png_dir = output_dir / "temp_png"
        temp_png_dir.mkdir(exist_ok=True)
        
        rgb_frames = FormatConverter.video_to_png_sequence(
            video_path=video_path,
            output_dir=temp_png_dir,
            frame_pattern="frame_%04d.png"
        )
        
        print(f"  ✓ Extracted {len(rgb_frames)} frames\n")
        
        # Step 2: Process depth for each frame
        print(f"[2/4] Generating depth maps ({depth_model.value})...")
        temp_depth_dir = output_dir / "temp_depth"
        temp_depth_dir.mkdir(exist_ok=True)
        
        depth_frames = []
        for i, rgb_frame in enumerate(rgb_frames, start=1):
            # Generate depth (grayscale, no colormap for processing)
            depth_result = self.preprocessor.depth(
                image_path=rgb_frame,
                output_name=f"depth_{i:04d}.png",
                model=depth_model,
                colormap=None,  # Grayscale for channel embedding
                normalize=True,
                invert=False,
            )
            
            depth_frames.append(Path(depth_result['output_path']))
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(rgb_frames)}")
        
        print(f"  ✓ Generated {len(depth_frames)} depth maps\n")
        
        # Step 3: Combine RGB + Depth into multi-channel EXR
        print("[3/4] Creating multi-channel EXR sequence...")
        exr_dir = output_dir / "exr_with_depth"
        exr_dir.mkdir(exist_ok=True)
        
        exr_frames = []
        for i, (rgb_path, depth_path) in enumerate(zip(rgb_frames, depth_frames), start=1):
            exr_path = exr_dir / f"frame_{i:04d}.exr"
            
            self._create_rgbd_exr(
                rgb_path=rgb_path,
                depth_path=depth_path,
                output_path=exr_path,
                linear_rgb=linear_rgb,
            )
            
            exr_frames.append(exr_path)
            
            if i % 10 == 0:
                print(f"  Created {i}/{len(rgb_frames)} EXR files")
        
        print(f"  ✓ Created {len(exr_frames)} multi-channel EXR frames\n")
        
        # Step 4: Cleanup temp files
        print("[4/4] Cleaning up...")
        import shutil
        shutil.rmtree(temp_png_dir)
        shutil.rmtree(temp_depth_dir)
        
        print(f"  ✓ Cleaned up temporary files\n")
        
        print(f"{'='*60}")
        print(f"✓ Export Complete!")
        print(f"{'='*60}")
        print(f"Output: {exr_dir}")
        print(f"Channels: RGB (color) + Z (depth)")
        print(f"Format: 32-bit float EXR")
        print(f"Frames: {len(exr_frames)}")
        print(f"{'='*60}\n")
        
        return exr_dir
    
    def _create_rgbd_exr(
        self,
        rgb_path: Path,
        depth_path: Path,
        output_path: Path,
        linear_rgb: bool = True,
    ):
        """
        Create multi-channel EXR with RGB + Depth
        
        Channels:
        - R, G, B: Color (linear or sRGB)
        - Z: Depth (normalized 0-1)
        """
        import cv2
        from PIL import Image
        
        # Load RGB
        rgb_img = Image.open(rgb_path)
        rgb_array = np.array(rgb_img).astype(np.float32) / 255.0
        
        # Convert to linear if requested
        if linear_rgb:
            rgb_array = self._srgb_to_linear(rgb_array)
        
        # Load depth (already normalized in preprocessing)
        depth_img = Image.open(depth_path).convert('L')  # Grayscale
        depth_array = np.array(depth_img).astype(np.float32) / 255.0
        
        height, width = rgb_array.shape[:2]
        
        # Create EXR header
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),  # Depth channel
        }
        
        # Write EXR
        exr = OpenEXR.OutputFile(str(output_path), header)
        
        channels = {
            'R': rgb_array[:, :, 0].astype(np.float32).tobytes(),
            'G': rgb_array[:, :, 1].astype(np.float32).tobytes(),
            'B': rgb_array[:, :, 2].astype(np.float32).tobytes(),
            'Z': depth_array.astype(np.float32).tobytes(),
        }
        
        exr.writePixels(channels)
        exr.close()
    
    @staticmethod
    def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear color space"""
        return np.where(
            img <= 0.04045,
            img / 12.92,
            np.power((img + 0.055) / 1.055, 2.4)
        )


# Example usage in video_generation_manager.py
"""
Add this method to VideoGenerationManager:

def export_with_depth_channel(
    self,
    gen_dir: Path,
    video_path: Path,
    config_path: Path,
    musubi_path: Path,
    depth_model: str = "depth_anything_v2",
    linear: bool = True,
) -> Path:
    '''
    Export video with embedded depth channel
    
    This creates multi-channel EXR with:
    - R, G, B: Color channels
    - Z: Depth channel
    
    Ready for compositing in professional tools
    '''
    
    from utils.depth_export import DepthExporter, DepthModel
    from utils.preprocessors import PreprocessorManager
    
    # Initialize preprocessor
    preprocessor = PreprocessorManager(gen_dir / "temp_preprocess")
    exporter = DepthExporter(preprocessor)
    
    # Map model string to enum
    depth_model_map = {
        "midas_small": DepthModel.MIDAS_SMALL,
        "midas_large": DepthModel.MIDAS_LARGE,
        "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
        "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
        "zoedepth": DepthModel.ZOEDEPTH,
    }
    
    model = depth_model_map.get(depth_model, DepthModel.DEPTH_ANYTHING_V2)
    
    # Export
    exr_dir = exporter.export_video_with_depth(
        video_path=video_path,
        output_dir=gen_dir,
        depth_model=model,
        linear_rgb=linear,
    )
    
    return exr_dir
"""


# Usage example
if __name__ == "__main__":
    from utils.preprocessors import PreprocessorManager, DepthModel
    
    # Initialize
    preprocessor = PreprocessorManager(Path("outputs/preprocessed"))
    exporter = DepthExporter(preprocessor)
    
    # Export video with depth
    exr_dir = exporter.export_video_with_depth(
        video_path=Path("outputs/video/2025-12-29/i2v_001/generated.mp4"),
        output_dir=Path("outputs/video/2025-12-29/i2v_001/export"),
        depth_model=DepthModel.DEPTH_ANYTHING_V2,
        linear_rgb=True,
    )
    
    print(f"\nReady for compositing!")
    print(f"Import {exr_dir} into Nuke/Fusion/After Effects")
    print(f"Available channels: R, G, B (color) + Z (depth)")
    print(f"\nUse depth for:")
    print(f"  • Depth-based color grading")
    print(f"  • Fog/atmosphere effects")
    print(f"  • Depth of field")
    print(f"  • Camera projection")
    print(f"  • 3D reconstruction")