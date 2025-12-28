#!/usr/bin/env python3
# test_exr_video_workflow.py
"""
EXR Video Workflow Test

WORKFLOW: PNG input → MP4 generation → EXR export
         (for model)   (from Wan)      (for compositing)

You CANNOT use EXR as input - PIL doesn't support it.
You CAN export MP4 output to EXR sequence for compositing.
"""
from pathlib import Path
from core.wan_video_wrapper import WanVideoGenerator, WanTask
from core.video_generation_manager import VideoGenerationManager
from utils.format_convert import FormatConverter


def test_exr_export_workflow():
    """
    WORKFLOW:
    1. Take PNG input image
    2. Generate MP4 video with Wan
    3. Export MP4 to EXR sequence for compositing
    """
    
    # ALL PATHS MUST BE ABSOLUTE - musubi runs from different directory
    config_path = Path("config/models.json").resolve()
    musubi_path = Path("vendor/musubi-tuner").resolve()
    defaults_path = Path("config/defaults.json").resolve()
    output_root = Path("outputs/video").resolve()
    
    # INPUT MUST BE PNG/JPG - NOT EXR
    start_image = Path("inputs/funtest_refimg01.png").resolve()
    
    # Verify input exists BEFORE starting
    if not start_image.exists():
        print(f"ERROR: Input image not found: {start_image}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Looking for: {start_image}")
        raise FileNotFoundError(f"Cannot find input: {start_image}")
    
    print(f"✓ Found input image: {start_image}")
    print(f"  Size: {start_image.stat().st_size / 1024:.1f} KB\n")
    
    generator = WanVideoGenerator(config_path, musubi_path, defaults_path)
    manager = VideoGenerationManager(output_root)
    
    gen_dir = manager.create_generation_dir(workflow_type="i2v_exr")
    paths = manager.get_output_paths(gen_dir)
    
    print(f"Generation directory: {gen_dir}\n")
    
    # STEP 1: Generate MP4 video from PNG input
    prompt = "A wide shot. A horse runs away and a man chases the horse."
    
    task = WanTask.I2V_14B
    video_size = (720, 1280)
    video_length = 41
    seed = 42
    
    try:
        # Copy input for reference
        manager.copy_control_inputs(gen_dir, image_path=start_image)
        
        print("=== STEP 1: Generate MP4 Video ===")
        print(f"Input: {start_image.name} (PNG)")
        print(f"Output: {paths['video_mp4'].name} (MP4)\n")
        
        generator.generate_video(
            prompt=prompt,
            output_path=paths["video_mp4"],
            task=task,
            video_size=video_size,
            video_length=video_length,
            seed=seed,
            image_path=start_image,  # This MUST be absolute path
            infer_steps=20,
            guidance_scale=2.5,
            flow_shift=2.1,
            fp8=True,
            fp8_scaled=False,
            fp8_t5=True,
            blocks_to_swap=15
        )
        
        # Verify MP4 was created
        if not paths["video_mp4"].exists():
            raise RuntimeError("MP4 generation failed - file not created")
        
        mp4_size = paths["video_mp4"].stat().st_size / (1024 * 1024)
        print(f"✓ Generated MP4: {paths['video_mp4']}")
        print(f"  Size: {mp4_size:.2f} MB\n")
        
        # STEP 2: Export MP4 to EXR sequence
        print("=== STEP 2: Export MP4 → EXR Sequence ===")
        print(f"Input: {paths['video_mp4'].name} (MP4)")
        print(f"Output: EXR frames (32-bit linear)\n")
        
        exr_dir = manager.export_to_exr_sequence(
            gen_dir=gen_dir,
            video_path=paths["video_mp4"],
            linear=True  # Linear color space for compositing
        )
        
        # Verify EXR frames were created
        exr_frames = sorted(exr_dir.glob("*.exr"))
        if not exr_frames:
            raise RuntimeError("EXR export failed - no frames created")
        
        total_exr_size = sum(f.stat().st_size for f in exr_frames) / (1024 * 1024)
        print(f"✓ Created {len(exr_frames)} EXR frames")
        print(f"  Total size: {total_exr_size:.2f} MB")
        print(f"  Location: {exr_dir}\n")
        
        # Save metadata
        manager.save_metadata(
            gen_dir=gen_dir,
            prompt=prompt,
            enhanced_prompt=prompt,
            task=task.value,
            video_size=video_size,
            video_length=video_length,
            seed=seed,
            image_path=start_image,
            exr_exported=True,
            exr_sequence_path=str(exr_dir)
        )
        print(f"✓ Metadata saved\n")
        
        print("=== Workflow Complete ===")
        print(f"MP4 video:    {paths['video_mp4']}")
        print(f"EXR sequence: {exr_dir}")
        print(f"\nImport {exr_dir} into Natron/Nuke for compositing")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_exr_import_workflow():
    """
    WORKFLOW:
    1. Take existing EXR sequence
    2. Convert to PNG for model input
    3. Use PNG sequence as Fun Control input
    4. Generate new MP4
    """
    
    exr_input_dir = Path("inputs/exr_sequence").resolve()
    
    if not exr_input_dir.exists():
        print(f"Skipping EXR import test - {exr_input_dir} not found")
        print("To test: Create 'inputs/exr_sequence/' with EXR frames")
        return
    
    config_path = Path("config/models.json").resolve()
    musubi_path = Path("vendor/musubi-tuner").resolve()
    defaults_path = Path("config/defaults.json").resolve()
    output_root = Path("outputs/video").resolve()
    
    generator = WanVideoGenerator(config_path, musubi_path, defaults_path)
    manager = VideoGenerationManager(output_root)
    
    gen_dir = manager.create_generation_dir(workflow_type="exr_control")
    paths = manager.get_output_paths(gen_dir)
    
    print(f"Generation directory: {gen_dir}\n")
    
    # STEP 1: Convert EXR sequence to PNG
    print("=== STEP 1: Convert EXR → PNG ===")
    print(f"Input: {exr_input_dir}")
    control_png_dir = gen_dir / "control_frames_from_exr"
    
    png_frames = FormatConverter.exr_sequence_to_png_sequence(
        exr_dir=exr_input_dir,
        output_dir=control_png_dir,
        linear=True  # Convert from linear to sRGB
    )
    print(f"✓ Converted {len(png_frames)} EXR → PNG\n")
    
    # Use first frame as start image
    start_image = png_frames[0]
    
    # STEP 2: Generate with Fun Control
    print("=== STEP 2: Generate with Control ===")
    prompt = "A cinematic shot with dramatic lighting"
    
    task = WanTask.I2V_14B_FC  # Fun Control
    video_size = (720, 1280)
    video_length = min(81, len(png_frames))  # Match sequence length
    seed = 42
    
    try:
        generator.generate_video(
            prompt=prompt,
            output_path=paths["video_mp4"],
            task=task,
            video_size=video_size,
            video_length=video_length,
            seed=seed,
            image_path=start_image,
            control_path=control_png_dir,
            infer_steps=20,
            guidance_scale=5.0,
            fp8=True,
            fp8_scaled=True,
            fp8_t5=True,
            blocks_to_swap=15
        )
        print(f"✓ Generated video: {paths['video_mp4']}\n")
        
        manager.save_metadata(
            gen_dir=gen_dir,
            prompt=prompt,
            enhanced_prompt=prompt,
            task=task.value,
            video_size=video_size,
            video_length=video_length,
            seed=seed,
            image_path=start_image,
            control_path=control_png_dir,
            exr_source=str(exr_input_dir)
        )
        
        print("=== Workflow Complete ===")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


def test_exr_roundtrip():
    """
    Quality test: MP4 → EXR → MP4
    """
    
    input_video = Path("inputs/test_video.mp4").resolve()
    
    if not input_video.exists():
        print(f"Skipping roundtrip test - {input_video} not found")
        return
    
    output_root = Path("outputs/test_exr_roundtrip").resolve()
    output_root.mkdir(exist_ok=True, parents=True)
    
    print("=== EXR Roundtrip Test ===")
    print(f"Input: {input_video}\n")
    
    # MP4 → EXR
    print("Step 1: MP4 → EXR sequence")
    exr_dir = output_root / "exr_frames"
    exr_frames = FormatConverter.video_to_exr_sequence(
        video_path=input_video,
        output_dir=exr_dir,
        linear=True
    )
    print(f"✓ Created {len(exr_frames)} EXR frames\n")
    
    # EXR → MP4
    print("Step 2: EXR sequence → MP4")
    output_video = output_root / "reconstructed.mp4"
    FormatConverter.exr_sequence_to_video(
        exr_dir=exr_dir,
        output_path=output_video,
        fps=24,
        linear=True,
        crf=18  # Visually lossless
    )
    print(f"✓ Created: {output_video}\n")
    
    print("=== Roundtrip Complete ===")
    print(f"Original: {input_video}")
    print(f"Reconstructed: {output_video}")
    print("Compare the two to verify quality preservation")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "export":
            test_exr_export_workflow()
        elif test_name == "import":
            test_exr_import_workflow()
        elif test_name == "roundtrip":
            test_exr_roundtrip()
        else:
            print(f"Unknown test: {test_name}")
            print("Available: export, import, roundtrip")
            sys.exit(1)
    else:
        # Default: most common workflow
        test_exr_export_workflow()