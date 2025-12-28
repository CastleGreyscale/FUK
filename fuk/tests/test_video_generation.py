# test_exr_video_workflow.py
"""
Test EXR import/export workflows for video generation
"""
from pathlib import Path
from core.wan_video_wrapper import WanVideoGenerator, WanTask
from core.video_generation_manager import VideoGenerationManager
from utils.format_convert import FormatConverter


def test_exr_export_workflow():
    """
    Test: Generate video -> Export to EXR sequence
    
    This workflow is for when you want professional-grade output
    for compositing in Natron/Nuke/After Effects
    """
    
    config_path = Path("config/models.json")
    musubi_path = Path("vendor/musubi-tuner")
    defaults_path = Path("config/defaults.json")
    output_root = Path("outputs/video")
    
    generator = WanVideoGenerator(config_path, musubi_path, defaults_path)
    manager = VideoGenerationManager(output_root)
    
    gen_dir = manager.create_generation_dir(workflow_type="i2v_exr")
    paths = manager.get_output_paths(gen_dir)
    
    print(f"Generation directory: {gen_dir}")
    
    # Generate video
    prompt = "A wide shot. A horse runs away and a man chases the horse."
    start_image = Path("inputs/funtest_refimg01.png")
    
    task = WanTask.I2V_14B
    video_size = (720, 1280)
    video_length = 61
    seed = 5366815
    
    try:
        # Copy control inputs
        manager.copy_control_inputs(gen_dir, image_path=start_image)
        
        # Generate video
        print("\n=== Step 1: Generate Video ===")
        generator.generate_video(
            prompt=prompt,
            output_path=paths["video_mp4"],
            task=task,
            video_size=video_size,
            video_length=video_length,
            seed=seed,
            image_path=start_image,
            infer_steps=20,
            guidance_scale=2.5,
            flow_shift=2.1,
            fp8=True,
            fp8_scaled=False,
            fp8_t5=True,
            blocks_to_swap=15
        )
        print(f"✓ Generated video: {paths['video_mp4']}")
        
        # Export to EXR sequence
        print("\n=== Step 2: Export to EXR Sequence ===")
        exr_dir = manager.export_to_exr_sequence(
            gen_dir=gen_dir,
            video_path=paths["video_mp4"],
            linear=True  # Linear color space for compositing
        )
        print(f"✓ EXR sequence: {exr_dir}")
        
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
        print(f"✓ Metadata saved")
        
        print("\n=== Workflow Complete ===")
        print(f"Video: {paths['video_mp4']}")
        print(f"EXR Frames: {exr_dir}")
        
    except Exception as e:
        print(f"â✓— Test failed: {e}")
        raise


def test_exr_import_workflow():
    """
    Test: Import EXR sequence -> Generate video with it
    
    This workflow is for when you have existing EXR renders
    that you want to use as control input for Fun Control
    """
    
    exr_input_dir = Path("inputs/exr_sequence")
    
    if not exr_input_dir.exists():
        print(f"Skipping EXR import test - {exr_input_dir} not found")
        print("Create this directory with EXR frames to test import workflow")
        return
    
    config_path = Path("config/models.json")
    musubi_path = Path("vendor/musubi-tuner")
    defaults_path = Path("config/defaults.json")
    output_root = Path("outputs/video")
    
    generator = WanVideoGenerator(config_path, musubi_path, defaults_path)
    manager = VideoGenerationManager(output_root)
    
    gen_dir = manager.create_generation_dir(workflow_type="exr_control")
    paths = manager.get_output_paths(gen_dir)
    
    print(f"Generation directory: {gen_dir}")
    
    # Convert EXR sequence to PNG for use as control
    print("\n=== Step 1: Convert EXR to PNG for control input ===")
    control_png_dir = gen_dir / "control_frames_from_exr"
    
    png_frames = FormatConverter.exr_sequence_to_png_sequence(
        exr_dir=exr_input_dir,
        output_dir=control_png_dir,
        linear=True  # Convert from linear to sRGB for model input
    )
    print(f"✓ Converted {len(png_frames)} frames to PNG")
    
    # Use first frame as start image
    start_image = png_frames[0]
    
    # Generate controlled video
    print("\n=== Step 2: Generate with EXR-derived control ===")
    prompt = "A cinematic shot with dramatic lighting"
    
    task = WanTask.I2V_14B_FC  # Fun Control
    video_size = (720, 1280)
    video_length = min(81, len(png_frames))  # Match sequence length or cap at 81
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
        print(f"✓ Generated video: {paths['video_mp4']}")
        
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
        
        print("\n=== Workflow Complete ===")
        
    except Exception as e:
        print(f"â✓— Test failed: {e}")
        raise


def test_exr_roundtrip():
    """
    Test: Video -> EXR sequence -> Video
    
    Quality verification workflow
    """
    
    input_video = Path("inputs/test_video.mp4")
    
    if not input_video.exists():
        print(f"Skipping roundtrip test - {input_video} not found")
        return
    
    output_root = Path("outputs/test_exr_roundtrip")
    output_root.mkdir(exist_ok=True)
    
    print("\n=== EXR Roundtrip Test ===")
    print(f"Input: {input_video}")
    
    # Step 1: Video -> EXR
    print("\nStep 1: Video -> EXR sequence")
    exr_dir = output_root / "exr_frames"
    exr_frames = FormatConverter.video_to_exr_sequence(
        video_path=input_video,
        output_dir=exr_dir,
        linear=True
    )
    print(f"✓ Created {len(exr_frames)} EXR frames")
    
    # Step 2: EXR -> Video
    print("\nStep 2: EXR sequence -> Video")
    output_video = output_root / "reconstructed.mp4"
    FormatConverter.exr_sequence_to_video(
        exr_dir=exr_dir,
        output_path=output_video,
        fps=24,
        linear=True,
        crf=18  # Visually lossless
    )
    print(f"✓ Created video: {output_video}")
    
    print("\n=== Roundtrip Complete ===")
    print("Compare original and reconstructed videos to verify quality")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "export":
            print("=== Testing EXR Export Workflow ===")
            test_exr_export_workflow()
        elif test_name == "import":
            print("=== Testing EXR Import Workflow ===")
            test_exr_import_workflow()
        elif test_name == "roundtrip":
            print("=== Testing EXR Roundtrip ===")
            test_exr_roundtrip()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: export, import, roundtrip")
            sys.exit(1)
    else:
        # Run export workflow by default (most common use case)
        print("=== Testing EXR Export Workflow ===")
        test_exr_export_workflow()
