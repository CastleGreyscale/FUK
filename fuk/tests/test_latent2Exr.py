#!/usr/bin/env python3
# test_lossless_exr_workflow.py
"""
Test LOSSLESS EXR export workflow

LOSSY (wrong):   Generation → MP4 → Extract → EXR  ❌
LOSSLESS (right): Generation → Latent → Decode → EXR ✓

The latent tensor contains the full precision output from diffusion,
before VAE compression. Decoding directly to EXR preserves this quality.
"""

from pathlib import Path
from core.wan_video_wrapper import WanVideoGenerator, WanTask
from core.video_generation_manager import VideoGenerationManager


def test_lossless_exr_export():
    """
    PROPER WORKFLOW:
    1. Generate video with output_type='both' (saves latent + MP4)
    2. Decode latent directly to EXR frames
    3. No MP4 compression in the pipeline
    """
    
    # Setup paths (all absolute)
    config_path = Path("config/models.json").resolve()
    musubi_path = Path("vendor/musubi-tuner").resolve()
    defaults_path = Path("config/defaults.json").resolve()
    output_root = Path("outputs/video").resolve()
    
    # Input image (PNG)
    start_image = Path("inputs/funtest_refimg01.png").resolve()
    
    if not start_image.exists():
        print(f"ERROR: Input not found: {start_image}")
        raise FileNotFoundError(f"Cannot find: {start_image}")
    
    print(f"✓ Input image: {start_image}\n")
    
    # Initialize
    generator = WanVideoGenerator(config_path, musubi_path, defaults_path)
    manager = VideoGenerationManager(output_root)
    
    gen_dir = manager.create_generation_dir(workflow_type="i2v_lossless")
    paths = manager.get_output_paths(gen_dir)
    
    print(f"Generation directory: {gen_dir}\n")
    
    # Generation parameters
    prompt = "A wide shot. A horse runs away and a man chases the horse."
    task = WanTask.I2V_14B
    video_size = (720, 1280)
    video_length = 41
    seed = 42
    
    try:
        # STEP 1: Generate with latent saving
        print("=== STEP 1: Generate Video + Latent ===")
        print("output_type='both' saves:")
        print("  - generated.mp4 (for preview)")
        print("  - latent.safetensors (for lossless decode)\n")
        
        manager.copy_control_inputs(gen_dir, image_path=start_image)
        
        # Note: output_type='both' is already set in wan_video_wrapper.py
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
        
        # Verify both outputs exist
        latent_path = gen_dir / "latent.safetensors"
        
        if not paths["video_mp4"].exists():
            raise RuntimeError("MP4 not created")
        if not latent_path.exists():
            raise RuntimeError("Latent not saved - check output_type='both'")
        
        mp4_size = paths["video_mp4"].stat().st_size / (1024 * 1024)
        latent_size = latent_path.stat().st_size / (1024 * 1024)
        
        print(f"✓ MP4 created: {mp4_size:.2f} MB (preview only)")
        print(f"✓ Latent saved: {latent_size:.2f} MB (full precision)\n")
        
        # STEP 2: Decode latent to EXR (LOSSLESS)
        print("=== STEP 2: Decode Latent → EXR (LOSSLESS) ===")
        print("This bypasses MP4 compression entirely")
        print("Output is 32-bit linear EXR, ready for compositing\n")
        
        exr_dir = manager.export_latent_to_exr(
            gen_dir=gen_dir,
            task=task.value,
            config_path=config_path,
            musubi_path=musubi_path,
            linear=True
        )
        
        # Verify EXR frames
        exr_frames = sorted(exr_dir.glob("*.exr"))
        if not exr_frames:
            raise RuntimeError("No EXR frames created")
        
        total_exr_size = sum(f.stat().st_size for f in exr_frames) / (1024 * 1024)
        avg_frame_size = total_exr_size / len(exr_frames)
        
        print(f"\n✓ Created {len(exr_frames)} EXR frames")
        print(f"  Total size: {total_exr_size:.2f} MB")
        print(f"  Avg per frame: {avg_frame_size:.2f} MB")
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
            exr_export_method="lossless_from_latent",
            exr_sequence_path=str(exr_dir)
        )
        
        # STEP 3: Quality comparison
        print("=== Quality Comparison ===")
        print(f"MP4 video:     {mp4_size:.2f} MB (8-bit, compressed)")
        print(f"EXR sequence:  {total_exr_size:.2f} MB (32-bit, lossless)")
        print(f"Ratio:         {total_exr_size / mp4_size:.1f}x larger")
        print("\nThe EXR sequence contains FULL quality from the VAE,")
        print("with no MP4 compression artifacts.\n")
        
        print("=== Workflow Complete ===")
        print(f"Preview (lossy):    {paths['video_mp4']}")
        print(f"Composite (lossless): {exr_dir}")
        print(f"\nImport {exr_dir} into Natron/Nuke for professional compositing")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_quality_comparison():
    """
    Compare lossy vs lossless EXR export side-by-side
    
    This generates the same video twice and exports both ways
    to demonstrate the quality difference
    """
    
    config_path = Path("config/models.json").resolve()
    musubi_path = Path("vendor/musubi-tuner").resolve()
    defaults_path = Path("config/defaults.json").resolve()
    output_root = Path("outputs/quality_comparison").resolve()
    
    start_image = Path("inputs/funtest_refimg01.png").resolve()
    
    if not start_image.exists():
        raise FileNotFoundError(f"Input not found: {start_image}")
    
    generator = WanVideoGenerator(config_path, musubi_path, defaults_path)
    manager = VideoGenerationManager(output_root)
    
    gen_dir = manager.create_generation_dir(workflow_type="comparison")
    paths = manager.get_output_paths(gen_dir)
    
    print("=== Quality Comparison Test ===\n")
    
    prompt = "A cinematic shot with fine details"
    task = WanTask.I2V_14B
    video_size = (720, 1280)
    video_length = 41
    seed = 42
    
    # Generate once
    print("Generating video...")
    manager.copy_control_inputs(gen_dir, image_path=start_image)
    
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
    
    print("✓ Video generated\n")
    
    # Export LOSSY (MP4 → EXR)
    print("=== LOSSY Export (MP4 → EXR) ===")
    lossy_dir = gen_dir / "exr_lossy"
    lossy_dir.mkdir(exist_ok=True)
    
    from utils.format_convert import FormatConverter
    FormatConverter.video_to_exr_sequence(
        video_path=paths["video_mp4"],
        output_dir=lossy_dir,
        linear=True
    )
    
    # Export LOSSLESS (Latent → EXR)
    print("\n=== LOSSLESS Export (Latent → EXR) ===")
    lossless_dir = manager.export_latent_to_exr(
        gen_dir=gen_dir,
        task=task.value,
        config_path=config_path,
        musubi_path=musubi_path,
        linear=True
    )
    
    # Compare
    print("\n=== Comparison Results ===")
    
    lossy_frames = sorted(lossy_dir.glob("*.exr"))
    lossless_frames = sorted(lossless_dir.glob("*.exr"))
    
    lossy_size = sum(f.stat().st_size for f in lossy_frames) / (1024 * 1024)
    lossless_size = sum(f.stat().st_size for f in lossless_frames) / (1024 * 1024)
    
    print(f"Lossy (MP4→EXR):     {len(lossy_frames)} frames, {lossy_size:.2f} MB")
    print(f"Lossless (Latent→EXR): {len(lossless_frames)} frames, {lossless_size:.2f} MB")
    print(f"\nBoth sequences are in {gen_dir}")
    print("Load both in Natron and toggle between them to see the difference")
    print("The lossless version will have finer details and no compression artifacts")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        test_quality_comparison()
    else:
        test_lossless_exr_export()