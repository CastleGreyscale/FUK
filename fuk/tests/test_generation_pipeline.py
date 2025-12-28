# test_generation_pipeline.py
from pathlib import Path
from core.musubi_wrapper import MusubiGenerator, QwenModel
from core.generation_manager import GenerationManager
from utils.format_convert import FormatConverter

def test_full_pipeline():
    """Test: generation -> metadata -> EXR conversion"""
    
    # Setup
    config_path = Path("config/models.json")
    defaults_path = Path("config/defaults.json")
    musubi_path = Path("vendor/musubi-tuner")
    output_root = Path("outputs")
    
    generator = MusubiGenerator(config_path, musubi_path, defaults_path)
    manager = GenerationManager(output_root)
    
    # Create generation directory
    gen_dir = manager.create_generation_dir()
    paths = manager.get_output_paths(gen_dir)
    
    print(f"Generation directory: {gen_dir}")
    
    # Test 1: Simple generation (no control image)
    print("\n=== Test 1: Simple generation ===")
    
    # Generation parameters
    prompt = "A noir film style image. An apple on a wooden table."
    model = QwenModel.IMAGE
    seed = 25738129
    image_size = (1024, 1024)
    lora = "noir_qwen"
    lora_multiplier = 1.7
    infer_steps = 20
    guidance_scale = 2.0  # Lower for more natural look
    blocks_to_swap = 10
    negative_prompt = None  # Will use default from config
    flow_shift = 2.1  # Your preferred value for natural results
    
    try:
        generator.generate(
            prompt=prompt,
            output_path=paths["generated_png"],
            model=model,
            image_size=image_size,
            seed=seed,
            lora=lora,
            lora_multiplier=lora_multiplier,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            blocks_to_swap=blocks_to_swap,
            negative_prompt=negative_prompt,
            flow_shift=flow_shift
        )
        print(f"✓ Generated: {paths['generated_png']}")
        
        # Save metadata
        manager.save_metadata(
            gen_dir=gen_dir,
            prompt=prompt,
            enhanced_prompt=prompt,  # no ollama enhancement yet
            model=model.value,
            seed=seed,
            image_size=image_size,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            lora=lora,
            lora_multiplier=lora_multiplier,
            negative_prompt=negative_prompt or generator.defaults.get("negative_prompt", ""),
            flow_shift=flow_shift
        )
        print(f"✓ Metadata saved: {paths['metadata']}")
        
        # Convert to EXR
        FormatConverter.png_to_exr_32bit(
            paths["generated_png"],
            paths["generated_exr"],
            linear=True
        )
        print(f"✓ EXR created: {paths['generated_exr']}")
        
        # Verify EXR file exists and has size
        exr_size = paths["generated_exr"].stat().st_size
        print(f"✓ EXR file size: {exr_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
    
    print("\n=== All tests passed ===")

if __name__ == "__main__":
    test_full_pipeline()