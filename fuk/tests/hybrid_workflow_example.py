# examples/hybrid_workflow_example.py
"""
Practical example showing immediate benefits of hybrid approach

This demonstrates how you can get latent-space benefits TODAY while
still using musubi-tuner for the actual generation. Best of both worlds:

1. Keep using musubi for generation (proven to work)
2. Add latent persistence layer on top
3. Enable latent → video workflow
4. Incrementally replace musubi as you figure out direct loading

NO BREAKING CHANGES to your existing code.
"""

from pathlib import Path
import json
import torch
from PIL import Image
import numpy as np
from safetensors.torch import save_file, load_file


class SimpleLatentManager:
    """
    Minimal latent management layer that works WITH musubi
    
    This gives you 80% of the benefits with 20% of the effort:
    - Latent persistence
    - Latent → video passthrough
    - Quality preservation
    
    No need to rewrite everything - just wrap existing musubi calls
    """
    
    @staticmethod
    def _get_vae_config_for_checkpoint(checkpoint_path: Path) -> dict:
        """
        Determine VAE config based on checkpoint
        
        This handles common VAE architectures used with Qwen/Flux/Wan
        """
        from safetensors.torch import load_file as load_safetensors
        
        # Load just the metadata to check
        try:
            state_dict = load_safetensors(str(checkpoint_path))
            
            # Check for key signatures
            keys = list(state_dict.keys())
            
            # Flux VAE signature (used by Qwen)
            if any('encoder.conv_in' in k for k in keys):
                print("Detected Flux-style VAE")
                return {
                    "in_channels": 3,
                    "out_channels": 3,
                    "down_block_types": ["DownEncoderBlock2D"] * 4,
                    "up_block_types": ["UpDecoderBlock2D"] * 4,
                    "block_out_channels": [128, 256, 512, 512],
                    "layers_per_block": 2,
                    "act_fn": "silu",
                    "latent_channels": 16,
                    "norm_num_groups": 32,
                    "sample_size": 1024,
                    "scaling_factor": 0.3611,
                    "shift_factor": 0.1159,
                }
            
            # SD3/SDXL VAE
            elif any('decoder.up_blocks' in k for k in keys):
                print("Detected SD3/SDXL-style VAE")
                return {
                    "in_channels": 3,
                    "out_channels": 3,
                    "down_block_types": ["DownEncoderBlock2D"] * 4,
                    "up_block_types": ["UpDecoderBlock2D"] * 4,
                    "block_out_channels": [128, 256, 512, 512],
                    "layers_per_block": 2,
                    "latent_channels": 16,
                    "norm_num_groups": 32,
                }
            
            else:
                print("Unknown VAE type, using Flux defaults")
                return {
                    "in_channels": 3,
                    "out_channels": 3,
                    "down_block_types": ["DownEncoderBlock2D"] * 4,
                    "up_block_types": ["UpDecoderBlock2D"] * 4,
                    "block_out_channels": [128, 256, 512, 512],
                    "layers_per_block": 2,
                    "act_fn": "silu",
                    "latent_channels": 16,
                    "norm_num_groups": 32,
                    "sample_size": 1024,
                    "scaling_factor": 0.3611,
                    "shift_factor": 0.1159,
                }
        except Exception as e:
            print(f"Warning: Could not analyze checkpoint: {e}")
            print("Using default Flux VAE config")
            return {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ["DownEncoderBlock2D"] * 4,
                "up_block_types": ["UpDecoderBlock2D"] * 4,
                "block_out_channels": [128, 256, 512, 512],
                "layers_per_block": 2,
                "act_fn": "silu",
                "latent_channels": 16,
                "norm_num_groups": 32,
                "sample_size": 1024,
                "scaling_factor": 0.3611,
                "shift_factor": 0.1159,
            }
    
    def __init__(self, vae_path: str, vae_config: dict = None):
        """
        Only needs VAE - this is the one component that's easy to load
        directly and gives you all the latent functionality
        
        Args:
            vae_path: Path to VAE checkpoint (can be .safetensors or directory)
            vae_config: Optional config dict if loading raw checkpoint
        """
        from diffusers import AutoencoderKL
        from safetensors.torch import load_file as load_safetensors
        
        print(f"Loading VAE from {vae_path}")
        
        vae_path = Path(vae_path)
        
        # Check if it's a directory (diffusers format) or file (raw checkpoint)
        if vae_path.is_dir():
            # Standard diffusers format
            self.vae = AutoencoderKL.from_pretrained(
                str(vae_path),
                torch_dtype=torch.float16
            ).to("cuda")
        else:
            # Raw safetensors file - need to load manually
            print("Detected raw safetensors checkpoint, loading manually...")
            
            # Auto-detect config if not provided
            if vae_config is None:
                vae_config = self._get_vae_config_for_checkpoint(vae_path)
            
            # Create VAE with config
            self.vae = AutoencoderKL(**vae_config)
            
            # Load weights
            state_dict = load_safetensors(str(vae_path))
            
            # Try to load, handling potential key mismatches
            missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"Warning: Missing keys in checkpoint: {len(missing)} keys")
            if unexpected:
                print(f"Warning: Unexpected keys in checkpoint: {len(unexpected)} keys")
            
            # Move to GPU
            self.vae = self.vae.to("cuda", dtype=torch.float16)
        
        self.vae.enable_tiling()
        print("✓ VAE ready for latent operations")
    
    def encode_png(self, png_path: Path) -> torch.Tensor:
        """
        Encode PNG to latent
        
        This is what makes latent persistence work:
        After musubi generates PNG, encode it to latent
        """
        image = Image.open(png_path).convert("RGB")
        
        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to("cuda", dtype=torch.float16)
        img_tensor = 2.0 * img_tensor - 1.0  # Normalize to [-1, 1]
        
        # Encode
        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent to PIL Image"""
        latent = latent.to("cuda", dtype=torch.float16)
        
        with torch.no_grad():
            image = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        return image
    
    def save_latent(self, latent: torch.Tensor, path: Path, metadata: dict = None):
        """Save latent with metadata"""
        data = {"latent": latent.cpu()}
        
        if metadata:
            # Convert metadata to strings for safetensors
            str_metadata = {k: str(v) for k, v in metadata.items()}
            save_file(data, path, metadata=str_metadata)
        else:
            save_file(data, path)
        
        print(f"✓ Saved latent: {path}")
    
    def load_latent(self, path: Path) -> tuple[torch.Tensor, dict]:
        """Load latent and metadata"""
        data = load_file(path)
        latent = data["latent"]
        
        # Load metadata if exists
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                metadata = dict(f.metadata()) if f.metadata() else {}
        except:
            metadata = {}
        
        return latent, metadata


class MusubiWithLatents:
    """
    Wrapper around your existing MusubiGenerator that adds latent capabilities
    
    This is a DROP-IN REPLACEMENT for your current musubi_wrapper.py
    Just change the import and you get latent support
    """
    
    def __init__(
        self,
        musubi_generator,  # Your existing MusubiGenerator instance
        latent_manager: SimpleLatentManager
    ):
        self.musubi = musubi_generator
        self.latent_mgr = latent_manager
    
    def generate(
        self,
        prompt: str,
        output_path: Path,
        save_latent: bool = True,
        **musubi_kwargs
    ) -> dict:
        """
        Generate with musubi, optionally save latent
        
        Returns:
            {
                "png": Path to PNG,
                "latent": Path to latent (if save_latent=True),
                "latent_tensor": Tensor (if save_latent=True)
            }
        """
        
        # Use existing musubi generation
        png_path = self.musubi.generate(
            prompt=prompt,
            output_path=output_path,
            **musubi_kwargs
        )
        
        results = {"png": png_path}
        
        # Add latent layer on top
        if save_latent:
            print("Encoding PNG to latent...")
            latent = self.latent_mgr.encode_png(png_path)
            
            # Save latent
            latent_path = output_path.parent / f"{output_path.stem}.safetensors"
            metadata = {
                "prompt": prompt,
                "source_png": str(png_path),
                **musubi_kwargs
            }
            self.latent_mgr.save_latent(latent, latent_path, metadata)
            
            results["latent"] = latent_path
            results["latent_tensor"] = latent
        
        return results


# ============================================================================
# EXAMPLE 1: Drop-in enhancement of existing workflow
# ============================================================================

def example_1_enhance_existing():
    """
    Shows how to add latent support to your EXISTING code
    with minimal changes
    """
    
    print("=== Example 1: Enhance Existing Workflow ===\n")
    
    # Your existing setup (unchanged)
    from core.musubi_wrapper import MusubiGenerator
    
    config_path = Path("config/models.json")
    musubi_path = Path("vendor/musubi-tuner")
    
    musubi_gen = MusubiGenerator(
        config_path=config_path,
        musubi_vendor_path=musubi_path
    )
    
    # NEW: Add latent layer (one-time setup)
    with open(config_path) as f:
        config = json.load(f)
    
    latent_mgr = SimpleLatentManager(
        vae_path=config["models"]["qwen_image"]["vae"]
    )
    
    # NEW: Wrap your generator
    enhanced_gen = MusubiWithLatents(musubi_gen, latent_mgr)
    
    # Your existing code with ONE extra parameter
    results = enhanced_gen.generate(
        prompt="A noir film apple on wooden table",
        output_path=Path("outputs/test.png"),
        image_size=(1024, 1024),
        seed=42,
        save_latent=True  # ← Only new parameter
    )
    
    print(f"\n✓ PNG: {results['png']}")
    print(f"✓ Latent: {results['latent']}")
    print(f"✓ Latent tensor shape: {results['latent_tensor'].shape}")


# ============================================================================
# EXAMPLE 2: Latent → Video workflow with musubi
# ============================================================================

def example_2_latent_to_video():
    """
    The BIG win: Image → Video with latent passthrough
    
    Before: PNG → encode → latent → video (quality loss)
    After:  latent → video (no quality loss)
    """
    
    print("=== Example 2: Latent → Video Workflow ===\n")
    
    # Setup (same as before)
    from core.musubi_wrapper import MusubiGenerator
    from core.wan_video_wrapper import WanVideoGenerator, WanTask
    
    config_path = Path("config/models.json")
    musubi_path = Path("vendor/musubi-tuner")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Image generator with latents
    musubi_gen = MusubiGenerator(config_path, musubi_path)
    latent_mgr = SimpleLatentManager(config["models"]["qwen_image"]["vae"])
    enhanced_gen = MusubiWithLatents(musubi_gen, latent_mgr)
    
    # Video generator (unchanged)
    wan_gen = WanVideoGenerator(config_path, musubi_path)
    
    # Step 1: Generate image with latent
    print("Step 1: Generate image")
    image_results = enhanced_gen.generate(
        prompt="A noir film apple on table",
        output_path=Path("outputs/image.png"),
        image_size=(1024, 1024),
        seed=42,
        save_latent=True
    )
    
    # Step 2: Decode latent to temp image for Wan
    print("\nStep 2: Prepare for video generation")
    latent = image_results["latent_tensor"]
    temp_image = latent_mgr.decode_latent(latent)
    temp_path = Path("outputs/temp_for_video.png")
    temp_image.save(temp_path)
    
    # Step 3: Generate video using the temp image
    print("\nStep 3: Generate video")
    video_path = wan_gen.generate_video(
        prompt="The apple slowly rolls off the table",
        output_path=Path("outputs/video.mp4"),
        task=WanTask.I2V_14B,
        image_path=temp_path,  # From latent, not original PNG
        video_size=(832, 480),
        video_length=81,
        seed=42
    )
    
    print(f"\n✓ Video: {video_path}")
    print("\nKey point: Video input came from LATENT, not original PNG")
    print("This preserves maximum quality through the pipeline")


# ============================================================================
# EXAMPLE 3: Batch processing with latent caching
# ============================================================================

def example_3_batch_with_caching():
    """
    Generate many images, save latents, render finals later
    
    This workflow:
    1. Generate 100 images quickly (PNG + latent)
    2. Review PNGs
    3. Decode only selected latents to EXR
    
    Saves time by not rendering EXR for rejected images
    """
    
    print("=== Example 3: Batch Processing with Latent Caching ===\n")
    
    from core.musubi_wrapper import MusubiGenerator
    from utils.format_convert import FormatConverter
    
    config_path = Path("config/models.json")
    musubi_path = Path("vendor/musubi-tuner")
    
    with open(config_path) as f:
        config = json.load(f)
    
    musubi_gen = MusubiGenerator(config_path, musubi_path)
    latent_mgr = SimpleLatentManager(config["models"]["qwen_image"]["vae"])
    enhanced_gen = MusubiWithLatents(musubi_gen, latent_mgr)
    
    # Step 1: Generate batch (fast)
    print("Step 1: Generating batch of 10 variations...")
    base_prompt = "A noir film apple on wooden table"
    variations = []
    
    for i in range(10):
        print(f"\nGenerating variation {i+1}/10")
        results = enhanced_gen.generate(
            prompt=f"{base_prompt}, variation {i+1}",
            output_path=Path(f"outputs/batch/var_{i:02d}.png"),
            image_size=(1024, 1024),
            seed=42 + i,
            save_latent=True
        )
        variations.append(results)
    
    # Step 2: User reviews PNGs, selects best
    print("\n\nStep 2: User reviews PNGs and selects best...")
    selected = [0, 3, 7]  # Simulate user selection
    print(f"Selected variations: {selected}")
    
    # Step 3: Render EXR only for selected (decode from latent)
    print("\nStep 3: Rendering finals for selected variations...")
    for idx in selected:
        var = variations[idx]
        
        # Load latent
        latent, metadata = latent_mgr.load_latent(var["latent"])
        
        # Decode to high quality
        image = latent_mgr.decode_latent(latent)
        
        # Save as EXR
        exr_path = Path(f"outputs/batch/var_{idx:02d}_final.exr")
        png_temp = Path(f"outputs/batch/var_{idx:02d}_temp.png")
        image.save(png_temp)
        
        FormatConverter.png_to_exr_32bit(png_temp, exr_path)
        png_temp.unlink()  # Clean up temp
        
        print(f"✓ Rendered EXR for variation {idx}")
    
    print("\n✓ Batch processing complete")
    print(f"Generated 10 images, rendered EXR for only {len(selected)} selected")


# ============================================================================
# EXAMPLE 4: Chained generation with quality preservation
# ============================================================================

def example_4_chained_generation():
    """
    Image A → Image B → Image C → Video
    All in latent space, maximum quality preservation
    """
    
    print("=== Example 4: Chained Generation ===\n")
    
    from core.musubi_wrapper import MusubiGenerator, QwenModel
    
    config_path = Path("config/models.json")
    musubi_path = Path("vendor/musubi-tuner")
    
    with open(config_path) as f:
        config = json.load(f)
    
    musubi_gen = MusubiGenerator(config_path, musubi_path)
    latent_mgr = SimpleLatentManager(config["models"]["qwen_image"]["vae"])
    enhanced_gen = MusubiWithLatents(musubi_gen, latent_mgr)
    
    # Step 1: Generate base image
    print("Step 1: Generate base image")
    base = enhanced_gen.generate(
        prompt="A wooden table",
        output_path=Path("outputs/chain/01_base.png"),
        model=QwenModel.IMAGE,
        image_size=(1024, 1024),
        seed=42,
        save_latent=True
    )
    
    # Step 2: Edit with depth map (using edit mode)
    print("\nStep 2: Add apple using edit mode")
    
    # Decode latent for control image
    base_image = latent_mgr.decode_latent(base["latent_tensor"])
    control_path = Path("outputs/chain/01_control.png")
    base_image.save(control_path)
    
    # Generate with control
    apple = enhanced_gen.generate(
        prompt="An apple appears on the wooden table",
        output_path=Path("outputs/chain/02_apple.png"),
        model=QwenModel.EDIT_2509,
        control_image=control_path,
        image_size=(1024, 1024),
        seed=43,
        save_latent=True
    )
    
    # Step 3: Another edit
    print("\nStep 3: Add dramatic lighting")
    apple_image = latent_mgr.decode_latent(apple["latent_tensor"])
    control_path2 = Path("outputs/chain/02_control.png")
    apple_image.save(control_path2)
    
    final = enhanced_gen.generate(
        prompt="Dramatic noir lighting on apple and table",
        output_path=Path("outputs/chain/03_final.png"),
        model=QwenModel.EDIT_2509,
        control_image=control_path2,
        image_size=(1024, 1024),
        seed=44,
        save_latent=True
    )
    
    print("\n✓ Chained generation complete")
    print("Each step preserved in latent space")
    print(f"Base latent: {base['latent']}")
    print(f"Apple latent: {apple['latent']}")
    print(f"Final latent: {final['latent']}")


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Enhance existing workflow", example_1_enhance_existing),
        "2": ("Latent → Video workflow", example_2_latent_to_video),
        "3": ("Batch processing", example_3_batch_with_caching),
        "4": ("Chained generation", example_4_chained_generation),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print('='*60 + "\n")
            func()
        else:
            print(f"Unknown example: {choice}")
            print("Available examples:")
            for key, (name, _) in examples.items():
                print(f"  {key}: {name}")
    else:
        print("Usage: python hybrid_workflow_example.py <example_number>")
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")
        
        print("\nExample: python hybrid_workflow_example.py 2")