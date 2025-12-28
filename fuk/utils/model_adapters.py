# core/model_adapters.py
"""
Adapters for loading Qwen and Wan models into diffusers pipelines
Bridges the gap between model checkpoints and FUK pipeline expectations
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer, CLIPModel, CLIPTokenizer


class ModelLoader:
    """Load models with proper memory management"""
    
    @staticmethod
    def load_vae(
        vae_path: str,
        dtype: torch.dtype = torch.float16,
        enable_tiling: bool = True
    ) -> AutoencoderKL:
        """Load VAE with memory optimizations"""
        
        print(f"Loading VAE from {vae_path}")
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )
        
        if enable_tiling:
            vae.enable_tiling()
        
        return vae
    
    @staticmethod
    def load_text_encoder_t5(
        t5_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ) -> tuple:
        """Load T5 text encoder and tokenizer"""
        
        print(f"Loading T5 from {t5_path}")
        tokenizer = T5Tokenizer.from_pretrained(t5_path)
        text_encoder = T5EncoderModel.from_pretrained(
            t5_path,
            torch_dtype=dtype
        )
        
        text_encoder.to(device)
        
        return text_encoder, tokenizer
    
    @staticmethod
    def load_clip(
        clip_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ) -> tuple:
        """Load CLIP vision encoder and tokenizer"""
        
        print(f"Loading CLIP from {clip_path}")
        tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        clip_model = CLIPModel.from_pretrained(
            clip_path,
            torch_dtype=dtype
        )
        
        clip_model.to(device)
        
        return clip_model, tokenizer


class QwenModelAdapter:
    """
    Adapter for Qwen models
    
    Qwen uses Flux-style architecture:
    - DiT (Diffusion Transformer)
    - VAE (Variational Autoencoder)
    - Text Encoder
    
    This adapter handles loading these components and creating a usable pipeline
    """
    
    @staticmethod
    def load_from_config(
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        enable_fp8: bool = False
    ) -> Dict[str, Any]:
        """
        Load Qwen model components
        
        Returns dict with loaded components that can be used to construct a pipeline
        """
        
        components = {}
        
        # Load VAE
        components["vae"] = ModelLoader.load_vae(vae_path, dtype)
        
        # Load text encoder
        # Note: Qwen might use T5, CLIP, or custom encoder
        # Adjust based on your actual model
        try:
            text_encoder, tokenizer = ModelLoader.load_text_encoder_t5(
                text_encoder_path, 
                dtype, 
                device
            )
            components["text_encoder"] = text_encoder
            components["tokenizer"] = tokenizer
        except:
            print("Warning: Could not load as T5, trying CLIP...")
            clip_model, tokenizer = ModelLoader.load_clip(
                text_encoder_path,
                dtype,
                device
            )
            components["text_encoder"] = clip_model
            components["tokenizer"] = tokenizer
        
        # Load DiT (transformer)
        # This is the tricky part - need to know Qwen's actual model class
        # Options:
        # 1. If Qwen provides diffusers-compatible checkpoint:
        #    components["transformer"] = FluxTransformer2DModel.from_pretrained(dit_path)
        # 2. If using custom checkpoint:
        #    components["transformer"] = load_custom_dit(dit_path)
        
        print("Note: DiT loading needs model-specific implementation")
        print(f"DiT path: {dit_path}")
        
        return components
    
    @staticmethod
    def create_pipeline(components: Dict[str, Any], device: str = "cuda"):
        """
        Create a usable pipeline from loaded components
        
        This is where you'd construct the actual inference pipeline
        For now, this is a placeholder showing the structure
        """
        
        # Example structure (adapt to your actual Qwen pipeline class):
        # from qwen_models import QwenPipeline
        # 
        # pipeline = QwenPipeline(
        #     vae=components["vae"],
        #     text_encoder=components["text_encoder"],
        #     tokenizer=components["tokenizer"],
        #     transformer=components["transformer"],
        # )
        # 
        # pipeline.to(device)
        # return pipeline
        
        print("Pipeline creation needs model-specific implementation")
        return None


class WanModelAdapter:
    """
    Adapter for Wan video models
    
    Wan models typically include:
    - DiT (video diffusion transformer)
    - VAE (shared with images or video-specific)
    - T5 text encoder
    - CLIP (for I2V tasks)
    """
    
    @staticmethod
    def load_from_config(
        dit_path: str,
        vae_path: str,
        t5_path: str,
        clip_path: Optional[str] = None,
        shared_vae: Optional[AutoencoderKL] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        enable_fp8: bool = False
    ) -> Dict[str, Any]:
        """Load Wan model components"""
        
        components = {}
        
        # Use shared VAE if provided, otherwise load new one
        if shared_vae is not None:
            components["vae"] = shared_vae
            print("Using shared VAE from Qwen pipeline")
        else:
            components["vae"] = ModelLoader.load_vae(vae_path, dtype)
        
        # Load T5 text encoder
        text_encoder, tokenizer = ModelLoader.load_text_encoder_t5(
            t5_path,
            dtype,
            device
        )
        components["text_encoder"] = text_encoder
        components["tokenizer"] = tokenizer
        
        # Load CLIP if needed (for I2V tasks)
        if clip_path:
            clip_model, clip_tokenizer = ModelLoader.load_clip(
                clip_path,
                dtype,
                device
            )
            components["clip"] = clip_model
            components["clip_tokenizer"] = clip_tokenizer
        
        # Load DiT
        print("Note: Video DiT loading needs model-specific implementation")
        print(f"DiT path: {dit_path}")
        
        return components
    
    @staticmethod
    def create_pipeline(components: Dict[str, Any], task: str, device: str = "cuda"):
        """Create task-specific video pipeline"""
        
        # Example structure:
        # from wan_models import WanVideoPipeline
        # 
        # pipeline = WanVideoPipeline(
        #     vae=components["vae"],
        #     text_encoder=components["text_encoder"],
        #     tokenizer=components["tokenizer"],
        #     clip=components.get("clip"),
        #     transformer=components["transformer"],
        #     task=task
        # )
        # 
        # pipeline.to(device)
        # return pipeline
        
        print(f"Video pipeline creation for task '{task}' needs implementation")
        return None


class MusubiCompatibilityLayer:
    """
    Fallback layer that uses musubi-tuner when direct loading isn't available
    
    This lets you incrementally migrate:
    1. Start with musubi for actual generation
    2. Add direct loading for models you figure out
    3. Eventually replace all musubi calls
    """
    
    def __init__(self, musubi_path: Path):
        self.musubi_path = Path(musubi_path)
    
    def generate_image_subprocess(
        self,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        prompt: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Fallback to musubi subprocess when direct loading not ready
        """
        import subprocess
        
        # Build musubi command
        cmd = [
            "python",
            str(self.musubi_path / "qwen_image_generate_image.py"),
            "--dit", dit_path,
            "--vae", vae_path,
            "--text_encoder", text_encoder_path,
            "--prompt", prompt,
            "--save_path", str(output_path.parent),
            # Add other kwargs...
        ]
        
        print("Using musubi subprocess fallback")
        subprocess.run(cmd, check=True)
        
        return output_path
    
    def generate_video_subprocess(
        self,
        dit_path: str,
        vae_path: str,
        t5_path: str,
        task: str,
        prompt: str,
        output_path: Path,
        **kwargs
    ) -> Path:
        """Fallback to musubi for video"""
        import subprocess
        
        cmd = [
            "python",
            str(self.musubi_path / "wan_generate_video.py"),
            "--task", task,
            "--dit", dit_path,
            "--vae", vae_path,
            "--t5", t5_path,
            "--prompt", prompt,
            "--save_path", str(output_path),
            # Add other kwargs...
        ]
        
        print("Using musubi subprocess fallback for video")
        subprocess.run(cmd, check=True)
        
        return output_path


# Hybrid approach during migration
class HybridPipeline:
    """
    Hybrid pipeline that uses direct loading where possible,
    falls back to musubi where needed
    
    This gives you the best of both worlds during migration:
    - Use direct diffusers for latent management
    - Use musubi subprocess for generation if needed
    - Incrementally replace musubi calls as you figure out models
    """
    
    def __init__(
        self,
        qwen_config: Dict[str, str],
        wan_config: Dict[str, str],
        musubi_path: Optional[Path] = None,
        use_direct: bool = False
    ):
        self.use_direct = use_direct
        self.musubi = MusubiCompatibilityLayer(musubi_path) if musubi_path else None
        
        # Load components we can
        try:
            # Try loading VAE directly (usually works)
            self.vae = ModelLoader.load_vae(qwen_config["vae"])
            print("✓ VAE loaded directly")
        except Exception as e:
            print(f"✗ VAE direct loading failed: {e}")
            self.vae = None
        
        # Store configs for fallback
        self.qwen_config = qwen_config
        self.wan_config = wan_config
    
    def generate_image(
        self,
        prompt: str,
        output_path: Path,
        return_latent: bool = True,
        **kwargs
    ):
        """
        Generate image using best available method
        """
        
        if self.use_direct and self.vae:
            # Use direct pipeline
            print("Using direct diffusers pipeline")
            # ... direct generation code
        else:
            # Use musubi subprocess
            print("Using musubi subprocess")
            result = self.musubi.generate_image_subprocess(
                dit_path=self.qwen_config["dit"],
                vae_path=self.qwen_config["vae"],
                text_encoder_path=self.qwen_config["text_encoder"],
                prompt=prompt,
                output_path=output_path,
                **kwargs
            )
            
            # If we want latents, encode the output
            if return_latent and self.vae:
                from PIL import Image
                img = Image.open(result)
                latent = self.encode_image(img)
                return result, latent
            
            return result, None
    
    def encode_image(self, image) -> torch.Tensor:
        """Encode image using loaded VAE"""
        if self.vae is None:
            raise RuntimeError("VAE not loaded")
        
        # Convert PIL to tensor
        import numpy as np
        from PIL import Image
        
        if isinstance(image, Path):
            image = Image.open(image)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.vae.device, dtype=self.vae.dtype)
        img_tensor = 2.0 * img_tensor - 1.0
        
        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    def decode_latent(self, latent: torch.Tensor):
        """Decode latent using loaded VAE"""
        if self.vae is None:
            raise RuntimeError("VAE not loaded")
        
        latent = latent.to(self.vae.device, dtype=self.vae.dtype)
        
        with torch.no_grad():
            image = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        
        from PIL import Image
        return Image.fromarray((image * 255).astype(np.uint8))


# Example usage showing migration path
if __name__ == "__main__":
    import json
    
    # Load your existing config
    with open("config/models.json") as f:
        config = json.load(f)
    
    # Start with hybrid approach
    hybrid = HybridPipeline(
        qwen_config=config["models"]["qwen_image"],
        wan_config=config["wan_models"]["i2v-14B"],
        musubi_path=Path("vendor/musubi-tuner"),
        use_direct=False  # Use musubi for generation
    )
    
    # Generate using musubi subprocess
    png_path, latent = hybrid.generate_image(
        prompt="A noir style apple",
        output_path=Path("test.png"),
        return_latent=True,  # Get latent even though using subprocess
        image_size=(1024, 1024),
        seed=42
    )
    
    print(f"Generated: {png_path}")
    if latent is not None:
        print(f"Latent shape: {latent.shape}")
        
        # Save latent
        from core.fuk_pipeline import LatentStore
        LatentStore.save(latent, Path("test_latent.safetensors"))
        
        # Can now use this latent for video generation
        print("Latent ready for video workflow")
