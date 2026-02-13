"""
Latent Manager for DiffSynth Backend

Handles saving, loading, and decoding latent tensors from the diffusion process.
This enables true lossless workflows: latent → VAE decode → EXR export.
"""

import torch
from pathlib import Path
from typing import Optional, Callable
import numpy as np


class LatentManager:
    """
    Manages latent tensor operations for DiffSynth pipelines.
    
    Provides callbacks for saving latents during generation and methods
    for decoding latents to images/videos without quality loss.
    """
    
    def __init__(self):
        self.saved_latent_path = None
    
    def create_save_callback(self, latent_path: Path) -> Callable:
        """
        Create a callback function that saves the final latent tensor.
        
        This callback is passed to DiffSynth pipelines via callback_on_step_end
        and captures the latent tensor at the final diffusion step.
        
        Args:
            latent_path: Path where latent tensor will be saved (.pt file)
            
        Returns:
            Callback function compatible with DiffSynth pipelines
        """
        latent_path = Path(latent_path)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            
            # Save on final step
            if step == pipe.num_inference_steps - 1:
                torch.save(latents.cpu(), latent_path)
                self.saved_latent_path = latent_path
                print(f"✓ Saved latent: {latent_path.name}")
            
            return callback_kwargs
        
        return callback
    
    def create_video_save_callback(self, latent_path: Path) -> Callable:
        """
        Create a callback for video generation that saves latent sequence.
        
        Video latents have shape [batch, channels, frames, height, width].
        We rearrange to [frames, channels, height, width] for easier frame access.
        
        Args:
            latent_path: Path where latent sequence will be saved
            
        Returns:
            Callback function for video pipelines
        """
        latent_path = Path(latent_path)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        
        def callback(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            
            # Save on final step
            if step == pipe.num_inference_steps - 1:
                # Rearrange from [B, C, F, H, W] to [F, C, H, W]
                latents_frames = latents.squeeze(0).permute(1, 0, 2, 3)
                torch.save(latents_frames.cpu(), latent_path)
                self.saved_latent_path = latent_path
                print(f"✓ Saved video latent: {latent_path.name} ({latents_frames.shape[0]} frames)")
            
            return callback_kwargs
        
        return callback
    
    def load_latent(self, latent_path: Path, device: str = 'cuda') -> torch.Tensor:
        """
        Load a saved latent tensor.
        
        Args:
            latent_path: Path to saved .pt file
            device: Device to load tensor to
            
        Returns:
            Loaded latent tensor
        """
        latent_path = Path(latent_path)
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        latents = torch.load(latent_path, map_location=device)
        return latents
    
    def decode_image_latent(
        self, 
        latents: torch.Tensor, 
        vae,
        to_numpy: bool = True
    ) -> np.ndarray:
        """
        Decode image latent through VAE.
        
        Args:
            latents: Latent tensor from diffusion process
            vae: VAE decoder (from pipeline)
            to_numpy: Convert to numpy float32 array
            
        Returns:
            Decoded image as numpy array (float32, scene-linear) or tensor
        """
        with torch.no_grad():
            # Ensure latents are on correct device
            if not latents.is_cuda and hasattr(vae, 'device'):
                latents = latents.to(vae.device)
            
            # Decode through VAE
            decoded = vae.decode(latents)
            
            if to_numpy:
                # Convert to numpy float32 for EXR export
                # Shape: [C, H, W] or [B, C, H, W]
                array = decoded.cpu().numpy().astype(np.float32)
                
                # Handle batch dimension
                if array.ndim == 4 and array.shape[0] == 1:
                    array = array.squeeze(0)
                
                # Convert from [C, H, W] to [H, W, C] for image processing
                if array.ndim == 3:
                    array = np.transpose(array, (1, 2, 0))
                
                return array
            
            return decoded
    
    def decode_video_latent_sequence(
        self,
        latents: torch.Tensor,
        vae,
        batch_size: int = 8
    ) -> list[np.ndarray]:
        """
        Decode video latent sequence frame by frame.
        
        Args:
            latents: Video latent tensor [frames, channels, height, width]
            vae: VAE decoder
            batch_size: Number of frames to decode at once (memory management)
            
        Returns:
            List of decoded frames as numpy arrays
        """
        frames = []
        num_frames = latents.shape[0]
        
        print(f"Decoding {num_frames} frames (batch size: {batch_size})...")
        
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                batch = latents[i:i+batch_size]
                
                # Ensure correct device
                if not batch.is_cuda and hasattr(vae, 'device'):
                    batch = batch.to(vae.device)
                
                # Decode batch
                decoded_batch = vae.decode(batch)
                
                # Convert each frame to numpy
                for frame_tensor in decoded_batch:
                    frame_array = frame_tensor.cpu().numpy().astype(np.float32)
                    # [C, H, W] -> [H, W, C]
                    frame_array = np.transpose(frame_array, (1, 2, 0))
                    frames.append(frame_array)
                
                print(f"  Decoded frames {i+1}-{min(i+batch_size, num_frames)}/{num_frames}")
        
        return frames
    
    def decode_to_exr(
        self,
        latent_path: Path,
        vae,
        output_exr: Path,
        layer_name: str = 'beauty'
    ) -> Path:
        """
        Decode latent directly to EXR file, bypassing PNG/MP4 compression.
        
        This is the true lossless path: latent → VAE → EXR
        
        Args:
            latent_path: Path to saved latent .pt file
            vae: VAE decoder from pipeline
            output_exr: Output EXR file path
            layer_name: EXR layer name
            
        Returns:
            Path to created EXR file
        """
        from exr_exporter import EXRExporter
        
        # Load latent
        latents = self.load_latent(latent_path)
        
        # Decode to numpy array
        image_array = self.decode_image_latent(latents, vae, to_numpy=True)
        
        # Write to EXR
        exporter = EXRExporter()
        output_exr = Path(output_exr)
        output_exr.parent.mkdir(parents=True, exist_ok=True)
        
        exporter.export_single_layer(
            output_path=output_exr,
            layer_name=layer_name,
            data=image_array
        )
        
        print(f"✓ Exported latent to EXR: {output_exr}")
        return output_exr


if __name__ == "__main__":
    # Simple test
    manager = LatentManager()
    print("LatentManager initialized successfully")
    
    # Test callback creation
    callback = manager.create_save_callback(Path("/tmp/test.latent.pt"))
    print("✓ Callback created")