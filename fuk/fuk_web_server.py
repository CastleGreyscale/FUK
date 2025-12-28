#!/usr/bin/env python3
"""
FUK Web Server - FastAPI backend for the FUK generation pipeline

Provides REST API endpoints for:
- Image generation (Qwen via musubi)
- Video generation (Wan via musubi)
- Progress monitoring (Server-Sent Events)
- File serving
- Generation history
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import asyncio
import uuid
from datetime import datetime
from enum import Enum
import sys

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try to import from different possible locations
try:
    # Option 1: Files in same directory
    from musubi_wrapper import MusubiGenerator, QwenModel
    from wan_video_wrapper import WanVideoGenerator, WanTask
    from generation_manager import GenerationManager
    from video_generation_manager import VideoGenerationManager
    from format_convert import FormatConverter
except ImportError:
    try:
        # Option 2: Files in core/ and utils/ subdirectories
        sys.path.insert(0, str(current_dir.parent))
        from core.musubi_wrapper import MusubiGenerator, QwenModel
        from core.wan_video_wrapper import WanVideoGenerator, WanTask
        from core.generation_manager import GenerationManager
        from core.video_generation_manager import VideoGenerationManager
        from utils.format_convert import FormatConverter
    except ImportError as e:
        print("\n" + "="*60)
        print("ERROR: Cannot import FUK modules")
        print("="*60)
        print("\nMake sure fuk_web_server.py is in the same directory as:")
        print("  - musubi_wrapper.py")
        print("  - wan_video_wrapper.py")
        print("  - generation_manager.py")
        print("  - video_generation_manager.py")
        print("  - format_convert.py")
        print("\nOr adjust PYTHONPATH to include the directory containing these files.")
        print(f"\nOriginal error: {e}")
        print("="*60 + "\n")
        sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Determine base directory (where this script is located)
BASE_DIR = Path(__file__).parent

# Look for config in multiple possible locations
CONFIG_PATHS = [
    BASE_DIR / "config" / "models.json",
    BASE_DIR.parent / "config" / "models.json",
    Path("config/models.json"),
]

DEFAULTS_PATHS = [
    BASE_DIR / "config" / "defaults.json",
    BASE_DIR.parent / "config" / "defaults.json",
    Path("config/defaults.json"),
]

MUSUBI_PATHS = [
    BASE_DIR / "vendor" / "musubi-tuner",
    BASE_DIR.parent / "vendor" / "musubi-tuner",
    Path("vendor/musubi-tuner"),
]

OUTPUT_ROOTS = [
    BASE_DIR / "outputs",
    BASE_DIR.parent / "outputs",
    Path("outputs"),
]

# Find first existing path for each
def find_path(paths, name):
    for p in paths:
        if p.exists():
            print(f"✓ Found {name}: {p}")
            return p
    print(f"✗ {name} not found in:")
    for p in paths:
        print(f"  - {p}")
    raise FileNotFoundError(f"{name} not found")

try:
    CONFIG_PATH = find_path(CONFIG_PATHS, "models.json")
    DEFAULTS_PATH = find_path(DEFAULTS_PATHS, "defaults.json")
    MUSUBI_PATH = find_path(MUSUBI_PATHS, "musubi-tuner")
    OUTPUT_ROOT = OUTPUT_ROOTS[0]  # Use first option, create if needed
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_ROOT}")
except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("\nPlease ensure your project has the following structure:")
    print("  config/models.json")
    print("  config/defaults.json")
    print("  vendor/musubi-tuner/")
    sys.exit(1)

# Ensure paths exist
OUTPUT_ROOT.mkdir(exist_ok=True)
(OUTPUT_ROOT / "image").mkdir(exist_ok=True)
(OUTPUT_ROOT / "video").mkdir(exist_ok=True)

# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="FUK Generation API", version="1.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve outputs directory as static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

# Active generations tracking
active_generations: Dict[str, Dict[str, Any]] = {}

# Load defaults
with open(DEFAULTS_PATH) as f:
    DEFAULTS = json.load(f)

# Initialize generators
image_generator = MusubiGenerator(CONFIG_PATH, MUSUBI_PATH, DEFAULTS_PATH)
video_generator = WanVideoGenerator(CONFIG_PATH, MUSUBI_PATH, DEFAULTS_PATH)
image_manager = GenerationManager(OUTPUT_ROOT / "image")
video_manager = VideoGenerationManager(OUTPUT_ROOT / "video")

# ============================================================================
# Request/Response Models
# ============================================================================

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "qwen_image"
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 2.1
    flow_shift: Optional[float] = 2.1
    seed: Optional[int] = None
    lora: Optional[str] = None
    lora_multiplier: float = 1.0
    blocks_to_swap: int = 10
    control_image_path: Optional[str] = None
    output_format: str = "png"  # png, exr, both
    
class VideoGenerationRequest(BaseModel):
    prompt: str
    task: str = "i2v-14B"
    negative_prompt: Optional[str] = None
    width: int = 832
    height: int = 480
    video_length: int = 81
    steps: int = 20
    guidance_scale: float = 5.0
    flow_shift: float = 5.0
    seed: Optional[int] = None
    image_path: Optional[str] = None
    end_image_path: Optional[str] = None
    control_path: Optional[str] = None
    lora: Optional[str] = None
    lora_multiplier: float = 1.0
    blocks_to_swap: int = 15
    fp8: bool = True
    fp8_scaled: bool = False
    fp8_t5: bool = True
    export_exr: bool = False
    
class GenerationResponse(BaseModel):
    generation_id: str
    status: str
    message: str
    
class GenerationStatus(BaseModel):
    generation_id: str
    status: str  # queued, running, complete, failed
    progress: float  # 0.0 to 1.0
    phase: str
    outputs: Dict[str, str]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# Progress Tracking
# ============================================================================

class ProgressCallback:
    """Callback for tracking generation progress"""
    
    def __init__(self, generation_id: str):
        self.generation_id = generation_id
        self.total_steps = 0
        
    def __call__(self, phase: str, current: int, total: int):
        """Update progress"""
        if total > 0:
            phase_progress = current / total
        else:
            phase_progress = 0.0
        
        # Store in active generations
        if self.generation_id in active_generations:
            active_generations[self.generation_id].update({
                "phase": phase,
                "progress": phase_progress,
                "current_step": current,
                "total_steps": total,
                "updated_at": datetime.now().isoformat()
            })

# ============================================================================
# VRAM Management
# ============================================================================

def clear_vram():
    """Aggressively clear VRAM after generation"""
    try:
        import torch
        import gc
        
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get VRAM stats
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            
            print(f"  VRAM - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            print(f"  ✓ VRAM cleared")
        else:
            print(f"  ⚠ CUDA not available, skipping VRAM clear")
            
    except Exception as e:
        print(f"  ⚠ VRAM clear failed: {e}")

# ============================================================================
# Image Generation
# ============================================================================

async def run_image_generation(generation_id: str, request: ImageGenerationRequest):
    """Background task for image generation"""
    
    try:
        print("\n" + "="*80)
        print(f"🚀 Starting Image Generation: {generation_id}")
        print("="*80)
        print(f"Prompt: {request.prompt}")
        print(f"Model: {request.model}")
        print(f"Size: {request.width}x{request.height} (W×H)")
        print(f"  → Musubi receives: {request.height}x{request.width} (H×W)")
        print(f"Steps: {request.steps}")
        print(f"Guidance: {request.guidance_scale}")
        print(f"Flow Shift: {request.flow_shift}")
        print(f"Seed: {request.seed}")
        print(f"LoRA: {request.lora} ({request.lora_multiplier}x)" if request.lora else "LoRA: None")
        print(f"Blocks to swap: {request.blocks_to_swap}")
        print("="*80 + "\n")
        
        # Update status
        active_generations[generation_id]["status"] = "running"
        active_generations[generation_id]["phase"] = "initialization"
        active_generations[generation_id]["total_steps"] = request.steps
        active_generations[generation_id]["current_step"] = 0
        print(f"[{generation_id}] Status: initialization")
        
        # Create generation directory
        print(f"[{generation_id}] Creating generation directory...")
        gen_dir = image_manager.create_generation_dir()
        paths = image_manager.get_output_paths(gen_dir)
        print(f"[{generation_id}] Output directory: {gen_dir}")
        
        active_generations[generation_id]["gen_dir"] = str(gen_dir)
        active_generations[generation_id]["phase"] = "generating"
        print(f"[{generation_id}] Status: generating")
        
        # Map model string to enum
        model = QwenModel.IMAGE if request.model == "qwen_image" else QwenModel.EDIT_2509
        print(f"[{generation_id}] Using model: {model}")
        
        # Generate with progress callback
        print(f"[{generation_id}] Calling musubi generator...")
        print(f"[{generation_id}] This may take several minutes depending on your hardware...")
        
        # Create a wrapper that updates progress
        def progress_callback(phase: str, current: int, total: int):
            """Update progress in active_generations"""
            if generation_id in active_generations:
                progress = current / total if total > 0 else 0
                active_generations[generation_id].update({
                    "phase": phase,
                    "current_step": current,
                    "total_steps": total,
                    "progress": progress,
                    "updated_at": datetime.now().isoformat()
                })
                print(f"[{generation_id}] Progress: {phase} {current}/{total} ({progress*100:.1f}%)")
        
        # Generate - we need to pass the progress callback through
        # For now, we'll manually track by monitoring the subprocess
        import subprocess
        import re
        
        # Build the musubi command manually to capture output
        model_config = image_generator.config["models"][request.model]
        save_dir = paths["generated_png"].parent
        
        cmd = [
            "python",
            str(image_generator.musubi_path / "qwen_image_generate_image.py"),
            "--dit", model_config["dit"],
            "--vae", model_config["vae"],
            "--text_encoder", model_config["text_encoder"],
            "--prompt", request.prompt,
            "--negative_prompt", request.negative_prompt or image_generator.defaults.get("negative_prompt", ""),
            "--output_type", "images",
            "--save_path", str(save_dir),
            "--image_size", str(request.height), str(request.width),  # Musubi expects HEIGHT, WIDTH
            "--infer_steps", str(request.steps),
            "--guidance_scale", str(request.guidance_scale),
            "--blocks_to_swap", str(request.blocks_to_swap),
            "--fp8_scaled",
            "--vae_enable_tiling"
        ]
        
        if request.flow_shift is not None:
            cmd.extend(["--flow_shift", str(request.flow_shift)])
        
        if request.control_image_path:
            cmd.extend(["--edit", "--image_path", str(request.control_image_path)])
        
        if request.seed is not None:
            cmd.extend(["--seed", str(request.seed)])
        
        if request.lora:
            lora_path = image_generator.config["loras"].get(request.lora, request.lora)
            cmd.extend([
                "--lora_weight", lora_path,
                "--lora_multiplier", str(request.lora_multiplier)
            ])
        
        # Run with real-time output parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Patterns to match step progress from musubi output
        # Pattern 1: "100%|██████████| 20/20 [00:45<00:00,  2.27s/it]"
        tqdm_pattern = re.compile(r'(\d+)/(\d+)\s+\[')
        # Pattern 2: "Step 5/20" or "step 5/20"
        step_pattern = re.compile(r'step\s+(\d+)/(\d+)', re.IGNORECASE)
        # Pattern 3: Just numbers like "5/20"
        simple_pattern = re.compile(r'^\s*(\d+)/(\d+)')
        
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            # Print to console
            print(line.rstrip())
            
            # Try to extract step progress with multiple patterns
            current = None
            total = None
            
            # Try tqdm pattern first (most common in musubi)
            match = tqdm_pattern.search(line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
            else:
                # Try step pattern
                match = step_pattern.search(line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                else:
                    # Try simple pattern
                    match = simple_pattern.search(line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
            
            # Update progress if we found a match
            if current is not None and total is not None:
                progress_callback("denoising", current, total)
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Musubi generation failed with code {process.returncode}")
        
        # Find the generated file
        generated_files = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        if not generated_files:
            raise RuntimeError("No output file found after generation")
        
        latest_file = generated_files[-1]
        if latest_file != paths["generated_png"]:
            latest_file.rename(paths["generated_png"])
        
        print(f"[{generation_id}] ✓ Generation complete!")
        
        outputs = {"png": str(paths["generated_png"].relative_to(OUTPUT_ROOT))}
        print(f"[{generation_id}] PNG saved: {paths['generated_png']}")
        
        # Convert to EXR if requested
        if request.output_format in ["exr", "both"]:
            active_generations[generation_id]["phase"] = "converting_to_exr"
            print(f"[{generation_id}] Status: converting to EXR...")
            
            FormatConverter.png_to_exr_32bit(
                paths["generated_png"],
                paths["generated_exr"],
                linear=True
            )
            outputs["exr"] = str(paths["generated_exr"].relative_to(OUTPUT_ROOT))
            print(f"[{generation_id}] EXR saved: {paths['generated_exr']}")
        
        # Save metadata
        print(f"[{generation_id}] Saving metadata...")
        image_manager.save_metadata(
            gen_dir=gen_dir,
            prompt=request.prompt,
            enhanced_prompt=request.prompt,
            model=request.model,
            seed=request.seed or 0,
            image_size=(request.width, request.height),
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            negative_prompt=request.negative_prompt or DEFAULTS.get("negative_prompt", ""),
            flow_shift=request.flow_shift
        )
        print(f"[{generation_id}] Metadata saved")
        
        # Mark complete
        active_generations[generation_id].update({
            "status": "complete",
            "phase": "complete",
            "progress": 1.0,
            "outputs": outputs,
            "completed_at": datetime.now().isoformat()
        })
        
        print("\n" + "="*80)
        print(f"✅ Generation Complete: {generation_id}")
        print(f"Output directory: {gen_dir}")
        print(f"Files: {', '.join(outputs.keys())}")
        print("="*80 + "\n")
        
        # Clear VRAM
        print(f"[{generation_id}] Clearing VRAM...")
        clear_vram()
        
    except Exception as e:
        active_generations[generation_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        
        print("\n" + "="*80)
        print(f"❌ Generation Failed: {generation_id}")
        print(f"Error: {e}")
        print("="*80)
        
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        
        # Clear VRAM even on failure
        print(f"[{generation_id}] Clearing VRAM after failure...")
        clear_vram()

@app.post("/api/generate/image", response_model=GenerationResponse)
async def generate_image(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """Start image generation"""
    
    generation_id = str(uuid.uuid4())
    
    # Initialize tracking
    active_generations[generation_id] = {
        "id": generation_id,
        "type": "image",
        "status": "queued",
        "phase": "queued",
        "progress": 0.0,
        "request": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    # Start background generation
    background_tasks.add_task(run_image_generation, generation_id, request)
    
    return GenerationResponse(
        generation_id=generation_id,
        status="queued",
        message="Image generation started"
    )

# ============================================================================
# Video Generation
# ============================================================================

async def run_video_generation(generation_id: str, request: VideoGenerationRequest):
    """Background task for video generation"""
    
    try:
        active_generations[generation_id]["status"] = "running"
        active_generations[generation_id]["phase"] = "initialization"
        
        # Create generation directory
        gen_dir = video_manager.create_generation_dir(workflow_type="i2v")
        paths = video_manager.get_output_paths(gen_dir)
        
        active_generations[generation_id]["gen_dir"] = str(gen_dir)
        active_generations[generation_id]["phase"] = "generating"
        
        # Copy control inputs if provided
        if request.image_path:
            video_manager.copy_control_inputs(
                gen_dir,
                image_path=Path(request.image_path) if request.image_path else None,
                end_image_path=Path(request.end_image_path) if request.end_image_path else None,
                control_path=Path(request.control_path) if request.control_path else None
            )
        
        # Create progress callback
        progress_cb = ProgressCallback(generation_id)
        
        # Map task string to enum
        task_enum = WanTask(request.task)
        
        # Generate video
        video_generator.generate_video(
            prompt=request.prompt,
            output_path=paths["video_mp4"],
            task=task_enum,
            video_size=(request.width, request.height),
            video_length=request.video_length,
            seed=request.seed,
            image_path=Path(request.image_path) if request.image_path else None,
            end_image_path=Path(request.end_image_path) if request.end_image_path else None,
            control_path=Path(request.control_path) if request.control_path else None,
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            negative_prompt=request.negative_prompt,
            blocks_to_swap=request.blocks_to_swap,
            fp8=request.fp8,
            fp8_scaled=request.fp8_scaled,
            fp8_t5=request.fp8_t5,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            progress_callback=progress_cb
        )
        
        outputs = {"mp4": str(paths["video_mp4"])}
        
        # Export to EXR if requested
        if request.export_exr:
            active_generations[generation_id]["phase"] = "exporting_exr"
            
            # Check if latent exists for lossless export
            latent_path = gen_dir / "latent.safetensors"
            if latent_path.exists():
                exr_dir = video_manager.export_latent_to_exr(
                    gen_dir=gen_dir,
                    task=request.task,
                    config_path=CONFIG_PATH,
                    musubi_path=MUSUBI_PATH,
                    linear=True
                )
            else:
                exr_dir = video_manager.export_to_exr_sequence(
                    gen_dir=gen_dir,
                    video_path=paths["video_mp4"],
                    linear=True
                )
            
            outputs["exr_sequence"] = str(exr_dir)
        
        # Save metadata
        video_manager.save_metadata(
            gen_dir=gen_dir,
            prompt=request.prompt,
            enhanced_prompt=request.prompt,
            task=request.task,
            video_size=(request.width, request.height),
            video_length=request.video_length,
            seed=request.seed or 0,
            image_path=Path(request.image_path) if request.image_path else None,
            end_image_path=Path(request.end_image_path) if request.end_image_path else None,
            control_path=Path(request.control_path) if request.control_path else None,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            negative_prompt=request.negative_prompt or DEFAULTS.get("negative_prompt", "")
        )
        
        # Mark complete
        active_generations[generation_id].update({
            "status": "complete",
            "phase": "complete",
            "progress": 1.0,
            "outputs": outputs,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        active_generations[generation_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        print(f"Generation {generation_id} failed: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/generate/video", response_model=GenerationResponse)
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Start video generation"""
    
    generation_id = str(uuid.uuid4())
    
    # Initialize tracking
    active_generations[generation_id] = {
        "id": generation_id,
        "type": "video",
        "status": "queued",
        "phase": "queued",
        "progress": 0.0,
        "request": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    # Start background generation
    background_tasks.add_task(run_video_generation, generation_id, request)
    
    return GenerationResponse(
        generation_id=generation_id,
        status="queued",
        message="Video generation started"
    )

# ============================================================================
# Cancel Generation
# ============================================================================

@app.post("/api/cancel/{generation_id}")
async def cancel_generation(generation_id: str):
    """Cancel an active generation"""
    
    if generation_id not in active_generations:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    gen = active_generations[generation_id]
    
    if gen["status"] in ["complete", "failed"]:
        return {"message": "Generation already finished"}
    
    # Mark as cancelled
    gen["status"] = "cancelled"
    gen["phase"] = "cancelled"
    gen["cancelled_at"] = datetime.now().isoformat()
    
    print(f"\n⚠ Generation Cancelled: {generation_id}")
    print(f"Note: Backend process may still be running (musubi doesn't support mid-generation cancellation)")
    
    # Clear VRAM
    print(f"[{generation_id}] Clearing VRAM after cancellation...")
    clear_vram()
    
    return {"message": "Generation marked as cancelled"}

# ============================================================================
# Status & Progress
# ============================================================================

@app.get("/api/status/{generation_id}", response_model=GenerationStatus)
async def get_status(generation_id: str):
    """Get generation status"""
    
    if generation_id not in active_generations:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    gen = active_generations[generation_id]
    
    return GenerationStatus(
        generation_id=generation_id,
        status=gen["status"],
        progress=gen.get("progress", 0.0),
        phase=gen.get("phase", "unknown"),
        outputs=gen.get("outputs", {}),
        error=gen.get("error"),
        metadata=gen.get("request")
    )

@app.get("/api/status")
async def get_all_status():
    """Get all active generations"""
    return {
        "active": [
            {
                "id": gen_id,
                "type": gen["type"],
                "status": gen["status"],
                "progress": gen.get("progress", 0.0),
                "phase": gen.get("phase", "unknown"),
                "created_at": gen["created_at"]
            }
            for gen_id, gen in active_generations.items()
        ]
    }

@app.get("/api/progress/{generation_id}")
async def stream_progress(generation_id: str):
    """Stream progress updates via Server-Sent Events"""
    
    if generation_id not in active_generations:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    async def event_generator():
        """Generate SSE events"""
        last_update = None
        last_status = None
        retry_count = 0
        max_retries = 60  # 30 seconds max
        
        # Send initial state immediately
        gen = active_generations.get(generation_id)
        if gen:
            yield f"data: {json.dumps(gen)}\n\n"
        
        while retry_count < max_retries:
            gen = active_generations.get(generation_id)
            
            if not gen:
                print(f"[SSE] Generation {generation_id} not found in active_generations")
                break
            
            # Send update if changed or periodically
            current_update = gen.get("updated_at")
            current_status = gen.get("status")
            
            if current_update != last_update or current_status != last_status:
                try:
                    yield f"data: {json.dumps(gen)}\n\n"
                    last_update = current_update
                    last_status = current_status
                except Exception as e:
                    print(f"[SSE] Error sending update: {e}")
                    break
            
            # Stop if complete or failed
            if gen["status"] in ["complete", "failed", "cancelled"]:
                print(f"[SSE] Generation {generation_id} finished with status: {gen['status']}")
                # Send final update one more time
                yield f"data: {json.dumps(gen)}\n\n"
                break
            
            await asyncio.sleep(0.5)
            retry_count += 1
        
        print(f"[SSE] Closing stream for {generation_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

# ============================================================================
# File Serving
# ============================================================================

@app.get("/api/file/{generation_id}/{filename}")
async def serve_file(generation_id: str, filename: str):
    """Serve generated files"""
    
    if generation_id not in active_generations:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    gen = active_generations[generation_id]
    gen_dir = Path(gen.get("gen_dir", ""))
    
    if not gen_dir.exists():
        raise HTTPException(status_code=404, detail="Generation directory not found")
    
    file_path = gen_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

# ============================================================================
# History & Gallery
# ============================================================================

@app.get("/api/history/images")
async def get_image_history(limit: int = 20):
    """Get recent image generations"""
    
    image_dir = OUTPUT_ROOT / "image"
    
    # Find all generation directories
    date_dirs = sorted(image_dir.glob("20*"), reverse=True)
    
    generations = []
    for date_dir in date_dirs:
        gen_dirs = sorted(date_dir.glob("generation_*"), reverse=True)
        
        for gen_dir in gen_dirs:
            metadata_path = gen_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Find output files
                png_file = gen_dir / "generated.png"
                
                generations.append({
                    "id": gen_dir.name,
                    "date": date_dir.name,
                    "timestamp": metadata.get("timestamp"),
                    "prompt": metadata.get("prompt", {}).get("original", ""),
                    "seed": metadata.get("seed"),
                    "metadata": metadata,
                    "thumbnail": f"/api/file/{gen_dir.name}/generated.png" if png_file.exists() else None
                })
                
                if len(generations) >= limit:
                    break
        
        if len(generations) >= limit:
            break
    
    return {"generations": generations}

@app.get("/api/history/videos")
async def get_video_history(limit: int = 20):
    """Get recent video generations"""
    
    video_dir = OUTPUT_ROOT / "video"
    
    date_dirs = sorted(video_dir.glob("20*"), reverse=True)
    
    generations = []
    for date_dir in date_dirs:
        gen_dirs = sorted(date_dir.glob("*_*"), reverse=True)
        
        for gen_dir in gen_dirs:
            metadata_path = gen_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                mp4_file = gen_dir / "generated.mp4"
                
                generations.append({
                    "id": gen_dir.name,
                    "date": date_dir.name,
                    "timestamp": metadata.get("timestamp"),
                    "prompt": metadata.get("prompt", {}).get("original", ""),
                    "seed": metadata.get("seed"),
                    "metadata": metadata,
                    "video": f"/api/file/{gen_dir.name}/generated.mp4" if mp4_file.exists() else None
                })
                
                if len(generations) >= limit:
                    break
        
        if len(generations) >= limit:
            break
    
    return {"generations": generations}

# ============================================================================
# Configuration & Info
# ============================================================================

@app.get("/api/config/defaults")
async def get_defaults():
    """Get default generation parameters"""
    return DEFAULTS

@app.get("/api/config/models")
async def get_models():
    """Get available models"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    return {
        "image_models": list(config.get("models", {}).keys()),
        "video_tasks": list(config.get("wan_models", {}).keys()),
        "loras": list(config.get("loras", {}).keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.3",
        "features": [
            "verbose_logging",
            "vram_clearing", 
            "time_tracking",
            "static_file_serving",
            "realtime_step_tracking",
            "persistent_state"
        ],
        "active_generations": len(active_generations),
        "musubi_path": str(MUSUBI_PATH),
        "output_root": str(OUTPUT_ROOT)
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("FUK Generation Web Server")
    print("="*60)
    print(f"Output directory: {OUTPUT_ROOT.absolute()}")
    print(f"Musubi path: {MUSUBI_PATH.absolute()}")
    print(f"Config: {CONFIG_PATH.absolute()}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")