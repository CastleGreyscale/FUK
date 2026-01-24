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

import sys
import os
from pathlib import Path

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

    # Determine base directory (where this script is located)
BASE_DIR = Path(__file__).parent

## Paths
UI_DIR = Path(__file__).parent          # fuk/ui/
ROOT_DIR = UI_DIR.parent                 # fuk/
CONFIG_DIR = ROOT_DIR / "config"         # fuk/config/
VENDOR_DIR = ROOT_DIR / "vendor"         # fuk/vendor/
sys.path.insert(0, str(ROOT_DIR))   # Add fuk/ to path

OUTPUT_ROOTS = [
    ROOT_DIR / "outputs",
    ROOT_DIR.parent / "outputs",
    Path("outputs"),
]



from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List, Dict, Any
from file_browser_endpoints import setup_file_browser_routes
from video_endpoints import setup_video_routes
from core.qwen_image_wrapper import create_generator as create_image_generator, QwenModel
from core.wan_video_wrapper import create_video_generator, WanTask
from core.video_processor import VideoProcessor
import json
import asyncio
import uuid
from datetime import datetime
from enum import Enum
import time
import traceback
from project_endpoints import (
    setup_project_routes,
    get_generation_output_dir,
    build_output_paths,
    get_project_relative_url,
    save_generation_metadata,
    get_cache_root,
    get_default_cache_root,
    cleanup_failed_generation
)


# ============================================================================
# Logging Utility
# ============================================================================

class FukLogger:
    """Centralized logging with timestamps and categories"""
    
    COLORS = {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    
    @staticmethod
    def timestamp():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    @classmethod
    def header(cls, title: str, char: str = "="):
        width = 70
        print(f"\n{cls.COLORS['cyan']}{char * width}")
        print(f"  {title}")
        print(f"{char * width}{cls.COLORS['end']}\n")
    
    @classmethod
    def section(cls, title: str):
        print(f"\n{cls.COLORS['blue']}--- {title} ---{cls.COLORS['end']}")
    
    @classmethod
    def info(cls, category: str, message: str):
        print(f"{cls.COLORS['cyan']}[{cls.timestamp()}]{cls.COLORS['end']} [{category}] {message}")
    
    @classmethod
    def success(cls, category: str, message: str):
        print(f"{cls.COLORS['green']}[{cls.timestamp()}] [{category}] {message}{cls.COLORS['end']}")
    
    @classmethod
    def warning(cls, category: str, message: str):
        print(f"{cls.COLORS['yellow']}[{cls.timestamp()}] [{category}] {message}{cls.COLORS['end']}")
    
    @classmethod
    def error(cls, category: str, message: str):
        print(f"{cls.COLORS['red']}[{cls.timestamp()}] [{category}] {message}{cls.COLORS['end']}")
    
    @classmethod
    def params(cls, title: str, params: Dict[str, Any]):
        """Log parameters in a formatted way"""
        print(f"\n{cls.COLORS['yellow']}  {title}:{cls.COLORS['end']}")
        for key, value in params.items():
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 80:
                str_val = str_val[:77] + "..."
            print(f"    {key}: {str_val}")
    
    @classmethod
    def command(cls, cmd: List[str]):
        """Log a command being executed"""
        print(f"\n{cls.COLORS['bold']}  Command:{cls.COLORS['end']}")
        # Format command nicely
        formatted = " \\\n    ".join(cmd)
        print(f"    {formatted}")
        print()
    
    @classmethod
    def paths(cls, title: str, paths: Dict[str, Any]):
        """Log path information"""
        print(f"\n{cls.COLORS['blue']}  {title}:{cls.COLORS['end']}")
        for key, value in paths.items():
            print(f"    {key}: {value}")
    
    @classmethod
    def timing(cls, category: str, start_time: float, message: str = ""):
        elapsed = time.time() - start_time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s"
        print(f"{cls.COLORS['green']}[{cls.timestamp()}] ÃƒÂ¢Ã‚ÂÃ‚Â± [{category}] {message} ({time_str}){cls.COLORS['end']}")
    
    @classmethod
    def exception(cls, category: str, e: Exception):
        """Log full exception with traceback"""
        print(f"\n{cls.COLORS['red']}{'=' * 70}")
        print(f"  EXCEPTION in {category}")
        print(f"{'=' * 70}")
        print(f"  Error: {type(e).__name__}: {e}")
        print(f"\n  Traceback:")
        for line in traceback.format_exc().split('\n'):
            print(f"    {line}")
        print(f"{'=' * 70}{cls.COLORS['end']}\n")

log = FukLogger()

# Add fuk/fuk/ root to path so we can import from core/
current_dir = Path(__file__).parent      # fuk/fuk/ui/
fuk_root = current_dir.parent            # fuk/fuk/
sys.path.insert(0, str(fuk_root))
sys.path.insert(0, str(current_dir))

print(f"[STARTUP] Server directory: {current_dir}")
print(f"[STARTUP] FUK root: {fuk_root}")
print(f"[STARTUP] Looking for core/ at: {fuk_root / 'core'}")

# Import from core/ directory
try:
    from core.qwen_image_wrapper import QwenImageGenerator, QwenModel
    from core.wan_video_wrapper import WanVideoGenerator, WanTask
    from core.image_generation_manager import ImageGenerationManager
    from core.video_generation_manager import VideoGenerationManager
    from core.format_convert import FormatConverter
    from core.preprocessors import PreprocessorManager, DepthModel, NormalsMethod, SAMModel
    from core.postprocessors import PostProcessorManager
    print("[STARTUP] Loaded modules from core/")
except ImportError as e:
    print(f"\n{'='*60}")
    print("ERROR: Cannot import FUK modules from core/")
    print(f"{'='*60}")
    print(f"\nLooking in: {fuk_root / 'core'}")
    print(f"\nExpected files:")
    print("  - core/qwen_image_wrapper.py")
    print("  - core/wan_video_wrapper.py")
    print("  - core/image_generation_manager.py")
    print("  - core/video_generation_manager.py")
    print("  - core/format_convert.py")
    print("  - core/preprocessors.py")
    print(f"\nImport error: {e}")
    print(f"{'='*60}\n")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================



# Find first existing path for each
def find_path(paths, name):
    for p in paths:
        if p.exists():
            print(f"[STARTUP] Found {name}: {p}", flush=True)
            return p
    print(f"[STARTUP] {name} not found in:", flush=True)
    for p in paths:
        print(f"  - {p}", flush=True)
    raise FileNotFoundError(f"{name} not found")

try:
    # Verify required config paths exist
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(f"Config directory not found: {CONFIG_DIR}")
    if not (CONFIG_DIR / "models.json").exists():
        raise FileNotFoundError(f"models.json not found in {CONFIG_DIR}")
    if not (CONFIG_DIR / "defaults.json").exists():
        raise FileNotFoundError(f"defaults.json not found in {CONFIG_DIR}")
    if not (VENDOR_DIR / "musubi-tuner").exists():
        raise FileNotFoundError(f"musubi-tuner not found in {VENDOR_DIR}")
    
    print(f"[STARTUP] Config directory: {CONFIG_DIR}", flush=True)
    print(f"[STARTUP] Vendor directory: {VENDOR_DIR}", flush=True)
    
    # Output directory (use first option, create if needed)
    OUTPUT_ROOT = OUTPUT_ROOTS[0]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] Output directory: {OUTPUT_ROOT}", flush=True)
    
except FileNotFoundError as e:
    print(f"\nError: {e}", flush=True)
    print("\nPlease ensure your project has the following structure:", flush=True)
    print("  config/models.json", flush=True)
    print("  config/defaults.json", flush=True)
    print("  vendor/musubi-tuner/", flush=True)
    sys.exit(1)

# Ensure output subdirs exist
(OUTPUT_ROOT / "image").mkdir(exist_ok=True)
(OUTPUT_ROOT / "video").mkdir(exist_ok=True)
# Uploads removed - files stay in place


# Cache directory for project-aware outputs
CACHE_ROOT = ROOT_DIR / "cache"
CACHE_ROOT.mkdir(exist_ok=True)

# ============================================================================
# Initialize Generators
# ============================================================================

# Initialize image generator
image_generator = create_image_generator(
    config_dir=CONFIG_DIR,
    vendor_dir=VENDOR_DIR
)

print("[STARTUP] Image generator initialized", flush=True)


# Initialize video generator
video_generator = create_video_generator(
    config_dir=CONFIG_DIR,
    vendor_dir=VENDOR_DIR
)
print("[STARTUP] Video generator initialized", flush=True)

# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="FUK Generation API", version="1.0.0")

# Initialize project system
from project_endpoints import initialize_project_system
initialize_project_system(CACHE_ROOT)

setup_project_routes(app)

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

# Uploads removed - files stay in place

# Serve project cache directory
app.mount("/project-cache", StaticFiles(directory=str(CACHE_ROOT)), name="project-cache")

# Active generations tracking
active_generations: Dict[str, Dict[str, Any]] = {}

# Load defaults
with open(CONFIG_DIR / "defaults.json") as f:
    DEFAULTS = json.load(f)



image_manager = ImageGenerationManager(OUTPUT_ROOT / "image")
video_manager = VideoGenerationManager(OUTPUT_ROOT / "video")
preprocessor_manager = PreprocessorManager(
output_dir=OUTPUT_ROOT / "preprocessed",
config_dir=CONFIG_DIR
)
postprocessor_manager = PostProcessorManager(OUTPUT_ROOT / "postprocessed")

# ============================================================================
# Path Resolution Helper
# ============================================================================

def resolve_input_path(path_str: str) -> Optional[Path]:
    """
    Convert URL paths or relative paths to absolute filesystem paths.
    
    Handles:
    - Already absolute paths -> return as-is
    - URL paths: api/project/files/... -> extract absolute path for external files
    - URL paths: api/project/cache/... -> project cache / relative
    - URL paths: project-cache/... -> project cache / relative (legacy)
    - Other relative paths -> project cache or OUTPUT_ROOT
    
    Falls back to checking BOTH project cache and default cache when files
    aren't found at the expected location.
    """
    if not path_str:
        return None
    
    # Get current cache root from project system
    cache_root = get_cache_root()
    default_cache = get_default_cache_root() or CACHE_ROOT
    
    print(f"[PATH] ========================================", flush=True)
    print(f"[PATH] Resolving: {path_str}", flush=True)
    print(f"[PATH] Project cache: {cache_root}", flush=True)
    print(f"[PATH] Default cache: {default_cache}", flush=True)
    
    if cache_root is None:
        print(f"[PATH] ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã‚Â¡Ãƒâ€š ÃƒÆ’Ã‚Â¯Ãƒâ€šÃ‚Â¸Ãƒâ€šÃ‚Â  WARNING: No project cache set! Is a project loaded?", flush=True)
    
    p = Path(path_str)
    
    # Already absolute path
    if p.is_absolute():
        print(f"[PATH] -> Absolute: {p}", flush=True)
        return p
    
    # URL path for external files: api/files/... -> extract absolute path (legacy format)
    if path_str.startswith('api/files/'):
        # Strip the api/files prefix to get absolute path
        absolute_path = '/' + path_str.replace('api/files/', '', 1)
        resolved = Path(absolute_path)
        print(f"[PATH] -> From api/files URL (legacy): {resolved}", flush=True)
        return resolved
    
    # URL path for external files: api/project/files/... -> extract absolute path (new format)
    if path_str.startswith('api/project/files/'):
        # Strip the api/project/files prefix to get absolute path
        absolute_path = '/' + path_str.replace('api/project/files/', '', 1)
        resolved = Path(absolute_path)
        print(f"[PATH] -> From api/project/files URL: {resolved}", flush=True)
        return resolved
    
    # URL path from dynamic cache endpoint: api/project/cache/...
    if path_str.startswith('api/project/cache/'):
        relative = path_str.replace('api/project/cache/', '', 1)
        print(f"[PATH] Relative from URL: {relative}", flush=True)
        
        # Try project cache first if set
        if cache_root:
            resolved = cache_root / relative
            print(f"[PATH] Project cache path: {resolved}", flush=True)
            if resolved.exists():
                print(f"[PATH] File exists in project cache", flush=True)
                return resolved
            else:
                print(f"[PATH]  File NOT found in project cache, checking default...", flush=True)
        
        # Fall back to default cache
        if default_cache:
            default_resolved = default_cache / relative
            print(f"[PATH] Default cache path: {default_resolved}", flush=True)
            if default_resolved.exists():
                print(f"[PATH] File found in default cache (fallback)", flush=True)
                return default_resolved
        
        # If still not found, return project cache path (for creation) or default
        if cache_root:
            print(f"[PATH] File not found anywhere, using project cache: {resolved}", flush=True)
            return resolved
        
        resolved = default_cache / relative
        print(f"[PATH] Using default cache (no project): {resolved}", flush=True)
        return resolved
    
    # Legacy URL path: project-cache/...
    if path_str.startswith('project-cache/'):
        relative = path_str.replace('project-cache/', '', 1)
        
        # Try project cache first
        if cache_root:
            resolved = cache_root / relative
            if resolved.exists():
                print(f"[PATH] -> Found in project cache (legacy URL): {resolved}", flush=True)
                return resolved
        
        # Fall back to default
        if default_cache:
            default_resolved = default_cache / relative
            if default_resolved.exists():
                print(f"[PATH] -> Found in default cache (legacy URL fallback): {default_resolved}", flush=True)
                return default_resolved
        
        # Return project cache path for creation, or default
        if cache_root:
            print(f"[PATH] -> From project cache (legacy URL): {resolved}", flush=True)
            return resolved
        
        resolved = default_cache / relative
        print(f"[PATH] -> From default cache (legacy URL): {resolved}", flush=True)
        return resolved
    

    # Other relative path - try various locations
    resolved = OUTPUT_ROOT / path_str
    if resolved.exists():
        print(f"[PATH] -> From OUTPUT_ROOT: {resolved}", flush=True)
        return resolved
    
    if cache_root:
        cache_resolved = cache_root / path_str
        if cache_resolved.exists():
            print(f"[PATH] -> From project cache: {cache_resolved}", flush=True)
            return cache_resolved
    
    if default_cache:
        default_cache_resolved = default_cache / path_str
        if default_cache_resolved.exists():
            print(f"[PATH] -> From default cache: {default_cache_resolved}", flush=True)
            return default_cache_resolved
    
    print(f"[PATH] -> Default (OUTPUT_ROOT): {resolved}", flush=True)
    return resolved


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
    control_image_path: Optional[str] = None  # For backward compatibility (single image)
    control_image_paths: Optional[List[str]] = None  # For multiple images
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
    blocks_to_swap: int = 0
    export_exr: bool = False
    # Removed: fp8, fp8_scaled, fp8_t5 - now in tool config
    
class GenerationResponse(BaseModel):
    generation_id: str
    status: str
    message: str
    
class UpscaleRequest(BaseModel):
    source_path: str
    scale: int = 4
    model: str = "realesrgan"
    denoise: float = 0.5

class InterpolateRequest(BaseModel):
    source_path: str
    source_fps: int = 16
    target_fps: int = 24
    model: str = "rife"

class GenerationStatus(BaseModel):
    generation_id: str
    status: str  # queued, running, complete, failed
    progress: float  # 0.0 to 1.0
    phase: str
    outputs: Dict[str, str]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PreprocessRequest(BaseModel):
    image_path: str
    method: str  # 'canny', 'openpose', 'depth'
    
    # Canny parameters
    low_threshold: int = 100
    high_threshold: int = 200
    canny_invert: bool = False
    blur_kernel: int = 3
    
    # OpenPose parameters
    detect_body: bool = True
    detect_hand: bool = False
    detect_face: bool = False
    
    # Depth parameters
    depth_model: str = "da3_mono_large"
    depth_invert: bool = False
    depth_normalize: bool = True
    depth_colormap: Optional[str] = "inferno"
    depth_process_res: int = 756  # NEW - higher res for better quality
    depth_process_res_method: str = "lower_bound_resize"  # NEW

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
            
            print(f"VRAM - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            print(f"VRAM cleared")
        else:
            print(f"CUDA not available, skipping VRAM clear")
            
    except Exception as e:
        print(f"VRAM clear failed: {e}")

# ============================================================================
# Image Generation
# ============================================================================

async def run_image_generation(generation_id: str, request: ImageGenerationRequest):
    """Background task for image generation"""
    
    try:
        print("\n" + "="*80)
        print(f"Starting Image Generation: {generation_id}")
        print("="*80)
        print(f"Prompt: {request.prompt}")
        print(f"Model: {request.model}")
        print(f"Size: {request.width}x{request.height}")
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
        
        # Create generation directory
        gen_dir = get_generation_output_dir("img_gen")
        paths = build_output_paths(gen_dir)
        print(f"[{generation_id}] Output directory: {gen_dir}")
        
        active_generations[generation_id]["gen_dir"] = str(gen_dir)
        active_generations[generation_id]["phase"] = "generating"
        
        # Map model string to enum
        model = QwenModel.IMAGE if request.model == "qwen_image" else QwenModel.EDIT_2509
        
        # Handle control images for edit mode (supports multiple)
        control_images = []
        if request.control_image_paths:
            for img_path in request.control_image_paths:
                resolved = resolve_input_path(img_path)
                if resolved and resolved.exists():
                    control_images.append(resolved)
            if control_images:
                print(f"[{generation_id}] Control images ({len(control_images)}): {[str(p) for p in control_images]}")
        elif request.control_image_path:
            # Backward compatibility for single image
            resolved = resolve_input_path(request.control_image_path)
            if resolved and resolved.exists():
                control_images.append(resolved)
                print(f"[{generation_id}] Control image: {resolved}")
        
        # Generate using the wrapper
        result = image_generator.generate(
            prompt=request.prompt,
            output_path=paths["generated_png"],
            model=model,
            width=request.width,
            height=request.height,
            seed=request.seed,
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            blocks_to_swap=request.blocks_to_swap,
            negative_prompt=request.negative_prompt,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            control_image=control_images if control_images else None,
        )
        
        print(f"[{generation_id}] Generation complete!")
        
        outputs = {
            "png": get_project_relative_url(paths["generated_png"])
        }
        print(f"[{generation_id}] PNG saved: {paths['generated_png']}")
        
        # Convert to EXR if requested
        if request.output_format in ["exr", "both"]:
            active_generations[generation_id]["phase"] = "converting_to_exr"
            print(f"[{generation_id}] Converting to EXR...")
            
            FormatConverter.png_to_exr_32bit(
                paths["generated_png"],
                paths["generated_exr"],
                linear=True
            )
            outputs["exr"] = get_project_relative_url(paths["generated_exr"])
            print(f"[{generation_id}] EXR saved: {paths['generated_exr']}")
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=request.prompt,
            model=request.model,
            seed=result.get("seed_used") or request.seed or 0,
            image_size=(request.width, request.height),
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt or "",
            flow_shift=request.flow_shift,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            control_image=[str(p) for p in control_images] if control_images else None
        )
        
        # Mark complete
        active_generations[generation_id].update({
            "status": "complete",
            "phase": "complete",
            "progress": 1.0,
            "outputs": outputs,
            "seed_used": result.get("seed_used") or request.seed,
            "completed_at": datetime.now().isoformat()
        })
        
        print("\n" + "="*80)
        print(f"Generation Complete: {generation_id}")
        print(f"Output: {outputs['png']}")
        print("="*80 + "\n")
        
        # Clear VRAM
        clear_vram()
        
    except Exception as e:
        active_generations[generation_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        print(f"Generation Failed: {e}")
        import traceback
        traceback.print_exc()

        # Clean up failed generation directory
        gen_dir = active_generations[generation_id].get("gen_dir")
        if gen_dir:
            cleanup_failed_generation(gen_dir, reason=str(e))

        # Clear VRAM even on failure
        print(f"[{generation_id}] Clearing VRAM after failure...")
        clear_vram()


@app.post("/api/generate/image", response_model=GenerationResponse)
async def generate_image(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """Start image generation"""
    
    generation_id = str(uuid.uuid4())
    
    active_generations[generation_id] = {
        "id": generation_id,
        "type": "image",
        "status": "queued",
        "phase": "queued",
        "progress": 0.0,
        "request": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
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
    
    start_time = time.time()
    
    try:
        log.header("VIDEO GENERATION")
        log.info("VideoGen", f"Generation ID: {generation_id}")
        
        active_generations[generation_id]["status"] = "running"
        active_generations[generation_id]["phase"] = "initialization"
        
        # Create generation directory in project cache
        gen_dir = get_generation_output_dir("video")
        paths = build_output_paths(gen_dir)
        log.info("VideoGen", f"Output directory: {gen_dir}")
        
        active_generations[generation_id]["gen_dir"] = str(gen_dir)
        active_generations[generation_id]["phase"] = "generating"
        
        # Resolve image paths
        image_path_abs = resolve_input_path(request.image_path)
        end_image_path_abs = resolve_input_path(request.end_image_path)
        control_path_abs = resolve_input_path(request.control_path)
        
        print(f"[VIDEO] Resolved paths:", flush=True)
        print(f"[VIDEO]   image_path: {request.image_path} -> {image_path_abs}", flush=True)
        print(f"[VIDEO]   end_image_path: {request.end_image_path} -> {end_image_path_abs}", flush=True)
        print(f"[VIDEO]   control_path: {request.control_path} -> {control_path_abs}", flush=True)
        
        # Copy control inputs for reference
        if image_path_abs:
            control_dir = gen_dir / "control"
            control_dir.mkdir(exist_ok=True)
            
            if image_path_abs.exists():
                import shutil
                shutil.copy(image_path_abs, control_dir / "start_image.png")
            
            if end_image_path_abs and end_image_path_abs.exists():
                shutil.copy(end_image_path_abs, control_dir / "end_image.png")
        
        # Create progress callback
        progress_cb = ProgressCallback(generation_id)
        
        # Map task string to enum
        task_enum = WanTask(request.task)
        
        # Generate video using the wrapper
        result = video_generator.generate_video(
            prompt=request.prompt,
            output_path=paths["generated_mp4"],
            task=task_enum,
            width=request.width,
            height=request.height,
            video_length=request.video_length,
            seed=request.seed,
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            blocks_to_swap=request.blocks_to_swap,
            negative_prompt=request.negative_prompt,
            image_path=image_path_abs,
            end_image_path=end_image_path_abs,
            control_path=control_path_abs,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            progress_callback=progress_cb,
        )
        
        # Extract thumbnail from generated video
        if paths["generated_mp4"].exists():
            thumb_path = paths["generated_mp4"].with_suffix('.thumb.jpg')
            video_processor = VideoProcessor()
            video_processor.extract_thumbnail(paths["generated_mp4"], thumb_path)
        
        # Build outputs with project-relative URLs
        outputs = {
            "mp4": get_project_relative_url(paths["generated_mp4"])
        }
        
        # Export to EXR if requested
        if request.export_exr:
            active_generations[generation_id]["phase"] = "exporting_exr"
            
            latent_path = gen_dir / "latent.safetensors"
            if latent_path.exists():
                exr_dir = video_manager.export_latent_to_exr(
                    gen_dir=gen_dir,
                    task=request.task,
                    config_path=CONFIG_DIR / "models.json",
                    musubi_path=VENDOR_DIR / "musubi-tuner",
                    linear=True
                )
            else:
                exr_dir = video_manager.export_to_exr_sequence(
                    gen_dir=gen_dir,
                    video_path=paths["generated_mp4"],
                    linear=True
                )
            
            outputs["exr_sequence"] = get_project_relative_url(exr_dir)
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=request.prompt,
            model=request.task,
            seed=result.get("seed_used") or request.seed or 0,
            image_size=(request.width, request.height),
            video_length=request.video_length,
            negative_prompt=request.negative_prompt or "",
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            infer_steps=request.steps,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            start_image=request.image_path,
            end_image=request.end_image_path,
        )
        
        # Mark complete
        active_generations[generation_id].update({
            "status": "complete",
            "phase": "complete",
            "progress": 1.0,
            "outputs": outputs,
            "seed_used": result.get("seed_used") or request.seed,
            "completed_at": datetime.now().isoformat()
        })
        
        # Clear VRAM
        clear_vram()
        
    except Exception as e:
        active_generations[generation_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        print(f"Generation {generation_id} failed: {e}")
        import traceback
        traceback.print_exc()

        # Clean up failed generation directory
        gen_dir = active_generations[generation_id].get("gen_dir")
        if gen_dir:
            cleanup_failed_generation(gen_dir, reason=str(e))

        # Clear VRAM even on failure (if not already done by wrapper)
        try:
            clear_vram()
        except:
            pass

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
    
    print(f"\nGeneration Cancelled: {generation_id}")
    print(f"Note: Backend process may still be running (musubi doesn't support mid-generation cancellation)")
    
    # Clear VRAM
    print(f"[{generation_id}] Clearing VRAM after cancellation...")
    clear_vram()
    
    return {"message": "Generation marked as cancelled"}

    # Clean up cancelled generation directory
    gen_dir = gen.get("gen_dir")
    if gen_dir:
        cleanup_failed_generation(gen_dir, reason="cancelled by user")

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
    with open(CONFIG_DIR / "models.json") as f:
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
            "persistent_state",
            "seed_favorites"
        ],
        "active_generations": len(active_generations),
        "musubi_path": str(MUSUBI_PATH),
        "output_root": str(OUTPUT_ROOT)
    }

# ============================================================================
# Saved Seeds Storage
# ============================================================================

# Seeds are stored in data/saved_seeds.json at project root (alongside config/, core/, etc.)
SEEDS_FILE = BASE_DIR / "data" / "saved_seeds.json"

def _load_seeds() -> Dict[str, list]:
    """Load saved seeds from file"""
    if not SEEDS_FILE.exists():
        return {}
    try:
        with open(SEEDS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load seeds file: {e}")
        return {}

def _save_seeds(seeds: Dict[str, list]):
    """Save seeds to file"""
    SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEDS_FILE, 'w') as f:
        json.dump(seeds, f, indent=2)

@app.get("/api/seeds")
async def get_all_seeds():
    """Get all saved seeds"""
    return _load_seeds()

@app.get("/api/seeds/{model}")
async def get_seeds_for_model(model: str):
    """Get saved seeds for a specific model"""
    seeds = _load_seeds()
    return seeds.get(model, [])

class SaveSeedRequest(BaseModel):
    model: str
    seed: int
    note: str = ""

@app.post("/api/seeds")
async def save_seed(request: SaveSeedRequest):
    """Save a seed for a model"""
    seeds = _load_seeds()
    
    if request.model not in seeds:
        seeds[request.model] = []
    
    model_seeds = seeds[request.model]
    
    # Check if seed already exists
    existing = next((s for s in model_seeds if s["seed"] == request.seed), None)
    if existing:
        # Update note
        existing["note"] = request.note
        existing["updatedAt"] = datetime.now().isoformat()
    else:
        # Add new seed
        model_seeds.insert(0, {
            "seed": request.seed,
            "note": request.note,
            "timestamp": datetime.now().isoformat()
        })
        # Limit to 50 per model
        seeds[request.model] = model_seeds[:50]
    
    _save_seeds(seeds)
    return {"success": True, "seeds": seeds[request.model]}

class RemoveSeedRequest(BaseModel):
    model: str
    seed: int

@app.delete("/api/seeds")
async def remove_seed(request: RemoveSeedRequest):
    """Remove a saved seed"""
    seeds = _load_seeds()
    
    if request.model in seeds:
        seeds[request.model] = [s for s in seeds[request.model] if s["seed"] != request.seed]
        if not seeds[request.model]:
            del seeds[request.model]
        _save_seeds(seeds)
    
    return {"success": True}

# ============================================================================
# PreProcessors
# ============================================================================

class PreprocessRequest(BaseModel):
    image_path: str
    method: str  # 'canny', 'openpose', 'depth', 'normals', 'crypto'
    
    # Canny parameters
    low_threshold: int = 100
    high_threshold: int = 200
    canny_invert: bool = False
    blur_kernel: int = 3
    
    # OpenPose parameters
    detect_body: bool = True
    detect_hand: bool = False
    detect_face: bool = False

    # Depth Parameters 
    depth_model: str = "da3_mono_large"  # Changed from depth_anything_v2
    depth_invert: bool = False
    depth_normalize: bool = True
    depth_colormap: Optional[str] = "inferno"
    
    # Normals parameters
    normals_method: str = "from_depth"  # 'from_depth' or 'dsine'
    normals_depth_model: str = "da3_mono_large"  # Changed from depth_anything_v2
    normals_space: str = "tangent"  # 'tangent', 'world', 'object'
    normals_flip_y: bool = False
    normals_flip_x: bool = False
    normals_intensity: float = 1.0

    # Crypto/SAM parameters
    crypto_model: str = "sam2_hiera_large"  # 'sam2_hiera_tiny', 'sam2_hiera_small', 'sam2_hiera_base_plus', 'sam2_hiera_large'
    crypto_max_objects: int = 50
    crypto_min_area: int = 500
    crypto_output_mode: str = "id_matte"  # 'id_matte', 'layers', 'both'



@app.post("/api/preprocess")
async def preprocess_image(request: PreprocessRequest):
    """
    Run a preprocessing method on an image
    
    Supported methods:
    - canny: Edge detection
    - openpose: Pose keypoints
    - depth: Depth estimation
    """
    
    try:
        # Resolve input path (handles URL paths from cache)
        input_path = resolve_input_path(request.image_path)
        
        if input_path is None:
            raise HTTPException(status_code=400, detail="No image path provided")
        
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path} (resolved to: {input_path})")
        
        print(f"\n{'='*60}", flush=True)
        print(f"Preprocessing: {request.method}", flush=True)
        print(f"Input (requested): {request.image_path}", flush=True)
        print(f"Input (resolved): {input_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Create generation directory in project cache
        gen_dir = get_generation_output_dir(f"preprocess_{request.method}")
        
        # Copy source image for reference
        import shutil
        shutil.copy(input_path, gen_dir / "source.png")
        
        # Generate output path
        output_path = gen_dir / "processed.png"
        
        # Run preprocessing
        if request.method == "canny":
            result = preprocessor_manager.canny(
                image_path=input_path,
                output_path=output_path,
                low_threshold=request.low_threshold,
                high_threshold=request.high_threshold,
                invert=request.canny_invert,
                blur_kernel=request.blur_kernel,
            )
        
        elif request.method == "openpose":
            result = preprocessor_manager.openpose(
                image_path=input_path,
                output_path=output_path,
                detect_body=request.detect_body,
                detect_hand=request.detect_hand,
                detect_face=request.detect_face,
            )
        
        elif request.method == "depth":
            # Map string to enum
            depth_model_map = {
                # DA3 models (new)
                "da3_mono_large": DepthModel.DA3_MONO_LARGE,
                "da3_metric_large": DepthModel.DA3_METRIC_LARGE,
                "da3_large": DepthModel.DA3_LARGE,
                "da3_giant": DepthModel.DA3_GIANT,
                # V2
                "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                # MiDaS
                "midas_small": DepthModel.MIDAS_SMALL,
                "midas_large": DepthModel.MIDAS_LARGE,
                # ZoeDepth
                "zoedepth": DepthModel.ZOEDEPTH,
                # Backwards compatibility
                "depth_anything_v3": DepthModel.DA3_MONO_LARGE,
            }
            
            depth_model = depth_model_map.get(
                request.depth_model,
                DepthModel.DEPTH_ANYTHING_V2
            )
            
            result = preprocessor_manager.depth(
                image_path=input_path,
                output_path=output_path,
                model=depth_model,
                invert=request.depth_invert,
                normalize=request.depth_normalize,
                colormap=request.depth_colormap,
            )

        elif request.method == "normals":
            # Map strings to enums
            normals_method_map = {
                "from_depth": NormalsMethod.FROM_DEPTH,
                "dsine": NormalsMethod.DSINE,
            }
            depth_model_map = {
                "midas_small": DepthModel.MIDAS_SMALL,
                "midas_large": DepthModel.MIDAS_LARGE,
                "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
                "zoedepth": DepthModel.ZOEDEPTH,
            }
            
            normals_method = normals_method_map.get(
                request.normals_method,
                NormalsMethod.FROM_DEPTH
            )
            depth_model = depth_model_map.get(
                request.normals_depth_model,
                DepthModel.DEPTH_ANYTHING_V2
            )
            
            result = preprocessor_manager.normals(
                image_path=input_path,
                output_path=output_path,
                method=normals_method,
                depth_model=depth_model,
                space=request.normals_space,
                flip_y=request.normals_flip_y,
                flip_x=request.normals_flip_x,
                intensity=request.normals_intensity,
            )
        
        elif request.method == "crypto":
            # Map string to enum
            sam_model_map = {
                "sam2_hiera_tiny": SAMModel.TINY,
                "sam2_hiera_small": SAMModel.SMALL,
                "sam2_hiera_base_plus": SAMModel.BASE,
                "sam2_hiera_large": SAMModel.LARGE,
            }
            
            sam_model = sam_model_map.get(
                request.crypto_model,
                SAMModel.LARGE
            )
            
            result = preprocessor_manager.crypto(
                image_path=input_path,
                output_path=output_path,
                model=sam_model,
                max_objects=request.crypto_max_objects,
                min_area=request.crypto_min_area,
                output_mode=request.crypto_output_mode,
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=f"Preprocessor: {request.method}",
            model=request.method,
            seed=None,
            image_size=(0, 0),
            source_image=str(request.image_path),
            parameters=result.get("parameters", {}),
        )
        
        # Generate thumbnail for preview in history
        output_path = Path(result["output_path"])
        thumbnail_path = gen_dir / "thumbnail.jpg"
        
        try:
            from PIL import Image
            with Image.open(output_path) as img:
                # Create thumbnail maintaining aspect ratio
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, "JPEG", quality=85)
        except Exception as e:
            print(f"[Preprocess] Failed to generate thumbnail: {e}")
        
        # Convert output path to project-relative URL
        output_path = Path(result["output_path"])
        result["url"] = get_project_relative_url(output_path)
        
        # Clear preprocessor cache to free VRAM
        # For demo/testing - remove if you want models to stay cached for fast iteration
        preprocessor_manager.clear_caches([request.method])
        clear_vram()
        
        return result
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        # Clear on error too
        preprocessor_manager.clear_caches([request.method])
        clear_vram()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preprocess/models")
async def get_preprocessor_models():
    """Get available preprocessor models and their info"""
    return {
        "canny": {
            "name": "Canny Edge Detection",
            "description": "Clean edge detection using Canny algorithm",
            "parameters": {
                "low_threshold": {"type": "int", "default": 1, "min": 0, "max": 40},
                "high_threshold": {"type": "int", "default": 20, "min": 0, "max": 40},
                "invert": {"type": "bool", "default": False},
                "blur_kernel": {"type": "int", "default": 3, "min": 0, "max": 15, "step": 2},
            }
        },
        "openpose": {
            "name": "OpenPose",
            "description": "Human pose keypoint detection",
            "parameters": {
                "detect_body": {"type": "bool", "default": True},
                "detect_hand": {"type": "bool", "default": False},
                "detect_face": {"type": "bool", "default": False},
            }
        },
        "depth": {
            "name": "Depth Estimation",
            "description": "Monocular depth estimation for Z-depth AOV",
            "models": {
                "midas_small": "MiDaS Small (Fast)",
                "midas_large": "MiDaS Large (Balanced)",
                "depth_anything_v2": "Depth Anything V2 (SOTA)",
                "depth_anything_v3": "Depth Anything V3 (Latest)",
                "zoedepth": "ZoeDepth (Metric)",
            },
            "parameters": {
                "invert": {"type": "bool", "default": False},
                "normalize": {"type": "bool", "default": True},
                "colormap": {
                    "type": "select",
                    "default": "inferno",
                    "options": [None, "inferno", "viridis", "magma", "plasma", "turbo"]
                },
            }
        },
        "normals": {
            "name": "Normal Map",
            "description": "Surface normal estimation for relighting",
            "methods": {
                "from_depth": "From Depth (Fast, uses depth model)",
                "dsine": "DSINE (Dedicated model, better quality)",
            },
            "depth_models": {
                "midas_small": "MiDaS Small (Fast)",
                "midas_large": "MiDaS Large (Balanced)",
                "depth_anything_v2": "Depth Anything V2 (SOTA)",
                "depth_anything_v3": "Depth Anything V3 (Latest)",
                "zoedepth": "ZoeDepth (Metric)",
            },
            "parameters": {
                "space": {
                    "type": "select",
                    "default": "tangent",
                    "options": ["tangent", "world", "object"]
                },
                "flip_y": {"type": "bool", "default": False, "description": "Flip Y for DirectX convention"},
                "flip_x": {"type": "bool", "default": False},
                "intensity": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "description": "Normal strength (depth-derived only)"},
            }
        },
        "crypto": {
            "name": "Cryptomatte",
            "description": "Instance segmentation for per-object mattes (uses SAM2)",
            "models": {
                "sam2_hiera_tiny": "SAM2 Tiny (Fast, ~39MB)",
                "sam2_hiera_small": "SAM2 Small (Balanced, ~46MB)",
                "sam2_hiera_base_plus": "SAM2 Base+ (Good, ~81MB)",
                "sam2_hiera_large": "SAM2 Large (Best, ~224MB)",
            },
            "parameters": {
                "max_objects": {"type": "int", "default": 50, "min": 1, "max": 200},
                "min_area": {"type": "int", "default": 500, "min": 100, "max": 10000, "description": "Minimum mask size in pixels"},
                "output_mode": {
                    "type": "select",
                    "default": "id_matte",
                    "options": ["id_matte", "layers", "both"],
                    "descriptions": {
                        "id_matte": "Single colored ID visualization",
                        "layers": "Individual mask files per object",
                        "both": "Both outputs"
                    }
                },
            }
        },
    }


@app.post("/api/preprocess/clear-cache")
async def clear_preprocessor_cache(models: Optional[List[str]] = None):
    """
    Clear preprocessor model caches to free VRAM
    
    Args:
        models: List of model types to clear ('depth', 'normals', 'crypto', 'openpose', 'canny')
               If None or empty, clears ALL cached models.
    
    Returns:
        Dict with VRAM stats after clearing
    """
    import torch
    
    try:
        # Clear specified or all caches
        preprocessor_manager.clear_caches(models if models else None)
        
        # Get VRAM stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            allocated = reserved = total = 0
        
        return {
            "success": True,
            "cleared": models if models else ["all"],
            "vram": {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Post-Processing Endpoints
# ============================================================================

@app.post("/api/postprocess/upscale")
async def upscale_image(request: UpscaleRequest):
    """
    Upscale an image using Real-ESRGAN or Lanczos
    
    Supports 2x, 4x, and 8x upscaling
    """
    try:
        print(f"\n{'='*60}")
        print(f"UPSCALE REQUEST")
        print(f"{'='*60}")
        print(f"  Source: {request.source_path}")
        print(f"  Scale: {request.scale}x")
        print(f"  Model: {request.model}")
        print(f"  Denoise: {request.denoise}")
        
        # Resolve input path
        source_path = resolve_input_path(request.source_path)
        if not source_path:
            raise HTTPException(status_code=400, detail=f"Source file not found: {request.source_path}")
        
        # Generate output directory and path
        gen_dir = get_generation_output_dir(f"upscale_{request.scale}x")
        output_path = gen_dir / f"upscaled_{request.scale}x.png"
        
        # Run upscaling
        result = postprocessor_manager.upscale_image(
            input_path=source_path,
            output_path=output_path,
            scale=request.scale,
            model=request.model,
            denoise=request.denoise,
        )
        
        # Get dimensions
        from PIL import Image
        with Image.open(output_path) as img:
            output_width, output_height = img.size
        with Image.open(source_path) as img:
            input_width, input_height = img.size
        
        # Generate thumbnail for preview in history
        thumbnail_path = gen_dir / "thumbnail.jpg"
        try:
            with Image.open(output_path) as img:
                # Create thumbnail maintaining aspect ratio
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                img.save(thumbnail_path, "JPEG", quality=85)
        except Exception as e:
            print(f"[Upscale] Failed to generate thumbnail: {e}")
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=f"Upscale {request.scale}x ({request.model})",
            model=request.model,
            seed=None,
            image_size=(output_width, output_height),
            source_image=str(source_path),
            parameters={
                "scale": request.scale,
                "denoise": request.denoise,
                "method": result.get("method", "unknown"),
            },
        )
        
        # Build URL using project-relative path (IMPORTANT!)
        output_url = get_project_relative_url(output_path)
        
        print(f"Upscaled: {input_width}x{input_height} -> {output_width}x{output_height}")
        print(f"  Output URL: {output_url}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "output_url": output_url,
            "input_size": {"width": input_width, "height": input_height},
            "output_size": {"width": output_width, "height": output_height},
            "scale": request.scale,
            "model": request.model,
            "method": result.get("method", "unknown"),
        }
        
    except Exception as e:
        print(f"Upscaling failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/postprocess/interpolate")
async def interpolate_video(request: InterpolateRequest):
    """
    Interpolate video frames to increase framerate
    
    Uses RIFE for AI-based interpolation with OpenCV blend fallback
    """
    try:
        print(f"\n{'='*60}")
        print(f"INTERPOLATION REQUEST")
        print(f"{'='*60}")
        print(f"  Source: {request.source_path}")
        print(f"  Source FPS: {request.source_fps}")
        print(f"  Target FPS: {request.target_fps}")
        print(f"  Model: {request.model}")
        
        # Resolve input path
        source_path = resolve_input_path(request.source_path)
        if not source_path:
            raise HTTPException(status_code=400, detail=f"Source file not found: {request.source_path}")
        
        # Generate output directory and path
        gen_dir = get_generation_output_dir(f"interpolate_{request.target_fps}fps")
        output_path = gen_dir / f"interpolated_{request.target_fps}fps.mp4"
        
        # Run interpolation
        result = postprocessor_manager.interpolate_video(
            input_path=source_path,
            output_path=output_path,
            source_fps=request.source_fps,
            target_fps=request.target_fps,
            model=request.model,
        )
        
        # Extract thumbnail from interpolated video
        if output_path.exists():
            thumb_path = output_path.with_suffix('.thumb.jpg')
            video_processor = VideoProcessor()
            video_processor.extract_thumbnail(output_path, thumb_path)
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=f"Interpolate {request.source_fps}fps -> {request.target_fps}fps",
            model=request.model,
            seed=None,
            image_size=(0, 0),
            source_image=str(source_path),
            parameters={
                "source_fps": request.source_fps,
                "target_fps": request.target_fps,
                "multiplier": result.get("multiplier", 0),
                "method": result.get("method", "unknown"),
            },
        )
        
        # Build URL using project-relative path (IMPORTANT!)
        output_url = get_project_relative_url(output_path)
        
        multiplier = result.get("multiplier", round(request.target_fps / request.source_fps))
        print(f"ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Interpolated: {request.source_fps}fps -> {request.target_fps}fps ({multiplier}x)")
        print(f"  Output URL: {output_url}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "output_url": output_url,
            "source_fps": request.source_fps,
            "target_fps": request.target_fps,
            "multiplier": multiplier,
            "model": request.model,
            "method": result.get("method", "unknown"),
        }
        
    except Exception as e:
        print(f"Interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/postprocess/capabilities")
async def get_postprocess_capabilities():
    """Get available post-processing capabilities"""
    return postprocessor_manager.get_capabilities()


@app.get("/api/postprocess/models")
async def get_postprocessor_models():
    """Get available post-processor models and their info"""
    return {
        "upscaling": {
            "realesrgan": {
                "name": "Real-ESRGAN",
                "description": "AI-based photo-realistic upscaling",
                "scales": [2, 4, 8],
                "parameters": {
                    "denoise": {"type": "float", "default": 0.5, "min": 0, "max": 1},
                }
            },
            "lanczos": {
                "name": "Lanczos",
                "description": "Fast traditional upscaling (fallback)",
                "scales": [2, 4, 8],
                "parameters": {}
            }
        },
        "interpolation": {
            "rife": {
                "name": "RIFE",
                "description": "Real-Time Intermediate Flow Estimation",
                "source_fps": [8, 12, 16, 24],
                "target_fps": [24, 30, 60],
                "parameters": {}
            }
        }
    }

# ============================================================================
# PreProcessors
# ============================================================================

class PreprocessRequest(BaseModel):
    image_path: str
    method: str  # 'canny', 'openpose', 'depth', 'normals', 'crypto'
    
    # Canny parameters
    low_threshold: int = 100
    high_threshold: int = 200
    canny_invert: bool = False
    blur_kernel: int = 3
    
    # OpenPose parameters
    detect_body: bool = True
    detect_hand: bool = False
    detect_face: bool = False
    
    # Depth parameters
    depth_model: str = "depth_anything_v2"
    depth_invert: bool = False
    depth_normalize: bool = True
    depth_colormap: Optional[str] = "inferno"
    
    # Normals parameters
    normals_method: str = "from_depth"  # 'from_depth' or 'dsine'
    normals_depth_model: str = "depth_anything_v2"
    normals_space: str = "tangent"  # 'tangent', 'world', 'object'
    normals_flip_y: bool = False
    normals_flip_x: bool = False
    normals_intensity: float = 1.0
    
    # Crypto/SAM parameters
    crypto_model: str = "sam2_hiera_large"
    crypto_max_objects: int = 50
    crypto_min_area: int = 500
    crypto_output_mode: str = "id_matte"  # 'id_matte', 'layers', 'both'


@app.post("/api/preprocess")
async def preprocess_image(request: PreprocessRequest):
    """
    Run a preprocessing method on an image
    
    Supported methods:
    - canny: Edge detection
    - openpose: Pose keypoints
    - depth: Depth estimation
    - normals: Surface normal maps
    - crypto: Instance segmentation / cryptomatte
    """
    
    try:
        # Resolve input path (handles URL paths from cache)
        input_path = resolve_input_path(request.image_path)
        
        if input_path is None:
            raise HTTPException(status_code=400, detail="No image path provided")
        
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path} (resolved to: {input_path})")
        
        print(f"\n{'='*60}", flush=True)
        print(f"Preprocessing: {request.method}", flush=True)
        print(f"Input (requested): {request.image_path}", flush=True)
        print(f"Input (resolved): {input_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Create generation directory in project cache
        gen_dir = get_generation_output_dir(f"preprocess_{request.method}")
        
        # Copy source image for reference
        import shutil
        shutil.copy(input_path, gen_dir / "source.png")
        
        # Generate output path
        output_path = gen_dir / "processed.png"
        
        # Run preprocessing based on method
        if request.method == "canny":
            result = preprocessor_manager.canny(
                image_path=input_path,
                output_path=output_path,
                low_threshold=request.low_threshold,
                high_threshold=request.high_threshold,
                invert=request.canny_invert,
                blur_kernel=request.blur_kernel,
            )
        
        elif request.method == "openpose":
            result = preprocessor_manager.openpose(
                image_path=input_path,
                output_path=output_path,
                detect_body=request.detect_body,
                detect_hand=request.detect_hand,
                detect_face=request.detect_face,
            )
        
        elif request.method == "depth":
            # Map string to enum
            depth_model_map = {
                "midas_small": DepthModel.MIDAS_SMALL,
                "midas_large": DepthModel.MIDAS_LARGE,
                "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
                "zoedepth": DepthModel.ZOEDEPTH,
            }
            
            depth_model = depth_model_map.get(
                request.depth_model,
                DepthModel.DEPTH_ANYTHING_V2
            )
            
            result = preprocessor_manager.depth(
                image_path=input_path,
                output_path=output_path,
                model=depth_model,
                invert=request.depth_invert,
                normalize=request.depth_normalize,
                colormap=request.depth_colormap,
            )
        
        elif request.method == "normals":
            # Map strings to enums
            normals_method_map = {
                "from_depth": NormalsMethod.FROM_DEPTH,
                "dsine": NormalsMethod.DSINE,
            }
            depth_model_map = {
                "midas_small": DepthModel.MIDAS_SMALL,
                "midas_large": DepthModel.MIDAS_LARGE,
                "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
                "zoedepth": DepthModel.ZOEDEPTH,
            }
            
            normals_method = normals_method_map.get(
                request.normals_method,
                NormalsMethod.FROM_DEPTH
            )
            depth_model = depth_model_map.get(
                request.normals_depth_model,
                DepthModel.DEPTH_ANYTHING_V2
            )
            
            result = preprocessor_manager.normals(
                image_path=input_path,
                output_path=output_path,
                method=normals_method,
                depth_model=depth_model,
                space=request.normals_space,
                flip_y=request.normals_flip_y,
                flip_x=request.normals_flip_x,
                intensity=request.normals_intensity,
            )
        
        elif request.method == "crypto":
            # Map string to enum
            sam_model_map = {
                "sam2_hiera_tiny": SAMModel.TINY,
                "sam2_hiera_small": SAMModel.SMALL,
                "sam2_hiera_base_plus": SAMModel.BASE,
                "sam2_hiera_large": SAMModel.LARGE,
            }
            
            sam_model = sam_model_map.get(
                request.crypto_model,
                SAMModel.LARGE
            )
            
            result = preprocessor_manager.crypto(
                image_path=input_path,
                output_path=output_path,
                model=sam_model,
                max_objects=request.crypto_max_objects,
                min_area=request.crypto_min_area,
                output_mode=request.crypto_output_mode,
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=f"Preprocessor: {request.method}",
            model=request.method,
            seed=None,
            image_size=(0, 0),
            source_image=str(request.image_path),
            parameters=result.get("parameters", {}),
        )
        
        # Convert output path to project-relative URL
        output_path = Path(result["output_path"])
        result["url"] = get_project_relative_url(output_path)
        
        # Clear preprocessor cache to free VRAM
        preprocessor_manager.clear_caches([request.method])
        clear_vram()
        
        return result
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        # Clear on error too
        preprocessor_manager.clear_caches([request.method])
        clear_vram()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preprocess/models")
async def get_preprocessor_models():
    """Get available preprocessor models and their info"""
    return {
        "canny": {
            "name": "Canny Edge Detection",
            "description": "Clean edge detection using Canny algorithm",
            "parameters": {
                "low_threshold": {"type": "int", "default": 100, "min": 0, "max": 500},
                "high_threshold": {"type": "int", "default": 200, "min": 0, "max": 500},
                "invert": {"type": "bool", "default": False},
                "blur_kernel": {"type": "int", "default": 3, "min": 0, "max": 15, "step": 2},
            }
        },
        "openpose": {
            "name": "OpenPose",
            "description": "Human pose keypoint detection",
            "parameters": {
                "detect_body": {"type": "bool", "default": True},
                "detect_hand": {"type": "bool", "default": False},
                "detect_face": {"type": "bool", "default": False},
            }
        },
        "depth": {
            "name": "Depth Estimation",
            "description": "Monocular depth estimation for Z-depth AOV",
            "models": {
                "midas_small": "MiDaS Small (Fast)",
                "midas_large": "MiDaS Large (Balanced)",
                "depth_anything_v2": "Depth Anything V2 (SOTA)",
                "depth_anything_v3": "Depth Anything V3 (Latest)",
                "zoedepth": "ZoeDepth (Metric)",
            },
            "parameters": {
                "invert": {"type": "bool", "default": False},
                "normalize": {"type": "bool", "default": True},
                "colormap": {
                    "type": "select",
                    "default": "inferno",
                    "options": [None, "inferno", "viridis", "magma", "plasma", "turbo"]
                },
            }
        },
        "normals": {
            "name": "Normal Map",
            "description": "Surface normal estimation for relighting",
            "methods": {
                "from_depth": "From Depth (Fast, uses depth model)",
                "dsine": "DSINE (Dedicated model, better quality)",
            },
            "depth_models": {
                "midas_small": "MiDaS Small (Fast)",
                "midas_large": "MiDaS Large (Balanced)",
                "depth_anything_v2": "Depth Anything V2 (SOTA)",
                "depth_anything_v3": "Depth Anything V3 (Latest)",
                "zoedepth": "ZoeDepth (Metric)",
            },
            "parameters": {
                "space": {
                    "type": "select",
                    "default": "tangent",
                    "options": ["tangent", "world", "object"]
                },
                "flip_y": {"type": "bool", "default": False, "description": "Flip Y for DirectX convention"},
                "flip_x": {"type": "bool", "default": False},
                "intensity": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "description": "Normal strength (depth-derived only)"},
            }
        },
        "crypto": {
            "name": "Cryptomatte",
            "description": "Instance segmentation for per-object mattes (uses SAM2)",
            "models": {
                "sam2_hiera_tiny": "SAM2 Tiny (Fast, ~39MB)",
                "sam2_hiera_small": "SAM2 Small (Balanced, ~46MB)",
                "sam2_hiera_base_plus": "SAM2 Base+ (Good, ~81MB)",
                "sam2_hiera_large": "SAM2 Large (Best, ~224MB)",
            },
            "parameters": {
                "max_objects": {"type": "int", "default": 50, "min": 1, "max": 200},
                "min_area": {"type": "int", "default": 500, "min": 100, "max": 10000, "description": "Minimum mask size in pixels"},
                "output_mode": {
                    "type": "select",
                    "default": "id_matte",
                    "options": ["id_matte", "layers", "both"],
                    "descriptions": {
                        "id_matte": "Single colored ID visualization",
                        "layers": "Individual mask files per object",
                        "both": "Both outputs"
                    }
                },
            }
        },
    }


# ============================================================================
# Layers (Batch AOV Generation)
# ============================================================================

class LayersRequest(BaseModel):
    """Request to generate multiple AOV layers from a single image"""
    image_path: str
    
    # Which layers to generate
    layers: Dict[str, bool] = {
        "depth": True,
        "normals": True,
        "crypto": False,
    }
    
    # Depth settings
    depth_model: str = "depth_anything_v2"
    depth_invert: bool = False
    depth_normalize: bool = True
    depth_colormap: Optional[str] = "inferno"
    
    # Normals settings
    normals_method: str = "from_depth"
    normals_depth_model: str = "depth_anything_v2"
    normals_space: str = "tangent"
    normals_flip_y: bool = False
    normals_intensity: float = 1.0
    
    # Crypto settings
    crypto_model: str = "sam2_hiera_large"
    crypto_max_objects: int = 50
    crypto_min_area: int = 500


@app.post("/api/layers/generate")
async def generate_layers(request: LayersRequest):
    """
    Generate multiple AOV layers from a single source image
    
    Returns all requested layers in one call, useful for the Layers tab.
    """
    try:
        # Resolve input path
        input_path = resolve_input_path(request.image_path)
        
        if input_path is None:
            raise HTTPException(status_code=400, detail="No image path provided")
        
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
        
        log.header("LAYERS GENERATION")
        log.info("Layers", f"Source: {input_path}")
        log.info("Layers", f"Requested: {[k for k, v in request.layers.items() if v]}")
        
        # Create generation directory
        gen_dir = get_generation_output_dir("layers")
        
        # Copy source image
        import shutil
        source_copy = gen_dir / "source.png"
        shutil.copy(input_path, source_copy)
        
        results = {
            "success": True,
            "source": get_project_relative_url(source_copy),
            "layers": {},
            "errors": {},
        }
        
        # Generate depth
        if request.layers.get("depth"):
            try:
                log.info("Layers", "Generating depth...")
                
                depth_model_map = {
                    "midas_small": DepthModel.MIDAS_SMALL,
                    "midas_large": DepthModel.MIDAS_LARGE,
                    "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                    "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
                    "zoedepth": DepthModel.ZOEDEPTH,
                }
                
                depth_result = preprocessor_manager.depth(
                    image_path=input_path,
                    output_path=gen_dir / "depth.png",
                    model=depth_model_map.get(request.depth_model, DepthModel.DEPTH_ANYTHING_V2),
                    invert=request.depth_invert,
                    normalize=request.depth_normalize,
                    colormap=request.depth_colormap,
                )
                
                results["layers"]["depth"] = {
                    "url": get_project_relative_url(Path(depth_result["output_path"])),
                    "path": depth_result["output_path"],
                    "model": depth_result.get("model", request.depth_model),
                }
                log.success("Layers", "Depth complete")
                
            except Exception as e:
                log.error("Layers", f"Depth failed: {e}")
                results["errors"]["depth"] = str(e)
        
        # Generate normals
        if request.layers.get("normals"):
            try:
                log.info("Layers", "Generating normals...")
                
                normals_method_map = {
                    "from_depth": NormalsMethod.FROM_DEPTH,
                    "dsine": NormalsMethod.DSINE,
                }
                depth_model_map = {
                    "midas_small": DepthModel.MIDAS_SMALL,
                    "midas_large": DepthModel.MIDAS_LARGE,
                    "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
                    "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
                    "zoedepth": DepthModel.ZOEDEPTH,
                }
                
                normals_result = preprocessor_manager.normals(
                    image_path=input_path,
                    output_path=gen_dir / "normals.png",
                    method=normals_method_map.get(request.normals_method, NormalsMethod.FROM_DEPTH),
                    depth_model=depth_model_map.get(request.normals_depth_model, DepthModel.DEPTH_ANYTHING_V2),
                    space=request.normals_space,
                    flip_y=request.normals_flip_y,
                    intensity=request.normals_intensity,
                )
                
                results["layers"]["normals"] = {
                    "url": get_project_relative_url(Path(normals_result["output_path"])),
                    "path": normals_result["output_path"],
                    "method": normals_result.get("estimation", request.normals_method),
                }
                log.success("Layers", "Normals complete")
                
            except Exception as e:
                log.error("Layers", f"Normals failed: {e}")
                results["errors"]["normals"] = str(e)
        
        # Generate cryptomatte
        if request.layers.get("crypto"):
            try:
                log.info("Layers", "Generating cryptomatte...")
                
                sam_model_map = {
                    "sam2_hiera_tiny": SAMModel.TINY,
                    "sam2_hiera_small": SAMModel.SMALL,
                    "sam2_hiera_base_plus": SAMModel.BASE,
                    "sam2_hiera_large": SAMModel.LARGE,
                }
                
                crypto_result = preprocessor_manager.crypto(
                    image_path=input_path,
                    output_path=gen_dir / "crypto.png",
                    model=sam_model_map.get(request.crypto_model, SAMModel.LARGE),
                    max_objects=request.crypto_max_objects,
                    min_area=request.crypto_min_area,
                    output_mode="id_matte",
                )
                
                results["layers"]["crypto"] = {
                    "url": get_project_relative_url(Path(crypto_result["output_path"])),
                    "path": crypto_result["output_path"],
                    "num_objects": crypto_result.get("num_objects", 0),
                }
                log.success("Layers", f"Cryptomatte complete ({crypto_result.get('num_objects', 0)} objects)")
                
            except Exception as e:
                log.error("Layers", f"Cryptomatte failed: {e}")
                results["errors"]["crypto"] = str(e)
        
        # Save metadata
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=f"Layers: {', '.join(results['layers'].keys())}",
            model="layers",
            seed=None,
            image_size=(0, 0),
            source_image=str(request.image_path),
            parameters={
                "layers": request.layers,
                "generated": list(results["layers"].keys()),
                "errors": list(results["errors"].keys()),
            },
        )
        
        log.success("Layers", f"Generated {len(results['layers'])} layers")
        
        return results
        
    except Exception as e:
        log.exception("Layers", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layers/capabilities")
async def get_layers_capabilities():
    """Get available layer types and their requirements"""
    return {
        "layers": {
            "depth": {
                "name": "Depth (Z)",
                "description": "Distance from camera for DOF/fog effects",
                "available": True,
                "requirements": ["Depth Anything V2 or MiDaS"],
            },
            "normals": {
                "name": "Normals",
                "description": "Surface orientation for relighting",
                "available": True,
                "requirements": ["Depth model (for from_depth) or DSINE"],
            },
            "crypto": {
                "name": "Cryptomatte",
                "description": "Per-object ID masks for selection",
                "available": True,
                "requirements": ["SAM2 model checkpoint"],
            },
        },
        "export_formats": ["png", "exr"],
    }

# ============================================================================
# Export Endpoints
# ============================================================================

class ExportEXRRequest(BaseModel):
    """Request to export layers to EXR"""
    layers: Dict[str, Optional[str]]  # layer_name -> image_path (can be None)
    
    # EXR settings
    bit_depth: int = 32  # 16 or 32
    compression: str = "ZIP"
    color_space: str = "Linear"  # 'Linear' or 'sRGB'
    
    # Export mode
    multi_layer: bool = True  # Single multi-layer EXR
    single_files: bool = False  # Also export individual EXRs
    
    # Output naming and location
    filename: Optional[str] = None  # Custom filename (without extension)
    export_path: Optional[str] = None  # Custom save directory (empty = use project cache)

class ExportEXRSequenceRequest(BaseModel):
    """Request to export video layers to EXR sequence"""
    layers: Dict[str, Optional[str]]  # layer_name -> video_path (MP4) or None
    
    # EXR settings
    bit_depth: int = 32  # 16 or 32
    compression: str = "ZIP"
    color_space: str = "Linear"  # 'Linear' or 'sRGB'
    
    # Export mode
    multi_layer: bool = True  # Each frame is multi-layer EXR
    single_files: bool = False  # Also export separate sequences per layer
    
    # Output naming and location
    filename: str = "export"  # Base filename for sequence
    export_path: Optional[str] = None  # Custom save directory (empty = use project cache)
    
    # Sequence-specific
    start_frame: int = 1  # First frame number
    filename_pattern: Optional[str] = None  # e.g. "{name}.{frame:04d}.exr"


@app.post("/api/export/exr")
async def export_to_exr(request: ExportEXRRequest):
    """
    Export AOV layers to EXR file(s)
    
    Creates:
    - Multi-layer EXR with all AOVs (if multi_layer=True)
    - Individual EXR files per layer (if single_files=True)
    
    If export_path is provided, saves directly to that location.
    Otherwise saves to project cache.
    """
    try:
        log.header("EXR EXPORT")
        log.info("Export", f"Layers: {list(request.layers.keys())}")
        log.info("Export", f"Bit Depth: {request.bit_depth}")
        log.info("Export", f"Compression: {request.compression}")
        
        # Import exporter
        from core.exr_exporter import EXRExporter
        
        # Determine output location
        custom_export = bool(request.export_path)
        
        if custom_export:
            # Use custom export path
            export_dir = Path(request.export_path)
            if not export_dir.exists():
                export_dir.mkdir(parents=True, exist_ok=True)
            gen_dir = export_dir  # For metadata
            log.info("Export", f"Custom export path: {export_dir}")
        else:
            # Use project cache
            gen_dir = get_generation_output_dir("export_exr")
            export_dir = gen_dir
            log.info("Export", f"Cache path: {gen_dir}")
        
        # Resolve layer paths
        resolved_layers = {}
        for layer_name, layer_path in request.layers.items():
            if layer_path is None:
                continue
            
            resolved_path = resolve_input_path(layer_path)
            if resolved_path and resolved_path.exists():
                resolved_layers[layer_name] = str(resolved_path)
                log.info("Export", f"  {layer_name}: {resolved_path.name}")
            else:
                log.warning("Export", f"  {layer_name}: not found ({layer_path})")
        
        if not resolved_layers:
            raise HTTPException(status_code=400, detail="No valid layers to export")
        
        exporter = EXRExporter()
        results = {"success": True, "outputs": {}}
        
        # Determine filename
        base_filename = request.filename or "export"
        
        # Export multi-layer EXR
        if request.multi_layer:
            output_path = export_dir / f"{base_filename}.exr"
            
            result = exporter.export_multilayer(
                layers=resolved_layers,
                output_path=output_path,
                bit_depth=request.bit_depth,
                compression=request.compression,
                linear=(request.color_space == "Linear"),
            )
            
            result["url"] = get_project_relative_url(output_path) if not custom_export else None
            result["saved_path"] = str(output_path)
            results["multi_layer"] = result
            log.success("Export", f"Multi-layer EXR: {output_path}")
        
        # Export single-layer EXRs
        if request.single_files:
            if custom_export:
                singles_dir = export_dir
            else:
                singles_dir = gen_dir / "single_layers"
                singles_dir.mkdir(exist_ok=True)
            
            single_results = exporter.export_single_layers(
                layers=resolved_layers,
                output_dir=singles_dir,
                bit_depth=request.bit_depth,
                compression=request.compression,
                linear=(request.color_space == "Linear"),
                filename_prefix=base_filename + "_" if custom_export else "",
            )
            
            # Add URLs to results
            for layer_name, layer_result in single_results.items():
                if not custom_export:
                    layer_result["url"] = get_project_relative_url(Path(layer_result["output_path"]))
                layer_result["saved_path"] = layer_result["output_path"]
            
            results["single_layers"] = single_results
            log.success("Export", f"Single-layer EXRs: {list(single_results.keys())}")
        
        # Add summary info
        results["saved_path"] = str(export_dir / f"{base_filename}.exr") if request.multi_layer else str(export_dir)
        results["custom_export"] = custom_export
        
        # Save metadata (only in project cache)
        if not custom_export:
            save_generation_metadata(
                gen_dir=gen_dir,
                prompt=f"EXR Export ({len(resolved_layers)} layers)",
                model="exr_export",
                seed=None,
                image_size=(0, 0),
                source_image=None,
                parameters={
                    "layers": list(resolved_layers.keys()),
                    "bit_depth": request.bit_depth,
                    "compression": request.compression,
                    "color_space": request.color_space,
                    "filename": base_filename,
                },
            )
        
        log.success("Export", "EXR export complete")
        
        return results
        
    except Exception as e:
        log.exception("Export", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export/exr/sequence")
async def export_video_to_exr_sequence(request: ExportEXRSequenceRequest):
    """
    Export video AOV layers to EXR image sequence
    
    Takes video files (MP4) for each layer and exports as frame-by-frame
    multi-layer EXR sequence.
    
    Output: {filename}.0001.exr, {filename}.0002.exr, ...
    Each frame contains all AOVs (beauty RGB, depth Z, normals N.XYZ, crypto)
    """
    try:
        log.header("EXR SEQUENCE EXPORT")
        log.info("SeqExport", f"Layers: {list(k for k, v in request.layers.items() if v)}")
        log.info("SeqExport", f"Bit Depth: {request.bit_depth}")
        log.info("SeqExport", f"Start Frame: {request.start_frame}")
        
        # Import exporter
        from core.exr_exporter import EXRExporter
        
        # Determine output location
        custom_export = bool(request.export_path)
        
        if custom_export:
            export_dir = Path(request.export_path)
            if not export_dir.exists():
                export_dir.mkdir(parents=True, exist_ok=True)
            gen_dir = export_dir
            log.info("SeqExport", f"Custom export path: {export_dir}")
        else:
            gen_dir = get_generation_output_dir("export_exr_seq")
            export_dir = gen_dir
            log.info("SeqExport", f"Cache path: {gen_dir}")
        
        # Resolve layer paths (video files)
        resolved_layers = {}
        for layer_name, layer_path in request.layers.items():
            if layer_path is None:
                continue
            
            resolved_path = resolve_input_path(layer_path)
            if resolved_path and resolved_path.exists():
                resolved_layers[layer_name] = str(resolved_path)
                log.info("SeqExport", f"  {layer_name}: {resolved_path.name}")
            else:
                log.warning("SeqExport", f"  {layer_name}: not found ({layer_path})")
        
        if not resolved_layers:
            raise HTTPException(status_code=400, detail="No valid video layers to export")
        
        exporter = EXRExporter()
        
        # Build filename pattern
        base_filename = request.filename or "export"
        filename_pattern = request.filename_pattern or f"{base_filename}.{{frame:04d}}.exr"
        
        # Export video sequence
        if request.multi_layer:
            result = exporter.export_video_sequence(
                layers=resolved_layers,
                output_dir=export_dir,
                filename_pattern=filename_pattern,
                bit_depth=request.bit_depth,
                compression=request.compression,
                linear=(request.color_space == "Linear"),
                start_frame=request.start_frame,
            )
            
            log.success("SeqExport", f"Exported {result['frame_count']} frames")
            
            results = {
                "success": True,
                "output_dir": str(export_dir),
                "frame_count": result["frame_count"],
                "start_frame": result["start_frame"],
                "end_frame": result["end_frame"],
                "total_size": result["total_size"],
                "total_size_mb": result["total_size_mb"],
                "filename_pattern": filename_pattern,
                "layers_included": result["layers_included"],
                "custom_export": custom_export,
            }
            
            # Add URL for cache exports
            if not custom_export:
                results["url"] = get_project_relative_url(export_dir)
            
        else:
            # Single files mode - export separate sequence per layer
            results = {
                "success": True,
                "output_dir": str(export_dir),
                "layers": {},
                "custom_export": custom_export,
            }
            
            total_frames = 0
            for layer_name, layer_path in resolved_layers.items():
                layer_pattern = f"{base_filename}_{layer_name}.{{frame:04d}}.exr"
                
                layer_result = exporter.export_video_sequence(
                    layers={layer_name: layer_path},
                    output_dir=export_dir,
                    filename_pattern=layer_pattern,
                    bit_depth=request.bit_depth,
                    compression=request.compression,
                    linear=(request.color_space == "Linear") if layer_name == "beauty" else False,
                    start_frame=request.start_frame,
                )
                
                results["layers"][layer_name] = {
                    "frame_count": layer_result["frame_count"],
                    "pattern": layer_pattern,
                }
                total_frames = max(total_frames, layer_result["frame_count"])
            
            results["frame_count"] = total_frames
        
        # Save metadata (only in project cache)
        if not custom_export:
            save_generation_metadata(
                gen_dir=gen_dir,
                prompt=f"EXR Sequence Export ({len(resolved_layers)} layers)",
                model="exr_sequence_export",
                seed=None,
                image_size=(0, 0),
                source_image=None,
                parameters={
                    "layers": list(resolved_layers.keys()),
                    "bit_depth": request.bit_depth,
                    "compression": request.compression,
                    "color_space": request.color_space,
                    "filename": base_filename,
                    "frame_count": results.get("frame_count", 0),
                    "start_frame": request.start_frame,
                },
            )
        
        log.success("SeqExport", "EXR sequence export complete")
        
        return results
        
    except Exception as e:
        log.exception("SeqExport", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/capabilities")
async def get_export_capabilities():
    """Get available export formats and options"""
    
    # Check for OpenEXR
    try:
        import OpenEXR
        has_openexr = True
    except ImportError:
        has_openexr = False
    
    return {
        "exr": {
            "available": has_openexr,
            "bit_depths": [16, 32],
            "compressions": [
                {"value": "ZIP", "name": "ZIP (Lossless, Recommended)"},
                {"value": "PIZ", "name": "PIZ (Lossless, Wavelet)"},
                {"value": "PXR24", "name": "PXR24 (Lossy, 24-bit)"},
                {"value": "B44", "name": "B44 (Lossy, Fast)"},
                {"value": "DWAA", "name": "DWAA (Lossy, Small)"},
                {"value": "NONE", "name": "None (Uncompressed)"},
            ],
            "color_spaces": ["Linear", "sRGB"],
            "missing_dependency": None if has_openexr else "pip install OpenEXR --break-system-packages",
        },
        "png": {
            "available": True,
            "bit_depths": [8, 16],
        },
        "jpg": {
            "available": True,
            "quality_range": [50, 100],
        },
    }

# ============================================================================
# Video Endpoints
# ============================================================================

# File browser routes (native dialogs)
setup_file_browser_routes(app, log)

# Video processing routes
setup_video_routes(
    app=app,
    preprocessor_manager=preprocessor_manager,
    postprocessor_manager=postprocessor_manager,
    resolve_input_path=resolve_input_path,
    get_generation_output_dir=get_generation_output_dir,
    get_project_relative_url=get_project_relative_url,
    save_generation_metadata=save_generation_metadata,
    DepthModel=DepthModel,
    NormalsMethod=NormalsMethod,
    SAMModel=SAMModel,
    log=log,
    clear_vram=clear_vram,                              
    cleanup_failed_generation=cleanup_failed_generation 
)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("FUK Generation Web Server")
    print("="*60)
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")