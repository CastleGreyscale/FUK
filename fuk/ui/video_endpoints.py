# video_endpoints.py
"""
Video Processing API Endpoints for FUK Web Server

Add these endpoints to fuk_web_server.py for video support:
- /api/preprocess/video - Apply preprocessor to video
- /api/postprocess/upscale/video - Upscale video frame by frame
- /api/layers/video/generate - Generate AOV layers for video

Import and register these routes in the main server file.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
import json
import time

# Shared video utilities (single source of truth)
from core.video_utils import (
    get_video_info as _get_video_info,
    extract_thumbnail,
)

# Create router for video endpoints
router = APIRouter(prefix="/api", tags=["video"])

# Progress tracking for async operations
video_progress: Dict[str, Dict[str, Any]] = {}


class VideoOutputMode(str, Enum):
    MP4 = "mp4"
    SEQUENCE = "sequence"


# ============================================================================
# Request Models
# ============================================================================

class VideoPreprocessRequest(BaseModel):
    """Request to preprocess a video"""
    video_path: str
    method: str  # 'canny', 'depth', 'normals', 'openpose', 'crypto'
    output_mode: VideoOutputMode = VideoOutputMode.MP4
    
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
    normals_method: str = "from_depth"
    normals_depth_model: str = "depth_anything_v2"
    normals_space: str = "tangent"
    normals_flip_y: bool = False
    normals_flip_x: bool = False
    normals_intensity: float = 1.0
    
    # Crypto parameters
    crypto_model: str = "sam2_hiera_large"
    crypto_max_objects: int = 50
    crypto_min_area: int = 500


class VideoUpscaleRequest(BaseModel):
    """Request to upscale a video"""
    source_path: str
    scale: int = 4  # 2, 4, or 8
    model: str = "realesrgan"
    denoise: float = 0.5
    output_mode: VideoOutputMode = VideoOutputMode.MP4


class VideoLayersRequest(BaseModel):
    """Request to generate layers from a video"""
    video_path: str
    
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


# ============================================================================
# Helper Functions (to be imported from main server)
# ============================================================================

def setup_video_routes(
    app,
    preprocessor_manager,
    postprocessor_manager,
    resolve_input_path,
    get_generation_output_dir,
    get_project_relative_url,
    save_generation_metadata,
    DepthModel,
    NormalsMethod,
    SAMModel,
    log,
    clear_vram,            
    cleanup_failed_generation
):
    """
    Setup video routes with access to server resources
    
    Call this from the main server file to register video endpoints.
    """
    
    from core.video_processor import VideoProcessor, OutputMode
    from core.preprocessors.depth import DepthPreprocessor
    from core.preprocessors.crypto import CryptoPreprocessor
    
    video_processor = VideoProcessor()
    
    # ==================================================================
    # Shared model string -> enum maps (defined once, used everywhere)
    # ==================================================================
    
    DEPTH_MODEL_MAP = {
        "midas_small": DepthModel.MIDAS_SMALL,
        "midas_large": DepthModel.MIDAS_LARGE,
        "depth_anything_v2": DepthModel.DEPTH_ANYTHING_V2,
        "depth_anything_v3": DepthModel.DEPTH_ANYTHING_V3,
        "da3_mono_large": DepthModel.DA3_MONO_LARGE,
        "da3_metric_large": DepthModel.DA3_METRIC_LARGE,
        "da3_large": DepthModel.DA3_LARGE,
        "da3_giant": DepthModel.DA3_GIANT,
        "zoedepth": DepthModel.ZOEDEPTH,
    }
    
    SAM_MODEL_MAP = {
        "sam2_hiera_tiny": SAMModel.TINY,
        "sam2_hiera_small": SAMModel.SMALL,
        "sam2_hiera_base_plus": SAMModel.BASE,
        "sam2_hiera_large": SAMModel.LARGE,
    }
    
    NORMALS_METHOD_MAP = {
        "from_depth": NormalsMethod.FROM_DEPTH,
        "dsine": NormalsMethod.DSINE,
    }
    
    # ==================================================================
    # Response Builders
    # ==================================================================
    
    def _generate_thumbnail(output_path: Path) -> Optional[Path]:
        """Generate thumbnail for MP4 output, returns thumb path or None"""
        if not output_path.exists():
            return None
        thumb_path = output_path.with_suffix('.thumb.jpg')
        if thumb_path.exists():
            return thumb_path
        try:
            result = extract_thumbnail(output_path, thumb_path)
            if result:
                log.info("VideoEndpoints", f"Generated thumbnail: {thumb_path.name}")
                return result
        except Exception as e:
            log.error("VideoEndpoints", f"Failed to generate thumbnail: {e}")
        return None
    
    def build_sequence_response(
        output_path: Path,
        result: Dict[str, Any],
        gen_dir: Path
    ) -> Dict[str, Any]:
        """
        Build response data for sequence output mode.
        
        Includes:
        - output_url: URL to the sequence directory
        - preview_url: URL to the first frame (for display)
        - frames: List of frame filenames
        """
        output_url = get_project_relative_url(output_path)
        
        # Get first frame for preview
        first_frame = result.get("first_frame")
        preview_url = None
        frames = result.get("frames", [])
        
        if first_frame:
            # Build preview URL to the first frame
            first_frame_path = output_path / first_frame
            preview_url = get_project_relative_url(first_frame_path)
        
        log.info("VideoResponse", f"Sequence output: {output_path}")
        log.info("VideoResponse", f"  Preview URL: {preview_url}")
        log.info("VideoResponse", f"  Frame count: {len(frames)}")
        
        return {
            "output_url": output_url,
            "preview_url": preview_url,
            "frames": frames,
            "is_sequence": True,
        }
    
    def build_video_response(
        output_path: Path,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build response data for MP4 output mode."""
        output_url = get_project_relative_url(output_path)
        
        # Generate thumbnail
        thumbnail_url = None
        thumb_path = _generate_thumbnail(output_path)
        if thumb_path:
            thumbnail_url = get_project_relative_url(thumb_path)
        
        response = {
            "output_url": output_url,
            "preview_url": output_url,  # Video can be its own preview
            "is_sequence": False,
        }
        
        if thumbnail_url:
            response["thumbnail_url"] = thumbnail_url
        
        return response
    
    def _build_response(output_path: Path, result: Dict, output_mode, gen_dir: Path) -> Dict:
        """Build response based on output mode (convenience wrapper)"""
        if output_mode == OutputMode.SEQUENCE:
            return build_sequence_response(output_path, result, gen_dir)
        else:
            return build_video_response(output_path, result)
    
    # ========================================================================
    # Video Preprocess Endpoint
    # ========================================================================
    
    @app.post("/api/preprocess/video")
    async def preprocess_video(request: VideoPreprocessRequest, background_tasks: BackgroundTasks):
        """
        Apply preprocessing to a video frame-by-frame
        
        Supports: canny, depth, normals, openpose, crypto
        
        Depth and crypto use their unified preprocessors which handle
        batch inference / temporal tracking internally.
        Canny, normals, openpose go through frame-by-frame VideoProcessor.
        """
        gen_dir = None
        
        try:
            # Resolve input path
            input_path = resolve_input_path(request.video_path)
            
            if input_path is None:
                raise HTTPException(status_code=400, detail="No video path provided")
            
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")
            
            log.header(f"VIDEO PREPROCESS: {request.method}")
            log.info("VideoPreprocess", f"Input: {input_path}")
            log.info("VideoPreprocess", f"Method: {request.method}")
            log.info("VideoPreprocess", f"Output Mode: {request.output_mode}")
            
            # Create output directory
            gen_dir = get_generation_output_dir(f"preprocess_video_{request.method}")
            
            output_mode = OutputMode.SEQUENCE if request.output_mode == "sequence" else OutputMode.MP4
            
            if output_mode == OutputMode.MP4:
                output_path = gen_dir / f"{request.method}.mp4"
            else:
                output_path = gen_dir / request.method
                # Don't create yet - let processor handle it
            
            log.info("VideoPreprocess", f"Output path: {output_path}")
            
            # Get video info
            video_info = _get_video_info(input_path)
            
            # Build processor kwargs based on method
            processor_kwargs = {}
            
            # ---- CANNY (frame-by-frame) ----
            if request.method == "canny":
                def frame_processor(inp, out):
                    return preprocessor_manager.canny(
                        image_path=inp,
                        output_path=out,
                        low_threshold=request.low_threshold,
                        high_threshold=request.high_threshold,
                        invert=request.canny_invert,
                        blur_kernel=request.blur_kernel,
                        exact_output=True,
                    )
                processor_kwargs = {
                    "low_threshold": request.low_threshold,
                    "high_threshold": request.high_threshold,
                }
                batch_processed = False
                    
            # ---- DEPTH (unified — handles batch vs frame-by-frame internally) ----
            elif request.method == "depth":
                depth_model = DEPTH_MODEL_MAP.get(request.depth_model, DepthModel.DEPTH_ANYTHING_V2)
                
                log.info("VideoPreprocess", f"Depth model: {depth_model.value}")
                
                # The unified DepthPreprocessor handles everything:
                #   DA3 models -> batch inference with global normalization
                #   Other models -> frame-by-frame via base class
                depth_processor = DepthPreprocessor(model_type=depth_model)
                
                start_time = time.time()
                
                result = depth_processor.process_video(
                    video_path=input_path,
                    output_path=output_path,
                    output_mode="mp4" if output_mode == OutputMode.MP4 else "sequence",
                    invert=request.depth_invert,
                    normalize=request.depth_normalize,
                    colormap=request.depth_colormap,
                    temporal_smooth=5,
                )
                
                elapsed = time.time() - start_time
                log.timing("VideoPreprocess", start_time, f"Complete - {result.get('frame_count', 0)} frames")
                
                # Thumbnail for MP4
                if output_mode == OutputMode.MP4:
                    _generate_thumbnail(output_path)
                
                url_data = _build_response(output_path, result, output_mode, gen_dir)
                processor_kwargs = {"depth_model": request.depth_model}
                
                # Unload model to free VRAM
                depth_processor.unload()
                
                batch_processed = True
                
            # ---- NORMALS (frame-by-frame) ----
            elif request.method == "normals":
                normals_method = NORMALS_METHOD_MAP.get(request.normals_method, NormalsMethod.FROM_DEPTH)
                depth_model = DEPTH_MODEL_MAP.get(request.normals_depth_model, DepthModel.DEPTH_ANYTHING_V2)
                
                def frame_processor(inp, out):
                    return preprocessor_manager.normals(
                        image_path=inp,
                        output_path=out,
                        method=normals_method,
                        depth_model=depth_model,
                        space=request.normals_space,
                        flip_y=request.normals_flip_y,
                        flip_x=request.normals_flip_x,
                        intensity=request.normals_intensity,
                        exact_output=True,
                    )
                processor_kwargs = {"normals_method": request.normals_method}
                batch_processed = False
                    
            # ---- OPENPOSE (frame-by-frame) ----
            elif request.method == "openpose":
                def frame_processor(inp, out):
                    return preprocessor_manager.openpose(
                        image_path=inp,
                        output_path=out,
                        detect_body=request.detect_body,
                        detect_hand=request.detect_hand,
                        detect_face=request.detect_face,
                        exact_output=True,
                    )
                processor_kwargs = {
                    "detect_body": request.detect_body,
                    "detect_hand": request.detect_hand,
                    "detect_face": request.detect_face,
                }
                batch_processed = False
                    
            # ---- CRYPTO (unified — temporal tracking via SAM2VideoPredictor) ----
            elif request.method == "crypto":
                sam_model = SAM_MODEL_MAP.get(request.crypto_model, SAMModel.LARGE)
                
                log.info("VideoPreprocess", f"Crypto model: {sam_model.value} (video tracking)")
                
                # The unified CryptoPreprocessor handles everything:
                #   process_video() -> SAM2VideoPredictor for temporal consistency
                crypto_processor = CryptoPreprocessor(model_size=sam_model)
                
                start_time = time.time()
                
                result = crypto_processor.process_video(
                    video_path=input_path,
                    output_path=output_path,
                    output_mode="mp4" if output_mode == OutputMode.MP4 else "sequence",
                    max_objects=request.crypto_max_objects,
                    min_area=request.crypto_min_area,
                )
                
                elapsed = time.time() - start_time
                log.timing("VideoPreprocess", start_time, f"Complete (video tracking) - {result.get('frame_count', 0)} frames, {result.get('num_objects', 0)} objects")
                
                # Thumbnail for MP4
                if output_mode == OutputMode.MP4:
                    _generate_thumbnail(output_path)
                
                url_data = _build_response(output_path, result, output_mode, gen_dir)
                processor_kwargs = {
                    "crypto_model": request.crypto_model,
                    "num_objects": result.get("num_objects", 0),
                    "sam_version": result.get("sam_version", "unknown"),
                }
                
                # Unload models to free VRAM
                crypto_processor.unload()
                
                batch_processed = True
            else:
                raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
            
            # ---- Frame-by-frame path (canny, normals, openpose) ----
            if not batch_processed:
                start_time = time.time()
                
                result = video_processor.process_video(
                    input_video=input_path,
                    output_path=output_path,
                    frame_processor=frame_processor,
                    output_mode=output_mode,
                )
                
                elapsed = time.time() - start_time
                log.timing("VideoPreprocess", start_time, f"Complete - {result.get('frame_count', 0)} frames")
                
                # Thumbnail for MP4
                if output_mode == OutputMode.MP4:
                    _generate_thumbnail(output_path)
                
                url_data = _build_response(output_path, result, output_mode, gen_dir)
            
            # Save metadata
            save_generation_metadata(
                gen_dir=gen_dir,
                prompt=f"Video Preprocess: {request.method}",
                model=request.method,
                seed=None,
                image_size=(video_info.get("width", 0), video_info.get("height", 0)),
                source_image=str(request.video_path),
                parameters={
                    "method": request.method,
                    "output_mode": request.output_mode,
                    "frame_count": result.get("frame_count", 0),
                    "fps": result.get("fps", 0),
                    "duration": result.get("duration", 0),
                    **processor_kwargs,
                },
            )

            # Clear preprocessor caches to free VRAM after video processing
            # Video preprocessing is memory-intensive, so unload models when done
            preprocessor_manager.clear_caches([request.method])
            clear_vram()

            return {
                "success": True,
                "output_path": str(output_path),
                "output_url": url_data["output_url"],
                "preview_url": url_data.get("preview_url"),
                "is_sequence": url_data.get("is_sequence", False),
                "frames": url_data.get("frames", []),
                "method": request.method,
                "output_mode": request.output_mode,
                "frame_count": result.get("frame_count", 0),
                "fps": result.get("fps", 0),
                "duration": result.get("duration", 0),
                "elapsed_seconds": elapsed,
                "errors": result.get("errors", []),
            }
            
        except Exception as e:
            if gen_dir:
                cleanup_failed_generation(gen_dir, reason=str(e))
            # Clear caches on error too
            preprocessor_manager.clear_caches([request.method])
            clear_vram()
            log.exception("VideoPreprocess", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # Video Upscale Endpoint
    # ========================================================================
    
    @app.post("/api/postprocess/upscale/video")
    async def upscale_video(request: VideoUpscaleRequest):
        """
        Upscale a video frame-by-frame using Real-ESRGAN
        """
        try:
            # Resolve input path
            input_path = resolve_input_path(request.source_path)
            
            if input_path is None:
                raise HTTPException(status_code=400, detail="No video path provided")
            
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {request.source_path}")
            
            log.header("VIDEO UPSCALE")
            log.info("VideoUpscale", f"Input: {input_path}")
            log.info("VideoUpscale", f"Scale: {request.scale}x")
            log.info("VideoUpscale", f"Model: {request.model}")
            log.info("VideoUpscale", f"Output Mode: {request.output_mode}")
            
            # Create output directory
            gen_dir = get_generation_output_dir(f"upscale_video_{request.scale}x")
            
            output_mode = OutputMode.SEQUENCE if request.output_mode == "sequence" else OutputMode.MP4
            
            if output_mode == OutputMode.MP4:
                output_path = gen_dir / f"upscaled_{request.scale}x.mp4"
            else:
                output_path = gen_dir / f"upscaled_{request.scale}x"
            
            log.info("VideoUpscale", f"Output path: {output_path}")
            
            # Get video info
            video_info = _get_video_info(input_path)
            
            # Frame processor for upscaling
            def frame_processor(inp, out):
                return postprocessor_manager.upscale_image(
                    input_path=inp,
                    output_path=out,
                    scale=request.scale,
                    model=request.model,
                    denoise=request.denoise,
                )
            
            # Process video
            start_time = time.time()
            
            result = video_processor.process_video(
                input_video=input_path,
                output_path=output_path,
                frame_processor=frame_processor,
                output_mode=output_mode,
            )

            # Generate thumbnail for the upscaled video
            if output_mode == OutputMode.MP4:
                _generate_thumbnail(output_path)

            elapsed = time.time() - start_time
            log.timing("VideoUpscale", start_time, f"Complete - {result.get('frame_count', 0)} frames")
            
            # Build response based on output mode
            url_data = _build_response(output_path, result, output_mode, gen_dir)
            
            # Calculate output dimensions
            input_width = video_info.get("width", 0)
            input_height = video_info.get("height", 0)
            output_width = input_width * request.scale
            output_height = input_height * request.scale
            
            # Save metadata
            save_generation_metadata(
                gen_dir=gen_dir,
                prompt=f"Video Upscale {request.scale}x",
                model=request.model,
                seed=None,
                image_size=(output_width, output_height),
                source_image=str(request.source_path),
                parameters={
                    "scale": request.scale,
                    "model": request.model,
                    "denoise": request.denoise,
                    "output_mode": request.output_mode,
                    "frame_count": result.get("frame_count", 0),
                    "input_size": {"width": input_width, "height": input_height},
                    "output_size": {"width": output_width, "height": output_height},
                },
            )
            
            return {
                "success": True,
                "output_path": str(output_path),
                "output_url": url_data["output_url"],
                "preview_url": url_data.get("preview_url"),
                "is_sequence": url_data.get("is_sequence", False),
                "frames": url_data.get("frames", []),
                "scale": request.scale,
                "model": request.model,
                "output_mode": request.output_mode,
                "input_size": {"width": input_width, "height": input_height},
                "output_size": {"width": output_width, "height": output_height},
                "frame_count": result.get("frame_count", 0),
                "fps": result.get("fps", 0),
                "duration": result.get("duration", 0),
                "elapsed_seconds": elapsed,
                "errors": result.get("errors", []),
            }
            
        except Exception as e:
            log.exception("VideoUpscale", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # Video Layers Endpoint
    # ========================================================================
    
    @app.post("/api/layers/video/generate")
    async def generate_video_layers(request: VideoLayersRequest):
        """
        Generate multiple AOV layers from a video
        
        Always produces MP4 previews + raw .npy data for each layer.
        Raw data is used by the EXR exporter for lossless output.
        
        Depth/crypto: raw .npy saved by their preprocessors
        Normals: raw .npy accumulated from frame callbacks
        """
        try:
            # Resolve input path
            input_path = resolve_input_path(request.video_path)
            
            if input_path is None:
                raise HTTPException(status_code=400, detail="No video path provided")
            
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")
            
            enabled_layers = [k for k, v in request.layers.items() if v]
            enabled_layers = [k for k, v in request.layers.items() if v]
            
            log.header("VIDEO LAYERS GENERATION")
            log.info("VideoLayers", f"Input: {input_path}")
            log.info("VideoLayers", f"Layers: {enabled_layers}")
            log.info("VideoLayers", f"Output: MP4 + raw .npy (always)")
            
            # Create output directory
            gen_dir = get_generation_output_dir("video_layers")
            
            log.info("VideoLayers", f"Output dir: {gen_dir}")
            
            # Get video info
            video_info = _get_video_info(input_path)
            
            # Copy source video for history preview (symlink to save space)
            import shutil
            source_copy = gen_dir / f"source{input_path.suffix}"
            try:
                source_copy.symlink_to(input_path)
                log.info("VideoLayers", f"Linked source: {source_copy.name}")
            except (OSError, NotImplementedError):
                shutil.copy2(input_path, source_copy)
                log.info("VideoLayers", f"Copied source: {source_copy.name}")
            
            # Copy latents from source generation dir so layers folder is self-contained
            # e.g. video_002/latents/generated.latent.pt -> video_layers_005/latents/
            source_latent_dir = input_path.parent / "latents"
            if source_latent_dir.exists():
                dest_latent_dir = gen_dir / "latents"
                dest_latent_dir.mkdir(exist_ok=True)
                latent_count = 0
                for latent_file in source_latent_dir.glob("*.latent.pt"):
                    shutil.copy2(latent_file, dest_latent_dir / latent_file.name)
                    latent_count += 1
                if latent_count:
                    log.info("VideoLayers", f"Copied {latent_count} latent(s) to layers dir ({sum(f.stat().st_size for f in dest_latent_dir.iterdir()) / (1024*1024):.1f} MiB)")
            else:
                log.warning("VideoLayers", f"No latents/ found at {source_latent_dir} — EXR export will need beauty_latent path")
            
            if not enabled_layers:
                raise HTTPException(status_code=400, detail="No layers enabled")
            
            start_time = time.time()
            
            # Collect all layer results here
            all_layer_results: Dict[str, Dict[str, Any]] = {}
            
            # ----------------------------------------------------------
            # Depth layer - always MP4 + raw .npy (saved by preprocessor)
            # ----------------------------------------------------------
            if request.layers.get("depth"):
                depth_model = DEPTH_MODEL_MAP.get(request.depth_model, DepthModel.DEPTH_ANYTHING_V2)
                log.info("VideoLayers", f"Processing depth layer ({depth_model.value})...")
                
                depth_output = gen_dir / "depth.mp4"
                depth_processor = DepthPreprocessor(model_type=depth_model)
                
                depth_result = depth_processor.process_video(
                    video_path=input_path,
                    output_path=depth_output,
                    output_mode="mp4",
                    invert=request.depth_invert,
                    normalize=request.depth_normalize,
                    colormap=request.depth_colormap,
                    temporal_smooth=5,
                )
                
                depth_processor.unload()
                
                all_layer_results["depth"] = {
                    "output_path": depth_result["output_path"],
                    "frame_count": depth_result.get("frame_count", 0),
                    "errors": 0,
                }
                log.info("VideoLayers", f"Depth complete: {depth_result.get('frame_count', 0)} frames")
            
            # ----------------------------------------------------------
            # Normals layer - MP4 preview + raw .npy for lossless EXR
            # Accumulates float32 normals from each frame callback
            # ----------------------------------------------------------
            if request.layers.get("normals"):
                normals_method = NORMALS_METHOD_MAP.get(request.normals_method, NormalsMethod.FROM_DEPTH)
                depth_model = DEPTH_MODEL_MAP.get(request.normals_depth_model, DepthModel.DEPTH_ANYTHING_V2)
                log.info("VideoLayers", f"Processing normals layer (method={normals_method.value})...")
                
                normals_output = gen_dir / "normals.mp4"
                raw_normals_frames = []
                
                def normals_frame_processor(inp, out):
                    result = preprocessor_manager.normals(
                        image_path=inp,
                        output_path=out,
                        method=normals_method,
                        depth_model=depth_model,
                        space=request.normals_space,
                        flip_y=request.normals_flip_y,
                        intensity=request.normals_intensity,
                        exact_output=True,
                    )
                    # Capture raw float32 normals [-1, 1] for lossless export
                    if "raw_normals" in result:
                        raw_normals_frames.append(result["raw_normals"])
                    return result
                
                normals_result = video_processor.process_video(
                    input_video=input_path,
                    output_path=normals_output,
                    frame_processor=normals_frame_processor,
                    output_mode=OutputMode.MP4,
                )
                
                # Save raw normals alongside MP4
                if raw_normals_frames:
                    import numpy as np
                    raw_normals_array = np.stack(raw_normals_frames, axis=0)  # [N, H, W, 3]
                    raw_normals_path = gen_dir / "normals_raw.npy"
                    np.save(str(raw_normals_path), raw_normals_array.astype(np.float32))
                    log.info("VideoLayers", f"Saved raw normals: {raw_normals_path} (shape: {raw_normals_array.shape})")
                    del raw_normals_frames, raw_normals_array
                
                all_layer_results["normals"] = {
                    "output_path": str(normals_output),
                    "frame_count": normals_result.get("frame_count", 0),
                    "errors": len(normals_result.get("errors", [])),
                }
                log.info("VideoLayers", f"Normals complete: {normals_result.get('frame_count', 0)} frames")
            
            # ----------------------------------------------------------
            # Crypto layer - always MP4 + raw .npy (saved by preprocessor)
            # ----------------------------------------------------------
            if request.layers.get("crypto"):
                sam_model = SAM_MODEL_MAP.get(request.crypto_model, SAMModel.LARGE)
                log.info("VideoLayers", f"Processing crypto layer ({sam_model.value}, video tracking)...")
                
                crypto_output = gen_dir / "crypto.mp4"
                crypto_processor = CryptoPreprocessor(model_size=sam_model)
                
                crypto_result = crypto_processor.process_video(
                    video_path=input_path,
                    output_path=crypto_output,
                    output_mode="mp4",
                    max_objects=request.crypto_max_objects,
                    min_area=request.crypto_min_area,
                )
                
                crypto_processor.unload()
                
                all_layer_results["crypto"] = {
                    "output_path": crypto_result["output_path"],
                    "frame_count": crypto_result.get("frame_count", 0),
                    "num_objects": crypto_result.get("num_objects", 0),
                    "sam_version": crypto_result.get("sam_version", "unknown"),
                    "errors": 0,
                }
                log.info("VideoLayers", f"Crypto complete: {crypto_result.get('num_objects', 0)} objects tracked")
            
            elapsed = time.time() - start_time
            log.timing("VideoLayers", start_time, f"Complete - {len(all_layer_results)} layers")
            
            # Build response — always MP4, no sequence handling
            layers_response = {}
            for layer_name, layer_data in all_layer_results.items():
                layer_path = Path(layer_data["output_path"])
                
                layers_response[layer_name] = {
                    "url": get_project_relative_url(layer_path),
                    "preview_url": get_project_relative_url(layer_path),
                    "path": str(layer_path),
                    "frame_count": layer_data.get("frame_count", 0),
                    "is_sequence": False,
                    "errors": layer_data.get("errors", 0),
                }
                
                log.info("VideoLayers", f"  {layer_name}: {layer_path}")
            
            # Save metadata
            save_generation_metadata(
                gen_dir=gen_dir,
                prompt=f"Video Layers: {', '.join(enabled_layers)}",
                model="video_layers",
                seed=None,
                image_size=(video_info.get("width", 0), video_info.get("height", 0)),
                source_image=str(request.video_path),
                parameters={
                    "layers": enabled_layers,
                    "output_mode": "mp4",
                    "frame_count": video_info.get("frame_count", 0),
                    "fps": video_info.get("fps", 0),
                },
            )
            
            return {
                "success": True,
                "layers": layers_response,
                "beauty_source": get_project_relative_url(source_copy),
                "has_latents": (gen_dir / "latents").exists(),
                "output_mode": "mp4",
                "frame_count": video_info.get("frame_count", 0),
                "fps": video_info.get("fps", 0),
                "elapsed_seconds": elapsed,
                "errors": {},
            }
            
        except Exception as e:
            log.exception("VideoLayers", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # Video Info Endpoint
    # ========================================================================
    
    @app.post("/api/video/info")
    async def video_info_endpoint(video_path: str):
        """Get video metadata (fps, duration, dimensions, etc.)"""
        try:
            input_path = resolve_input_path(video_path)
            
            if input_path is None:
                raise HTTPException(status_code=400, detail="No video path provided")
            
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
            
            info = _get_video_info(input_path)
            
            return {
                "success": True,
                **info,
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    log.success("Routes", "Video endpoints registered")