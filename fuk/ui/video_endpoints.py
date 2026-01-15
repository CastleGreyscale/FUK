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
from fuk.core.preprocessors.depth_video_batch import VideoDepthBatchProcessor
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
import json
import time

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
    output_mode: VideoOutputMode = VideoOutputMode.MP4
    
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
    log
):
    """
    Setup video routes with access to server resources
    
    Call this from the main server file to register video endpoints.
    """
    
    from core.video_processor import VideoProcessor, OutputMode
    
    video_processor = VideoProcessor()
    
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
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build response data for MP4 output mode."""
        output_url = get_project_relative_url(output_path)
        
        return {
            "output_url": output_url,
            "preview_url": output_url,  # Video can be its own preview
            "is_sequence": False,
        }
    
    # ========================================================================
    # Video Preprocess Endpoint
    # ========================================================================
    
    @app.post("/api/preprocess/video")
    async def preprocess_video(request: VideoPreprocessRequest, background_tasks: BackgroundTasks):
        """
        Apply preprocessing to a video frame-by-frame
        
        Supports: canny, depth, normals, openpose, crypto
        """
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
                # Don't create yet - let video_processor handle it
            
            log.info("VideoPreprocess", f"Output path: {output_path}")
            
            # Get video info
            video_info = video_processor.get_video_info(input_path)
            
            # Build processor kwargs based on method
            processor_kwargs = {}
            
            if request.method == "canny":
                def frame_processor(inp, out):
                    return preprocessor_manager.canny(
                        image_path=inp,
                        output_path=out,
                        low_threshold=request.low_threshold,
                        high_threshold=request.high_threshold,
                        invert=request.canny_invert,
                        blur_kernel=request.blur_kernel,
                        exact_output=True,  # Use exact path for video frames
                    )
                processor_kwargs = {
                    "low_threshold": request.low_threshold,
                    "high_threshold": request.high_threshold,
                }
                batch_processed = False
                    
            elif request.method == "depth":
                depth_model_map = {
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
                depth_model = depth_model_map.get(request.depth_model, DepthModel.DEPTH_ANYTHING_V2)
                
                # Check if we should use batch processing (DA3 models only)
                use_batch = depth_model in [
                    DepthModel.DEPTH_ANYTHING_V3,
                    DepthModel.DA3_MONO_LARGE,
                    DepthModel.DA3_METRIC_LARGE,
                    DepthModel.DA3_LARGE,
                    DepthModel.DA3_GIANT,
                ]
                
                if use_batch:
                    # Use batch processor for DA3 models (temporal consistency)
                    log.info("VideoPreprocess", "Using BATCH processing for DA3 (temporal consistency)")
                    
                    # Initialize depth model
                    from fuk.core.preprocessors.depth import DepthPreprocessor
                    depth_processor = DepthPreprocessor(model_type=depth_model)
        
                    
                    # Create batch processor
                    batch_processor = VideoDepthBatchProcessor()
                    
                    # Process with batch inference (includes timing)
                    start_time = time.time()
                    
                    result = batch_processor.process_video_batch(
                        video_path=input_path,
                        output_path=output_path,
                        depth_model=depth_processor,
                        model_type=depth_model,
                        invert=request.depth_invert,
                        normalize=request.depth_normalize,
                        colormap=request.depth_colormap,
                        process_res=504,  # DA3 default
                        process_res_method="upper_bound_resize",  # DA3 default
                        output_mode="mp4" if output_mode == OutputMode.MP4 else "sequence",
                    )
                    
                    elapsed = time.time() - start_time
                    log.timing("VideoPreprocess", start_time, f"Complete (batch) - {result.get('frame_count', 0)} frames")
                    
                    # Build response
                    if output_mode == OutputMode.SEQUENCE:
                        url_data = build_sequence_response(output_path, result, gen_dir)
                    else:
                        url_data = build_video_response(output_path, result)
                    
                    processor_kwargs = {"depth_model": request.depth_model}
                    
                    # Set flag to skip common video_processor call
                    batch_processed = True
                
                else:
                    # Frame-by-frame for other models (V2, MiDaS, ZoeDepth)
                    def frame_processor(inp, out):
                        return preprocessor_manager.depth(
                            image_path=inp,
                            output_path=out,
                            model=depth_model,
                            invert=request.depth_invert,
                            normalize=request.depth_normalize,
                            colormap=request.depth_colormap,
                            exact_output=True,
                        )
                    processor_kwargs = {"depth_model": request.depth_model}
                    batch_processed = False  
            elif request.method == "normals":
                normals_method_map = {
                    "from_depth": NormalsMethod.FROM_DEPTH,
                    "dsine": NormalsMethod.DSINE,
                }
                depth_model_map = {
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
                
                normals_method = normals_method_map.get(request.normals_method, NormalsMethod.FROM_DEPTH)
                depth_model = depth_model_map.get(request.normals_depth_model, DepthModel.DEPTH_ANYTHING_V2)
                
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
                        exact_output=True,  # Use exact path for video frames
                    )
                processor_kwargs = {"normals_method": request.normals_method}
                batch_processed = False
                    
            elif request.method == "openpose":
                def frame_processor(inp, out):
                    return preprocessor_manager.openpose(
                        image_path=inp,
                        output_path=out,
                        detect_body=request.detect_body,
                        detect_hand=request.detect_hand,
                        detect_face=request.detect_face,
                        exact_output=True,  # Use exact path for video frames
                    )
                processor_kwargs = {
                    "detect_body": request.detect_body,
                    "detect_hand": request.detect_hand,
                    "detect_face": request.detect_face,
                }
                batch_processed = False
                    
            elif request.method == "crypto":
                sam_model_map = {
                    "sam2_hiera_tiny": SAMModel.TINY,
                    "sam2_hiera_small": SAMModel.SMALL,
                    "sam2_hiera_base_plus": SAMModel.BASE,
                    "sam2_hiera_large": SAMModel.LARGE,
                }
                sam_model = sam_model_map.get(request.crypto_model, SAMModel.LARGE)
                
                def frame_processor(inp, out):
                    return preprocessor_manager.crypto(
                        image_path=inp,
                        output_path=out,
                        model=sam_model,
                        max_objects=request.crypto_max_objects,
                        min_area=request.crypto_min_area,
                        output_mode="id_matte",
                        exact_output=True,  # Use exact path for video frames
                    )
                processor_kwargs = {"crypto_model": request.crypto_model}
                batch_processed = False
            else:
                raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
            
            # Process video (skip if already batch processed)
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
                
                # Build response based on output mode
                if output_mode == OutputMode.SEQUENCE:
                    url_data = build_sequence_response(output_path, result, gen_dir)
                else:
                    url_data = build_video_response(output_path, result)
            
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
            video_info = video_processor.get_video_info(input_path)
            
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
            if output_mode == OutputMode.MP4 and output_path.exists():
                thumb_path = output_path.with_suffix('.thumb.jpg')
                video_processor.extract_thumbnail(output_path, thumb_path)


            elapsed = time.time() - start_time
            log.timing("VideoUpscale", start_time, f"Complete - {result.get('frame_count', 0)} frames")
            
            # Build response based on output mode
            if output_mode == OutputMode.SEQUENCE:
                url_data = build_sequence_response(output_path, result, gen_dir)
            else:
                url_data = build_video_response(output_path, result)
            
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
        
        Produces one output video (or sequence) per enabled layer.
        """
        try:
            # Resolve input path
            input_path = resolve_input_path(request.video_path)
            
            if input_path is None:
                raise HTTPException(status_code=400, detail="No video path provided")
            
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")
            
            enabled_layers = [k for k, v in request.layers.items() if v]
            
            log.header("VIDEO LAYERS GENERATION")
            log.info("VideoLayers", f"Input: {input_path}")
            log.info("VideoLayers", f"Layers: {enabled_layers}")
            log.info("VideoLayers", f"Output Mode: {request.output_mode}")
            
            # Create output directory
            gen_dir = get_generation_output_dir("video_layers")
            
            log.info("VideoLayers", f"Output dir: {gen_dir}")
            
            # Get video info
            video_info = video_processor.get_video_info(input_path)
            
            output_mode = OutputMode.SEQUENCE if request.output_mode == "sequence" else OutputMode.MP4
            
            # Build layer processors
            layer_processors = {}
            
            if request.layers.get("depth"):
                depth_model_map = {
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
                depth_model = depth_model_map.get(request.depth_model, DepthModel.DEPTH_ANYTHING_V2)
                
                def depth_processor(inp, out):
                    return preprocessor_manager.depth(
                        image_path=inp,
                        output_path=out,
                        model=depth_model,
                        invert=request.depth_invert,
                        normalize=request.depth_normalize,
                        colormap=request.depth_colormap,
                        exact_output=True,  # Use exact path for video frames
                    )
                layer_processors["depth"] = depth_processor
            
            if request.layers.get("normals"):
                normals_method_map = {
                    "from_depth": NormalsMethod.FROM_DEPTH,
                    "dsine": NormalsMethod.DSINE,
                }
                depth_model_map = {
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
                normals_method = normals_method_map.get(request.normals_method, NormalsMethod.FROM_DEPTH)
                depth_model = depth_model_map.get(request.normals_depth_model, DepthModel.DEPTH_ANYTHING_V2)
                
                def normals_processor(inp, out):
                    return preprocessor_manager.normals(
                        image_path=inp,
                        output_path=out,
                        method=normals_method,
                        depth_model=depth_model,
                        space=request.normals_space,
                        flip_y=request.normals_flip_y,
                        intensity=request.normals_intensity,
                        exact_output=True,  # Use exact path for video frames
                    )
                layer_processors["normals"] = normals_processor
            
            if request.layers.get("crypto"):
                sam_model_map = {
                    "sam2_hiera_tiny": SAMModel.TINY,
                    "sam2_hiera_small": SAMModel.SMALL,
                    "sam2_hiera_base_plus": SAMModel.BASE,
                    "sam2_hiera_large": SAMModel.LARGE,
                }
                sam_model = sam_model_map.get(request.crypto_model, SAMModel.LARGE)
                
                def crypto_processor(inp, out):
                    return preprocessor_manager.crypto(
                        image_path=inp,
                        output_path=out,
                        model=sam_model,
                        max_objects=request.crypto_max_objects,
                        min_area=request.crypto_min_area,
                        output_mode="id_matte",
                        exact_output=True,  # Use exact path for video frames
                    )
                layer_processors["crypto"] = crypto_processor
            
            if not layer_processors:
                raise HTTPException(status_code=400, detail="No layers enabled")
            
            # Process all layers
            start_time = time.time()
            
            result = video_processor.process_video_layers(
                input_video=input_path,
                output_dir=gen_dir,
                layer_processors=layer_processors,
                output_mode=output_mode,
            )
            
            elapsed = time.time() - start_time
            log.timing("VideoLayers", start_time, f"Complete - {len(result.get('layers', {}))} layers")
            
            # Build response with URLs for each layer
            layers_response = {}
            for layer_name, layer_data in result.get("layers", {}).items():
                layer_path = Path(layer_data["output_path"])
                
                if output_mode == OutputMode.SEQUENCE:
                    # For sequences, include preview URL (first frame)
                    first_frame = layer_data.get("first_frame")
                    preview_url = None
                    if first_frame:
                        first_frame_path = layer_path / first_frame
                        preview_url = get_project_relative_url(first_frame_path)
                    
                    layers_response[layer_name] = {
                        "url": get_project_relative_url(layer_path),
                        "preview_url": preview_url,
                        "path": str(layer_path),
                        "frame_count": layer_data.get("frame_count", 0),
                        "frames": layer_data.get("frames", []),
                        "is_sequence": True,
                        "errors": layer_data.get("errors", 0),
                    }
                else:
                    # For MP4, URL is the video itself
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
                    "output_mode": request.output_mode,
                    "frame_count": result.get("frame_count", 0),
                    "fps": result.get("fps", 0),
                },
            )
            
            return {
                "success": True,
                "layers": layers_response,
                "output_mode": request.output_mode,
                "frame_count": result.get("frame_count", 0),
                "fps": result.get("fps", 0),
                "elapsed_seconds": elapsed,
                "errors": result.get("errors", {}),
            }
            
        except Exception as e:
            log.exception("VideoLayers", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # Video Info Endpoint
    # ========================================================================
    
    @app.post("/api/video/info")
    async def get_video_info(video_path: str):
        """Get video metadata (fps, duration, dimensions, etc.)"""
        try:
            input_path = resolve_input_path(video_path)
            
            if input_path is None:
                raise HTTPException(status_code=400, detail="No video path provided")
            
            if not input_path.exists():
                raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
            
            info = video_processor.get_video_info(input_path)
            
            return {
                "success": True,
                **info,
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    log.success("Routes", "Video endpoints registered")