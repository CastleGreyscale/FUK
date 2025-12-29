"""
FUK Web Server - Integration Patch
==================================

This file shows the EXACT changes needed in your fuk_web_server.py
to integrate project management with proper image paths and metadata.

Apply these changes to your existing fuk_web_server.py:
"""

# =============================================================================
# STEP 1: Add imports at the TOP of fuk_web_server.py
# =============================================================================

# Add this import near the other imports:
from project_endpoints import (
    setup_project_routes,
    get_generation_output_dir,
    build_output_paths,
    get_project_relative_url,
    save_generation_metadata
)


# =============================================================================
# STEP 2: After creating the FastAPI app, add the project routes
# =============================================================================

# Find this line in your server:
#   app = FastAPI(title="FUK Generation API", version="1.0.0")

# Add this line AFTER it:
setup_project_routes(app)


# =============================================================================
# STEP 3: Modify run_image_generation function
# =============================================================================

# Find the run_image_generation function and make these changes:

async def run_image_generation(generation_id: str, request):  # ImageGenerationRequest
    """Background task for image generation - UPDATED VERSION"""
    
    try:
        print("\n" + "="*80)
        print(f"🚀 Starting Image Generation: {generation_id}")
        print("="*80)
        
        # Update status
        active_generations[generation_id]["status"] = "running"
        active_generations[generation_id]["phase"] = "initialization"
        
        # =====================================================================
        # CHANGED: Use project-aware output directory
        # OLD: gen_dir = image_manager.create_generation_dir()
        # NEW:
        gen_dir = get_generation_output_dir("img_gen")
        paths = build_output_paths(gen_dir)
        # =====================================================================
        
        active_generations[generation_id]["gen_dir"] = str(gen_dir)
        active_generations[generation_id]["phase"] = "generating"
        
        print(f"[{generation_id}] Output directory: {gen_dir}")
        
        # ... your existing generation code here ...
        # (the part that calls musubi and creates the image)
        
        # After generation completes successfully:
        
        # =====================================================================
        # CHANGED: Build output URLs that work with project cache
        # OLD: outputs = {"png": str(paths["generated_png"].relative_to(OUTPUT_ROOT))}
        # NEW:
        outputs = {
            "png": get_project_relative_url(paths["generated_png"])
        }
        # =====================================================================
        
        # If EXR export:
        if request.output_format in ["exr", "both"]:
            # ... your EXR conversion code ...
            outputs["exr"] = get_project_relative_url(paths["generated_exr"])
        
        # =====================================================================
        # CHANGED: Save metadata using the helper function
        # This ensures metadata.json is created in the generation folder
        # OLD: image_manager.save_metadata(...)
        # NEW:
        save_generation_metadata(
            gen_dir=gen_dir,
            prompt=request.prompt,
            model=request.model,
            seed=request.seed or 0,
            image_size=(request.width, request.height),
            infer_steps=request.steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt or DEFAULTS.get("negative_prompt", ""),
            flow_shift=request.flow_shift,
            lora=request.lora,
            lora_multiplier=request.lora_multiplier,
            control_image=str(request.control_image_path) if request.control_image_path else None
        )
        # =====================================================================
        
        # Mark complete
        active_generations[generation_id].update({
            "status": "complete",
            "phase": "complete",
            "progress": 1.0,
            "outputs": outputs,  # This now has the correct URL format
            "completed_at": datetime.now().isoformat()
        })
        
        print(f"✅ Generation Complete: {generation_id}")
        print(f"   Output PNG: {outputs['png']}")
        
    except Exception as e:
        active_generations[generation_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        print(f"❌ Generation Failed: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
"""
1. Import the helper functions from project_endpoints.py
2. Call setup_project_routes(app) after creating the FastAPI app
3. In run_image_generation:
   - Replace output directory creation with get_generation_output_dir()
   - Replace path building with build_output_paths()
   - Replace output URL creation with get_project_relative_url()
   - Replace metadata saving with save_generation_metadata()

The key change is that get_project_relative_url() returns paths like:
  - "project-cache/img_gen_20251229_143052/generated.png" (when project folder is set)
  - "image/2025-12-29/generation_001/generated.png" (fallback to old behavior)

The frontend's buildImageUrl() function handles both formats correctly.
"""
