"""
Project Endpoints and Cache Management
Handles project-aware output directories and file serving
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import mimetypes
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog

# Will be set by main server
_project_folder = None
_project_state = None  # Current loaded project state
_cache_root = None
_default_cache_root = None  # Original cache root before any project is opened

router = APIRouter(prefix="/api/project", tags=["project"])


def initialize_project_system(cache_root: Path):
    """Initialize the project system with cache root"""
    global _cache_root, _default_cache_root
    _cache_root = Path(cache_root)
    _default_cache_root = Path(cache_root)  # Remember the default location
    _cache_root.mkdir(exist_ok=True, parents=True)
    print(f"[PROJECT] Initialized cache root: {_cache_root}", flush=True)


def get_current_project_info() -> Optional[Dict[str, str]]:
    """Get current project name, shot, and version from loaded state"""
    if not _project_state:
        return None
    
    project_info = _project_state.get("project", {})
    return {
        "name": project_info.get("name", "default"),
        "shot": project_info.get("shot", "01"),
        "version": project_info.get("version", "unnamed"),
    }


def get_project_cache_dir() -> Path:
    """
    Get project-specific cache directory
    
    Format: cache/{projectname}_shot{shot}_{version}/
    Example: cache/fuktest_shot01_251229/
    """
    if not _cache_root:
        raise RuntimeError("Project system not initialized")
    
    project_info = get_current_project_info()
    
    if project_info:
        # Build project-specific cache dir
        project_name = project_info["name"]
        shot = project_info["shot"]
        version = project_info["version"]
        
        cache_dir = _cache_root / f"{project_name}_shot{shot}_{version}"
    else:
        # Fallback to default if no project loaded
        cache_dir = _cache_root / "default"
    
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def get_generation_output_dir(gen_type: str = "img_gen") -> Path:
    """
    Create numbered output directory for a generation
    
    Args:
        gen_type: Type of generation (img_gen, video, preprocess, etc.)
    
    Returns:
        Path to generation directory
        
    Example:
        cache/fuktest_shot01_251229/img_gen_001/
        cache/fuktest_shot01_251229/video_001/
        cache/fuktest_shot01_251229/preprocess_001/
    """
    project_cache = get_project_cache_dir()
    
    # Find next number for this generation type
    existing = [d for d in project_cache.iterdir() 
                if d.is_dir() and d.name.startswith(f"{gen_type}_")]
    
    if existing:
        # Extract numbers and find max
        numbers = []
        for d in existing:
            try:
                num = int(d.name.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1
    
    gen_dir = project_cache / f"{gen_type}_{next_num:03d}"
    gen_dir.mkdir(exist_ok=True, parents=True)
    
    return gen_dir


def build_output_paths(gen_dir: Path) -> Dict[str, Path]:
    """
    Build standard output paths for a generation directory
    
    Returns dict with common output file paths
    """
    return {
        "generated_png": gen_dir / "generated.png",
        "generated_exr": gen_dir / "generated.exr",
        "generated_mp4": gen_dir / "generated.mp4",
        "latent": gen_dir / "latent.safetensors",
        "metadata": gen_dir / "metadata.json",
        "source": gen_dir / "source.png",
        "control": gen_dir / "control",
    }


def get_project_relative_url(file_path: Path) -> str:
    """
    Convert absolute file path to URL for serving via /api/project/cache/
    
    Args:
        file_path: Absolute path to file
    
    Returns:
        Relative URL for the dynamic cache endpoint
        
    Example:
        /home/user/myproject/Projects/Fuk/cache/fuktest_shot01_251229/img_gen_001/generated.png
        -> api/project/cache/fuktest_shot01_251229/img_gen_001/generated.png
    """
    if not _cache_root:
        raise RuntimeError("Project system not initialized")
    
    try:
        # Get path relative to cache root
        rel_path = Path(file_path).relative_to(_cache_root)
        return f"api/project/cache/{rel_path}"
    except ValueError:
        # If file is not in cache, return as-is (might be in outputs/)
        return str(file_path)


def save_generation_metadata(
    gen_dir: Path,
    prompt: str,
    model: str,
    seed: Optional[int],
    image_size: tuple,
    **kwargs
) -> Path:
    """
    Save generation metadata to JSON file
    
    Args:
        gen_dir: Generation directory
        prompt: User prompt
        model: Model used
        seed: Random seed
        image_size: (width, height)
        **kwargs: Additional metadata
    
    Returns:
        Path to metadata.json file
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "model": model,
        "seed": seed,
        "image_size": list(image_size),
        **kwargs
    }
    
    metadata_path = gen_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


# ============================================================================
# FastAPI Routes for Project Management
# ============================================================================

@router.post("/browse-folder")
async def browse_folder():
    """Open native folder browser dialog"""
    print("[PROJECT] Opening folder browser...", flush=True)
    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.attributes('-topmost', True)  # Bring to front
        
        folder = filedialog.askdirectory(
            title="Select FUK Project Folder (Projects/Fuk/)",
            mustexist=True
        )
        
        root.destroy()
        
        if folder:
            print(f"[PROJECT] User selected: {folder}", flush=True)
            return {"path": folder, "cancelled": False}
        else:
            print("[PROJECT] User cancelled folder selection", flush=True)
            return {"path": None, "cancelled": True}
            
    except Exception as e:
        print(f"[PROJECT] ✗ Folder browser error: {e}", flush=True)
        return {"error": str(e), "cancelled": True}


@router.post("/browse-save")
async def browse_save_location(data: dict = Body(...)):
    """
    Open native save file dialog.
    
    Args (in body):
        title: Dialog title
        defaultName: Default filename
        fileTypes: List of [description, pattern] tuples
        initialDir: Starting directory (optional)
    
    Returns:
        path: Selected file path or None if cancelled
        cancelled: True if user cancelled
    """
    print("[PROJECT] Opening save dialog...", flush=True)
    
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Parse file types
        file_types = data.get('fileTypes', [['All Files', '*.*']])
        tk_filetypes = [(desc, pattern) for desc, pattern in file_types]
        
        # Get initial directory
        initial_dir = data.get('initialDir')
        if not initial_dir and _project_folder:
            initial_dir = str(_project_folder)
        
        # Open save dialog
        file_path = filedialog.asksaveasfilename(
            title=data.get('title', 'Save File'),
            initialfile=data.get('defaultName', ''),
            initialdir=initial_dir,
            filetypes=tk_filetypes,
            defaultextension='.exr',
        )
        
        root.destroy()
        
        if file_path:
            print(f"[PROJECT] User selected save location: {file_path}", flush=True)
            return {"path": file_path, "cancelled": False}
        else:
            print("[PROJECT] User cancelled save dialog", flush=True)
            return {"path": None, "cancelled": True}
            
    except Exception as e:
        print(f"[PROJECT] ✗ Save dialog error: {e}", flush=True)
        return {"error": str(e), "cancelled": True}


@router.post("/set-folder")
async def set_folder(data: dict = Body(...)):
    """Set the active project folder and update cache location"""
    global _project_folder, _cache_root
    
    folder_path = data.get("path")
    if not folder_path:
        raise HTTPException(status_code=400, detail="No path provided")
    
    folder = Path(folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Folder does not exist")
    
    _project_folder = folder
    
    # Update cache root to be inside the project folder
    _cache_root = folder / "cache"
    _cache_root.mkdir(exist_ok=True, parents=True)
    
    print(f"[PROJECT] Set project folder: {folder}", flush=True)
    print(f"[PROJECT] Cache root updated to: {_cache_root}", flush=True)
    
    return {"folder": str(folder), "success": True, "cacheRoot": str(_cache_root)}


@router.get("/current")
async def get_current():
    """Get current project folder"""
    return {
        "isSet": _project_folder is not None,
        "folder": str(_project_folder) if _project_folder else None,
        "projectInfo": get_current_project_info()
    }


@router.get("/list")
async def list_files():
    """List project files in current folder"""
    if not _project_folder:
        return {"files": [], "folder": None}
    
    # Find all .json files
    json_files = list(_project_folder.glob("*.json"))
    
    files = []
    for f in json_files:
        files.append({
            "name": f.name,
            "path": str(f),
            "modifiedAt": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })
    
    return {
        "files": sorted(files, key=lambda x: x["modifiedAt"], reverse=True),
        "folder": str(_project_folder)
    }


@router.get("/load/{filename}")
async def load_file(filename: str):
    """Load a project file"""
    if not _project_folder:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    file_path = _project_folder / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Update global project state
        global _project_state
        _project_state = data
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load file: {str(e)}")


@router.post("/save/{filename}")
async def save_file(filename: str, state: dict = Body(...)):
    """Save project state to file"""
    if not _project_folder:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    file_path = _project_folder / filename
    
    try:
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Update global project state
        global _project_state
        _project_state = state
        
        return {"success": True, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@router.post("/new")
async def create_new(data: dict = Body(...)):
    """Create a new project file"""
    if not _project_folder:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    project_name = data.get("projectName", "untitled")
    shot_number = data.get("shotNumber", "01")
    
    # Generate version (YYMMDD format)
    now = datetime.now()
    version = now.strftime("%y%m%d")
    
    # Build filename
    filename = f"{project_name}_shot{shot_number}_{version}.json"
    file_path = _project_folder / filename
    
    # Check if exists
    if file_path.exists():
        # Add letter suffix (a, b, c, etc.)
        suffix = 'a'
        while (_project_folder / f"{project_name}_shot{shot_number}_{version}{suffix}.json").exists():
            suffix = chr(ord(suffix) + 1)
        filename = f"{project_name}_shot{shot_number}_{version}{suffix}.json"
        file_path = _project_folder / filename
    
    # Create empty project state
    state = {
        "meta": {
            "version": "1.0",
            "createdAt": datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
            "fukVersion": "1.0.0",
        },
        "project": {
            "name": project_name,
            "shot": shot_number,
            "version": version,
        },
        "tabs": {
            "image": {},
            "video": {},
            "preprocess": {},
        },
        "assets": {},
        "lastState": {},
        "notes": "",
    }
    
    try:
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Update global project state
        global _project_state
        _project_state = state
        
        return {"success": True, "filename": filename, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create file: {str(e)}")


@router.get("/config")
async def get_config():
    """Get project configuration"""
    return {
        "versionFormat": "date",  # or "sequential"
        "cacheFolder": str(_cache_root) if _cache_root else None,
    }


@router.get("/cache-info")
async def get_cache_info():
    """Get cache folder information"""
    if not _cache_root:
        return {"exists": False}
    
    try:
        project_cache = get_project_cache_dir()
    except RuntimeError:
        return {"exists": False}
    
    # Count generations by type
    counts = {}
    for item in project_cache.iterdir():
        if item.is_dir():
            gen_type = item.name.rsplit("_", 1)[0]
            counts[gen_type] = counts.get(gen_type, 0) + 1
    
    return {
        "exists": True,
        "path": str(project_cache),
        "projectInfo": get_current_project_info(),
        "generations": counts,
    }


@router.get("/generations")
async def list_generations(
    days: int = Query(default=1, description="Number of days to load (0 = all)"),
    pinned: str = Query(default="", description="Comma-separated list of pinned IDs to always include")
):
    """
    List generations in the project cache with pagination support.
    
    Args:
        days: Number of days of history to load (default 1 = today only, 0 = all)
        pinned: Comma-separated pinned item IDs to always include regardless of date
    
    Returns generation metadata for the history panel.
    """
    print(f"[HISTORY] Fetching generations (days={days}, pinned={pinned[:50]}...)", flush=True)
    
    if not _cache_root:
        print("[HISTORY] No cache root set", flush=True)
        return {"generations": [], "error": "No project loaded", "hasMore": False}
    
    try:
        project_cache = get_project_cache_dir()
        print(f"[HISTORY] Project cache dir: {project_cache}", flush=True)
    except RuntimeError as e:
        print(f"[HISTORY] Error getting project cache: {e}", flush=True)
        return {"generations": [], "error": "Project system not initialized", "hasMore": False}
    
    if not project_cache.exists():
        print(f"[HISTORY] Project cache doesn't exist: {project_cache}", flush=True)
        return {"generations": [], "hasMore": False}
    
    # Parse pinned IDs
    pinned_ids = set(p.strip() for p in pinned.split(",") if p.strip())
    
    # Calculate date cutoff
    if days > 0:
        cutoff_date = datetime.now() - timedelta(days=days)
    else:
        cutoff_date = None
    
    generations = []
    has_more = False
    
    # Scan all generation directories
    dirs = list(project_cache.iterdir())
    print(f"[HISTORY] Found {len(dirs)} items in cache", flush=True)
    
    for gen_dir in sorted(dirs, key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True):
        if not gen_dir.is_dir():
            continue
        
        gen_name = gen_dir.name
        dir_mtime = datetime.fromtimestamp(gen_dir.stat().st_mtime)
        
        # Check if within date range (unless pinned)
        is_pinned = gen_name in pinned_ids
        if cutoff_date and dir_mtime < cutoff_date and not is_pinned:
            # Check for any pinned sub-items (for layers)
            has_pinned_child = any(f"{gen_name}/{prefix}" in pinned_ids 
                                   for prefix in ["depth", "normals", "crypto"])
            if not has_pinned_child:
                has_more = True
                continue
        
        # Determine type based on directory name prefix
        if gen_name.startswith("layers"):
            gen_type = "layers"
        elif gen_name.startswith("preprocess"):
            gen_type = "preprocess"
        elif gen_name.startswith("video") or gen_name.endswith("_video"):
            gen_type = "video"
        elif gen_name.startswith("upscale"):
            gen_type = "upscale"
        elif gen_name.startswith("interpolate"):
            gen_type = "interpolate"
        elif gen_name.startswith("export"):
            gen_type = "export"
        else:
            gen_type = "image"
        
        # Load metadata if available
        metadata = {}
        metadata_path = gen_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Special handling for layers - create entry for EACH layer file
        if gen_type == "layers":
            layer_entries = _get_layer_entries(gen_dir, gen_name, metadata, pinned_ids, _cache_root)
            generations.extend(layer_entries)
            continue
        
        # Find the main output file based on type
        preview_path = None
        output_path = None
        
        if gen_type == "video":
            for ext in [".mp4", ".webm"]:
                candidate = gen_dir / f"generated{ext}"
                if candidate.exists():
                    output_path = candidate
                    preview_path = candidate
                    break
        elif gen_type == "interpolate":
            video_files = list(gen_dir.glob("interpolated*.mp4")) + list(gen_dir.glob("interpolated*.webm"))
            if video_files:
                output_path = video_files[0]
                preview_path = output_path
        elif gen_type == "upscale":
            upscaled_files = list(gen_dir.glob("upscaled*.png")) + list(gen_dir.glob("upscaled*.jpg"))
            if upscaled_files:
                output_path = upscaled_files[0]
                preview_path = output_path
        elif gen_type == "preprocess":
            processed_files = list(gen_dir.glob("processed*.png"))
            if processed_files:
                output_path = processed_files[0]
                preview_path = output_path
        elif gen_type == "export":
            exr_files = list(gen_dir.glob("*.exr"))
            if exr_files:
                output_path = exr_files[0]
            png_files = list(gen_dir.glob("*.png"))
            if png_files:
                preview_path = png_files[0]
            elif exr_files:
                preview_path = exr_files[0]
        else:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = gen_dir / f"generated{ext}"
                if candidate.exists():
                    output_path = candidate
                    preview_path = candidate
                    break
        
        if not output_path:
            print(f"[HISTORY]   {gen_name}: no output file found, skipping", flush=True)
            continue
        
        # Get file stats
        stat = output_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        # Build relative path for API URL
        rel_path = output_path.relative_to(_cache_root)
        api_path = f"api/project/cache/{rel_path}"
        
        # Preview path
        preview_rel = preview_path.relative_to(_cache_root) if preview_path else rel_path
        preview_api = f"api/project/cache/{preview_rel}"
        
        generations.append({
            "id": gen_name,
            "name": gen_name,
            "type": gen_type,
            "path": api_path,
            "preview": preview_api,
            "date": mtime.strftime("%Y-%m-%d"),
            "timestamp": mtime.strftime("%H:%M:%S"),
            "size": stat.st_size,
            "prompt": metadata.get("prompt", ""),
            "seed": metadata.get("seed"),
            "model": metadata.get("model", ""),
            "pinned": gen_name in pinned_ids,
        })
    
    # Sort: pinned first, then by date
    generations.sort(key=lambda g: (not g.get("pinned", False), g["date"], g["timestamp"]), reverse=True)
    
    # Summary
    types = {}
    for g in generations:
        types[g['type']] = types.get(g['type'], 0) + 1
    print(f"[HISTORY] Returning {len(generations)} generations: {types}, hasMore={has_more}", flush=True)
    
    return {"generations": generations, "hasMore": has_more}


def _get_layer_entries(gen_dir: Path, gen_name: str, metadata: dict, pinned_ids: set, cache_root: Path) -> List[dict]:
    """
    Create separate history entries for each layer file in a layers directory.
    
    Returns list of generation entries for depth, normals, crypto files.
    """
    entries = []
    
    # Layer type mapping
    layer_types = {
        "depth_": ("depth", "Depth"),
        "normals_": ("normals", "Normals"),
        "crypto_": ("crypto", "Crypto"),
    }
    
    for prefix, (layer_type, label) in layer_types.items():
        layer_files = list(gen_dir.glob(f"{prefix}*.png"))
        
        for layer_file in layer_files:
            stat = layer_file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            rel_path = layer_file.relative_to(cache_root)
            api_path = f"api/project/cache/{rel_path}"
            
            # Create unique ID for this specific layer
            layer_id = f"{gen_name}/{layer_type}"
            
            entries.append({
                "id": layer_id,
                "name": f"{gen_name}/{label}",
                "type": "layers",
                "subtype": layer_type,  # depth, normals, or crypto
                "path": api_path,
                "preview": api_path,
                "date": mtime.strftime("%Y-%m-%d"),
                "timestamp": mtime.strftime("%H:%M:%S"),
                "size": stat.st_size,
                "prompt": metadata.get("prompt", ""),
                "seed": metadata.get("seed"),
                "model": metadata.get("model", ""),
                "pinned": layer_id in pinned_ids,
                "parentDir": gen_name,
            })
    
    return entries


@router.delete("/generations/{gen_id:path}")
async def delete_generation(gen_id: str):
    """Delete a generation directory or specific layer file"""
    if not _cache_root:
        raise HTTPException(status_code=400, detail="No project loaded")
    
    try:
        project_cache = get_project_cache_dir()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Project system not initialized")
    
    # Check if it's a layer-specific ID (contains /)
    if "/" in gen_id:
        # It's a specific layer - just delete that file? Or the whole dir?
        # For now, delete the whole layers directory
        gen_id = gen_id.split("/")[0]
    
    gen_dir = project_cache / gen_id
    
    # Security check
    try:
        gen_dir.resolve().relative_to(project_cache.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid generation ID")
    
    if not gen_dir.exists():
        raise HTTPException(status_code=404, detail="Generation not found")
    
    if not gen_dir.is_dir():
        raise HTTPException(status_code=400, detail="Not a generation directory")
    
    # Delete the directory
    import shutil
    try:
        shutil.rmtree(gen_dir)
        print(f"[HISTORY] Deleted generation: {gen_id}", flush=True)
        return {"success": True, "deleted": gen_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


def setup_project_routes(app):
    """Setup project routes on FastAPI app"""
    app.include_router(router)


# ============================================================================
# Dynamic File Serving (replaces static mount for project cache)
# ============================================================================

@router.get("/cache/{file_path:path}")
async def serve_cache_file(file_path: str):
    """
    Dynamically serve files from the project cache directory.
    This allows the cache location to change when projects are opened.
    
    Falls back to default cache if file not found in project cache.
    
    URL: /api/project/cache/{relative_path}
    Example: /api/project/cache/fuktest_shot01_251229/img_gen_001/generated.png
    """
    print(f"[CACHE] Serving request for: {file_path}", flush=True)
    print(f"[CACHE] Current cache root: {_cache_root}", flush=True)
    print(f"[CACHE] Default cache root: {_default_cache_root}", flush=True)
    
    if not _cache_root and not _default_cache_root:
        print("[CACHE] ✗ Cache not initialized", flush=True)
        raise HTTPException(status_code=503, detail="Cache not initialized. Please restart the server.")
    
    # Try current cache root first
    full_path = None
    tried_paths = []
    
    if _cache_root:
        candidate = _cache_root / file_path
        tried_paths.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            full_path = candidate
            print(f"[CACHE] Found in project cache: {full_path}", flush=True)
    
    # Fall back to default cache if not found
    if not full_path and _default_cache_root and _default_cache_root != _cache_root:
        candidate = _default_cache_root / file_path
        tried_paths.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            full_path = candidate
            print(f"[CACHE] Found in default cache (fallback): {full_path}", flush=True)
    
    if not full_path:
        print(f"[CACHE] ✗ File not found in any cache location", flush=True)
        print(f"[CACHE]   Tried: {tried_paths}", flush=True)
        raise HTTPException(
            status_code=404, 
            detail=f"File not found: {file_path}. Tried: {', '.join(tried_paths)}"
        )
    
    # Security: ensure path is within one of the cache roots
    try:
        if _cache_root:
            full_path.resolve().relative_to(_cache_root.resolve())
    except ValueError:
        try:
            if _default_cache_root:
                full_path.resolve().relative_to(_default_cache_root.resolve())
        except ValueError:
            print(f"[CACHE] ✗ Access denied - path escapes cache roots", flush=True)
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Determine content type
    content_type, _ = mimetypes.guess_type(str(full_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
    print(f"[CACHE] ✓ Serving {full_path} as {content_type}", flush=True)
    
    return FileResponse(
        path=full_path,
        media_type=content_type,
        filename=full_path.name
    )

@router.get("/debug")
async def debug_project_state():
    """Debug endpoint to check project system state"""
    return {
        "project_folder": str(_project_folder) if _project_folder else None,
        "cache_root": str(_cache_root) if _cache_root else None,
        "default_cache_root": str(_default_cache_root) if _default_cache_root else None,
        "cache_exists": _cache_root.exists() if _cache_root else False,
        "project_state_keys": list(_project_state.keys()) if _project_state else [],
    }

def get_cache_root() -> Optional[Path]:
    """Get current cache root (for external use)"""
    global _cache_root
    # Add debug logging
    if _cache_root is None:
        print(f"[PROJECT] get_cache_root() -> None (no project loaded)", flush=True)
    return _cache_root


def get_default_cache_root() -> Optional[Path]:
    """Get the default cache root (original location before any project was opened)"""
    return _default_cache_root