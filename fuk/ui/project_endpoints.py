"""
Project Endpoints and Cache Management
Handles project-aware output directories and file serving
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, Dict, Any
import json
import mimetypes
from datetime import datetime
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
    
    # IMPORTANT: Set cache root to be inside the project folder
    # This ensures all generations go to projectname/project/fuk/cache/
    _cache_root = folder / "cache"
    _cache_root.mkdir(exist_ok=True, parents=True)
    
    print(f"[PROJECT] ✓ Project folder set: {folder}", flush=True)
    print(f"[PROJECT] ✓ Cache root updated: {_cache_root}", flush=True)
    
    return {
        "folder": str(folder), 
        "cacheFolder": str(_cache_root),
        "success": True
    }


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
        raise HTTPException(status_code=400, detail="No project folder set. Please open a project folder first.")
    
    file_path = _project_folder / filename
    
    print(f"[PROJECT] Saving: {file_path}", flush=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Update global project state
        global _project_state
        _project_state = state
        
        print(f"[PROJECT] ✓ Saved successfully", flush=True)
        return {"success": True, "path": str(file_path)}
    except Exception as e:
        print(f"[PROJECT] ✗ Save failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@router.post("/new")
async def create_new(data: dict = Body(...)):
    """Create a new project file"""
    if not _project_folder:
        raise HTTPException(status_code=400, detail="No project folder set. Please open a project folder first.")
    
    project_name = data.get("projectName", "untitled")
    shot_number = data.get("shotNumber", "01")
    
    print(f"[PROJECT] Creating new project: {project_name}_shot{shot_number}", flush=True)
    
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
    
    project_cache = get_project_cache_dir()
    
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
async def list_generations():
    """
    List all generations in the project cache.
    Returns generation metadata for the history panel.
    """
    if not _cache_root:
        return {"generations": [], "error": "No project loaded"}
    
    try:
        project_cache = get_project_cache_dir()
    except RuntimeError:
        return {"generations": [], "error": "Project system not initialized"}
    
    if not project_cache.exists():
        return {"generations": []}
    
    generations = []
    
    # Scan all generation directories
    for gen_dir in sorted(project_cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not gen_dir.is_dir():
            continue
        
        gen_name = gen_dir.name
        
        # Determine type based on directory name prefix
        if gen_name.startswith("preprocess"):
            gen_type = "preprocess"
        elif gen_name.startswith("video") or gen_name.endswith("_video"):
            gen_type = "video"
        else:
            gen_type = "image"
        
        # Find the main output file based on type
        preview_path = None
        output_path = None
        
        if gen_type == "video":
            # Look for video file
            for ext in [".mp4", ".webm"]:
                candidate = gen_dir / f"generated{ext}"
                if candidate.exists():
                    output_path = candidate
                    preview_path = candidate
                    break
        elif gen_type == "preprocess":
            # Preprocessor output is processed.png
            candidate = gen_dir / "processed.png"
            if candidate.exists():
                output_path = candidate
                preview_path = candidate
        else:
            # Image generation - look for generated.png
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = gen_dir / f"generated{ext}"
                if candidate.exists():
                    output_path = candidate
                    preview_path = candidate
                    break
        
        if not output_path:
            continue
        
        # Get file stats
        stat = output_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        # Build relative path for API URL
        rel_path = output_path.relative_to(_cache_root)
        api_path = f"api/project/cache/{rel_path}"
        
        # Load metadata if available
        metadata = {}
        metadata_path = gen_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except:
                pass
        
        generations.append({
            "id": gen_name,
            "name": gen_name,
            "type": gen_type,
            "path": api_path,
            "preview": api_path,
            "date": mtime.strftime("%Y-%m-%d"),
            "timestamp": mtime.strftime("%H:%M:%S"),
            "size": stat.st_size,
            "prompt": metadata.get("prompt", ""),
            "seed": metadata.get("seed"),
            "model": metadata.get("model", ""),
        })
    
    return {"generations": generations}


@router.delete("/generations/{gen_id}")
async def delete_generation(gen_id: str):
    """Delete a generation directory"""
    if not _cache_root:
        raise HTTPException(status_code=400, detail="No project loaded")
    
    try:
        project_cache = get_project_cache_dir()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Project system not initialized")
    
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


def get_cache_root() -> Optional[Path]:
    """Get current cache root (for external use)"""
    return _cache_root


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