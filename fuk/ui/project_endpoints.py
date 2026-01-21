"""
Project Endpoints and Cache Management
Handles project-aware output directories and file serving

IMPORTANT: All tkinter dialogs run via subprocess to avoid blocking
the FastAPI event loop. See file_browser.py for the subprocess handler.
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import mimetypes
from datetime import datetime, timedelta
import subprocess
import sys
import os

# Will be set by main server
_project_folder = None
_project_state = None  # Current loaded project state
_cache_root = None
_default_cache_root = None  # Original cache root before any project is opened

router = APIRouter(prefix="/api/project", tags=["project"])


# ============================================================================
# Subprocess-based Dialog Helpers (avoid tkinter/asyncio conflicts)
# ============================================================================

def _get_file_browser_script() -> Path:
    """Get path to file_browser.py script"""
    this_dir = Path(__file__).parent
    
    candidates = [
        this_dir / "file_browser.py",
        this_dir.parent / "core" / "file_browser.py",
        this_dir.parent / "file_browser.py",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        "file_browser.py not found. Expected locations:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )

def _run_dialog_subprocess(args: List[str], timeout: int = 300) -> Dict[str, Any]:
    """
    Run file_browser.py as subprocess for dialog operations.
    This avoids tkinter/asyncio conflicts that cause the server to hang.
    
    Args:
        args: Arguments to pass to file_browser.py
        timeout: Max seconds to wait
        
    Returns:
        Parsed JSON response from script
    """
    script_path = _get_file_browser_script()
    cmd = [sys.executable, str(script_path)] + args
    
    try:
        # Run with display environment for tkinter
        env = os.environ.copy()
        if 'DISPLAY' not in env:
            env['DISPLAY'] = ':0'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            return {"success": False, "error": error_msg, "cancelled": True}
        
        # Parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid response: {e}", "cancelled": True}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Dialog timed out", "cancelled": True}
    except FileNotFoundError:
        return {"success": False, "error": "file_browser.py not found", "cancelled": True}
    except Exception as e:
        return {"success": False, "error": str(e), "cancelled": True}

# ============================================================================
# Initialization
# ============================================================================

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


def ensure_project_loaded():
    """
    Ensure a project is loaded. If not, try to:
    1. Load the most recent project file in the folder
    2. If no files exist, create a default project
    
    Returns True if a project is loaded, False otherwise.
    """
    global _project_state
    
    # Already have a loaded project
    if _project_state:
        return True
    
    # No project folder set
    if not _project_folder:
        print("[PROJECT] No project folder set, cannot auto-load", flush=True)
        return False
    
    # Look for project files in the folder
    project_files = sorted(_project_folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if project_files:
        # Load the most recent project file
        latest_file = project_files[0]
        print(f"[PROJECT] Auto-loading most recent project: {latest_file.name}", flush=True)
        
        try:
            with open(latest_file) as f:
                _project_state = json.load(f)
            print(f"[PROJECT] âœ“ Auto-loaded: {latest_file.name}", flush=True)
            return True
        except Exception as e:
            print(f"[PROJECT] Failed to auto-load {latest_file.name}: {e}", flush=True)
            return False
    else:
        # No project files exist, create a default one
        print("[PROJECT] No project files found, creating default project", flush=True)
        
        try:
            # Get folder name for project name
            folder_name = _project_folder.name
            now = datetime.now()
            version = now.strftime("%y%m%d")
            filename = f"{folder_name}_shot01_{version}.json"
            file_path = _project_folder / filename
            
            # Create default project state
            state = {
                "meta": {
                    "version": "1.0",
                    "createdAt": now.isoformat(),
                    "updatedAt": now.isoformat(),
                    "fukVersion": "1.0.0",
                },
                "project": {
                    "name": folder_name,
                    "shot": "01",
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
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            _project_state = state
            print(f"[PROJECT] âœ“ Created default project: {filename}", flush=True)
            return True
            
        except Exception as e:
            print(f"[PROJECT] Failed to create default project: {e}", flush=True)
            return False


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

def cleanup_failed_generation(gen_dir: Path, reason: str = "generation failed") -> bool:
    """
    Clean up a failed generation directory.
    
    Removes the entire generation directory and all its contents when a generation
    fails, keeping the cache folder clean from partial/broken outputs.
    
    Args:
        gen_dir: Path to the generation directory to clean up
        reason: Reason for cleanup (for logging)
    
    Returns:
        True if cleanup succeeded, False otherwise
    """
    import shutil
    
    if not gen_dir:
        print(f"[CLEANUP] No directory provided, skipping cleanup", flush=True)
        return False
    
    gen_dir = Path(gen_dir)
    
    if not gen_dir.exists():
        print(f"[CLEANUP] Directory doesn't exist: {gen_dir}", flush=True)
        return False
    
    # Safety check - only delete directories within cache root
    if _cache_root:
        try:
            gen_dir.resolve().relative_to(_cache_root.resolve())
        except ValueError:
            print(f"[CLEANUP] Ã¢Å“â€” Refusing to delete - path outside cache: {gen_dir}", flush=True)
            return False
    
    try:
        # List contents before deleting for logging
        contents = list(gen_dir.iterdir()) if gen_dir.is_dir() else []
        content_names = [f.name for f in contents[:5]]
        if len(contents) > 5:
            content_names.append(f"... and {len(contents) - 5} more")
        
        shutil.rmtree(gen_dir)
        print(f"[CLEANUP] Ã¢Å“â€œ Removed failed generation: {gen_dir.name}", flush=True)
        print(f"[CLEANUP]   Reason: {reason}", flush=True)
        if content_names:
            print(f"[CLEANUP]   Deleted: {', '.join(content_names)}", flush=True)
        return True
        
    except Exception as e:
        print(f"[CLEANUP] Ã¢Å“â€” Failed to clean up {gen_dir}: {e}", flush=True)
        return False

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
    """
    Open native folder browser dialog.
    
    Uses subprocess to avoid tkinter/asyncio conflicts.
    """
    print("[PROJECT] Opening folder browser via subprocess...", flush=True)
    
    result = _run_dialog_subprocess([
        "directory",
        "--title", "Select FUK Project Folder (Projects/Fuk/)",
    ])
    
    if result.get("success") and result.get("directory"):
        folder = result["directory"]
        print(f"[PROJECT] User selected: {folder}", flush=True)
        return {"path": folder, "cancelled": False}
    else:
        error = result.get("error", "")
        if error:
            print(f"[PROJECT] Dialog error: {error}", flush=True)
        else:
            print("[PROJECT] User cancelled folder selection", flush=True)
        return {"path": None, "cancelled": True, "error": error if error else None}

@router.post("/browse-save")
async def browse_save_location(data: dict = Body(...)):
    """
    Open native save file dialog.
    
    Note: This still uses tkinter directly since file_browser.py
    doesn't have a save dialog. Consider adding one if this causes issues.
    
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
    
    # For save dialogs, we need to use tkinter directly for now
    # Run in a thread to avoid blocking
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def run_save_dialog():
        import tkinter as tk
        from tkinter import filedialog
        
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
                return {"path": file_path, "cancelled": False}
            else:
                return {"path": None, "cancelled": True}
                
        except Exception as e:
            return {"error": str(e), "cancelled": True}
    
    # Run in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_save_dialog)
    
    if result.get("path"):
        print(f"[PROJECT] User selected save location: {result['path']}", flush=True)
    elif result.get("error"):
        print(f"[PROJECT] Save dialog error: {result['error']}", flush=True)
    else:
        print("[PROJECT] User cancelled save dialog", flush=True)
    
    return result

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
    
    # AUTO-LOAD OR CREATE PROJECT
    ensure_project_loaded()
    
    return {
        "folder": str(folder), 
        "success": True, 
        "cacheRoot": str(_cache_root),
        "projectLoaded": _project_state is not None,
        "projectInfo": get_current_project_info()
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
        
        # Get the new project-specific cache directory
        project_cache = get_project_cache_dir()
        
        print(f"[PROJECT] Loaded: {filename}", flush=True)
        print(f"[PROJECT] Cache directory: {project_cache}", flush=True)
        
        # Return comprehensive info for UI to refresh
        return {
            "success": True,
            "data": data,
            "projectInfo": get_current_project_info(),
            "cacheRoot": str(_cache_root) if _cache_root else None,
            "projectCache": str(project_cache),
            "filename": filename,
        }
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
        
        print(f"[PROJECT] Saved: {filename}", flush=True)
        
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
    
    # Import DEFAULTS from main module
    from fuk_web_server import DEFAULTS
    
    # Create project state populated with defaults from defaults.json
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
            "image": {
                "positive_prompt": DEFAULTS.get("image", {}).get("positive_prompt", ""),
                "negative_prompt": DEFAULTS.get("image", {}).get("negative_prompt", ""),
                "width": DEFAULTS.get("image", {}).get("width", 1344),
                "aspect_ratio": DEFAULTS.get("image", {}).get("aspect_ratio", "Widescreen"),
                "infer_steps": DEFAULTS.get("image", {}).get("infer_steps", 20),
                "guidance_scale": DEFAULTS.get("image", {}).get("guidance_scale", 2.1),
                "flow_shift": DEFAULTS.get("image", {}).get("flow_shift", 2.1),
                "blocks_to_swap": DEFAULTS.get("image", {}).get("blocks_to_swap", 0),
                "lora_multiplier": DEFAULTS.get("image", {}).get("lora_multiplier", 1.0),
                "seed": DEFAULTS.get("image", {}).get("seed"),
            },
            "video": {
                "task": DEFAULTS.get("video", {}).get("task", "i2v-A14B"),
                "positive_prompt": DEFAULTS.get("video", {}).get("positive_prompt", ""),
                "negative_prompt": DEFAULTS.get("video", {}).get("negative_prompt", ""),
                "length": DEFAULTS.get("video", {}).get("length", 41),
                "scale_factor": DEFAULTS.get("video", {}).get("scale_factor", 1.0),
                "infer_steps": DEFAULTS.get("video", {}).get("infer_steps", 20),
                "guidance_scale": DEFAULTS.get("video", {}).get("guidance_scale", 5.0),
                "flow_shift": DEFAULTS.get("video", {}).get("flow_shift", 2.1),
                "blocks_to_swap": DEFAULTS.get("video", {}).get("blocks_to_swap", 5),
                "lora_multiplier": DEFAULTS.get("video", {}).get("lora_multiplier", 1.0),
                "seed": DEFAULTS.get("video", {}).get("seed"),
            },
            "preprocess": DEFAULTS.get("preprocess", {}),
            "postprocess": DEFAULTS.get("postprocess", {}),
            "export": DEFAULTS.get("export", {}),
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
        
        print(f"[PROJECT] Created new project: {filename}", flush=True)
        
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

@router.get("/defaults")
async def get_defaults():
    """Get user defaults from defaults.json"""
    from fuk_web_server import DEFAULTS
    return DEFAULTS

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
    except RuntimeError:
        return {"generations": [], "error": "Project system not initialized", "hasMore": False}
    
    if not project_cache.exists():
        return {"generations": [], "hasMore": False}
    
    # Parse pinned IDs
    pinned_ids = set(p.strip() for p in pinned.split(",") if p.strip())
    
    # Calculate cutoff time
    if days > 0:
        cutoff = datetime.now() - timedelta(days=days)
    else:
        cutoff = None
    
    generations = []
    has_more = False
    
    # Scan generation directories
    for gen_dir in sorted(project_cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not gen_dir.is_dir():
            continue
        
        # Skip non-generation directories
        if not any(gen_dir.name.startswith(prefix) for prefix in ["img_gen_", "video_", "video_layers_", "preprocess_", "layers_", "interpolate_", "upscale_", "import_"]):
            continue
        
        # Get path relative to cache root for URL building
        try:
            rel_path = gen_dir.relative_to(_cache_root)
        except ValueError:
            rel_path = Path(gen_dir.name)
        
        # Get modification time
        mtime = datetime.fromtimestamp(gen_dir.stat().st_mtime)
        
        # Check if within date range or pinned (check both full path and just dir name for backwards compat)
        is_pinned = str(rel_path) in pinned_ids or gen_dir.name in pinned_ids
        if cutoff and mtime < cutoff and not is_pinned:
            has_more = True
            continue
        
        # Load metadata if exists
        metadata_path = gen_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Find preview image/video
        preview = None
        gen_type = "unknown"
        source_path = None  # For imports
        
        # Handle imports specially - they reference external files
        if gen_dir.name.startswith("import_") and metadata.get("source_path"):
            source_path = metadata["source_path"]
            # Strip leading slash for URL path
            preview = f"api/project/files{source_path}"
            gen_type = metadata.get("media_type", "import")
        # Standard generation outputs
        elif (gen_dir / "generated.png").exists():
            preview = f"api/project/cache/{rel_path}/generated.png"
            gen_type = "image"
        elif (gen_dir / "generated.mp4").exists():
            preview = f"api/project/cache/{rel_path}/generated.mp4"
            gen_type = "video"
        # Preprocessor outputs (processed_*.png)
        elif (gen_dir / "preprocessed.png").exists():
            preview = f"api/project/cache/{rel_path}/preprocessed.png"
            gen_type = "preprocess"
        # Layers outputs (depth, normals, crypto) - both image and video
        elif gen_dir.name.startswith("layers_") or gen_dir.name.startswith("video_layers_"):
            is_video_layers = gen_dir.name.startswith("video_layers_")
            
            # Check for layer files (depth, normals, crypto)
            # Handle both simple names (depth.png) and hashed names (depth_abc123_456.png)
            layer_files = []
            available_layers = {}  # Track all available layers
            
            for layer_name in ["depth", "normals", "crypto"]:
                # Priority: video > EXR > PNG
                # Check for exact match first, then pattern match
                found = False
                
                # Video
                if (gen_dir / f"{layer_name}.mp4").exists():
                    layer_files.append((f"{layer_name}.mp4", layer_name))
                    available_layers[layer_name] = f"api/project/cache/{rel_path}/{layer_name}.mp4"
                    found = True
                elif list(gen_dir.glob(f"{layer_name}_*.mp4")):
                    matches = list(gen_dir.glob(f"{layer_name}_*.mp4"))
                    layer_files.append((matches[0].name, layer_name))
                    available_layers[layer_name] = f"api/project/cache/{rel_path}/{matches[0].name}"
                    found = True
                
                # EXR
                if not found:
                    if (gen_dir / f"{layer_name}.exr").exists():
                        layer_files.append((f"{layer_name}.exr", layer_name))
                        available_layers[layer_name] = f"api/project/cache/{rel_path}/{layer_name}.exr"
                        found = True
                    elif list(gen_dir.glob(f"{layer_name}_*.exr")):
                        matches = list(gen_dir.glob(f"{layer_name}_*.exr"))
                        layer_files.append((matches[0].name, layer_name))
                        available_layers[layer_name] = f"api/project/cache/{rel_path}/{matches[0].name}"
                        found = True
                
                # PNG
                if not found:
                    if (gen_dir / f"{layer_name}.png").exists():
                        layer_files.append((f"{layer_name}.png", layer_name))
                        available_layers[layer_name] = f"api/project/cache/{rel_path}/{layer_name}.png"
                    elif list(gen_dir.glob(f"{layer_name}_*.png")):
                        matches = list(gen_dir.glob(f"{layer_name}_*.png"))
                        layer_files.append((matches[0].name, layer_name))
                        available_layers[layer_name] = f"api/project/cache/{rel_path}/{matches[0].name}"
            
            # Determine preview and type
            # For video layers, check for source.mp4 first, then source.png
            # For image layers, check source.png
            preview_found = False
            
            if is_video_layers and (gen_dir / "source.mp4").exists():
                preview = f"api/project/cache/{rel_path}/source.mp4"
                available_layers["beauty"] = preview  # Include source as beauty
                gen_type = "layers"
                metadata["available_layers"] = available_layers
                metadata["layer_count"] = len(available_layers)
                metadata["is_video"] = True
                preview_found = True
            elif (gen_dir / "source.png").exists():
                preview = f"api/project/cache/{rel_path}/source.png"
                available_layers["beauty"] = preview  # Include source as beauty
                gen_type = "layers"
                # Store available layers in metadata for UI
                metadata["available_layers"] = available_layers
                metadata["layer_count"] = len(available_layers)
                metadata["is_video"] = is_video_layers
                preview_found = True
            
            # For video layers without source file, try to get source from metadata
            if not preview_found and is_video_layers and metadata.get("source_image"):
                source_path = metadata["source_image"]
                # Convert to API path if it's an absolute path
                if source_path.startswith("/"):
                    available_layers["beauty"] = f"api/project/files{source_path}"
                elif source_path.startswith("api/"):
                    available_layers["beauty"] = source_path
                else:
                    # Try to resolve as relative to cache
                    available_layers["beauty"] = f"api/project/cache/{source_path}"
            
            if not preview_found and layer_files:
                # Fallback to first layer as preview if source missing
                preview_file, layer_type = layer_files[0]
                preview = f"api/project/cache/{rel_path}/{preview_file}"
                gen_type = "layers"
                metadata["available_layers"] = available_layers
                metadata["layer_count"] = len(available_layers)
                metadata["is_video"] = is_video_layers
                preview_found = True
            
            if not preview_found:
                # No layers found at all, skip this directory
                continue
        else:
            # Look for pattern-based outputs
            # Video preprocessing: {method}.mp4 in preprocess_video_* folders
            if gen_dir.name.startswith("preprocess_video_"):
                # Look for method-named video files (canny.mp4, depth.mp4, etc.)
                video_files = list(gen_dir.glob("*.mp4"))
                if video_files:
                    # Skip source.mp4 if it exists, prefer processed output
                    for vid in video_files:
                        if vid.name != "source.mp4":
                            preview = f"api/project/cache/{rel_path}/{vid.name}"
                            gen_type = "preprocess"
                            # Mark as video preprocessing for frontend
                            metadata["subtype"] = "video"
                            break
                    # If only source.mp4 exists, use it
                    if not preview and video_files:
                        preview = f"api/project/cache/{rel_path}/{video_files[0].name}"
                        gen_type = "preprocess"
                        metadata["subtype"] = "video"
            # Image preprocessors: processed.png or processed_*.png
            elif (gen_dir / "processed.png").exists():
                preview = f"api/project/cache/{rel_path}/processed.png"
                gen_type = "preprocess"
            elif list(gen_dir.glob("processed_*.png")):
                processed_files = list(gen_dir.glob("processed_*.png"))
                preview_file = processed_files[0].name
                preview = f"api/project/cache/{rel_path}/{preview_file}"
                gen_type = "preprocess"
            # Interpolation: interpolated_*.mp4
            elif list(gen_dir.glob("interpolated_*.mp4")):
                interp_files = list(gen_dir.glob("interpolated_*.mp4"))
                preview_file = interp_files[0].name
                preview = f"api/project/cache/{rel_path}/{preview_file}"
                gen_type = "video"
            # Upscaling: upscaled_*.png or upscaled_*.mp4
            elif list(gen_dir.glob("upscaled_*.png")):
                upscaled_files = list(gen_dir.glob("upscaled_*.png"))
                preview_file = upscaled_files[0].name
                preview = f"api/project/cache/{rel_path}/{preview_file}"
                gen_type = "image"
            elif list(gen_dir.glob("upscaled_*.mp4")):
                upscaled_files = list(gen_dir.glob("upscaled_*.mp4"))
                preview_file = upscaled_files[0].name
                preview = f"api/project/cache/{rel_path}/{preview_file}"
                gen_type = "upscale"
                metadata["subtype"] = "video"
            # Fallback: find any image or video
            else:
                image_exts = ['.png', '.jpg', '.jpeg', '.webp', '.exr']
                video_exts = ['.mp4', '.mov', '.webm', '.avi']
                
                # Try to find any image
                for ext in image_exts:
                    image_files = list(gen_dir.glob(f"*{ext}"))
                    if image_files:
                        # Skip source.png and .thumb.jpg files, prefer other images
                        for img in image_files:
                            if img.name != "source.png" and not img.name.endswith('.thumb.jpg'):
                                preview = f"api/project/cache/{rel_path}/{img.name}"
                                gen_type = "image"
                                break
                        if preview:
                            break
                
                # If no image, try video
                if not preview:
                    for ext in video_exts:
                        video_files = list(gen_dir.glob(f"*{ext}"))
                        if video_files:
                            preview = f"api/project/cache/{rel_path}/{video_files[0].name}"
                            gen_type = "video"
                            break
        # Check for video thumbnail (extract filename from preview URL)
        thumbnail_url = None
        if preview and (gen_type == "video" or 
                       (gen_type == "preprocess" and metadata.get("subtype") == "video") or
                       (gen_type == "upscale" and metadata.get("subtype") == "video")):
            # Extract filename from preview URL (e.g., "api/project/cache/xxx/upscaled_4x.mp4" -> "upscaled_4x.mp4")
            preview_filename = preview.split("/")[-1] if "/" in preview else preview
            thumb_candidates = [
                gen_dir / preview_filename.replace('.mp4', '.thumb.jpg'),
                gen_dir / "thumbnail.jpg",
            ]
            for thumb in thumb_candidates:
                if thumb.exists():
                    thumbnail_url = f"api/project/cache/{rel_path}/{thumb.name}"
                    break

        # Build generation object
        gen_obj = {
            "id": str(rel_path),  # Use full relative path as ID
            "name": metadata.get("display_name"),  # For imports
            "type": gen_type,
            "subtype": metadata.get("subtype"),  # For video preprocessing detection
            "preview": preview,
            "path": preview,  # Alias for drag-drop compatibility
            "sourcePath": source_path,  # Original path for imports
            "date": mtime.strftime("%Y-%m-%d"),
            "timestamp": mtime.strftime("%H:%M:%S"),
            "prompt": metadata.get("prompt", ""),
            "seed": metadata.get("seed"),
            "model": metadata.get("model", ""),
            "pinned": is_pinned,
            "thumbnailUrl": thumbnail_url,  # ADD THIS
        }
        
        # Include metadata for special types that need extra data
        # Layers need available_layers for drag-to-export functionality
        if gen_type == "layers" and metadata.get("available_layers"):
            gen_obj["metadata"] = {
                "available_layers": metadata["available_layers"],
                "layer_count": metadata.get("layer_count", len(metadata["available_layers"])),
                "is_video": metadata.get("is_video", False),
            }
        
        generations.append(gen_obj)
    
    return {"generations": generations, "hasMore": has_more}

@router.get("/generations/layers")
async def list_layer_generations(
    days: int = Query(default=7, description="Number of days to load"),
    pinned: str = Query(default="", description="Comma-separated list of pinned IDs")
):
    """
    List layer generations (depth, normals, crypto) in the project cache.
    Returns entries for the Layers history panel.
    """
    print(f"[HISTORY] Fetching layer generations (days={days})", flush=True)
    
    if not _cache_root:
        return []
    
    try:
        project_cache = get_project_cache_dir()
    except RuntimeError:
        return []
    
    if not project_cache.exists():
        return []
    
    pinned_ids = set(p.strip() for p in pinned.split(",") if p.strip())
    
    if days > 0:
        cutoff = datetime.now() - timedelta(days=days)
    else:
        cutoff = None
    
    entries = []
    
    # Look for layer outputs in preprocess directories
    layer_types = {
        "depth.mp4": ("depth", "Depth Video"),
        "depth.png": ("depth", "Depth"),
        "depth.exr": ("depth", "Depth EXR"),
        "normals.mp4": ("normals", "Normals Video"),
        "normals.png": ("normals", "Normals"),
        "normals.exr": ("normals", "Normals EXR"),
        "crypto.mp4": ("crypto", "Cryptomatte Video"),
        "crypto.png": ("crypto", "Cryptomatte"),
        "crypto.exr": ("crypto", "Cryptomatte EXR"),
    }
    
    for gen_dir in sorted(project_cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not gen_dir.is_dir():
            continue
        
        if not gen_dir.name.startswith("preprocess_") and not gen_dir.name.startswith("layers_"):
            continue
        
        # Get path relative to cache root for URL building
        try:
            rel_path = gen_dir.relative_to(_cache_root)
        except ValueError:
            rel_path = Path(gen_dir.name)
        
        # Check for layer files
        for filename, (layer_type, label) in layer_types.items():
            layer_path = gen_dir / filename
            if not layer_path.exists():
                continue
            
            stat = layer_path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            # Date filter - use full relative path for IDs
            layer_id = f"{rel_path}/{layer_type}"
            is_pinned = layer_id in pinned_ids or f"{gen_dir.name}/{layer_type}" in pinned_ids
            if cutoff and mtime < cutoff and not is_pinned:
                continue
            
            # Load metadata
            metadata_path = gen_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except:
                    pass
            
            api_path = f"api/project/cache/{rel_path}/{filename}"
            
            entries.append({
                "id": layer_id,
                "name": f"{rel_path}/{label}",
                "type": "layers",
                "subtype": layer_type,
                "path": api_path,
                "preview": api_path,
                "date": mtime.strftime("%Y-%m-%d"),
                "timestamp": mtime.strftime("%H:%M:%S"),
                "size": stat.st_size,
                "prompt": metadata.get("prompt", ""),
                "seed": metadata.get("seed"),
                "model": metadata.get("model", ""),
                "pinned": is_pinned,
                "parentDir": str(rel_path),
            })
    
    return entries

@router.post("/import")
async def register_import(data: dict = Body(...)):
    """
    Register an imported asset in history.
    
    Creates an import_### directory with metadata pointing to the original file.
    No copying or symlinking - just a reference. This means:
    - Zero disk duplication
    - Re-renders (e.g., Blender depth passes) automatically update
    - Files served directly from original location
    
    Args (in body):
        path: Source file path (absolute)
        name: Display name (optional)
        auto_pin: Whether to auto-pin (default: True)
    
    Returns:
        Import info including the new history ID
    """
    source_path = data.get("path")
    display_name = data.get("name")
    auto_pin = data.get("auto_pin", True)
    
    if not source_path:
        raise HTTPException(status_code=400, detail="No path provided")
    
    source = Path(source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {source_path}")
    
    # Get project cache directory
    try:
        project_cache = get_project_cache_dir()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Project system not initialized")
    
    # Create import directory (just for metadata)
    import_dir = get_generation_output_dir("import")
    
    # Determine file name
    if not display_name:
        display_name = source.name
    
    # Determine media type
    ext = source.suffix.lower()
    if ext in ['.mp4', '.mov', '.webm', '.avi', '.mkv']:
        media_type = "video"
    elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']:
        media_type = "image"
    elif ext in ['.exr', '.dpx']:
        media_type = "exr"
    else:
        media_type = "unknown"
    
    # Save metadata with reference to original path (no copying!)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source_path": str(source),
        "display_name": display_name,
        "media_type": media_type,
        "link_type": "reference",  # Just a pointer, no duplication
        "imported": True,
    }
    
    metadata_path = import_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Build the import ID (relative path from cache root)
    try:
        rel_path = import_dir.relative_to(_cache_root)
        import_id = str(rel_path)
    except ValueError:
        import_id = import_dir.name
    
    # Build API path for serving
    api_path = f"api/project/files{source}"  # Use files endpoint for external paths
    
    print(f"[IMPORT] Registered: {source.name} -> {import_id} (reference only, no copy)", flush=True)
    
    return {
        "success": True,
        "id": import_id,
        "path": api_path,
        "name": display_name,
        "media_type": media_type,
        "auto_pin": auto_pin,
    }

@router.delete("/generations/{gen_id:path}")
async def delete_generation(gen_id: str):
    """Delete a generation directory or specific layer file"""
    if not _cache_root:
        raise HTTPException(status_code=400, detail="No project loaded")
    
    # gen_id is now relative to _cache_root, e.g.:
    # - "fuktest_shot08_260110/video_002" 
    # - "fuktest_shot08_260110/preprocess_001/depth" (layer-specific)
    
    # For layer-specific IDs, remove the layer type suffix to get the directory
    # Layer IDs end with /depth, /normals, /crypto
    layer_suffixes = ['/depth', '/normals', '/crypto']
    dir_id = gen_id
    for suffix in layer_suffixes:
        if gen_id.endswith(suffix):
            dir_id = gen_id[:-len(suffix)]
            break
    
    # Build path relative to cache root
    gen_dir = _cache_root / dir_id
    
    # Security check - must be within cache root
    try:
        gen_dir.resolve().relative_to(_cache_root.resolve())
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
        print(f"[HISTORY] Deleted generation: {dir_id}", flush=True)
        return {"success": True, "deleted": dir_id}
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
    if not _cache_root and not _default_cache_root:
        raise HTTPException(status_code=503, detail="Cache not initialized. Please restart the server.")
    
    # Try current cache root first
    full_path = None
    tried_paths = []
    
    if _cache_root:
        candidate = _cache_root / file_path
        tried_paths.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            full_path = candidate
    
    # Fall back to default cache if not found
    if not full_path and _default_cache_root and _default_cache_root != _cache_root:
        candidate = _default_cache_root / file_path
        tried_paths.append(str(candidate))
        if candidate.exists() and candidate.is_file():
            full_path = candidate
    
    if not full_path:
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
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Determine content type
    content_type, _ = mimetypes.guess_type(str(full_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
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
    return _cache_root

def get_default_cache_root() -> Optional[Path]:
    """Get the default cache root (original location before any project was opened)"""
    return _default_cache_root

# ============================================================================
# External File Serving (for imports and files outside cache)
# ============================================================================

@router.get("/files/{file_path:path}")
async def serve_external_file(file_path: str):
    """
    Serve files from absolute paths (for imports and external references).
    
    URL: /api/project/files/{absolute_path_without_leading_slash}
    Example: /api/project/files/home/brad/Projects/projects/fukTests/assets/images/test.png
    
    Security: Only serves files from allowed directories (project assets, common media paths)
    """
    # Reconstruct absolute path
    full_path = Path("/" + file_path)
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")
    
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {full_path}")
    
    # Security: Check if file is in an allowed location
    allowed_roots = [
        Path.home(),  # User's home directory
        Path("/tmp"),  # Temp files
    ]
    
    # Add project folder if loaded
    if _project_folder:
        allowed_roots.append(Path(_project_folder))
    
    # Check if path is under an allowed root
    is_allowed = False
    resolved_path = full_path.resolve()
    
    for allowed_root in allowed_roots:
        try:
            resolved_path.relative_to(allowed_root.resolve())
            is_allowed = True
            break
        except ValueError:
            continue
    
    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied - path not in allowed locations")
    
    # Determine content type
    content_type, _ = mimetypes.guess_type(str(full_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
    return FileResponse(
        path=full_path,
        media_type=content_type,
        filename=full_path.name
    )