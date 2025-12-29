"""
FUK Project API Endpoints
Handles project files, cache routing, and serves project cache files

=============================================================================
INTEGRATION GUIDE
=============================================================================

1. Copy this file next to fuk_web_server.py

2. At the TOP of fuk_web_server.py, add:

    from project_endpoints import (
        setup_project_routes, 
        get_generation_output_dir, 
        build_output_paths,
        get_project_relative_url
    )

3. After creating the FastAPI app, add:

    setup_project_routes(app)

4. In run_image_generation(), REPLACE the output path logic:

    # OLD CODE:
    # gen_dir = image_manager.create_generation_dir()
    # paths = image_manager.get_output_paths(gen_dir)
    
    # NEW CODE:
    gen_dir = get_generation_output_dir("img_gen")
    paths = build_output_paths(gen_dir)
    
5. When setting outputs at the end of generation, use:

    outputs = {
        "png": get_project_relative_url(paths["generated_png"]),
    }
    
    # If you also have EXR:
    if "exr" in outputs:
        outputs["exr"] = get_project_relative_url(paths["generated_exr"])

=============================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import threading

# ============================================================================
# Global Project State
# ============================================================================

_project_state = {
    "folder": None,
    "config": {
        "versionFormat": "date",
        "cacheFolder": "cache",
    }
}


def get_project_folder() -> Optional[Path]:
    """Get current project folder path"""
    if _project_state["folder"]:
        return Path(_project_state["folder"])
    return None


def get_cache_folder() -> Optional[Path]:
    """Get cache folder path for current project"""
    folder = get_project_folder()
    if folder:
        cache = folder / _project_state["config"]["cacheFolder"]
        cache.mkdir(exist_ok=True)
        return cache
    return None


# ============================================================================
# OUTPUT PATH HELPERS - Use these in fuk_web_server.py
# ============================================================================

def get_generation_output_dir(generation_type: str = "img_gen") -> Path:
    """
    Get output directory for a new generation.
    Creates timestamped subfolder in project cache if project is set,
    otherwise falls back to default outputs folder.
    
    Args:
        generation_type: Type prefix for folder name (img_gen, vid_gen, etc.)
        
    Returns:
        Path to generation folder
    """
    cache = get_cache_folder()
    
    if cache:
        # Project cache path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_folder = cache / f"{generation_type}_{timestamp}"
        gen_folder.mkdir(parents=True, exist_ok=True)
        return gen_folder
    else:
        # Fallback to default outputs folder (original behavior)
        default = Path("outputs") / "image" / datetime.now().strftime("%Y-%m-%d")
        default.mkdir(parents=True, exist_ok=True)
        
        # Find next generation number
        existing = [d for d in default.iterdir() if d.is_dir() and d.name.startswith("generation_")]
        next_num = len(existing) + 1
        gen_folder = default / f"generation_{next_num:03d}"
        gen_folder.mkdir()
        
        return gen_folder


def build_output_paths(gen_dir: Path) -> Dict[str, Path]:
    """
    Build standard output paths for a generation directory.
    
    Returns dict with:
        - generated_png: Path to PNG output
        - generated_exr: Path to EXR output  
        - source: Path to source/control image copy
        - metadata: Path to metadata.json
    """
    return {
        "generated_png": gen_dir / "generated.png",
        "generated_exr": gen_dir / "generated.exr",
        "source": gen_dir / "source.png",
        "metadata": gen_dir / "metadata.json"
    }


def get_project_relative_url(file_path: Path) -> str:
    """
    Convert a file path to a URL that can be served by the API.
    
    If in project cache: returns "project-cache/img_gen_20251229.../generated.png"
    If in outputs: returns "image/2025-12-29/generation_001/generated.png"
    
    The frontend will prepend the appropriate base path.
    """
    cache = get_cache_folder()
    
    if cache:
        try:
            # Check if path is in cache folder
            file_path = Path(file_path).resolve()
            cache_resolved = cache.resolve()
            
            if str(file_path).startswith(str(cache_resolved)):
                relative = file_path.relative_to(cache_resolved)
                return f"project-cache/{relative}"
        except:
            pass
    
    # Fallback - try to make relative to outputs
    try:
        outputs_root = Path("outputs").resolve()
        file_resolved = Path(file_path).resolve()
        
        if str(file_resolved).startswith(str(outputs_root)):
            return str(file_resolved.relative_to(outputs_root))
    except:
        pass
    
    # Last resort - just the filename
    return str(Path(file_path).name)


def save_generation_metadata(
    gen_dir: Path,
    prompt: str,
    model: str,
    seed: Optional[int],
    image_size: tuple,
    infer_steps: int = 20,
    guidance_scale: float = 2.1,
    negative_prompt: str = "",
    flow_shift: Optional[float] = None,
    lora: Optional[str] = None,
    lora_multiplier: float = 1.0,
    control_image: Optional[str] = None,
    **kwargs
) -> Path:
    """
    Save generation metadata to JSON file.
    Call this in run_image_generation after the generation completes.
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt": {
            "original": prompt,
            "enhanced": prompt  # Can be different if you have prompt enhancement
        },
        "model": model,
        "seed": seed,
        "image_size": list(image_size),
        "infer_steps": infer_steps,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "flow_shift": flow_shift,
        "lora": lora,
        "lora_multiplier": lora_multiplier,
        "control_image": control_image,
        **kwargs
    }
    
    metadata_path = gen_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


# ============================================================================
# Request Models
# ============================================================================

class SetFolderRequest(BaseModel):
    path: str


class NewProjectRequest(BaseModel):
    projectName: str
    shotNumber: str = "01"


# ============================================================================
# Folder Browser (tkinter)
# ============================================================================

_dialog_result = {"path": None, "done": False}


def _open_folder_dialog() -> Optional[str]:
    """Open native folder selection dialog"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        folder_path = filedialog.askdirectory(
            title="Select FUK Project Folder (ProjectName/Projects/Fuk/)",
            mustexist=True
        )
        
        root.destroy()
        return folder_path if folder_path else None
        
    except Exception as e:
        print(f"Folder dialog error: {e}")
        return None


def _run_dialog_thread():
    """Run dialog in thread"""
    global _dialog_result
    _dialog_result["path"] = _open_folder_dialog()
    _dialog_result["done"] = True


# ============================================================================
# Route Setup
# ============================================================================

def setup_project_routes(app: FastAPI):
    """Add project management routes to FastAPI app"""
    
    # ========================================================================
    # Serve files from project cache
    # ========================================================================
    
    @app.get("/project-cache/{file_path:path}")
    async def serve_project_cache(file_path: str):
        """Serve files from project cache folder"""
        cache = get_cache_folder()
        if not cache:
            raise HTTPException(status_code=404, detail="No project folder set")
        
        full_path = cache / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        # Security check
        try:
            full_path.resolve().relative_to(cache.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(full_path)
    
    # ========================================================================
    # Folder Browser
    # ========================================================================
    
    @app.post("/api/project/browse-folder")
    async def browse_folder():
        """Open native folder browser dialog"""
        global _dialog_result
        
        _dialog_result = {"path": None, "done": False}
        
        thread = threading.Thread(target=_run_dialog_thread)
        thread.start()
        thread.join(timeout=120)
        
        if not _dialog_result["done"]:
            return {"path": None, "error": "Dialog timeout"}
        
        if _dialog_result["path"]:
            folder_path = Path(_dialog_result["path"])
            _project_state["folder"] = str(folder_path)
            
            # Ensure cache exists
            cache_folder = folder_path / _project_state["config"]["cacheFolder"]
            cache_folder.mkdir(exist_ok=True)
            
            # Load config if exists
            config_path = folder_path / "fuk_config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        user_config = json.load(f)
                        _project_state["config"].update(user_config)
                except:
                    pass
            
            return {"path": str(folder_path), "success": True}
        
        return {"path": None, "cancelled": True}
    
    # ========================================================================
    # Folder Management
    # ========================================================================
    
    @app.post("/api/project/set-folder")
    async def set_project_folder(request: SetFolderRequest):
        """Set project folder by path"""
        folder_path = Path(request.path).expanduser().resolve()
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {folder_path}")
        
        _project_state["folder"] = str(folder_path)
        
        cache_folder = folder_path / _project_state["config"]["cacheFolder"]
        cache_folder.mkdir(exist_ok=True)
        
        config_path = folder_path / "fuk_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    _project_state["config"].update(user_config)
            except Exception as e:
                print(f"Warning: Could not load project config: {e}")
        
        return {
            "folder": str(folder_path),
            "config": _project_state["config"],
            "success": True
        }
    
    @app.get("/api/project/current")
    async def get_current_project():
        """Get current project state"""
        return {
            "folder": _project_state["folder"],
            "config": _project_state["config"],
            "isSet": _project_state["folder"] is not None
        }
    
    # ========================================================================
    # Project Files CRUD
    # ========================================================================
    
    @app.get("/api/project/list")
    async def list_project_files():
        """List project files"""
        if not _project_state["folder"]:
            raise HTTPException(status_code=400, detail="No project folder set")
        
        folder = Path(_project_state["folder"])
        files = []
        
        for json_file in folder.glob("*.json"):
            if json_file.name in ["fuk_config.json"]:
                continue
            
            stat = json_file.stat()
            files.append({
                "name": json_file.name,
                "path": str(json_file),
                "modifiedAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
            })
        
        files.sort(key=lambda f: f["modifiedAt"], reverse=True)
        
        return {"folder": str(folder), "files": files}
    
    @app.get("/api/project/load/{filename}")
    async def load_project(filename: str):
        """Load a project file"""
        if not _project_state["folder"]:
            raise HTTPException(status_code=400, detail="No project folder set")
        
        file_path = Path(_project_state["folder"]) / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        try:
            with open(file_path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    @app.post("/api/project/save/{filename}")
    async def save_project(filename: str, state: Dict[str, Any]):
        """Save project state"""
        if not _project_state["folder"]:
            raise HTTPException(status_code=400, detail="No project folder set")
        
        file_path = Path(_project_state["folder"]) / filename
        
        if "meta" not in state:
            state["meta"] = {}
        
        state["meta"]["updatedAt"] = datetime.now().isoformat()
        
        if not file_path.exists():
            state["meta"]["createdAt"] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        return {"success": True, "filename": filename, "path": str(file_path)}
    
    @app.post("/api/project/new")
    async def create_new_project(request: NewProjectRequest):
        """Create new project file"""
        if not _project_state["folder"]:
            raise HTTPException(status_code=400, detail="No project folder set")
        
        folder = Path(_project_state["folder"])
        
        if _project_state["config"]["versionFormat"] == "date":
            now = datetime.now()
            version = f"{now.year % 100:02d}{now.month:02d}{now.day:02d}"
        else:
            version = "v01"
        
        shot = request.shotNumber.zfill(2)
        filename = f"{request.projectName}_shot{shot}_{version}.json"
        file_path = folder / filename
        
        if file_path.exists():
            suffix = 'a'
            while (folder / f"{request.projectName}_shot{shot}_{version}{suffix}.json").exists():
                suffix = chr(ord(suffix) + 1)
            version = f"{version}{suffix}"
            filename = f"{request.projectName}_shot{shot}_{version}.json"
            file_path = folder / filename
        
        state = {
            "meta": {
                "version": "1.0",
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat(),
                "fukVersion": "1.0.0"
            },
            "project": {
                "name": request.projectName,
                "shot": shot,
                "version": version
            },
            "tabs": {
                "image": {},
                "video": {},
                "preprocess": {},
                "postprocess": {},
                "export": {}
            },
            "assets": {},
            "lastState": {"activeTab": "image"},
            "notes": ""
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        return {"success": True, "filename": filename, "state": state}
    
    # ========================================================================
    # Config & Cache Info
    # ========================================================================
    
    @app.get("/api/project/config")
    async def get_project_config():
        """Get project config"""
        return _project_state["config"]
    
    @app.get("/api/project/cache-info")
    async def get_cache_info():
        """Get cache folder stats"""
        if not _project_state["folder"]:
            return {"path": None, "exists": False, "size": 0, "itemCount": 0}
        
        cache_folder = Path(_project_state["folder"]) / _project_state["config"]["cacheFolder"]
        
        if not cache_folder.exists():
            return {"path": str(cache_folder), "exists": False, "size": 0, "itemCount": 0}
        
        total_size = 0
        item_count = 0
        
        for item in cache_folder.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
                item_count += 1
        
        return {
            "path": str(cache_folder),
            "exists": True,
            "size": total_size,
            "sizeFormatted": _format_size(total_size),
            "itemCount": item_count,
        }


def _format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
