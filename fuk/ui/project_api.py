"""
FUK Project API Endpoints
Handles project file management, saving, loading, and versioning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

router = APIRouter(prefix="/project", tags=["project"])

# Global state for current project folder
_project_state = {
    "folder": None,
    "config": {
        "versionFormat": "date",  # or "sequential"
        "cacheFolder": "cache",
    }
}


class SetFolderRequest(BaseModel):
    path: str


class ProjectFile(BaseModel):
    name: str
    path: str
    modifiedAt: str
    size: int


class ProjectListResponse(BaseModel):
    folder: str
    files: List[ProjectFile]


class NewVersionRequest(BaseModel):
    baseFilename: str
    newVersion: str


# ============================================================================
# Project Folder Management
# ============================================================================

@router.post("/set-folder")
async def set_project_folder(request: SetFolderRequest):
    """Set the active project folder"""
    
    folder_path = Path(request.path).expanduser().resolve()
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder_path}")
    
    _project_state["folder"] = str(folder_path)
    
    # Ensure cache folder exists
    cache_folder = folder_path / _project_state["config"]["cacheFolder"]
    cache_folder.mkdir(exist_ok=True)
    
    # Try to load project config if it exists
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
    }


@router.get("/list", response_model=ProjectListResponse)
async def list_project_files():
    """List all .json project files in current folder"""
    
    if not _project_state["folder"]:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    folder = Path(_project_state["folder"])
    
    files = []
    for json_file in folder.glob("*.json"):
        # Skip config files
        if json_file.name in ["fuk_config.json"]:
            continue
            
        stat = json_file.stat()
        files.append(ProjectFile(
            name=json_file.name,
            path=str(json_file),
            modifiedAt=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            size=stat.st_size,
        ))
    
    # Sort by modified time (newest first)
    files.sort(key=lambda f: f.modifiedAt, reverse=True)
    
    return ProjectListResponse(
        folder=str(folder),
        files=files,
    )


# ============================================================================
# Project Load/Save
# ============================================================================

@router.get("/load/{filename}")
async def load_project(filename: str):
    """Load a project file"""
    
    if not _project_state["folder"]:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    file_path = Path(_project_state["folder"]) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Project file not found: {filename}")
    
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {e}")


@router.post("/save/{filename}")
async def save_project(filename: str, state: Dict[str, Any]):
    """Save project state to file"""
    
    if not _project_state["folder"]:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    file_path = Path(_project_state["folder"]) / filename
    
    try:
        # Add/update timestamps
        if "meta" not in state:
            state["meta"] = {}
        
        state["meta"]["updatedAt"] = datetime.now().isoformat()
        
        if not file_path.exists():
            state["meta"]["createdAt"] = datetime.now().isoformat()
        
        # Write with nice formatting
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        return {
            "success": True,
            "filename": filename,
            "path": str(file_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save project: {e}")


@router.post("/new-version")
async def create_new_version(request: NewVersionRequest):
    """Create a new version of a project file"""
    
    if not _project_state["folder"]:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    folder = Path(_project_state["folder"])
    base_path = folder / request.baseFilename
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Base file not found: {request.baseFilename}")
    
    # Load base file
    try:
        with open(base_path) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load base file: {e}")
    
    # Generate new filename
    # Parse: projectname_shot##_version.json
    base_name = request.baseFilename.rsplit('.', 1)[0]  # Remove .json
    parts = base_name.rsplit('_', 1)  # Split off version
    
    if len(parts) >= 2:
        new_filename = f"{parts[0]}_{request.newVersion}.json"
    else:
        new_filename = f"{base_name}_{request.newVersion}.json"
    
    new_path = folder / new_filename
    
    # Update metadata
    data["meta"] = data.get("meta", {})
    data["meta"]["createdAt"] = datetime.now().isoformat()
    data["meta"]["updatedAt"] = datetime.now().isoformat()
    data["meta"]["basedOn"] = request.baseFilename
    
    if "project" in data:
        data["project"]["version"] = request.newVersion
    
    # Save new file
    try:
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            "success": True,
            "filename": new_filename,
            "path": str(new_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create new version: {e}")


# ============================================================================
# Project Config
# ============================================================================

@router.get("/config")
async def get_project_config():
    """Get current project configuration"""
    return _project_state["config"]


@router.get("/cache-info")
async def get_cache_info():
    """Get cache folder information"""
    
    if not _project_state["folder"]:
        raise HTTPException(status_code=400, detail="No project folder set")
    
    folder = Path(_project_state["folder"])
    cache_folder = folder / _project_state["config"]["cacheFolder"]
    
    if not cache_folder.exists():
        return {
            "path": str(cache_folder),
            "exists": False,
            "size": 0,
            "itemCount": 0,
        }
    
    # Calculate size and count
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
        "sizeFormatted": format_size(total_size),
        "itemCount": item_count,
    }


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ============================================================================
# Folder Browser (placeholder - needs system integration)
# ============================================================================

@router.post("/browse-folder")
async def browse_folder():
    """
    Placeholder for native folder browser.
    In a real implementation, this would trigger a system dialog.
    For now, returns instructions.
    """
    return {
        "message": "Native folder browser not implemented. Use set-folder with a path.",
        "hint": "Enter the path manually or integrate with a file browser library.",
    }
