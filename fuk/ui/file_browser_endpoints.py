# file_browser_endpoints.py
"""
File Browser API Endpoints for FUK Web Server

Provides endpoints for native file dialogs via tkinter.
Since tkinter can't run in the async event loop, we spawn
a subprocess to handle the dialog.

Add to fuk_web_server.py:
    from file_browser_endpoints import setup_file_browser_routes
    setup_file_browser_routes(app, log)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import json
import sys
import os


# ============================================================================
# Request/Response Models
# ============================================================================

class FileDialogRequest(BaseModel):
    """Request to open a file dialog"""
    title: str = "Select Media"
    initial_dir: Optional[str] = None
    multiple: bool = True
    detect_sequences: bool = True
    filter: str = "all"  # all, images, videos, exr


class DirectoryDialogRequest(BaseModel):
    """Request to open a directory dialog"""
    title: str = "Select Directory"
    initial_dir: Optional[str] = None


class ScanDirectoryRequest(BaseModel):
    """Request to scan a directory for media"""
    directory: str
    recursive: bool = False
    detect_sequences: bool = True


class MediaFile(BaseModel):
    """Media file info returned from browser"""
    path: str
    display_name: str
    media_type: str  # image, video, sequence
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    frame_count: Optional[int] = None
    frame_pattern: Optional[str] = None


class FileDialogResponse(BaseModel):
    """Response from file dialog"""
    success: bool
    files: List[MediaFile] = []
    error: Optional[str] = None


class DirectoryDialogResponse(BaseModel):
    """Response from directory dialog"""
    success: bool
    directory: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_file_browser_script() -> Path:
    """Get path to file_browser.py script"""
    # Check same directory as this file
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


def run_file_browser(args: List[str], timeout: int = 300) -> Dict[str, Any]:
    """
    Run file_browser.py as subprocess
    
    Args:
        args: Arguments to pass to file_browser.py
        timeout: Max seconds to wait (dialogs can take a while)
        
    Returns:
        Parsed JSON response from script
    """
    script_path = get_file_browser_script()
    
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
            return {"success": False, "error": error_msg, "files": []}
        
        # Parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid response: {e}", "files": []}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Dialog timed out", "files": []}
    except Exception as e:
        return {"success": False, "error": str(e), "files": []}


# ============================================================================
# Route Setup
# ============================================================================

def setup_file_browser_routes(app, log):
    """
    Setup file browser API routes
    
    Args:
        app: FastAPI application
        log: Logger instance
    """
    
    @app.post("/api/browser/open", response_model=FileDialogResponse)
    async def open_file_dialog(request: FileDialogRequest):
        """
        Open native file dialog for selecting media files
        
        Supports images, videos, and image sequences.
        Sequences are automatically detected and collapsed.
        """
        log.info("FileBrowser", f"Opening file dialog: {request.title}")
        
        args = [
            "open",
            "--title", request.title,
            "--filter", request.filter,
        ]
        
        if request.initial_dir:
            args.extend(["--initial-dir", request.initial_dir])
        
        if request.multiple:
            args.append("--multiple")
        else:
            args.append("--no-multiple")
        
        if request.detect_sequences:
            args.append("--detect-sequences")
        else:
            args.append("--no-detect-sequences")
        
        result = run_file_browser(args)
        
        if result.get("success"):
            files = [MediaFile(**f) for f in result.get("files", [])]
            log.success("FileBrowser", f"Selected {len(files)} file(s)")
            return FileDialogResponse(success=True, files=files)
        else:
            error = result.get("error", "Dialog failed")
            log.error("FileBrowser", error)
            return FileDialogResponse(success=False, error=error)
    
    @app.post("/api/browser/directory", response_model=DirectoryDialogResponse)
    async def open_directory_dialog(request: DirectoryDialogRequest):
        """Open native directory selection dialog"""
        log.info("FileBrowser", f"Opening directory dialog: {request.title}")
        
        args = [
            "directory",
            "--title", request.title,
        ]
        
        if request.initial_dir:
            args.extend(["--initial-dir", request.initial_dir])
        
        result = run_file_browser(args)
        
        if result.get("success"):
            directory = result.get("directory")
            log.success("FileBrowser", f"Selected: {directory}")
            return DirectoryDialogResponse(success=True, directory=directory)
        else:
            error = result.get("error", "Dialog failed")
            log.error("FileBrowser", error)
            return DirectoryDialogResponse(success=False, error=error)
    
    @app.post("/api/browser/scan", response_model=FileDialogResponse)
    async def scan_directory(request: ScanDirectoryRequest):
        """
        Scan a directory for media files
        
        Returns all images, videos, and sequences found.
        """
        log.info("FileBrowser", f"Scanning: {request.directory}")
        
        args = [
            "scan",
            "--initial-dir", request.directory,
        ]
        
        if request.recursive:
            args.append("--recursive")
        
        if request.detect_sequences:
            args.append("--detect-sequences")
        else:
            args.append("--no-detect-sequences")
        
        result = run_file_browser(args, timeout=60)
        
        if result.get("success"):
            files = [MediaFile(**f) for f in result.get("files", [])]
            log.success("FileBrowser", f"Found {len(files)} file(s)")
            return FileDialogResponse(success=True, files=files)
        else:
            error = result.get("error", "Scan failed")
            log.error("FileBrowser", error)
            return FileDialogResponse(success=False, error=error)
    
    @app.get("/api/browser/recent")
    async def get_recent_directories():
        """Get recently used directories (from project history)"""
        # This could be expanded to track actual recent directories
        home = str(Path.home())
        return {
            "directories": [
                {"path": home, "name": "Home"},
                {"path": str(Path.home() / "Downloads"), "name": "Downloads"},
            ]
        }
    
    log.success("Routes", "File browser endpoints registered")