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

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import json
import sys
import os

from file_browser import list_directory, resolve_browse_path, selection_for, BROWSER_FILTERS


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


def browse_roots() -> List[Path]:
    """
    Folders the in-app browser may list: the same places the file endpoint
    (/api/project/files) will serve, so every listed image can be previewed.
    """
    from project_endpoints import get_project_folder

    roots = [Path.home(), Path("/tmp")]
    project = get_project_folder()
    if project:
        roots.append(Path(project))
    return [r.resolve() for r in roots]


def _within_roots(path: Path, roots: List[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == r or r in resolved.parents for r in roots)


def browse_listing(raw_path: Optional[str], filter: str, sequences: bool, offset: int, limit: int) -> Dict[str, Any]:
    """Blocking body of /api/browser/list (runs in a worker thread)."""
    roots = browse_roots()
    directory, target, missing = resolve_browse_path(raw_path or str(Path.home()))
    if not _within_roots(directory, roots):
        raise HTTPException(status_code=403, detail=f"Outside the folders FUK can browse: {directory}")

    try:
        entries, aliases = list_directory(directory, filter=filter, detect_sequences=sequences)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {directory}")

    # The entry a pasted file/sequence points at, wherever it falls in the
    # pages, so the UI can pick it straight away.
    selected = selection_for(target, aliases)
    selected_entry = next((e for e in entries if e["path"] == selected), None) if selected else None

    page = entries[offset:offset + limit]
    for entry in page:
        if entry["kind"] in ("image", "video", "file"):
            try:
                st = os.stat(entry["path"])
                entry["size"], entry["mtime"] = st.st_size, st.st_mtime
            except OSError:
                pass

    parent = directory.parent
    return {
        "path": str(directory),
        "parent": str(parent) if parent != directory and _within_roots(parent, roots) else None,
        "home": str(Path.home()),
        "selected": selected,
        "selected_entry": selected_entry,
        "missing": missing,
        "entries": page,
        "total": len(entries),
        "offset": offset,
        "has_more": offset + limit < len(entries),
    }


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
        
        result = await run_in_threadpool(run_file_browser, args)
        
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
        
        result = await run_in_threadpool(run_file_browser, args)
        
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
        
        result = await run_in_threadpool(run_file_browser, args, 60)
        
        if result.get("success"):
            files = [MediaFile(**f) for f in result.get("files", [])]
            log.success("FileBrowser", f"Found {len(files)} file(s)")
            return FileDialogResponse(success=True, files=files)
        else:
            error = result.get("error", "Scan failed")
            log.error("FileBrowser", error)
            return FileDialogResponse(success=False, error=error)
    
    @app.get("/api/browser/list")
    async def browse_list(
        path: Optional[str] = Query(default=None, description="Folder, file, or sequence pattern; pasted paths welcome"),
        filter: str = Query(default="all", description=", ".join(BROWSER_FILTERS)),
        sequences: bool = Query(default=True, description="Collapse frame sequences into one entry"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=300, ge=1, le=2000),
    ):
        """
        List a folder for the in-app file browser, a page at a time.

        Subfolders come first, then media matching `filter`. Frame sequences
        collapse to one entry. If `path` names a file or sequence, its folder
        is listed and `selected` names the matching entry.
        """
        return await run_in_threadpool(browse_listing, path, filter, sequences, offset, limit)

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
    
