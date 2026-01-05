# file_browser.py
"""
Native File Browser for FUK Pipeline

Uses tkinter for native OS file dialogs with:
- Image sequence detection (render.0001.exr → render.####.exr)
- Video file support
- Network path handling
- VFX-friendly file patterns

This runs as a separate process from the web server to avoid 
tkinter/asyncio conflicts.
"""

import os
import re
import sys
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    SEQUENCE = "sequence"


@dataclass
class MediaFile:
    """Represents a media file or sequence"""
    path: str                    # For sequences: path with #### pattern
    display_name: str            # User-friendly name
    media_type: str              # image, video, sequence
    first_frame: Optional[int]   # For sequences
    last_frame: Optional[int]    # For sequences
    frame_count: Optional[int]   # For sequences
    frame_pattern: Optional[str] # Original pattern (e.g., %04d)
    
    def to_dict(self):
        return asdict(self)


# File extension categories
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.exr', '.dpx'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mxf', '.prores'}
SEQUENCE_EXTENSIONS = {'.exr', '.dpx', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}

# Regex for detecting frame numbers in filenames
FRAME_PATTERNS = [
    # render.0001.exr, render.001.exr
    re.compile(r'^(.+?)\.(\d{3,})\.(\w+)$'),
    # render_0001.exr, render_001.exr  
    re.compile(r'^(.+?)_(\d{3,})\.(\w+)$'),
    # render0001.exr (no separator)
    re.compile(r'^(.+?)(\d{4,})\.(\w+)$'),
]


def detect_sequence(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Detect if a file is part of an image sequence
    
    Returns dict with sequence info or None if not a sequence
    """
    if file_path.suffix.lower() not in SEQUENCE_EXTENSIONS:
        return None
    
    filename = file_path.name
    parent = file_path.parent
    
    for pattern in FRAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)
            frame_str = match.group(2)
            ext = match.group(3)
            padding = len(frame_str)
            
            # Build pattern to find all frames
            frame_pattern = f"{prefix}.{'#' * padding}.{ext}"
            glob_pattern = f"{prefix}.*[0-9].{ext}"
            
            # Find all matching frames
            frames = []
            for f in parent.glob(f"{prefix}*"):
                m = pattern.match(f.name)
                if m and m.group(3) == ext:
                    try:
                        frames.append(int(m.group(2)))
                    except ValueError:
                        pass
            
            if len(frames) > 1:
                frames.sort()
                return {
                    "is_sequence": True,
                    "prefix": prefix,
                    "extension": ext,
                    "padding": padding,
                    "first_frame": min(frames),
                    "last_frame": max(frames),
                    "frame_count": len(frames),
                    "pattern": frame_pattern,
                    "path_pattern": str(parent / frame_pattern),
                    # Python-style pattern for actual file access
                    "printf_pattern": str(parent / f"{prefix}.%0{padding}d.{ext}"),
                }
    
    return None


def collapse_sequences(files: List[Path]) -> List[MediaFile]:
    """
    Collapse a list of files, grouping image sequences
    
    Returns list of MediaFile objects with sequences collapsed
    """
    results = []
    seen_sequences = set()
    
    for file_path in sorted(files):
        suffix = file_path.suffix.lower()
        
        # Check for sequence
        seq_info = detect_sequence(file_path)
        if seq_info:
            # Use pattern as unique key
            seq_key = seq_info["path_pattern"]
            if seq_key not in seen_sequences:
                seen_sequences.add(seq_key)
                results.append(MediaFile(
                    path=seq_info["printf_pattern"],
                    display_name=f"{seq_info['pattern']} [{seq_info['first_frame']}-{seq_info['last_frame']}]",
                    media_type=MediaType.SEQUENCE.value,
                    first_frame=seq_info["first_frame"],
                    last_frame=seq_info["last_frame"],
                    frame_count=seq_info["frame_count"],
                    frame_pattern=seq_info["pattern"],
                ))
            continue
        
        # Single image
        if suffix in IMAGE_EXTENSIONS:
            results.append(MediaFile(
                path=str(file_path),
                display_name=file_path.name,
                media_type=MediaType.IMAGE.value,
                first_frame=None,
                last_frame=None,
                frame_count=None,
                frame_pattern=None,
            ))
            continue
        
        # Video
        if suffix in VIDEO_EXTENSIONS:
            results.append(MediaFile(
                path=str(file_path),
                display_name=file_path.name,
                media_type=MediaType.VIDEO.value,
                first_frame=None,
                last_frame=None,
                frame_count=None,
                frame_pattern=None,
            ))
    
    return results


def open_file_dialog(
    title: str = "Select Media",
    initial_dir: Optional[str] = None,
    file_types: Optional[List[Tuple[str, str]]] = None,
    multiple: bool = True,
    detect_sequences: bool = True,
) -> List[MediaFile]:
    """
    Open a native file dialog and return selected files
    
    Args:
        title: Dialog title
        initial_dir: Starting directory
        file_types: List of (name, pattern) tuples for file filter
        multiple: Allow multiple file selection
        detect_sequences: Collapse image sequences
        
    Returns:
        List of MediaFile objects
    """
    import tkinter as tk
    from tkinter import filedialog
    
    # Create hidden root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Bring to front
    
    # Default file types
    if file_types is None:
        file_types = [
            ("All Media", "*.png *.jpg *.jpeg *.webp *.exr *.dpx *.mp4 *.mov *.avi *.mkv"),
            ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif"),
            ("EXR/DPX", "*.exr *.dpx"),
            ("Videos", "*.mp4 *.mov *.avi *.mkv *.webm"),
            ("All Files", "*.*"),
        ]
    
    # Initial directory
    if initial_dir and Path(initial_dir).exists():
        init_dir = initial_dir
    else:
        init_dir = str(Path.home())
    
    # Open dialog
    if multiple:
        selected = filedialog.askopenfilenames(
            title=title,
            initialdir=init_dir,
            filetypes=file_types,
        )
    else:
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=init_dir,
            filetypes=file_types,
        )
        selected = [selected] if selected else []
    
    root.destroy()
    
    if not selected:
        return []
    
    # Convert to Path objects
    files = [Path(f) for f in selected if f]
    
    # Collapse sequences if requested
    if detect_sequences:
        return collapse_sequences(files)
    else:
        # Return as individual files
        results = []
        for f in files:
            suffix = f.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                media_type = MediaType.VIDEO.value
            else:
                media_type = MediaType.IMAGE.value
            
            results.append(MediaFile(
                path=str(f),
                display_name=f.name,
                media_type=media_type,
                first_frame=None,
                last_frame=None,
                frame_count=None,
                frame_pattern=None,
            ))
        return results


def open_directory_dialog(
    title: str = "Select Directory",
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    """Open a directory selection dialog"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if initial_dir and Path(initial_dir).exists():
        init_dir = initial_dir
    else:
        init_dir = str(Path.home())
    
    selected = filedialog.askdirectory(
        title=title,
        initialdir=init_dir,
    )
    
    root.destroy()
    
    return selected if selected else None


def scan_directory(
    directory: str,
    recursive: bool = False,
    detect_sequences: bool = True,
) -> List[MediaFile]:
    """
    Scan a directory for media files
    
    Args:
        directory: Directory path to scan
        recursive: Scan subdirectories
        detect_sequences: Collapse image sequences
        
    Returns:
        List of MediaFile objects
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    files = []
    
    if recursive:
        for ext in all_extensions:
            files.extend(dir_path.rglob(f"*{ext}"))
    else:
        for ext in all_extensions:
            files.extend(dir_path.glob(f"*{ext}"))
    
    if detect_sequences:
        return collapse_sequences(files)
    else:
        results = []
        for f in files:
            suffix = f.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                media_type = MediaType.VIDEO.value
            else:
                media_type = MediaType.IMAGE.value
            
            results.append(MediaFile(
                path=str(f),
                display_name=f.name,
                media_type=media_type,
                first_frame=None,
                last_frame=None,
                frame_count=None,
                frame_pattern=None,
            ))
        return results


# ============================================================================
# CLI Interface (for subprocess calls from web server)
# ============================================================================

def main():
    """CLI entry point for subprocess invocation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FUK File Browser")
    parser.add_argument("command", choices=["open", "directory", "scan"])
    parser.add_argument("--title", default="Select Media")
    parser.add_argument("--initial-dir", default=None)
    parser.add_argument("--multiple", action="store_true", default=True)
    parser.add_argument("--no-multiple", action="store_false", dest="multiple")
    parser.add_argument("--detect-sequences", action="store_true", default=True)
    parser.add_argument("--no-detect-sequences", action="store_false", dest="detect_sequences")
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--filter", choices=["all", "images", "videos", "exr"], default="all")
    
    args = parser.parse_args()
    
    # Build file type filter
    if args.filter == "images":
        file_types = [
            ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif *.exr *.dpx"),
        ]
    elif args.filter == "videos":
        file_types = [
            ("Videos", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
        ]
    elif args.filter == "exr":
        file_types = [
            ("EXR/DPX", "*.exr *.dpx"),
        ]
    else:
        file_types = None
    
    result = {"success": False, "files": [], "directory": None}
    
    try:
        if args.command == "open":
            files = open_file_dialog(
                title=args.title,
                initial_dir=args.initial_dir,
                file_types=file_types,
                multiple=args.multiple,
                detect_sequences=args.detect_sequences,
            )
            result = {
                "success": True,
                "files": [f.to_dict() for f in files],
            }
            
        elif args.command == "directory":
            directory = open_directory_dialog(
                title=args.title,
                initial_dir=args.initial_dir,
            )
            result = {
                "success": True,
                "directory": directory,
            }
            
        elif args.command == "scan":
            if not args.initial_dir:
                result = {"success": False, "error": "Directory required for scan"}
            else:
                files = scan_directory(
                    directory=args.initial_dir,
                    recursive=args.recursive,
                    detect_sequences=args.detect_sequences,
                )
                result = {
                    "success": True,
                    "files": [f.to_dict() for f in files],
                }
                
    except Exception as e:
        result = {"success": False, "error": str(e)}
    
    # Output as JSON
    print(json.dumps(result))


if __name__ == "__main__":
    main()