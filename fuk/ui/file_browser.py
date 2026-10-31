# file_browser.py
"""
Native File Browser for FUK Pipeline

Uses tkinter for native OS file dialogs with:
- Image sequence detection (render.0001.exr â†’ render.####.exr)
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
# Each pattern includes a group for the separator to preserve it
FRAME_PATTERNS = [
    # render.0001.exr, render.001.exr (dot separator)
    (re.compile(r'^(.+?)(\.?)(\d{3,})\.(\w+)$'), '.'),
    # render_0001.exr, render_001.exr (underscore separator)
    (re.compile(r'^(.+?)(_)(\d{3,})\.(\w+)$'), '_'),
    # render0001.exr (no separator - 4+ digits required)
    (re.compile(r'^(.+?)()(\d{4,})\.(\w+)$'), ''),
]


def _match_frame(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """Split a frame filename into (prefix, separator, frame digits, extension)."""
    if Path(filename).suffix.lower() not in SEQUENCE_EXTENSIONS:
        return None
    for pattern, default_sep in FRAME_PATTERNS:
        match = pattern.match(filename)
        if match:
            prefix, separator, frame_str, ext = match.groups()
            # Use the actual separator from the filename
            # For no-separator pattern, default_sep is already ''
            return prefix, separator or default_sep, frame_str, ext
    return None


def find_sequences(parent: Path, names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Group the frame-numbered files among `names` (all in `parent`) into
    sequences, in one pass.

    Supports common VFX naming patterns:
    - render.0001.exr (dot separator)
    - render_0001.png (underscore separator)
    - render0001.dpx (no separator)

    Returns {filename: sequence info} for every file in a sequence of two or
    more frames. Files that aren't part of one are left out.
    """
    buckets: Dict[Tuple[str, str, str], List[Tuple[int, str, str]]] = {}
    for name in names:
        parts = _match_frame(name)
        if parts:
            prefix, sep, frame_str, ext = parts
            buckets.setdefault((prefix, sep, ext), []).append((int(frame_str), frame_str, name))

    result = {}
    for (prefix, sep, ext), frames in buckets.items():
        if len(frames) < 2:
            continue
        frames.sort()
        padding = len(frames[0][1])
        # Build pattern preserving original separator
        frame_pattern = f"{prefix}{sep}{'#' * padding}.{ext}"
        info = {
            "is_sequence": True,
            "prefix": prefix,
            "separator": sep,
            "extension": ext,
            "padding": padding,
            "first_frame": frames[0][0],
            "last_frame": frames[-1][0],
            "frame_count": len(frames),
            "pattern": frame_pattern,
            "path_pattern": str(parent / frame_pattern),
            # Python-style pattern for actual file access
            "printf_pattern": str(parent / f"{prefix}{sep}%0{padding}d.{ext}"),
            "first_file": str(parent / frames[0][2]),
        }
        for _, _, name in frames:
            result[name] = info
    return result


def detect_sequence(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Detect if a file is part of an image sequence on disk.

    Returns dict with sequence info or None if not a sequence
    """
    if not _match_frame(file_path.name):
        return None
    return find_sequences(file_path.parent, os.listdir(file_path.parent)).get(file_path.name)


def collapse_sequences(files: List[Path]) -> List[MediaFile]:
    """
    Collapse a list of files, grouping image sequences

    A file counts as a frame if its folder on disk holds the rest of the
    sequence, so picking one frame yields the whole sequence. Each folder is
    listed once: this used to glob the folder per frame, which took 10 s for
    2,000 frames and grew with the square of the count.

    Returns list of MediaFile objects with sequences collapsed
    """
    results = []
    seen_sequences = set()
    folder_sequences: Dict[Path, Dict[str, Dict[str, Any]]] = {}

    for file_path in sorted(files):
        suffix = file_path.suffix.lower()

        # Check for sequence
        parent = file_path.parent
        if parent not in folder_sequences:
            try:
                folder_sequences[parent] = find_sequences(parent, os.listdir(parent))
            except OSError:
                folder_sequences[parent] = {}
        seq_info = folder_sequences[parent].get(file_path.name)
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


def save_file_dialog(
    title: str = "Save As",
    initial_dir: Optional[str] = None,
    initial_file: Optional[str] = None,
    file_types: Optional[List] = None,
) -> Optional[str]:
    """Open a native 'Save As' dialog and return the chosen destination path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if initial_dir and Path(initial_dir).exists():
        init_dir = initial_dir
    else:
        init_dir = str(Path.home())

    kwargs = {"title": title, "initialdir": init_dir}
    if initial_file:
        kwargs["initialfile"] = initial_file
        ext = Path(initial_file).suffix
        if ext:
            kwargs["defaultextension"] = ext
            kwargs["filetypes"] = file_types or [(f"{ext.lstrip('.').upper()} file", f"*{ext}"), ("All files", "*.*")]
    elif file_types:
        kwargs["filetypes"] = file_types

    selected = filedialog.asksaveasfilename(**kwargs)

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
# In-app browser listing (used by /api/browser/list)
# ============================================================================

# Extension sets for the browser's filter. None = show every file.
BROWSER_FILTERS = {
    "all": IMAGE_EXTENSIONS | VIDEO_EXTENSIONS,
    "images": IMAGE_EXTENSIONS,
    "videos": VIDEO_EXTENSIONS,
    "exr": {".exr", ".dpx"},
    "layered": {".psd", ".ora"},
    "any": None,
}

# Name patterns that stand for a whole sequence: ####, %04d, $F4.
_SEQUENCE_TOKEN = re.compile(r"#+|%0?(\d*)d|\$F(\d*)")
# Nuke and Resolve append the frame range: "shot.%04d.exr 1001-1100".
_TRAILING_RANGE = re.compile(r"\s+-?\d+-\d+$")


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _hash_pattern(name: str) -> str:
    """Normalise %04d / $F4 frame tokens in a filename to ####."""
    def repl(m):
        if m.group(0).startswith("#"):
            return m.group(0)
        width = m.group(1) or m.group(2) or "4"
        return "#" * int(width)
    return _SEQUENCE_TOKEN.sub(repl, name)


def resolve_browse_path(raw: str) -> Tuple[Path, Optional[Path], bool]:
    """
    Turn a typed or pasted path into (folder to list, entry to select, missing).

    Accepts quotes, file:// URLs, ~, a file (lists its folder and selects it),
    a sequence pattern (####, %04d, $F4, with or without a trailing frame
    range), or a folder. A path that doesn't exist resolves to its nearest
    existing ancestor with missing=True.
    """
    from urllib.parse import unquote, urlparse

    s = raw.strip().strip("'\"")
    if s.startswith("file://"):
        s = unquote(urlparse(s).path)
    s = _TRAILING_RANGE.sub("", s)
    p = Path(os.path.expanduser(s))
    if not p.is_absolute():
        p = Path.home() / p

    if _SEQUENCE_TOKEN.search(p.name) and p.parent.is_dir():
        return p.parent, p, False
    if p.is_dir():
        return p, None, False
    if p.is_file():
        return p.parent, p, False

    ancestor = p.parent
    while not ancestor.is_dir() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    return ancestor, None, True


def list_directory(
    directory: Path,
    filter: str = "all",
    detect_sequences: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    List a folder for the in-app browser: subfolders first, then media.

    Hidden entries are skipped. Frame sequences collapse to one entry (path =
    printf pattern, display name "shot_010.####.exr [1001-1100]"). Files and
    sequences carry `thumb`, an image path the preview endpoint can downscale.

    Returns (entries, aliases): aliases maps every member frame and pattern
    spelling of a sequence to its entry path, so a pasted frame or pattern
    selects the right entry.
    """
    exts = BROWSER_FILTERS.get(filter, BROWSER_FILTERS["all"])
    dirs, files = [], []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.is_file():
                    if exts is None or Path(entry.name).suffix.lower() in exts:
                        files.append(entry.name)
            except OSError:
                continue  # broken symlink, permission

    entries = [
        {"kind": "dir", "name": name, "path": str(directory / name)}
        for name in sorted(dirs, key=_natural_key)
    ]
    sequences = find_sequences(directory, files) if detect_sequences else {}
    aliases: Dict[str, str] = {}
    seen = set()

    for name in sorted(files, key=_natural_key):
        seq = sequences.get(name)
        if seq:
            aliases[str(directory / name)] = seq["printf_pattern"]
            if seq["printf_pattern"] in seen:
                continue
            seen.add(seq["printf_pattern"])
            aliases[seq["path_pattern"]] = seq["printf_pattern"]
            entries.append({
                "kind": "sequence",
                "name": f"{seq['pattern']} [{seq['first_frame']}-{seq['last_frame']}]",
                "path": seq["printf_pattern"],
                "thumb": seq["first_file"],
                "first_frame": seq["first_frame"],
                "last_frame": seq["last_frame"],
                "frame_count": seq["frame_count"],
                "frame_pattern": seq["pattern"],
            })
            continue

        suffix = Path(name).suffix.lower()
        kind = "image" if suffix in IMAGE_EXTENSIONS else "video" if suffix in VIDEO_EXTENSIONS else "file"
        entries.append({
            "kind": kind,
            "name": name,
            "path": str(directory / name),
            "thumb": str(directory / name) if kind == "image" else None,
        })

    return entries, aliases


def selection_for(target: Optional[Path], aliases: Dict[str, str]) -> Optional[str]:
    """Entry path to preselect for a resolved paste target."""
    if target is None:
        return None
    key = str(target)
    if key in aliases:
        return aliases[key]
    return aliases.get(str(target.parent / _hash_pattern(target.name)), key)


# ============================================================================
# CLI Interface (for subprocess calls from web server)
# ============================================================================

def main():
    """CLI entry point for subprocess invocation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FUK File Browser")
    parser.add_argument("command", choices=["open", "directory", "scan", "save"])
    parser.add_argument("--title", default="Select Media")
    parser.add_argument("--initial-dir", default=None)
    parser.add_argument("--initial-file", default=None)
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

        elif args.command == "save":
            dest = save_file_dialog(
                title=args.title,
                initial_dir=args.initial_dir,
                initial_file=args.initial_file,
                file_types=file_types,
            )
            result = {
                "success": True,
                "path": dest,
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