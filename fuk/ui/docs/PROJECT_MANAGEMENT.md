# FUK Project Management - Integration Guide

## Overview

This adds project file management to FUK, supporting the standard VFX project structure:

```
ProjectName/
├── Assets/
├── Projects/
│   ├── Blender/
│   ├── Resolve/
│   └── Fuk/                          ← FUK project folder
│       ├── fuk_config.json           ← Optional config overrides
│       ├── cache/                    ← Auto-generated cache folder
│       │   └── img_gen_20251229_143052/
│       ├── projectname_shot01_251229.json
│       └── projectname_shot02_251229.json
├── Renders/
│   └── Shot01/
│       └── 251229/
└── Deliverables/
```

## Backend Integration

### Step 1: Copy the endpoints file

Copy `src/backend/project_endpoints.py` to your server directory (same folder as `fuk_web_server.py`).

### Step 2: Add to your server

At the top of `fuk_web_server.py`, add:

```python
from project_endpoints import setup_project_routes, get_cache_output_path
```

After creating the FastAPI app (after `app = FastAPI(...)`), add:

```python
# Add project management routes
setup_project_routes(app)
```

### Step 3: Route generations to project cache

In your `run_image_generation` function, change the output path logic:

```python
# OLD: Uses fixed output folder
# gen_dir = image_manager.create_generation_dir()

# NEW: Uses project cache if set, otherwise default
from project_endpoints import get_cache_output_path
gen_dir = get_cache_output_path("img_gen")
```

Do the same for video generation.

### Step 4: Install tkinter (for folder browser)

On Ubuntu/Debian:
```bash
sudo apt-get install python3-tk
```

On macOS (usually pre-installed):
```bash
# If needed: brew install python-tk
```

## Frontend Files

```
src/
├── components/
│   └── ProjectBar.jsx      # Header navigation component
├── hooks/
│   └── useProject.js       # Project state management hook
├── utils/
│   ├── projectManager.js   # Filename parsing, versioning logic
│   └── projectApi.js       # API calls for project management
└── styles/
    └── project-bar.css     # ProjectBar styling
```

## How It Works

### Workflow

1. Click folder icon → Native folder browser opens
2. Navigate to `ProjectName/Projects/Fuk/`
3. If empty: "New Project File" button appears → Enter project name
4. If files exist: Auto-loads most recent version
5. Use dropdowns to switch Shot/Version
6. "Version Up" creates new version with today's date
7. All generations go to `cache/` subfolder

### File Naming Convention

```
{projectname}_shot{##}_{version}.json

Examples:
  commercial_shot01_251229.json    (date format)
  commercial_shot01_251229a.json   (same-day suffix)
  commercial_shot02_v01.json       (sequential format)
```

### Project Config

Create `fuk_config.json` in your Fuk folder to customize:

```json
{
  "versionFormat": "date",
  "cacheFolder": "cache"
}
```

- `versionFormat`: `"date"` (251229) or `"sequential"` (v01)
- `cacheFolder`: Name of cache subfolder (default: "cache")

## JSON Project File Structure

```json
{
  "meta": {
    "version": "1.0",
    "createdAt": "2025-12-29T10:00:00Z",
    "updatedAt": "2025-12-29T14:30:00Z",
    "fukVersion": "1.0.0"
  },
  "project": {
    "name": "projectname",
    "shot": "01",
    "version": "251229"
  },
  "tabs": {
    "image": {
      "prompt": "...",
      "model": "qwen_image",
      "aspectRatio": "16:9",
      "width": 1344,
      ...
    },
    "video": { ... },
    "preprocess": { },
    "postprocess": { },
    "export": { }
  },
  "assets": {
    "controlImages": [],
    "referenceImages": [],
    "inputVideos": []
  },
  "lastState": {
    "activeTab": "image",
    "lastImagePreview": "cache/img_gen_251229_143052/generated.png"
  },
  "notes": "Artist notes"
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/project/browse-folder` | POST | Open native folder dialog |
| `/api/project/set-folder` | POST | Set folder by path |
| `/api/project/current` | GET | Get current project state |
| `/api/project/list` | GET | List all project files |
| `/api/project/load/{filename}` | GET | Load a project file |
| `/api/project/save/{filename}` | POST | Save project state |
| `/api/project/new` | POST | Create new project file |
| `/api/project/config` | GET | Get project config |
| `/api/project/cache-info` | GET | Get cache folder stats |

## Cache vs Renders

- **Cache** (`Fuk/cache/`): All trial/error generations, organized by timestamp
- **Renders** (`Renders/Shot##/`): Final exports only (via Export tab)

The cache is never auto-cleaned. Add a "Clear Cache" button later if needed.
