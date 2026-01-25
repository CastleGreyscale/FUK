# FUK: Framework for Unified Kreation

**Local-first rendering pipeline for professional AI generation workflows.**

Built for post-production professionals who need AI tools that integrate with existing pipelines rather than replacing them.

---

## Core Philosophy

- **Stay in the zone** - Minimize context switching and file wrangling
- **Your files, your structure** - No forced file locations or arbitrary project hierarchies
- **Stability over novelty** - Production tools shouldn't break with every update
- **Professional outputs** - Multi-layer EXR, proper bit depth, real compositing integration
- **Modular design** - Use the components you need, ignore the rest

FUK treats AI generation as source material that enters a professional pipeline, not as final output.

---

## Features

### Current Implementation

**Generation:**
- Image generation via Qwen (musubi-tuner wrapper)
- Video generation via Wan (musubi-tuner wrapper)
- Preprocessors: OpenPose, Depth Anything V2, Canny edge detection
- Seed tracking and management with save/recall functionality

**Post-Processing:**
- Upscaling via Real-ESRGAN
- Frame interpolation via RIFE
- Style transfer and image-to-image workflows

**Technical Outputs:**
- Depth map generation (Depth Anything V2/V3)
- Normal map generation
- Cryptomatte generation (SAM2)
- Multi-layer 32-bit EXR export with proper AOV organization

**Project Management:**
- Industry-standard project/shot/version hierarchy
- Non-destructive file organization (no forced copying)
- Comprehensive metadata storage
- Version-aware caching system
- Per-shot configuration and state persistence

### In Development

- **Direct model integration** - Moving away from musubi wrappers for extended functionality
- **Temporal consistency** - Improved frame-to-frame coherence for depth and cryptomattes
- **Full Wan implementation** - Complete video generation feature set
- **True lossless passthrough** - EXR and raw latent workflows throughout pipeline
- **Advanced ControlNet** - Direct implementation with extended control options

### Roadmap

- Latent space manipulation for camera control and scene modification
- Multi-camera scene coverage from single generations
- Enhanced project collection and packaging tools
- Prompt expansion and learning via Ollama integration
- Per-project configuration overrides
- Native sequence handling and metadata extraction

---
## Installation

### System Requirements

- **OS:** Linux (recommended), macOS, or Windows with WSL2
- **Python:** 3.10 or higher
- **GPU:** CUDA-compatible NVIDIA GPU with 12GB+ VRAM
- **Storage:** ~50GB for models and dependencies
- **Software:** Git, Node.js (for frontend)

### Quick Install

### Linux/Mac

```bash
git clone https://github.com/yourusername/FUK.git
cd FUK
chmod +x setup.sh
./setup.sh
```

The setup script handles everything:
- Installs core Python dependencies
- Clones vendor repositories (SAM2, Depth-Anything-3, musubi-tuner)
- Installs vendor packages
- Downloads SAM2 checkpoints
- Installs frontend dependencies (bun or npm)

### Windows

```bash
git clone https://github.com/yourusername/FUK.git
cd FUK

# Install core package
pip install -e .

# Clone vendor dependencies
mkdir fuk\vendor
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git fuk/vendor/Depth-Anything-3
git clone https://github.com/facebookresearch/segment-anything-2 fuk/vendor/segment-anything-2
git clone https://github.com/kohya-ss/musubi-tuner.git fuk/vendor/musubi-tuner
git clone https://github.com/baegwangbin/DSINE.git fuk/vendor/DSINE

# Install vendor packages
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2
pip install -e ./fuk/vendor/musubi-tuner

# Install Depth Anything V3 deps
pip install -e ".[depth]"

# Frontend
cd fuk/ui
npm install  # or: bun install
cd ../..
```

### Optional: RIFE and Real-ESRGAN (Recommended)

For frame interpolation and upscaling, download pre-built binaries:

**RIFE (Frame Interpolation):**
1. Download from: https://github.com/nihui/rife-ncnn-vulkan/releases
2. Extract contents to: `fuk/vendor/rife-ncnn/`
3. Ensure `rife-ncnn-vulkan` executable is in that directory

**Real-ESRGAN (Upscaling):**

Linux:
```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d fuk/vendor/realesrgan-ncnn/
```

Windows:
1. Download: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
2. Extract contents to: `fuk\vendor\realesrgan-ncnn\`
3. Ensure `realesrgan-ncnn-vulkan.exe` is in that directory

### Model Downloads

FUK requires these AI models:

### Required Models

**Image Generation (Qwen):**
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

**Video Generation (Wan):**
- [WanX-FluxV1](https://huggingface.co/wanx-video/WanX-FluxV1)

Download via musubi-tuner's model manager or manually:

```bash
cd vendor/musubi-tuner
python download_models.py
```

See [musubi-tuner documentation](https://github.com/kohya-ss/musubi-tuner) for detailed model setup.

### Auto-Downloaded Models

These download automatically on first use:
- SAM2 segmentation checkpoints (via setup script)
- Depth Anything V3 (on first depth generation)
- Real-ESRGAN models (on first upscale)

## Configuration

Edit these files before first run:

- `models.json` - Model paths and configurations
- `defaults.json` - UI defaults and system settings
- Model-specific configs: `depth-anything-v3.json`, `sam2.json`, etc.

## First Run

```bash
python start_web_ui.py
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## Troubleshooting

### Depth Anything V3 Not Found

If you see `No module named 'depth_anything_3'`:

```bash
# Verify installation
cd fuk/vendor/Depth-Anything-3
pip install -e .

# If that fails, install dependencies first
pip install -e ".[depth]"  # from FUK root directory
cd fuk/vendor/Depth-Anything-3
pip install -e .

# Verify it's importable
python -c "import depth_anything_3; print('✓ DA3 installed')"
```

Common causes:
- Installed with `--no-deps` flag (don't do this)
- Missing dependencies from `[depth]` optional group
- Virtual environment not activated

### CUDA/GPU Issues
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Should return `True`. If not, reinstall PyTorch with CUDA support.

### VRAM Issues
Edit `defaults.json` and reduce batch sizes or enable CPU offloading.

### Model Download Fails
Some models require Hugging Face authentication:
```bash
huggingface-cli login
```

### Frontend Won't Start
```bash
cd fuk/ui
rm -rf node_modules
npm install  # or: bun install
```

## Development Install

For development with hot reload:

```bash
# Backend (auto-reloads on file changes)
uvicorn fuk_web_server:app --reload --host 0.0.0.0 --port 8000

# Frontend (Vite dev server)
cd fuk/ui
npm run dev
```

## Uninstall

```bash
pip uninstall fuk
rm -rf vendor/
```
## Usage

```bash
# Start the server
python start_web_ui.py

# Frontend runs on http://localhost:5173
# Backend API on http://localhost:8000
```

The UI follows a left-to-right workflow:
1. **Image Tab** - Generation and preprocessing
2. **Video Tab** - Video generation and temporal processing
3. **Layers Tab** - Technical output generation (depth, normals, crypto)
4. **Post-Process Tab** - Upscaling and interpolation
5. **Export Tab** - Multi-layer EXR assembly and delivery

Projects are organized by date-based versions (`YYMMDD_NN`) with automatic versioning on export.

---

## Why FUK?

Most AI generation tools are built for hobbyists creating single images. FUK is built for professionals who need:

- AI-generated content that integrates into existing compositing workflows
- Reproducible results with proper version control
- Technical outputs (depth, normals, mattes) for downstream work
- Stable tools that don't break between projects
- Local control without cloud dependencies

FUK doesn't replace your workflow - it extends it.

---

## Technical Notes

**VRAM Management:**
FUK implements intelligent model loading/unloading to maximize available VRAM. Models are loaded on-demand and offloaded when not in use. For lower VRAM systems, consider reducing batch sizes in generation configs.

**File Organization:**
Unlike tools that force centralized file structures, FUK respects your project organization. Set your project root, and FUK will create necessary subdirectories while leaving your existing files untouched.

**Temporal Consistency:**
Video and sequence processing includes batch-based temporal coherence for preprocessors and technical outputs. Currently in active development for depth and cryptomatte generation.

---

## Known Issues

- Video upscaling UI freezing when loading results (functional, UI issue only)
- Progress bar implementation inconsistent across operations
- Image dropzone sensitivity (may require precise targeting)

---

## Contributing

This project is currently in active development. Contribution guidelines coming soon.

For bug reports and feature requests, please use GitHub issues.

---

## License

MIT License - see LICENSE file for details

---

## Name

Yes, it's called FUK. **F**ramework for **U**nified **K**reation.

Because professional tools shouldn't require professional patience.

**Get FUKed. Render locally.**
