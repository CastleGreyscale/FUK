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

**System Requirements:**
- Python 3.10+
- CUDA-compatible GPU (12GB+ VRAM recommended)
- ~50GB storage for models and dependencies

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/fuk.git
cd fuk

# Install Python dependencies
pip install -e .

# Install frontend dependencies (bun recommended, npm works)
cd frontend
bun install  # or: npm install
cd ..
```

### Core Models

FUK uses musubi-tuner as a backend wrapper for generation models. You'll need:

**Image Generation (Qwen):**
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

**Video Generation (Wan):**
- [WanX-FluxV1](https://huggingface.co/wanx-video/WanX-FluxV1)

Download these via musubi-tuner's model manager or manually place in your models directory. See [musubi-tuner documentation](https://github.com/tdrussell/musubi-tuner) for detailed setup.

### Vendor Dependencies

These tools provide specialized functionality and need separate installation:

#### SAM2 (Segmentation/Cryptomattes)
```bash
cd vendor/sam2
pip install -e .
# Download checkpoint:
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

#### Depth Anything V3 (Depth Maps)
```bash
cd vendor/Depth-Anything-V2
pip install -e .
# Models will auto-download on first use
```

#### Real-ESRGAN-ncnn (Upscaling)
```bash
cd vendor/Real-ESRGAN-ncnn-vulkan
# Download pre-built binaries from:
# https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases
# Place realesrgan-ncnn-vulkan executable in this directory
```

#### RIFE-ncnn (Frame Interpolation)
```bash
cd vendor/rife-ncnn-vulkan
# Download pre-built binaries from:
# https://github.com/nihui/rife-ncnn-vulkan/releases
# Place rife-ncnn-vulkan executable in this directory
```

#### musubi-tuner (Generation Backend)
```bash
cd vendor/musubi-tuner
pip install -e .
# Follow musubi setup for model downloads and configuration
```

### Configuration

FUK uses JSON configs for defaults and model-specific parameters:

- `defaults.json` - Global settings and UI defaults
- `models.json` - Model configurations and paths
- Model-specific configs in root directory (e.g., `depth-anything-v3.json`)

Edit these to match your system and preferences before first run.

---

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
