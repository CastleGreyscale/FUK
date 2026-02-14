# FUK: Framework for Unified Kreation

**Local-first AI rendering pipeline for professional VFX workflows.**

Built for post-production professionals who need AI tools that integrate with existing pipelines rather than replacing them.

---

## Architecture

FUK is built as a modular, extensible pipeline that orchestrates specialized AI models through a unified interface:

```
┌─────────────────────────────────────────────────────────────┐
│                    React UI (Vite + FastAPI)                 │
│          Project Management • Generation History             │
│              Drag-and-drop • Real-time Preview               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   FUK Core Pipeline Layer                    │
│        Python Managers • Config System • VRAM Control        │
│       Latent Management • EXR Export • File Organization     │
└─┬──────┬──────┬──────┬──────┬──────┬──────┬────────────────┘
  │      │      │      │      │      │      │
  ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ DS │ │DA3 │ │SAM2│ │DSIN│ │RIFE│ │RE  │ │... │  Vendor Layer
│    │ │    │ │    │ │ E  │ │    │ │ GAN│ │    │  (External Models)
└────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘

DS = DiffSynth-Studio (Qwen, Wan generation)
DA3 = Depth-Anything-V3 (monocular depth)
SAM2 = Segment Anything 2 (cryptomattes)
DSINE = Surface normals estimation
RIFE = Frame interpolation
REGAN = Real-ESRGAN upscaling
```

**Design Principles:**
- **Modular vendor integration** - Each AI model is a self-contained module
- **Unified configuration** - Single config system for all tools
- **Extensible by design** - Add new models by extending base classes
- **Stay close to source** - Direct model APIs, minimal wrapper dependencies
- **Professional outputs** - Industry-standard formats (EXR, proper color space)

Adding new capabilities means extending managers and configs, not rewriting core logic.

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

**Generation (via DiffSynth-Studio):**
- Image generation via Qwen models (2509, 2512, Edit, Control-Union)
- Video generation via Wan models (I2V-A14B, VACE-Fun)
- Direct model integration with flexible configuration
- Seed tracking and management with save/recall functionality
- Latent-space operations and caching

**Preprocessing:**
- Depth: Depth-Anything-V3 (mono/stereo, indoor/outdoor variants)
- Cryptomatte: SAM2 segmentation with automatic mask ID generation
- Normals: DSINE surface normal estimation
- Pose: OpenPose (body, hands, face)
- Edge: Canny edge detection
- Batch video processing with temporal consistency

**Post-Processing:**
- Upscaling: Real-ESRGAN (X4 models)
- Frame interpolation: RIFE (2x, 4x, 8x)
- Video stabilization and temporal coherence
- Sequence-aware processing

**Technical Outputs:**
- Multi-layer 32-bit float EXR export
- Proper AOV organization (beauty, depth, normals, crypto)
- Scene-linear color space workflows
- Lossless latent-to-EXR conversion
- Native Nuke/Resolve/Blender integration

**Project Management:**
- Industry-standard project/shot/version hierarchy
- Non-destructive file organization
- Comprehensive generation history and metadata
- Per-shot configuration persistence
- Intelligent VRAM management and model offloading

### In Development

- **Extended latent operations** - Camera control, lighting adjustment via latent manipulation
- **Scene builder workflows** - Multi-camera coverage from single generations
- **Advanced temporal processing** - Enhanced frame-to-frame coherence
- **LoRA training pipeline** - Automated character consistency workflows
- **Extended model support** - SUPIR upscaling, FILM interpolation

### Roadmap

- Multi-shot project packaging and delivery
- Prompt expansion via Ollama integration
- Per-project configuration inheritance and overrides
- Native sequence handling with automatic metadata extraction
- Render farm integration for batch processing

---

## Installation

### System Requirements

- **OS:** Linux (primary), macOS, Windows with WSL2
- **Python:** 3.10 or 3.11 (3.12+ not yet tested)
- **GPU:** NVIDIA GPU with 12GB+ VRAM (24GB recommended for video)
- **CUDA:** 11.8 or 12.1+
- **Storage:** ~50GB for base models, more for LoRAs and custom checkpoints
- **Software:** Git, Node.js 18+ (for frontend)

### Quick Install

#### Linux/macOS

```bash
# Clone repository
git clone https://github.com/CastleGreyscale/FUK.git
cd FUK

# Run setup (installs dependencies, vendor packages, configs)
chmod +x setup.sh
./setup.sh

# Edit configuration files
nano config/models.json     # Set your model root directory
nano config/defaults.json   # Adjust generation defaults

# Download models (separate step, allows config review first)
python scripts/download_models.py

# Start the server
python start_web_ui.py
```

The setup script handles:
- Python dependencies from pyproject.toml
- Vendor repository cloning (Depth-Anything-3, SAM2, DSINE)
- Vendor package installation
- SAM2 checkpoint downloads
- Frontend dependencies (npm/bun)
- Configuration file initialization

**Model downloading is separate** to allow you to review and modify `config/models.json` first (set your preferred model directory, select which models to download, etc.)

#### Windows

```bash
git clone https://github.com/CastleGreyscale/FUK.git
cd FUK

# Install core package
pip install -e .

# Clone vendor dependencies
mkdir fuk\vendor
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git fuk/vendor/Depth-Anything-3
git clone https://github.com/facebookresearch/segment-anything-2 fuk/vendor/segment-anything-2
git clone https://github.com/baegwangbin/DSINE.git fuk/vendor/DSINE

# Install vendor packages
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2

# Install Depth Anything V3 dependencies
pip install -e ".[depth]"

# Download SAM2 checkpoints
cd fuk/vendor/segment-anything-2/checkpoints
bash download_ckpts.sh
cd ../../../..

# Frontend
cd fuk/ui
npm install  # or: bun install
cd ../..

# Edit configs
notepad config\models.json
notepad config\defaults.json

# Download models
python scripts\download_models.py

# Start server
python start_web_ui.py
```

### Optional: Binary Tools (Recommended)

For frame interpolation and upscaling, download pre-built binaries:

**RIFE (Frame Interpolation):**
1. Download: https://github.com/nihui/rife-ncnn-vulkan/releases
2. Extract to: `fuk/vendor/rife-ncnn/`
3. Ensure executable is in that directory

**Real-ESRGAN (Upscaling):**

Linux:
```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d fuk/vendor/realesrgan-ncnn/
```

Windows:
1. Download: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
2. Extract to: `fuk\vendor\realesrgan-ncnn\`

---

## Model Downloads

FUK uses DiffSynth-Studio for direct model access. Models are downloaded via Hugging Face.

### Automated Download (Recommended)

After editing `config/models.json` to set your model root directory:

```bash
python scripts/download_models.py
```

This downloads all configured models from Hugging Face to your specified directory.

### Manual Download

If you prefer to manage downloads yourself or have bandwidth constraints:

**Image Generation (Qwen):**
- Base: `Qwen/Qwen-Image`
- Updated: `Qwen/Qwen-Image-2512`
- Editing: `Qwen/Qwen-Image-Edit-2511`
- Control: `DiffSynth-Studio/Qwen-Image-In-Context-Control-Union`

**Video Generation (Wan):**
- I2V: `Wan-AI/Wan2.2-I2V-A14B`
- VACE: `PAI/Wan2.2-VACE-Fun-A14B`

**Download via Hugging Face CLI:**
```bash
huggingface-cli login  # authenticate first
huggingface-cli download Qwen/Qwen-Image --local-dir /your/models/path/Qwen/Qwen-Image
```

### Auto-Downloaded Models

These download automatically on first use:
- SAM2 segmentation checkpoints (via setup.sh)
- Depth-Anything-V3 weights (on first depth generation)
- Real-ESRGAN models (on first upscale operation)
- RIFE models (on first interpolation)

---

## Configuration

FUK uses a modular configuration system with separation between git-tracked templates and user-specific settings.

### Core Configuration Files

**`config/models.json`** - Model registry (YOUR PATHS):
```json
{
  "qwen_image": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"},
    "supports": ["negative_prompt"]
  }
}
```

**`config/defaults.json`** - Generation defaults:
```json
{
  "image": {
    "width": 1280,
    "height": 720,
    "steps": 20,
    "guidance_scale": 4.0
  },
  "vram": {
    "preset": "low"
  }
}
```

**`config/tools/*.json`** - Tool-specific configs:
- `depth-anything-v3.json` - Depth estimation parameters
- `sam2.json` - Segmentation settings

### VRAM Management

FUK includes intelligent VRAM presets for different hardware:

- **None** - No offloading (48GB+ VRAM)
- **Low** - CPU offload in bf16 (16-24GB VRAM, 64GB+ RAM) ← **Default**
- **Medium** - CPU offload in fp8 (16-24GB VRAM, 32GB+ RAM)
- **High** - Disk offload (8GB+ VRAM, slow but functional)

Edit `config/defaults.json` → `vram.preset` to change.

---

## First Run

```bash
python start_web_ui.py
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### UI Workflow

The interface follows a left-to-right pipeline:

1. **Image Tab** - Text-to-image, image-to-image, editing
2. **Video Tab** - Image-to-video, video generation
3. **Preprocess Tab** - Depth, normals, cryptomatte, pose
4. **Postprocess Tab** - Upscale, interpolate, temporal processing
5. **Layers Tab** - Multi-layer EXR assembly
6. **Export Tab** - Project delivery and packaging

Projects auto-version using date-based identifiers (`YYMMDD_NN`).

---

## Troubleshooting

### Python Dependencies

If you see import errors:

```bash
# Verify installation
pip list | grep fuk
pip list | grep diffsynth

# Reinstall if needed
pip install -e . --force-reinstall
```

### Depth-Anything-V3 Not Found

```bash
cd fuk/vendor/Depth-Anything-3
pip install -e .

# Verify
python -c "import depth_anything_3; print('✓ DA3 installed')"
```

Common causes:
- Missing `[depth]` optional dependencies
- Virtual environment not activated
- Installed with `--no-deps` flag

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"

# Should return True and your GPU name
```

If False, reinstall PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### VRAM Issues

Edit `config/defaults.json`:
```json
{
  "vram": {
    "preset": "medium"  // or "high" for aggressive offloading
  }
}
```

### Model Download Failures

Some models require Hugging Face authentication:
```bash
huggingface-cli login
```

### Frontend Won't Start

```bash
cd fuk/ui
rm -rf node_modules
npm install  # or: bun install
npm run dev
```

### Generation Hangs or Crashes

Check logs at `fuk/logs/fuk.log` for detailed error messages.

Common fixes:
- Reduce batch size in generation config
- Enable more aggressive VRAM offloading
- Verify model files are complete (re-download if corrupted)

---

## Development

### Development Install

For hot-reload during development:

```bash
# Backend (auto-reloads on code changes)
uvicorn fuk.fuk_web_server:app --reload --host 0.0.0.0 --port 8000

# Frontend (Vite dev server with HMR)
cd fuk/ui
npm run dev
```

### Adding New Models

1. Add model entry to `config/models.json` following the schema
2. Models are automatically loaded via DiffSynth-Studio registry
3. No code changes needed for compatible models

### Extending Preprocessors

1. Subclass `BasePreprocessor` in `fuk/core/preprocessors/`
2. Implement `process_image()` or `process_video()`
3. Register in preprocessor manager
4. Add config file in `config/tools/`

See `fuk/core/preprocessors/depth.py` as reference implementation.

---

## Why FUK?

Most AI generation tools are built for hobbyists creating single images. FUK is built for professionals who need:

- **Pipeline integration** - AI content that fits into existing compositing workflows
- **Reproducible results** - Proper versioning, seed tracking, generation history
- **Technical outputs** - Depth, normals, mattes for downstream work
- **Stability** - Tools that don't break between projects
- **Local control** - No cloud dependencies, your hardware, your data

FUK doesn't replace your workflow - it extends it.

---

## Technical Notes

### Direct Model Integration

FUK uses DiffSynth-Studio for direct model access rather than wrapper libraries. This provides:
- Stable APIs that don't change with every release
- Access to full model capabilities and parameters
- Better VRAM management and model offloading
- Reduced dependency chains and breakage points

### Latent-Space Operations

FUK maintains latent representations throughout the pipeline for:
- Lossless iteration without quality degradation
- Faster experimentation (no decode/encode cycles)
- Future latent manipulation features (camera control, lighting)

### Temporal Consistency

Video and sequence processing includes:
- Batch-based normalization for depth/normals
- Cross-frame coherence in cryptomatte generation
- Optical flow-guided upscaling and interpolation

### File Organization

FUK respects your project structure. Set your project root and FUK creates necessary subdirectories while leaving existing files untouched. No forced copying, no centralized file dumps.

---

## Known Issues

- Video upscaling UI may freeze when loading large results (processing continues, UI issue only)
- Progress bar implementation inconsistent across some operations
- Image dropzone requires precise targeting in some browsers

See GitHub Issues for current bugs and feature requests.

---

## Contributing

This project is in active development. Contribution guidelines coming soon.

For bug reports and feature requests, use GitHub Issues with detailed logs from `fuk/logs/`.

---

## License

MIT License - see LICENSE file for details

---

## Name

Yes, it's called FUK. **F**ramework for **U**nified **K**reation.

Because professional tools shouldn't require professional patience.

**Get FUKed. Render locally.**
