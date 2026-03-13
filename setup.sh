#!/bin/bash
set -e

echo "============================================"
echo "FUK: Framework for Unified Kreation"
echo "Installation Script"
echo "============================================"
echo ""

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠  Warning: nvidia-smi not found. GPU acceleration may not work."
fi

# ── Virtual Environment (Python 3.10 required) ───────────────────────────────
echo ""
echo "[0/7] Setting up virtual environment..."

VENV_DIR="venv"
PYTHON_BIN=""

# Prefer explicit python3.10; fall back with a hard error
for candidate in python3.10 python3; do
    if command -v "$candidate" &> /dev/null; then
        VER=$("$candidate" -c "import sys; print('%d.%d' % sys.version_info[:2])")
        if [ "$VER" = "3.10" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "  ✗ Error: Python 3.10 is required but was not found."
    echo "    Install it with:  sudo apt install python3.10 python3.10-venv"
    echo "    Or via pyenv:     pyenv install 3.10 && pyenv local 3.10"
    exit 1
fi

echo "  ✓ Found Python 3.10: $(which $PYTHON_BIN)"

if [ ! -d "$VENV_DIR" ]; then
    echo "  → Creating virtual environment at ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created"
else
    # Verify the existing venv is actually 3.10
    VENV_VER=$("$VENV_DIR/bin/python" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || echo "unknown")
    if [ "$VENV_VER" != "3.10" ]; then
        echo "  ⚠  Existing venv is Python $VENV_VER, not 3.10 — recreating..."
        rm -rf "$VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        echo "  ✓ Virtual environment recreated with Python 3.10"
    else
        echo "  ✓ Virtual environment already exists (Python 3.10)"
    fi
fi

echo "  → Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "  ✓ Using Python: $(python --version) at $(which python)"

# Upgrade pip and install wheel up front — prevents legacy setup.py install
# warnings for packages like antlr4-python3-runtime, basicsr, moviepy
echo "  → Upgrading pip + installing build tools..."
pip install --quiet --upgrade pip wheel setuptools
echo "  ✓ pip, wheel, setuptools up to date"

# ── PyTorch (CUDA) ────────────────────────────────────────────────────────────
# PyPI only hosts CPU-only torch wheels. CUDA builds must come from PyTorch's
# own index. This must run before `pip install -e .` so the CUDA wheels are
# already present and pip won't replace them with CPU-only versions.
echo ""
echo "[1/7] Installing PyTorch (CUDA)..."

# Detect CUDA version from nvcc to pick the right wheel index.
# Supports: cu118, cu124, cu126, cu128 (PyTorch 2.7 matrix)
CUDA_INDEX="cu128"  # safe default — most recent stable
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | tr -d '.')
    case "$CUDA_VER" in
        118|11*) CUDA_INDEX="cu118" ;;
        124)     CUDA_INDEX="cu124" ;;
        126)     CUDA_INDEX="cu126" ;;
        128|12*) CUDA_INDEX="cu128" ;;
        *)       CUDA_INDEX="cu128" ;;
    esac
    echo "  ✓ Detected CUDA $CUDA_VER → $CUDA_INDEX"
else
    echo "  ⚠  nvcc not found, defaulting to $CUDA_INDEX"
fi

echo "  → https://download.pytorch.org/whl/$CUDA_INDEX"
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
echo "  ✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  ✓ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Core FUK ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/7] Installing core FUK dependencies..."
# --no-deps first so pip resolves our already-installed CUDA torch
# rather than re-pulling CPU-only wheels from PyPI
pip install -e . --no-deps
pip install -e .

# ── Vendor directory ─────────────────────────────────────────────────────────
echo ""
echo "[3/7] Setting up vendor directory..."
mkdir -p fuk/vendor

# ── Vendor repos ─────────────────────────────────────────────────────────────
echo ""
echo "[4/7] Cloning vendor dependencies..."

# DiffSynth-Studio
if [ ! -d "fuk/vendor/DiffSynth-Studio" ]; then
    echo "  → Cloning DiffSynth-Studio..."
    git clone https://github.com/modelscope/DiffSynth-Studio.git fuk/vendor/DiffSynth-Studio
else
    echo "  ✓ DiffSynth-Studio already exists"
fi

# Depth Anything V3
if [ ! -d "fuk/vendor/Depth-Anything-3" ]; then
    echo "  → Cloning Depth-Anything-3..."
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git fuk/vendor/Depth-Anything-3
else
    echo "  ✓ Depth-Anything-3 already exists"
fi

# SAM2
if [ ! -d "fuk/vendor/segment-anything-2" ]; then
    echo "  → Cloning SAM2..."
    git clone https://github.com/facebookresearch/segment-anything-2 fuk/vendor/segment-anything-2
    echo "  → Downloading SAM2 checkpoints..."
    cd fuk/vendor/segment-anything-2/checkpoints
    bash download_ckpts.sh
    cd ../../../..
else
    echo "  ✓ segment-anything-2 already exists"
fi

# DSINE (normals)
if [ ! -d "fuk/vendor/DSINE" ]; then
    echo "  → Cloning DSINE..."
    git clone https://github.com/baegwangbin/DSINE.git fuk/vendor/DSINE
else
    echo "  ✓ DSINE already exists"
fi

# ── Vendor packages ───────────────────────────────────────────────────────────
echo ""
echo "[5/7] Installing vendor packages..."

# basicsr uses setup.py version introspection that breaks in isolated build
# environments (KeyError: '__version__'). --no-build-isolation lets it see
# the already-installed setuptools in our venv and avoids the error.
# Install it explicitly before DiffSynth-Studio pulls it in as a dep.
echo "  → Pre-installing basicsr (no-build-isolation workaround)..."
pip install basicsr --no-build-isolation

pip install -e ./fuk/vendor/DiffSynth-Studio
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2

# ── Configuration files ───────────────────────────────────────────────────────
echo ""
echo "[6/7] Setting up configuration files..."
mkdir -p fuk/config

# Initialize .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "  → Creating .gitignore..."
    cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# FUK specific
cache/
renders/
outputs/

# Configuration files (personal settings - gitignored, edit freely)
fuk/config/*.json
fuk/config/tools/*.json
!fuk/config/*.json.template
!fuk/config/tools/*.json.template
EOL
else
    echo "  ✓ .gitignore already exists"
fi

# Add config patterns to .gitignore if not already present
if ! grep -q "fuk/config/\*.json" .gitignore 2>/dev/null; then
    echo "  → Adding config patterns to .gitignore..."
    cat >> .gitignore << 'EOL'

# Configuration files (personal settings - gitignored, edit freely)
fuk/config/*.json
fuk/config/tools/*.json
!fuk/config/*.json.template
!fuk/config/tools/*.json.template
EOL
fi

# ── Templates ─────────────────────────────────────────────────────────────────
# Templates are ALWAYS written — they are the canonical reference and must
# stay in sync with the codebase. Personal fuk/config/*.json files are never
# touched after first creation.
echo ""
echo "  → Writing template configuration files..."

cat > fuk/config/models.json.template << 'EOL'
{
  "_comment": "FUK Model Registry — each entry defines everything needed to construct a DiffSynth pipeline. Add new models by adding entries here. Visit https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Qwen-Image.md and https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Wan.md for full model list. Note: not all options are setup. Below is the current list of working models. Before running download_models.py remove unwanted models. Roughly 15-30GB per model",

  "qwen_image": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "category": "image",
    "description": "Base text-to-image generation",
    "aliases": ["qwen", "t2i"],
    "supports": ["negative_prompt"],
    "parameter_map": {},
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"}
  },

  "qwen_image_2512": {
    "model_id": "Qwen/Qwen-Image-2512",
    "pipeline": "qwen",
    "category": "image",
    "description": "Qwen text-to-image (2512 update, improved quality)",
    "aliases": ["qwen-2512"],
    "supports": ["negative_prompt"],
    "parameter_map": {},
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors", "model_id": "Qwen/Qwen-Image"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors", "model_id": "Qwen/Qwen-Image"}
    ],
    "tokenizer": {"pattern": "tokenizer/", "model_id": "Qwen/Qwen-Image"}
  },

  "qwen_image_edit_2511": {
    "model_id": "Qwen/Qwen-Image-Edit-2511",
    "pipeline": "qwen",
    "category": "image",
    "description": "Multi-image editing (supports multiple input images)",
    "aliases": ["qwen-edit", "edit-2511"],
    "supports": ["edit_image", "negative_prompt"],
    "parameter_map": {
      "edit_targets": "edit_image"
    },
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors", "model_id": "Qwen/Qwen-Image"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors", "model_id": "Qwen/Qwen-Image"}
    ],
    "tokenizer": {"pattern": "tokenizer/", "model_id": "Qwen/Qwen-Image"},
    "processor": {"pattern": "processor/", "model_id": "Qwen/Qwen-Image-Edit"}
  },

  "qwen_image_control_union": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "category": "image",
    "description": "In-context control with preprocessed control images",
    "aliases": ["control-union", "qwen-control"],
    "supports": ["context_image", "negative_prompt"],
    "parameter_map": {
      "control_input": "context_image"
    },
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"},
    "lora": {
      "model_id": "DiffSynth-Studio/Qwen-Image-In-Context-Control-Union",
      "pattern": "model.safetensors",
      "target": "dit"
    }
  },

  "qwen_eligen": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "category": "image",
    "description": "Entity-level composition control (masks + per-entity prompts)",
    "aliases": ["eligen", "eligen-v2"],
    "supports": ["eligen", "negative_prompt"],
    "parameter_map": {},
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"},
    "lora": {
      "model_id": "DiffSynth-Studio/Qwen-Image-EliGen-V2",
      "pattern": "model.safetensors",
      "target": "dit"
    }
  },

  "wan_i2v_a14b": {
    "model_id": "Wan-AI/Wan2.2-I2V-A14B",
    "pipeline": "wan",
    "category": "video",
    "description": "Image-to-video (Wan 2.2 A14B - latest)",
    "aliases": ["i2v-A14B", "i2v-2.2"],
    "supports": ["input_image", "negative_prompt", "tiled"],
    "parameter_map": {
      "reference_image": "input_image"
    },
    "components": [
      {"pattern": "high_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "low_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "models_t5_umt5-xxl-enc-bf16.pth"},
      {"pattern": "Wan2.1_VAE.pth"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "wan_vace_a14b": {
    "model_id": "PAI/Wan2.2-VACE-Fun-A14B",
    "pipeline": "wan",
    "category": "video",
    "description": "VACE-controlled video (control video + reference image)",
    "aliases": ["vace", "vace-a14b"],
    "supports": ["vace_video", "vace_reference_image", "vace_scale", "negative_prompt", "tiled"],
    "parameter_map": {
      "control_input": "vace_video",
      "reference_image": "vace_reference_image"
    },
    "components": [
      {"pattern": "high_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "low_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "models_t5_umt5-xxl-enc-bf16.pth"},
      {"pattern": "Wan2.1_VAE.pth"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "wan_inp_a14b": {
    "model_id": "PAI/Wan2.2-Fun-A14B-InP",
    "pipeline": "wan",
    "category": "video",
    "description": "First+Last frame inpainting (Wan 2.2 A14B)",
    "aliases": ["inp", "inp-a14b", "inpaint-video"],
    "supports": ["input_image", "end_image", "negative_prompt", "tiled"],
    "parameter_map": {
      "reference_image": "input_image",
      "end_image": "end_image"
    },
    "components": [
      {"pattern": "high_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "low_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "models_t5_umt5-xxl-enc-bf16.pth"},
      {"pattern": "Wan2.1_VAE.pth"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "_deferred": {
    "_comment": "Complex models deferred for post-launch",
    "wan_t2v_1.3b": "Excluding T2V - not professional use case",
    "wan_t2v_14b": "Excluding T2V - not professional use case",
    "wan_s2v_14b": "Requires: s2v_pose_video + motion_video + audio pipeline",
    "wan_animate_14b": "Requires: animate_pose_video + animate_face_video + animate_inpaint_video + animate_mask_video",
    "wan_control_camera": "Requires: camera_control_direction parsing + special UI",
    "qwen_blockwise_controlnet": "Requires: preprocessor integration for canny/depth/inpaint"
  }
}
EOL
echo "    ✓ fuk/config/models.json.template"

cat > fuk/config/defaults.json.template << 'EOL'
{
  "models_root": "/path/to/your/models",

  "aspect_ratios": [
    { "label": "1:1 (Square)",        "value": "1:1",    "ratio": 1.0    },
    { "label": "1.33:1 (Fullscreen)", "value": "1.33:1", "ratio": 1.3333 },
    { "label": "1.78:1 (Widescreen)", "value": "1.78:1", "ratio": 1.7778 },
    { "label": "1.85:1 (Academy)",    "value": "1.85:1", "ratio": 1.85   },
    { "label": "2.39:1 (Anamorphic)", "value": "2.39:1", "ratio": 2.39   },
    { "label": "2.75:1 (Panavision)", "value": "2.75:1", "ratio": 2.76   }
  ],

  "image": {
    "prompt": "",
    "negative_prompt": "low resolution, low quality, limb deformities, finger deformities, over-saturated image, wax figure appearance, lack of facial detail, overly smooth, AI-generated look. Chaotic composition. Blurry, distorted text",
    "model": "qwen_image",
    "width": 1280,
    "aspectRatio": "2.39",
    "steps": 20,
    "stepsMode": "preset",
    "guidance_scale": 4,
    "cfg_scale": 5.0,
    "denoising_strength": 1,
    "lora": null,
    "lora_multiplier": 1.0,
    "seed": null,
    "seedMode": "random",
    "lastUsedSeed": null,
    "output_format": "png",
    "edit_strength": 1,
    "exponential_shift_mu": null,
    "save_latent": true,
    "control_image_paths": []
  },

  "video": {
    "model": "wan_i2v_a14b",
    "prompt": "",
    "negative_prompt": "bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
    "task": "i2v-A14B",
    "video_length": 41,
    "scale_factor": 1.0,
    "steps": 20,
    "stepsMode": "preset",
    "guidance_scale": 5.0,
    "cfg_scale": 6.0,
    "denoising_strength": 1.0,
    "sigma_shift": 5.0,
    "sliding_window_size": null,
    "sliding_window_stride": null,
    "motion_bucket_id": null,
    "lora": null,
    "lora_multiplier": 1.0,
    "seed": null,
    "save_latent": true,
    "seedMode": "random",
    "lastUsedSeed": null,
    "image_path": null,
    "end_image_path": null,
    "width": null,
    "height": null,
    "source_width": null,
    "source_height": null
  },

  "lora_dirs": [
    "/path/to/your/models/loras"
  ],

  "vram": {
    "preset": "low",
    "presets": {
      "none": {
        "label": "None — Full VRAM",
        "description": "No offloading. Fastest but needs 48GB+ for 14B models.",
        "config": null,
        "buffer_gb": 0.5
      },
      "low": {
        "label": "Low — CPU Offload (bf16)",
        "description": "Offload to RAM in bf16. Best for 16-24GB VRAM with 64GB+ RAM.",
        "config": {
          "offload_dtype": "bfloat16",
          "offload_device": "cpu",
          "onload_dtype": "bfloat16",
          "onload_device": "cpu",
          "preparing_dtype": "bfloat16",
          "preparing_device": "cuda",
          "computation_dtype": "bfloat16",
          "computation_device": "cuda"
        },
        "buffer_gb": 2
      },
      "medium": {
        "label": "Medium — CPU Offload (fp8)",
        "description": "Offload to RAM in fp8. Halves RAM usage, minor quality risk.",
        "config": {
          "offload_dtype": "float8_e4m3fn",
          "offload_device": "cpu",
          "onload_dtype": "float8_e4m3fn",
          "onload_device": "cpu",
          "preparing_dtype": "float8_e4m3fn",
          "preparing_device": "cuda",
          "computation_dtype": "bfloat16",
          "computation_device": "cuda"
        },
        "buffer_gb": 2
      },
      "high": {
        "label": "High — Disk Offload",
        "description": "Offload to disk. Runs on 8GB VRAM but significantly slower.",
        "config": {
          "offload_dtype": "disk",
          "offload_device": "disk",
          "onload_dtype": "bfloat16",
          "onload_device": "cpu",
          "preparing_dtype": "bfloat16",
          "preparing_device": "cuda",
          "computation_dtype": "bfloat16",
          "computation_device": "cuda"
        },
        "buffer_gb": 2
      }
    }
  },

  "export": {
    "prefer_latent": true,
    "exr_compression": "ZIP",
    "exr_precision": "float32"
  },

  "preprocess": {
    "depth": {
      "model": "da3_mono_large",
      "invert": false,
      "normalize": true,
      "colormap": "greyscale"
    },
    "normals": {
      "method": "depth_gradient",
      "depth_model": "da3_mono_large",
      "blur_radius": 3,
      "normalize": true
    },
    "canny": {
      "low_threshold": 1,
      "high_threshold": 20,
      "blur_kernel": 3,
      "invert": false
    },
    "openpose": {
      "include_hands": true,
      "include_face": true
    },
    "crypto": {
      "model": "sam2",
      "mode": "auto",
      "points_per_side": 32,
      "pred_iou_thresh": 0.86,
      "stability_score_thresh": 0.92
    }
  },

  "postprocess": {
    "upscale": {
      "model": "real_esrgan_x4",
      "scale": 4
    },
    "interpolate": {
      "model": "film",
      "multiplier": 2
    }
  },

  "project": {
    "shot_label": "shot",
    "versioning": "DATE",
    "autosave_interval": 30,
    "cache_cleanup_days": 7
  }
}
EOL
echo "    ✓ fuk/config/defaults.json.template"

# Copy templates to live configs only if they don't exist yet
echo ""
echo "  → Initializing live configuration files..."

if [ ! -f "fuk/config/models.json" ]; then
    cp fuk/config/models.json.template fuk/config/models.json
    echo "    ✓ Created fuk/config/models.json — set your models_root path"
else
    echo "    ✓ fuk/config/models.json already exists (not overwriting)"
fi

if [ ! -f "fuk/config/defaults.json" ]; then
    cp fuk/config/defaults.json.template fuk/config/defaults.json
    echo "    ✓ Created fuk/config/defaults.json — set your models_root and lora_dirs"
else
    echo "    ✓ fuk/config/defaults.json already exists (not overwriting)"
fi

# ── Frontend ──────────────────────────────────────────────────────────────────
echo ""
echo "Note: Real-ESRGAN is optional but recommended for upscaling."
echo "Download pre-built binaries:"
echo ""
echo "  Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN/releases"
echo ""
echo "Extract to: fuk/vendor/realesrgan-ncnn/"
echo ""

echo ""
echo "[7/7] Installing frontend dependencies..."
cd fuk/ui

if command -v bun &> /dev/null; then
    echo "  → Using bun..."
    bun install
elif command -v npm &> /dev/null; then
    echo "  → Using npm..."
    npm install
else
    echo "  ✗ Error: Neither bun nor npm found. Please install Node.js."
    exit 1
fi

cd ../..

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Virtual environment: ./venv"
echo "  Activate manually: source venv/bin/activate"
echo ""
echo "Next steps:"
echo ""
echo "1. IMPORTANT: Edit your configuration files:"
echo "   fuk/config/models.json      — set your models_root path"
echo "   fuk/config/defaults.json    — set your models_root and lora_dirs"
echo ""
echo "   Templates (tracked in git):"
echo "   fuk/config/models.json.template"
echo "   fuk/config/defaults.json.template"
echo ""
echo "2. Download models:"
echo "   python scripts/download_models.py"
echo ""
echo "3. Start FUK:"
echo "   ./start.sh"
echo ""
echo "Configuration notes:"
echo "  • Your config files (fuk/config/*.json) are gitignored"
echo "  • Templates (fuk/config/*.json.template) are tracked in git"
echo "  • Templates are always refreshed by setup.sh — edit freely"
echo ""