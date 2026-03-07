#!/bin/bash
set -e

echo "============================================"
echo "FUK: Framework for Unified Kreation"
echo "Installation Script"
echo "============================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠  Warning: nvidia-smi not found. GPU acceleration may not work."
fi

echo ""
echo "[1/6] Installing core FUK dependencies..."
pip install -e . --no-deps
pip install -e .

echo ""
echo "[2/6] Setting up vendor directory..."
mkdir -p fuk/vendor

echo ""
echo "[3/6] Cloning vendor dependencies..."

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

echo ""
echo "[4/6] Installing vendor packages..."
pip install -e ./fuk/vendor/DiffSynth-Studio
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2

echo ""
echo "[5/6] Setting up configuration files..."
mkdir -p config

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
config/*.json
config/tools/*.json
!config/*.json.template
!config/tools/*.json.template
EOL
else
    echo "  ✓ .gitignore already exists"
fi

# Add config patterns to .gitignore if not already present
if ! grep -q "config/\*.json" .gitignore 2>/dev/null; then
    echo "  → Adding config patterns to .gitignore..."
    cat >> .gitignore << 'EOL'

# Configuration files (personal settings - gitignored, edit freely)
config/*.json
config/tools/*.json
!config/*.json.template
!config/tools/*.json.template
EOL
fi

# Create example config files if they don't exist
echo ""
echo "  → Setting up example configuration files..."

# models.json.template
if [ ! -f "config/models.json.template" ]; then
    cat > config/models.json.template << 'EOL'
{
  "_comment": "FUK Model Registry — each entry defines everything needed to construct a DiffSynth pipeline. Add new models by adding entries here.",

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
    "parameter_map": {"edit_targets": "edit_image"},
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
    "parameter_map": {"control_input": "context_image"},
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
    "parameter_map": {"reference_image": "input_image"},
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
    "pipeline_kwargs": {"tiled": true}
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
    "pipeline_kwargs": {"tiled": true}
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
    "pipeline_kwargs": {"tiled": true}
  }
}
EOL
    echo "    ✓ Created config/models.json.template"
fi

# defaults.json.template
if [ ! -f "config/defaults.json.template" ]; then
    cat > config/defaults.json.template << 'EOL'
{
  "models_root": "/path/to/your/models",

  "aspect_ratios": [
    { "label": "1:1 (Square)",      "value": "1:1",    "ratio": 1.0    },
    { "label": "1.33:1 (Fullscreen)","value": "1.33:1", "ratio": 1.3333 },
    { "label": "1.78:1 (Widescreen)","value": "1.78:1", "ratio": 1.7778 },
    { "label": "1.85:1 (Academy)",  "value": "1.85:1", "ratio": 1.85   },
    { "label": "2.39:1 (Anamorphic)","value": "2.39:1", "ratio": 2.39   },
    { "label": "2.75:1 (Panavision)","value": "2.75:1", "ratio": 2.76   }
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
    echo "    ✓ Created config/defaults.json.template"
fi

# Copy example configs to actual configs if they don't exist
echo ""
echo "  → Initializing configuration files from examples..."

if [ ! -f "config/models.json" ]; then
    cp config/models.json.template config/models.json
    echo "    ✓ Created config/models.json (set your models_root path)"
else
    echo "    ✓ config/models.json already exists (not overwriting)"
fi

if [ ! -f "config/defaults.json" ]; then
    cp config/defaults.json.template config/defaults.json
    echo "    ✓ Created config/defaults.json (set your models_root and lora_dirs)"
else
    echo "    ✓ config/defaults.json already exists (not overwriting)"
fi

echo ""
echo "Note: Real-ESRGAN is optional but recommended for upscaling."
echo "Download pre-built binaries:"
echo ""
echo "  Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN/releases"
echo ""
echo "Extract to: fuk/vendor/realesrgan-ncnn/"
echo ""

echo ""
echo "[6/6] Installing frontend dependencies..."
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

echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. IMPORTANT: Edit your configuration files:"
echo "   config/models.json      - Set your models_root path"
echo "   config/defaults.json    - Set your models_root and lora_dirs"
echo ""
echo "   Templates are in:"
echo "   config/*.json.template"
echo ""
echo "2. Download models:"
echo "   python scripts/download_models.py"
echo ""
echo "3. Start FUK:"
echo "   ./start.sh"
echo ""
echo "Configuration notes:"
echo "  • Your config files (config/*.json) are gitignored"
echo "  • Example configs (config/*.json.example) are tracked in git"
echo "  • Update examples when adding new config options"
echo ""