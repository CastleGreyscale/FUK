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
echo "[1/7] Installing core FUK dependencies..."
pip install -e . --no-deps
pip install -e .

echo ""
echo "[2/7] Setting up vendor directory..."
mkdir -p fuk/vendor

echo ""
echo "[3/7] Cloning vendor dependencies..."

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

# musubi-tuner
if [ ! -d "fuk/vendor/musubi-tuner" ]; then
    echo "  → Cloning musubi-tuner..."
    git clone https://github.com/kohya-ss/musubi-tuner.git fuk/vendor/musubi-tuner
else
    echo "  ✓ musubi-tuner already exists"
fi

# DSINE (normals)
if [ ! -d "fuk/vendor/DSINE" ]; then
    echo "  → Cloning DSINE..."
    git clone https://github.com/baegwangbin/DSINE.git fuk/vendor/DSINE
else
    echo "  ✓ DSINE already exists"
fi

echo ""
echo "[4/7] Installing vendor packages..."
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2
pip install -e ./fuk/vendor/musubi-tuner

echo ""
echo "[5/7] Installing Depth Anything V3 dependencies..."
pip install -e ".[depth]"

echo ""
echo "[6/7] Setting up configuration files..."
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

# Configuration files (personal settings)
config/models.json
config/defaults.json
config/tools/*.json
!config/tools/*.json.example
config/*.json
!config/*.json.example
EOL
else
    echo "  ✓ .gitignore already exists"
fi

# Add config patterns to .gitignore if not already present
if ! grep -q "config/models.json" .gitignore 2>/dev/null; then
    echo "  → Adding config patterns to .gitignore..."
    cat >> .gitignore << 'EOL'

# Configuration files (personal settings)
config/models.json
config/defaults.json
config/tools/*.json
!config/tools/*.json.example
config/*.json
!config/*.json.example
EOL
fi

# Create example config files if they don't exist
echo ""
echo "  → Setting up example configuration files..."

# models.json.example
if [ ! -f "config/models.json.example" ]; then
    cat > config/models.json.example << 'EOL'
{
  "models": {
    "qwen-7b": {
      "path": "/path/to/Qwen2-VL-7B-Instruct",
      "type": "qwen"
    }
  },
  "wan_models": {
    "wan2.1": {
      "path": "/path/to/wan2.1-fp16",
      "version": "2.1"
    },
    "wan2.2": {
      "path": "/path/to/wan2.2-fp16",
      "version": "2.2"
    }
  },
  "loras": {
    "example-lora": {
      "path": "/path/to/lora.safetensors",
      "strength": 1.0
    }
  }
}
EOL
    echo "    ✓ Created config/models.json.example"
fi

# defaults.json.example
if [ ! -f "config/defaults.json.example" ]; then
    cat > config/defaults.json.example << 'EOL'
{
  "image": {
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "guidance_scale": 3.5,
    "seed": -1
  },
  "video": {
    "width": 512,
    "height": 512,
    "frames": 49,
    "steps": 28,
    "guidance_scale": 7.0,
    "seed": -1,
    "fps": 24
  },
  "depth": {
    "max_depth": 20.0
  },
  "performance": {
    "fp8": false,
    "cpu_offload": false,
    "attention_mode": "xformers"
  }
}
EOL
    echo "    ✓ Created config/defaults.json.example"
fi

# Create tools directory
mkdir -p config/tools

# musubi-wan.json.example
if [ ! -f "config/tools/musubi-wan.json.example" ]; then
    cat > config/tools/musubi-wan.json.example << 'EOL'
{
  "script": "wan_generate_video.py",
  "performance": {
    "fp8": false,
    "cpu_offload": false,
    "attention_mode": "xformers"
  },
  "arg_mapping": {
    "guidance_scale": "guidance_scale",
    "steps": "num_inference_steps",
    "seed": "seed"
  },
  "cli_flags": {
    "enable_tiling": "--enable-tiling"
  }
}
EOL
    echo "    ✓ Created config/tools/musubi-wan.json.example"
fi

# Copy example configs to actual configs if they don't exist
echo ""
echo "  → Initializing configuration files from examples..."

if [ ! -f "config/models.json" ]; then
    cp config/models.json.example config/models.json
    echo "    ✓ Created config/models.json from example (EDIT THIS FILE)"
else
    echo "    ✓ config/models.json already exists (not overwriting)"
fi

if [ ! -f "config/defaults.json" ]; then
    cp config/defaults.json.example config/defaults.json
    echo "    ✓ Created config/defaults.json from example"
else
    echo "    ✓ config/defaults.json already exists (not overwriting)"
fi

if [ ! -f "config/tools/musubi-wan.json" ]; then
    cp config/tools/musubi-wan.json.example config/tools/musubi-wan.json
    echo "    ✓ Created config/tools/musubi-wan.json from example"
else
    echo "    ✓ config/tools/musubi-wan.json already exists (not overwriting)"
fi

echo ""
echo "Note: RIFE and Real-ESRGAN are optional but recommended for"
echo "frame interpolation and upscaling. Download pre-built binaries:"
echo ""
echo "  RIFE: https://github.com/nihui/rife-ncnn-vulkan/releases"
echo "  Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN/releases"
echo ""
echo "Extract to:"
echo "  fuk/vendor/rife-ncnn/"
echo "  fuk/vendor/realesrgan-ncnn/"
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

echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. IMPORTANT: Edit your configuration files:"
echo "   config/models.json      - Set your model paths"
echo "   config/defaults.json    - Adjust default parameters"
echo ""
echo "   Example configs are in:"
echo "   config/*.json.example"
echo ""
echo "2. Download models (see README for details):"
echo "   - Qwen2-VL-7B-Instruct"
echo "   - WanX-FluxV1"
echo ""
echo "3. Start FUK:"
echo "   python start_web_ui.py"
echo ""
echo "For model downloads, see:"
echo "https://github.com/kohya-ss/musubi-tuner"
echo ""
echo "Configuration notes:"
echo "  • Your config files (config/*.json) are gitignored"
echo "  • Example configs (config/*.json.example) are tracked in git"
echo "  • Update examples when adding new config options"
echo ""