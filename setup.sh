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
    echo "⚠ Warning: nvidia-smi not found. GPU acceleration may not work."
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
echo "[4/6] Installing vendor packages..."
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2
pip install -e ./fuk/vendor/musubi-tuner

echo ""
echo "[5/6] Installing Depth Anything V3 dependencies..."
pip install -e ".[depth]"

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
echo "1. Download models (see README for details):"
echo "   - Qwen2-VL-7B-Instruct"
echo "   - WanX-FluxV1"
echo ""
echo "2. Configure model paths in models.json"
echo ""
echo "3. Start FUK:"
echo "   python start_web_ui.py"
echo ""
echo "For model downloads, see:"
echo "https://github.com/kohya-ss/musubi-tuner"
echo ""
