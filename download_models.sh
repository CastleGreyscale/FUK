#!/bin/bash
set -e

# Always run relative to repo root regardless of where the script is called from
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

echo "============================================"
echo "FUK — Model Downloader"
echo "============================================"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────

# Make sure setup has been run
if [ ! -d "venv" ]; then
    echo "✗ No virtual environment found."
    echo "  Run setup.sh first, then come back here."
    echo ""
    exit 1
fi

# Make sure config exists and has been edited
CONFIG="fuk/config/defaults.json"
if [ ! -f "$CONFIG" ]; then
    echo "✗ fuk/config/defaults.json not found."
    echo "  Run setup.sh first — it will create your config files."
    echo ""
    exit 1
fi

# Warn if models_root still looks like the template placeholder
MODELS_ROOT=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('models_root',''))" 2>/dev/null || echo "")
if [ -z "$MODELS_ROOT" ] || [ "$MODELS_ROOT" = "/path/to/your/models" ]; then
    echo "✗ models_root in fuk/config/defaults.json has not been set."
    echo ""
    echo "  Open the file and set it to where you want models stored, e.g.:"
    echo "    \"models_root\": \"/home/brad/ai/models\""
    echo ""
    echo "  Then re-run this script."
    echo ""
    exit 1
fi

echo "  ✓ Config found"
echo "  ✓ models_root: $MODELS_ROOT"
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────────
echo "Activating virtual environment..."
source venv/bin/activate
echo "  ✓ $(python --version) at $(which python)"
echo ""

# ── Run downloader ────────────────────────────────────────────────────────────
# DiffSynth defaults to ModelScope. HuggingFace is usually much faster from
# outside China, but a few repos (PAI/*, DiffSynth-Studio's converted Wan
# safetensors) exist only on ModelScope — the downloader falls back per
# component, so either choice completes.
echo "Starting model downloads..."
echo "  (Edit fuk/config/models.json to remove models you don't want)"
echo ""
echo "  Download source: ${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope (default)}"
if [ -z "${DIFFSYNTH_DOWNLOAD_SOURCE:-}" ]; then
    echo "  Slow or unstable? Try:  DIFFSYNTH_DOWNLOAD_SOURCE=huggingface ./download_models.sh"
fi
echo ""

# SeedVR2 is fetched at the end of the same run. It is not in models.json —
# it is not a DiffSynth pipeline — and it comes from HuggingFace regardless of
# DIFFSYNTH_DOWNLOAD_SOURCE. Skipped automatically if setup.sh has not vendored
# the engine, in which case video upscaling falls back to per-frame Real-ESRGAN.
if [ -d "fuk/vendor/SeedVR2" ]; then
    echo "  Also fetching: SeedVR2 video restoration weights (~20GB, HuggingFace)"
    echo "                 3B + 7B, each at fp8 and quantized. The 7B Sharp"
    echo "                 variants are not pulled here — they download on first use."
else
    echo "  Skipping SeedVR2 — engine not vendored (run setup.sh to enable)"
fi
echo ""

python fuk/utils/download_models.py "$@"
