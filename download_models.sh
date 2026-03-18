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
echo "Starting model downloads..."
echo "  (Edit fuk/config/models.json to remove models you don't want)"
echo ""

python fuk/utils/download_models.py "$@"
