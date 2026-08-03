#!/bin/bash
# FUK Start Script
# Activates venv, starts frontend dev server, starts backend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: ./start.sh [options]

  --no-sage            Disable SageAttention, fall back to PyTorch SDPA
  --attention IMPL     Force an attention backend. One of:
                       torch, sage_attention, xformers,
                       flash_attention_2, flash_attention_3
  -h, --help           Show this help

With no options, DiffSynth auto-picks the fastest installed backend.
EOF
}

# Parse args. The attention backend is resolved at diffsynth import time, so it
# has to be in the environment before python starts.
while [ $# -gt 0 ]; do
    case "$1" in
        --no-sage)
            export DIFFSYNTH_ATTENTION_IMPLEMENTATION="torch"
            shift
            ;;
        --attention)
            if [ -z "$2" ]; then
                echo "Error: --attention needs a value" >&2
                exit 1
            fi
            export DIFFSYNTH_ATTENTION_IMPLEMENTATION="$2"
            shift 2
            ;;
        --attention=*)
            export DIFFSYNTH_ATTENTION_IMPLEMENTATION="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -n "$DIFFSYNTH_ATTENTION_IMPLEMENTATION" ]; then
    echo "→ Attention backend: $DIFFSYNTH_ATTENTION_IMPLEMENTATION"
fi

# Activate virtual environment if present
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "→ Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Start frontend dev server in background
echo "→ Starting frontend..."
cd "$SCRIPT_DIR/fuk/ui"
if command -v bun &> /dev/null; then
    bun run dev &
else
    npm run dev &
fi
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

# Trap Ctrl+C to kill both processes cleanly
trap "echo ''; echo 'Shutting down...'; kill $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo "→ Starting backend..."
echo ""
python fuk/ui/fuk_web_server.py

# If backend exits normally, clean up frontend too
kill $FRONTEND_PID 2>/dev/null
