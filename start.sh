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

# Activate virtual environment if present. FUK_VENV picks a different one,
# e.g. FUK_VENV=venv312 ./start.sh (matches setup.sh).
VENV_DIR="${FUK_VENV:-venv}"
if [ -f "$SCRIPT_DIR/$VENV_DIR/bin/activate" ]; then
    echo "→ Activating virtual environment ($VENV_DIR)..."
    source "$SCRIPT_DIR/$VENV_DIR/bin/activate"
fi

# Backend first. It takes several seconds to import torch and DiffSynth, and a
# frontend started alongside it fills the terminal with vite proxy
# ECONNREFUSED errors until port 8000 opens.
echo "→ Starting backend..."
echo ""
python fuk/ui/fuk_web_server.py &
BACKEND_PID=$!
FRONTEND_PID=""

# Trap Ctrl+C to kill both processes cleanly
trap 'echo ""; echo "Shutting down..."; kill $FRONTEND_PID $BACKEND_PID 2>/dev/null; exit 0' INT TERM

until curl -fsS -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "✗ Backend exited before it was ready — see the error above." >&2
        exit 1
    fi
    sleep 0.5
done

echo "→ Backend ready. Starting frontend..."
cd "$SCRIPT_DIR/fuk/ui"
if command -v bun &> /dev/null; then
    bun run dev &
else
    npm run dev &
fi
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

# Stay attached to the backend; when it exits, take the frontend down too.
wait "$BACKEND_PID"
kill $FRONTEND_PID 2>/dev/null
