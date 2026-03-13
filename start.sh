#!/bin/bash
# FUK Start Script
# Activates venv, starts frontend dev server, starts backend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
