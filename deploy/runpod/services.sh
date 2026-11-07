#!/bin/bash
# Start / stop the FUK backend and frontend inside the pod.
#
#   services.sh start | stop | restart | status
#
# Used by entrypoint.sh at boot and by `fukpod restart` over SSH. Each process
# is started detached with its log on the volume, so a crash leaves the pod up
# and the log readable (`fukpod logs`) instead of taking the container down.
set -uo pipefail

FUK=/opt/fuk
W="${FUK_WORKSPACE:-/workspace/fuk}"
LOGS="$W/logs"
mkdir -p "$LOGS"

# FUK's $HOME on the pod. It has to be on the volume, because the file browser
# only browses under $HOME (and /tmp) and projects must survive the pod. It
# should also be the *same path* as on the workstation, because project files
# store absolute paths. So the workstation's home (sent by `fukpod push-config`)
# is made a symlink to $W/home, and FUK runs with that as $HOME: a project at
# /home/you/Projects/x is at /home/you/Projects/x on both machines.
pod_home() {
    local h=""
    if [ -f "$W/config/.home_paths.json" ]; then
        h="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("home", ""))' \
             "$W/config/.home_paths.json" 2>/dev/null)"
    fi
    # Only ever replace nothing or an old symlink, never a real directory.
    if [ -n "$h" ] && [ "$h" != "/root" ] && { [ ! -e "$h" ] || [ -L "$h" ]; }; then
        mkdir -p "$(dirname "$h")"
        ln -sfn "$W/home" "$h"
        echo "$h"
    else
        echo "$W/home"
    fi
}

# What the FUK processes run with.
fuk_env() {
    export HOME="$(pod_home)"
    export HF_HOME="$W/hf"
    export HF_XET_HIGH_PERFORMANCE=1
    # RunPod secrets arrive as env vars; ssh sessions don't inherit the
    # container's environment, so entrypoint.sh saves them here.
    if [ -f "$W/.pod_env" ]; then
        set -a; . "$W/.pod_env"; set +a
    fi
}

backend_pid()  { pgrep -f "fuk/ui/fuk_web_server.py" | head -1; }
frontend_pid() { pgrep -f "vite --host" | head -1; }

start() {
    fuk_env
    # The redirects sit on the backgrounded subshell itself, not on the command
    # inside it. Otherwise the subshell keeps the caller's stdout open, and an
    # `ssh pod services.sh restart` never returns.
    if [ -z "$(backend_pid)" ]; then
        (cd "$FUK" && exec setsid "$FUK/venv/bin/python" fuk/ui/fuk_web_server.py) \
            >> "$LOGS/server.log" 2>&1 < /dev/null &
        echo "backend started  (log: $LOGS/server.log)"
    else
        echo "backend already running"
    fi
    if [ -z "$(frontend_pid)" ]; then
        (cd "$FUK/fuk/ui" && exec setsid npm run dev -- --port 3000 --strictPort) \
            >> "$LOGS/ui.log" 2>&1 < /dev/null &
        echo "frontend started (log: $LOGS/ui.log)"
    else
        echo "frontend already running"
    fi
}

stop() {
    pkill -f "fuk/ui/fuk_web_server.py" && echo "backend stopped"
    pkill -f "vite --host" && echo "frontend stopped"
    # The SeedVR2 / LTX-HDR engines run as children; don't leave one holding VRAM.
    pkill -f "ltx_hdr_runner.py" 2>/dev/null
    pkill -f "inference_cli.py" 2>/dev/null
    sleep 2
    return 0
}

status() {
    [ -f "$W/.pod_env" ] && . "$W/.pod_env"
    echo "backend:  ${FUK_GIT_SHA:-?} $( [ -n "$(backend_pid)" ] && echo "running (pid $(backend_pid))" || echo stopped)"
    echo "frontend: $( [ -n "$(frontend_pid)" ] && echo "running (pid $(frontend_pid))" || echo stopped)"
    echo "home:     $(pod_home)"
    curl -s -m 3 localhost:8000/health > /dev/null && echo "health:   ok" || echo "health:   not answering"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    *) echo "usage: $0 start|stop|restart|status" >&2; exit 1 ;;
esac
