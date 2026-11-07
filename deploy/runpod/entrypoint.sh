#!/bin/bash
# Pod boot: lay out the volume, wire FUK onto it, start SSH and FUK, stay up.
#
# Volume layout (all persistent):
#   /workspace/fuk/config   FUK's fuk/config      (pushed with `fukpod push-config`)
#   /workspace/fuk/data     FUK's fuk/ui/data     (tags, seeds, perf history)
#   /workspace/fuk/models   models_root           (weights, LoRAs under loras/)
#   /workspace/fuk/cache    fuk/cache             (generation outputs, latents)
#   /workspace/fuk/outputs  fuk/outputs
#   /workspace/fuk/home     $HOME for FUK         (projects go here)
#   /workspace/fuk/hf       HF_HOME
#   /workspace/fuk/logs     server.log, ui.log
#
# Env (set on the RunPod template):
#   PUBLIC_KEY     injected by RunPod from your account's SSH keys
#   HF_TOKEN       a RunPod secret; needed for gated repos
#   FUK_PREFETCH   optional, e.g. "ltx2 ltx2:loras tool:ltx_hdr" — starts those
#                  downloads once the server is up (see prefetch.sh)
set -uo pipefail

FUK=/opt/fuk
W="${FUK_WORKSPACE:-/workspace/fuk}"
SEED=/opt/fuk-seed
HERE="$FUK/deploy/runpod"

echo "[pod] FUK ${FUK_GIT_SHA:-?} — workspace $W"
mkdir -p "$W"/{config,data,models/loras,cache,outputs,home,hf,logs}

# ── Config and data ──────────────────────────────────────────────────────────
# First boot on an empty volume: seed from the image's setup.sh output so the
# server can start. Push your real configs afterwards with `fukpod push-config`.
if [ ! -f "$W/config/models.json" ]; then
    echo "[pod] empty config — seeding from image templates"
    cp -r --update=none "$SEED/config/." "$W/config/"
fi
if [ -z "$(ls -A "$W/data" 2>/dev/null)" ]; then
    cp -r --update=none "$SEED/data/." "$W/data/"
fi

link() {  # link <path-in-repo> <target-on-volume>
    rm -rf "$FUK/$1"
    ln -sfn "$2" "$FUK/$1"
}
link fuk/config  "$W/config"
link fuk/ui/data "$W/data"
link fuk/cache   "$W/cache"
link fuk/outputs "$W/outputs"

python3 "$HERE/pod_paths.py" "$W"

# ── Environment for later SSH sessions ───────────────────────────────────────
# Processes started over SSH (fukpod restart) don't see the container's env.
: > "$W/.pod_env"
chmod 600 "$W/.pod_env"
for var in HF_TOKEN FUK_PREFETCH FUK_GIT_SHA; do
    if [ -n "${!var:-}" ]; then
        printf '%s=%q\n' "$var" "${!var}" >> "$W/.pod_env"
    fi
done

# ── SSH ──────────────────────────────────────────────────────────────────────
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
    while IFS= read -r key; do
        [ -n "$key" ] && ! grep -qxF "$key" /root/.ssh/authorized_keys \
            && echo "$key" >> /root/.ssh/authorized_keys
    done <<< "$PUBLIC_KEY"
else
    echo "[pod] WARNING: no PUBLIC_KEY — you will not be able to SSH in"
fi
ssh-keygen -A > /dev/null
mkdir -p /run/sshd
# Key-only. Port forwarding stays on: it's how the UI is reached.
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password

# ── FUK ──────────────────────────────────────────────────────────────────────
"$HERE/services.sh" start

if [ -n "${FUK_PREFETCH:-}" ]; then
    # Background: a 70GB download must not hold up the pod coming up.
    setsid "$HERE/prefetch.sh" $FUK_PREFETCH >> "$W/logs/prefetch.log" 2>&1 < /dev/null &
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/[pod] GPU: /'
echo "[pod] up. From your machine:  fukpod connect <ip> <port> && fukpod ui"

# Stay up even if FUK dies, so you can SSH in, read the logs and restart it.
# This shell is PID 1, so every process orphaned by a restart is re-parented to
# it. `exec sleep infinity` would never reap those; bash's SIGCHLD handling
# does, each time `wait` returns.
while true; do
    sleep 3600 &
    wait $!
done
