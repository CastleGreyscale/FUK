#!/usr/bin/env bash
#
# SeedVR2 isolated environment installer
#
# SeedVR2 needs flash-attn and apex built against its own torch. FUK's main
# venv runs torch 2.9+cu130 for Qwen and Wan; installing SeedVR2's stack into
# it would put generation at risk. So SeedVR2 gets its own self-contained
# environment and FUK talks to it over a subprocess (see seedvr2_worker.py).
#
# Same shape as install_trellis_env.sh. Everything lands under fuk/vendor/
# (gitignored). No system packages, no conda in the user's shell, no changes
# to the main venv.
#
# Usage:  bash fuk/core/install_seedvr2_env.sh
#
# Env overrides:
#   SEEDVR2_WEIGHTS_DIR   where checkpoints go   (default ~/ai/models/seedvr2)
#   SEEDVR2_VARIANT       3b | 7b | both         (default 3b)
#
set -uo pipefail

CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$(cd "$CORE_DIR/.." && pwd)/vendor"
ENV_ROOT="$VENDOR_DIR/SEEDVR2_ENV"
MAMBA_BIN="$ENV_ROOT/bin/micromamba"
ENV_PREFIX="$ENV_ROOT/env"
SEEDVR_SRC="$VENDOR_DIR/SeedVR"
READY_MARKER="$ENV_ROOT/READY"
WEIGHTS_DIR="${SEEDVR2_WEIGHTS_DIR:-$HOME/ai/models/seedvr2}"
VARIANT="${SEEDVR2_VARIANT:-3b}"

# Pinned to the combination SeedVR2 is developed against and for which
# flash-attn ships prebuilt wheels. cu121 is deliberate: flash-attn's cu124
# wheels for torch 2.4 are patchier, and the isolated env carries its own
# runtime so it does not matter what the system CUDA is.
PY_VERSION="3.10"
CUDA_VERSION="12.1"
TORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
FLASH_ATTN_VERSION="2.5.9.post1"

log()  { printf '\033[96m[seedvr2-env]\033[0m %s\n' "$*"; }
ok()   { printf '\033[92m[seedvr2-env] ✔\033[0m %s\n' "$*"; }
warn() { printf '\033[93m[seedvr2-env] ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[91m[seedvr2-env] ✗\033[0m %s\n' "$*"; exit 1; }

rm -f "$READY_MARKER"

# ---------------------------------------------------------------------------
# 1. Source
# ---------------------------------------------------------------------------
if [ ! -d "$SEEDVR_SRC/projects" ]; then
    log "cloning SeedVR (Apache 2.0)…"
    git clone --depth 1 https://github.com/ByteDance-Seed/SeedVR.git "$SEEDVR_SRC" \
        || die "clone failed"
else
    ok "source present at $SEEDVR_SRC"
fi

mkdir -p "$ENV_ROOT/bin" "$WEIGHTS_DIR"

# ---------------------------------------------------------------------------
# 2. micromamba — a single static binary, used only to provision a private
#    python + CUDA runtime. Never added to the user's shell.
# ---------------------------------------------------------------------------
if [ ! -x "$MAMBA_BIN" ]; then
    log "fetching micromamba…"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "$ENV_ROOT" bin/micromamba >/dev/null 2>&1 \
        || die "could not fetch micromamba"
fi
ok "micromamba ready"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    log "creating python $PY_VERSION environment…"
    "$MAMBA_BIN" create -y -p "$ENV_PREFIX" -r "$ENV_ROOT" \
        -c conda-forge \
        "python=$PY_VERSION" "cuda-runtime=$CUDA_VERSION" ffmpeg \
        || die "environment creation failed"
fi
ok "python environment ready"

PY="$ENV_PREFIX/bin/python"

# ---------------------------------------------------------------------------
# 3. Torch stack
# ---------------------------------------------------------------------------
log "installing torch $TORCH_VERSION (cu121)…"
"$PY" -m pip install --quiet --upgrade pip || die "pip upgrade failed"
"$PY" -m pip install --quiet \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX" || die "torch install failed"
ok "torch installed"

log "installing flash-attn $FLASH_ATTN_VERSION (prebuilt wheel, no compile)…"
"$PY" -m pip install --quiet "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation \
    || warn "flash-attn install failed — SeedVR2 will fall back to SDPA (slower)"

log "installing SeedVR requirements…"
if [ -f "$SEEDVR_SRC/requirements.txt" ]; then
    # torch is already pinned above; letting the requirements file resolve it
    # again is how the env ends up with a second, wrong torch.
    grep -viE '^(torch|torchvision|torchaudio)([=<>~!]|$)' \
        "$SEEDVR_SRC/requirements.txt" > "$ENV_ROOT/requirements.filtered.txt"
    "$PY" -m pip install --quiet -r "$ENV_ROOT/requirements.filtered.txt" \
        || warn "some requirements failed — check the log above"
fi
"$PY" -m pip install --quiet mediapy einops omegaconf tqdm || warn "helper install incomplete"
ok "dependencies installed"

# ---------------------------------------------------------------------------
# 4. Weights  (Apache 2.0 — commercial use permitted)
# ---------------------------------------------------------------------------
"$PY" -m pip install --quiet "huggingface_hub[cli]" || die "huggingface_hub install failed"

fetch_variant() {
    local size="$1" repo ckpt
    case "$size" in
        3b) repo="ByteDance-Seed/SeedVR2-3B"; ckpt="seedvr2_ema_3b.pth" ;;
        7b) repo="ByteDance-Seed/SeedVR2-7B"; ckpt="seedvr2_ema_7b.pth" ;;
        *)  warn "unknown variant '$size', skipping"; return ;;
    esac

    if [ -f "$WEIGHTS_DIR/$ckpt" ]; then
        ok "$ckpt already present"
        return
    fi
    log "downloading $repo → $WEIGHTS_DIR (this is several GB)…"
    "$ENV_PREFIX/bin/huggingface-cli" download "$repo" \
        --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False \
        || warn "download of $repo failed — fetch it manually into $WEIGHTS_DIR"
}

case "$VARIANT" in
    both) fetch_variant 3b; fetch_variant 7b ;;
    *)    fetch_variant "$VARIANT" ;;
esac

# SeedVR2 needs the shared VAE and text embeddings alongside the DiT weights;
# without them the runner fails at config load, not at inference, which is a
# confusing place to discover a missing file.
log "fetching VAE + text embeddings…"
"$ENV_PREFIX/bin/huggingface-cli" download ByteDance-Seed/SeedVR2-3B \
    --include "*.pth" "*.yaml" "*.json" \
    --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False >/dev/null 2>&1 \
    || warn "auxiliary file fetch incomplete"

# ---------------------------------------------------------------------------
# 5. Verify — the marker is written only if the imports the worker needs
#    actually resolve. A half-built env must not look installed.
# ---------------------------------------------------------------------------
log "verifying…"
if "$PY" - <<'PYEOF'
import sys
sys.path.insert(0, __import__("os").environ.get("SEEDVR_SRC", "."))
import torch
assert torch.cuda.is_available(), "CUDA not available inside the isolated env"
import torchvision, mediapy, omegaconf  # noqa: F401
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
try:
    import flash_attn  # noqa: F401
    print("flash-attn present")
except ImportError:
    print("flash-attn absent — SDPA fallback")
PYEOF
then
    if compgen -G "$WEIGHTS_DIR/*.pth" > /dev/null; then
        touch "$READY_MARKER"
        ok "SeedVR2 environment ready"
        echo
        echo "  Weights:  $WEIGHTS_DIR"
        echo "  Env:      $ENV_PREFIX"
        echo "  Restart the FUK server to pick it up."
    else
        die "no checkpoints in $WEIGHTS_DIR — environment built but unusable"
    fi
else
    die "verification failed — environment built but unusable"
fi
