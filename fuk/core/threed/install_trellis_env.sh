#!/usr/bin/env bash
#
# TRELLIS isolated environment installer
#
# TRELLIS needs torch 2.4–2.6 era CUDA extensions (spconv, nvdiffrast,
# diff_gaussian_rasterization). FUK's main venv runs torch 2.9+cu130 and the
# system toolchain is CUDA 13.3 / gcc 16 — neither of which those extensions
# build against. Installing them into the main venv would put Qwen and Wan
# generation at risk, so TRELLIS gets its own self-contained environment and
# FUK talks to it over a subprocess (see trellis_worker.py).
#
# Everything lands under fuk/vendor/ (gitignored). Nothing outside that
# directory and the model cache is touched — no system packages, no conda in
# the user's shell, no changes to the main venv.
#
# Usage:  bash fuk/core/threed/install_trellis_env.sh
#
set -uo pipefail

THREED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$(cd "$THREED_DIR/../.." && pwd)/vendor"
ENV_ROOT="$VENDOR_DIR/TRELLIS_ENV"
MAMBA_BIN="$ENV_ROOT/bin/micromamba"
ENV_PREFIX="$ENV_ROOT/env"
TRELLIS_SRC="$VENDOR_DIR/TRELLIS"
BUILD_DIR="$ENV_ROOT/build"
WEIGHTS_DIR="${TRELLIS_WEIGHTS_DIR:-$HOME/ai/models/threed}"

# Pinned to the last combination with prebuilt wheels for every hard dep:
# spconv-cu126 exists, torch 2.6.0+cu126 matches it, and CUDA 12.6's nvcc
# accepts gcc 13 (supplied by micromamba, not the system).
PY_VERSION="3.10"
CUDA_VERSION="12.6"
GXX_VERSION="13"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
# Pinned to the xformers release built against torch 2.6.0. Leaving this
# unpinned resolves to a current xformers, which drags torch forward with it.
XFORMERS_VERSION="0.0.29.post3"

log()  { printf '\033[96m[trellis-env]\033[0m %s\n' "$*"; }
ok()   { printf '\033[92m[trellis-env] ✔\033[0m %s\n' "$*"; }
warn() { printf '\033[93m[trellis-env] ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[91m[trellis-env] ✗\033[0m %s\n' "$*"; exit 1; }

[ -d "$TRELLIS_SRC" ] || die "TRELLIS source not found at $TRELLIS_SRC
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git $TRELLIS_SRC"

mkdir -p "$ENV_ROOT/bin" "$BUILD_DIR"

# ---------------------------------------------------------------------------
# 1. micromamba — a single static binary, used only to provision a private
#    toolchain (gcc 13 + CUDA 12.6). It is never added to the user's shell.
# ---------------------------------------------------------------------------
if [ ! -x "$MAMBA_BIN" ]; then
    log "Fetching micromamba…"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "$ENV_ROOT" bin/micromamba >/dev/null 2>&1 \
        || die "micromamba download failed"
    chmod +x "$MAMBA_BIN"
fi
ok "micromamba: $("$MAMBA_BIN" --version 2>/dev/null || echo unknown)"

export MAMBA_ROOT_PREFIX="$ENV_ROOT/mamba_root"

# ---------------------------------------------------------------------------
# 2. Private toolchain env
# ---------------------------------------------------------------------------
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    log "Creating env (python $PY_VERSION, gcc $GXX_VERSION, CUDA $CUDA_VERSION)…"
    "$MAMBA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge -c nvidia \
        "python=$PY_VERSION" \
        "gxx_linux-64=$GXX_VERSION" \
        "gcc_linux-64=$GXX_VERSION" \
        "cuda-toolkit=$CUDA_VERSION" \
        ninja cmake git \
        || die "env creation failed"
fi
ok "env: $("$ENV_PREFIX/bin/python" -V)"

PY="$ENV_PREFIX/bin/python"
PIP="$PY -m pip"

# nvcc and the private gcc must win over the system ones for every build below.
export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CC="$(ls "$ENV_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -1)"
export CXX="$(ls "$ENV_PREFIX"/bin/*-linux-gnu-g++ 2>/dev/null | head -1)"
[ -x "${CC:-}" ] || die "private gcc not found in $ENV_PREFIX/bin"
export CUDAHOSTCXX="$CXX"
# conda-forge's cuda-toolkit puts headers and libs under targets/<arch>/, but
# torch's cpp_extension only ever looks in $CUDA_HOME/include and
# $CUDA_HOME/lib64. Without these, every extension build fails on a missing
# cuda_runtime.h.
CUDA_TARGET="$ENV_PREFIX/targets/x86_64-linux"
if [ -d "$CUDA_TARGET/include" ]; then
    export CPATH="$CUDA_TARGET/include${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="$CUDA_TARGET/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LIBRARY_PATH="$CUDA_TARGET/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$CUDA_TARGET/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # nvcc resolves -I paths itself and ignores CPATH for device code.
    export NVCC_PREPEND_FLAGS="-I$CUDA_TARGET/include ${NVCC_PREPEND_FLAGS:-}"
fi
# 4090 = sm_89. Building one arch keeps the build minutes instead of hours.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
log "CC=$CC"
log "nvcc=$("$ENV_PREFIX/bin/nvcc" --version 2>/dev/null | tail -1)"

# ---------------------------------------------------------------------------
# 3. torch, then the pure-python / wheel-only deps
# ---------------------------------------------------------------------------
CURRENT_TORCH="$($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
if [ "$CURRENT_TORCH" != "$TORCH_VERSION+cu126" ]; then
    log "Installing torch $TORCH_VERSION+cu126 (found: $CURRENT_TORCH)…"
    $PIP install --quiet --force-reinstall \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "$TORCH_INDEX" || die "torch install failed"
fi
ok "torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"

# Every CUDA extension below is compiled against this exact torch ABI. If a
# later dependency quietly upgrades torch, those .so files stop loading with
# an "undefined symbol" error and torchvision's ops vanish. A pip constraints
# file makes that upgrade impossible rather than merely unlikely.
CONSTRAINTS="$ENV_ROOT/constraints.txt"
cat > "$CONSTRAINTS" <<EOF
torch==$TORCH_VERSION+cu126
torchvision==$TORCHVISION_VERSION+cu126
EOF
export PIP_CONSTRAINT="$CONSTRAINTS"

log "Installing TRELLIS python deps…"
# One package per install so a single bad resolve can't take the whole set
# down with it — pip aborts the entire transaction on any unsatisfiable spec.
for dep in "numpy<2" pillow imageio imageio-ffmpeg tqdm easydict \
           opencv-python-headless scipy ninja rembg onnxruntime trimesh \
           xatlas pyvista pymeshfix igraph safetensors transformers \
           huggingface_hub einops open3d; do
    $PIP install --quiet "$dep" || warn "dep failed: $dep"
done
# open3d is not optional despite only being used by the text-to-3D pipeline:
# trellis/pipelines/__init__.py imports that module unconditionally, so the
# image pipeline cannot be imported without it.

# utils3d must be the revision TRELLIS pins — the current PyPI package has a
# different API and postprocessing_utils will not import against it.
$PIP install --quiet \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
    || warn "utils3d pinned revision failed"

log "Installing spconv (sparse convs for the SLAT decoder)…"
$PIP install --quiet spconv-cu126 || die "spconv install failed"

# kaolin supplies exactly one symbol on the inference path —
# kaolin.utils.testing.check_tensor, used for shape assertions in FlexiCubes —
# but flexicubes.py imports it at module scope, so the mesh representation
# cannot load without it. NVIDIA publishes wheels per torch build; this URL is
# the one matching the pinned torch above. Nothing else here needs kaolin,
# which is why the torch pin and this index have to stay in step.
log "Installing kaolin (FlexiCubes shape assertions)…"
$PIP install --quiet kaolin \
    -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_cu126.html" \
    || warn "kaolin install failed — the mesh representation will not import"

# TRELLIS's *sparse* attention accepts only 'xformers' or 'flash_attn' —
# unlike its dense attention, 'sdpa' is not an option there. Install both
# attention backends BEFORE compiling any CUDA extension, so the extensions
# build against the torch that will actually be present at runtime.
$PIP install --quiet "xformers==$XFORMERS_VERSION" --index-url "$TORCH_INDEX" \
    && ok "xformers installed (sparse attention backend)" \
    || die "xformers install failed — TRELLIS sparse attention has no usable backend"

$PIP install --quiet flash-attn --no-build-isolation 2>/dev/null \
    && ok "flash-attn installed" \
    || warn "flash-attn unavailable — using xformers instead"

# Anything above may still have tried to move torch. Catch it here rather
# than at import time, where it surfaces as an inscrutable symbol error.
FINAL_TORCH="$($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
[ "$FINAL_TORCH" = "$TORCH_VERSION+cu126" ] \
    || die "torch drifted to $FINAL_TORCH — extensions would not load. Check $CONSTRAINTS"

# ---------------------------------------------------------------------------
# 4. CUDA extensions built from source
# ---------------------------------------------------------------------------
build_ext() {
    local name="$1" url="$2" subdir="${3:-}" rev="${4:-}"
    if $PY -c "import $5" 2>/dev/null; then ok "$name already built"; return 0; fi
    log "Building $name…"
    local src="$BUILD_DIR/$name"
    [ -d "$src" ] || git clone --recursive "$url" "$src" >/dev/null 2>&1 || {
        warn "$name clone failed"; return 1; }
    if [ -n "$rev" ]; then (cd "$src" && git checkout -q "$rev" 2>/dev/null); fi
    if $PIP install --no-build-isolation "$src/$subdir" 2>&1 | tail -20; then
        ok "$name built"
    else
        warn "$name FAILED to build"
        return 1
    fi
}

# nvdiffrast — rasterizes the mesh for UV texture baking. Required for GLB.
build_ext nvdiffrast https://github.com/NVlabs/nvdiffrast.git "" "" nvdiffrast

# diff_gaussian_rasterization (mip-splatting fork) — bakes the SLAT gaussian
# appearance onto the extracted mesh's texture. Required for a textured GLB.
build_ext mip-splatting https://github.com/autonomousvision/mip-splatting.git \
    "submodules/diff-gaussian-rasterization" "" diff_gaussian_rasterization

# Optional: radiance-field output only, which Phase 1 does not ship. A
# failure here does not affect the mesh path.
build_ext diffoctreerast https://github.com/JeffreyXiang/diffoctreerast.git "" "" diffoctreerast

# vox2seq is deliberately not built. It backs the 'serialized' sparse
# attention mode; every shipped TRELLIS config uses 'swin', so it never gets
# imported — and it is no longer present in the TRELLIS repo to build from.

# ---------------------------------------------------------------------------
# 5. Weights
# ---------------------------------------------------------------------------
TRELLIS_MODEL="${TRELLIS_MODEL:-microsoft/TRELLIS-image-large}"
TARGET="$WEIGHTS_DIR/$(basename "$TRELLIS_MODEL")"
if [ ! -d "$TARGET" ]; then
    log "Downloading $TRELLIS_MODEL → $TARGET…"
    $PY - "$TRELLIS_MODEL" "$TARGET" <<'PYEOF' || warn "weight download failed"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PYEOF
fi
[ -d "$TARGET" ] && ok "weights: $TARGET"

# ---------------------------------------------------------------------------
# 6. Verify the pieces the mesh path actually needs
# ---------------------------------------------------------------------------
log "Verifying…"
TRELLIS_SRC="$TRELLIS_SRC" $PY - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["TRELLIS_SRC"])
os.environ.setdefault("SPCONV_ALGO", "native")
try:
    import flash_attn  # noqa: F401
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
except ImportError:
    os.environ.setdefault("ATTN_BACKEND", "xformers")

required = ["torch", "spconv", "nvdiffrast.torch", "diff_gaussian_rasterization",
            "utils3d", "xatlas", "pymeshfix", "pyvista", "igraph", "trimesh",
            "open3d", "xformers", "kaolin"]
optional = ["diffoctreerast", "flash_attn"]

missing = []
for m in required:
    try:
        __import__(m); print(f"  ok       {m}")
    except Exception as e:
        print(f"  MISSING  {m}: {type(e).__name__}: {e}"); missing.append(m)
for m in optional:
    try:
        __import__(m); print(f"  ok       {m} (optional)")
    except Exception:
        print(f"  absent   {m} (optional — mesh path does not need it)")

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    print("  ok       trellis.pipelines")
except Exception as e:
    print(f"  MISSING  trellis.pipelines: {type(e).__name__}: {e}")
    missing.append("trellis")

print()
if missing:
    print("INCOMPLETE — missing:", ", ".join(missing))
    sys.exit(1)
print("TRELLIS environment is READY")
PYEOF

# FUK treats this marker as the signal that TRELLIS is usable — an
# interpreter on disk proves nothing, since a failed build leaves one behind.
if [ $? -eq 0 ]; then
    date -Iseconds > "$ENV_ROOT/READY"
    ok "Done. FUK will detect the environment at $ENV_PREFIX"
else
    rm -f "$ENV_ROOT/READY"
    warn "Environment is incomplete — the TRELLIS model will report as unavailable in the UI."
    warn "The VGGT multi-view path is unaffected and runs in the main venv."
    exit 1
fi
