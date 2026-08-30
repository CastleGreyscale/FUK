#!/bin/bash
set -e

echo "============================================"
echo "FUK: Framework for Unified Kreation"
echo "Installation Script"
echo "============================================"
echo ""

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠  Warning: nvidia-smi not found. GPU acceleration may not work."
fi

# ── Virtual Environment (Python 3.10 required) ───────────────────────────────
echo ""
echo "[0/7] Setting up virtual environment..."

VENV_DIR="venv"
PYTHON_BIN=""

# Prefer explicit python3.10; fall back with a hard error
for candidate in python3.10 python3; do
    if command -v "$candidate" &> /dev/null; then
        VER=$("$candidate" -c "import sys; print('%d.%d' % sys.version_info[:2])")
        if [ "$VER" = "3.10" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "  ✗ Error: Python 3.10 is required but was not found."
    echo "    Install it with:  sudo apt install python3.10 python3.10-venv"
    echo "    Or via pyenv:     pyenv install 3.10 && pyenv local 3.10"
    exit 1
fi

echo "  ✓ Found Python 3.10: $(which $PYTHON_BIN)"

if [ ! -d "$VENV_DIR" ]; then
    echo "  → Creating virtual environment at ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created"
else
    # Verify the existing venv is actually 3.10
    VENV_VER=$("$VENV_DIR/bin/python" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || echo "unknown")
    if [ "$VENV_VER" != "3.10" ]; then
        echo "  ⚠  Existing venv is Python $VENV_VER, not 3.10 — recreating..."
        rm -rf "$VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        echo "  ✓ Virtual environment recreated with Python 3.10"
    else
        echo "  ✓ Virtual environment already exists (Python 3.10)"
    fi
fi

echo "  → Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "  ✓ Using Python: $(python --version) at $(which python)"

# Upgrade pip and install wheel up front — prevents legacy setup.py install
# warnings for packages like antlr4-python3-runtime, basicsr, moviepy
echo "  → Upgrading pip + installing build tools..."
pip install --quiet --upgrade pip wheel setuptools
echo "  ✓ pip, wheel, setuptools up to date"

# ── PyTorch (CUDA) ────────────────────────────────────────────────────────────
# PyPI only hosts CPU-only torch wheels. CUDA builds must come from PyTorch's
# own index. This must run before `pip install -e .` so the CUDA wheels are
# already present and pip won't replace them with CPU-only versions.
echo ""
echo "[1/7] Installing PyTorch (CUDA)..."

# Detect CUDA version from nvcc to pick the right wheel index.
# Supports: cu118, cu124, cu126, cu128 (PyTorch 2.7 matrix)
CUDA_INDEX="cu128"  # safe default — most recent stable
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | tr -d '.')
    case "$CUDA_VER" in
        118|11*) CUDA_INDEX="cu118" ;;
        124)     CUDA_INDEX="cu124" ;;
        126)     CUDA_INDEX="cu126" ;;
        128|12*) CUDA_INDEX="cu128" ;;
        *)       CUDA_INDEX="cu128" ;;
    esac
    echo "  ✓ Detected CUDA $CUDA_VER → $CUDA_INDEX"
else
    echo "  ⚠  nvcc not found, defaulting to $CUDA_INDEX"
fi

echo "  → https://download.pytorch.org/whl/$CUDA_INDEX"
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
echo "  ✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  ✓ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Core FUK ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/7] Installing core FUK dependencies..."
# --no-deps first so pip resolves our already-installed CUDA torch
# rather than re-pulling CPU-only wheels from PyPI
pip install -e . --no-deps
pip install -e .

# ── Vendor directory ─────────────────────────────────────────────────────────
echo ""
echo "[3/7] Setting up vendor directory..."
mkdir -p fuk/vendor

# ── Vendor repos ─────────────────────────────────────────────────────────────
echo ""
echo "[4/7] Cloning vendor dependencies..."

# Every vendor repo is pinned to an exact commit. These used to be bare clones
# of the default branch, which meant a fresh install silently got whatever
# upstream HEAD happened to be that day — that is how this tree ended up running
# a DiffSynth seven months newer than the one its patches were written against.
# Bump these deliberately, and re-run the regression matrix when you do.
DIFFSYNTH_SHA=102fe9980b9375ecb6436d360297a00327472535   # 2.1.5, 2026-08-28
DEPTH_ANYTHING_3_SHA=2c21ea849ceec7b469a3e62ea0c0e270afc3281a
SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4
DSINE_SHA=ef0c2afa32b4dd19cb8ca4567c652802cd92591c
VGGT_SHA=a288dd0f14786c93483e45524328726ab7b1b4ce
TRELLIS_SHA=442aa1e1afb9014e80681d3bf604e8d728a86ee7
SEEDVR2_SHA=4490bd1f482e026674543386bb2a4d176da245b9

# clone_pinned <dir-name> <url> <sha> [extra git-clone args...]
# Clones at the pinned commit. If the directory already exists it is left alone,
# but a mismatched HEAD is reported loudly rather than ignored — a stale vendor
# tree produces bugs that look like FUK bugs.
clone_pinned() {
    local name="$1" url="$2" sha="$3"
    shift 3
    local dir="fuk/vendor/$name"

    if [ ! -d "$dir" ]; then
        echo "  → Cloning $name @ ${sha:0:8}..."
        git clone "$@" "$url" "$dir"
        git -C "$dir" checkout --quiet --detach "$sha"
        return 0
    fi

    local current
    current="$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ "$current" = "$sha" ]; then
        echo "  ✓ $name already at pinned ${sha:0:8}"
    else
        echo "  ⚠ $name is at ${current:0:8} but the pin is ${sha:0:8}"
        echo "    Leaving it untouched. To adopt the pin (discards local edits):"
        echo "      git -C $dir fetch origin $sha && git -C $dir checkout --detach $sha"
    fi
    return 1
}

# DiffSynth-Studio — patched; see fuk/vendor/patches/README.md for what and why.
if clone_pinned DiffSynth-Studio https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH_SHA"; then
    echo "  → Applying FUK patches to DiffSynth-Studio..."
    for patch in "$(pwd)"/fuk/vendor/patches/0*.patch; do
        [ -e "$patch" ] || continue
        if git -C fuk/vendor/DiffSynth-Studio apply "$patch"; then
            echo "    ✓ $(basename "$patch")"
        else
            echo "    ✗ FAILED to apply $(basename "$patch")"
            echo "      DiffSynth is unpatched. Expect VACE OOM at 720p and ring"
            echo "      artifacts on reference-image renders. See fuk/vendor/patches/README.md"
            exit 1
        fi
    done
fi

# Depth Anything V3
clone_pinned Depth-Anything-3 https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DEPTH_ANYTHING_3_SHA" || true

# SAM2
clone_pinned segment-anything-2 https://github.com/facebookresearch/segment-anything-2 "$SAM2_SHA" || true
# Checkpoint presence is checked separately from the clone so an interrupted
# first run can be resumed without deleting the repo.
if ! ls fuk/vendor/segment-anything-2/checkpoints/*.pt >/dev/null 2>&1; then
    echo "  → Downloading SAM2 checkpoints..."
    (cd fuk/vendor/segment-anything-2/checkpoints && bash download_ckpts.sh)
else
    echo "  ✓ SAM2 checkpoints already present"
fi

# DSINE (normals)
clone_pinned DSINE https://github.com/baegwangbin/DSINE.git "$DSINE_SHA" || true

# VGGT (multi-view 3D reconstruction)
# Source-only: threed/vggt_backend.py puts this directory on sys.path rather
# than pip-installing it, and the weights come from HuggingFace at first use.
# Do NOT install its requirements.txt — it pins torch==2.3.1 and would drag
# the main venv's CUDA torch backwards, breaking Qwen and Wan generation.
clone_pinned VGGT https://github.com/facebookresearch/vggt.git "$VGGT_SHA" || true

# TRELLIS (single-image 3D reconstruction) — source only.
# The runtime lives in an isolated environment built separately; see the
# 3D reconstruction section below for why it is not built here.
clone_pinned TRELLIS https://github.com/microsoft/TRELLIS.git "$TRELLIS_SHA" --recurse-submodules || true

# SeedVR2 (temporal video restoration / upscaling) — source only, runs in this
# venv like VGGT does. It reimplements the SeedVR2 architecture with VAE tiling,
# block swapping and quantized weights, which is what makes it fit a 24GB card;
# ByteDance's own inference path does not and was removed from FUK for that
# reason. Do NOT install its requirements.txt — it lists an unpinned torch and
# would fight the venv's cu130 build. Its only two extra deps (gguf,
# rotary_embedding_torch) are in pyproject.toml instead. Weights download on
# first use, SHA256-validated, into {models_root}/seedvr2.
clone_pinned SeedVR2 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEEDVR2_SHA" || true

# ── Vendor packages ───────────────────────────────────────────────────────────
echo ""
echo "[5/7] Installing vendor packages..."

# basicsr uses setup.py version introspection that breaks in isolated build
# environments (KeyError: '__version__'). --no-build-isolation lets it see
# the already-installed setuptools in our venv and avoids the error.
# Install it explicitly before DiffSynth-Studio pulls it in as a dep.
echo "  → Pre-installing basicsr (no-build-isolation workaround)..."
pip install basicsr --no-build-isolation

pip install -e ./fuk/vendor/DiffSynth-Studio
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2

# ── 3D reconstruction ─────────────────────────────────────────────────────────
# VGGT needs nothing installed: it is plain PyTorch and runs in this venv off
# the vendored source. TRELLIS is the opposite — it needs torch 2.6/cu126 plus
# spconv, nvdiffrast and diff_gaussian_rasterization compiled from source,
# which cannot coexist with the torch this venv runs for Qwen and Wan. It
# therefore gets a private micromamba environment under fuk/vendor/TRELLIS_ENV
# and is driven over a subprocess.
#
# That build is ~18GB and takes a long while, so it is opt-in rather than part
# of a default install. Everything else in FUK works without it; only the
# single-image 3D path goes dark.
echo ""
echo "  → 3D reconstruction backends..."

if python -c "import sys; sys.path.insert(0, 'fuk/vendor/VGGT'); import vggt" 2>/dev/null; then
    echo "    ✓ VGGT ready (runs in this venv, no install needed)"
else
    echo "    ⚠  VGGT source missing or unimportable — multi-view 3D will be unavailable"
fi

# ── Video restoration ─────────────────────────────────────────────────────────
echo ""
echo "  → Video restoration backend..."

if python -c "import sys; sys.path.insert(0, 'fuk'); from core import seedvr2_backend as s; import sys as _s; _s.exit(0 if s.availability()['available'] else 1)" 2>/dev/null; then
    echo "    ✓ SeedVR2 ready (runs in this venv, weights fetch on first use)"
else
    echo "    ⚠  SeedVR2 unavailable — video upscaling falls back to per-frame Real-ESRGAN"
fi

if [ -f "fuk/vendor/TRELLIS_ENV/READY" ]; then
    echo "    ✓ TRELLIS isolated environment already built"
elif [ "${FUK_INSTALL_TRELLIS_ENV:-0}" = "1" ]; then
    echo "    → Building TRELLIS isolated environment (~18GB, this takes a while)..."
    # Never let a TRELLIS build failure abort the rest of the install — the
    # script exits non-zero when the environment comes out incomplete, and
    # this whole file runs under `set -e`.
    if bash fuk/core/threed/install_trellis_env.sh; then
        echo "    ✓ TRELLIS environment built"
    else
        echo "    ⚠  TRELLIS environment incomplete — single-image 3D will show as"
        echo "       unavailable in the UI. Everything else is unaffected."
    fi
else
    echo "    ○ TRELLIS isolated environment not built (optional, ~18GB)"
    echo "      Build it with:  bash fuk/core/threed/install_trellis_env.sh"
    echo "      Or re-run this script with:  FUK_INSTALL_TRELLIS_ENV=1 ./setup.sh"
fi

# ── Configuration files ───────────────────────────────────────────────────────
echo ""
echo "[6/7] Setting up configuration files..."
mkdir -p fuk/config fuk/config/tools

# Initialize .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "  → Creating .gitignore..."
    cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# FUK specific
cache/
renders/
outputs/

# Configuration files (personal settings - gitignored, edit freely)
fuk/config/*.json
fuk/config/tools/*.json
!fuk/config/*.json.template
!fuk/config/tools/*.json.template
EOL
else
    echo "  ✓ .gitignore already exists"
fi

# Add config patterns to .gitignore if not already present
if ! grep -q "fuk/config/\*.json" .gitignore 2>/dev/null; then
    echo "  → Adding config patterns to .gitignore..."
    cat >> .gitignore << 'EOL'

# Configuration files (personal settings - gitignored, edit freely)
fuk/config/*.json
fuk/config/tools/*.json
!fuk/config/*.json.template
!fuk/config/tools/*.json.template
EOL
fi

# ── Templates ─────────────────────────────────────────────────────────────────
# Templates are ALWAYS written — they are the canonical reference and must
# stay in sync with the codebase. Personal fuk/config/*.json files are never
# touched after first creation.
echo ""
echo "  → Writing template configuration files..."

cat > fuk/config/models.json.template << 'EOL'
{
  "_comment": "FUK Model Registry — each entry defines everything needed to construct a DiffSynth pipeline. Add new models by adding entries here. Visit https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Qwen-Image.md and https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Wan.md for full model list. Note: not all options are setup. Below is the current list of working models. Before running download_models.py remove unwanted models. Roughly 15-30GB per model",

  "qwen_image": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "category": "image",
    "description": "Qwen text-to-image",
    "aliases": ["qwen", "t2i"],
    "supports": ["negative_prompt"],
    "parameter_map": {},
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"}
  },

  "qwen_image_2512": {
    "model_id": "Qwen/Qwen-Image-2512",
    "pipeline": "qwen",
    "category": "image",
    "description": "Qwen 2512 text-to-image",
    "aliases": ["qwen-2512"],
    "supports": ["negative_prompt"],
    "parameter_map": {},
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors", "model_id": "Qwen/Qwen-Image"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors", "model_id": "Qwen/Qwen-Image"}
    ],
    "tokenizer": {"pattern": "tokenizer/", "model_id": "Qwen/Qwen-Image"}
  },

  "qwen_image_edit_2511": {
    "model_id": "Qwen/Qwen-Image-Edit-2511",
    "pipeline": "qwen",
    "category": "image",
    "description": "Qwen Edit 2511 image-editing",
    "aliases": ["qwen-edit", "edit-2511", "qwen_edit"],
    "supports": ["edit_image", "negative_prompt"],
    "parameter_map": {
      "edit_targets": "edit_image"
    },
    "pipeline_kwargs": {
      "zero_cond_t": true
    },
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors", "model_id": "Qwen/Qwen-Image"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors", "model_id": "Qwen/Qwen-Image"}
    ],
    "tokenizer": {"pattern": "tokenizer/", "model_id": "Qwen/Qwen-Image"},
    "processor": {"pattern": "processor/", "model_id": "Qwen/Qwen-Image-Edit"}
  },

  "qwen_image_control_union": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "category": "image",
    "description": "Qwen Image Control",
    "aliases": ["control-union", "qwen-control"],
    "supports": ["context_image", "negative_prompt"],
    "parameter_map": {
      "control_input": "context_image"
    },
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"},
    "lora": {
      "model_id": "DiffSynth-Studio/Qwen-Image-In-Context-Control-Union",
      "pattern": "model.safetensors",
      "target": "dit"
    }
  },

  "qwen_image_control_union_2512": {
    "model_id": "Qwen/Qwen-Image-2512",
    "pipeline": "qwen",
    "category": "image",
    "description": "Qwen Image 2512 Control",
    "aliases": ["control-union-2512", "qwen-control-2512"],
    "supports": ["context_image", "negative_prompt"],
    "parameter_map": {
      "control_input": "context_image"
    },
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors", "model_id": "Qwen/Qwen-Image"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors", "model_id": "Qwen/Qwen-Image"}
    ],
    "tokenizer": {"pattern": "tokenizer/", "model_id": "Qwen/Qwen-Image"},
    "lora": {
      "model_id": "DiffSynth-Studio/Qwen-Image-In-Context-Control-Union",
      "pattern": "model.safetensors",
      "target": "dit"
    }
  },


  "qwen_eligen": {
    "model_id": "Qwen/Qwen-Image",
    "pipeline": "qwen",
    "category": "image",
    "description": "Zone composition control",
    "aliases": ["eligen", "eligen-v2"],
    "supports": ["eligen", "negative_prompt"],
    "parameter_map": {},
    "components": [
      {"pattern": "transformer/diffusion_pytorch_model*.safetensors"},
      {"pattern": "text_encoder/model*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"},
    "lora": {
      "model_id": "DiffSynth-Studio/Qwen-Image-EliGen-V2",
      "pattern": "model.safetensors",
      "target": "dit",
      "alpha": 0.55
    }
  },

  "wan_i2v_a14b": {
    "model_id": "Wan-AI/Wan2.2-I2V-A14B",
    "pipeline": "wan",
    "category": "video",
    "description": "Wan 2.2 Image-to-video",
    "aliases": ["i2v-A14B", "i2v-2.2"],
    "supports": ["input_image", "negative_prompt", "tiled"],
    "parameter_map": {
      "reference_image": "input_image"
    },
    "components": [
      {"pattern": "high_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "low_noise_model/diffusion_pytorch_model*.safetensors"},
      {"model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors", "pattern": "models_t5_umt5-xxl-enc-bf16.safetensors"},
      {"model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors", "pattern": "Wan2.1_VAE.safetensors"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "wan_vace_a14b": {
    "model_id": "alibaba-pai/Wan2.2-VACE-Fun-A14B",
    "pipeline": "wan",
    "category": "video",
    "description": "Wan 2.2 VACE-controlled video",
    "aliases": ["vace", "vace-a14b"],
    "supports": ["vace_video", "vace_reference_image", "vace_scale", "negative_prompt", "tiled"],
    "parameter_map": {
      "control_input": "vace_video",
      "reference_image": "vace_reference_image"
    },
    "components": [
      {"pattern": "high_noise_model/diffusion_pytorch_model*.safetensors"},
      {"pattern": "low_noise_model/diffusion_pytorch_model*.safetensors"},
      {"model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors", "pattern": "models_t5_umt5-xxl-enc-bf16.safetensors"},
      {"model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors", "pattern": "Wan2.1_VAE.safetensors"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "wan_inp_a14b": {
    "model_id": "alibaba-pai/Wan2.2-Fun-A14B-InP",
    "pipeline": "wan",
    "category": "video",
    "description": "Wan 2.2 FirstFrame - LastFrame",
    "aliases": ["inp", "inp-a14b", "inpaint-video"],
    "supports": ["input_image", "end_image", "negative_prompt", "tiled"],
    "parameter_map": {
        "reference_image": "input_image",
        "end_image": "end_image"
    },
    "components": [
        {"pattern": "high_noise_model/diffusion_pytorch_model*.safetensors"},
        {"pattern": "low_noise_model/diffusion_pytorch_model*.safetensors"},
        {"model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors", "pattern": "models_t5_umt5-xxl-enc-bf16.safetensors"},
        {"model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors", "pattern": "Wan2.1_VAE.safetensors"}
    ],
    "tokenizer": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
        "tiled": true
    }
  },

  "wan_animate_a14b": {
    "model_id": "Wan-AI/Wan2.2-Animate-14B",
    "pipeline": "wan",
    "category": "video",
    "description": "Wan 2.2 Animate (pose + face control)",
    "aliases": ["animate", "animate-a14b"],
    "supports": ["input_image", "animate_pose_video", "animate_face_video", "negative_prompt", "tiled"],
    "parameter_map": {
      "reference_image": "input_image",
      "control_input": "animate_pose_video"
    },
    "components": [
      {"pattern": "diffusion_pytorch_model*.safetensors"},
      {"pattern": "models_t5_umt5-xxl-enc-bf16.pth"},
      {"pattern": "Wan2.1_VAE.pth"},
      {"pattern": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "wan_animate_replace_a14b": {
    "model_id": "Wan-AI/Wan2.2-Animate-14B",
    "pipeline": "wan",
    "category": "video",
    "description": "Wan 2.2 Animate Replace (pose + face + inpaint + mask)",
    "aliases": ["animate-replace", "animate-replace-a14b"],
    "supports": ["input_image", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video", "negative_prompt", "tiled"],
    "parameter_map": {
      "reference_image": "input_image",
      "control_input": "animate_pose_video"
    },
    "components": [
      {"pattern": "diffusion_pytorch_model*.safetensors"},
      {"pattern": "models_t5_umt5-xxl-enc-bf16.pth"},
      {"pattern": "Wan2.1_VAE.pth"},
      {"pattern": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}
    ],
    "tokenizer": {
      "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
      "pattern": "google/umt5-xxl/"
    },
    "lora": {
      "model_id": "Wan-AI/Wan2.2-Animate-14B",
      "pattern": "relighting_lora.ckpt",
      "target": "dit"
    },
    "pipeline_kwargs": {
      "tiled": true
    }
  },

  "flux2_dev": {
    "model_id": "black-forest-labs/FLUX.2-dev",
    "pipeline": "flux2",
    "category": "image",
    "description": "FLUX.2-dev multi-image editing",
    "aliases": ["flux2", "flux2-dev"],
    "supports": ["edit_image", "negative_prompt"],
    "parameter_map": {
      "edit_targets": "edit_image"
    },
    "components": [
      {"pattern": "text_encoder/*.safetensors"},
      {"pattern": "transformer/*.safetensors"},
      {"pattern": "vae/diffusion_pytorch_model.safetensors"}
    ],
    "tokenizer": {"pattern": "tokenizer/"}
  },

  "_threed_comment": "3D reconstruction models. Not DiffSynth pipelines — they load from their own vendored repos (see docs/3D_RECONSTRUCTION_SYSTEM.md). Entries stay flat like every other model so resolve_model_type/aliases keep working; the 'threed' pipeline value is what groups them.",

  "trellis": {
    "model_id": "microsoft/TRELLIS-image-large",
    "pipeline": "threed",
    "category": "threed",
    "type": "single_image",
    "name": "TRELLIS",
    "description": "Single-image 3D — Structured 3D Latents, seconds to a mesh",
    "aliases": ["trellis-image-large", "single-image-3d"],
    "path": "/path/to/your/models/threed/TRELLIS-image-large",
    "supports": ["glb", "ply", "obj"],
    "vram_gb_estimate": 16,
    "isolated_env": "fuk/vendor/TRELLIS_ENV/env",
    "install": "bash fuk/core/threed/install_trellis_env.sh"
  },

  "vggt": {
    "model_id": "facebook/VGGT-1B",
    "pipeline": "threed",
    "category": "threed",
    "type": "multi_view",
    "name": "VGGT",
    "description": "Multi-view reconstruction — feed-forward transformer, 1 to hundreds of images",
    "aliases": ["vggt-1b", "multi-view-3d"],
    "path": "/path/to/your/models/threed/VGGT-1B",
    "supports": ["glb", "ply", "obj"],
    "vram_gb_estimate": 14,
    "max_input_dim": 518
  },

  "_deferred": {
    "_comment": "Complex models deferred for post-launch",
    "seedvr2": "Not a DiffSynth pipeline. Video restoration runs through fuk/core/seedvr2_backend.py against the vendored engine in fuk/vendor/SeedVR2; its weights are fetched by download_models.sh into {models_root}/seedvr2. Same arrangement as trellis and vggt.",
    "wan_t2v_1.3b": "Excluding T2V - not professional use case",
    "wan_t2v_14b": "Excluding T2V - not professional use case",
    "wan_s2v_14b": "Requires: s2v_pose_video + motion_video + audio pipeline",
    "wan_control_camera": "Requires: camera_control_direction parsing + special UI",
    "qwen_blockwise_controlnet": "Requires: preprocessor integration for canny/depth/inpaint"
  }
}
EOL
echo "    ✓ fuk/config/models.json.template"

cat > fuk/config/defaults.json.template << 'EOL'
{
  "models_root": "/path/to/your/models",

  "aspect_ratios": [
    { "label": "1:1 (Square)",        "value": "1:1",    "ratio": 1.0    },
    { "label": "1.33:1 (Fullscreen)", "value": "1.33:1", "ratio": 1.3333 },
    { "label": "1.78:1 (Widescreen)", "value": "1.78:1", "ratio": 1.7778 },
    { "label": "1.85:1 (Academy)",    "value": "1.85:1", "ratio": 1.85   },
    { "label": "2.39:1 (Anamorphic)", "value": "2.39:1", "ratio": 2.39   },
    { "label": "2.75:1 (Panavision)", "value": "2.75:1", "ratio": 2.76   }
  ],

  "image": {
    "prompt": "",
    "negative_prompt": "low resolution, low quality, limb deformities, finger deformities, over-saturated image, wax figure appearance, lack of facial detail, overly smooth, AI-generated look. Chaotic composition. Blurry, distorted text",
    "model": "qwen_image",
    "width": 1280,
    "aspectRatio": "2.39:1",
    "steps": 20,
    "stepsMode": "preset",
    "guidance_scale": 4,
    "cfg_scale": 5.0,
    "denoising_strength": 1,
    "lora": null,
    "lora_multiplier": 1.0,
    "seed": null,
    "seedMode": "random",
    "lastUsedSeed": null,
    "output_format": "png",
    "edit_strength": 1,
    "detail_bias": 0.9,
    "exponential_shift_mu": null,
    "save_latent": true,
    "control_image_paths": []
  },

  "video": {
    "model": "wan_i2v_a14b",
    "prompt": "",
    "negative_prompt": "bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
    "task": "wan_i2v_a14b",
    "video_length": 41,
    "scale_factor": 1.0,
    "steps": 20,
    "stepsMode": "preset",
    "guidance_scale": 5.0,
    "cfg_scale": 6.0,
    "denoising_strength": 1.0,
    "sigma_shift": 5.0,
    "switch_dit_boundary": 0.875,
    "sliding_window_size": null,
    "sliding_window_stride": null,
    "tea_cache_l1_thresh": null,
    "tea_cache_model_id": null,
    "lora": null,
    "lora_multiplier": 1.0,
    "seed": null,
    "save_latent": true,
    "seedMode": "random",
    "lastUsedSeed": null,
    "image_path": null,
    "end_image_path": null,
    "width": null,
    "height": null,
    "source_width": null,
    "source_height": null
  },

  "flux2": {
    "width": 1024,
    "steps": 30,
    "cfg_scale": 1.0,
    "embedded_guidance": 4.0,
    "denoising_strength": 1.0,
    "negative_prompt": ""
  },

  "export": {
    "prefer_latent": true,
    "exr_compression": "ZIP",
    "exr_precision": "float32"
  },

  "preprocess": {
    "depth": {
      "model": "da3_mono_large",
      "invert": false,
      "normalize": true,
      "colormap": "greyscale"
    },
    "normals": {
      "method": "depth_gradient",
      "depth_model": "da3_mono_large",
      "blur_radius": 3,
      "normalize": true,
      "num_iter": 5,
      "fov_deg": 60.0
    },
    "canny": {
      "low_threshold": 1,
      "high_threshold": 20,
      "blur_kernel": 3,
      "invert": false
    },
    "openpose": {
      "include_hands": true,
      "include_face": true
    },
    "crypto": {
      "model": "sam2",
      "mode": "auto",
      "points_per_side": 32,
      "pred_iou_thresh": 0.86,
      "stability_score_thresh": 0.92
    }
  },

  "_postprocess_comment": "Seeds tabs.postprocess for new projects. Keys must match PostprocessTab.jsx DEFAULT_SETTINGS exactly — the tab spreads this over its own defaults, so a key spelled differently here is silently ignored rather than reported.",
  "postprocess": {
    "upscaleMethod": "realesrgan",
    "upscaleFactor": 4,

    "videoUpscaleMethod": "seedvr2",
    "videoUpscaleFactor": 2,

    "seedvr2Variant": "seedvr2_7b_fp8",
    "seedvr2FrameWindow": 13,
    "seedvr2ResolutionCap": 1920,
    "seedvr2MemoryMode": "auto",
    "seedvr2TemporalOverlap": -1,
    "seedvr2MotionProtection": 0.7,

    "interpolationMethod": "film",
    "targetFramerate": 24,
    "sourceFramerate": 16,

    "videoOutputMode": "mp4"
  },

  "project": {
    "shot_label": "shot",
    "versioning": "DATE",
    "autosave_interval": 30,
    "cache_cleanup_days": 7
  },

  "tools": {
    "blender": "/usr/bin/blender",
    "blender_template": "/path/to/your/template.blend"
  }
}
EOL
echo "    ✓ fuk/config/defaults.json.template"

cat > fuk/config/defaults_loras.json.template << 'EOL'
{
  "_comment": "defined_loras_path is the base directory for the curated 'loras' entries below — their 'path' is resolved relative to it. scanned_loras_path is swept for every .safetensors found and needs no entries here. 'model' accepts a single model key or a list of keys (a LoRA trained on Qwen 2512 usually applies to both the base and control-union 2512 entries in models.json).",

  "defined_loras_path": "/path/to/your/models/loras",
  "scanned_loras_path": "/path/to/your/models/loras",

  "loras": [
    {
      "name": "example_lora",
      "path": "example.safetensors",
      "model": ["qwen_image", "qwen_image_control_union", "qwen_eligen"],
      "default_strength": 1.0,
      "trigger_word": "",
      "inject_text": ""
    },
    {
      "name": "example_lora_2512",
      "path": "example_2512.safetensors",
      "model": ["qwen_image_2512", "qwen_image_control_union_2512"],
      "default_strength": 0.7,
      "trigger_word": "example_style",
      "inject_text": "A cinematic still from an example_style movie"
    }
  ]
}
EOL
echo "    ✓ fuk/config/defaults_loras.json.template"

cat > fuk/config/defaults_vram.json.template << 'EOL'
{
  "vram": {
    "preset": "low",

    "_comment_cache": "pipeline_cache_slots — how many loaded pipelines stay resident before the oldest is evicted. 2 lets you flip between (say) an image and a video model without a reload; drop to 1 if RAM is tight. vae_encode_cache reuses VAE encodings of unchanged input images across generations.",
    "pipeline_cache_slots": 2,
    "vae_encode_cache": true,

    "_buffer_gb_comment": "Headroom left free after resident weights. It must cover the largest single ACTIVATION the model allocates, not just slack \u2014 Wan 2.2's rope_apply asks for one contiguous 4.16GB tensor at 720p, which is why 2GB OOM'd at the first step. Activations are bf16 in every preset regardless of offload dtype, so this value does not shrink for the fp8 or disk presets.",

    "presets": {
      "none": {
        "label": "None — Full VRAM",
        "description": "No offloading. Fastest but needs 48GB+ for 14B models.",
        "config": null,
        "buffer_gb": 2
      },
      "low": {
        "label": "Low — CPU Offload (bf16)",
        "description": "Offload to RAM in bf16. Best for 16-24GB VRAM with 64GB+ RAM.",
        "config": {
          "offload_dtype": "bfloat16",
          "offload_device": "cpu",
          "onload_dtype": "bfloat16",
          "onload_device": "cpu",
          "preparing_dtype": "bfloat16",
          "preparing_device": "cuda",
          "computation_dtype": "bfloat16",
          "computation_device": "cuda"
        },
        "buffer_gb": 6
      },
      "medium": {
        "label": "Medium — CPU Offload (fp8)",
        "description": "Offload to RAM in fp8. Halves RAM usage, minor quality risk.",
        "config": {
          "offload_dtype": "float8_e4m3fn",
          "offload_device": "cpu",
          "onload_dtype": "float8_e4m3fn",
          "onload_device": "cpu",
          "preparing_dtype": "float8_e4m3fn",
          "preparing_device": "cuda",
          "computation_dtype": "bfloat16",
          "computation_device": "cuda"
        },
        "buffer_gb": 6
      },
      "high": {
        "label": "High — Disk Offload",
        "description": "Offload to disk. Runs on 8GB VRAM but significantly slower.",
        "config": {
          "offload_dtype": "disk",
          "offload_device": "disk",
          "onload_dtype": "bfloat16",
          "onload_device": "cpu",
          "preparing_dtype": "bfloat16",
          "preparing_device": "cuda",
          "computation_dtype": "bfloat16",
          "computation_device": "cuda"
        },
        "buffer_gb": 6
      }
    }
  }
}
EOL
echo "    ✓ fuk/config/defaults_vram.json.template"

cat > fuk/config/defaults_spec_tool.json.template << 'EOL'
{
  "spec_tool": {
    "ratio_idx":        6,
    "resolution_width": 1920,
    "guides": {
      "thirds":     true,
      "golden":     false,
      "vp":         false,
      "actionSafe": false,
      "titleSafe":  false,
      "center":     false
    },
    "guide_color":    "#ffffff",
    "bg_color":       "#1a1a1a",
    "bg_alpha":       0,
    "burn_in":        true,
    "blend_fps":      24,
    "blend_filename": "template"
  }
}
EOL
echo "    ✓ fuk/config/defaults_spec_tool.json.template"

cat > fuk/config/defaults_dataset.json.template << 'EOL'
{
  "lora_dataset": {
    "denoising_strength": 1.0,
    "seed": 509327136,
    "seed_strategy": "fixed",
    "cfg_scale": 5,
    "steps": 28,

    "variation_prompts": {
      "character": {
        "base": "",

        "angles": {

          "front_full_body":          "On a plain neutral grey background with flat studio lighting, reframe to a full-body front view, feet to crown fully visible with a small margin at top and bottom.",
          "front_seven_eighths":      "On a plain neutral grey background with flat studio lighting, reframe to a front view cropped from mid-thigh to just above the crown.",
          "front_medium":             "On a plain neutral grey background with flat studio lighting, reframe to a front medium shot cropped from the waist to just above the crown.",
          "front_medium_tight":       "On a plain neutral grey background with flat studio lighting, reframe to a front shot cropped from just below the chest to just above the crown.",
          "front_bust":               "On a plain neutral grey background with flat studio lighting, reframe to a front bust shot cropped from the mid-chest to just above the crown.",
          "front_shoulder":           "On a plain neutral grey background with flat studio lighting, reframe to a front shot cropped from the tops of the shoulders to just above the crown.",
          "front_closeup":            "On a plain neutral grey background with flat studio lighting, reframe to a front close-up cropped from the collarbone to just above the crown, face prominent.",
          "front_face":               "On a plain neutral grey background with flat studio lighting, reframe to a front shot cropped from the chin to just above the crown, full face filling the frame.",
          "front_extreme_closeup":    "On a plain neutral grey background with flat studio lighting, reframe to an extreme close-up cropped from the chin to the eyebrows, eyes and nose centered.",
          "front_eyes":               "On a plain neutral grey background with flat studio lighting, reframe to an extreme close-up of just the eye region, cropped from the upper lip to the hairline, both eyes fully visible.",

          "left_profile_full":        "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly left, full body visible feet to crown.",
          "left_profile_medium":      "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly left, cropped from the waist to just above the crown.",
          "left_profile_bust":        "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly left, cropped from the chest to just above the crown.",
          "left_profile_face":        "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly left in a strict profile, cropped from the chin to just above the crown, full side of the face visible.",
          "left_profile_closeup":     "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly left, tight close-up showing only the side profile of the face from chin to crown.",

          "right_profile_full":       "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly right, full body visible feet to crown.",
          "right_profile_medium":     "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly right, cropped from the waist to just above the crown.",
          "right_profile_bust":       "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly right, cropped from the chest to just above the crown.",
          "right_profile_face":       "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly right in a strict profile, cropped from the chin to just above the crown, full side of the face visible.",
          "right_profile_closeup":    "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly right, tight close-up showing only the side profile of the face from chin to crown.",

          "three_quarter_left_full":  "On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the left, full body visible feet to crown.",
          "three_quarter_left_medium":"On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the left, cropped from the waist to just above the crown.",
          "three_quarter_left_bust":  "On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the left, cropped from the chest to just above the crown.",
          "three_quarter_left_face":  "On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the left, cropped from the chin to just above the crown.",

          "three_quarter_right_full": "On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the right, full body visible feet to crown.",
          "three_quarter_right_medium":"On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the right, cropped from the waist to just above the crown.",
          "three_quarter_right_bust": "On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the right, cropped from the chest to just above the crown.",
          "three_quarter_right_face": "On a plain neutral grey background with flat studio lighting, turn the subject 45 degrees to the right, cropped from the chin to just above the crown.",

          "slight_left":              "On a plain neutral grey background with flat studio lighting, rotate the subject very slightly to the left, approximately 15 degrees off front-facing, framed from the chest up.",
          "slight_right":             "On a plain neutral grey background with flat studio lighting, rotate the subject very slightly to the right, approximately 15 degrees off front-facing, framed from the chest up.",

          "back_full":                "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly away from camera, full body visible feet to crown.",
          "back_medium":              "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly away from camera, cropped from the waist to just above the crown.",
          "back_bust":                "On a plain neutral grey background with flat studio lighting, rotate the subject to face directly away from camera, cropped from the mid-back to just above the crown.",
          "back_three_quarter_left":  "On a plain neutral grey background with flat studio lighting, turn the subject to a rear three-quarter angle facing back-left, full body visible.",
          "back_three_quarter_right": "On a plain neutral grey background with flat studio lighting, turn the subject to a rear three-quarter angle facing back-right, full body visible.",
          "back_three_quarter_left_medium": "On a plain neutral grey background with flat studio lighting, turn the subject to a rear three-quarter angle facing back-left, cropped from the waist up.",
          "back_three_quarter_right_medium":"On a plain neutral grey background with flat studio lighting, turn the subject to a rear three-quarter angle facing back-right, cropped from the waist up.",
          "over_shoulder_left":       "On a plain neutral grey background with flat studio lighting, the subject's back is toward the camera with their right shoulder and the back of their head filling the foreground. The subject faces directly away from camera. The viewpoint is from just over their right shoulder, close behind them. A small amount of the face is visible.",
          "over_shoulder_right":      "On a plain neutral grey background with flat studio lighting, the subject's back is toward the camera with their left shoulder and the back of their head filling the foreground. The subject faces directly away from camera. The viewpoint is from just over their left shoulder, close behind them. A small amount of the face is visible.",

          "low_angle_full":           "On a plain neutral grey background with flat studio lighting, shoot from a low angle with the camera at knee level looking upward, full body visible.",
          "low_angle_medium":         "On a plain neutral grey background with flat studio lighting, shoot from a low angle with the camera at waist level looking upward, cropped from the waist to above the crown.",
          "low_angle_face":           "On a plain neutral grey background with flat studio lighting, shoot from a low angle with the camera at chin level looking upward, face and crown visible with chin prominent in frame.",
          "low_angle_three_quarter_left": "On a plain neutral grey background with flat studio lighting, shoot from a low angle at knee level with the subject turned 45 degrees to the left, full body visible.",
          "low_angle_three_quarter_right":"On a plain neutral grey background with flat studio lighting, shoot from a low angle at knee level with the subject turned 45 degrees to the right, full body visible.",

          "high_angle_full":          "On a plain neutral grey background with flat studio lighting, shoot from a high angle with the camera well above head height looking steeply downward, full body visible.",
          "high_angle_medium":        "On a plain neutral grey background with flat studio lighting, shoot from a high angle with the camera above eye level looking downward, cropped from the chest up.",
          "high_angle_face":          "On a plain neutral grey background with flat studio lighting, shoot from a high angle directly above the face looking steeply downward, face filling the frame crown prominent.",
          "top_down":                 "On a plain neutral grey background with flat studio lighting, shoot from directly above in a true top-down bird's eye view, full body visible.",

          "seated_front_full":        "On a plain neutral grey background with flat studio lighting, show the subject seated facing front, full body visible from feet to crown.",
          "seated_front_medium":      "On a plain neutral grey background with flat studio lighting, show the subject seated facing front, cropped from the lap to just above the crown.",
          "seated_three_quarter_left":"On a plain neutral grey background with flat studio lighting, show the subject seated and turned 45 degrees to the left, full body visible.",
          "seated_three_quarter_right":"On a plain neutral grey background with flat studio lighting, show the subject seated and turned 45 degrees to the right, full body visible.",
          "kneeling_front":           "On a plain neutral grey background with flat studio lighting, show the subject kneeling facing front, full body visible.",
          "kneeling_three_quarter_left":"On a plain neutral grey background with flat studio lighting, show the subject kneeling and turned 45 degrees to the left, full body visible."

        },

        "lighting": {
          "overcast":           "On a plain neutral grey background, change the lighting to soft diffuse overcast with no directional shadows, even illumination across the entire face.",
          "hard_frontal":       "On a plain neutral grey background, change the lighting to a hard on-axis frontal spot, sharp cast shadows falling directly behind facial features.",
          "three_quarter_soft": "On a plain neutral grey background, change the lighting to a soft key light from the upper right at 45 degrees, with a gentle fill light from the left reducing shadow depth.",
          "overhead":           "On a plain neutral grey background, change the lighting to a direct overhead source, casting downward shadows deep into the eye sockets, under the nose, and beneath the chin.",
          "under_light":        "On a plain neutral grey background, change the lighting to a source from directly below, casting upward shadows across the jaw, cheeks, nose bridge, and brow ridge.",
          "split":              "On a plain neutral grey background, change the lighting to a hard split light with one side of the face fully lit and the opposite side in complete shadow.",
          "rim":                "On a plain neutral grey background, change the lighting to a strong rim light from behind and to one side, creating a bright edge outline on the hair and shoulder with the face in shadow."
        }
      },

      "object": {
        "base": "",

        "angles": {

          "front_full":             "On a plain neutral grey background with flat studio lighting, show the object from directly in front, full object visible with small margin on all sides.",
          "front_closeup":          "On a plain neutral grey background with flat studio lighting, show the object from directly in front in a tight close-up, filling the frame with the object's front face and primary details.",
          "front_detail":           "On a plain neutral grey background with flat studio lighting, show the most visually distinctive feature of the object in an extreme close-up from directly in front.",

          "slight_left":            "On a plain neutral grey background with flat studio lighting, rotate the object approximately 15 degrees to the left, full object visible.",
          "slight_right":           "On a plain neutral grey background with flat studio lighting, rotate the object approximately 15 degrees to the right, full object visible.",

          "45_left":                "On a plain neutral grey background with flat studio lighting, rotate the object 45 degrees to the left, full object visible.",
          "45_right":               "On a plain neutral grey background with flat studio lighting, rotate the object 45 degrees to the right, full object visible.",
          "45_left_closeup":        "On a plain neutral grey background with flat studio lighting, rotate the object 45 degrees to the left in a tight close-up filling the frame.",
          "45_right_closeup":       "On a plain neutral grey background with flat studio lighting, rotate the object 45 degrees to the right in a tight close-up filling the frame.",

          "90_left":                "On a plain neutral grey background with flat studio lighting, rotate the object 90 degrees to show the strict left side profile, full object visible.",
          "90_right":               "On a plain neutral grey background with flat studio lighting, rotate the object 90 degrees to show the strict right side profile, full object visible.",
          "90_left_closeup":        "On a plain neutral grey background with flat studio lighting, rotate the object 90 degrees to the left side profile in a tight close-up filling the frame.",
          "90_right_closeup":       "On a plain neutral grey background with flat studio lighting, rotate the object 90 degrees to the right side profile in a tight close-up filling the frame.",

          "135_left":               "On a plain neutral grey background with flat studio lighting, rotate the object 135 degrees to the left, showing the rear-left three-quarter angle, full object visible.",
          "135_right":              "On a plain neutral grey background with flat studio lighting, rotate the object 135 degrees to the right, showing the rear-right three-quarter angle, full object visible.",

          "back_full":              "On a plain neutral grey background with flat studio lighting, rotate the object to face directly away, showing the full back face, full object visible.",
          "back_closeup":           "On a plain neutral grey background with flat studio lighting, rotate the object to face directly away in a tight close-up, back face filling the frame.",

          "orbit_high_front":       "On a plain neutral grey background with flat studio lighting, elevate the camera to 45 degrees above and shoot downward at the front of the object, full object visible.",
          "orbit_high_45_left":     "On a plain neutral grey background with flat studio lighting, elevate the camera to 45 degrees above and orbit to the front-left, shooting downward at the object at a 45 degree horizontal angle, full object visible.",
          "orbit_high_45_right":    "On a plain neutral grey background with flat studio lighting, elevate the camera to 45 degrees above and orbit to the front-right, shooting downward at the object at a 45 degree horizontal angle, full object visible.",
          "orbit_high_90_left":     "On a plain neutral grey background with flat studio lighting, elevate the camera to 45 degrees above and orbit directly to the left side, shooting downward at the object, full object visible.",
          "orbit_high_90_right":    "On a plain neutral grey background with flat studio lighting, elevate the camera to 45 degrees above and orbit directly to the right side, shooting downward at the object, full object visible.",
          "orbit_high_back":        "On a plain neutral grey background with flat studio lighting, elevate the camera to 45 degrees above and orbit to directly behind the object, shooting downward, full object visible.",

          "top_down":               "On a plain neutral grey background with flat studio lighting, shoot the object from directly above in a true top-down view, full object visible with small margin on all sides.",
          "top_down_slight_front":  "On a plain neutral grey background with flat studio lighting, shoot the object from above at a very steep angle tilted slightly toward the front face, full object visible.",

          "orbit_low_front":        "On a plain neutral grey background with flat studio lighting, lower the camera to 20 degrees below center and shoot upward at the front of the object, full object visible.",
          "orbit_low_45_left":      "On a plain neutral grey background with flat studio lighting, lower the camera to 20 degrees below center and orbit to the front-left, shooting upward at the object, full object visible.",
          "orbit_low_45_right":     "On a plain neutral grey background with flat studio lighting, lower the camera to 20 degrees below center and orbit to the front-right, shooting upward at the object, full object visible.",
          "orbit_low_90_left":      "On a plain neutral grey background with flat studio lighting, lower the camera to 20 degrees below center and orbit directly to the left side shooting upward at the object, full object visible.",
          "orbit_low_90_right":     "On a plain neutral grey background with flat studio lighting, lower the camera to 20 degrees below center and orbit directly to the right side shooting upward at the object, full object visible.",

          "underside":              "On a plain neutral grey background with flat studio lighting, flip the object to show the underside or bottom face directly facing the camera, full object visible.",
          "underside_slight_tilt":  "On a plain neutral grey background with flat studio lighting, tilt the object to show the underside at a slight angle, bottom and one side visible simultaneously."

        },

        "lighting": {
          "overcast":           "On a plain neutral grey background, change the lighting to soft diffuse overcast with no directional shadows, even illumination across the entire object.",
          "hard_frontal":       "On a plain neutral grey background, change the lighting to a hard on-axis frontal spot, sharp cast shadows falling directly behind the object revealing surface planes.",
          "three_quarter_soft": "On a plain neutral grey background, change the lighting to a soft 45-degree key light from the upper right with a gentle fill light from the left reducing shadow depth.",
          "overhead":           "On a plain neutral grey background, change the lighting to a direct overhead source casting downward shadows that reveal the object's top surface geometry.",
          "under_light":        "On a plain neutral grey background, change the lighting to a source from directly below casting upward shadows that reveal the object's underside geometry.",
          "split":              "On a plain neutral grey background, change the lighting to a hard split with one side of the object fully lit and the opposite side in complete shadow, revealing surface contours.",
          "rim":                "On a plain neutral grey background, change the lighting to a single, key light strikes the subject from the left and right sides creating a sharp silhouette with crisp highlights that gently wrap along objects contours.",
          "top_backlit":        "On a plain neutral grey background, change the lighting to a strong backlight from directly above and behind, creating a halo edge on the top surface with the front face in shadow."
        }
      },

      "environment": {
        "time_of_day": {
          "dawn":   "Change the lighting to early dawn with soft pink and purple pre-sunrise light on the horizon",
          "noon":   "Change the lighting to bright midday noon sun with harsh overhead light and short shadows",
          "sunset": "Change the lighting to golden sunset with warm orange-red light from the horizon",
          "night":  "Change the scene to night-time with dark sky, stars, and ambient artificial or moonlight lighting"
        },
        "weather": {
          "clear":    "Change the weather to a clear sunny day with bright blue sky and no clouds",
          "overcast": "Change the weather to overcast with thick cloud cover and diffuse grey light",
          "rain":     "Change the weather to rainy with visible rainfall, wet glistening surfaces, and grey atmosphere",
          "fog":      "Change the weather to foggy with thick mist reducing visibility and creating soft diffuse ambient lighting",
          "snow":     "Change the weather to snowy with gently falling snow and white snow coverage on all surfaces"
        },
        "season": {
          "spring": "Change the season to spring with fresh green leaves, blooming flowers, and mild soft light",
          "summer": "Change the season to summer with full lush green foliage, bright warm light, and vibrant atmosphere",
          "autumn": "Change the season to autumn with orange, red, and yellow falling leaves and warm earthy tones",
          "winter": "Change the season to winter with bare trees, frost on surfaces, and cold blue-grey atmosphere"
        }
      }
    }
  }
}
EOL
echo "    ✓ fuk/config/defaults_dataset.json.template"

# ── Tool configs ──────────────────────────────────────────────────────────────
# Per-vendor-repo settings, read directly by the preprocessors. Without these
# the depth preprocessor silently falls back to different inference settings
# than the ones tuned here, so they are part of a correct install.

cat > fuk/config/tools/depth-anything-v3.json.template << 'EOL'
{
  "_comment": "Depth Anything V3 - Repo-specific settings",
  "_repo": "https://github.com/DepthAnything/Depth-Anything-3",

  "models": {
    "da3_mono_large": {
      "model_id": "depth-anything/DA3MONO-LARGE",
      "description": "Monocular depth, best for single images"
    },
    "da3_metric_large": {
      "model_id": "depth-anything/DA3METRIC-LARGE",
      "description": "Metric depth in meters"
    },
    "da3_large": {
      "model_id": "depth-anything/DA3-LARGE-1.1",
      "description": "Multi-view capable"
    },
    "da3_giant": {
      "model_id": "depth-anything/DA3-GIANT-1.1",
      "description": "Largest model, highest quality"
    }
  },

  "inference_defaults": {
    "process_res": 1280,
    "process_res_method": "upper_bound_resize"
  },

  "process_res_methods": {
    "_comment": "Available resize methods for inference",
    "available": ["lower_bound_resize", "upper_bound_resize", "exact_resize"],
    "recommended": "exact_resize"
  },

  "output": {
    "dtype": "float32",
    "range": [0, 1],
    "normalized": true
  },

  "fallback": {
    "_comment": "What to use if DA3 fails to load",
    "model": "depth_anything_v2",
    "reason": "DA3 requires separate installation"
  }
}
EOL
echo "    ✓ fuk/config/tools/depth-anything-v3.json.template"

cat > fuk/config/tools/sam2.json.template << 'EOL'
{
  "_comment": "SAM2 (Segment Anything Model 2) - Repo-specific settings",
  "_repo": "https://github.com/facebookresearch/sam2",

  "models": {
    "sam2.1_hiera_large": {
      "config": "sam2.1_hiera_l",
      "checkpoint": "sam2.1_hiera_large.pt",
      "description": "Large model, best quality"
    },
    "sam2.1_hiera_base": {
      "config": "sam2.1_hiera_b+",
      "checkpoint": "sam2.1_hiera_base_plus.pt",
      "description": "Base+ model, balanced"
    },
    "sam2.1_hiera_small": {
      "config": "sam2.1_hiera_s",
      "checkpoint": "sam2.1_hiera_small.pt",
      "description": "Small model, fastest"
    },
    "sam2.1_hiera_tiny": {
      "config": "sam2.1_hiera_t",
      "checkpoint": "sam2.1_hiera_tiny.pt",
      "description": "Tiny model, minimal VRAM"
    }
  },

  "default_model": "sam2.1_hiera_large",

  "checkpoint_paths": {
    "search_order": [
      "{models_dir}/sam/{checkpoint}",
      "{models_dir}/checkpoints/{checkpoint}",
      "{vendor_dir}/sam2/checkpoints/{checkpoint}"
    ]
  },

  "download": {
    "base_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824",
    "files": {
      "sam2.1_hiera_large.pt": "sam2.1_hiera_large.pt",
      "sam2.1_hiera_base_plus.pt": "sam2.1_hiera_base_plus.pt",
      "sam2.1_hiera_small.pt": "sam2.1_hiera_small.pt",
      "sam2.1_hiera_tiny.pt": "sam2.1_hiera_tiny.pt"
    }
  },

  "auto_mask_generator": {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100
  },

  "output": {
    "mask_format": "binary",
    "cryptomatte_bits": 32
  }
}
EOL
echo "    ✓ fuk/config/tools/sam2.json.template"

# Copy templates to live configs only if they don't exist yet
echo ""
echo "  → Initializing live configuration files..."

if [ ! -f "fuk/config/models.json" ]; then
    cp fuk/config/models.json.template fuk/config/models.json
    echo "    ✓ Created fuk/config/models.json — set your models_root path"
else
    echo "    ✓ fuk/config/models.json already exists (not overwriting)"
fi

for fragment in defaults.json defaults_loras.json defaults_vram.json defaults_spec_tool.json defaults_dataset.json; do
    if [ ! -f "fuk/config/$fragment" ]; then
        cp "fuk/config/$fragment.template" "fuk/config/$fragment"
        echo "    ✓ Created fuk/config/$fragment"
    else
        echo "    ✓ fuk/config/$fragment already exists (not overwriting)"
    fi
done

for tool in depth-anything-v3.json sam2.json; do
    if [ ! -f "fuk/config/tools/$tool" ]; then
        cp "fuk/config/tools/$tool.template" "fuk/config/tools/$tool"
        echo "    ✓ Created fuk/config/tools/$tool"
    else
        echo "    ✓ fuk/config/tools/$tool already exists (not overwriting)"
    fi
done

# ── Frontend ──────────────────────────────────────────────────────────────────
echo ""
echo "Note: Real-ESRGAN is optional but recommended for upscaling."
echo "Download pre-built binaries:"
echo ""
echo "  Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN/releases"
echo ""
echo "Extract to: fuk/vendor/realesrgan-ncnn/"
echo ""

echo ""
echo "[7/7] Installing frontend dependencies..."
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

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Virtual environment: ./venv"
echo "  Activate manually: source venv/bin/activate"
echo ""
echo "Next steps:"
echo ""
echo "1. IMPORTANT: Edit your configuration files:"
echo "   fuk/config/models.json             — set your models_root path"
echo "   fuk/config/defaults.json           — set models_root, aspect_ratios, image/video defaults"
echo "   fuk/config/defaults_loras.json     — set defined_loras_path, scanned_loras_path, and define loras"
echo "   fuk/config/defaults_vram.json      — set VRAM offload preset"
echo "   fuk/config/defaults_spec_tool.json — spec tool defaults"
echo "   fuk/config/defaults_dataset.json   — dataset generation defaults"
echo "   fuk/config/tools/*.json            — per-tool settings (DA3, SAM2)"
echo ""
echo "   Reference templates written by this script:"
echo "   fuk/config/models.json.template"
echo "   fuk/config/defaults*.json.template"
echo "   fuk/config/tools/*.json.template"
echo ""
echo "2. Download models:"
echo "   ./download_models.sh"
echo ""
echo "   Trim unwanted entries from fuk/config/models.json first —"
echo "   it downloads everything listed, at roughly 15-30GB per model."
echo ""
echo "3. Optional — single-image 3D (TRELLIS):"
if [ -f "fuk/vendor/TRELLIS_ENV/READY" ]; then
echo "   Already built — nothing to do."
else
echo "   bash fuk/core/threed/install_trellis_env.sh"
echo ""
echo "   Builds a private environment (~18GB) because TRELLIS needs a torch"
echo "   and CUDA toolchain that cannot share this venv. Skip it and only"
echo "   single-image 3D is unavailable; multi-view (VGGT) works regardless."
fi
echo ""
echo "4. Start FUK:"
echo "   ./start.sh"
echo ""
echo "Configuration notes:"
echo "  • Your config files (fuk/config/*.json) are gitignored — edit freely"
echo "  • Templates (*.json.template) are rewritten on every setup.sh run;"
echo "    they are the canonical reference, so don't edit them directly"
echo "  • Live configs are never overwritten once they exist — after an"
echo "    update, diff yours against the template to pick up new keys"
echo ""
