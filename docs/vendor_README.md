# Vendor Dependencies

This directory contains third-party repositories installed during setup.

## Contents (installed by setup.sh)

- `DiffSynth-Studio/` - Core generation backend (Qwen, Wan pipelines)
- `Depth-Anything-3/` - Depth estimation models
- `segment-anything-2/` - SAM2 segmentation
- `DSINE/` - Normal map generation
- `Real-ESRGAN-ncnn-vulkan/` - Upscaling (optional, pre-built binary)

## Installation

These are automatically cloned and installed by the main setup script:

```bash
./setup.sh
```

Or manually:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git fuk/vendor/DiffSynth-Studio
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git fuk/vendor/Depth-Anything-3
git clone https://github.com/facebookresearch/segment-anything-2 fuk/vendor/segment-anything-2
git clone https://github.com/baegwangbin/DSINE.git fuk/vendor/DSINE

pip install -e ./fuk/vendor/DiffSynth-Studio
pip install -e ./fuk/vendor/Depth-Anything-3
pip install -e ./fuk/vendor/segment-anything-2
```

## Note

This directory is excluded from version control. Each installation clones fresh copies.