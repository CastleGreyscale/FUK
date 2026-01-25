# Vendor Dependencies

This directory contains third-party repositories installed during setup.

## Contents (installed by setup.sh)

- `Depth-Anything-3/` - Depth estimation models
- `segment-anything-2/` - SAM2 segmentation
- `musubi-tuner/` - Generation model backend
- `DSINE/` - Normal map generation
- `Real-ESRGAN-ncnn-vulkan/` - Upscaling (optional)
- `rife-ncnn-vulkan/` - Frame interpolation (optional)

## Installation

These are automatically cloned and installed by the main setup script:

```bash
./setup.sh
```

Or manually:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git vendor/Depth-Anything-3
git clone https://github.com/facebookresearch/segment-anything-2 vendor/segment-anything-2
git clone https://github.com/kohya-ss/musubi-tuner.git vendor/musubi-tuner
git clone https://github.com/baegwangbin/DSINE.git vendor/DSINE

pip install -e ./vendor/Depth-Anything-3 --no-deps
pip install -e ./vendor/segment-anything-2
pip install -e ./vendor/musubi-tuner
```

## Note

This directory is excluded from version control. Each installation clones fresh copies.
```bash
mkdir rife-ncnn
cd rife-ncnn
# download correct interpolation package
# https://github.com/nihui/rife-ncnn-vulkan/releases
```