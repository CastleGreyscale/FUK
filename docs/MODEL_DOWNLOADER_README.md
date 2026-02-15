# FUK Model Downloader

Automated model download script using DiffSynth's built-in download mechanism.

## Setup

1. **Add `models_root` to your `defaults.json`:**
   ```json
   {
     "models_root": "/home/brad/ai/models",
     "aspect_ratios": [
       ...
   ```

2. **Set the download path:**
   The script reads `models_root` from `defaults.json` and sets it as:
   ```bash
   export DIFFSYNTH_MODEL_BASE_PATH="/home/brad/ai/models"
   ```

## Usage

```bash
python download_models.py
```

The script will:
1. Load `defaults.json` and `models.json` from the same directory
2. Set `DIFFSYNTH_MODEL_BASE_PATH` from `defaults.models_root`
3. Show a summary of what will be downloaded
4. Ask for confirmation before proceeding
5. Download all model components defined in `models.json`:
   - Main components (transformers, text encoders, VAEs)
   - Tokenizers
   - Processors
   - LoRAs

## What Gets Downloaded

For each model in `models.json`, the script downloads:

### Example: `qwen_image`
- Component 1: `Qwen/Qwen-Image` → `transformer/diffusion_pytorch_model*.safetensors`
- Component 2: `Qwen/Qwen-Image` → `text_encoder/model*.safetensors`
- Component 3: `Qwen/Qwen-Image` → `vae/diffusion_pytorch_model.safetensors`
- Tokenizer: `Qwen/Qwen-Image` → `tokenizer/`

### Example: `qwen_edit`
- Component 1: `Qwen/Qwen-Image-Edit-2511` → `transformer/diffusion_pytorch_model*.safetensors`
- Component 2: `Qwen/Qwen-Image` → `text_encoder/model*.safetensors` (override model_id)
- Component 3: `Qwen/Qwen-Image` → `vae/diffusion_pytorch_model.safetensors` (override model_id)
- Processor: `Qwen/Qwen-Image-Edit` → `processor/`

### Example: `qwen_control_union`
- Standard components (transformer, text encoder, VAE)
- Tokenizer
- LoRA: `DiffSynth-Studio/Qwen-Image-In-Context-Control-Union` → `model.safetensors`

## Output

Downloads are organized by model_id under `models_root`:
```
/home/brad/ai/models/
├── Qwen/
│   ├── Qwen-Image/
│   │   ├── transformer/
│   │   ├── text_encoder/
│   │   ├── vae/
│   │   └── tokenizer/
│   └── Qwen-Image-Edit-2511/
│       └── transformer/
├── Wan-AI/
│   └── Wan2.2-I2V-A14B/
│       ├── high_noise_model/
│       ├── low_noise_model/
│       └── ...
└── DiffSynth-Studio/
    └── Qwen-Image-In-Context-Control-Union/
        └── model.safetensors
```

## Progress Tracking

The script shows detailed progress:
```
================================================================================
# Processing Model: qwen_image
# Description: Qwen text-to-image (base model)
================================================================================

--- Main Components ---

================================================================================
Downloading: Component 1: Qwen/Qwen-Image → transformer/diffusion_pytorch_model*.safetensors
================================================================================
✓ Downloaded to: /home/brad/ai/models/Qwen/Qwen-Image/transformer/...
```

## Integration

After downloading, you can use `defaults.models_root` consistently throughout FUK:
- Backend model loading
- Pipeline construction
- LoRA discovery
- Any future model operations

Replace hardcoded paths like:
```python
# Old
models_root = "/home/brad/ai/models"

# New
models_root = defaults['models_root']
```
