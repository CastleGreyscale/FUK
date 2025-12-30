#!/usr/bin/env python3
"""
Test script for FUK preprocessor auto-download system

Tests:
1. Config reading
2. Path resolution
3. Download simulation
4. Integration with preprocessors
"""

from pathlib import Path
import json
import sys

print("="*60)
print("FUK Preprocessor Auto-Download Test")
print("="*60)

# Test 1: Config reading
print("\n[Test 1] Reading models.json...")

# Determine config path relative to script location
script_dir = Path(__file__).parent
config_path = script_dir.parent / "config" / "models.json"

print(f"  Looking for config at: {config_path}")

if not config_path.exists():
    print(f"✗ Config not found: {config_path}")
    print(f"  Script location: {script_dir}")
    sys.exit(1)

with open(config_path) as f:
    config = json.load(f)

if "preprocessor_models" not in config:
    print("✗ No preprocessor_models section in config")
    sys.exit(1)

print(f"✓ Config loaded")
print(f"  Found {len(config['preprocessor_models'])} preprocessor models:")
for model_name, model_info in config['preprocessor_models'].items():
    if 'checkpoint' in model_info:
        print(f"    - {model_name}: {model_info['checkpoint']}")
    else:
        print(f"    - {model_name}: {model_info.get('note', 'Auto-download')}")

# Test 2: Check Depth Anything V2 config
print("\n[Test 2] Checking Depth Anything V2 config...")
da_config = config['preprocessor_models'].get('depth_anything_v2')

if not da_config:
    print("✗ depth_anything_v2 not found in config")
    sys.exit(1)

checkpoint_path = Path(da_config['checkpoint']).expanduser()
url = da_config.get('url')
size_mb = da_config.get('size_mb', 0)

print(f"✓ Depth Anything V2 configured:")
print(f"  Checkpoint: {checkpoint_path}")
print(f"  URL: {url}")
print(f"  Size: ~{size_mb} MB")

# Test 3: Check if checkpoint exists
print("\n[Test 3] Checking for existing checkpoint...")
if checkpoint_path.exists():
    size_actual = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"✓ Checkpoint found: {checkpoint_path}")
    print(f"  Size: {size_actual:.1f} MB")
else:
    print(f"⚠ Checkpoint not found: {checkpoint_path}")
    print(f"  Will be auto-downloaded on first use")
    
    # Check if parent directory exists
    if not checkpoint_path.parent.exists():
        print(f"⚠ Parent directory doesn't exist: {checkpoint_path.parent}")
        print(f"  Create it with: mkdir -p {checkpoint_path.parent}")
    else:
        print(f"✓ Parent directory exists: {checkpoint_path.parent}")

# Test 4: Check vendor directory
print("\n[Test 4] Checking vendor directory...")
vendor_path = script_dir.parent / "vendor" / "depth-anything-v2"
vendor_checkpoint = vendor_path / "checkpoints" / "depth_anything_v2_vitl.pth"

if vendor_path.exists():
    print(f"✓ Vendor directory exists: {vendor_path}")
    
    if vendor_checkpoint.exists():
        size_vendor = vendor_checkpoint.stat().st_size / (1024 * 1024)
        print(f"✓ Vendor checkpoint found: {vendor_checkpoint}")
        print(f"  Size: {size_vendor:.1f} MB")
    else:
        print(f"⚠ No checkpoint in vendor directory")
        print(f"  Expected: {vendor_checkpoint}")
else:
    print(f"⚠ Vendor directory not found: {vendor_path}")
    print(f"  Clone with: cd {script_dir.parent} && git clone https://github.com/DepthAnything/Depth-Anything-V2 vendor/depth-anything-v2")

# Test 5: Test model_downloader import
print("\n[Test 5] Testing model_downloader...")
try:
    from utils.model_downloader import ModelDownloader, ensure_model_downloaded
    print("✓ model_downloader module imported successfully")
    
    # Try to initialize downloader
    downloader = ModelDownloader(config_path)
    print("✓ ModelDownloader initialized")
    
    # Check if model info is readable
    model_info = downloader.get_preprocessor_model_info("depth_anything_v2")
    if model_info:
        print(f"✓ Model info accessible:")
        print(f"  Checkpoint: {model_info.get('checkpoint')}")
        print(f"  URL: {model_info.get('url')}")
    else:
        print("✗ Could not read model info")
        
except ImportError as e:
    print(f"✗ Failed to import model_downloader: {e}")
except Exception as e:
    print(f"✗ Error testing downloader: {e}")

# Test 6: Check preprocessor integration
print("\n[Test 6] Checking preprocessor integration...")
try:
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.preprocessors import PreprocessorManager, DepthModel
    print("✓ preprocessors module imported")
    
    # Try to initialize with config
    manager = PreprocessorManager(
        output_dir=Path("outputs/test_preprocessed"),
        config_path=config_path
    )
    print("✓ PreprocessorManager initialized with config_path")
    print(f"  Output dir: {manager.output_dir}")
    print(f"  Config path: {manager.config_path}")
    
except ImportError as e:
    print(f"✗ Failed to import preprocessors: {e}")
except Exception as e:
    print(f"✗ Error testing preprocessors: {e}")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)

if checkpoint_path.exists() or vendor_checkpoint.exists():
    print("✓ Ready to use - checkpoint found")
    print(f"  The system will use existing checkpoint")
elif checkpoint_path.parent.exists():
    print("⚠ First-time setup - checkpoint will auto-download")
    print(f"  On first depth preprocessing, checkpoint will download to:")
    print(f"  {checkpoint_path}")
else:
    print("⚠ Setup needed - create checkpoint directory")
    print(f"  Run: mkdir -p {checkpoint_path.parent}")
    print(f"  Then checkpoint will auto-download on first use")

print("\nTo test auto-download manually:")
print(f"  python utils/model_downloader.py depth_anything_v2")

print("\n" + "="*60)