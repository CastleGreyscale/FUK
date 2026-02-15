#!/usr/bin/env python3
"""
FUK Model Downloader
Downloads all models defined in models.json using DiffSynth's built-in download mechanism.
Sets DIFFSYNTH_MODEL_BASE_PATH from defaults.json to control download location.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from diffsynth.core import ModelConfig


def find_config_file(filename: str, search_paths: list[Path]) -> Path:
    """Search for config file in multiple locations."""
    for search_path in search_paths:
        config_path = search_path / filename
        if config_path.exists():
            return config_path
    
    # If not found, show where we looked
    locations = '\n  '.join(str(p / filename) for p in search_paths)
    raise FileNotFoundError(
        f"Could not find {filename}. Searched:\n  {locations}"
    )


def load_config(config_path: Path) -> dict:
    """Load JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def download_model_component(model_id: str, pattern: str, component_name: str = None):
    """Download a single model component using DiffSynth's ModelConfig."""
    display_name = f"{model_id} → {pattern}" if not component_name else f"{component_name}: {model_id} → {pattern}"
    print(f"\n{'='*80}")
    print(f"Downloading: {display_name}")
    print(f"{'='*80}")
    
    config = ModelConfig(
        model_id=model_id,
        origin_file_pattern=pattern,
    )
    
    config.download_if_necessary()
    print(f"✓ Downloaded to: {config.path}")
    return config.path


def download_all_models(models_config: dict):
    """Process all models from models.json and download their components."""
    total_downloads = 0
    
    for model_key, model_spec in models_config.items():
        # Skip comment entries
        if model_key.startswith('_'):
            continue
            
        print(f"\n\n{'#'*80}")
        print(f"# Processing Model: {model_key}")
        print(f"# Description: {model_spec.get('description', 'No description')}")
        print(f"{'#'*80}")
        
        base_model_id = model_spec['model_id']
        
        # Download main components
        if 'components' in model_spec:
            print(f"\n--- Main Components ---")
            for idx, component in enumerate(model_spec['components'], 1):
                # Use override model_id if specified, otherwise use base
                model_id = component.get('model_id', base_model_id)
                pattern = component['pattern']
                
                download_model_component(
                    model_id=model_id,
                    pattern=pattern,
                    component_name=f"Component {idx}"
                )
                total_downloads += 1
        
        # Download tokenizer if present
        if 'tokenizer' in model_spec:
            print(f"\n--- Tokenizer ---")
            tokenizer = model_spec['tokenizer']
            model_id = tokenizer.get('model_id', base_model_id)
            pattern = tokenizer['pattern']
            
            download_model_component(
                model_id=model_id,
                pattern=pattern,
                component_name="Tokenizer"
            )
            total_downloads += 1
        
        # Download processor if present
        if 'processor' in model_spec:
            print(f"\n--- Processor ---")
            processor = model_spec['processor']
            model_id = processor.get('model_id', base_model_id)
            pattern = processor['pattern']
            
            download_model_component(
                model_id=model_id,
                pattern=pattern,
                component_name="Processor"
            )
            total_downloads += 1
        
        # Download LoRA if present
        if 'lora' in model_spec:
            print(f"\n--- LoRA ---")
            lora = model_spec['lora']
            model_id = lora['model_id']
            pattern = lora['pattern']
            
            download_model_component(
                model_id=model_id,
                pattern=pattern,
                component_name="LoRA"
            )
            total_downloads += 1
    
    return total_downloads


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download all models defined in models.json'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        help='Directory containing defaults.json and models.json (default: auto-search)'
    )
    parser.add_argument(
        '--defaults',
        type=Path,
        help='Path to defaults.json (overrides --config-dir)'
    )
    parser.add_argument(
        '--models',
        type=Path,
        help='Path to models.json (overrides --config-dir)'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Determine search paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if args.config_dir:
        # User specified config directory
        search_paths = [args.config_dir]
    else:
        # Auto-search: project root, config/, script dir
        search_paths = [
            project_root,                    # /path/to/project/
            project_root / 'config',         # /path/to/project/config/
            script_dir,                      # /path/to/project/utils/
        ]
    
    # Find config files
    try:
        if args.defaults:
            defaults_path = args.defaults
        else:
            defaults_path = find_config_file('defaults.json', search_paths)
        
        if args.models:
            models_path = args.models
        else:
            models_path = find_config_file('models.json', search_paths)
        
        print(f"Using configs:")
        print(f"  defaults.json: {defaults_path}")
        print(f"  models.json:   {models_path}\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load configurations
    print("Loading configuration files...")
    defaults = load_config(defaults_path)
    models = load_config(models_path)
    
    # Set model root from defaults
    models_root = defaults.get('models_root', '/home/brad/ai/models')
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = models_root
    
    print(f"\n{'='*80}")
    print(f"FUK Model Download Configuration")
    print(f"{'='*80}")
    print(f"Models root: {models_root}")
    print(f"Total models to process: {len([k for k in models.keys() if not k.startswith('_')])}")
    print(f"{'='*80}\n")
    
    # Confirm before proceeding
    if not args.yes:
        response = input("Proceed with download? [y/N]: ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    # Download all models
    total = download_all_models(models)
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"Download Complete!")
    print(f"{'='*80}")
    print(f"Total components downloaded: {total}")
    print(f"Models location: {models_root}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()