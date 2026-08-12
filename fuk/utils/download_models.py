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


# DiffSynth resolves the download source per ModelConfig, defaulting to
# ModelScope. Neither host carries everything: the PAI/* repos and
# DiffSynth-Studio's converted Wan safetensors are ModelScope-only, while
# HuggingFace is usually far faster for everything that exists on both. So
# each component tries the preferred source and falls back to the other
# rather than taking the whole run down with it.
SOURCES = ("huggingface", "modelscope")


def preferred_source() -> str:
    """Source to try first — DIFFSYNTH_DOWNLOAD_SOURCE, else DiffSynth's default."""
    env = os.environ.get('DIFFSYNTH_DOWNLOAD_SOURCE')
    if env and env.lower() in SOURCES:
        return env.lower()
    return "modelscope"


def download_model_component(model_id: str, pattern: str, component_name: str = None):
    """
    Download one component, trying the preferred source then the other.

    Returns the local path on success, or None if every source failed. A
    failure is reported and skipped rather than raised — one unreachable
    component should not abandon a download measured in hundreds of GB.
    """
    display_name = f"{model_id} → {pattern}" if not component_name else f"{component_name}: {model_id} → {pattern}"
    print(f"\n{'='*80}")
    print(f"Downloading: {display_name}")
    print(f"{'='*80}")

    first = preferred_source()
    order = [first] + [s for s in SOURCES if s != first]

    errors = {}
    for source in order:
        try:
            config = ModelConfig(
                model_id=model_id,
                origin_file_pattern=pattern,
                download_source=source,
            )
            config.download_if_necessary()
            print(f"✓ Downloaded from {source}: {config.path}")
            return config.path
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors[source] = f"{type(e).__name__}: {e}"
            remaining = [s for s in order if s not in errors]
            if remaining:
                print(f"  ⚠ {source} failed ({type(e).__name__}) — trying {remaining[0]}...")

    print(f"  ✗ FAILED on every source:")
    for source, err in errors.items():
        print(f"      {source}: {err[:160]}")
    return None


def download_all_models(models_config: dict):
    """
    Process all models from models.json and download their components.

    Returns (succeeded, failures) where failures is a list of
    (model_key, label, model_id, pattern) for everything that could not be
    fetched from any source.
    """
    succeeded = 0
    failures = []

    for model_key, model_spec in models_config.items():
        # Skip comment entries
        if model_key.startswith('_'):
            continue

        print(f"\n\n{'#'*80}")
        print(f"# Processing Model: {model_key}")
        print(f"# Description: {model_spec.get('description', 'No description')}")
        print(f"{'#'*80}")

        base_model_id = model_spec['model_id']

        # Flatten every downloadable piece into one list so the retry and
        # bookkeeping below stay in a single place.
        targets = []
        for idx, component in enumerate(model_spec.get('components', []), 1):
            targets.append((
                f"Component {idx}",
                component.get('model_id', base_model_id),
                component['pattern'],
            ))
        for section, label in (('tokenizer', 'Tokenizer'),
                               ('processor', 'Processor'),
                               ('lora', 'LoRA')):
            if section in model_spec:
                spec = model_spec[section]
                targets.append((
                    label,
                    spec.get('model_id', base_model_id),
                    spec['pattern'],
                ))

        if not targets:
            # 3D entries (trellis, vggt) carry no components — their weights
            # arrive via install_trellis_env.sh or on first use from the hub.
            print(f"\n  (no downloadable components — skipping)")
            continue

        for label, model_id, pattern in targets:
            print(f"\n--- {label} ---")
            path = download_model_component(
                model_id=model_id,
                pattern=pattern,
                component_name=label,
            )
            if path is None:
                failures.append((model_key, label, model_id, pattern))
            else:
                succeeded += 1

    return succeeded, failures


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
    succeeded, failures = download_all_models(models)

    # Summary
    print(f"\n\n{'='*80}")
    print("Download Complete!" if not failures else "Download Finished — With Failures")
    print(f"{'='*80}")
    print(f"Components downloaded: {succeeded}")
    print(f"Models location: {models_root}")

    if failures:
        print(f"\nFailed on every source ({len(failures)}):")
        for model_key, label, model_id, pattern in failures:
            print(f"  {model_key} / {label}: {model_id} → {pattern}")
        print("\nThese models are incomplete and will not load. Re-running is safe —")
        print("anything already fetched is skipped, so only the gaps are retried.")

    print(f"{'='*80}\n")
    # Non-zero on partial completion so a wrapper script or CI can tell the
    # difference between "everything landed" and "most things landed".
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()