#!/usr/bin/env python3
"""
FUK Model Downloader
Downloads all models defined in models.json using DiffSynth's built-in download mechanism.
Sets DIFFSYNTH_MODEL_BASE_PATH from defaults.json to control download location.
"""



import argparse
import json
import os
import shutil
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
        # stage2_lora is LTX-2's distilled refine LoRA. It is merged at
        # from_pretrained time rather than loaded at runtime, but it still has
        # to be on disk before the first two-stage generation.
        for section, label in (('tokenizer', 'Tokenizer'),
                               ('processor', 'Processor'),
                               ('lora', 'LoRA'),
                               ('stage2_lora', 'Stage-2 LoRA')):
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


def download_seedvr2(models_root: str):
    """
    Pre-fetch SeedVR2 video-restoration weights.

    Returns the backend's result dict, or None when SeedVR2 is not installed —
    it is an optional vendor dependency, and a missing engine is a reason to
    skip quietly, not to fail a download run that otherwise succeeded.
    """
    print(f"\n\n{'#'*80}")
    print("# Processing: SeedVR2 (video restoration)")
    print(f"{'#'*80}")

    # core/ is not on sys.path for this script; add it the same way the
    # config search does, relative to this file.
    fuk_dir = Path(__file__).resolve().parent.parent
    if str(fuk_dir) not in sys.path:
        sys.path.insert(0, str(fuk_dir))

    try:
        from core import seedvr2_backend
    except ImportError as exc:
        print(f"\n  (SeedVR2 backend unavailable: {exc} — skipping)")
        return None

    status = seedvr2_backend.availability()
    if not status["available"]:
        print(f"\n  (not installed: {', '.join(status['missing'])} — skipping)")
        print(f"  {status['hint']}")
        return None

    # Honour models_root even if defaults.json was overridden on the command
    # line: the backend reads config itself, so point it at the same place.
    os.environ.setdefault("FUK_SEEDVR2_WEIGHTS", str(Path(models_root) / "seedvr2"))

    try:
        result = seedvr2_backend.download_weights()
    except Exception as exc:
        print(f"\n  ✗ SeedVR2 download failed: {exc}")
        return None

    if result["failed"]:
        print(f"\n  ⚠ Could not fetch: {', '.join(result['failed'])}")
    return result


def download_ltx2_loras(loras_config: dict):
    """
    Fetch the LTX-2 function LoRAs into the curated LoRA directory.

    These cannot ride the models.json loop: they are LoRAs, not pipeline
    components, and FUK's LoRA registry resolves them by filesystem path under
    defined_loras_path rather than by model_id. Each lives in its own upstream
    repo, so the repo is derived from the filename declared in
    defaults_loras.json.

    Downloads land in the shared model cache and are symlinked into the LoRA
    directory — the files are up to 2.4GB each and there is no reason to hold
    two copies. Falls back to copying where symlinks are unavailable.

    Returns (fetched, failed) counts, or None when nothing is configured.
    """
    entries = [
        e for e in loras_config.get("loras", [])
        if "ltx2" in (e.get("model") or []) and str(e.get("path", "")).startswith("ltx2/")
    ]
    if not entries:
        return None

    base = loras_config.get("defined_loras_path")
    if not base:
        print("\n  (defined_loras_path not set — skipping LTX-2 LoRAs)")
        return None

    print(f"\n\n{'#'*80}")
    print(f"# Processing: LTX-2 function LoRAs ({len(entries)})")
    print(f"{'#'*80}")
    print("# One 35GB base transformer, one small LoRA per camera move or control mode.")

    dest_dir = Path(base).expanduser() / "ltx2"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Repo names follow the filenames, which is why they are derived rather
    # than listed twice: camera moves live in LTX-2-19b-LoRA-Camera-Control-*,
    # in-context LoRAs in LTX-2-19b-IC-LoRA-*.
    fetched, failed = 0, 0
    for entry in entries:
        filename = Path(entry["path"]).name
        dest = dest_dir / filename
        if dest.exists():
            print(f"\n  ✓ already present: {filename}")
            fetched += 1
            continue

        stem = filename.replace("ltx-2-19b-", "").replace(".safetensors", "")
        if stem.startswith("lora-camera-control-"):
            move = stem[len("lora-camera-control-"):]
            repo = "Lightricks/LTX-2-19b-LoRA-Camera-Control-" + "-".join(
                p.capitalize() for p in move.split("-"))
        elif stem.startswith("ic-lora-union-control"):
            repo = "Lightricks/LTX-2-19b-IC-LoRA-Union-Control"
        elif stem.startswith("ic-lora-detailer"):
            repo = "Lightricks/LTX-2-19b-IC-LoRA-Detailer"
        else:
            print(f"\n  ⚠ no known repo for {filename} — skipping")
            failed += 1
            continue

        path = download_model_component(repo, filename, component_name=entry.get("name"))
        if path is None:
            failed += 1
            continue

        src = Path(path)
        if src.is_dir():
            src = src / filename
        try:
            dest.symlink_to(src)
            print(f"  ✓ linked → {dest}")
        except OSError:
            shutil.copy2(src, dest)
            print(f"  ✓ copied → {dest}")
        fetched += 1

    return fetched, failed


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

    # SeedVR2 is not a DiffSynth pipeline and carries no models.json components,
    # so the loop above cannot reach it. Fetch it here instead — otherwise the
    # first video upscale stalls on a silent 3.4GB download.
    seedvr2_result = download_seedvr2(models_root)

    # LTX-2's function LoRAs are likewise invisible to the models.json loop —
    # they are LoRAs resolved by path, not pipeline components. defaults_loras
    # lives beside defaults.json, so read it from the same directory.
    ltx2_lora_result = None
    loras_path = defaults_path.parent / "defaults_loras.json"
    if loras_path.exists():
        try:
            ltx2_lora_result = download_ltx2_loras(load_config(loras_path))
        except Exception as exc:
            print(f"\n  ✗ LTX-2 LoRA download failed: {exc}")

    # Summary
    print(f"\n\n{'='*80}")
    print("Download Complete!" if not failures else "Download Finished — With Failures")
    print(f"{'='*80}")
    print(f"Components downloaded: {succeeded}")
    print(f"Models location: {models_root}")
    if seedvr2_result is not None:
        fetched = len(seedvr2_result["fetched"])
        total = fetched + len(seedvr2_result["failed"])
        print(f"SeedVR2 variants ready: {fetched}/{total} in {seedvr2_result['weights_dir']}")
    if ltx2_lora_result is not None:
        got, bad = ltx2_lora_result
        print(f"LTX-2 function LoRAs ready: {got}/{got + bad}")

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