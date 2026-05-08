"""
LoRA Dataset Manager
Manages dataset jobs: variation presets, job state, and sequential generation
using qwen_edit to produce structured training image sets.
"""

import json
import uuid
import shutil
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# ============================================================================
# Variation Presets — derived from defaults.json at runtime
# ============================================================================

_prompt_config: Optional[Dict] = None


def set_prompt_config(cfg: Optional[Dict]):
    global _prompt_config
    _prompt_config = cfg or {}


def _key_to_label(key: str) -> str:
    return key.replace('_', ' ').title()


def get_variation_presets() -> Dict:
    """Build the presets structure from the loaded prompt config."""
    if not _prompt_config:
        return {}
    presets = {}
    for subject_type, subject_data in _prompt_config.items():
        if not isinstance(subject_data, dict):
            continue
        presets[subject_type] = {}
        for pack_key, pack_data in subject_data.items():
            if pack_key == 'base' or not isinstance(pack_data, dict):
                continue
            presets[subject_type][pack_key] = {
                'label': _key_to_label(pack_key),
                'variations': [
                    {'id': var_id, 'label': _key_to_label(var_id)}
                    for var_id in pack_data
                ],
            }
    return presets


# ============================================================================
# Job state tracking (module-level, shared across endpoints)
# ============================================================================

# job_id -> dict with full job state
dataset_jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helpers
# ============================================================================

def _safe_name(name: str) -> str:
    """Slug-safe subject name."""
    import re
    return re.sub(r"[^\w\-]", "_", name.lower().strip()).strip("_") or "dataset"


def _build_variation_list(
    subject_type: str,
    selected_packs: List[str],
    seed_strategy: str,
    base_seed: int,
    prompt_config: Optional[Dict] = None,
    excluded_variation_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """Resolve selected pack keys into an ordered variation list with seeds.
    Variations and prompts are sourced entirely from prompt_config
    (defaults.json lora_dataset.variation_prompts).
    """
    excluded = set(excluded_variation_ids or [])
    cfg = prompt_config or _prompt_config or {}
    type_cfg = cfg.get(subject_type, {})
    base_prompt = type_cfg.get("base", "")
    variations = []
    idx = 1
    for pack_key in selected_packs:
        pack_prompts = type_cfg.get(pack_key)
        if not isinstance(pack_prompts, dict):
            continue
        for var_id, var_prompt in pack_prompts.items():
            if var_id in excluded:
                continue
            prompt = f"{base_prompt} {var_prompt}".strip() if base_prompt else var_prompt
            seed = (base_seed + idx) % (2**32) if seed_strategy == "fixed" else random.randint(0, 2**32 - 1)
            variations.append({
                "id":       f"{idx:03d}_{var_id}",
                "pack":     pack_key,
                "label":    _key_to_label(var_id),
                "prompt":   prompt,
                "seed":     seed,
                "status":   "pending",
                "approved": None,
                "error":    None,
            })
            idx += 1
    return variations


def _manifest_dict(job: Dict) -> Dict:
    """Return the dataset_manifest.json representation of a job."""
    return {
        "subject_name": job["subject_name"],
        "subject_type": job["subject_type"],
        "created":      job["created"],
        "sources":      [f"sources/{Path(p).name}" for p in job["source_paths"]],
        "params":       job["params"],
        "variations":   job["variations"],
    }


def _save_manifest(job: Dict):
    job_dir = Path(job["job_dir"])
    manifest_path = job_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(_manifest_dict(job), f, indent=2)


# ============================================================================
# Job creation
# ============================================================================

def create_dataset_job(
    datasets_root: Path,
    subject_name: str,
    subject_type: str,
    source_paths: List[str],
    selected_packs: List[str],
    params: Dict,
    prompt_config: Optional[Dict] = None,
    excluded_variation_ids: Optional[List[str]] = None,
) -> str:
    """Create a new dataset job, set up directories, copy sources. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]
    date_str = datetime.now().strftime("%Y%m%d")
    dir_name = f"{_safe_name(subject_name)}_{date_str}_{job_id}"
    job_dir = datasets_root / dir_name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "sources").mkdir(exist_ok=True)
    (job_dir / "generated").mkdir(exist_ok=True)
    (job_dir / "approved").mkdir(exist_ok=True)

    # Copy source images
    copied_sources = []
    for i, src in enumerate(source_paths):
        src_path = Path(src)
        if src_path.exists():
            dst = job_dir / "sources" / f"source_{i+1:03d}{src_path.suffix}"
            shutil.copy2(src_path, dst)
            copied_sources.append(str(dst))

    seed_strategy = params.get("seed_strategy", "fixed")
    base_seed = params.get("seed", 42)
    base_variations = _build_variation_list(
        subject_type, selected_packs, seed_strategy, base_seed, prompt_config, excluded_variation_ids
    )

    # When multiple sources, replicate variations per source so each image is generated independently
    if len(copied_sources) > 1:
        variations = []
        for src_idx, _ in enumerate(copied_sources):
            src_num = src_idx + 1
            for v in base_variations:
                new_v = {**v,
                         "id": f"s{src_num}_{v['id']}",
                         "label": f"[Src {src_num}] {v['label']}",
                         "source_idx": src_idx}
                if seed_strategy == "random":
                    new_v["seed"] = random.randint(0, 2**32 - 1)
                variations.append(new_v)
    else:
        variations = [{**v, "source_idx": 0} for v in base_variations]

    job = {
        "job_id":       job_id,
        "job_dir":      str(job_dir),
        "subject_name": subject_name,
        "subject_type": subject_type,
        "source_paths": copied_sources,
        "variations":   variations,
        "params":       params,
        "created":      datetime.now().isoformat(),
        "status":       "queued",
        "progress":     0.0,
        "current_idx":  0,
        "error":        None,
    }

    dataset_jobs[job_id] = job
    _save_manifest(job)
    return job_id


# ============================================================================
# Runner
# ============================================================================

async def run_dataset_job(job_id: str, generation_backend):
    """
    Sequentially generate all variations for the job.
    Updates dataset_jobs[job_id] in place so the SSE stream can report progress.
    """
    job = dataset_jobs.get(job_id)
    if not job:
        return

    job_dir = Path(job["job_dir"])
    params = job["params"]
    source_paths = [Path(p) for p in job["source_paths"] if Path(p).exists()]
    total = len(job["variations"])

    job["status"] = "running"
    job["current_idx"] = 0

    for idx, variation in enumerate(job["variations"]):
        # Check for cancellation
        if job.get("cancelled"):
            job["status"] = "cancelled"
            _save_manifest(job)
            return

        var_dir = job_dir / "generated" / variation["id"]
        var_dir.mkdir(parents=True, exist_ok=True)
        output_path = var_dir / "generated.png"

        variation["status"] = "running"
        job["current_idx"] = idx
        job["progress"] = idx / total

        print(f"[LoRA Dataset] Generating {idx+1}/{total} — {variation['label']}")

        # Use per-variation source when multiple inputs were provided
        src_idx = variation.get("source_idx", 0)
        ctrl_image = [source_paths[src_idx]] if source_paths and src_idx < len(source_paths) else (source_paths or None)

        try:
            seed = variation["seed"]
            result = await asyncio.to_thread(
                generation_backend.run,
                "image",
                prompt=variation["prompt"],
                output_path=output_path,
                model="qwen_edit",
                seed=seed,
                steps=params.get("steps", 28),
                guidance_scale=params.get("cfg_scale", 5.0),
                denoising_strength=1.0,
                lora=params.get("lora") or None,
                lora_multiplier=params.get("lora_alpha", 1.0),
                control_image=ctrl_image,
            )
            variation["status"] = "completed"
            print(f"[LoRA Dataset] Done: {variation['label']}")

        except Exception as e:
            variation["status"] = "failed"
            variation["error"] = str(e)
            print(f"[LoRA Dataset] Failed: {variation['label']} — {e}")

        job["progress"] = (idx + 1) / total
        _save_manifest(job)

    job["status"] = "complete"
    job["progress"] = 1.0
    _save_manifest(job)
    print(f"[LoRA Dataset] Job {job_id} complete — {total} variations")


# ============================================================================
# Rerun a single variation
# ============================================================================

async def rerun_single_variation(job_id: str, variation_id: str, generation_backend):
    """Re-generate one variation in place, replacing its previous output."""
    job = dataset_jobs.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    variation = next((v for v in job["variations"] if v["id"] == variation_id), None)
    if not variation:
        raise ValueError(f"Variation {variation_id!r} not found")

    job_dir = Path(job["job_dir"])
    params = job["params"]
    source_paths = [Path(p) for p in job["source_paths"] if Path(p).exists()]

    var_dir = job_dir / "generated" / variation["id"]
    var_dir.mkdir(parents=True, exist_ok=True)
    output_path = var_dir / "generated.png"

    variation["status"] = "running"
    variation["error"] = None
    variation["approved"] = None
    variation["seed"] = random.randint(0, 2**32 - 1)
    _save_manifest(job)

    src_idx = variation.get("source_idx", 0)
    ctrl_image = [source_paths[src_idx]] if source_paths and src_idx < len(source_paths) else (source_paths or None)

    try:
        await asyncio.to_thread(
            generation_backend.run,
            "image",
            prompt=variation["prompt"],
            output_path=output_path,
            model="qwen_edit",
            seed=variation["seed"],
            steps=params.get("steps", 28),
            guidance_scale=params.get("cfg_scale", 5.0),
            denoising_strength=1.0,
            lora=params.get("lora") or None,
            lora_multiplier=params.get("lora_alpha", 1.0),
            control_image=ctrl_image,
        )
        variation["status"] = "completed"
        print(f"[LoRA Dataset] Rerun done: {variation['label']}")
    except Exception as e:
        variation["status"] = "failed"
        variation["error"] = str(e)
        print(f"[LoRA Dataset] Rerun failed: {variation['label']} — {e}")

    _save_manifest(job)


# ============================================================================
# Export approved images
# ============================================================================

def export_approved(job_id: str, target_dir: Optional[str] = None) -> tuple:
    """Copy all approved generated images to target_dir (or job's approved/ if not given).
    Returns (count, resolved_dir_str).
    """
    job = dataset_jobs.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job_dir = Path(job["job_dir"])

    if target_dir:
        out_dir = Path(target_dir)
    else:
        out_dir = job_dir / "approved"

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for variation in job["variations"]:
        if variation.get("approved") is True and variation["status"] == "completed":
            src = job_dir / "generated" / variation["id"] / "generated.png"
            if src.exists():
                dst = out_dir / f"{variation['id']}.png"
                shutil.copy2(src, dst)
                count += 1

    return count, str(out_dir)
