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
# Variation Presets
# ============================================================================

# Variation structure: ids + labels only. Prompts are loaded from defaults.json
# at runtime via the `prompt_config` passed to _build_variation_list().
VARIATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "character": {
        "angles": {
            "label": "Angles",
            "variations": [
                {"id": "left_profile",        "label": "Left Profile"},
                {"id": "right_profile",       "label": "Right Profile"},
                {"id": "back_view",           "label": "Back View"},
                {"id": "three_quarter_left",  "label": "¾ Left"},
                {"id": "three_quarter_right", "label": "¾ Right"},
                {"id": "low_angle",           "label": "Low Angle"},
                {"id": "top_down",            "label": "Top-Down"},
            ],
        },
        "expressions": {
            "label": "Expressions",
            "variations": [
                {"id": "neutral",   "label": "Neutral"},
                {"id": "smile",     "label": "Smile"},
                {"id": "laughing",  "label": "Laughing"},
                {"id": "angry",     "label": "Angry"},
                {"id": "sad",       "label": "Sad"},
                {"id": "surprised", "label": "Surprised"},
                {"id": "thinking",  "label": "Thinking"},
            ],
        },
        "environments": {
            "label": "Environments",
            "variations": [
                {"id": "outdoor_park",   "label": "Outdoor Park"},
                {"id": "urban_street",   "label": "Urban Street"},
                {"id": "indoor_office",  "label": "Indoor Office"},
                {"id": "forest",         "label": "Forest"},
                {"id": "studio_neutral", "label": "Studio Neutral"},
            ],
        },
        "lighting": {
            "label": "Lighting",
            "variations": [
                {"id": "golden_hour",   "label": "Golden Hour"},
                {"id": "overcast",      "label": "Overcast"},
                {"id": "night",         "label": "Night"},
                {"id": "dramatic_side", "label": "Dramatic Side"},
                {"id": "backlit",       "label": "Backlit"},
            ],
        },
    },
    "product": {
        "angles": {
            "label": "Angles",
            "variations": [
                {"id": "45_left",        "label": "45° Left"},
                {"id": "45_right",       "label": "45° Right"},
                {"id": "top_down",       "label": "Top-Down"},
                {"id": "underside",      "label": "Underside"},
                {"id": "front_straight", "label": "Front Straight"},
            ],
        },
        "surfaces": {
            "label": "Surfaces",
            "variations": [
                {"id": "wooden_table",    "label": "Wooden Table"},
                {"id": "marble",          "label": "Marble"},
                {"id": "white_seamless",  "label": "White Seamless"},
                {"id": "outdoor_natural", "label": "Outdoor / Natural"},
            ],
        },
        "lighting": {
            "label": "Lighting",
            "variations": [
                {"id": "studio_3point", "label": "Studio 3-Point"},
                {"id": "single_key",    "label": "Single Key"},
                {"id": "soft_diffuse",  "label": "Soft Diffuse"},
                {"id": "dramatic",      "label": "Dramatic"},
            ],
        },
    },
    "environment": {
        "time_of_day": {
            "label": "Time of Day",
            "variations": [
                {"id": "dawn",   "label": "Dawn"},
                {"id": "noon",   "label": "Noon"},
                {"id": "sunset", "label": "Sunset"},
                {"id": "night",  "label": "Night"},
            ],
        },
        "weather": {
            "label": "Weather",
            "variations": [
                {"id": "clear",    "label": "Clear"},
                {"id": "overcast", "label": "Overcast"},
                {"id": "rain",     "label": "Rain"},
                {"id": "fog",      "label": "Fog"},
                {"id": "snow",     "label": "Snow"},
            ],
        },
        "season": {
            "label": "Season",
            "variations": [
                {"id": "spring", "label": "Spring"},
                {"id": "summer", "label": "Summer"},
                {"id": "autumn", "label": "Autumn"},
                {"id": "winter", "label": "Winter"},
            ],
        },
    },
}


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
) -> List[Dict]:
    """Resolve selected pack keys into an ordered variation list with seeds.
    Prompts are sourced from prompt_config (defaults.json lora_dataset.variation_prompts).
    """
    packs = VARIATION_PRESETS.get(subject_type, {})
    type_prompts = (prompt_config or {}).get(subject_type, {})
    variations = []
    idx = 1
    for pack_key in selected_packs:
        pack = packs.get(pack_key)
        if not pack:
            continue
        pack_prompts = type_prompts.get(pack_key, {})
        for v in pack["variations"]:
            prompt = pack_prompts.get(v["id"], f"Edit: {v['label']}")
            seed = base_seed if seed_strategy == "fixed" else random.randint(0, 2**32 - 1)
            variations.append({
                "id":       f"{idx:03d}_{v['id']}",
                "pack":     pack_key,
                "label":    v["label"],
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
    variations = _build_variation_list(subject_type, selected_packs, seed_strategy, base_seed, prompt_config)

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
                denoising_strength=params.get("denoising_strength", 0.65),
                lora=params.get("lora") or None,
                lora_multiplier=params.get("lora_alpha", 1.0),
                control_image=source_paths if source_paths else None,
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
# Export approved images
# ============================================================================

def export_approved(job_id: str) -> int:
    """Copy all approved generated images into the approved/ folder. Returns count."""
    job = dataset_jobs.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job_dir = Path(job["job_dir"])
    approved_dir = job_dir / "approved"
    approved_dir.mkdir(exist_ok=True)

    count = 0
    for variation in job["variations"]:
        if variation.get("approved") is True and variation["status"] == "completed":
            src = job_dir / "generated" / variation["id"] / "generated.png"
            if src.exists():
                dst = approved_dir / f"{variation['id']}.png"
                shutil.copy2(src, dst)
                count += 1

    return count
