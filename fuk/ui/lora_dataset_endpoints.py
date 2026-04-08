"""
LoRA Dataset Endpoints
FastAPI router for dataset creation, streaming, curation, and export.

Register with: setup_dataset_routes(app, generation_backend, datasets_root)
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lora_dataset_manager import (
    VARIATION_PRESETS,
    dataset_jobs,
    create_dataset_job,
    run_dataset_job,
    export_approved,
)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# Injected by setup_dataset_routes
_generation_backend = None
_datasets_root: Optional[Path] = None
_prompt_config: Optional[dict] = None   # defaults["lora_dataset"]["variation_prompts"]
_resolve_path = None                    # resolve_input_path fn from the main server


# ============================================================================
# Request / Response models
# ============================================================================

class CreateDatasetRequest(BaseModel):
    subject_name: str
    subject_type: str                    # character | product | environment
    source_paths: List[str]
    selected_packs: List[str]
    params: dict = {}


class ApproveRequest(BaseModel):
    variation_id: str
    approved: bool


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/presets")
async def get_presets():
    """Return the full variation preset library."""
    return VARIATION_PRESETS


@router.get("/list")
async def list_jobs():
    """Return all jobs with summary status."""
    summaries = []
    for job in dataset_jobs.values():
        total = len(job["variations"])
        done = sum(1 for v in job["variations"] if v["status"] in ("completed", "failed"))
        summaries.append({
            "job_id":       job["job_id"],
            "subject_name": job["subject_name"],
            "subject_type": job["subject_type"],
            "created":      job["created"],
            "status":       job["status"],
            "progress":     job["progress"],
            "total":        total,
            "completed":    done,
        })
    summaries.sort(key=lambda j: j["created"], reverse=True)
    return summaries


@router.post("/create")
async def create_job(request: CreateDatasetRequest, background_tasks: BackgroundTasks):
    """Create and start a new dataset generation job."""
    if not request.source_paths:
        raise HTTPException(status_code=400, detail="At least one source image is required")
    if not request.selected_packs:
        raise HTTPException(status_code=400, detail="At least one variation pack must be selected")
    if request.subject_type not in VARIATION_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown subject_type: {request.subject_type!r}")

    # Resolve API-relative paths (e.g. api/project/cache/...) to absolute filesystem paths
    resolved_sources = []
    for p in request.source_paths:
        resolved = _resolve_path(p) if _resolve_path else None
        if resolved and resolved.exists():
            resolved_sources.append(str(resolved))
        else:
            resolved_sources.append(p)  # pass through; manager will skip if not found

    job_id = create_dataset_job(
        datasets_root=_datasets_root,
        subject_name=request.subject_name,
        subject_type=request.subject_type,
        source_paths=resolved_sources,
        selected_packs=request.selected_packs,
        params=request.params,
        prompt_config=_prompt_config,
    )

    background_tasks.add_task(run_dataset_job, job_id, _generation_backend)
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream — sends job state updates until the job finishes."""
    if job_id not in dataset_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    async def event_generator():
        last_snapshot = None
        idle_ticks = 0

        while True:
            job = dataset_jobs.get(job_id)
            if not job:
                yield _sse({"status": "not_found"})
                break

            # Build a compact snapshot for diffing
            snapshot = {
                "status":      job["status"],
                "progress":    round(job["progress"], 4),
                "current_idx": job["current_idx"],
                "total":       len(job["variations"]),
                "variations":  [
                    {k: v[k] for k in ("id", "label", "status", "approved", "error")}
                    for v in job["variations"]
                ],
            }
            if job["status"] in ("running", "queued"):
                cur = job["variations"][job["current_idx"]] if job["current_idx"] < len(job["variations"]) else None
                snapshot["current_label"] = cur["label"] if cur else ""

            if snapshot != last_snapshot:
                yield _sse(snapshot)
                last_snapshot = snapshot
                idle_ticks = 0
            else:
                idle_ticks += 1

            # Terminal states — send final event then stop
            if job["status"] in ("complete", "failed", "cancelled"):
                break

            # Timeout after ~5 minutes of silence
            if idle_ticks > 600:
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Return full job detail including manifest."""
    job = dataset_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_dir = Path(job["job_dir"])

    # Enrich variations with image URL if generated
    enriched_variations = []
    for v in job["variations"]:
        img_path = job_dir / "generated" / v["id"] / "generated.png"
        enriched_variations.append({
            **v,
            "image_url": str(img_path) if img_path.exists() else None,
        })

    return {
        **{k: job[k] for k in ("job_id", "subject_name", "subject_type", "created", "status", "progress", "params")},
        "source_paths": job["source_paths"],
        "variations":   enriched_variations,
        "total":        len(job["variations"]),
        "approved_count": sum(1 for v in job["variations"] if v.get("approved") is True),
    }


@router.post("/{job_id}/approve")
async def approve_variation(job_id: str, request: ApproveRequest):
    """Set approved state for a single variation."""
    job = dataset_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    for v in job["variations"]:
        if v["id"] == request.variation_id:
            v["approved"] = request.approved
            from lora_dataset_manager import _save_manifest
            _save_manifest(job)
            return {"ok": True, "variation_id": request.variation_id, "approved": request.approved}

    raise HTTPException(status_code=404, detail=f"Variation {request.variation_id!r} not found")


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Signal the running job to stop after the current variation completes."""
    job = dataset_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] not in ("queued", "running"):
        return {"ok": False, "detail": f"Job is already {job['status']}"}
    job["cancelled"] = True
    return {"ok": True}


class ExportRequest(BaseModel):
    target_dir: Optional[str] = None   # if omitted, defaults to job's approved/ subfolder


@router.post("/{job_id}/export")
async def export_job(job_id: str, request: ExportRequest = ExportRequest()):
    """Copy all approved images to target_dir (or job's approved/ if not specified)."""
    if job_id not in dataset_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        count, out_dir = export_approved(job_id, request.target_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "exported": count, "path": out_dir}


# ============================================================================
# SSE helper
# ============================================================================

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ============================================================================
# Setup
# ============================================================================

def setup_dataset_routes(app, generation_backend, datasets_root: Path, defaults: dict = None, resolve_input_path=None):
    """Register the dataset router and inject dependencies."""
    global _generation_backend, _datasets_root, _prompt_config, _resolve_path
    _generation_backend = generation_backend
    _datasets_root = datasets_root
    _prompt_config = (defaults or {}).get("lora_dataset", {}).get("variation_prompts")
    _resolve_path = resolve_input_path
    _datasets_root.mkdir(parents=True, exist_ok=True)
    app.include_router(router)
