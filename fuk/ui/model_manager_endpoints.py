"""
Model Manager endpoints — /api/models/manage/*

Backs the Utilities → Models panel, which replaces download_models.sh as the
primary way to acquire and enable models. The script stays for headless runs.

Two kinds of thing appear in the panel and they behave differently on purpose:

  Tools    Ollama, SeedVR2, TRELLIS, VGGT, SAM2, Depth-Anything-3, DSINE.
           Read-only status, green or red, exactly as they already report
           elsewhere in the app. They are installed by setup.sh, not from here,
           so the panel surfaces their existing availability checks and the
           hint each one already carries rather than inventing new ones.

  Models   Registry entries from models.json. Each gets a download button and
           an active checkbox. Unchecking one unregisters it: it disappears
           from the generation dropdowns but stays listed here, so turning it
           back on is one click and does not need a re-download.

Downloads prefer HuggingFace. DiffSynth defaults to ModelScope, but HF is
usually faster and steadier, and the per-component fallback in
fuk/utils/download_models.py means the ModelScope-only repos (the DiffSynth
converted Wan safetensors, LTX-2-Repackage, the PAI mirrors) still resolve.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SetEnabledRequest(BaseModel):
    enabled: bool


class DownloadRequest(BaseModel):
    # Omitted means "everything this model needs". Named components let the UI
    # retry only the pieces that failed rather than re-walking a 60GB model.
    patterns: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Download jobs
#
# Downloads run on a thread rather than in the request: these are tens of GB
# and the panel has to stay responsive. State is in-process and deliberately
# not persisted — a job that dies with the server is a job the user restarts,
# and every download is resumable because already-present files are skipped.
# ---------------------------------------------------------------------------

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(model_key: str, targets: List[tuple]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "model": model_key,
            "status": "running",
            "started_at": time.time(),
            "total": len(targets),
            "completed": 0,
            "current": None,
            "failed": [],
            "error": None,
        }
    return job_id


def _update_job(job_id: str, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

# Pipelines whose entries are DiffSynth models with downloadable components.
# threed entries are listed too, but their weights arrive through
# install_trellis_env.sh or on first use, so they carry no components.
_CATEGORY_BY_PIPELINE = {
    "qwen": "image",
    "flux2": "image",
    "krea2": "image",
    "wan": "video",
    "ltx2": "video",
    "minimax_h3": "video",
    "threed": "threed",
}


def _iter_models(models_config: dict):
    """Yield (key, entry) for real model entries, skipping comment keys."""
    for key, entry in models_config.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if "pipeline" not in entry:
            continue
        yield key, entry


def _entry_targets(entry: dict) -> List[tuple]:
    """Every (label, model_id, pattern) a model needs on disk.

    Mirrors download_all_models in fuk/utils/download_models.py — components
    plus the tokenizer/processor/lora/stage2_lora sections — so the panel and
    the CLI downloader agree on what "complete" means.
    """
    base = entry["model_id"]
    targets = [
        (f"Component {i}", c.get("model_id", base), c["pattern"])
        for i, c in enumerate(entry.get("components", []), 1)
    ]
    for section, label in (("tokenizer", "Tokenizer"),
                           ("processor", "Processor"),
                           ("lora", "LoRA"),
                           ("stage2_lora", "Stage-2 LoRA")):
        if section in entry:
            spec = entry[section]
            targets.append((label, spec.get("model_id", base), spec["pattern"]))
    return targets


def _expand_pattern(pattern: str) -> str:
    """Match DiffSynth's ModelConfig.parse_original_file_pattern."""
    if pattern in (None, "", "./"):
        return "*"
    if pattern.endswith("/"):
        return pattern + "*"
    return pattern


def _target_status(models_root: Path, model_id: str, pattern: str) -> dict:
    """Whether one download target is present, and how big it is on disk."""
    root = models_root / model_id
    matches = glob.glob(_expand_pattern(pattern), root_dir=str(root)) if root.exists() else []
    size = 0
    for m in matches:
        p = root / m
        if p.is_file():
            size += p.stat().st_size
        elif p.is_dir():
            size += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return {"present": bool(matches), "files": len(matches), "size_bytes": size}


def _model_status(models_root: Path, key: str, entry: dict) -> dict:
    """Full panel row for one registry entry."""
    targets = _entry_targets(entry)
    parts, present_count, total_bytes = [], 0, 0
    for label, model_id, pattern in targets:
        st = _target_status(models_root, model_id, pattern)
        present_count += bool(st["present"])
        total_bytes += st["size_bytes"]
        parts.append({
            "label": label,
            "model_id": model_id,
            "pattern": pattern,
            "present": st["present"],
            "size_bytes": st["size_bytes"],
            "install_path": str(models_root / model_id),
        })

    repos = sorted({model_id for _, model_id, _ in targets})
    pipeline = entry.get("pipeline", "")

    return {
        "key": key,
        "name": entry.get("name", key),
        "description": entry.get("description", ""),
        "pipeline": pipeline,
        "category": entry.get("category") or _CATEGORY_BY_PIPELINE.get(pipeline, "other"),
        "aliases": entry.get("aliases", []),
        # Absent means enabled: every model predating this panel stays visible.
        "enabled": entry.get("enabled", True),
        "downloaded": bool(targets) and present_count == len(targets),
        "partial": 0 < present_count < len(targets),
        "components_present": present_count,
        "components_total": len(targets),
        "size_on_disk_bytes": total_bytes,
        # Declared in the registry where known; the panel shows on-disk size
        # once something is present, so this only has to cover the "not yet
        # downloaded, how big is this?" case.
        "size_gb_estimate": entry.get("size_gb"),
        "components": parts,
        "repos": [
            {
                "model_id": mid,
                "huggingface_url": f"https://huggingface.co/{mid}",
                "modelscope_url": f"https://modelscope.cn/models/{mid}",
                "install_path": str(models_root / mid),
            }
            for mid in repos
        ],
        # threed entries have no components — their weights come from
        # install_trellis_env.sh or on first use.
        "downloadable": bool(targets),
        "notes": entry.get("notes"),
    }


# ---------------------------------------------------------------------------
# Tool status
# ---------------------------------------------------------------------------

def _vendor_dir(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "vendor" / name


def _tool_status(generation_backend, log) -> List[dict]:
    """Aggregate the availability checks the app already performs.

    Each tool reports through its own existing check so the panel cannot drift
    from what the feature itself believes. Failures are caught per tool: one
    unreachable service should grey out its own row, not blank the panel.
    """
    tools: List[dict] = []

    def add(key, name, description, checker, hint=None, docs_url=None):
        try:
            state = checker()
        except Exception as e:
            state = {"available": False, "missing": [f"{type(e).__name__}: {e}"]}
        tools.append({
            "key": key,
            "name": name,
            "description": description,
            "available": bool(state.get("available")),
            "missing": state.get("missing") or [],
            "hint": state.get("hint") or hint,
            "detail": state.get("detail"),
            "docs_url": docs_url,
        })

    # Ollama — powers Describe & Tag and the prompt LLM features.
    def _ollama():
        from llm_endpoints import _ollama_health
        h = _ollama_health()
        if not h.get("available"):
            return {"available": False,
                    "missing": ["Ollama server not reachable"],
                    "hint": "Start it with `ollama serve`"}
        if not h.get("model_present"):
            return {"available": False,
                    "missing": [f"model '{h.get('model')}' not pulled"],
                    "hint": f"ollama pull {h.get('model')}",
                    "detail": f"server up, {len(h.get('models', []))} other models present"}
        return {"available": True, "detail": f"model {h.get('model')} ready"}

    add("ollama", "Ollama — Describe & Tag",
        "Vision-language model that writes cinematographer-style descriptions and prompt tags.",
        _ollama, docs_url="https://ollama.com")

    # SeedVR2 — temporal video restoration in Postprocess.
    def _seedvr2():
        from core import seedvr2_backend
        st = seedvr2_backend.availability()
        n = len(st.get("downloaded") or [])
        return {**st, "detail": f"{n} weight variant(s) present" if n else
                                "engine ready, weights download on first use"}

    add("seedvr2", "SeedVR2 — video restoration",
        "Upscales a sequence as a sequence, so it does not flicker like per-frame models.",
        _seedvr2, docs_url="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler")

    # 3D reconstruction.
    def _trellis():
        from core.threed import trellis_backend
        return trellis_backend.availability()

    add("trellis", "TRELLIS — single-image 3D",
        "Structured 3D latents; one image to a mesh in seconds. Runs in its own environment.",
        _trellis, hint="bash install_trellis_env.sh",
        docs_url="https://github.com/microsoft/TRELLIS")

    def _vggt():
        from core.threed import vggt_backend
        ok = vggt_backend.vendor_available()
        return {"available": ok,
                "missing": [] if ok else ["VGGT source (fuk/vendor/VGGT)"],
                "hint": None if ok else "Re-run setup.sh to vendor it"}

    add("vggt", "VGGT — multi-view 3D",
        "Multi-view reconstruction. Weights download from HuggingFace on first use.",
        _vggt, docs_url="https://github.com/facebookresearch/vggt")

    # Preprocessor vendor trees. These are plain source checkouts, so presence
    # of the directory is the whole check.
    def _vendor_check(dirname, extra: Optional[Callable[[Path], List[str]]] = None):
        def check():
            d = _vendor_dir(dirname)
            missing = [] if d.exists() else [f"fuk/vendor/{dirname}"]
            if not missing and extra:
                missing = extra(d)
            return {"available": not missing, "missing": missing,
                    "hint": None if not missing else "Re-run setup.sh to vendor it",
                    "detail": str(d) if d.exists() else None}
        return check

    add("sam2", "SAM2 — segmentation",
        "Segment Anything 2, behind the mask and cryptomatte preprocessors.",
        _vendor_check(
            "segment-anything-2",
            lambda d: [] if list((d / "checkpoints").glob("*.pt")) else ["SAM2 checkpoints"],
        ),
        docs_url="https://github.com/facebookresearch/segment-anything-2")

    add("depth_anything_3", "Depth Anything 3 — depth",
        "Depth maps for control passes and the depth AOV.",
        _vendor_check("Depth-Anything-3"),
        docs_url="https://github.com/ByteDance-Seed/Depth-Anything-3")

    add("dsine", "DSINE — surface normals",
        "Surface-normal estimation. Note: non-commercial weights.",
        _vendor_check("DSINE"),
        docs_url="https://github.com/baegwangbin/DSINE")

    return tools


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def setup_model_manager_routes(
    app,
    *,
    config_dir: Path,
    generation_backend,
    log,
):
    """Register /api/models/manage/* routes."""

    def _models_path() -> Path:
        return Path(config_dir) / "models.json"

    def _load_models() -> dict:
        with open(_models_path()) as f:
            return json.load(f)

    def _models_root() -> Path:
        with open(Path(config_dir) / "defaults.json") as f:
            return Path(json.load(f).get("models_root", "./models")).expanduser()

    @app.get("/api/models/manage")
    async def list_managed_models():
        """Everything the Models panel renders: tools, models, and where they live."""
        models_config = _load_models()
        root = _models_root()

        models = [_model_status(root, k, e) for k, e in _iter_models(models_config)]
        # Group ordering is the UI's business, but a stable sort here keeps rows
        # from jumping around between polls.
        models.sort(key=lambda m: (m["category"], m["key"]))

        return {
            "models_root": str(root),
            "download_source": os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface"),
            "tools": _tool_status(generation_backend, log),
            "models": models,
        }

    @app.post("/api/models/manage/{model_key}/enabled")
    async def set_model_enabled(model_key: str, request: SetEnabledRequest):
        """Register or unregister a model.

        Writes `enabled` into models.json and reloads the backend registry, so
        the generation dropdowns reflect the change without a server restart.
        """
        models_config = _load_models()
        entry = models_config.get(model_key)
        if not isinstance(entry, dict) or "pipeline" not in entry:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'")

        entry["enabled"] = bool(request.enabled)

        path = _models_path()
        # Write via a temp file in the same directory so an interrupted write
        # cannot leave models.json truncated — losing it would take every model
        # with it, including the ones that were working.
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(models_config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)

        try:
            generation_backend.reload_config()
        except Exception as e:
            log.warning("ModelManager", f"Config reload after toggle failed: {e}")

        log.info("ModelManager",
                 f"{model_key} {'enabled' if request.enabled else 'disabled'}")
        return {"success": True, "key": model_key, "enabled": entry["enabled"]}

    @app.post("/api/models/manage/{model_key}/download")
    async def download_model(model_key: str, request: DownloadRequest):
        """Start a background download for one model. Returns a job id to poll."""
        models_config = _load_models()
        entry = models_config.get(model_key)
        if not isinstance(entry, dict) or "pipeline" not in entry:
            raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'")

        targets = _entry_targets(entry)
        if request.patterns:
            wanted = set(request.patterns)
            targets = [t for t in targets if t[2] in wanted]
        if not targets:
            raise HTTPException(
                status_code=400,
                detail=f"'{model_key}' has no downloadable components. "
                       f"3D models acquire weights via install_trellis_env.sh or on first use.",
            )

        root = _models_root()
        job_id = _new_job(model_key, targets)

        def worker():
            # download_models lives in fuk/utils and imports diffsynth directly.
            utils_dir = str(Path(__file__).resolve().parent.parent / "utils")
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(root)

            try:
                from download_models import download_model_component
            except Exception as e:
                _update_job(job_id, status="failed", error=f"downloader unavailable: {e}")
                return

            done, failed = 0, []
            for label, model_id, pattern in targets:
                _update_job(job_id, current=f"{label}: {model_id} → {pattern}")
                try:
                    # HuggingFace first — faster and steadier — with a
                    # per-component fallback to ModelScope inside the helper for
                    # the repos HF does not carry. Passed explicitly rather than
                    # via DIFFSYNTH_DOWNLOAD_SOURCE, which is process-global and
                    # would leak out of this worker thread.
                    path = download_model_component(
                        model_id, pattern, component_name=label, prefer="huggingface")
                except Exception as e:
                    path = None
                    log.warning("ModelManager", f"{model_key} {label} raised: {e}")
                # Falsy covers both None and the empty-glob case: a source can
                # resolve without producing files, and that is not a success.
                if not path:
                    failed.append({"label": label, "model_id": model_id, "pattern": pattern})
                else:
                    done += 1
                _update_job(job_id, completed=done, failed=failed)

            _update_job(
                job_id,
                status="completed" if not failed else "partial",
                current=None,
                finished_at=time.time(),
            )
            log.info("ModelManager",
                     f"{model_key} download finished: {done}/{len(targets)}"
                     + (f", {len(failed)} failed" if failed else ""))

        threading.Thread(target=worker, name=f"dl-{model_key}", daemon=True).start()
        log.info("ModelManager", f"{model_key} download started ({len(targets)} targets)")
        return {"success": True, "job_id": job_id, "total": len(targets)}

    @app.get("/api/models/manage/jobs/{job_id}")
    async def download_job_status(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
        return job

    @app.get("/api/models/manage/jobs")
    async def download_jobs():
        """All jobs this process knows about, newest first."""
        with _jobs_lock:
            jobs = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
        return {"jobs": jobs}
