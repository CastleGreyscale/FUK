"""
TRELLIS single-image reconstruction (Tier 1) — host side.

Runs in FUK's main venv but executes nothing itself: it writes a job file,
launches trellis_worker.py under the isolated environment's interpreter, and
streams that subprocess's output into FUK's console panel.

The isolation is deliberate. TRELLIS needs torch 2.6/cu126 plus spconv,
nvdiffrast and diff_gaussian_rasterization; the main venv runs torch
2.9/cu130 for Qwen and Wan. Installing one into the other breaks generation,
so they stay apart and talk over a subprocess boundary.

Set up the environment with:
    bash fuk/core/threed/install_trellis_env.sh

Source: https://github.com/microsoft/TRELLIS   (Microsoft, MIT)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .mesh_export import _log
from .trellis_worker import RESULT_MARKER

_VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor"
_TRELLIS_SRC = _VENDOR_DIR / "TRELLIS"
_ENV_ROOT = _VENDOR_DIR / "TRELLIS_ENV"
_ENV_PYTHON = _ENV_ROOT / "env" / "bin" / "python"
# Written by install_trellis_env.sh only after every import the mesh path
# needs has been verified inside the isolated env. An interpreter existing
# proves nothing — a half-finished build leaves one behind.
_READY_MARKER = _ENV_ROOT / "READY"
_WORKER = Path(__file__).resolve().parent / "trellis_worker.py"

INSTALL_HINT = "Run: bash fuk/core/threed/install_trellis_env.sh"


def env_python() -> Optional[Path]:
    """Interpreter for the isolated environment, or None if not installed."""
    override = os.environ.get("FUK_TRELLIS_PYTHON")
    if override and Path(override).exists():
        return Path(override)
    return _ENV_PYTHON if _ENV_PYTHON.exists() else None


def availability() -> Dict[str, Any]:
    """Report whether TRELLIS can run, and what's missing if it can't."""
    missing = []
    if not (_TRELLIS_SRC / "trellis" / "pipelines").exists():
        missing.append("TRELLIS source (vendor/TRELLIS)")
    if env_python() is None:
        missing.append("isolated environment (vendor/TRELLIS_ENV/env)")
    elif not _READY_MARKER.exists() and not os.environ.get("FUK_TRELLIS_PYTHON"):
        missing.append("CUDA extensions (environment built but incomplete)")
    return {
        "available": not missing,
        "missing": missing,
        "hint": INSTALL_HINT if missing else None,
    }


def resolve_weights(entry: dict) -> str:
    configured = entry.get("path")
    if configured:
        expanded = Path(configured).expanduser()
        if expanded.exists():
            return str(expanded)
    return entry.get("model_id", "microsoft/TRELLIS-image-large")


def reconstruct(
    image: str,
    output_dir: Path,
    entry: dict,
    export_formats: List[str],
    seed: int = 42,
    steps: int = 12,
    cfg_strength: float = 7.5,
    simplify: float = 0.95,
    texture_size: int = 1024,
    fill_holes: bool = True,
    progress_callback=None,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Reconstruct a mesh from a single image via the isolated TRELLIS env."""
    status = availability()
    if not status["available"]:
        raise RuntimeError(
            "TRELLIS is not installed: " + ", ".join(status["missing"]) + f"\n  {INSTALL_HINT}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "image": str(image),
        "output_dir": str(output_dir),
        "weights": resolve_weights(entry),
        "trellis_src": str(_TRELLIS_SRC),
        "export_formats": export_formats,
        "seed": seed,
        "steps": steps,
        "cfg_strength": cfg_strength,
        "simplify": simplify,
        "texture_size": texture_size,
        "fill_holes": fill_holes,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(job, fh)
        job_path = Path(fh.name)

    python = env_python()
    _log(f"Launching TRELLIS worker under {python}")
    if progress_callback:
        progress_callback(0.10, "Starting TRELLIS (isolated environment)")

    result: Optional[Dict[str, Any]] = None
    tail: List[str] = []
    timed_out = False

    # Progress is coarse — the worker's phases aren't individually
    # instrumented, so each recognised milestone bumps the bar. Mesh
    # postprocessing is the long tail: to_glb simplifies, fills holes with a
    # 1000-view render pass and bakes a texture, which on a large SLAT mesh
    # runs for minutes. Its tqdm output is relayed so the console keeps
    # moving; without that this stage looks like a hang.
    milestones = [
        ("pipeline loaded", 0.30, "Weights loaded"),
        ("running SLAT", 0.40, "Generating structured latents"),
        ("generation complete", 0.60, "Decoding geometry"),
        ("raw mesh:", 0.65, "Mesh decoded"),
        ("postprocessing mesh", 0.70, "Postprocessing mesh (this is the slow part)"),
        ("Simplifying", 0.75, "Simplifying mesh"),
        ("Filling holes", 0.80, "Filling holes"),
        ("Parametrizing", 0.86, "Unwrapping UVs"),
        ("Baking texture", 0.90, "Baking texture"),
        ("wrote mesh", 0.96, "Writing outputs"),
    ]

    process = None
    try:
        process = subprocess.Popen(
            [str(python), str(_WORKER), str(job_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # The worker must not inherit the main venv's paths, or it will
            # import torch 2.9 from site-packages and fail on the extensions.
            env={
                **{k: v for k, v in os.environ.items()
                   if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")},
                "PYTHONUNBUFFERED": "1",
            },
        )

        # A wall-clock watchdog, because `process.wait(timeout=…)` after the
        # read loop can never fire: if the worker wedges, its stdout never
        # reaches EOF and the loop below blocks forever. Killing the child is
        # what unblocks the read.
        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            _log(f"TRELLIS worker exceeded {timeout}s — terminating", "error")
            try:
                process.kill()
            except Exception:
                pass

        watchdog = threading.Timer(timeout, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()

        try:
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith(RESULT_MARKER):
                    result = json.loads(line[len(RESULT_MARKER):].strip())
                    continue
                # Relayed so it lands in the GenerationModal console like any
                # other FUK task.
                print(line, flush=True)
                tail.append(line)
                del tail[:-40]
                if progress_callback:
                    for needle, value, label in milestones:
                        if needle in line:
                            progress_callback(value, label)
                            break

            code = process.wait()
        finally:
            watchdog.cancel()
    finally:
        job_path.unlink(missing_ok=True)
        if process and process.poll() is None:
            process.kill()

    if timed_out:
        raise RuntimeError(
            f"TRELLIS timed out after {timeout}s.\n"
            f"Mesh postprocessing is the usual cause — a dense SLAT mesh can spend "
            f"minutes filling holes. Turn off 'Fill holes' or raise 'Simplify', "
            f"then try again.\n" + "\n".join(tail[-10:])
        )
    if code != 0:
        detail = "\n".join(tail[-15:])
        raise RuntimeError(f"TRELLIS worker exited with code {code}\n{detail}")
    if result is None:
        raise RuntimeError("TRELLIS worker produced no result")

    result.setdefault("input_count", 1)
    result.setdefault("point_count", 0)
    return result
