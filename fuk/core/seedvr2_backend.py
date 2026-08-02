"""
SeedVR2 video restoration / upscaling — host side.

Runs in FUK's main venv but executes nothing itself: it writes a job file,
launches seedvr2_worker.py under the isolated environment's interpreter, and
streams that subprocess's output into FUK's console panel.

Same isolation rationale as TRELLIS. SeedVR2 wants flash-attn and apex built
against its own torch; the main venv runs torch 2.9/cu130 for Qwen and Wan.
They stay apart and talk over a subprocess boundary.

Why this exists at all
    The previous video upscale path ran Real-ESRGAN frame by frame. ESRGAN has
    no temporal model, so each frame is enhanced independently and the result
    flickers — detail crawls between frames on exactly the kind of content Wan
    produces. SeedVR2 restores the sequence as a sequence, which is the whole
    point of using it.

Set up the environment with:
    bash fuk/core/install_seedvr2_env.sh

Source:  https://github.com/ByteDance-Seed/SeedVR   (ByteDance, Apache 2.0)
Weights: ByteDance-Seed/SeedVR2-3B, SeedVR2-7B        (Apache 2.0)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .seedvr2_worker import RESULT_MARKER

_CORE_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _CORE_DIR.parent / "vendor"
_SEEDVR_SRC = _VENDOR_DIR / "SeedVR"
_ENV_ROOT = _VENDOR_DIR / "SEEDVR2_ENV"
_ENV_PYTHON = _ENV_ROOT / "env" / "bin" / "python"
# Written by install_seedvr2_env.sh only after every import the inference path
# needs has been verified inside the isolated env. An interpreter existing
# proves nothing — a half-finished build leaves one behind.
_READY_MARKER = _ENV_ROOT / "READY"
_WORKER = _CORE_DIR / "seedvr2_worker.py"

DEFAULT_WEIGHTS_DIR = Path.home() / "ai" / "models" / "seedvr2"

INSTALL_HINT = "Run: bash fuk/core/install_seedvr2_env.sh"

# 3B is the only variant that fits a 24GB card with room for the VAE. 7B is
# offered for larger cards; it is not the default for a reason.
VARIANTS = {
    "seedvr2_3b": {
        "label": "SeedVR2 3B",
        "checkpoint": "seedvr2_ema_3b.pth",
        "script": "projects/inference_seedvr2_3b.py",
        "repo": "ByteDance-Seed/SeedVR2-3B",
        "min_vram_gb": 18,
    },
    "seedvr2_7b": {
        "label": "SeedVR2 7B",
        "checkpoint": "seedvr2_ema_7b.pth",
        "script": "projects/inference_seedvr2_7b.py",
        "repo": "ByteDance-Seed/SeedVR2-7B",
        "min_vram_gb": 40,
    },
}

DEFAULT_VARIANT = "seedvr2_3b"


def env_python() -> Optional[Path]:
    """Interpreter for the isolated environment, or None if not installed."""
    override = os.environ.get("FUK_SEEDVR2_PYTHON")
    if override and Path(override).exists():
        return Path(override)
    return _ENV_PYTHON if _ENV_PYTHON.exists() else None


def weights_dir() -> Path:
    return Path(os.environ.get("FUK_SEEDVR2_WEIGHTS", DEFAULT_WEIGHTS_DIR)).expanduser()


def availability() -> Dict[str, Any]:
    """Report whether SeedVR2 can run, and what's missing if it can't."""
    missing: List[str] = []

    if not (_SEEDVR_SRC / "projects").exists():
        missing.append("SeedVR source (vendor/SeedVR)")
    if env_python() is None:
        missing.append("isolated environment (vendor/SEEDVR2_ENV/env)")
    elif not _READY_MARKER.exists() and not os.environ.get("FUK_SEEDVR2_PYTHON"):
        missing.append("environment built but incomplete")

    installed = [
        key for key, spec in VARIANTS.items()
        if (weights_dir() / spec["checkpoint"]).exists()
    ]
    if not installed:
        missing.append(f"model weights ({weights_dir()})")

    return {
        "available": not missing,
        "missing": missing,
        "variants": installed,
        "weights_dir": str(weights_dir()),
        "hint": INSTALL_HINT if missing else None,
    }


def _target_resolution(width: int, height: int, scale: int, cap: int) -> tuple[int, int]:
    """
    SeedVR2 takes an absolute output resolution, not a scale factor.

    Dimensions are forced even — the VAE downsamples by 2 and an odd dimension
    fails at the encode step rather than at argument parsing, which is a
    confusing place to discover a typo.
    """
    tw, th = width * scale, height * scale

    longest = max(tw, th)
    if cap and longest > cap:
        ratio = cap / longest
        tw, th = int(tw * ratio), int(th * ratio)

    return (tw - tw % 2, th - th % 2)


def upscale_video(
    input_path: Path,
    output_path: Path,
    scale: int = 2,
    variant: str = DEFAULT_VARIANT,
    frame_window: int = 0,
    seed: int = 42,
    resolution_cap: int = 1920,
    progress_callback: Optional[Callable] = None,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """
    Restore and upscale a video via the isolated SeedVR2 environment.

    frame_window
        Frames processed per forward pass. 0 lets the worker choose from
        available VRAM. Larger windows give better temporal coherence and cost
        more memory; this is the knob that decides whether a clip fits.
    resolution_cap
        Longest output edge. SeedVR2's attention window scales with resolution,
        so an uncapped 4x on a 1080p source will OOM a 24GB card.
    """
    status = availability()
    if not status["available"]:
        raise RuntimeError(
            "SeedVR2 is not installed: " + ", ".join(status["missing"])
            + f"\n  {INSTALL_HINT}"
        )

    if variant not in VARIANTS:
        raise ValueError(f"Unknown SeedVR2 variant: {variant}")
    if variant not in status["variants"]:
        raise RuntimeError(
            f"{VARIANTS[variant]['label']} weights not found in {weights_dir()}.\n"
            f"  Download: huggingface-cli download {VARIANTS[variant]['repo']}"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job = {
        "input": str(input_path),
        "output": str(output_path),
        "seedvr_src": str(_SEEDVR_SRC),
        "weights_dir": str(weights_dir()),
        "variant": variant,
        "checkpoint": VARIANTS[variant]["checkpoint"],
        "script": VARIANTS[variant]["script"],
        "scale": int(scale),
        "frame_window": int(frame_window),
        "resolution_cap": int(resolution_cap),
        "seed": int(seed),
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(job, fh)
        job_path = Path(fh.name)

    python = env_python()
    print(f"[SeedVR2] Launching worker under {python}", flush=True)
    if progress_callback:
        progress_callback(0.05, "Starting SeedVR2 (isolated environment)")

    result: Optional[Dict[str, Any]] = None
    tail: List[str] = []
    timed_out = False

    # Coarse milestones, same approach as the TRELLIS host: the worker's phases
    # aren't individually instrumented, so each recognised line bumps the bar.
    # Per-chunk progress is emitted by the worker itself and parsed below.
    milestones = [
        ("probing input", 0.08, "Reading source video"),
        ("extracting frames", 0.12, "Extracting frames"),
        ("loading weights", 0.18, "Loading SeedVR2 weights"),
        ("runner ready", 0.25, "Model loaded"),
        ("encoding output", 0.92, "Encoding output video"),
        ("wrote", 0.97, "Writing result"),
    ]

    process = None
    code = 1
    try:
        process = subprocess.Popen(
            [str(python), str(_WORKER), str(job_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # The worker must not inherit the main venv's paths, or it will
            # import the wrong torch and fail at the flash-attn import.
            env={
                **{k: v for k, v in os.environ.items()
                   if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")},
                "PYTHONUNBUFFERED": "1",
            },
        )

        # Wall-clock watchdog. `process.wait(timeout=…)` after the read loop
        # can never fire: if the worker wedges, its stdout never reaches EOF and
        # the loop blocks forever. Killing the child is what unblocks the read.
        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            print(f"[SeedVR2] worker exceeded {timeout}s — terminating", flush=True)
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

                print(line, flush=True)
                tail.append(line)
                del tail[:-40]

                if not progress_callback:
                    continue

                # Chunk lines carry real progress and dominate the runtime, so
                # they get the 0.25–0.90 band; milestones only fill the ends.
                if "chunk " in line and "/" in line:
                    try:
                        frag = line.split("chunk ", 1)[1].split()[0]
                        done, total = (int(x) for x in frag.split("/"))
                        progress_callback(
                            0.25 + 0.65 * (done / max(total, 1)),
                            f"Restoring chunk {done}/{total}",
                        )
                        continue
                    except (ValueError, IndexError):
                        pass

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
            f"SeedVR2 timed out after {timeout}s.\n"
            f"Long clips at high resolution are the usual cause. Lower the scale, "
            f"reduce the resolution cap, or split the clip.\n" + "\n".join(tail[-10:])
        )
    if code != 0:
        detail = "\n".join(tail[-15:])
        # OOM is the failure people will actually hit, so name the fix rather
        # than leaving a CUDA traceback as the whole message.
        if any("out of memory" in t.lower() for t in tail):
            raise RuntimeError(
                "SeedVR2 ran out of VRAM.\n"
                "  Reduce Scale, lower the resolution cap, or set a smaller "
                "frame window in Postprocess settings.\n" + detail
            )
        raise RuntimeError(f"SeedVR2 worker exited with code {code}\n{detail}")
    if result is None:
        raise RuntimeError("SeedVR2 worker produced no result")

    if progress_callback:
        progress_callback(1.0, "Complete")

    result.setdefault("method", "seedvr2")
    return result
