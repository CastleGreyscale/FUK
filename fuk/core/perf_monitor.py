"""
Lightweight performance watchdog for FUK.

Feeds on the wall-clock timings already measured by the `[timing]` log lines
(pipeline loads, per-step denoise speed), keeps a small rolling history per
metric on disk, and logs a loud warning when a new sample is much slower than
the recent median. Catches slow-burn regressions (disk cold-data aging, VRAM
pressure, thermal throttling) when they cross a threshold instead of months
later. Benchmarks are exposed via GET /api/system/perf.

Standalone on purpose: stdlib only, and it never raises — a watchdog failure
must not break generation.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

_HISTORY_PATH = Path(__file__).resolve().parent.parent / "ui" / "data" / "perf_history.json"
_MAX_SAMPLES = 40      # rolling window per metric
_MIN_BASELINE = 3      # prior samples needed before judging
_WARN_RATIO = 1.5      # slower than 1.5x the median -> warn
_WARN_FLOOR_S = 1.0    # ...and at least 1s over the median (kills noise)

_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(_HISTORY_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _HISTORY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(_HISTORY_PATH)


def _median(values):
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def record_timing(metric: str, seconds: float, context: str = "") -> None:
    """Record one timing sample and warn if it's far off the recent median."""
    try:
        with _lock:
            data = _load()
            samples = data.get(metric, [])
            prior = [s["v"] for s in samples]
            entry = {"t": datetime.now().isoformat(timespec="seconds"),
                     "v": round(float(seconds), 3)}
            if context:
                entry["ctx"] = context
            samples.append(entry)
            data[metric] = samples[-_MAX_SAMPLES:]
            _save(data)

        if len(prior) >= _MIN_BASELINE:
            base = _median(prior)
            if seconds > base * _WARN_RATIO and seconds > base + _WARN_FLOOR_S:
                print(
                    f"\033[93m⚠ [PERF] {metric}: {seconds:.1f}s — {seconds / base:.1f}x "
                    f"slower than typical ({base:.1f}s median of last {len(prior)}). "
                    f"Slow loads → check disk cold-read speed (NAND aging); "
                    f"slow steps → check VRAM pressure/thermals.\033[0m",
                    flush=True,
                )
    except Exception:
        pass  # never let the watchdog break generation


def summary() -> dict:
    """Per-metric stats for the /api/system/perf endpoint."""
    out = {}
    try:
        with _lock:
            data = _load()
        for metric, samples in sorted(data.items()):
            vals = [s["v"] for s in samples]
            if not vals:
                continue
            base = _median(vals)
            last = samples[-1]
            out[metric] = {
                "count": len(vals),
                "median_s": round(base, 3),
                "last_s": last["v"],
                "last_at": last.get("t"),
                "min_s": min(vals),
                "max_s": max(vals),
                "last_vs_median": round(last["v"] / base, 2) if base > 0 else None,
            }
    except Exception:
        pass
    return out
