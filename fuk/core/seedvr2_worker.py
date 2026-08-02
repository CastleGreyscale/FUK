"""
SeedVR2 worker — runs INSIDE the isolated SeedVR2 environment.

This file is never imported by FUK for its inference path. The main venv
launches it as a subprocess using vendor/SEEDVR2_ENV/env/bin/python, because
SeedVR2 needs flash-attn and apex built against its own torch, which would
conflict with the main venv's torch 2.9/cu130 (see install_seedvr2_env.sh).

    Only RESULT_MARKER is imported by the host, at module scope, so this file
    must stay importable under the main venv's interpreter. Keep every heavy
    import inside main().

Protocol
    argv[1]   path to a JSON job file
    stdout    human-readable progress, streamed to FUK's console panel, plus
              one machine-readable line:  __SEEDVR2_RESULT__ {json}
    exit 0    success

Job file keys: input, output, seedvr_src, weights_dir, variant, checkpoint,
script, scale, frame_window, resolution_cap, seed.

Why this shells out
    SeedVR2 is driven through its own `projects/inference_seedvr2_*.py` entry
    point rather than by importing VideoDiffusionInfer directly. That script is
    the interface ByteDance document and keep stable; their internal runner
    classes are neither. Driving the CLI costs one subprocess and buys immunity
    from upstream refactors.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

RESULT_MARKER = "__SEEDVR2_RESULT__"

# Frames per forward pass, chosen from total VRAM. SeedVR2's adaptive window
# attention means memory scales with window * resolution, and a too-large
# window fails at the first chunk rather than degrading. These are deliberately
# conservative — a completed slow run beats an OOM at 90%.
_WINDOW_BY_VRAM_GB = [
    (80, 97),   # H100 — the configuration ByteDance benchmark
    (40, 49),   # A100 40G / 5090-class
    (24, 25),   # 4090 / 3090
    (16, 13),
    (0,   5),   # SeedVR2 degrades badly below ~5 frames of context
]


def emit(message):
    print(f"[SeedVR2] {message}", flush=True)


def _probe(video: Path) -> dict:
    """Source dimensions, fps and frame count via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    stream = (json.loads(out.stdout).get("streams") or [{}])[0]

    fps_str = stream.get("r_frame_rate", "16/1")
    num, _, den = fps_str.partition("/")
    fps = float(num) / float(den or 1)

    frames = int(stream.get("nb_frames") or 0)
    if not frames:
        # nb_frames is absent on some containers; count the hard way rather
        # than guess, since chunking arithmetic depends on it.
        cnt = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nk=1:nw=1", str(video)],
            capture_output=True, text=True,
        )
        frames = int((cnt.stdout or "0").strip() or 0)

    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "frames": frames,
    }


def _pick_window(requested: int) -> int:
    if requested > 0:
        return requested
    try:
        import torch
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        gb = 24
    for threshold, window in _WINDOW_BY_VRAM_GB:
        if gb >= threshold:
            emit(f"detected {gb:.0f}GB VRAM — frame window {window}")
            return window
    return 5


def _target_resolution(width, height, scale, cap):
    tw, th = width * scale, height * scale
    longest = max(tw, th)
    if cap and longest > cap:
        ratio = cap / longest
        tw, th = int(tw * ratio), int(th * ratio)
    return tw - tw % 2, th - th % 2


def main():
    if len(sys.argv) < 2:
        print("usage: seedvr2_worker.py <job.json>", file=sys.stderr)
        return 2

    job = json.loads(Path(sys.argv[1]).read_text())

    seedvr_src = Path(job["seedvr_src"])
    if str(seedvr_src) not in sys.path:
        sys.path.insert(0, str(seedvr_src))

    source = Path(job["input"])
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    emit(f"probing input: {source.name}")
    info = _probe(source)
    if not info["width"] or not info["frames"]:
        raise RuntimeError(f"Could not read video stream from {source}")
    emit(f"source: {info['width']}x{info['height']}, "
         f"{info['frames']} frames @ {info['fps']:.2f}fps")

    res_w, res_h = _target_resolution(
        info["width"], info["height"], job["scale"], job["resolution_cap"]
    )
    effective_scale = round(res_w / info["width"], 2)
    if effective_scale < job["scale"]:
        emit(f"resolution cap applied — {job['scale']}x requested, "
             f"{effective_scale}x achievable within {job['resolution_cap']}px")
    emit(f"target: {res_w}x{res_h}")

    window = _pick_window(job["frame_window"])
    chunks = max(1, math.ceil(info["frames"] / window))

    checkpoint = Path(job["weights_dir"]) / job["checkpoint"]
    script = seedvr_src / job["script"]
    if not script.exists():
        raise RuntimeError(f"SeedVR2 inference script not found: {script}")

    t0 = time.perf_counter()
    start_mem = 0

    with tempfile.TemporaryDirectory(prefix="fuk_seedvr2_") as tmp:
        tmp = Path(tmp)
        in_dir = tmp / "in"
        out_dir = tmp / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # SeedVR2's script takes a *directory* of clips. Chunking is done by
        # splitting the source into segments rather than by asking the model
        # for a smaller window, because the window is a property of the config
        # and the segment length is not.
        if chunks == 1:
            emit("extracting frames (single pass)")
            shutil.copy2(source, in_dir / source.name)
        else:
            emit(f"extracting frames — splitting into {chunks} chunks "
                 f"of {window} frames")
            seg_seconds = window / max(info["fps"], 1)
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(source),
                 "-c", "copy", "-f", "segment",
                 "-segment_time", f"{seg_seconds:.4f}",
                 "-reset_timestamps", "1",
                 str(in_dir / "chunk_%04d.mp4")],
                check=True,
            )

        emit(f"loading weights: {checkpoint.name}")
        cmd = [
            "torchrun", "--nproc-per-node=1", str(script),
            "--video_path", str(in_dir),
            "--output_dir", str(out_dir),
            "--seed", str(job["seed"]),
            "--res_h", str(res_h),
            "--res_w", str(res_w),
            "--sp_size", "1",
        ]
        emit("runner ready — " + " ".join(cmd[2:]))

        env = {
            **os.environ,
            "PYTHONPATH": str(seedvr_src),
            "PYTHONUNBUFFERED": "1",
            # Fragmentation is what actually kills long runs on 24GB; the
            # allocator holds enough freed-but-uncoalesced blocks to fail a
            # later chunk that the first one fitted into fine.
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
            "SEEDVR2_CKPT": str(checkpoint),
        }

        proc = subprocess.Popen(
            cmd, cwd=str(seedvr_src), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        done = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(f"  {line}", flush=True)
            # Their script logs one line per clip written; that is the only
            # reliable per-chunk signal, so it drives the host's progress bar.
            if "saved" in line.lower() or "writing" in line.lower():
                done += 1
                emit(f"chunk {min(done, chunks)}/{chunks}")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"SeedVR2 inference exited with code {rc}")

        produced = sorted(
            [p for p in out_dir.rglob("*") if p.suffix.lower() in (".mp4", ".mkv", ".mov")]
        )
        if not produced:
            listing = [p.name for p in out_dir.rglob("*")][:20]
            raise RuntimeError(f"SeedVR2 produced no video output. Got: {listing}")

        emit("encoding output")
        if len(produced) == 1:
            shutil.move(str(produced[0]), str(output))
        else:
            # Concat back in segment order. Re-encoding here would undo the
            # restoration, so the segments are stream-copied.
            listfile = tmp / "concat.txt"
            listfile.write_text(
                "\n".join(f"file '{p.as_posix()}'" for p in produced)
            )
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(listfile), "-c", "copy", str(output)],
                check=True,
            )

    elapsed = time.perf_counter() - t0
    emit(f"wrote {output.name} in {elapsed:.1f}s")

    try:
        import torch
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        peak = None

    result = {
        "output_path": str(output),
        "input_size": {"width": info["width"], "height": info["height"]},
        "output_size": {"width": res_w, "height": res_h},
        "scale": effective_scale,
        "frame_count": info["frames"],
        "frame_window": window,
        "chunks": chunks,
        "variant": job["variant"],
        "method": "seedvr2",
        "inference_seconds": round(elapsed, 2),
        "peak_vram_gb": peak,
    }
    print(f"{RESULT_MARKER} {json.dumps(result)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
