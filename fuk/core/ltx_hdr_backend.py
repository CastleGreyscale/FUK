"""
LTX-2.5 SDR -> HDR conversion — host side.

Lightricks' SDR-To-HDR IC-LoRA for LTX-2.5 22B takes a finished 8-bit SDR clip
and returns it as scene-linear HDR: a half-float EXR sequence (ACEScg by
default) plus a 10-bit BT.2020/HLG HEVC master. The SDR clip is the whole
control signal — there is no prompt, no CFG and no text encoder; the LoRA ships
a precomputed embedding of the one caption it was trained on.

This is the "real range" path the latent -> EXR exporter cannot be. That
exporter's ceiling is the VAE's [0, 1] clamp — bracketing recovers precision,
not range nothing captured. Here the model was trained on paired SDR/HDR plates
and synthesises the highlights SDR clipped away.

Why upstream's pipeline and not DiffSynth
    FUK's LTX-2.5 runs through DiffSynth, which has no HDR code at all, and the
    HDR LoRA leans on more than a LoRA: an ACEScct input transform, a frame-
    aligned full-resolution reference, DFR-style seam keyframes (a 1-frame SDR
    guide plus a generated HDR slot at every 24/32-frame border), and a
    keyframe-aware *diffusion* VAE decode (decoder.type_emb, joint attention
    with the nearest keyframe planes). Porting that took the diffusers team a
    dedicated PR series validated against upstream to 50dB. So this runs
    Lightricks' own ltx-pipelines from a pinned checkout in fuk/vendor/LTX-2,
    driven by ltx_hdr_runner.py in a subprocess — the SeedVR2 arrangement, for
    the same VRAM and OOM-recovery reasons given in seedvr2_backend.

    It runs in FUK's venv: ltx-core asks for torch~=2.7 and transformers<5.15,
    both already satisfied. Its only new imports are colour-science,
    openimageio and cloudpickle, which pyproject.toml carries. NATTEN — pinned
    upstream to torch 2.13 — is NOT needed: keyframe-aware decode cannot use it
    and always runs on Triton, and the keyframe-free path falls back to Triton
    without it.

Memory, measured on a 24GB 4090
    The FP8-cast transformer does not fit: loading it alone reached 21.1GB and
    OOMed before the first step. With --offload cpu the bf16 weights are pinned
    in RAM (~42GB) and streamed per layer; a 17-frame 768x432 clip then peaked
    at 7.7GB and finished in 39s including load. Resident FP8 is still tried
    first on cards big enough to hold it, since it is the faster path.

Frame count
    The pipeline requires 8k+1 frames and raises otherwise. Rather than drop up
    to seven frames of someone's plate, the source is padded up to the next
    8k+1 by holding its last frame (lossless FFV1, so the reference is
    bit-identical) and the runner trims the decode back before writing.

Licence: LTX-2.x Community License, same as the LTX-2.5 base FUK already runs.
Source:  https://github.com/Lightricks/LTX-2  (packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py)
Weights: Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR, plus the 2.5 distilled
         transformer and video VAE from Lightricks/LTX-2.5.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .seedvr2_backend import ALLOC_CONF_VARS, _free_vram_for_seedvr2, _models_root

_CORE_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _CORE_DIR.parent / "vendor" / "LTX-2"
_ENGINE_SRC = [
    _ENGINE_DIR / "packages" / "ltx-core" / "src",
    _ENGINE_DIR / "packages" / "ltx-pipelines" / "src",
]
_RUNNER = _CORE_DIR / "ltx_hdr_runner.py"

INSTALL_HINT = "Run: bash setup.sh   (clones fuk/vendor/LTX-2)"

# (model_id, file) pairs, resolved under models_root the way DiffSynth's
# ModelConfig lays out downloads — so download_models.download_model_component
# puts them exactly where this looks.
WEIGHTS = {
    "transformer": ("Lightricks/LTX-2.5",
                    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
    "video_vae": ("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-bf16.safetensors"),
    "hdr_lora": ("Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR",
                 "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"),
    "text_embeddings": ("Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR",
                        "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"),
}

# EXR sidecar colour spaces upstream can write. ACEScg is the default and what
# a grading pipeline expects; acescct writes the VAE's log codes untouched.
EXR_COLORSPACES = ("acescg", "srgb_linear", "acescct")

# Tried in order under memory_mode="auto". A rung whose min_vram_gb exceeds the
# card is skipped outright rather than spending a minute loading 21GB to OOM.
LADDER: List[Dict[str, Any]] = [
    {"name": "resident", "label": "resident FP8", "offload": "none",
     "quantize": True, "min_vram_gb": 30},
    # Upstream disables quantization whenever offload is on, so this streams
    # bf16 — slower per step than resident FP8, but more faithful. Upstream
    # calls it "cpu" offload, but only the weights live there: every layer is
    # computed on the GPU. The label says so, because "CPU offload" next to a
    # 2%-busy CPU reads as a hang.
    {"name": "offload", "label": "GPU, weights streamed from RAM", "offload": "cpu",
     "quantize": False, "min_vram_gb": 0},
]

# Seconds of subprocess silence before the host prints a "still working" line.
_HEARTBEAT_AFTER_S = 60

# Largest clip measured to fit one denoise window on a 24GB card, in latent
# tokens: (frames-1)/8+1 latent frames x (H/32) x (W/32). 121 frames of
# 1920x1088 (16 x 34 x 60 = 32,640) fits; 225 frames (29 x 34 x 60 = 59,160)
# OOMs on the first step. The true ceiling is somewhere between — this is a
# hint for the error message, not a gate.
_KNOWN_GOOD_TOKENS_24GB = 32_640

# tqdm step lines: " 38%|███▊      | 3/8 [00:06<00:09,  1.84s/it]"
_TQDM_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")

_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
)


def weight_paths() -> Dict[str, Path]:
    root = _models_root()
    return {key: root / model_id / pattern for key, (model_id, pattern) in WEIGHTS.items()}


def availability() -> Dict[str, Any]:
    """
    Report whether SDR -> HDR can run, and what's missing if it can't.

    Unlike SeedVR2, weights are required up front: nothing downloads them on
    first use, and the transformer alone is 42GB — not something to start
    silently from a Postprocess click.
    """
    missing: List[str] = []

    if not all(p.exists() for p in _ENGINE_SRC):
        missing.append("LTX-2 pipelines (fuk/vendor/LTX-2)")

    import importlib.util

    for mod, pkg in (("colour", "colour-science"), ("OpenImageIO", "openimageio"),
                     ("cloudpickle", "cloudpickle")):
        if importlib.util.find_spec(mod) is None:
            missing.append(f"python package '{pkg}'")

    paths = weight_paths()
    absent = [key for key, path in paths.items() if not path.exists()]
    for key in absent:
        missing.append(f"weights: {WEIGHTS[key][0]}/{WEIGHTS[key][1]}")

    hint = None
    if missing:
        hint = (INSTALL_HINT if any("fuk/vendor" in m or "package" in m for m in missing)
                else "Run: bash download_models.sh   (fetches the HDR LoRA and distilled transformer)")

    return {
        "available": not missing,
        "missing": missing,
        "weights": {key: str(path) for key, path in paths.items()},
        "hint": hint,
    }


def _total_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        pass
    return 0.0


def _warm_page_cache(paths: List[Path], log: Callable[[str], None],
                     progress: Optional[Callable[[float, str], None]] = None) -> None:
    """
    Read the weights sequentially so the subprocess finds them in page cache.

    Upstream's offload path fills its pinned RAM copy of the transformer by
    reading tensors through safetensors' mmap. On a cold cache that faults the
    file in a page at a time: measured on the 980 PRO, the 42GB DiT loaded at
    ~42MB/s (≈10k reads/s of 4KB) and took over eight minutes, while a plain
    sequential read of the same file ran at 5.2GB/s and took 8s. Every run
    in a fresh session paid that, and the silent load also tripped the
    progress stream's 5-minute idle timeout.

    Already-cached files read at memory speed, so this costs a few seconds when
    it is not needed. The cache is ordinary reclaimable memory.
    """
    total = sum(p.stat().st_size for p in paths)
    done = 0
    buf = memoryview(bytearray(64 << 20))
    start = time.perf_counter()
    for path in paths:
        with open(path, "rb", buffering=0) as fh:
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            while n := fh.readinto(buf):
                done += n
                if progress and done % (2 << 30) < n:
                    progress(done / total, f"Reading weights {done / 1e9:.0f}/{total / 1e9:.0f}GB")
    elapsed = time.perf_counter() - start
    log(f"Weights read into page cache: {total / 1e9:.1f}GB in {elapsed:.1f}s "
        f"({total / 1e9 / max(elapsed, 1e-3):.1f}GB/s)")


def _exact_frame_count(path: Path) -> int:
    """
    Count video packets by demuxing — no decode, so it is quick.

    get_video_info falls back to duration * fps when a container (MKV, WebM)
    records no frame count, and an estimate is not good enough here: one frame
    off and the 8k+1 padding is wrong, so upstream refuses the clip or the trim
    cuts a real frame.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return int(out.stdout.strip().split(",")[0])


def _pad_to_8k1(input_path: Path, frames: int, work_dir: Path,
                log: Callable[[str], None]) -> Path:
    """
    Hold the last frame until the clip is 8k+1 long, losslessly.

    FFV1 keeps the source's own pixel format, so the frames the model sees as
    its reference decode identically to the original. MKV rather than MP4
    because MP4 has no FFV1 mapping; upstream decodes through PyAV and counts
    frames by decoding when the container does not record them, so it takes
    the file as-is.
    """
    target = ((frames - 1 + 7) // 8) * 8 + 1
    pad = target - frames
    if pad == 0:
        return input_path

    padded = work_dir / f"{input_path.stem}_pad{target}.mkv"
    log(f"{frames} frames is not 8k+1 — holding the last frame {pad}x to {target}, "
        f"trimmed back after decode")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(input_path),
         "-vf", f"tpad=stop_mode=clone:stop={pad}",
         "-map", "0:v:0", "-c:v", "ffv1", "-level", "3", str(padded)],
        check=True,
    )
    return padded


def _build_command(input_path: Path, keep_frames: int, output_path: Path, rung: Dict[str, Any],
                   exr_colorspace: str, seed: int, keyframes: bool, keyframe_strength: float,
                   high_quality: bool) -> List[str]:
    paths = weight_paths()
    cmd = [
        sys.executable, str(_RUNNER),
        "--input", str(input_path),
        "--keep-frames", str(keep_frames),
        "--output-path", str(output_path),
        "--transformer-path", str(paths["transformer"]),
        "--video-vae-path", str(paths["video_vae"]),
        "--hdr-lora", str(paths["hdr_lora"]),
        "--text-embeddings", str(paths["text_embeddings"]),
        "--exr-colorspace", exr_colorspace,
        "--seed", str(seed),
        "--offload", rung["offload"],
        "--keyframe-strength", str(keyframe_strength),
    ]
    if not rung["quantize"]:
        cmd.append("--no-quantization")
    if not keyframes:
        cmd.append("--no-keyframes")
    if high_quality:
        cmd.append("--high-quality")
    return cmd


def _run_once(cmd: List[str], rung_label: str, progress_callback: Optional[Callable],
              progress_base: float, progress_span: float, timeout: int) -> Dict[str, Any]:
    """
    Run one attempt. Returns {"code", "tail", "timed_out", "result"}.

    Never raises on a failed run; the caller decides whether a non-zero exit is
    an OOM worth escalating or a real error worth reporting.
    """
    tail: List[str] = []
    timed_out = False
    result: Optional[Dict[str, Any]] = None
    process = None
    code = 1
    phase = None

    # Upstream logs one line per phase; the denoise loop is a tqdm bar, which
    # text-mode pipes split on its carriage returns into one line per step.
    phases = (
        ("building transformer", 0.05, "Loading transformer"),
        ("running denoising loop", 0.15, "Denoising"),
        ("building video decoder", 0.70, "Decoding HDR"),
        ("decode oom — retrying", 0.70, "Decode OOM, retrying with smaller tiles"),
        ("encoding bt.2020/hlg", 0.90, "Writing EXR + HLG"),
    )

    last = {"frac": 0.0, "label": "Starting", "at": time.monotonic()}

    def _emit(frac: float, label: str) -> None:
        last.update(frac=frac, label=label)
        if progress_callback:
            progress_callback(progress_base + progress_span * frac, f"{rung_label}: {label}")

    # Upstream can be silent for minutes — a long denoise step, pinning 42GB of
    # weights. Say so, both for whoever is watching and because the progress
    # stream closes after five minutes without an update.
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(30):
            quiet = time.monotonic() - last["at"]
            if quiet >= _HEARTBEAT_AFTER_S:
                print(f"[LTX-HDR] still working: {last['label']} "
                      f"(no output for {quiet:.0f}s)", flush=True)
                _emit(last["frac"], last["label"])

    pythonpath = os.pathsep.join(str(p) for p in _ENGINE_SRC)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_ENGINE_DIR),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": pythonpath,
                **{var: "expandable_segments:True" for var in ALLOC_CONF_VARS},
            },
        )

        # Same watchdog as seedvr2_backend: a wedged child never reaches EOF,
        # so only killing it can unblock the read loop.
        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            print(f"[LTX-HDR] attempt exceeded {timeout}s — terminating", flush=True)
            try:
                process.kill()
            except Exception:
                pass

        watchdog = threading.Timer(timeout, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()
        threading.Thread(target=_heartbeat, daemon=True, name="ltx-hdr-heartbeat").start()

        try:
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                last["at"] = time.monotonic()

                if line.startswith("FUK_RESULT "):
                    result = json.loads(line[len("FUK_RESULT "):])
                    continue

                # x265 prints a dozen lines of encoder settings per run.
                if line.startswith("x265 [info]"):
                    continue

                print(f"[LTX-HDR] {line}", flush=True)
                tail.append(line)
                del tail[:-60]

                lowered = line.lower()
                for marker, frac, label in phases:
                    if marker in lowered:
                        phase = label
                        _emit(frac, label)
                        break
                else:
                    # Only the denoise bar maps onto the progress band; any
                    # other tqdm upstream prints would run the bar backwards.
                    match = _TQDM_STEP_RE.search(line)
                    if match and phase == "Denoising":
                        done, total = int(match.group(1)), max(int(match.group(2)), 1)
                        _emit(0.15 + 0.55 * done / total, f"Denoising {done}/{total}")

            code = process.wait()
        finally:
            watchdog.cancel()
            stop_heartbeat.set()
    finally:
        if process and process.poll() is None:
            process.kill()

    return {"code": code, "tail": tail, "timed_out": timed_out, "result": result,
            "phase": phase}


def _is_oom(tail: List[str]) -> bool:
    joined = "\n".join(tail).lower()
    return any(marker in joined for marker in _OOM_MARKERS)


def _oom_message(phase: Optional[str], frames: int, width: int, height: int,
                 high_quality: bool) -> str:
    """Say where it ran out and what will actually fix it."""
    if phase in ("Decoding HDR", "Decode OOM, retrying with smaller tiles", "Writing EXR + HLG"):
        # The runner has already retried decode at smaller tiles by now.
        return ("SDR -> HDR denoised, but ran out of VRAM decoding even at the "
                "smallest tiling. Trim the clip or lower its resolution.")

    # Out in (or before) the denoise: the whole clip is one attention window,
    # so no tiling helps — only fewer frames or fewer pixels.
    gen_frames = 2 * frames - 1 if high_quality else frames
    per_frame = max(1, ((height + 31) // 32) * ((width + 31) // 32))
    tokens = ((gen_frames - 1) // 8 + 1) * per_frame
    fit_frames = (max(1, _KNOWN_GOOD_TOKENS_24GB // per_frame) - 1) * 8 + 1
    if high_quality:
        fit_frames = (fit_frames + 1) // 2
    return (
        f"SDR -> HDR ran out of VRAM in the denoise: {frames} frames at {width}x{height}"
        f"{' (high quality doubles that)' if high_quality else ''} is ~{tokens:,} latent "
        f"tokens, all denoised as one window, and the model has no smaller window to "
        f"fall back to. On a 24GB card about {fit_frames} frames at this resolution is "
        f"known to fit — split the clip into shots of that length or less."
    )


def _make_preview(hlg_path: Path, preview_path: Path, log: Callable[[str], None]) -> Optional[Path]:
    """
    Tonemap the HLG master to an SDR H.264 the browser can play.

    Browsers on Linux mostly cannot decode 10-bit HEVC at all, and none of them
    show HLG as HLG on an SDR panel. This is a viewing copy only — Hable, so
    recovered highlights roll off visibly instead of clipping. Needs ffmpeg's
    zscale (libzimg); without it the UI is handed the HLG master instead.
    """
    vf = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
          "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(hlg_path), "-vf", vf,
             "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
             "-movflags", "+faststart", str(preview_path)],
            check=True, capture_output=True, text=True,
        )
        return preview_path
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        log(f"SDR preview skipped ({detail.strip()[:160]}) — serving the HLG master")
        return None


def convert_to_hdr(
    input_path: Path,
    output_path: Path,
    exr_colorspace: str = "acescg",
    seed: int = 42,
    keyframes: bool = True,
    keyframe_strength: float = 0.95,
    high_quality: bool = False,
    memory_mode: str = "auto",
    progress_callback: Optional[Callable] = None,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """
    Convert an SDR video to HDR with the LTX-2.5 SDR-To-HDR IC-LoRA.

    output_path
        The HLG master (.mp4). Upstream writes the EXR sequence beside it as
        <stem>_<colourspace>_exr/frame_NNNNN.exr, and an SDR viewing copy is
        written as <stem>_preview.mp4.
    keyframes / keyframe_strength
        Seam keyframes are the trained default and only engage on clips long
        enough to have a seam (past ~24 frames). Upstream warns explicitly
        against strength 0 — leave it at 0.95 unless comparing.
    high_quality
        Generates 2N-1 frames from a doubled reference and keeps every other
        one. Fewer temporal artifacts at roughly twice the time and memory.
    memory_mode
        "auto" tries resident FP8 on cards that can hold it, then CPU offload.
        "resident" or "offload" pins one and reports OOM instead of retrying.
    """
    status = availability()
    if not status["available"]:
        raise RuntimeError(
            "SDR -> HDR is not available: " + ", ".join(status["missing"])
            + f"\n  {status['hint']}"
        )
    if exr_colorspace not in EXR_COLORSPACES:
        raise ValueError(f"exr_colorspace must be one of {EXR_COLORSPACES}, got {exr_colorspace!r}")

    # Absolute, because the subprocess runs with cwd=_ENGINE_DIR.
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log = lambda msg: print(f"[LTX-HDR] {msg}", flush=True)

    from .video_utils import get_video_info

    info = get_video_info(input_path)
    frames = _exact_frame_count(input_path)
    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    if frames < 1 or not width or not height:
        raise RuntimeError(f"Could not read frame count/dimensions from {input_path}")

    if memory_mode == "auto":
        total = _total_vram_gb()
        rungs = [r for r in LADDER if total >= r["min_vram_gb"]]
        skipped = [r["label"] for r in LADDER if r not in rungs]
        if skipped:
            log(f"{total:.0f}GB card — skipping {', '.join(skipped)}")
    else:
        rungs = [r for r in LADDER if r["name"] == memory_mode]
        if not rungs:
            raise ValueError(f"Unknown memory_mode {memory_mode!r}. Use 'auto', "
                             + " or ".join(repr(r["name"]) for r in LADDER))

    log(f"Source {width}x{height}, {frames} frames @ {info.get('fps')}fps → "
        f"EXR {exr_colorspace} + HLG master")
    log(f"Seam keyframes {'@ ' + str(keyframe_strength) if keyframes else 'off'}"
        f"{', high-quality (2x frames)' if high_quality else ''}, seed {seed}")

    padded = _pad_to_8k1(input_path, frames, output_path.parent, log)

    _free_vram_for_seedvr2(log)
    if progress_callback:
        progress_callback(0.02, "Reading weights")
    paths = weight_paths()
    _warm_page_cache(
        [paths["transformer"], paths["hdr_lora"], paths["video_vae"]], log,
        (lambda f, label: progress_callback(0.02 + 0.06 * f, label)) if progress_callback else None,
    )

    last: Optional[Dict[str, Any]] = None
    rung = rungs[0]
    try:
        for index, rung in enumerate(rungs):
            label = f"{rung['label']}"
            log(f"Attempt {index + 1}/{len(rungs)} — {label}")
            cmd = _build_command(padded, frames, output_path, rung, exr_colorspace, seed,
                                 keyframes, keyframe_strength, high_quality)
            span = 0.87 / len(rungs)
            last = _run_once(cmd, label, progress_callback,
                             progress_base=0.08 + span * index, progress_span=span,
                             timeout=timeout)

            if last["timed_out"]:
                raise RuntimeError(
                    f"SDR -> HDR timed out after {timeout}s. Long or large plates "
                    "are the usual cause; trim the clip.\n" + "\n".join(last["tail"][-10:])
                )
            if last["code"] == 0 and last["result"] and output_path.exists():
                break
            if _is_oom(last["tail"]) and index + 1 < len(rungs):
                log(f"{label} ran out of VRAM — retrying with {rungs[index + 1]['label']}")
                _free_vram_for_seedvr2(log)
                continue

            detail = "\n".join(last["tail"][-15:])
            if _is_oom(last["tail"]):
                raise RuntimeError(_oom_message(last.get("phase"), frames, width, height,
                                                high_quality) + "\n" + detail)
            raise RuntimeError(f"SDR -> HDR exited with code {last['code']}\n{detail}")
    finally:
        if padded != input_path:
            padded.unlink(missing_ok=True)

    result = last["result"]
    exr_dir = Path(result["exr_dir"]) if result.get("exr_dir") else None
    exr_frames = len(list(exr_dir.glob("*.exr"))) if exr_dir and exr_dir.exists() else 0
    if exr_frames != frames:
        log(f"WARNING: wrote {exr_frames} EXR frames for a {frames}-frame source")

    if progress_callback:
        progress_callback(0.96, "SDR preview")
    preview = _make_preview(output_path, output_path.with_name(output_path.stem + "_preview.mp4"), log)

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "method": "ltx_hdr",
        "memory_rung": rung["name"],
        "exr_colorspace": exr_colorspace,
        "exr_dir": str(exr_dir) if exr_dir else None,
        "exr_frames": exr_frames,
        "hlg_path": str(output_path),
        "preview_path": str(preview) if preview else None,
        "frame_count": result["frames"],
        "padded_frames": result["generated_frames"] - result["frames"],
        "fps": result["fps"],
        "size": {"width": result["width"], "height": result["height"]},
        "keyframes": keyframes,
        "keyframe_strength": keyframe_strength if keyframes else None,
        "high_quality": high_quality,
        "seed": seed,
        "peak_vram_gb": result.get("peak_vram_gb"),
    }
