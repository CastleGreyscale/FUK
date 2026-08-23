"""
SeedVR2 video restoration / upscaling — host side.

Runs in FUK's main venv but executes nothing itself: it launches the vendored
SeedVR2 CLI in a subprocess and streams that process's output into FUK's
console panel.

Why a subprocess and not an in-process import
    Two reasons, and neither is about dependency isolation any more.

    First, VRAM. FUK keeps up to `pipeline_cache_slots` DiffSynth pipelines
    resident, and a Wan pipeline can occupy most of a 24GB card. We evict them
    before launching (see _free_vram_for_seedvr2), but a subprocess is the only
    thing that guarantees *everything* — allocator arenas, fragmented blocks,
    cuBLAS workspaces — is actually returned when the run ends.

    Second, OOM recovery. An out-of-memory error inside the web server process
    is a coin flip between a clean exception and a wedged CUDA context. As a
    subprocess it is just a non-zero exit code, which is what makes the
    escalation ladder below possible at all.

History
    A previous integration (commit 1d4e822, removed in 20b77fe) drove
    ByteDance's own inference script under an isolated torch-2.4 environment.
    That reference implementation has no quantization, no block swapping and no
    VAE tiling, so the only memory knob was frame-window size — and it still
    would not fit 24GB. This engine is a reimplementation that adds all three,
    which is what makes SeedVR2 viable here.

Status: under evaluation
    Wired up, measured and usable, but not signed off as part of the finishing
    pipeline. The open question is whether the detail it invents is acceptable
    on clean generated footage — it is a restoration model, so on source with
    little real detail to recover it hallucinates, and what it hallucinates
    varies frame to frame. See motion_protect for the largest of those effects.

Bit depth — why this is not in the latent -> EXR path
    The engine is 8-bit display-referred at both ends and there is no option to
    change that: input is divided by 255 on load (so 16-bit PNG is *misread*,
    not merely truncated), internals are fp16, and every write path multiplies
    by 255 into uint8. No float, no linear light, nothing above 1.0.

    Patching that would be a few lines and would still be wrong: the model was
    trained on display-referred video, so linear HDR is out of distribution for
    it. Display space is where it belongs.

    That matters because exr_exporter._decode_beauty_latents does real work
    8-bit would discard — float32 VAE decode, optional scale/noise bracketing
    fused with Mertens to recover range, then sRGB -> linear, all of it before
    any quantisation. Running SeedVR2 downstream of that writes float32 EXR
    containers holding round-tripped 8-bit sRGB, with the bracketing wasted.

    There is also no latent-space shortcut: SeedVR2 carries its own VAE and
    cannot consume Wan latents, so the latent must be decoded to pixels first
    whichever route is taken.

    The EXR path therefore stays clean and linear, and this runs on the delivery
    side. If an upscaled EXR is ever needed, the options are a lanczos resample
    of the float decode (nothing invented) or a ratio-based detail transfer
    (upscale the HDR cleanly, run SeedVR2 on a tonemapped copy, multiply by
    enhanced/unenhanced so values above 1.0 survive). See the wiki.

Source:  https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler  (Apache 2.0)
Weights: AInVFX/SeedVR2_comfyUI, from ByteDance-Seed/SeedVR2-3B (Apache 2.0)
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_PEAK_VRAM_RE = re.compile(r"Peak:\s*([\d.]+)\s*GB", re.IGNORECASE)


# Both spellings of the CUDA allocator config variable. torch 2.9.1 warns the
# first is deprecated yet it is the only one that build actually parses, so
# setting just the "modern" one silently disables expandable_segments. Kept in
# step with diffsynth_backend.ALLOC_CONF_VARS but duplicated rather than
# imported: that module pulls in DiffSynth and mutates sys.path at import time,
# which is far too much to drag in on a path that also runs without a server.
ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")

_CORE_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _CORE_DIR.parent / "vendor"
_ENGINE_DIR = _VENDOR_DIR / "SeedVR2"
_ENGINE_CLI = _ENGINE_DIR / "inference_cli.py"
_CONFIG_DIR = _CORE_DIR.parent / "config"

INSTALL_HINT = "Run: bash setup.sh   (clones fuk/vendor/SeedVR2)"

# Variants are grouped into families. A family is one model at two precisions:
# the one you ask for, and a smaller quantized one the escalation ladder can
# drop to without leaving the model you chose. That distinction matters — a 7B
# run that runs out of VRAM should become a quantized 7B, not silently become a
# 3B, because the 7B is what you picked it for.
#
# `max_blocks` is the engine's per-architecture BlockSwap ceiling: 32 for the
# 3B, 36 for the 7B. Asking for more is rejected by the engine.
#
# fp16 variants exist upstream for both sizes and are deliberately absent: at
# ~14GB resident for the 7B they leave nothing for the VAE, which is the phase
# that actually peaks on a 24GB card.
VARIANTS = {
    "seedvr2_3b_fp8": {
        "label": "SeedVR2 3B (fp8)",
        "checkpoint": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "family": "3b",
        "precision": "fp8",
        "max_blocks": 32,
    },
    "seedvr2_3b_q8": {
        "label": "SeedVR2 3B (GGUF Q8_0)",
        "checkpoint": "seedvr2_ema_3b-Q8_0.gguf",
        "family": "3b",
        "precision": "quant",
        "max_blocks": 32,
    },
    "seedvr2_7b_fp8": {
        "label": "SeedVR2 7B (fp8, block 35 fp16)",
        # The last block is kept at fp16 upstream specifically to stop fp8
        # quantization artifacts showing in the output.
        "checkpoint": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "family": "7b",
        "precision": "fp8",
        "max_blocks": 36,
    },
    "seedvr2_7b_q4": {
        "label": "SeedVR2 7B (GGUF Q4_K_M)",
        "checkpoint": "seedvr2_ema_7b-Q4_K_M.gguf",
        "family": "7b",
        "precision": "quant",
        "max_blocks": 36,
    },
    "seedvr2_7b_sharp_fp8": {
        "label": "SeedVR2 7B Sharp (fp8, block 35 fp16)",
        "checkpoint": "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "family": "7b_sharp",
        "precision": "fp8",
        "max_blocks": 36,
    },
    "seedvr2_7b_sharp_q4": {
        "label": "SeedVR2 7B Sharp (GGUF Q4_K_M)",
        "checkpoint": "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        "family": "7b_sharp",
        "precision": "quant",
        "max_blocks": 36,
    },
}

# Measured on a 45-frame 640x360 clip upscaled to 720p, window 13, against the
# source's own temporal statistics (higher lag-1 = smoother motion, lower lag-4
# = less of the 4-frame stepping the VAE's temporal compression introduces):
#
#                 mean delta   lag-1   lag-4
#   source            1.009     0.65    0.08
#   3B fp8            1.409     0.31    0.10     38.8s
#   7B fp8            1.388     0.35    0.05     54.0s
#
# The 7B halves the stepping — below the source's own figure — and invents
# slightly less spurious detail, for about 40% more time. It peaks at 13.4GB on
# a 24GB card, so it fits without touching the escalation ladder. Worth the
# default for a finishing pass; pick a 3B variant when iterating.
DEFAULT_VARIANT = "seedvr2_7b_fp8"

# Pulled by download_models.sh. The sharp 7B is left out on purpose: it is a
# stylistic alternative rather than a quality tier, another ~12GB, and the
# engine fetches it on first use if anyone selects it.
DEFAULT_DOWNLOAD_VARIANTS = (
    "seedvr2_3b_fp8", "seedvr2_3b_q8", "seedvr2_7b_fp8", "seedvr2_7b_q4",
)


def _validate_ladder() -> None:
    """
    Check rung invariants at import time.

    BlockSwap and I/O swapping both need somewhere to put what they offload, and
    the engine raises rather than ignoring the flag:

        ValueError: BlockSwap enabled (blocks_to_swap=16) but
        dit_offload_device is invalid.

    Rung 2 shipped with exactly that combination. It went unnoticed because the
    broken rung is only reached after two OOMs, so every test that fit in VRAM
    passed straight over it — the failure surfaced only when the ladder was
    finally needed. Asserting here means a bad rung fails on import instead of
    forty minutes into someone's escalation.
    """
    for index, rung in enumerate(LADDER):
        if (rung["blocks_to_swap"] > 0 or rung["swap_io"]) and rung["dit_offload"] == "none":
            raise ValueError(
                f"LADDER rung {index} ({rung['name']!r}) enables block swapping "
                f"(blocks_to_swap={rung['blocks_to_swap']}, "
                f"swap_io={rung['swap_io']}) but sets dit_offload='none'. "
                "The engine rejects this combination."
            )
        if rung["batch_cap"] < 5:
            raise ValueError(
                f"LADDER rung {index} ({rung['name']!r}) caps the frame window "
                f"at {rung['batch_cap']}; 5 is the minimum that gives any "
                "temporal context."
            )


def _variant_for(family: str, precision: str) -> str:
    """The variant key naming a given precision within a family."""
    for key, spec in VARIANTS.items():
        if spec["family"] == family and spec["precision"] == precision:
            return key
    raise KeyError(f"no {precision} variant for family {family!r}")

# Shared by every variant — the DiT changes, the VAE does not.
VAE_CHECKPOINT = "ema_vae_fp16.safetensors"

# Above this many frames, stream the clip in chunks rather than loading it all.
# Purely a host-RAM measure: decoded RGB frames at 1080p run ~6MB each, so a few
# hundred is nothing against this machine's RAM, and every chunk boundary is one
# more seam risk. Long-form footage is the only case that needs it.
_STREAM_THRESHOLD_FRAMES = 600

# Progressively harder memory settings, tried in order until one survives.
#
# Each rung is a full subprocess run, so a rung that OOMs costs time but leaves
# nothing behind.
#
# The ordering is driven by measurement on a 24GB 4090, not by the upstream
# README's advice. Per-phase peaks for a 640x360 -> 1920x1080 clip:
#
#                       batch 5, decode-tiled   batch 13, both tiled
#   VAE encode                14.50 GB                11.08 GB
#   DiT upscale                6.18 GB                 8.67 GB
#   VAE decode                11.29 GB                17.31 GB
#
# Three conclusions, all of which contradict the obvious approach:
#
#   1. The VAE dominates, not the DiT. `blocks_to_swap` only offloads DiT
#      blocks, so the headline BlockSwap feature is the *weakest* lever here —
#      it is applied late and never alone.
#   2. Frame window is the strongest lever. Decode peak scales with it directly
#      (11.3 GB at 5 frames, 17.3 GB at 13), so each rung caps it lower.
#   3. Untiled decode OOMs at 1080p output even at a 5-frame window, so tiling
#      is on from rung 0 rather than being an escalation step.
#
# Frame window is capped rather than set: a caller asking for 5 gets 5 at every
# rung. Lowering it costs temporal coherence, which is the reason to use SeedVR2
# over ESRGAN in the first place, so it is spent reluctantly.
#
# Rungs name a *precision*, not a variant. The family comes from whatever the
# caller chose, so escalating a 7B run yields a quantized 7B rather than
# silently substituting a 3B. `blocks_to_swap` is clamped to the family's
# ceiling at build time (32 for the 3B, 36 for the 7B).
#
# The peaks quoted above are the 3B's. The 7B roughly doubles the DiT phase but
# leaves the VAE phases identical, and since the VAE is what peaks, the same
# rung ordering holds; the 7B simply starts escalating sooner.
LADDER: List[Dict[str, Any]] = [
    {
        "name": "balanced",
        "precision": "fp8",
        "batch_cap": 13,
        "encode_tile": 1024,
        "decode_tile": 1024,
        "blocks_to_swap": 0,
        "swap_io": False,
        "vae_offload": "none",
        "dit_offload": "none",
    },
    {
        "name": "narrow window",
        "precision": "fp8",
        "batch_cap": 9,
        "encode_tile": 768,
        "decode_tile": 768,
        "blocks_to_swap": 0,
        "swap_io": False,
        "vae_offload": "cpu",
        "dit_offload": "none",
    },
    {
        "name": "narrow window + block swap",
        "precision": "fp8",
        # 9, not 5. Block swapping brings the measured peak down to 6.9GB at a
        # 5-frame window, so there is ample room to keep a window that actually
        # looks good — a 5-frame window scores worst of all for 4-frame stepping
        # (lag-4 0.47 vs 0.05 at 13), which would mean the ladder rescues the
        # run and hands back the exact artifact the user was trying to avoid.
        "batch_cap": 9,
        "encode_tile": 512,
        "decode_tile": 512,
        "blocks_to_swap": 16,
        "swap_io": False,
        "vae_offload": "cpu",
        # Must not be "none": BlockSwap needs somewhere to put the blocks it
        # swaps out, and the engine hard-errors rather than ignoring the flag.
        "dit_offload": "cpu",
    },
    {
        "name": "quantized + everything",
        "precision": "quant",
        # Last resort, but still 9: measured peak here is 3.9GB at a 5-frame
        # window, so the window is not what is saving us at this rung.
        "batch_cap": 9,
        "encode_tile": 384,
        "decode_tile": 384,
        "blocks_to_swap": 32,
        "swap_io": True,
        "vae_offload": "cpu",
        "dit_offload": "cpu",
    },
]


def _models_root() -> Path:
    """
    Resolve models_root the same way DiffSynthBackend does.

    Top-level key first (FUK convention), then the nested paths.models_root
    some older configs used. Kept in step with diffsynth_backend.py:220-226.
    """
    try:
        with open(_CONFIG_DIR / "defaults.json") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}

    root = cfg.get("models_root") or cfg.get("paths", {}).get("models_root")
    return Path(root).expanduser() if root else Path.home() / "ai" / "models"


def weights_dir() -> Path:
    override = os.environ.get("FUK_SEEDVR2_WEIGHTS")
    if override:
        return Path(override).expanduser()
    return _models_root() / "seedvr2"


def availability() -> Dict[str, Any]:
    """
    Report whether SeedVR2 can run, and what's missing if it can't.

    Weights are deliberately not required: the engine downloads them on first
    use (SHA256-validated), the same way TRELLIS and VGGT acquire theirs. Only
    the vendored source and its two extra imports are hard prerequisites.
    """
    missing: List[str] = []

    if not _ENGINE_CLI.exists():
        missing.append("SeedVR2 engine (fuk/vendor/SeedVR2)")

    import importlib.util

    for mod, hint in (("gguf", "gguf"), ("rotary_embedding_torch", "rotary_embedding_torch")):
        if importlib.util.find_spec(mod) is None:
            missing.append(f"python package '{hint}'")

    installed = [
        key for key, spec in VARIANTS.items()
        if (weights_dir() / spec["checkpoint"]).exists()
    ]

    return {
        "available": not missing,
        "missing": missing,
        # Every variant is offerable — anything not on disk is fetched on
        # demand. `downloaded` is what the UI uses to say so.
        "variants": list(VARIANTS),
        "downloaded": installed,
        "weights_dir": str(weights_dir()),
        "hint": INSTALL_HINT if missing else None,
    }


def download_weights(variants: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Fetch SeedVR2 weights ahead of time instead of on first upscale.

    Inference works without this — the engine downloads what it needs on demand
    — but that turns the first upscale into a silent multi-gigabyte stall. This
    exists so download_models.sh can get it out of the way with everything else.

    Idempotent: the engine checks for an existing file and validates its SHA256
    before deciding to fetch, so re-running costs a hash check, not a download.
    """
    status = availability()
    if not status["available"]:
        raise RuntimeError(
            "SeedVR2 engine not present: " + ", ".join(status["missing"])
            + f"\n  {INSTALL_HINT}"
        )

    target = weights_dir()
    target.mkdir(parents=True, exist_ok=True)

    # The engine's downloader lives in the vendored tree and imports its
    # siblings by package path, so it needs the engine root on sys.path.
    engine_root = str(_ENGINE_DIR)
    added = engine_root not in sys.path
    if added:
        sys.path.insert(0, engine_root)
    try:
        from src.utils.downloads import download_weight
    finally:
        if added:
            sys.path.remove(engine_root)

    wanted = variants or list(DEFAULT_DOWNLOAD_VARIANTS)
    fetched, failed = [], []
    for key in wanted:
        if key not in VARIANTS:
            raise ValueError(f"Unknown SeedVR2 variant: {key}")
        checkpoint = VARIANTS[key]["checkpoint"]
        print(f"  → {VARIANTS[key]['label']} ({checkpoint})", flush=True)
        try:
            ok = download_weight(checkpoint, VAE_CHECKPOINT, model_dir=str(target))
        except Exception as exc:
            ok = False
            print(f"    ✗ {exc}", flush=True)
        (fetched if ok else failed).append(key)

    return {"weights_dir": str(target), "fetched": fetched, "failed": failed}


def snap_batch_size(frames: int) -> int:
    """
    Snap a frame window to the nearest valid 4n+1 value.

    SeedVR2 batches frames as 4n+1 (1, 5, 9, 13, ...) because the VAE's temporal
    compression is 4x with one extra key frame. A value off that lattice is
    rejected deep inside the encoder. 5 is the floor: a batch of 1 is a single
    frame with no temporal context at all, which defeats the point of using
    SeedVR2 over ESRGAN.
    """
    if frames <= 5:
        return 5
    return ((frames - 1 + 2) // 4) * 4 + 1



_validate_ladder()

def default_temporal_overlap(batch_size: int) -> int:
    """
    Frames shared between consecutive batches, blended with a Hann crossfade.

    This must not be left at zero. SeedVR2 restores each batch as an independent
    diffusion problem, so two adjacent batches invent different fine detail; with
    no overlap the seam between them is a hard cut, which reads as the picture
    "stepping" every batch_size frames. The engine only runs its crossfade
    (generation_phases.py:979) when this is > 0.

    Scaled to the window rather than fixed: overlap costs real time, because the
    batch stride becomes batch_size - overlap and the number of batches rises
    accordingly. A quarter of the window is enough for the crossfade — which
    itself ramps over the middle third of the overlap — without doubling the run.
    Clamped to 16 (the engine's maximum) and kept strictly below batch_size,
    since the engine silently resets overlap >= batch_size back to 0.
    """
    return max(2, min(batch_size // 4, 16, batch_size - 1))


def _target_short_side(width: int, height: int, scale: int) -> int:
    """
    SeedVR2 takes an absolute *short side* target, not a scale factor.

    This differs from the old integration, which computed a longest-edge target
    — passing that here silently upscales by the aspect ratio on top of the
    requested scale.
    """
    short = min(width, height) * scale
    return short - short % 2


def _free_vram_for_seedvr2(log: Callable[[str], None]) -> None:
    """
    Evict cached DiffSynth pipelines before the subprocess launches.

    Mirrors ThreeDPipelineRunner._free_vram_for_reconstruction. The previous
    SeedVR2 integration skipped this entirely, so a resident Wan pipeline was
    still holding most of the card when the upscaler started — which is a
    plausible share of why it never fit.
    """
    try:
        import torch
    except ImportError:
        return

    # Reach the live backend through the web server's already-imported module.
    # sys.modules rather than an import: importing fuk_web_server here would
    # re-execute the server module and reset its cache root, which has caused a
    # double-load bug before.
    backend = None
    for mod_name in ("fuk_web_server", "ui.fuk_web_server"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            backend = getattr(mod, "diffsynth_backend", None)
            if backend is not None:
                break

    if backend is not None:
        cached = list(getattr(backend, "pipelines", {}).keys())
        if cached:
            log(f"Evicting {len(cached)} cached pipeline(s) to free VRAM")
            for key in cached:
                backend._evict_pipeline(key)

    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        log(f"VRAM before launch: {free / 1024**3:.1f}GB free / {total / 1024**3:.1f}GB")


def _build_command(
    input_path: Path,
    output_path: Path,
    rung: Dict[str, Any],
    short_side: int,
    max_resolution: int,
    batch_size: int,
    seed: int,
    chunk_size: int,
    temporal_overlap: int,
    prepend_frames: int,
) -> List[str]:
    checkpoint = VARIANTS[rung["variant"]]["checkpoint"]

    cmd = [
        sys.executable, str(_ENGINE_CLI), str(input_path),
        "--output", str(output_path),
        "--output_format", "mp4",
        "--video_backend", "ffmpeg",
        "--model_dir", str(weights_dir()),
        "--dit_model", checkpoint,
        "--resolution", str(short_side),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        # Pad the final batch out to a full window. Without this the last batch
        # is short, gets restored with less temporal context than every other
        # batch, and lands as a visible change in character at the tail. The
        # engine's own help calls this "recommended for optimal quality".
        "--uniform_batch_size",
        # sdpa unconditionally. FUK ships sageattention 1.0.6, but the engine
        # only offers sageattn_2/_3 — selecting those against 1.x fails at the
        # kernel call, well after the model is on the card.
        "--attention_mode", "sdpa",
        "--tensor_offload_device", "cpu",
        "--blocks_to_swap", str(rung["blocks_to_swap"]),
        # The per-phase "[VRAM] ... Peak: N GB" summaries only appear under
        # --debug, and they are the whole basis for reporting peak_vram_gb and
        # for judging whether a rung has headroom. Worth the extra output: it
        # is a handful of lines per batch, and it lands in the console panel
        # where the rest of the subprocess output already goes.
        "--debug",
    ]

    if max_resolution:
        cmd += ["--max_resolution", str(max_resolution)]
    if chunk_size:
        cmd += ["--chunk_size", str(chunk_size)]
    if temporal_overlap:
        cmd += ["--temporal_overlap", str(temporal_overlap)]
    if prepend_frames:
        # The first batch has no preceding frames to condition on, so it is
        # restored differently from every batch after it. Feeding it a reversed
        # run-up (auto-removed from the output) gives it context to match.
        cmd += ["--prepend_frames", str(prepend_frames)]
    if rung["swap_io"]:
        cmd += ["--swap_io_components"]
    if rung["vae_offload"] != "none":
        cmd += ["--vae_offload_device", rung["vae_offload"]]
    if rung["dit_offload"] != "none":
        cmd += ["--dit_offload_device", rung["dit_offload"]]

    # Overlap is kept at 1/8 of the tile. Below that the seams start showing on
    # flat gradients, which is exactly the content the VAE is worst at.
    if rung["encode_tile"]:
        size = rung["encode_tile"]
        cmd += [
            "--vae_encode_tiled",
            "--vae_encode_tile_size", str(size),
            "--vae_encode_tile_overlap", str(max(size // 8, 32)),
        ]
    if rung["decode_tile"]:
        size = rung["decode_tile"]
        cmd += [
            "--vae_decode_tiled",
            "--vae_decode_tile_size", str(size),
            "--vae_decode_tile_overlap", str(max(size // 8, 32)),
        ]

    return cmd


def _run_once(
    cmd: List[str],
    rung_label: str,
    progress_callback: Optional[Callable],
    progress_base: float,
    progress_span: float,
    timeout: int,
) -> Dict[str, Any]:
    """
    Run one attempt. Returns {"code", "tail", "timed_out", "peak_vram_gb"}.

    Never raises on a failed run — the caller decides whether a non-zero exit is
    an OOM worth escalating or a real error worth reporting.
    """
    tail: List[str] = []
    timed_out = False
    peak_vram: Optional[float] = None
    process = None
    code = 1

    # The engine announces phases as "Phase N: <name>", then counts batches
    # within each one ("Decoding batch 2/4"). Both are needed: the batch counter
    # restarts every phase, so a fixed band for batches would run the bar
    # backwards each time a new phase began.
    #
    # Matching the numbered banner rather than the phase word also matters —
    # bare words like "upscale" and "decode" occur in file paths and
    # tracebacks, which made the bar jump to "Restoring" on a file-not-found.
    phase_bands = {
        1: (0.05, 0.30, "Encoding to latents"),
        2: (0.30, 0.60, "Restoring"),
        3: (0.60, 0.92, "Decoding frames"),
        4: (0.92, 0.97, "Colour matching"),
    }
    band = phase_bands[1]

    def _emit(frac: float, label: str) -> None:
        if progress_callback:
            progress_callback(progress_base + progress_span * frac, label)

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
                # Fragmentation is what actually kills long runs on 24GB, and
                # the ladder's measured peaks were all taken with this on.
                **{var: "expandable_segments:True" for var in ALLOC_CONF_VARS},
            },
        )

        # Wall-clock watchdog. process.wait(timeout=...) after the read loop can
        # never fire: if the child wedges, its stdout never reaches EOF and the
        # loop blocks forever. Killing the child is what unblocks the read.
        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            print(f"[SeedVR2] attempt exceeded {timeout}s — terminating", flush=True)
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

                print(f"[SeedVR2] {line}", flush=True)
                tail.append(line)
                del tail[:-60]

                lowered = line.lower()

                # The engine prints a per-phase memory summary shaped like:
                #   [VRAM] 0.48GB allocated / 13.53GB reserved / Peak: 11.08GB /
                #          6.46GB free / 23.54GB total
                # Anchor on "Peak:" specifically — taking the first float in the
                # line yields the *allocated* figure, which is near zero between
                # phases and badly understates the run.
                match = _PEAK_VRAM_RE.search(line)
                if match:
                    value = float(match.group(1))
                    if peak_vram is None or value > peak_vram:
                        peak_vram = value

                if not progress_callback:
                    continue

                phase_hit = False
                for number, candidate in phase_bands.items():
                    if f"phase {number}" in lowered:
                        band = candidate
                        _emit(band[0], f"{rung_label}: {band[2]}")
                        phase_hit = True
                        break
                if phase_hit:
                    continue

                if "batch" in lowered and "/" in line:
                    for token in line.split():
                        if "/" not in token:
                            continue
                        done, _, total = token.partition("/")
                        try:
                            frac = int(done.strip("[](),")) / max(int(total.strip("[](),")), 1)
                        except ValueError:
                            continue
                        lo, hi, label = band
                        _emit(lo + (hi - lo) * frac, f"{rung_label}: {label} {token}")
                        break

            code = process.wait()
        finally:
            watchdog.cancel()
    finally:
        if process and process.poll() is None:
            process.kill()

    return {"code": code, "tail": tail, "timed_out": timed_out, "peak_vram_gb": peak_vram}


# Markers the engine/torch actually emit on exhaustion. `torch.OutOfMemoryError`
# is what a 3B fp8 decode raises on a 24GB card — the message is
# "Allocation on device" with no "out of memory" substring, so matching only the
# classic CUDA text misses it.
_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
)


def _is_oom(tail: List[str]) -> bool:
    joined = "\n".join(tail).lower()
    return any(marker in joined for marker in _OOM_MARKERS)


def upscale_video(
    input_path: Path,
    output_path: Path,
    scale: int = 2,
    variant: str = DEFAULT_VARIANT,
    frame_window: int = 0,
    seed: int = 42,
    resolution_cap: int = 1920,
    memory_mode: str = "auto",
    temporal_overlap: int = -1,
    # Off by default: the engine's help says the reversed run-up is
    # "auto-removed", but measured on a 45-frame clip, --prepend_frames 4
    # returns 49 frames. A silent change in clip length desyncs audio and
    # breaks frame bookkeeping downstream, which is not worth a marginal
    # improvement to the first batch. Opt in only if you check the count.
    prepend_frames: int = 0,
    motion_protection: float = 0.7,
    progress_callback: Optional[Callable] = None,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """
    Restore and upscale a video with SeedVR2.

    memory_mode
        "auto" walks the escalation ladder from the fastest configuration until
        one completes. A rung index ("0".."3") or rung name pins a single
        configuration and reports OOM rather than retrying.
    frame_window
        Frames per forward pass, snapped to the 4n+1 lattice. 0 uses 13.

        Wider is NOT monotonically better, which is the opposite of what the
        upstream docs imply. Measured on a 45-frame clip with the 7B, scoring
        invented detail (mean frame delta) and 4-frame stepping (lag-4
        autocorrelation, 0.08 in the source):

            window     5      9     13     17     21
            mean d  1.469  1.367  1.388  1.674  1.715
            lag-4    0.47  -0.16   0.05   0.33   0.36

        Both ends are bad. Too narrow and each batch has too little context to
        place detail consistently; too wide and the VAE's 4x temporal
        compression has to carry more motion than it can represent, so stepping
        returns. 9-13 is the usable band for clips of this length.
    resolution_cap
        Upper bound on any output edge, passed through as --max_resolution.
        SeedVR2's attention cost scales with resolution, so this is the knob
        that keeps a 4x on 1080p source from being hopeless.
    temporal_overlap
        Frames shared between consecutive batches and crossfaded. -1 derives it
        from the window (see default_temporal_overlap); 0 disables blending,
        which makes every batch boundary a visible step. Do not pass 0 unless
        you are deliberately measuring that artifact.
    prepend_frames
        Reversed run-up fed to the first batch so it has the same temporal
        context as the rest. Removed from the output automatically.
    motion_protection
        0..1. How far moving pixels are blended back toward the source to keep
        their motion blur. 0 disables the pass. Measured on a Wan clip at 0.7,
        the fast-moving region keeps only 19% of SeedVR2's sharpening while the
        still background keeps 78% of it — which is the whole point. Harmless on
        static footage, where the motion mask is near zero.
    """
    status = availability()
    if not status["available"]:
        raise RuntimeError(
            "SeedVR2 is not installed: " + ", ".join(status["missing"])
            + f"\n  {INSTALL_HINT}"
        )
    if variant not in VARIANTS:
        raise ValueError(f"Unknown SeedVR2 variant: {variant}")

    # Absolute, because the subprocess runs with cwd=_ENGINE_DIR so the engine
    # can find its bundled configs. A relative path would resolve against the
    # vendor directory instead of FUK's.
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_dir().mkdir(parents=True, exist_ok=True)

    # Probe the source for the short-side target. video_utils.get_video_info is
    # the shared helper — it already returns width/height and falls back to
    # OpenCV when ffprobe is absent.
    from .video_utils import get_video_info

    info = get_video_info(input_path)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if not width or not height:
        raise RuntimeError(f"Could not read dimensions from {input_path}")

    short_side = _target_short_side(width, height, scale)
    requested_batch = snap_batch_size(frame_window or 13)

    # Pick the rungs to try.
    if memory_mode == "auto":
        rungs = list(LADDER)
    else:
        try:
            rungs = [LADDER[int(memory_mode)]]
        except (ValueError, IndexError):
            matched = [r for r in LADDER if r["name"] == memory_mode]
            if not matched:
                raise ValueError(
                    f"Unknown memory_mode {memory_mode!r}. Use 'auto', an index "
                    f"0-{len(LADDER) - 1}, or one of: "
                    + ", ".join(r["name"] for r in LADDER)
                )
            rungs = matched

    # Bind each rung's precision to the chosen model's family, and clamp block
    # swapping to that architecture's ceiling. Asking for a quantized rung when
    # the caller already selected the quantized variant is a no-op, so the
    # ladder simply repeats that variant with harder memory settings.
    family = VARIANTS[variant]["family"]
    chosen_precision = VARIANTS[variant]["precision"]
    resolved = []
    for rung in rungs:
        # Never step *up* in precision: if the caller picked the quantized
        # variant, every rung stays quantized.
        precision = "quant" if chosen_precision == "quant" else rung["precision"]
        rung_variant = _variant_for(family, precision)
        resolved.append(dict(
            rung,
            variant=rung_variant,
            blocks_to_swap=min(rung["blocks_to_swap"],
                               VARIANTS[rung_variant]["max_blocks"]),
        ))
    rungs = resolved

    log = lambda msg: print(f"[SeedVR2] {msg}", flush=True)
    log(f"Source {width}x{height} -> short side {short_side}px (scale {scale}x)")
    log(f"Frame window {requested_batch} (4n+1), cap {resolution_cap}px, seed {seed}")

    # The engine writes here; the motion-protection pass then composites it
    # against the source into output_path. Kept alongside the final file so a
    # crash leaves the raw restoration behind for inspection rather than in a
    # temp dir that gets swept.
    engine_out = (output_path.with_name(output_path.stem + "_raw" + output_path.suffix)
                  if motion_protection > 0 else output_path)

    _free_vram_for_seedvr2(log)

    if progress_callback:
        progress_callback(0.03, "Starting SeedVR2")

    last: Optional[Dict[str, Any]] = None
    batch_size = requested_batch
    overlap = 0
    for index, rung in enumerate(rungs):
        batch_size = min(requested_batch, rung["batch_cap"])
        overlap = (default_temporal_overlap(batch_size) if temporal_overlap < 0
                   else min(temporal_overlap, batch_size - 1))
        label = f"rung {index} ({rung['name']})"
        log(f"Attempt {index + 1}/{len(rungs)} — {label}, "
            f"{VARIANTS[rung['variant']]['label']}, window={batch_size}, "
            f"overlap={overlap}, "
            f"tiles={rung['encode_tile']}/{rung['decode_tile']}, "
            f"blocks_to_swap={rung['blocks_to_swap']}")
        if batch_size < requested_batch:
            log(f"  frame window reduced {requested_batch} -> {batch_size} "
                f"(temporal coherence will suffer)")

        # Only stream in chunks when the clip is long enough to be worth it.
        # Chunking is a host-RAM measure, and every chunk boundary is another
        # place for continuity to break; a typical FUK clip is a few hundred
        # frames at most, so load the whole thing and keep one continuous run.
        frame_count = int(info.get("frame_count") or 0)
        chunk_size = 0 if frame_count <= _STREAM_THRESHOLD_FRAMES else batch_size * 10

        cmd = _build_command(
            input_path, engine_out, rung, short_side, resolution_cap,
            batch_size, seed, chunk_size=chunk_size,
            temporal_overlap=overlap, prepend_frames=prepend_frames,
        )

        # Give each attempt a slice of the bar so an escalation doesn't appear
        # to run the progress backwards.
        span = 0.94 / len(rungs)
        last = _run_once(
            cmd, label, progress_callback,
            progress_base=0.03 + span * index, progress_span=span,
            timeout=timeout,
        )

        if last["timed_out"]:
            raise RuntimeError(
                f"SeedVR2 timed out after {timeout}s.\n"
                "Long clips at high resolution are the usual cause. Lower the "
                "scale, reduce the resolution cap, or split the clip.\n"
                + "\n".join(last["tail"][-10:])
            )

        if last["code"] == 0 and engine_out.exists():
            break

        if _is_oom(last["tail"]) and index + 1 < len(rungs):
            log(f"{label} ran out of VRAM — escalating to a lower-memory configuration")
            _free_vram_for_seedvr2(log)
            continue

        detail = "\n".join(last["tail"][-15:])
        if _is_oom(last["tail"]):
            raise RuntimeError(
                "SeedVR2 ran out of VRAM at every configuration.\n"
                "  Reduce Scale, lower the resolution cap, or set a smaller "
                "frame window in Postprocess settings.\n" + detail
            )
        raise RuntimeError(f"SeedVR2 exited with code {last['code']}\n{detail}")

    if not engine_out.exists():
        raise RuntimeError("SeedVR2 reported success but wrote no output file")

    # Motion-adaptive protection. SeedVR2 de-blurs, and on footage whose
    # smoothness depends on motion blur that reads as stepping — see
    # motion_protect for the measurements. Blending moving pixels back toward
    # the source restores the blur where it matters and nowhere else. A clip
    # with no movement produces a near-zero mask and passes through unchanged,
    # so this is safe to leave on.
    protection: Optional[Dict[str, Any]] = None
    if motion_protection > 0:
        from .motion_protect import apply_motion_protection

        if progress_callback:
            progress_callback(0.95, "Motion protection")
        protection = apply_motion_protection(
            source_path=input_path,
            restored_path=engine_out,
            output_path=output_path,
            strength=motion_protection,
            log=log,
        )
        engine_out.unlink(missing_ok=True)
    else:
        engine_out.replace(output_path)

    out_info = get_video_info(output_path)

    # SeedVR2 must return the clip it was given, frame for frame. Some engine
    # options (--prepend_frames) quietly change the count, which desyncs audio
    # and breaks anything downstream that pairs frames with the source.
    in_frames = int(info.get("frame_count") or 0)
    out_frames = int(out_info.get("frame_count") or 0)
    if in_frames and out_frames and in_frames != out_frames:
        log(f"WARNING: frame count changed {in_frames} -> {out_frames}. "
            f"Check prepend_frames/uniform_batch_size; downstream timing will drift.")

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "method": "seedvr2",
        "variant": rung["variant"],
        "memory_rung": rung["name"],
        "scale": scale,
        "frame_window": batch_size,
        "temporal_overlap": overlap,
        "motion_protection": motion_protection,
        "motion_mask_coverage": protection["mask_coverage"] if protection else None,
        "frame_count": int(out_info.get("frame_count") or 0),
        "input_size": {"width": width, "height": height},
        "output_size": {
            "width": int(out_info.get("width") or 0),
            "height": int(out_info.get("height") or 0),
        },
        "peak_vram_gb": last.get("peak_vram_gb") if last else None,
        "output_path": str(output_path),
    }
