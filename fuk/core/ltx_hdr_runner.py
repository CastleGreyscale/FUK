"""
LTX-2.5 SDR -> HDR — subprocess driver.

Not imported by FUK. ltx_hdr_backend launches this file with the vendored
LTX-2 packages on PYTHONPATH, the same way seedvr2_backend launches the
SeedVR2 CLI.

Why a driver of our own rather than `python -m ltx_pipelines.hdr_ic_lora`
    Upstream's main() is about thirty lines around HDRICLoraPipeline, and it
    gets two things wrong for FUK:

    1. It requires an 8k+1 frame count and refuses anything else. FUK pads the
       source up to the next 8k+1 by holding the last frame, so the decoded
       clip has to be trimmed back before anything is written. Trimming the
       tensor here means the EXRs and the HLG master are both written at the
       source length; trimming afterwards would mean deleting EXR files and
       re-encoding the master.
    2. It writes at round(fps). The HDR encode path takes a float, so a
       23.976 source keeps its rate here instead of coming back at 24.
    3. It encodes the whole clip as one chunk with the pipeline still alive,
       which OOMs on 24GB at 121 frames of 1080p (see the encode call).
    4. A decode OOM throws away the denoise. Upstream sizes its decode tiles
       once, from free VRAM before denoising starts, and its estimate counts
       activations at 2 bytes while the HDR path decodes in float32. On a 24GB
       card a 121-frame 1080p clip denoised for eight minutes and then died on
       the first decode tile. _retry_decode_on_oom re-plans the tiling at half
       the budget and decodes the same latent again.

    Everything else — the pipeline, the ACEScct transforms, seam keyframes,
    keyframe-aware decode, the EXR + HLG writers — is upstream's, unchanged.

Output protocol
    Upstream's own logging goes to stdout as-is (the host parses it for
    progress). The last line is `FUK_RESULT {json}`; the host treats a run
    without it as failed.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch

log = logging.getLogger("fuk.ltx_hdr")

# Each retry halves the budget handed to upstream's tiling planner. Three
# retries reach an eighth of the original — below that the planner finds no
# legal tile and raises, which ends the loop anyway.
_DECODE_RETRIES = 3

# Frames per encode chunk. EXR numbering runs on across chunks upstream.
_ENCODE_CHUNK_FRAMES = 16


def _retry_decode_on_oom(pipeline) -> None:
    """
    Wrap HDRICLoraPipeline._decode_video so a decode OOM retries with smaller
    tiles instead of discarding the denoised latent.

    Patches a private method of the pinned upstream checkout; if a bump changes
    its signature this fails loudly on the first decode rather than silently.
    """
    from ltx_core.devices import cuda_activation_budget_bytes
    from ltx_pipelines.utils.helpers import tiling_config_for_vae

    original = pipeline._decode_video

    def decode(video_state, tiling_config, generator, **kwargs):
        _, _, _, lat_h, lat_w = video_state.latent.shape
        budget = None
        for attempt in range(_DECODE_RETRIES + 1):
            try:
                return original(video_state, tiling_config, generator, **kwargs)
            except torch.OutOfMemoryError:
                if attempt == _DECODE_RETRIES:
                    raise
            # Outside the handler: while the exception is live its traceback
            # pins the failed decode's activations, and empty_cache frees nothing.
            gc.collect()
            torch.cuda.empty_cache()
            budget = (budget or cuda_activation_budget_bytes(pipeline.device)) // 2
            try:
                tiling_config = tiling_config_for_vae(
                    pipeline.video_decoder.checkpoint_path,
                    height=lat_h * 32,
                    width=lat_w * 32,
                    num_frames=kwargs["gen_frames"],
                    diffvae_optimization=pipeline.video_decoder.diffvae_optimization,
                    device=pipeline.device,
                    free_bytes=budget,
                    keyframes=bool(kwargs.get("generated_kf")),
                )
            except ValueError:
                # No legal tile fits this budget — surface it as the OOM it is.
                raise torch.OutOfMemoryError(
                    f"VAE decode does not fit: no tile layout within {budget / 1024**3:.1f}GB"
                ) from None
            log.warning("Decode OOM — retrying with tiling planned for %.1fGB (attempt %d/%d)",
                        budget / 1024**3, attempt + 2, _DECODE_RETRIES + 1)

    pipeline._decode_video = decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", required=True, help="SDR video (display-referred sRGB/Rec.709)")
    parser.add_argument("--keep-frames", type=int, required=True,
                        help="Source frame count before 8k+1 padding; output is trimmed to this")
    parser.add_argument("--output-path", required=True, help="HLG master .mp4; EXRs go beside it")
    parser.add_argument("--transformer-path", required=True)
    parser.add_argument("--video-vae-path", required=True)
    parser.add_argument("--hdr-lora", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--exr-colorspace", default="acescg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", default="none", choices=["none", "cpu", "disk"])
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--keyframe-strength", type=float, default=0.95)
    parser.add_argument("--no-keyframes", action="store_true")
    parser.add_argument("--high-quality", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_pipelines.hdr_ic_lora import _DEFAULT_QUANTIZATION, HDRICLoraPipeline
    from ltx_pipelines.utils.media_io import EXRColorSpace, VideoInput, encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode

    with torch.inference_mode():
        pipeline = HDRICLoraPipeline(
            model_paths=ModelPaths.from_split(
                transformer_path=args.transformer_path,
                video_vae_path=args.video_vae_path,
            ),
            hdr_lora=args.hdr_lora,
            text_embeddings_path=args.text_embeddings,
            offload_mode=OffloadMode(args.offload),
            quantization=None if args.no_quantization else _DEFAULT_QUANTIZATION,
        )
        _retry_decode_on_oom(pipeline)

        acescct_hdr, fps = pipeline(
            video=VideoInput(path=Path(args.input), gamma_encoded=True),
            seed=args.seed,
            tiling_config=AUTO_TILING,
            high_quality_hdr=args.high_quality,
            keyframe_strength=None if args.no_keyframes else args.keyframe_strength,
        )
        generated = int(acescct_hdr.shape[0])
        acescct_hdr = acescct_hdr[: args.keep_frames]

        # Free everything the pipeline still holds before encoding, and hand
        # the encoder chunks rather than the whole clip: per chunk it makes a
        # linear Rec.709 copy for HLG and another in the EXR primaries, so one
        # 121-frame 1080p chunk wants three ~3GB float32 volumes at once. That
        # OOMed on 24GB after a successful decode.
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        exr_dir = encode_video(
            video=iter(acescct_hdr.split(_ENCODE_CHUNK_FRAMES)),
            fps=fps,
            audio=None,
            output_path=args.output_path,
            video_chunks_number=1,
            color_space=EXRColorSpace[args.exr_colorspace.upper()],
        )

    peak = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
    print("FUK_RESULT " + json.dumps({
        "exr_dir": str(exr_dir) if exr_dir else None,
        "fps": float(fps),
        "frames": int(acescct_hdr.shape[0]),
        "generated_frames": generated,
        "height": int(acescct_hdr.shape[1]),
        "width": int(acescct_hdr.shape[2]),
        "peak_vram_gb": round(peak, 2) if peak is not None else None,
    }), flush=True)


if __name__ == "__main__":
    main()
