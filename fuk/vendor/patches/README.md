# Vendor patches

Local fixes FUK applies on top of pinned vendor source trees. `setup.sh` clones each
vendor repo at a fixed commit and then replays these with `git apply`, so a fresh install
gets byte-identical code to a development machine.

Everything else under `fuk/vendor/` is gitignored. These patches are the exception —
they are the only record of our changes, so they must stay version-controlled.

## Active patches — DiffSynth-Studio

Base commit: `c458cb42ab1ee838bff85c6546e14bb01c3571e9` (version 2.1.7, 2026-09-14).

### `0001-wan-vace-seq-len-clamp.patch`

`diffsynth/models/wan_video_vace.py` — `VaceWanModel.forward` right-pads the vace context
to the latent sequence length with `u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))`.
That assumes the context is never *longer* than the latent sequence. When a reference
image pushes it past `x`, the length goes negative and the `cat` raises. Clamp with a
slice when the context is already long enough.

### `0002-wan-vace-vram-and-untiled-ref-encode.patch`

`diffsynth/pipelines/wan_video.py` — `WanVideoUnit_VACE.process`, two related fixes:

1. **VRAM.** Upstream keeps `vace_video`, `inactive`, `reactive` and all their latents
   live simultaneously. At 720p that alone OOMs a 24GB card before denoising starts.
   Free each intermediate as it is consumed and `empty_cache()` once at the end.
2. **Ring artifacts.** Upstream encodes the reference frames with the same `tiled=True`
   settings as the video. Tiled encoding of a single 832x480 frame leaves tile-boundary
   seams in the conditioning latent, which the DiT then reinforces at every denoising
   step — visible as concentric rings on curved surfaces. Reference frames are typically
   1-2 images, so encode them untiled; the extra VRAM is negligible.

   This is the regression signal to watch when upgrading: render a Wan VACE job with a
   reference image and inspect curved surfaces for ringing.

### `0003-ltx25-optional-gemma4-import.patch`

`diffsynth/models/ltx25_text_encoder.py` — added at the 2.1.7 upgrade. LTX-2.5 arrived in
`a98c6d4` and its text encoder does
`from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration`.
Those symbols do not exist in transformers 5.1.0, which is what FUK has installed, and
upstream leaves `transformers` unpinned so nothing flags the mismatch.

The blast radius is much larger than LTX-2.5. `pipelines/ltx2_audio_video.py` imports this
module unconditionally, and `diffsynth_backend._setup_diffsynth_env` imports *that*
eagerly at startup — so the missing symbol took down the entire FUK backend, every model
included, not just the one nobody had registered yet.

`LTX25TextEncoder` subclasses `Gemma4UnifiedForConditionalGeneration` in its class body, so
deferring the import into a function is not an option — the name has to resolve at module
import. The patch instead catches the `ImportError` and substitutes stubs that raise a
message naming the real cause if anything actually tries to construct them. LTX-2 is
untouched and keeps working; only a genuine LTX-2.5 load fails.

Drop this patch as soon as FUK pins a transformers that ships Gemma4Unified (5.1.0 does
not; latest on PyPI at the time of writing is 5.17.0, unverified). Upgrading transformers
under the Qwen, FLUX and Krea2 pipelines is its own piece of work and was deliberately not
bundled into the DiffSynth upgrade.

## Dropped patches

### `layers.py` LoRA device co-location — dropped at the 2.1.5 upgrade

FUK previously patched `AutoWrappedLinear` in `diffsynth/core/vram/layers.py` with a
`_move_lora_weights()` helper, called from `offload`/`onload`/`preparing`. `lora_A_weights`
and `lora_B_weights` are plain Python lists, so `torch.nn.Module.to()` does not see them
and they were left stranded on the wrong device when the surrounding weights moved.

Upstream fixed the same bug independently: LoRA handling moved into a `LoRAHotLoadMixin`,
whose `lora_forward` now casts inline with
`lora_A.T.to(device=x.device, dtype=x.dtype)`. FUK drives LoRA through the ordinary
`load_lora`/`clear_lora` path, which that fix covers, so the patch is redundant.

Caveat worth revisiting: upstream casts on *every* forward rather than moving the tensors
once, which is strictly more work per step. If LoRA generation shows a measurable slowdown
versus 2.0.4, a perf-oriented variant of this patch can come back — but only backed by a
benchmark, not on suspicion. The original is preserved under `.orig-2.0.4/`.

## `.orig-2.0.4/`

Snapshot of the pre-upgrade patched files and the full working-tree diff as it stood
against DiffSynth `c0f7e1db` (2.0.4). Reference material for the upgrade only; nothing
reads these at build time. Safe to delete once 2.1.5 has been in use for a while.
