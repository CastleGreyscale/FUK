# Vendor patches

Local fixes FUK applies on top of pinned vendor source trees. `setup.sh` clones each
vendor repo at a fixed commit and then replays these with `git apply`, so a fresh install
gets byte-identical code to a development machine.

Everything else under `fuk/vendor/` is gitignored. These patches are the exception —
they are the only record of our changes, so they must stay version-controlled.

## Active patches — DiffSynth-Studio

Base commit: `acf2ad29c65f436633f57beec1caaa1296760331` (version 2.1.9, 2026-10-09).

Moved from 2.1.7 for two things FUK runs that 2.1.7 does not have: Qwen-Image-2.1 itself
(2.1.8) and the control branch `qwen_image_21_control` loads as a fourth component (2.1.9,
PR #1730). None of the three patched files changed between the two, so all three patches
apply unmodified.

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

### `0003-nonrecurse-wrapper-tensor-device.patch`

`diffsynth/core/vram/layers.py` — `AutoWrappedNonRecurseModule.cast_to`, added at the
LTX-2.5 swap when every generation died with *"Expected all tensors to be on the same
device, but found at least two devices, cuda:0 and cpu"* inside the text encoder.

That wrapper deliberately does not move the block it wraps: its children are wrapped
individually and cast themselves at their own forward, which is what "parameter casting is
implemented in the model architecture" in the original comment means. The contract holds
for every block DiffSynth wraps this way, because they are all DiffSynth's own and own
nothing but small modulation / scale-shift tables that their forward casts itself —
`wan_video_dit.DiTBlock`, `ltx2_dit.BasicAVTransformerBlock`,
`ace_step_dit.AceStepDiTLayer` and the rest.

The LTX-2.5 text-encoder map is the exception: it wraps a foreign class, transformers'
`Gemma4UnifiedTextDecoderLayer`, whose forward ends with
`hidden_states *= self.layer_scalar`. By then the wrapped children have put the
activations on the GPU while that scalar sits wherever the block was onloaded.

Worth knowing where `layer_scalar` comes from, because it is not where the traceback
suggests. Gemma4 registers it as a *buffer*, but `ltx25_text_encoder.py` converts it to a
frozen Parameter at construction — deliberately, because DiffSynth's disk-offload reload
restores `named_parameters()` only and a persistent buffer would never be fetched back.
So the stranded tensor is a **parameter**, and a fix that moves only buffers does nothing
at all. (Ask the loaded model: `layer_scalar` is absent from every `_buffers` and present
in `_parameters`, 48 of them.)

It only bites under pressure, which is why it can read as intermittent: with the "low"
preset the onload device is the CPU and `forward` normally calls `preparing()` first,
moving the whole block — scalar included — to CUDA. Once resident weights pass the VRAM
limit, `check_free_vram()` returns False, `preparing()` is skipped, and `computation()`
falls through to `cast_to`, which returned the block untouched. On a 24GB card with a 26GB
text encoder that is every run.

The patch moves the block's own parameters and buffers (`recurse=False` — the children own
the actual weights) to the computation device. That is kilobytes per block, and for
DiffSynth's own blocks it only pre-does what their forward already does each step;
`offload()` moves everything back. Device only, never dtype: some of these tensors are
deliberately kept above the computation precision.

Verified by loading the real 26GB encoder outside the server and encoding a prompt: fails
before, succeeds after, and the offload that follows is clean (the parameters are swapped
under `torch.inference_mode()`, which is how generation runs).

Regression signal when upgrading: run any LTX-2.5 generation on the `low` preset with the
text encoder resident. If upstream fixes this themselves — by moving the block's own
tensors in the wrapper, by leaving `layer_scalar` a buffer, or by dropping
`Gemma4UnifiedTextDecoderLayer` from the map — drop this patch.

### `0004-ltx2-ref-list-aliasing.patch`

`diffsynth/pipelines/ltx2_audio_video.py` — `model_fn_ltx2`, added when every LTX-2.5
generation that combined a control video with a last frame died with a CUDA OOM of
1-2GB partway into the denoise, at sizes that run fine without the control video.

Upstream builds the reference-token list with
`total_ref_latents = ref_frames_latents if ref_frames_latents is not None else []` and
then `total_ref_latents += [in_context_video_latents]`. When `ref_frames_latents` is a
real list that is an alias, not a copy, and `+=` on a list extends in place — so the
in-context latents are appended to the list living in `inputs_shared`, and it is never
reset. Every forward pass (two per step with CFG on) therefore adds one more copy of the
control video to the sequence. The positions list has the same bug.

It needs both inputs to bite, which is why it reads as "openpose is broken" rather than
"LTX is broken": `ref_frames_latents` is only a list when a keyframe other than index 0 is
given (FUK's last frame), and nothing is appended unless `in_context_videos` is set. A
control video alone starts from a fresh `[]` each call, and a last frame alone appends
nothing.

Measured on CPU with a stub DiT at 832x480x81 (control at half resolution): the video
sequence should be 5824 tokens every call and instead went 5824, 6968, 8112, 9256… —
+1144 per call, about 12x the intended length by the end of a 30-step CFG run. After the
patch it holds at 5824.

The patch copies both lists before extending them. Regression signal when upgrading: an
LTX-2.5 job with a first frame, a last frame and a control video; watch that per-step time
stays flat instead of climbing. Drop the patch if upstream stops mutating its arguments.

## Dropped patches

### `0003-ltx25-optional-gemma4-import.patch` — dropped at the LTX-2.5 swap

(The number was freed and immediately reused by the tensor-device patch above. Same day,
same model, different problem — do not confuse the two.)

`diffsynth/models/ltx25_text_encoder.py`. Added at the 2.1.7 upgrade and removed a day
later, when `ltx2` was repointed from LTX-2 to LTX-2.5 and the dependency it worked around
had to be faced rather than deferred.

LTX-2.5's text encoder does
`from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration`, and
those symbols did not exist in transformers 5.1.0. The blast radius was much larger than
LTX-2.5: `pipelines/ltx2_audio_video.py` imports that module unconditionally and
`diffsynth_backend._setup_diffsynth_env` imports *that* eagerly at startup, so the missing
symbol took down the entire FUK backend, every model included. `LTX25TextEncoder`
subclasses `Gemma4UnifiedForConditionalGeneration` in its class body, so deferring the
import into a function was not an option — the patch caught the `ImportError` and
substituted stubs that raised a message naming the cause.

The fix is now the pin: `transformers>=5.10,<5.13` in `pyproject.toml`. Gemma4Unified first
ships in **5.10.0** (absent in 5.9.0). The upper bound is what keeps this cheap — 5.10.0
through 5.12.1 still declare `tokenizers>=0.22,<=0.23` and `safetensors>=0.4.3`, which the
existing 0.22.2 and 0.4.5 satisfy, so the upgrade moved only transformers,
huggingface_hub, click and hf-xet. From 5.13.0 `safetensors>=0.8.0` is required and from
5.16 `tokenizers>=0.23.1`; raise the ceiling only with a reason and a regression pass.

What to watch if that ceiling moves: DiffSynth reaches into transformers privates —
`models.qwen3.modeling_qwen3`, `qwen2_5_vl`, `qwen3_vl`, `siglip`, `dinov3_vit`,
`cache_utils`, `processing_utils`, `modeling_flash_attention_utils`. All present at
v5.12.1; none of them are public API and any of them can move in a minor release.

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
