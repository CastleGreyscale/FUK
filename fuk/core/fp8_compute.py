"""
FP8 matmul for a denoiser's transformer-block linears.

Ada (sm_89) and Hopper tensor cores run FP8 matmuls at twice the bf16 rate.
Measured on the 4090 at Wan 2.2 FFN shapes: bf16 27.9 ms, FP8 13.8 ms.

Why this doesn't use DiffSynth's own fp8_linear (core/vram/layers.py): it scales
activations per row and leaves weights unscaled. torch._scaled_mm only has a
fast per-row kernel on Hopper — on Ada the same matmul took 51.4 ms, slower than
bf16 — and Wan's weights (~0.02) sit near FP8's precision floor unscaled. Here
both operands get a per-tensor scale, which is the fast path on Ada.

This is lossy: images drift visibly (FLUX.2-klein ~30 dB vs bf16), which is why
it is a user-selected precision mode and off for images by default.

Only linears inside numbered transformer blocks are switched. Embeddings,
modulation (adaLN), norms and the output head stay bf16 — they are tiny in
compute and the most sensitive to rounding. Text encoders and VAEs are never
passed in.

attach()/detach() work on a loaded model, so toggling costs no reload. They
hook DiffSynth's per-instance linear_forward on its offload wrapper, or forward
on a bare nn.Linear; no class is modified.
"""

import re
import types
from typing import Iterable, List

import torch
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8).max  # 448

_INCLUDE = re.compile(r"blocks\.\d+\.")
# Spelled out: a bare "mod" would also match Wan's "blocks.N.module.*" wrappers.
_EXCLUDE = re.compile(r"modulation|img_mod|txt_mod|norm|adaln|embed", re.I)


def supported() -> bool:
    """FP8 tensor cores: Ada (8.9) and newer NVIDIA GPUs."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    k, n = x.shape[-1], weight.shape[0]
    # _scaled_mm needs both inner dims divisible by 16.
    if k % 16 or n % 16 or x.numel() == 0:
        return F.linear(x, weight, bias)
    shape = x.shape
    x2 = x.reshape(-1, k)
    sx = (x2.abs().amax().float() / _FP8_MAX).clamp(min=1e-12)
    sw = (weight.abs().amax().float() / _FP8_MAX).clamp(min=1e-12)
    xq = (x2 / sx.to(x2.dtype)).to(FP8)
    wq = (weight / sw.to(weight.dtype)).to(FP8)
    out = torch._scaled_mm(
        xq, wq.t(), scale_a=sx, scale_b=sw,
        bias=None if bias is None else bias.to(x.dtype),
        out_dtype=x.dtype,
    )
    return out.reshape(*shape[:-1], n)


def _wrapped_linear_forward(self, x, weight, bias):  # DiffSynth AutoWrappedLinear hook
    return fp8_linear(x, weight, bias)


def _plain_forward(self, x):  # bare nn.Linear (no offload wrapper)
    return fp8_linear(x, self.weight, self.bias)


def attach(model: torch.nn.Module) -> int:
    """Switch the model's block linears to FP8. Returns how many were switched."""
    count = 0
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or getattr(m, "_fuk_fp8", False):
            continue
        if not _INCLUDE.search(name) or _EXCLUDE.search(name):
            continue
        if hasattr(m, "linear_forward"):
            m.linear_forward = types.MethodType(_wrapped_linear_forward, m)
        else:
            m.forward = types.MethodType(_plain_forward, m)
        m._fuk_fp8 = True
        count += 1
    return count


def detach(model: torch.nn.Module) -> None:
    """Undo attach(): the class methods take over again."""
    for m in model.modules():
        if getattr(m, "_fuk_fp8", False):
            for attr in ("linear_forward", "forward"):
                m.__dict__.pop(attr, None)
            del m._fuk_fp8


def denoisers(pipeline) -> List[torch.nn.Module]:
    """The denoiser module(s) of a DiffSynth pipeline (Wan 2.2 has two experts)."""
    return [m for m in (getattr(pipeline, a, None) for a in ("dit", "dit2"))
            if isinstance(m, torch.nn.Module)]


def set_enabled(pipeline, enabled: bool) -> int:
    """Attach or detach on every denoiser in the pipeline. Returns linears switched."""
    total = 0
    for m in denoisers(pipeline):
        if enabled:
            total += attach(m)
        else:
            detach(m)
    return total
