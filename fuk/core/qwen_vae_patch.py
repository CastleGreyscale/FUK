"""
Qwen VAE memory fixes for FUK.

Two independent pieces, both aimed at the same failure: decoding a large
still image through Qwen's video-shaped VAE used to OOM at the very last
step of generation.

  1. `QwenImageCausalConv3d` folded to a 2D conv for single-frame input.
     This is the actual fix, and it is exact — see the note on the patch.

  2. `tiled_vae_decode`, a spatial fallback for whatever the fold cannot
     make fit. Only ever used when a decode does not have the VRAM for it,
     since tiling makes the VAE's mid-block attention per-tile.

Importing this module applies (1). It is imported from diffsynth_backend so
every entry point that touches a Qwen VAE — generation, EXR export, chained
runs — gets it, not just the image pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffsynth.models.qwen_image_vae import QwenImageCausalConv3d


# --- 1. Single-frame causal conv3d -> conv2d ------------------------------

_causal_conv3d_original = QwenImageCausalConv3d.forward


def _causal_conv3d_patched(self, x, cache_x=None):
    """Run 1-frame input as a 2D conv. Exact, not an approximation.

    A still image goes through this VAE as a 1-frame video. The causal
    temporal padding is *zeros* (F.pad's default mode), so with D=1 the
    padded stack is [0, ..., 0, x] and every temporal kernel tap except the
    last is multiplied by zero. What is left is precisely a 2D convolution
    with weight[:, :, -1] — matches the 3D path to float64 round-off.

    Worth special-casing because conv3d has no cuDNN algorithm for this
    shape and falls back to im2col, whose workspace is
    in_ch * kt*kh*kw * H * W elements. For the decoder's last 96->96 block
    at 2048x2048 that is a single 20.25 GiB allocation, which is what used
    to blow up at the end of a long generation. The conv2d path peaks at
    3.0 GiB for the same output.

    Guarded on 2*temporal_pad == kt-1, which is what makes the real frame
    line up with the last tap. Every conv in this VAE satisfies it (k=3/p=1
    and k=1/p=0); anything that does not falls through to the vendor path.
    """
    if (
        cache_x is None
        and x.shape[2] == 1
        and self.dilation[0] == 1
        and self._padding[4] == self.kernel_size[0] - 1
    ):
        x = F.pad(x[:, :, 0], self._padding[:4])
        x = F.conv2d(
            x, self.weight[:, :, -1], self.bias,
            stride=self.stride[1:], dilation=self.dilation[1:], groups=self.groups,
        )
        return x.unsqueeze(2)
    return _causal_conv3d_original(self, x, cache_x)


QwenImageCausalConv3d.forward = _causal_conv3d_patched


# --- 2. Tiled decode fallback ---------------------------------------------

# Bytes of peak GPU allocation per output pixel for a bf16 Qwen VAE decode on
# the folded conv2d path. Measured end-to-end at 1346 B/px, holding to within
# 0.3% from 512x512 up to 2048x2048 and across aspect ratios — the decoder's
# full-resolution stage dominates and scales with H*W. Carries ~20% margin for
# the extra copy diffsynth's VRAM manager can make when it casts a module.
_DECODE_BYTES_PER_PIXEL = 1600

# Never tile below this, in latent units — small tiles cost quality at the
# seams and buy little, and the VAE needs some context to work with.
_MIN_TILE = 32


def estimate_decode_bytes(latent: torch.Tensor, upsampling_factor: int = 8) -> int:
    """Rough peak allocation for decoding `latent`, in bytes.

    Deliberately an overestimate — being wrong low means an OOM at the end
    of a generation, being wrong high means a needless tiled decode.
    """
    h, w = latent.shape[-2:]
    pixels = h * w * upsampling_factor * upsampling_factor
    return int(pixels * _DECODE_BYTES_PER_PIXEL)


def _blend_ramp(length: int, head: bool, tail: bool, border: int, **kw) -> torch.Tensor:
    """1D weight ramp: linear fade in over `border` unless this edge is the image edge."""
    # A short final tile can be narrower than two borders; without the clamp the
    # two ramps would overlap and write over each other.
    border = min(border, length // 2)
    x = torch.ones(length, **kw)
    if border > 0:
        ramp = (torch.arange(border, **kw) + 1) / border
        if not head:
            x[:border] = ramp
        if not tail:
            x[-border:] = ramp.flip(0)
    return x


def _tile_starts(n: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering [0, n), stopping once a tile reaches the end."""
    starts = []
    for s in range(0, n, stride):
        if starts and starts[-1] + tile >= n:
            break
        starts.append(s)
    return starts


def tiled_vae_decode(decode_fn, latent: torch.Tensor, tile: int, stride: int) -> torch.Tensor:
    """Decode a (B, C, h, w) latent in overlapping spatial tiles.

    `decode_fn` takes a latent tile and returns (B, 3, H, W). The upsampling
    factor is read off the first tile rather than assumed, so this does not
    care which VAE it is driving.

    Overlaps are cross-faded, which hides the seams but does not eliminate
    them entirely: the VAE's mid-block attention sees one tile at a time, so
    a tiled decode is not bit-identical to a whole-image one.
    """
    h, w = latent.shape[-2:]

    hs = _tile_starts(h, tile, stride)
    ws = _tile_starts(w, tile, stride)

    out = None
    weight = None
    factor = None

    for y in hs:
        y_ = min(y + tile, h)
        for x in ws:
            x_ = min(x + tile, w)
            piece = decode_fn(latent[:, :, y:y_, x:x_])

            if out is None:
                factor = piece.shape[-2] // (y_ - y)
                out = torch.zeros(
                    (*piece.shape[:2], h * factor, w * factor),
                    dtype=torch.float32, device=piece.device,
                )
                weight = torch.zeros(
                    (1, 1, h * factor, w * factor),
                    dtype=torch.float32, device=piece.device,
                )

            border = (tile - stride) * factor
            kw = {"dtype": torch.float32, "device": piece.device}
            mask = torch.minimum(
                _blend_ramp(piece.shape[-2], y == 0, y_ >= h, border, **kw)[:, None],
                _blend_ramp(piece.shape[-1], x == 0, x_ >= w, border, **kw)[None, :],
            )[None, None]

            ty, tx = y * factor, x * factor
            out[:, :, ty:ty + piece.shape[-2], tx:tx + piece.shape[-1]] += piece.float() * mask
            weight[:, :, ty:ty + piece.shape[-2], tx:tx + piece.shape[-1]] += mask
            del piece

    out /= weight.clamp_min(1e-6)
    return out.to(latent.dtype)


def choose_tile(latent: torch.Tensor, budget_bytes: int, upsampling_factor: int = 8) -> tuple[int, int]:
    """Largest (tile, stride) in latent units whose decode fits in `budget_bytes`.

    Biggest tile that fits means fewest seams and the least attention
    distortion, which is why this is not a fixed tile size.
    """
    per_tile_pixel = _DECODE_BYTES_PER_PIXEL * upsampling_factor * upsampling_factor
    max_latent_px = max(1, budget_bytes // per_tile_pixel)
    tile = int(max_latent_px ** 0.5)
    tile = max(_MIN_TILE, (tile // 8) * 8)
    tile = min(tile, max(latent.shape[-2], latent.shape[-1]))
    # A quarter-tile of overlap is enough for the ramp to hide the seam.
    stride = max(_MIN_TILE // 2, tile - max(8, tile // 4))
    return tile, stride
