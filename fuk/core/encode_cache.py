"""
Content-addressed cache for VAE encode results.

Iterating on a shot re-encodes the same control video / reference / input
image every generation — VACE even encodes the control video twice per run
(inactive + reactive variants). Both Qwen and Wan VAE encodes are
deterministic (mean projection, log_var discarded, no sampling), so the
output depends only on input content and tiling params — identical inputs
can skip the encode entirely.

install_encode_cache(pipe) wraps pipe.vae.encode with a small LRU keyed by a
blake2b hash of the input bytes + shape/dtype + tiling kwargs. Results are
parked on CPU (latents are a few MB each) and moved to the requested device
on a hit. Any input or result structure the cache doesn't recognize bypasses
it — behavior is then identical to an unwrapped pipeline. Never raises from
the wrapper's cache logic; a cache failure falls back to a plain encode.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch

_MAX_ENTRIES = 12
# Kwargs that change encode output. `device` deliberately excluded — it only
# affects where the result lives, which the hit path handles.
_KEY_KWARGS = ("tiled", "tile_size", "tile_stride")


def _hash_one(h, obj) -> bool:
    """Feed one input object into the hash. False = type we can't key safely."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        h.update(f"T{tuple(t.shape)}{t.dtype}".encode())
        h.update(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return True
    if isinstance(obj, (list, tuple)):
        h.update(f"L{len(obj)}".encode())
        return all(_hash_one(h, o) for o in obj)
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            h.update(f"I{obj.size}{obj.mode}".encode())
            h.update(obj.tobytes())
            return True
    except Exception:
        pass
    return False


def _to_cpu(res):
    """CPU copy of an encode result, or None if the structure is unknown."""
    if isinstance(res, torch.Tensor):
        return res.detach().to("cpu", copy=True)
    if isinstance(res, (list, tuple)):
        out = [_to_cpu(r) for r in res]
        if any(o is None for o in out):
            return None
        return type(res)(out)
    return None


def _to_device(res, device):
    if isinstance(res, torch.Tensor):
        return res.to(device)
    return type(res)(_to_device(r, device) for r in res)


def _first_device(res) -> str:
    if isinstance(res, torch.Tensor):
        return str(res.device)
    for r in res:
        return _first_device(r)
    return "cpu"


def install_encode_cache(pipe, log=None) -> bool:
    """Wrap pipe.vae.encode with a content-addressed LRU. Idempotent."""
    log = log or (lambda msg: None)
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "encode"):
        return False
    if getattr(vae, "_fuk_encode_cache_installed", False):
        return True

    original_encode = vae.encode
    cache: "OrderedDict[str, tuple]" = OrderedDict()

    def cached_encode(x, *args, **kwargs):
        try:
            # Positional extras would be invisible to the key — don't risk it.
            key = None
            if not args:
                h = hashlib.blake2b(digest_size=16)
                if _hash_one(h, x):
                    for k in _KEY_KWARGS:
                        h.update(f"{k}={kwargs.get(k)!r}".encode())
                    key = h.hexdigest()

            if key is not None:
                hit = cache.get(key)
                if hit is not None:
                    cache.move_to_end(key)
                    cpu_res, stored_device = hit
                    log(f"[encode-cache] hit — skipped VAE encode ({key[:10]})")
                    return _to_device(cpu_res, kwargs.get("device") or stored_device)
        except Exception as e:
            log(f"[encode-cache] lookup failed (falling back to encode): {e}")
            key = None

        result = original_encode(x, *args, **kwargs)

        try:
            if key is not None:
                cpu_res = _to_cpu(result)
                if cpu_res is not None:
                    cache[key] = (cpu_res, _first_device(result))
                    while len(cache) > _MAX_ENTRIES:
                        cache.popitem(last=False)
        except Exception as e:
            log(f"[encode-cache] store failed (result unaffected): {e}")
        return result

    vae.encode = cached_encode
    vae._fuk_encode_cache_installed = True
    return True
