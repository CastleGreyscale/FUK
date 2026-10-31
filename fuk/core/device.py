"""
Compute-device layer: the one place FUK decides which accelerator it runs on.

Everything else asks this module instead of calling torch.cuda directly, so the
same code runs on NVIDIA (CUDA), AMD (ROCm) and Apple Silicon (MPS), with CPU as
the last resort. See docs/PLATFORM_SUPPORT_PLAN.md.

Selection order: FUK_DEVICE if set, then CUDA, then MPS, then CPU. CUDA always
wins when it's there — it's the reference platform. ROCm builds of torch report
through the torch.cuda namespace and use the "cuda" device string, so AMD takes
the CUDA branch everywhere below; backend() is what tells the two apart.

On CUDA the wrappers call torch.cuda exactly as the code did before this module
existed, so NVIDIA behaviour is unchanged. MPS has no mem_get_info: Apple
Silicon memory is unified, and the budget the OS will let Metal use is
recommended_max_memory(), so "free" is that budget minus what Metal holds.

Results are cached per process. FUK's two import shapes (`core.device` and bare
`device`) can load this module twice; that's harmless here, since both copies
compute the same answer.
"""

import os
from functools import lru_cache
from typing import Tuple

import torch

_VALID = ("cuda", "mps", "cpu")


@lru_cache(maxsize=None)
def device_type() -> str:
    """'cuda', 'mps' or 'cpu'. AMD GPUs are 'cuda' (see backend())."""
    forced = os.environ.get("FUK_DEVICE", "").strip().lower()
    if forced:
        if forced not in _VALID:
            raise ValueError(f"FUK_DEVICE={forced!r}; expected one of {', '.join(_VALID)}")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=None)
def backend() -> str:
    """'cuda', 'rocm', 'mps' or 'cpu' — device_type() with AMD split out."""
    kind = device_type()
    if kind == "cuda" and torch.version.hip is not None:
        return "rocm"
    return kind


def device(index: int = 0) -> torch.device:
    """The torch.device to run on."""
    kind = device_type()
    return torch.device(kind, index) if kind == "cuda" else torch.device(kind)


def is_accelerated() -> bool:
    """True on any GPU (CUDA, ROCm or MPS)."""
    return device_type() != "cpu"


def unified_memory() -> bool:
    """True when the GPU shares system RAM, so offloading to 'cpu' frees nothing."""
    return device_type() == "mps"


def empty_cache() -> None:
    kind = device_type()
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def synchronize() -> None:
    kind = device_type()
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def memory_allocated() -> int:
    """Bytes held by live tensors."""
    kind = device_type()
    if kind == "cuda":
        return torch.cuda.memory_allocated()
    if kind == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def memory_reserved() -> int:
    """Bytes held by the allocator, including its cache."""
    kind = device_type()
    if kind == "cuda":
        return torch.cuda.memory_reserved()
    if kind == "mps":
        return torch.mps.driver_allocated_memory()
    return 0


def max_memory_allocated() -> int:
    """Peak bytes since the last reset. CUDA only; MPS keeps no peak counter."""
    return torch.cuda.max_memory_allocated() if device_type() == "cuda" else 0


def reset_peak_memory_stats() -> None:
    if device_type() == "cuda":
        torch.cuda.reset_peak_memory_stats()


def total_memory() -> int:
    """Bytes the GPU can use: card VRAM, or Metal's budget on Apple Silicon."""
    kind = device_type()
    if kind == "cuda":
        return torch.cuda.get_device_properties(0).total_memory
    if kind == "mps":
        return torch.mps.recommended_max_memory()
    return 0


def mem_get_info() -> Tuple[int, int]:
    """(free, total) bytes, like torch.cuda.mem_get_info()."""
    kind = device_type()
    if kind == "cuda":
        return torch.cuda.mem_get_info()
    if kind == "mps":
        total = torch.mps.recommended_max_memory()
        return max(total - torch.mps.driver_allocated_memory(), 0), total
    return 0, 0
