"""
CANN Bench Utils - Framework operators for anti-cheat

Provides custom AscendC operators:
- cann_bench_warmup: MatMul (10240x10240, fp16) for NPU frequency boost
- cann_bench_cache_clean: ReduceMax (96x1024x1024, fp16) for L2 cache flush
- cann_bench_copy: device-to-device memory copy (fp16/fp32), safe alternative
  to aclnnInplaceCopy when TBE kernel tree is deleted by anti-cheat
- cann_bench_clone: clone a tensor via the framework copy kernel
"""

import torch
import torch_npu  # noqa: F401  ensure NPU backend (PrivateUse1) is initialized

from . import _C

torch.library.define("cann_bench_utils::cann_bench_warmup", "(Tensor x, Tensor y) -> Tensor")
torch.library.define("cann_bench_utils::cann_bench_cache_clean", "(Tensor x) -> Tensor")
# The dispatcher op mutates ``dst`` and deliberately has no return value.  The
# Python wrapper below returns ``dst`` for callers that need the copied tensor.
torch.library.define("cann_bench_utils::cann_bench_copy", "(Tensor src, Tensor(a!) dst) -> ()")
torch.library.define("cann_bench_utils::cann_bench_zero", "(Tensor(a!) dst) -> ()")


@torch.library.impl("cann_bench_utils::cann_bench_zero", "PrivateUse1")
def _zero_npu(dst: torch.Tensor) -> None:
    _C.device_zero_npu(dst)


@torch.library.impl("cann_bench_utils::cann_bench_zero", "Meta")
def _zero_meta(dst: torch.Tensor) -> None:
    if not dst.is_contiguous():
        raise ValueError("cann_bench_zero requires a contiguous tensor")


def cann_bench_zero(dst: torch.Tensor) -> None:
    """Zero tensor bytes on the current stream; caller synchronizes before release."""
    torch.ops.cann_bench_utils.cann_bench_zero(dst)


@torch.library.impl("cann_bench_utils::cann_bench_warmup", "PrivateUse1")
def _warmup_npu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    _C.warmup_npu(x, y, z)
    return z


@torch.library.impl("cann_bench_utils::cann_bench_warmup", "Meta")
def _warmup_meta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _C.warmup_meta(x, y)


@torch.library.impl("cann_bench_utils::cann_bench_cache_clean", "PrivateUse1")
def _cache_clean_npu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((), dtype=x.dtype, device=x.device)
    _C.cache_clean_npu(x, out)
    return out


@torch.library.impl("cann_bench_utils::cann_bench_cache_clean", "Meta")
def _cache_clean_meta(x: torch.Tensor) -> torch.Tensor:
    return _C.cache_clean_meta(x)


@torch.library.impl("cann_bench_utils::cann_bench_copy", "PrivateUse1")
def _copy_npu(src: torch.Tensor, dst: torch.Tensor) -> None:
    _C.device_memcpy_npu(src, dst)


@torch.library.impl("cann_bench_utils::cann_bench_copy", "Meta")
def _copy_meta(src: torch.Tensor, dst: torch.Tensor) -> None:
    _C.device_memcpy_meta(src, dst)


def cann_bench_warmup(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """NPU warmup operation (MatMul for frequency boost)."""
    return torch.ops.cann_bench_utils.cann_bench_warmup(x, y)


def cann_bench_cache_clean(x: torch.Tensor) -> torch.Tensor:
    """L2 cache clean operation (ReduceMax for cache flush)."""
    return torch.ops.cann_bench_utils.cann_bench_cache_clean(x)


def cann_bench_copy(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Device-to-device copy using framework AscendC kernel.

    Safe alternative to aclnnInplaceCopy which may be unavailable after
    anti-cheat TBE kernel tree deletion.  Supports fp16 and fp32.
    """
    torch.ops.cann_bench_utils.cann_bench_copy(src, dst)
    return dst


def cann_bench_clone(x: torch.Tensor) -> torch.Tensor:
    """Clone a tensor using the framework copy kernel.

    Safe alternative to torch.Tensor.clone() when aclnnInplaceCopy is
    unavailable due to anti-cheat TBE kernel deletion.
    """
    y = torch.empty_like(x)
    cann_bench_copy(x, y)
    return y


__all__ = [
    'cann_bench_warmup',
    'cann_bench_cache_clean',
    'cann_bench_copy',
    'cann_bench_clone',
    'cann_bench_zero',
]
