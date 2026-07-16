"""
CANN Bench Utils - Framework warmup and cache clean operators

Provides two custom operators for v3 anti-cheat:
- cann_bench_warmup: MatMul (10240x10240, fp16) for NPU frequency boost
- cann_bench_cache_clean: ReduceMax (96x1024x1024, fp16) for L2 cache flush

These operators use specialized naming (CannBenchWarmup/CannBenchCacheClean)
for profiling filtering without shape matching.
"""

import torch

# Import C++ extension
from . import _C

def cann_bench_warmup(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """NPU warmup operation (MatMul for frequency boost).

    Args:
        x: Input tensor (10240, 10240), dtype=float16, device='npu'
        y: Input tensor (10240, 10240), dtype=float16, device='npu'

    Returns:
        Output tensor (10240, 10240), dtype=float16

    Note:
        This is NOT a full MatMul implementation - just enough to boost NPU frequency.
        Profiling Type: "CannBenchWarmup"
    """
    return torch.ops.cann_bench_utils.cann_bench_warmup(x, y)


def cann_bench_cache_clean(x: torch.Tensor) -> torch.Tensor:
    """L2 cache clean operation (ReduceMax for cache flush).

    Args:
        x: Input tensor (96, 1024, 1024), dtype=float16, device='npu'

    Returns:
        Scalar tensor, dtype=float16

    Note:
        This is NOT a full ReduceMax implementation - just enough to flush L2 cache.
        Profiling Type: "CannBenchCacheClean"
    """
    return torch.ops.cann_bench_utils.cann_bench_cache_clean(x)


__all__ = ['cann_bench_warmup', 'cann_bench_cache_clean']
