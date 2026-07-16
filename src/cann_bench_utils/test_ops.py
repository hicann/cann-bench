#!/usr/bin/env python3
"""
Simple test for cann_bench_utils operators
"""

import torch
import torch_npu

def test_warmup():
    print("Testing cann_bench_warmup...")
    from cann_bench_utils import cann_bench_warmup

    x = torch.randn(10240, 10240, dtype=torch.float16).npu()
    y = torch.randn(10240, 10240, dtype=torch.float16).npu()

    out = cann_bench_warmup(x, y)

    assert out.shape == (10240, 10240), f"Expected (10240, 10240), got {out.shape}"
    assert out.dtype == torch.float16, f"Expected float16, got {out.dtype}"
    assert out.device.type == 'npu', f"Expected npu, got {out.device.type}"

    print("✓ cann_bench_warmup passed")


def test_cache_clean():
    print("Testing cann_bench_cache_clean...")
    from cann_bench_utils import cann_bench_cache_clean

    x = torch.randn(96, 1024, 1024, dtype=torch.float16).npu()

    out = cann_bench_cache_clean(x)

    assert out.shape == (), f"Expected scalar, got {out.shape}"
    assert out.dtype == torch.float16, f"Expected float16, got {out.dtype}"
    assert out.device.type == 'npu', f"Expected npu, got {out.device.type}"

    print("✓ cann_bench_cache_clean passed")


if __name__ == "__main__":
    print("=" * 60)
    print("CANN Bench Utils - Operator Tests")
    print("=" * 60)

    try:
        test_warmup()
        test_cache_clean()
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
