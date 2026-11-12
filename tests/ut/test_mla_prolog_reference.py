# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""MLAProlog preserves FP32 intermediates and final output stores."""

import importlib.util
from pathlib import Path
import torch
import pytest

TASK = Path(__file__).resolve().parents[2] / "tasks/level4/mla_prolog/golden.py"


def load():
    spec = importlib.util.spec_from_file_location("prolog_reference", TASK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [17, 31, 371])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reference_matches_continuous_fp32_independent_products(seed, dtype):
    gen = torch.Generator().manual_seed(seed)
    shapes = [(1, 3, 16), (16, 8), (8, 32), (2, 8, 8), (16, 16), (8,), (8,)]
    args = [(torch.randn(s, generator=gen) * 0.25).to(dtype) for s in shapes]
    angle = torch.randn(1, 3, 4, generator=gen).repeat(1, 1, 2)
    sin, cos = angle.sin().to(dtype), angle.cos().to(dtype)
    x, wdq, wuq, wuk, wdkv, gq, gk = [a.float() for a in args]

    # Explicit products/sums: no Golden matmul, RMSNorm or RoPE helpers.
    def mm(a, b):
        return (a.unsqueeze(-1) * b).sum(-2)

    def norm(z, g):
        return g * z / (z.square().mean(-1, keepdim=True) + 1e-5).sqrt()

    def rope(z):
        return z * cos.float() + torch.cat((-z[..., 4:], z[..., :4]), -1) * sin.float()

    cq = norm(mm(x, wdq), gq)
    qr = mm(cq, wuq).reshape(1, 3, 2, 16)
    query = torch.stack([mm(qr[:, :, h, :8], wuk[h]) for h in range(2)], 2)
    query_rope = torch.stack([rope(qr[:, :, h, 8:]) for h in range(2)], 2)
    kv = mm(x, wdkv)
    expected = [query, query_rope, norm(kv[..., :8], gk), rope(kv[..., 8:])]
    actual = load().mla_prolog(*args, sin, cos, n_heads=2)
    for a, b in zip(actual, expected):
        assert a.dtype == dtype and a.is_contiguous()
        torch.testing.assert_close(a, b.to(dtype), atol=0, rtol=0)
