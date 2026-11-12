# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Independent dense algebra and precision-policy checks for sparse native bench."""

import importlib.util
from pathlib import Path
import pytest
import torch

P = (
    Path(__file__).resolve().parents[2]
    / "tasks/level4/sparse_flash_attention/golden.py"
)
spec = importlib.util.spec_from_file_location("sfa_reference", P)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def inputs(dtype, layout, causal):
    gen = torch.Generator().manual_seed(19)
    q = torch.randn(1, 4, 7, 64, generator=gen).to(dtype)
    k = torch.randn(1, 2, 11, 64, generator=gen).to(dtype)
    v = k.clone()
    ids = torch.tensor([0, 2, 5, 8]).expand(1, 2, 7, 4).clone().int()
    if layout == "BSND":
        q, k, v, ids = (x.transpose(1, 2) for x in (q, k, v, ids))
    return q, k, v, ids


def dense(q, k, v, ids, layout, causal):
    if layout == "BSND":
        q, k, v, ids = (x.transpose(1, 2) for x in (q, k, v, ids))
    kk = k.repeat_interleave(2, 1)
    vv = v.repeat_interleave(2, 1)
    mask = (
        torch.zeros(1, 2, 7, 11, dtype=torch.bool)
        .scatter_(-1, ids.long(), True)
        .repeat_interleave(2, 1)
    )
    if causal:
        mask &= (
            torch.arange(11)[None, None, None, :]
            <= torch.arange(7)[None, None, :, None] + 4
        )
    score = (q @ kk.transpose(-1, -2)) * 0.125
    score.masked_fill_(~mask, -torch.inf)
    mx = score.amax(-1, keepdim=True)
    empty = torch.isneginf(mx)
    ex = (score - torch.where(empty, torch.zeros_like(mx), mx)).exp()
    den = ex.sum(-1, keepdim=True)
    prob = ex / torch.where(empty, torch.ones_like(den), den)
    y = prob @ vv
    return y.transpose(1, 2) if layout == "BSND" else y


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("causal", [False, True])
def test_native_reference_matches_independent_dense(dtype, layout, causal):
    x = inputs(dtype, layout, causal)
    actual = g.sparse_flash_attention_bench(*x, 0.125, layout, causal)
    expected = dense(*x, layout, causal)
    # Sparse and dense reduction order can differ by native rounding.
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=4 * eps, atol=eps)
    assert actual.dtype == dtype and actual.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_keeps_native_bmm_stages_without_dense_scores(dtype, monkeypatch):
    q, k, v, ids = inputs(dtype, "BNSD", False)
    seen = []
    real = torch.bmm

    def observe(a, b):
        seen.append((a.dtype, b.dtype, a.shape, b.shape))
        return real(a, b)

    monkeypatch.setattr(torch, "bmm", observe)
    g.sparse_flash_attention_bench(q, k, v, ids, 0.125, "BNSD")
    assert seen and all(a == dtype and b == dtype for a, b, _, _ in seen)
    assert all(11 not in ashape and 11 not in bshape for _, _, ashape, bshape in seen)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_fp64_oracle_matches_independent_math(layout):
    x = inputs(torch.float64, layout, True)
    actual = g.sparse_flash_attention_oracle(*x, 0.125, layout, True)
    torch.testing.assert_close(actual, dense(*x, layout, True), rtol=1e-12, atol=1e-12)
    assert actual.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fully_masked_row_remains_zero(dtype):
    q, k, v, ids = inputs(dtype, "BNSD", True)
    ids[:] = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    y = g.sparse_flash_attention_bench(q, k, v, ids, 0.125, "BNSD", True)
    assert torch.isfinite(y).all() and torch.count_nonzero(y[:, :, :3]) == 0
