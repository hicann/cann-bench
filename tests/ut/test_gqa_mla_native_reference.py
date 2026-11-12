# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Native stage stores match independent dense per-head attention equations."""

import importlib.util
from pathlib import Path
import pytest
import torch

TASKS = Path(__file__).resolve().parents[2] / "tasks/level4"


def load(op):
    spec = importlib.util.spec_from_file_location(op, TASKS / op / "golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense(q, k, v, causal, scale):
    # Independent head lookup; no grouping reshape or tiled helper.
    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype)
    s, sk, nq, nk = q.shape[1], k.shape[1], q.shape[2], k.shape[2]
    mask = torch.arange(sk)[None] > torch.arange(s)[:, None] + sk - s
    for batch in range(q.shape[0]):
        for head in range(nq):
            logits = (q[batch, :, head] @ k[batch, :, head * nk // nq].T) * scale
            if causal:
                logits = logits.masked_fill(mask, -torch.inf)
            out[batch, :, head] = logits.softmax(-1) @ v[batch, :, head * nk // nq]
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_native_equations_and_bounded_workspace(
    dtype, kv_heads, causal, monkeypatch
):
    m = load("gqa")
    gen = torch.Generator().manual_seed(81)
    q = torch.randn(2, 3, 4, 8, generator=gen).to(dtype)
    k, v = [torch.randn(2, 7, kv_heads, 8, generator=gen).to(dtype) for _ in range(2)]
    expected = dense(q, k, v, causal, 0.3)
    monkeypatch.setattr(m, "_NATIVE_SCORE_ELEMENTS", 14)
    original = torch.nn.functional.softmax
    observed = []

    def softmax(x, *args, **kwargs):
        observed.append((x.numel(), x.dtype, x.shape[-1]))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softmax", softmax)
    actual = m.gqa_bench(q, k, v, 0.3, causal)
    assert observed and all(
        n <= 14 and dt == dtype and sk == 7 for n, dt, sk in observed
    )
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-14 if dtype == torch.float64 else 0,
        rtol=1e-14 if dtype == torch.float64 else 0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("causal", [False, True])
def test_mla_native_layout_shared_latent_and_full_key_domain(
    dtype, layout, causal, monkeypatch
):
    m = load("mla")
    gen = torch.Generator().manual_seed(29)
    qn = torch.randn(2, 3, 4, 8, generator=gen).to(dtype)
    qr = torch.randn(2, 3, 4, 4, generator=gen).to(dtype)
    kn = torch.randn(2, 7, 2, 8, generator=gen).to(dtype)
    kr = torch.randn(2, 7, 2, 4, generator=gen).to(dtype)
    expected = dense(
        torch.cat((qn, qr), -1), torch.cat((kn, kr), -1), kn, causal, 12**-0.5
    )
    tensors = (qn, qr, kn, kr, kn.clone())
    if layout == "BNSD":
        tensors = tuple(t.permute(0, 2, 1, 3) for t in tensors)
        expected = expected.permute(0, 2, 1, 3)
    monkeypatch.setattr(m, "_NATIVE_SCORE_ELEMENTS", 14)
    actual = m.mla_bench(*tensors, numKVHeads=2, inputLayout=layout, is_causal=causal)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-14 if dtype == torch.float64 else 0,
        rtol=1e-14 if dtype == torch.float64 else 0,
    )
    assert actual.is_contiguous()


@pytest.mark.parametrize("op", ["gqa", "mla"])
def test_reject_invalid_rectangular_causal_inputs(op):
    m = load(op)
    q = torch.zeros(1, 8, 4, 8)
    k = torch.zeros(1, 7, 2, 8)
    with pytest.raises(ValueError, match="S <= S_kv"):
        if op == "gqa":
            m.gqa_bench(q, k, k, is_causal=True)
        else:
            m.mla_bench(q, q, k, k, k, numKVHeads=2, is_causal=True)
