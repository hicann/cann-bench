# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""CPU semantic regressions for reference changes; not NPU performance acceptance."""

import itertools
import math
import re
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from kernel_eval.benches.cann_loader import GoldenLoader

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"


@lru_cache(None)
def golden(path):
    return GoldenLoader(str(TASKS))._load_module(path)


def scalar_roi(x, boxes, oh, ow, scale, ratio, aligned):
    """Independent scalar interpolation over legal in-map ROIs."""
    out = torch.zeros((len(boxes), x.shape[1], oh, ow), dtype=torch.float64)
    h, w = x.shape[-2:]
    for k, b in enumerate(boxes):
        batch = int(b[0])
        shift = 0.5 if aligned else 0.0
        x0, y0, x1, y1 = [float(v) * scale - shift for v in b[1:]]
        rh, rw = y1 - y0, x1 - x0
        if not aligned:
            rh, rw = max(1.0, rh), max(1.0, rw)
        gh = ratio if ratio > 0 else math.ceil(rh / oh)
        gw = ratio if ratio > 0 else math.ceil(rw / ow)
        for c, ph, pw in itertools.product(range(x.shape[1]), range(oh), range(ow)):
            value = 0.0
            for iy, ix in itertools.product(range(gh), range(gw)):
                yy = max(0.0, y0 + (ph + (iy + 0.5) / gh) * rh / oh)
                xx = max(0.0, x0 + (pw + (ix + 0.5) / gw) * rw / ow)
                yl, xl = min(int(yy), h - 1), min(int(xx), w - 1)
                yh, xh = min(yl + 1, h - 1), min(xl + 1, w - 1)
                ly = 0.0 if yl >= h - 1 else yy - yl
                lx = 0.0 if xl >= w - 1 else xx - xl
                value += (
                    (1 - ly) * (1 - lx) * float(x[batch, c, yl, xl])
                    + (1 - ly) * lx * float(x[batch, c, yl, xh])
                    + ly * (1 - lx) * float(x[batch, c, yh, xl])
                    + ly * lx * float(x[batch, c, yh, xh])
                )
            out[k, c, ph, pw] = value / max(gh * gw, 1)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("ratio", [-1, 0, 1, 2, 4])
def test_roi_fallback_matches_independent_interpolation(dtype, aligned, ratio):
    g = torch.Generator().manual_seed(13)
    x = torch.randn(2, 2, 8, 9, generator=g).to(dtype)
    boxes = torch.tensor(
        [[0, 0, 0, 8, 7], [1, 0.25, 0.5, 4.5, 5.25], [0, 3, 2, 3.25, 2.125]],
        dtype=dtype,
    )
    actual = golden("level3/roi_align")._roi_align_fallback(
        x, boxes, 2, 3, 1.0, ratio, aligned
    )
    expected = scalar_roi(x.double(), boxes.double(), 2, 3, 1.0, ratio, aligned).to(
        dtype
    )
    tol = {torch.float16: 1e-3, torch.float32: 1e-6, torch.float64: 1e-12}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


def test_roi_fp64_oracle_preserves_sub_fp32_input_precision():
    x = torch.full((1, 1, 2, 2), 1 + 2**-30, dtype=torch.float64)
    boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    y = golden("level3/roi_align")._roi_align_fallback(x, boxes, 1, 1, 1.0, 1, False)
    assert y.dtype == torch.float64
    assert y.item() == x[0, 0, 0, 0].item()
    assert y.item() != x.float()[0, 0, 0, 0].item()


def test_roi_adaptive_grid_uses_active_slots(monkeypatch):
    module = golden("level3/roi_align")
    original = module._bilinear_interpolate
    sampled = []

    def observe(x, indices, y, x_pos, ymask, xmask):
        sampled.append((y.shape[-1], x_pos.shape[-1]))
        return original(x, indices, y, x_pos, ymask, xmask)

    monkeypatch.setattr(module, "_bilinear_interpolate", observe)
    x = torch.ones(1, 1, 64, 64)
    boxes = torch.tensor([[0.0, 1.0, 1.0, 21.0, 22.0], [0.0, 3.0, 2.0, 17.0, 16.0]])
    y = module._roi_align_fallback(x, boxes, 7, 7, 1.0, 0, True)
    assert sampled and all(h <= 3 and w <= 3 for h, w in sampled)
    torch.testing.assert_close(y, torch.ones_like(y))


@pytest.mark.parametrize("special", [float("nan"), float("inf"), -float("inf")])
def test_roi_special_values_follow_sampled_interpolation(special):
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    x[0, 0, 0, 0] = special
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 7.0, 7.0], [0.0, 3.0, 3.0, 6.0, 6.0]], dtype=torch.float64
    )
    y = golden("level3/roi_align")._roi_align_fallback(x, boxes, 3, 3, 1.0, 0, True)
    expected = scalar_roi(x, boxes.float(), 3, 3, 1.0, 0, True).float()
    torch.testing.assert_close(y, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("mask_kind", ["none", "partial", "all"])
def test_linear_ce_global_reduction_across_chunk_tail(dtype, reduction, mask_kind):
    g = torch.Generator().manual_seed(29)
    h = torch.randn(257, 7, generator=g).to(dtype)
    w = torch.randn(19, 7, generator=g).to(dtype)
    labels = torch.randint(0, 19, (257,), generator=g, dtype=torch.int32)
    if mask_kind == "partial":
        labels[::3] = -1
    elif mask_kind == "all":
        labels[:] = -1
    module = golden("level4/linear_cross_entropy")
    func = (
        module.linear_cross_entropy_oracle
        if dtype == torch.float64
        else module.linear_cross_entropy
    )
    y = func(h, w, labels, reduction, -1)
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    logits = h.to(calc_dtype) @ w.to(calc_dtype).t()
    valid = labels != -1
    ids = torch.where(valid)[0]
    losses = torch.logsumexp(logits[ids], -1) - logits[ids, labels[ids].long()]
    expected = losses.sum()
    if reduction == "mean":
        expected = expected / valid.sum()
    tol = 1e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(
        y.reshape(()), expected, rtol=tol, atol=tol, equal_nan=True
    )
    assert y.dtype == calc_dtype


def test_linear_ce_bounds_logits_rows(monkeypatch):
    module = golden("level4/linear_cross_entropy")
    original = torch.matmul
    shapes = []

    def observe(*args, **kwargs):
        y = original(*args, **kwargs)
        shapes.append(y.shape)
        return y

    monkeypatch.setattr(torch, "matmul", observe)
    module.linear_cross_entropy(
        torch.ones(257, 7), torch.ones(19, 7), torch.zeros(257, dtype=torch.int32)
    )
    assert [s[0] for s in shapes] == [128, 128, 1]
    assert all(s[1] == 19 for s in shapes)


@pytest.mark.parametrize(
    "pads", [[0, 0, 0, 0], [1, 1, 2, 2], [0, 1, 2, 0], [2, 0, 0, 1]]
)
@pytest.mark.parametrize("stride", [[1, 1], [2, 1]])
@pytest.mark.parametrize("dilation", [[1, 1], [2, 1]])
def test_conv_padding_matches_scalar_convolution(pads, stride, dilation):
    x = torch.arange(30, dtype=torch.float64).reshape(1, 1, 5, 6) / 10
    w = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=torch.float64)
    bias = torch.tensor([0.25], dtype=torch.float64)
    actual = golden("level3/conv_2d").conv_2d(x, w, bias, stride, pads, dilation)
    expected = torch.empty_like(actual)
    for oh, ow in itertools.product(range(actual.shape[-2]), range(actual.shape[-1])):
        value = 0.25
        for kh, kw in itertools.product(range(2), range(2)):
            ih = oh * stride[0] + kh * dilation[0] - pads[0]
            iw = ow * stride[1] + kw * dilation[1] - pads[2]
            if 0 <= ih < 5 and 0 <= iw < 6:
                value += float(x[0, 0, ih, iw]) * float(w[0, 0, kh, kw])
        expected[0, 0, oh, ow] = value
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", [2, 3, 4])
@pytest.mark.parametrize("sq,sk", [(2049, 2049), (17, 2053)])
def test_attention_grad_preserves_compressed_candidate_mask(mode, sq, sk):
    module = golden("level4/flash_attention_score_grad_enhance")
    q = torch.zeros(sq, 1, 1)
    k = torch.zeros(sk, 1, 1)
    prepared = module.get_input(
        q,
        k,
        k,
        q,
        sparse_mode=mode,
        head_num=1,
        input_layout="TND",
        actual_seq_qlen=[sq],
        actual_seq_kvlen=[sk],
    )
    assert prepared[6].shape == (2048, 2048)
    assert prepared[6].dtype == torch.uint8
    assert prepared[10].shape == q.shape


@pytest.mark.parametrize("reduction,tokens", [("sum", 0), ("mean", 0), ("none", 1)])
def test_linear_ce_retains_legacy_edge_behavior(reduction, tokens):
    module = golden("level4/linear_cross_entropy")
    h, w = torch.ones(tokens, 7), torch.ones(19, 7)
    labels = torch.zeros(tokens, dtype=torch.int32)
    expected = torch.nn.functional.cross_entropy(
        h @ w.t(), labels.long(), reduction=reduction
    ).reshape(1)
    actual = module.linear_cross_entropy(h, w, labels, reduction)
    torch.testing.assert_close(actual, expected, equal_nan=True)


def reference_attention(q, k, v, dy, q_lengths, kv_lengths, mode, pre, nxt, sink):
    """Independent small attention oracle; no Golden mask/layout helpers."""
    q, k, v = [x.detach().double().requires_grad_() for x in (q, k, v)]
    q_start, kv_start = 0, sink
    losses, outputs = [], []
    for sq, sk in zip(q_lengths, kv_lengths):
        qi = q[q_start : q_start + sq].transpose(0, 1)
        ids = list(range(sink)) + list(range(kv_start, kv_start + sk))
        ki = k[ids].repeat_interleave(q.shape[1] // k.shape[1], dim=1).transpose(0, 1)
        vi = v[ids].repeat_interleave(q.shape[1] // v.shape[1], dim=1).transpose(0, 1)
        rows = torch.arange(sq)[:, None]
        cols = torch.arange(sk)[None, :]
        if mode == 2:
            mask = cols > rows
        elif mode == 3:
            mask = cols > sk - sq + rows
        else:
            mask = (cols < sk - sq + rows - pre) | (cols > sk - sq + rows + nxt)
        mask = torch.cat([torch.zeros(sq, sink, dtype=torch.bool), mask], dim=1)
        scores = (qi @ ki.transpose(-1, -2)) * 0.5
        yi = (torch.softmax(scores.masked_fill(mask, -40000.0), dim=-1) @ vi).transpose(
            0, 1
        )
        outputs.append(yi)
        losses.append((yi * dy[q_start : q_start + sq]).sum())
        q_start += sq
        kv_start += sk
    grads = torch.autograd.grad(sum(losses), (q, k, v))
    return torch.cat(outputs), grads


@pytest.mark.parametrize("mode", [2, 3, 4])
@pytest.mark.parametrize("sink_groups", [0, 1])
def test_attention_cpu_rebuilds_each_segment_without_mutating_input(mode, sink_groups):
    module = golden("level4/flash_attention_score_grad_enhance")
    g = torch.Generator().manual_seed(73)
    sink = sink_groups * 64
    q = torch.randn(8, 2, 3, generator=g, dtype=torch.float64)
    k = torch.randn(7 + sink, 1, 3, generator=g, dtype=torch.float64)
    v = torch.randn(7 + sink, 1, 2, generator=g, dtype=torch.float64)
    dy = torch.randn(8, 2, 2, generator=g, dtype=torch.float64)
    mask = torch.ones(2048, 2048, dtype=torch.uint8).triu(1)
    original = mask.clone()
    args = dict(
        head_num=2,
        input_layout="TND",
        atten_mask=mask,
        actual_seq_qlen=[3, 5],
        actual_seq_kvlen=[5, 2],
        sparse_mode=mode,
        sink_num=sink_groups,
        pre_tokens=1,
        next_tokens=0,
    )
    expected_y, expected_grad = reference_attention(
        q, k, v, dy, [3, 5], [5, 2], mode, 1, 0, sink
    )
    actual_grad = module.flash_attention_score_grad_enhance(
        q, k, v, dy, scale_value=0.5, **args
    )
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    prepared = module.get_input(q, k, v, dy, scale_value=0.5, **args)
    assert prepared[6] is mask
    assert torch.equal(mask, original)
    torch.testing.assert_close(prepared[10], expected_y, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode", [2, 3, 4])
def test_attention_cpu_bsnd_retains_layout_and_mask(mode):
    module = golden("level4/flash_attention_score_grad_enhance")
    g = torch.Generator().manual_seed(41)
    q = torch.randn(2, 5, 2, 3, generator=g, dtype=torch.float64)
    k = torch.randn(2, 7, 1, 3, generator=g, dtype=torch.float64)
    v = torch.randn(2, 7, 1, 2, generator=g, dtype=torch.float64)
    dy = torch.randn(2, 5, 2, 2, generator=g, dtype=torch.float64)
    expected_y, expected_grad = reference_attention(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        dy.flatten(0, 1),
        [5, 5],
        [7, 7],
        mode,
        2,
        1,
        0,
    )
    mask = torch.ones(2048, 2048, dtype=torch.uint8).triu(1)
    args = dict(
        head_num=2,
        input_layout="BSND",
        atten_mask=mask,
        sparse_mode=mode,
        pre_tokens=2,
        next_tokens=1,
        scale_value=0.5,
    )
    grads = module.flash_attention_score_grad_enhance(q, k, v, dy, **args)
    for actual, expected in zip(grads, expected_grad):
        torch.testing.assert_close(
            actual, expected.reshape(actual.shape), rtol=1e-12, atol=1e-12
        )
    prepared = module.get_input(q, k, v, dy, **args)
    assert prepared[6] is mask
    torch.testing.assert_close(
        prepared[10], expected_y.reshape(2, 5, 2, 2), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize(
    "path", ["level3/roi_align", "level3/conv_2d", "level4/linear_cross_entropy"]
)
def test_embedded_reference_matches_golden(path):
    desc = (TASKS / path / "desc.md").read_text(encoding="utf-8")
    section = desc.split("## 5. 标准 Golden 代码", 1)[1]
    source = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    namespace = {}
    exec(compile(source, str(TASKS / path / "desc.md"), "exec"), namespace)
    module = golden(path)
    if path.endswith("roi_align"):
        x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        boxes = torch.tensor([[0.0, 0.0, 0.0, 7.0, 7.0]])
        actual = namespace["_roi_align_fallback"](x, boxes, 3, 3, 1.0, 0, True)
        expected = module._roi_align_fallback(x, boxes, 3, 3, 1.0, 0, True)
    elif path.endswith("conv_2d"):
        args = (
            torch.ones(1, 1, 5, 6),
            torch.ones(1, 1, 2, 2),
            torch.zeros(1),
            [1, 1],
            [0, 1, 2, 0],
        )
        actual = namespace["conv_2d"](*args)
        expected = module.conv_2d(*args)
    else:
        args = (
            torch.ones(129, 7),
            torch.ones(19, 7),
            torch.zeros(129, dtype=torch.int32),
        )
        actual = namespace["linear_cross_entropy"](*args)
        expected = module.linear_cross_entropy(*args)
    torch.testing.assert_close(actual, expected)
