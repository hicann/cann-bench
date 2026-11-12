# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Independent sparse-indexer gradient regressions, including repeated and future indices."""

import importlib.util
from pathlib import Path
import pytest
import torch

TASKS = Path(__file__).resolve().parents[2] / "tasks"


def load(rel):
    p = TASKS / rel / "golden.py"
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_indexer_key_gradient_matches_independent_gather_autograd(repeated, dtype):
    m = load("level5/sparse_lightning_indexer_grad_kl_loss_enhance")
    gen = torch.Generator().manual_seed(283)
    q = torch.randn(3, 2, 8, generator=gen, dtype=dtype)
    k = torch.randn(5, 1, 8, generator=gen, dtype=dtype)
    qi = torch.randn(3, 2, 8, generator=gen, dtype=dtype, requires_grad=True)
    ki = torch.randn(5, 1, 8, generator=gen, dtype=dtype, requires_grad=True)
    w = torch.randn(3, 2, generator=gen, dtype=dtype, requires_grad=True)
    ids = torch.tensor([0, 1, 1 if repeated else 2, 4]).expand(3, 4)
    selected_k = k[:, 0][ids]
    target = torch.softmax(
        (q[:, :, :, None] * selected_k.transpose(1, 2)[:, None]).sum(2) * 0.2, -1
    ).mean(1)
    selected_index = ki[:, 0][ids]
    index_logits = (qi[:, :, :, None] * selected_index.transpose(1, 2)[:, None]).sum(2)
    prediction = torch.softmax((torch.relu(index_logits) * w[:, :, None]).sum(1), -1)
    pt = target.clamp_min(1e-8)
    pp = prediction.clamp_min(1e-8)
    loss = (pt * (pt.log() - pp.log())).sum()
    gradients = torch.autograd.grad(loss, (qi, ki, w))
    actual = m.sparse_lightning_indexer_grad_kl_loss_enhance(
        q,
        k,
        qi,
        ki,
        w,
        ids[:, None].int(),
        scale_value=0.2,
        layout="TND",
        actual_seq_lengths_query=[3],
        actual_seq_lengths_key=[5],
        sparse_mode=0,
    )
    for a, b in zip(actual, (*gradients, loss.reshape(1))):
        torch.testing.assert_close(
            a,
            b,
            atol=2e-6 if dtype == torch.float32 else 1e-10,
            rtol=2e-5 if dtype == torch.float32 else 1e-10,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pattern", ["future", "all_future", "noncontiguous"])
def test_indexer_causal_target_and_independent_head_groups(pattern, dtype):
    """Target-only causal mask; indexer heads need not equal attention heads."""
    m = load("level5/sparse_lightning_indexer_grad_kl_loss_enhance")
    gen = torch.Generator().manual_seed(283)
    q = torch.randn(3, 2, 8, generator=gen, dtype=dtype) * 0.7
    k = torch.randn(5, 1, 8, generator=gen, dtype=dtype) * 0.7
    qi = (torch.randn(3, 4, 8, generator=gen, dtype=dtype) * 0.7).requires_grad_()
    ki = (torch.randn(5, 1, 8, generator=gen, dtype=dtype) * 0.7).requires_grad_()
    w = (torch.randn(3, 4, generator=gen, dtype=dtype) * 0.7).requires_grad_()
    patterns = {
        "future": [[0, 4], [0, 4], [0, 4]],
        "all_future": [[3, 4], [4, -1], [0, 1]],
        "noncontiguous": [[0, 2], [0, 3], [1, 4]],
    }
    ids = torch.tensor(patterns[pattern], dtype=torch.int32)
    loss = qi.sum() * 0
    for s, row in enumerate(ids):
        selected = row[row >= 0].long()
        # Explicit dot products and autograd, independent of analytic backward.
        logits = (q[s, :, None] * k[selected, 0][None]).sum(-1) * 0.2
        logits = logits.masked_fill(selected[None] >= 5 - 3 + s + 1, -40000.0)
        target = logits.softmax(-1).mean(0).clamp_min(1e-8)
        dot = (qi[s, :, None] * ki[selected, 0][None]).sum(-1)
        pred = (dot.relu() * w[s, :, None]).sum(0).softmax(-1).clamp_min(1e-8)
        loss = loss + (target * (target.log() - pred.log())).sum()
    gradients = torch.autograd.grad(loss, (qi, ki, w))
    actual = m.sparse_lightning_indexer_grad_kl_loss_enhance(
        q,
        k,
        qi,
        ki,
        w,
        ids[:, None],
        scale_value=0.2,
        layout="TND",
        actual_seq_lengths_query=[3],
        actual_seq_lengths_key=[5],
        sparse_mode=3,
    )
    for a, b in zip(actual, (*gradients, loss.reshape(1))):
        torch.testing.assert_close(
            a,
            b,
            atol=2e-6 if dtype == torch.float32 else 1e-10,
            rtol=2e-5 if dtype == torch.float32 else 1e-10,
        )
