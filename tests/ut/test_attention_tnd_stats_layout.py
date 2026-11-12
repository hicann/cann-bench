# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""TND forward-statistics rows must address the same token/head as Q."""

import importlib.util
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize("mode", [0, 3, 4])
@pytest.mark.parametrize("sink_groups", [0, 1])
def test_tnd_stats_follow_token_then_head_across_ragged_segments(mode, sink_groups):
    path = (
        Path(__file__).resolve().parents[2]
        / "tasks/level4"
        / "flash_attention_score_grad_enhance/golden.py"
    )
    spec = importlib.util.spec_from_file_location("attention_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qlens, klens = [3, 2], [4, 3]
    sink = sink_groups * 64
    q = torch.zeros(5, 2, 64, dtype=torch.bfloat16)
    k = torch.zeros(7 + sink, 1, 64, dtype=torch.bfloat16)
    q[..., 0] = (torch.arange(10).reshape(5, 2) + 1) / 16
    k[..., 0] = (torch.arange(7 + sink).reshape(-1, 1) + 1) / 128
    v = torch.ones_like(k)
    dy = torch.ones_like(q)
    prepared = module.get_input(
        q,
        k,
        v,
        dy,
        head_num=2,
        input_layout="TND",
        actual_seq_qlen=qlens,
        actual_seq_kvlen=klens,
        sparse_mode=mode,
        pre_tokens=1,
        next_tokens=0,
        sink_num=sink_groups,
        scale=0.125,
    )
    maximum, denominator = prepared[7:9]
    assert maximum.shape == denominator.shape == (5, 2, 8)
    qoff, koff = 0, sink
    for qlen, klen in zip(qlens, klens):
        ids = list(range(sink)) + list(range(koff, koff + klen))
        for i in range(qlen):
            for h in range(2):
                score = (k[ids, 0].double() * q[qoff + i, h].double()).sum(-1) * 0.125
                for j in range(klen):
                    visible = (
                        (j <= klen - qlen + i)
                        if mode == 3
                        else (klen - qlen + i - 1 <= j <= klen - qlen + i)
                    )
                    if not visible:
                        score[sink + j] = -40000.0
                expected_max = score.max()
                expected_sum = (score - expected_max).exp().sum()
                torch.testing.assert_close(
                    maximum[qoff + i, h],
                    expected_max.float().expand(8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    denominator[qoff + i, h],
                    expected_sum.float().expand(8),
                    rtol=2e-6,
                    atol=2e-6,
                )
        qoff += qlen
        koff += klen


@pytest.mark.parametrize("mode", [0, 3, 4])
def test_bsnd_auxiliary_packed_stats_follow_valid_token_then_head(mode):
    path = (
        Path(__file__).resolve().parents[2]
        / "tasks/level4"
        / "flash_attention_score_grad_enhance/golden.py"
    )
    spec = importlib.util.spec_from_file_location("attention_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = torch.zeros(2, 3, 2, 64, dtype=torch.bfloat16)
    k = torch.zeros(2, 4, 1, 64, dtype=torch.bfloat16)
    q[..., 0] = (torch.arange(12).reshape(2, 3, 2) + 1) / 16
    k[..., 0] = (torch.arange(8).reshape(2, 4, 1) + 1) / 128
    qlens, klens = [3, 2], [4, 3]
    prepared = module.get_input(
        q,
        k,
        torch.ones_like(k),
        torch.ones_like(q),
        head_num=2,
        input_layout="BSND",
        actual_seq_qlen=qlens,
        actual_seq_kvlen=klens,
        sparse_mode=mode,
        pre_tokens=1,
        next_tokens=0,
        scale=0.125,
    )
    # Preserve this helper's existing packed-valid-token representation;
    # this does not claim the native CANN BSND statistics format.
    maximum, denominator = prepared[7:9]
    assert maximum.shape == denominator.shape == (5, 2, 8)
    off = 0
    for b, (ql, kl) in enumerate(zip(qlens, klens)):
        for i in range(ql):
            for h in range(2):
                score = (k[b, :kl, 0].double() * q[b, i, h].double()).sum(-1) * 0.125
                for j in range(kl):
                    visible = (
                        (j <= kl - ql + i)
                        if mode == 3
                        else (kl - ql + i - 1 <= j <= kl - ql + i)
                    )
                    if not visible:
                        score[j] = -40000.0
                mx = score.max()
                sm = (score - mx).exp().sum()
                torch.testing.assert_close(
                    maximum[off + i, h], mx.float().expand(8), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    denominator[off + i, h], sm.float().expand(8), rtol=2e-6, atol=2e-6
                )
        off += ql
