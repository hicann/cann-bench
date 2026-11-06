#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
HstuAttention 算子 Torch Golden 参考实现

HSTU（生成式推荐）的变长因果注意力：以 SiLU 取代 softmax，且不做归一化。
q / k / v 为按序列拼接（varlen）的 [sum_L, H, D] 张量，seq_offsets 给出各序列边界。
公式:
    scores[i, j] = alpha * (q[i] . k[j])
    w[i, j]      = SiLU(scores[i, j]) / max_seq_len, 仅保留 j <= i（因果）
    y[i]         = sum_j w[i, j] * v[j]
"""


def hstu_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets,
    max_seq_len: int,
    alpha: float,
) -> torch.Tensor:
    """
    HSTU 变长因果注意力

    公式:
        scores[i, j] = alpha * (q[i] . k[j])
        w[i, j]      = SiLU(scores[i, j]) / max_seq_len, 仅保留 j <= i（因果）
        y[i]         = sum_j w[i, j] * v[j]

    Args:
        q: query 张量 [sum_L, H, D]
        k: key 张量 [sum_L, H, D]
        v: value 张量 [sum_L, H, Dv]
        seq_offsets: 长度 B+1 的序列边界（cumsum），seq_offsets[0] == 0，末值 == sum_L
        max_seq_len: 序列最大长度，同时作为打分的缩放分母
        alpha: 打分缩放系数

    Returns:
        输出张量 [sum_L, H, Dv]
    """
    if isinstance(seq_offsets, torch.Tensor):
        seq_offsets = seq_offsets.tolist()
    offsets = [int(x) for x in seq_offsets]
    total, heads, dim = q.shape
    dim_v = v.shape[-1]
    batch = len(offsets) - 1
    rows = int(max_seq_len)

    pos = torch.arange(rows, dtype=torch.long)
    lengths = torch.tensor([offsets[i + 1] - offsets[i] for i in range(batch)], dtype=torch.long)
    valid = pos.unsqueeze(0) < lengths.unsqueeze(1)
    starts = torch.tensor(offsets[:-1], dtype=torch.long)
    gidx = (starts.unsqueeze(1) + pos.unsqueeze(0)).reshape(-1).clamp(max=total - 1)

    q_pad = q.index_select(0, gidx).float().reshape(batch, rows, heads, dim)
    k_pad = k.index_select(0, gidx).float().reshape(batch, rows, heads, dim)
    v_pad = v.index_select(0, gidx).float().reshape(batch, rows, heads, dim_v)

    # 按 query 分块，避免一次物化 [B, H, L, L]
    out = torch.zeros(batch, rows, heads, dim_v, dtype=torch.float32)
    budget = 2 ** 24
    blk = max(1, min(rows, budget // max(1, batch * heads * rows)))
    for start in range(0, rows, blk):
        end = min(start + blk, rows)
        scores = torch.einsum("bqhd,bkhd->bhqk", q_pad[:, start:end], k_pad[:, :end]) * alpha
        weights = torch.nn.functional.silu(scores) / rows
        keep = (pos[start:end].unsqueeze(1) >= pos[:end].unsqueeze(0)).reshape(1, 1, end - start, end)
        weights = weights * keep
        weights = weights * valid[:, :end].reshape(batch, 1, 1, end)
        out[:, start:end] = torch.einsum("bhqk,bkhd->bqhd", weights, v_pad[:, :end])
    out = out.reshape(batch * rows, heads, dim_v)

    flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
    buf = torch.zeros(total + 1, heads, dim_v, dtype=torch.float32)
    buf.index_copy_(0, flat_idx, out)
    return buf[:total].to(v.dtype)
