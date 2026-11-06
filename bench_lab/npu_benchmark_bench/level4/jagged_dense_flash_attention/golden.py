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
JaggedDenseFlashAttention 算子 Torch Golden 参考实现

变长 query（jagged）× dense key/value 的 attention 前向：
    logits = Q_b @ K_b^T + attn_bias[b, :L_b]
    P_b    = softmax(logits, dim=0)          # 沿 jagged 序列维归一化
    out[b] = P_b^T @ V_b                     # [T, D]
全程 fp32（allow_tf32=False），最后回铸 q.dtype。

公式:
    L_b = min(jagged_offsets[b+1] - jagged_offsets[b], max_seq_len)
    S_b = Q_b @ K_b^T + attn_bias[b, :L_b]
    out[b] = softmax(S_b, dim=0)^T @ V_b
"""


def _build_offsets(seed_offsets: torch.Tensor, total: int, batch: int,
                   max_seq_len: int, device: torch.device) -> torch.Tensor:
    """把随机占位 offsets 规整为合法、单调不减的 int32 offsets。

    cases.yaml 的 value_range 无法表达 offsets 的单调性与长度上界，因此按
    (total, batch, max_seq_len) 重建：
      - total == batch * max_seq_len：等长批，offsets = [0, msl, 2*msl, ...]
      - 否则把 total 行按占位值分配的抖动长度分给 batch 段，每段长度 <= max_seq_len
    """
    if batch <= 0:
        return torch.zeros(1, dtype=torch.int32, device=device)
    if total == batch * max_seq_len:
        return (torch.arange(batch + 1, dtype=torch.int64, device=device) * max_seq_len).to(torch.int32)

    total_eff = min(total, batch * max_seq_len)
    weights = [int(v) % 1024 + 1 for v in seed_offsets.reshape(-1).to(torch.int64).abs().tolist()]
    if len(weights) < batch:
        weights.extend([1] * (batch - len(weights)))

    lengths = [0] * batch
    remaining = total_eff
    for i in range(batch):
        segs_left = batch - i - 1
        if total_eff >= batch:
            lo = max(1, remaining - segs_left * max_seq_len)
            hi = min(max_seq_len, remaining - segs_left)
        else:
            lo = max(0, remaining - segs_left * max_seq_len)
            hi = min(max_seq_len, remaining)
        take = lo if hi <= lo else lo + weights[i] % (hi - lo + 1)
        lengths[i] = take
        remaining -= take

    offsets = [0] * (batch + 1)
    acc = 0
    for i, length in enumerate(lengths):
        acc += length
        offsets[i + 1] = acc
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def jagged_dense_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 attn_bias: torch.Tensor, jagged_offsets: torch.Tensor,
                                 max_seq_len: int) -> torch.Tensor:
    """
    变长 query × dense key/value attention 前向

    公式:
        L_b = min(jagged_offsets[b+1] - jagged_offsets[b], max_seq_len)
        S_b = Q_b @ K_b^T + attn_bias[b, :L_b]
        out[b] = softmax(S_b, dim=0)^T @ V_b

    Args:
        q: 变长 query [sum_L, D]
        k: dense key [B, D, T]
        v: 变长 value [sum_L, D]
        attn_bias: 每个 batch 的加性 bias [B, max_seq_len, T]
        jagged_offsets: 各 batch 在 sum_L 维上的起止偏移 [B+1]，int32，单调不减
        max_seq_len: 每条序列参与计算的最大行数

    Returns:
        输出张量 [B, T, D]，dtype 与 q 相同
    """
    batch, dim, slots = k.shape
    total = q.shape[0]
    rows = max_seq_len

    if total == batch * rows:
        # 等长批：q / v 本身即 [B, rows, D] 连续布局
        q_pad = q.reshape(batch, rows, dim).to(torch.float32)
        v_pad = v.reshape(batch, rows, dim).to(torch.float32)
        logits = torch.baddbmm(attn_bias[:, :rows, :].to(torch.float32), q_pad, k.to(torch.float32))
        probs = torch.softmax(logits, dim=1)
        return torch.bmm(probs.transpose(1, 2), v_pad).to(q.dtype)

    pos = torch.arange(rows, dtype=torch.int64)
    lengths = (jagged_offsets[1:] - jagged_offsets[:-1]).to(torch.int64)
    valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
    idx = (jagged_offsets[:-1].reshape(batch, 1).to(torch.int64) + pos.reshape(1, rows)).clamp(min=0, max=total - 1)

    q_pad = torch.index_select(q, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, dim)
    v_pad = torch.index_select(v, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, dim)

    logits = torch.baddbmm(attn_bias[:, :rows, :].to(torch.float32), q_pad, k.to(torch.float32))
    logits = logits.masked_fill(~valid[:, :, None], float("-inf"))
    probs = torch.softmax(logits, dim=1)
    return torch.bmm(probs.transpose(1, 2), v_pad).to(q.dtype)


def get_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: torch.Tensor,
              jagged_offsets: torch.Tensor, max_seq_len: int = 1, **kwargs) -> list:
    """把随机生成的 offsets 规整为合法单调范围（cases.yaml 无法表达该约束）。"""
    batch = jagged_offsets.reshape(-1).numel() - 1
    total = q.shape[0]
    return [q, k, v, attn_bias, _build_offsets(jagged_offsets, total, batch, max_seq_len, q.device)]
