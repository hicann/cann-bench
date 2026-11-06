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
JaggedSoftmax 算子 Torch Golden 参考实现

变长序列（jagged）softmax：对每条序列在拼接维上的前 max_seq_len 行沿序列维做归一化，
超出 max_seq_len 的行输出 0。累加全程 fp32，最后回铸输入 dtype。

公式:
    N_b = min(offsets[b+1] - offsets[b], max_seq_len)
    y[start_b + i, h] = exp(x[start_b + i, h] - m_b) / sum_{j=0}^{N_b-1} exp(x[start_b + j, h] - m_b),  i < N_b
    y[start_b + i, h] = 0,  i >= N_b
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


def jagged_softmax(x: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    变长序列 softmax

    公式:
        N_b = min(offsets[b+1] - offsets[b], max_seq_len)
        y[start_b + i, h] = softmax(x[start_b : start_b + N_b], dim=0)[i, h],  i < N_b
        y[start_b + i, h] = 0,  i >= N_b

    Args:
        x: 变长序列拼接后的打分/特征 [sum_L, H]
        x_offsets: 各序列起止偏移 [B+1]，int32，单调不减
        max_seq_len: 每条序列参与归一化的最大行数

    Returns:
        与 x 同形同 dtype 的 softmax 结果 [sum_L, H]
    """
    total, heads = x.shape
    batch = x_offsets.shape[0] - 1
    rows = max_seq_len

    if total == batch * rows:
        # 等长批：x 本身即 [B, rows, H] 连续布局
        padded = x.reshape(batch, rows, heads).to(torch.float32)
        return torch.softmax(padded, dim=1).reshape(total, heads).to(x.dtype)

    pos = torch.arange(rows, dtype=torch.int64)
    lengths = (x_offsets[1:] - x_offsets[:-1]).to(torch.int64)
    valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
    idx = (x_offsets[:-1].reshape(batch, 1).to(torch.int64) + pos.reshape(1, rows)).clamp(min=0, max=total - 1)

    padded = torch.index_select(x, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, heads)
    padded = padded.masked_fill(~valid[:, :, None], float("-inf"))
    probs = torch.softmax(padded, dim=1)

    flat_idx = torch.where(valid.reshape(-1), idx.reshape(-1),
                           torch.full_like(idx.reshape(-1), total))
    buf = torch.zeros(total + 1, heads, dtype=torch.float32, device=x.device)
    buf.index_copy_(0, flat_idx, probs.reshape(-1, heads))
    return buf[:total].to(x.dtype)


def get_input(x: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int = 1, **kwargs) -> list:
    """把随机生成的 offsets 规整为合法单调范围（cases.yaml 无法表达该约束）。"""
    batch = x_offsets.reshape(-1).numel() - 1
    total = x.shape[0]
    return [x, _build_offsets(x_offsets, total, batch, max_seq_len, x.device)]
