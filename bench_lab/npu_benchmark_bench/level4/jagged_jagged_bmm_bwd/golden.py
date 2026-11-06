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
JaggedJaggedBmmBwd 算子 Torch Golden 参考实现

变长序列批量矩阵乘 Z[b] = X_b^T @ Y_b 的反向：
    dX_b = Y_b @ dZ_b^T
    dY_b = X_b @ dZ_b
每条序列只取前 L_b = min(len, max_seq_len) 行参与，超出行梯度保持 0。
累加全程 fp32，最后回铸输入 dtype。

公式:
    L_b = min(offsets[b+1] - offsets[b], max_seq_len)
    dX_b = Y_b @ dZ_b^T
    dY_b = X_b @ dZ_b
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


def jagged_jagged_bmm_bwd(x: torch.Tensor, y: torch.Tensor, x_offsets: torch.Tensor,
                          grad_output: torch.Tensor, max_seq_len: int):
    """
    变长序列批量矩阵乘反向

    公式:
        L_b = min(offsets[b+1] - offsets[b], max_seq_len)
        dX_b = Y_b @ dZ_b^T      # [L_b, T] @ [T, D] -> [L_b, D]
        dY_b = X_b @ dZ_b        # [L_b, D] @ [D, T] -> [L_b, T]

    Args:
        x: 前向左矩阵 [sum_L, D]
        y: 前向右矩阵 [sum_L, T]
        x_offsets: 各序列起止偏移 [B+1]，int32，单调不减
        grad_output: 上游梯度 dZ [B, D, T]
        max_seq_len: 每条序列参与计算的最大行数

    Returns:
        (grad_x, grad_y)：分别为 x、y 的梯度，形状 [sum_L, D]、[sum_L, T]
    """
    total, dim = x.shape
    cols = y.shape[1]
    batch = x_offsets.shape[0] - 1
    rows = max_seq_len
    grad_z = grad_output.to(torch.float32)

    if total == batch * rows:
        # 等长批：无需 padding 与掩码
        x_pad = x.reshape(batch, rows, dim).to(torch.float32)
        y_pad = y.reshape(batch, rows, cols).to(torch.float32)
        grad_x = torch.bmm(y_pad, grad_z.transpose(1, 2)).reshape(total, dim)
        grad_y = torch.bmm(x_pad, grad_z).reshape(total, cols)
        return grad_x.to(x.dtype), grad_y.to(y.dtype)

    pos = torch.arange(rows, dtype=torch.int64)
    lengths = (x_offsets[1:] - x_offsets[:-1]).to(torch.int64)
    valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
    idx = (x_offsets[:-1].reshape(batch, 1).to(torch.int64) + pos.reshape(1, rows)).clamp(min=0, max=total - 1)

    x_pad = torch.index_select(x, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, dim)
    y_pad = torch.index_select(y, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, cols)
    grad_x = torch.bmm(y_pad, grad_z.transpose(1, 2))
    grad_y = torch.bmm(x_pad, grad_z)

    flat_idx = torch.where(valid.reshape(-1), idx.reshape(-1),
                           torch.full_like(idx.reshape(-1), total))
    buf_x = torch.zeros(total + 1, dim, dtype=torch.float32, device=x.device)
    buf_y = torch.zeros(total + 1, cols, dtype=torch.float32, device=x.device)
    buf_x.index_copy_(0, flat_idx, grad_x.reshape(-1, dim))
    buf_y.index_copy_(0, flat_idx, grad_y.reshape(-1, cols))
    return buf_x[:total].to(x.dtype), buf_y[:total].to(y.dtype)


def get_input(x: torch.Tensor, y: torch.Tensor, x_offsets: torch.Tensor,
              grad_output: torch.Tensor, max_seq_len: int = 1, **kwargs) -> list:
    """把随机生成的 offsets 规整为合法单调范围（cases.yaml 无法表达该约束）。"""
    batch = x_offsets.reshape(-1).numel() - 1
    total = x.shape[0]
    return [x, y, _build_offsets(x_offsets, total, batch, max_seq_len, x.device), grad_output]
