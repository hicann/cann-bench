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
DecoderMhaAttention 算子 Torch Golden 参考实现

SGLang 风格 decode 阶段 MHA 注意力（split-KV 两段式的数学等价实现）。
对每条序列按 kv_indptr / kv_indices 取出命中的 KV pool token，做全局 softmax
注意力；split-KV 归并后的结果与一次全局 softmax 等价。

公式:
    idx[b] = kv_indices[kv_indptr[b] : kv_indptr[b+1]]
    scores[b, h, j] = sm_scale * q[b, h, :] . k_buffer[idx[b][j], h, :]
    o[b, h, :] = softmax(scores[b, h, :]) @ v_buffer[idx[b], h, :]
"""


def decoder_mha_attention(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    max_kv_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """
    Decoder MHA 注意力前向（decode，split-KV 等价全局 softmax）

    公式:
        scores = sm_scale * q k^T，softmax 后与 v 相乘

    Args:
        q: 每条序列一个新 token 的 query [B, H, D]
        k_buffer: 全局 KV pool 中的 key [P, H, D]
        v_buffer: 全局 KV pool 中的 value [P, H, Dv]
        kv_indptr: 每条序列在 kv_indices 上的起止 [B + 1]，int32
        kv_indices: 每条序列命中的 KV pool token 号 [nnz]，int32
        max_kv_len: 批内最长 KV 长度（形状提示）
        sm_scale: 打分缩放因子，通常为 1/sqrt(D)

    Returns:
        注意力输出 [B, H, Dv]，dtype 与 q 一致
    """
    device = q.device
    batch, heads, dim = q.shape
    dim_v = v_buffer.shape[-1]
    total = kv_indices.shape[0]
    rows = int(max_kv_len)

    if total == batch * rows:
        # 等长批：kv_indices 本身即 [B, max_kv_len] 连续布局，无需 padding 与掩码
        token = kv_indices
        valid = None
    else:
        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = kv_indptr[1:] - kv_indptr[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        gather = kv_indptr[:-1].reshape(batch, 1) + pos.reshape(1, rows)
        token = torch.index_select(kv_indices, 0, gather.reshape(-1).clamp(max=total - 1))

    k = torch.index_select(k_buffer, 0, token.reshape(-1)).to(torch.float32)
    v = torch.index_select(v_buffer, 0, token.reshape(-1)).to(torch.float32)
    k = k.reshape(batch, rows, heads, dim)
    v = v.reshape(batch, rows, heads, dim_v)

    scores = torch.einsum("bhd,bkhd->bhk", q.to(torch.float32), k) * sm_scale
    if valid is not None:
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhk,bkhd->bhd", probs, v)
    return out.to(q.dtype)


def get_input(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    max_kv_len: int = 1,
    sm_scale: float = 1.0,
    **kwargs,
):
    """重建满足算子里约的 KV 变长索引结构。

    kv_indptr 必须是单调前缀和，kv_indices 必须是 KV pool 中的合法（可被打散的）
    token 号，单区间 value_range 无法表达。这里按实际 shape 重新构造：
      - nnz 均分到 B 条序列（每条不超过 max_kv_len）；
      - kv_indices 用固定种子对 pool 做打乱后取前 nnz 个。
    q / k_buffer / v_buffer 原样保留。

    Returns:
        [q, k_buffer, v_buffer, kv_indptr, kv_indices]
    """
    batch = int(kv_indptr.numel()) - 1
    pool = int(k_buffer.shape[0])
    total = int(kv_indices.numel())
    rows = int(max_kv_len)
    device = kv_indptr.device

    if total == batch * rows and rows > 0:
        lens = [rows] * batch
    else:
        base, rem = divmod(total, max(batch, 1))
        lens = [min(base + (1 if b < rem else 0), rows) for b in range(batch)]
        short = total - sum(lens)
        for b in range(batch):
            take = min(short, rows - lens[b])
            if take > 0:
                lens[b] += take
                short -= take

    indptr = [0]
    for length in lens:
        indptr.append(indptr[-1] + length)

    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(pool, generator=gen).to(torch.int32)
    indices = perm[:total]
    if total > 0:
        indices = indices.clamp(max=max(pool - 1, 0))

    new_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    new_indices = indices.to(device)
    return [q, k_buffer, v_buffer, new_indptr, new_indices]
