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
NsaTopkSparseAttention 算子 Torch Golden 参考实现

DeepSeek Native Sparse Attention 的选块注意力前向。对每个 query 位置，仅在与
topk_idx 选中的 KV 块且因果可见（j <= i）的 key 上做 softmax 注意力。

公式:
    allow(i, h) = { j | (j // block_size) in topk_idx[h, i], j <= i }
    p = softmax_{j in allow(i, h)}(sm_scale * q_i . k_j)
    y_i = sum_{j in allow(i, h)} p_j * v_j
"""


def nsa_topk_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_size: int,
    max_seqlen: int,
    sm_scale: float,
) -> torch.Tensor:
    """
    NSA TopK 稀疏注意力前向

    公式: y_i = softmax_{j in allow(i,h)}(sm_scale * q_i . k_j) @ v

    Args:
        q: varlen 拼接的 query [total_len, Hq, D]
        k: varlen 拼接的 key [total_len, Hk, D]
        v: varlen 拼接的 value [total_len, Hk, D]
        topk_idx: 每个 query 选中的 KV 块号 [Hk, total_len, topk]，-1 表示 padding
        cu_seqlens: 序列前缀和 [B + 1] int32
        block_size: KV 分块大小
        max_seqlen: 批内最长序列长度
        sm_scale: softmax 缩放因子

    Returns:
        注意力输出 [total_len, Hq, D]
    """
    device = q.device
    total, hq, dim = q.shape
    hk = k.shape[1]
    group = hq // hk
    batch = cu_seqlens.shape[0] - 1
    rows = int(max_seqlen)
    blocks = (rows + block_size - 1) // block_size
    topk = topk_idx.shape[-1]
    pos = torch.arange(rows, device=device)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    uniform = total == batch * rows
    if uniform:
        # 等长批：各条长度均等于 max_seqlen，q/k/v/topk_idx 本身即按 [B, rows, ...] 排布
        q_pad = q.reshape(batch, rows, hk, group, dim).to(torch.float32)
        k_pad = k.reshape(batch, rows, hk, dim).to(torch.float32)
        v_pad = v.reshape(batch, rows, hk, dim).to(torch.float32)
        sel_idx = topk_idx.reshape(hk, batch, rows, topk)
        valid = None
    else:
        gidx = (cu_seqlens[:-1].reshape(batch, 1) + pos.reshape(1, rows)).reshape(-1)
        gidx = gidx.clamp(min=0, max=total - 1).to(torch.int64)
        q_pad = torch.index_select(q, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
        k_pad = torch.index_select(k, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
        v_pad = torch.index_select(v, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
        sel_idx = torch.index_select(topk_idx, 1, gidx).reshape(hk, batch, rows, topk)
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)

    jblk = torch.arange(rows, device=device) // block_size       # [rows]，每个 key 位置所属块号
    out = torch.empty(batch, rows, hk, group, dim, dtype=torch.float32, device=device)
    # 按 query 分块，避免物化 [B, Hk, G, rows, rows] 的完整 score 张量
    chunk = max(1, 16_000_000 // max(1, hk * group * rows))

    for b in range(batch):
        for h in range(hk):
            sel_h = sel_idx[h, b].long()                          # [rows, topk]
            # 选块表 -> 布尔矩阵 [rows, blocks + 1]，-1 落到末尾哨兵列后丢弃
            sel_clamped = torch.where(sel_h < 0, torch.full_like(sel_h, blocks), sel_h).clamp(0, blocks)
            selmat = torch.zeros(rows, blocks + 1, dtype=torch.bool, device=device)
            selmat.scatter_(-1, sel_clamped, True)

            k_bh = k_pad[b, :, h, :]                              # [rows, D]
            v_bh = v_pad[b, :, h, :]                              # [rows, D]
            q_bh = q_pad[b, :, h, :, :]                           # [rows, G, D]

            for c0 in range(0, rows, chunk):
                c1 = min(rows, c0 + chunk)
                q_pos = torch.arange(c0, c1, device=device)
                allowed = selmat[c0:c1, :blocks].index_select(1, jblk)
                causal = q_pos.reshape(-1, 1) >= pos.reshape(1, rows)
                keep = allowed & causal
                if valid is not None:
                    keep = keep & valid[b].reshape(1, rows)

                scores = torch.einsum("qgd,kd->gqk", q_bh[c0:c1], k_bh) * sm_scale
                scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[b, c0:c1, h, :, :] = torch.einsum("gqk,kd->qgd", probs, v_bh)

    if uniform:
        return out.reshape(batch * rows, hq, dim).to(q.dtype)

    flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
    buf = torch.zeros(total + 1, hq, dim, dtype=torch.float32, device=device)
    buf.index_copy_(0, flat_idx, out.reshape(batch * rows, hq, dim))
    return buf[:total].to(q.dtype)


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_size: int = 64,
    max_seqlen: int = 1,
    sm_scale: float = 1.0,
    **kwargs,
):
    """重建满足本算子契约的结构化输入（同时替换 golden 与候选输入，比较公平）。

    单区间 value_range 无法表达两条契约：
      1. cu_seqlens 必须是单调前缀和，且 cu_seqlens[-1] == total_len；
      2. topk_idx 的每个块号必须落在 [0, ceil(len/block_size)) 内，且每个 query
         至少选中一个因果可见块（保证 softmax 分母非空）。
    这里按实际 shape 重生成 cu_seqlens 与 topk_idx，q/k/v 原样保留。

    Returns:
        [q, k, v, topk_idx, cu_seqlens]，顺序与 nsa_topk_sparse_attention 签名一致。
    """
    total = int(q.shape[0])
    hk = int(k.shape[1])
    batch = int(cu_seqlens.numel()) - 1
    block_size = int(block_size)
    rows = int(max_seqlen)
    topk = int(topk_idx.shape[-1])

    # 均匀分配 total 到 batch 条序列，且每条不超过 max_seqlen
    base = total // batch
    rem = total % batch
    lengths = [base + (1 if b < rem else 0) for b in range(batch)]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    new_cu = torch.tensor(cu, dtype=cu_seqlens.dtype, device=cu_seqlens.device)

    # 每个 query 在因果可见块中取前 min(visible, topk) 个（升序），不足补 -1
    chunks = []
    for length in lengths:
        nblk = (length + block_size - 1) // block_size
        local = torch.arange(length).reshape(1, length, 1)
        idx = torch.arange(min(topk, max(nblk, 1)), dtype=torch.int32).reshape(1, 1, -1)
        idx = idx.expand(hk, length, -1).clone()
        if idx.shape[-1] < topk:
            pad = torch.full((hk, length, topk - idx.shape[-1]), -1, dtype=torch.int32)
            idx = torch.cat([idx, pad], dim=-1)
        idx = torch.where(idx > (local // block_size).to(torch.int32),
                          torch.full_like(idx, -1), idx)
        chunks.append(idx)
    new_topk = torch.cat(chunks, dim=1).contiguous() if chunks else \
        torch.full((hk, 0, topk), -1, dtype=torch.int32)
    new_topk = new_topk.to(dtype=topk_idx.dtype, device=topk_idx.device)
    if new_topk.shape[1] != total:
        pad = torch.full((hk, total - new_topk.shape[1], topk), -1, dtype=new_topk.dtype)
        new_topk = torch.cat([new_topk[:, :total], pad], dim=1)

    return [q, k, v, new_topk, new_cu]
