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
FusionAttention 算子 Torch Golden 参考实现

融合注意力前向：支持 BNSD / BSH 布局与 MHA / GQA 头扩展，按 scale 缩放 QK^T
后做 softmax，并同时返回每行 softmax 的最大值与归一化分母。

公式:
    q, k, v = layout_to_bnsd(query, key, value)
    k, v    = repeat_kv(k, v, num_q_heads // num_kv_heads)
    scores  = (q @ k^T) * scale
    p       = softmax(scores, dim=-1)
    out     = p @ v
    attention_out = bnsd_to_layout(out)
    softmax_max[i] = max_j scores[i, j]
    softmax_sum[i] = sum_j exp(scores[i, j] - softmax_max[i])
"""


def _to_bnsd(tensor: torch.Tensor, layout: str, heads: int) -> torch.Tensor:
    if layout == "BNSD":
        return tensor
    if layout == "BSH":
        b, s, h = tensor.shape
        d = h // heads
        return tensor.view(b, s, heads, d).transpose(1, 2)
    raise ValueError(f"Unsupported layout: {layout}")


def _from_bnsd(tensor: torch.Tensor, layout: str, heads: int) -> torch.Tensor:
    if layout == "BNSD":
        return tensor
    if layout == "BSH":
        b, n, s, d = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(b, s, n * d)
    raise ValueError(f"Unsupported layout: {layout}")


def fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: int,
    input_layout: str,
    scale: float = 1.0,
) -> tuple:
    """
    FusionAttention 前向

    公式:
        q, k, v = layout_to_bnsd(query, key, value)
        k, v    = repeat_kv(k, v, num_q_heads // num_kv_heads)
        scores  = (q @ k^T) * scale
        attention_out = bnsd_to_layout(softmax(scores, dim=-1) @ v)
        softmax_max[i] = max_j scores[i, j]
        softmax_sum[i] = sum_j exp(scores[i, j] - softmax_max[i])

    Args:
        query: query 张量，BNSD [B, N, Sq, D] 或 BSH [B, Sq, N*D]
        key: key 张量，BNSD [B, Nkv, Skv, Dkv] 或 BSH [B, Skv, Nkv*Dkv]
        value: value 张量，BNSD [B, Nkv, Skv, Dv] 或 BSH [B, Skv, Nkv*Dv]
        head_num: query 头数 N
        input_layout: "BNSD" 或 "BSH"
        scale: QK^T 的打分缩放系数

    Returns:
        (attention_out, softmax_max, softmax_sum)
    """
    nq = int(head_num)
    if input_layout == "BSH":
        q_hidden = query.shape[-1]
        k_hidden = key.shape[-1]
        nkv = nq * k_hidden // q_hidden if q_hidden != k_hidden else nq
    else:
        nkv = nq

    q = _to_bnsd(query, input_layout, nq).float()
    k = _to_bnsd(key, input_layout, nkv).float()
    v = _to_bnsd(value, input_layout, nkv).float()

    b, n, sq, dk = q.shape
    skv = k.shape[2]
    dv = v.shape[-1]

    group = nq // nkv
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    out = torch.zeros(b, n, sq, dv, dtype=torch.float32)
    softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32)
    softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32)

    # 按 query 行分块，避免一次物化 [B, N, Sq, Skv]
    budget = 2 ** 24
    blk = max(1, min(sq, budget // max(1, b * n * skv)))
    for start in range(0, sq, blk):
        end = min(start + blk, sq)
        scores = torch.matmul(q[:, :, start:end], k.transpose(-2, -1)) * float(scale)
        row_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - row_max)
        row_sum = exp_scores.sum(dim=-1, keepdim=True)
        out[:, :, start:end] = torch.matmul(exp_scores / row_sum, v)
        softmax_max[:, :, start:end] = row_max
        softmax_sum[:, :, start:end] = row_sum

    attention_out = _from_bnsd(out, input_layout, nq).to(query.dtype)
    softmax_max = softmax_max.expand(b, n, sq, 8).contiguous().to(torch.float32)
    softmax_sum = softmax_sum.expand(b, n, sq, 8).contiguous().to(torch.float32)
    return attention_out, softmax_max, softmax_sum
