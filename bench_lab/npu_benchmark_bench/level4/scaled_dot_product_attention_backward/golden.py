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

import math

import torch

"""
ScaledDotProductAttentionBackward 算子 Torch Golden 参考实现

融合缩放点积注意力反向：由前向保存的 logsumexp 重建 softmax 概率，计算
query/key/value 梯度，支持因果掩码。全程 fp32 计算，输出回铸输入 dtype。

公式:
    P = exp(scale * Q K^T - logsumexp)   # causal 时上三角置 0
    dV = P^T dO
    dP = dO V^T
    D  = rowsum(dO ⊙ O)
    dS = P ⊙ (dP - D)
    dQ = dS K scale
    dK = dS^T Q scale
"""


def _causal_mask(seq_q, seq_k):
    return torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=seq_k - seq_q + 1)


def scaled_dot_product_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
) -> tuple:
    """
    缩放点积注意力反向

    公式:
        P = exp(scale * Q K^T - logsumexp)   # causal 时上三角置 0
        dV = P^T dO
        dP = dO V^T
        D  = rowsum(dO ⊙ O)
        dS = P ⊙ (dP - D)
        dQ = dS K scale
        dK = dS^T Q scale

    Args:
        grad_output: 输出梯度 dO [B, H, S_q, D]
        query: 前向 query [B, H, S_q, D]
        key: 前向 key [B, H, S_k, D]
        value: 前向 value [B, H, S_k, D]
        attn_output: 前向注意力输出 O [B, H, S_q, D]
        logsumexp: 前向 softmax 的 logsumexp [B, H, S_q]
        is_causal: 是否使用因果掩码

    Returns:
        (grad_query, grad_key, grad_value)，shape 分别与 query/key/value 一致
    """
    batch, n_heads, seq_q, dim = query.shape
    seq_k = key.shape[2]
    scale = 1.0 / math.sqrt(dim)

    mask = _causal_mask(seq_q, seq_k) if is_causal else None

    grad_q = torch.empty(batch, n_heads, seq_q, dim, dtype=torch.float32)
    grad_k = torch.empty(batch, n_heads, seq_k, dim, dtype=torch.float32)
    grad_v = torch.empty(batch, n_heads, seq_k, dim, dtype=torch.float32)

    for b in range(batch):
        for h in range(n_heads):
            q = query[b, h].float()
            k = key[b, h].float()
            v = value[b, h].float()
            do = grad_output[b, h].float()
            o = attn_output[b, h].float()
            lse = logsumexp[b, h].float()

            scores = torch.matmul(q, k.transpose(0, 1)) * scale
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            probs = torch.exp(scores - lse.unsqueeze(-1))

            d_value = torch.matmul(probs.transpose(0, 1), do)
            d_probs = torch.matmul(do, v.transpose(0, 1))
            d_term = (do * o).sum(dim=-1, keepdim=True)
            d_scores = probs * (d_probs - d_term)

            grad_q[b, h] = torch.matmul(d_scores, k) * scale
            grad_k[b, h] = torch.matmul(d_scores.transpose(0, 1), q) * scale
            grad_v[b, h] = d_value

    return grad_q.to(query.dtype), grad_k.to(key.dtype), grad_v.to(value.dtype)


def get_input(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
    **kwargs,
) -> list:
    """按前向公式重算 attn_output 与 logsumexp，保证反向输入自洽。"""
    q = query.float()
    k = key.float()
    v = value.float()
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        scores = scores.masked_fill(_causal_mask(seq_q, seq_k), float('-inf'))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse.unsqueeze(-1))
    attn_output = torch.matmul(probs, v).to(attn_output.dtype)
    logsumexp = lse.to(logsumexp.dtype)
    return [grad_output, query, key, value, attn_output, logsumexp]
