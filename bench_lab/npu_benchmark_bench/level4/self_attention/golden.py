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
SelfAttention 算子 Torch Golden 参考实现

多头自注意力前向（Q=K=V=x）：Q/K/V 输入投影 + 按 sqrt(head_dim) 缩放的
softmax 注意力 + 输出投影。原实现内部随机初始化并缓存的四个 nn.Linear
权重已全部提升为显式张量入参。

公式:
    q = x @ Wq^T,  k = x @ Wk^T,  v = x @ Wv^T
    q, k, v = split_heads(q, k, v, n_heads)
    scores  = (q / sqrt(D)) @ k^T
    y       = softmax(scores, dim=-1) @ v
    y       = merge_heads(y) @ Wo^T
"""


def self_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    SelfAttention 前向

    公式:
        q = x @ Wq^T,  k = x @ Wk^T,  v = x @ Wv^T
        q, k, v = split_heads(q, k, v, n_heads)
        scores  = (q / sqrt(D)) @ k^T
        y       = softmax(scores, dim=-1) @ v
        y       = merge_heads(y) @ Wo^T

    Args:
        x: 输入序列特征 [B, L, d_model]
        q_proj_weight: query 投影权重 [d_model, d_model]
        k_proj_weight: key 投影权重 [d_model, d_model]
        v_proj_weight: value 投影权重 [d_model, d_model]
        out_proj_weight: 输出投影权重 [d_model, d_model]
        n_heads: 注意力头数

    Returns:
        自注意力输出 [B, L, d_model]
    """
    heads = int(n_heads)
    batch, length, d_model = x.shape
    if d_model % heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = d_model // heads
    inv_scale = 1.0 / (head_dim ** 0.5)

    xf = x.float()
    q = torch.matmul(xf, q_proj_weight.float().transpose(-1, -2))
    k = torch.matmul(xf, k_proj_weight.float().transpose(-1, -2))
    v = torch.matmul(xf, v_proj_weight.float().transpose(-1, -2))

    q = q.view(batch, length, heads, head_dim).transpose(1, 2)
    k = k.view(batch, length, heads, head_dim).transpose(1, 2)
    v = v.view(batch, length, heads, head_dim).transpose(1, 2)

    out = torch.zeros(batch, heads, length, head_dim, dtype=torch.float32)
    # 按 query 行分块，避免一次物化 [B, N, L, L]
    budget = 2 ** 24
    blk = max(1, min(length, budget // max(1, batch * heads * length)))
    for start in range(0, length, blk):
        end = min(start + blk, length)
        scores = torch.matmul(q[:, :, start:end] * inv_scale, k.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, start:end] = torch.matmul(probs, v)

    out = out.transpose(1, 2).contiguous().view(batch, length, d_model)
    y = torch.matmul(out, out_proj_weight.float().transpose(-1, -2))
    return y.to(x.dtype)
