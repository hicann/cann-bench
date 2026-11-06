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
LinformerAttention 算子 Torch Golden 参考实现

Linformer 低秩线性注意力：query/key/value 先做 d_model 维线性投影，key/value 再沿
序列维做低秩投影（S -> projection_dim），随后在投影后的 key/value 上做缩放点积注意力。
投影层与低秩投影权重全部由输入给出。

公式:
    q = Q x,  k = K x,  v = V x
    k' = E k,  v' = F v                      # E, F: S -> projection_dim
    P = softmax(q k'^T / sqrt(head_dim))
    y = O (P v')
"""


def linformer_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    key_reduce_weight: torch.Tensor,
    value_reduce_weight: torch.Tensor,
    n_heads: int,
    projection_dim: int,
) -> torch.Tensor:
    """
    Linformer 低秩线性注意力前向

    公式:
        q = Q x,  k = K x,  v = V x
        k' = E k,  v' = F v
        P = softmax(q k'^T / sqrt(head_dim))
        y = O (P v')

    Args:
        x: 输入序列特征 [B, S, D]
        q_proj_weight: query 投影权重 [D, D]
        k_proj_weight: key 投影权重 [D, D]
        v_proj_weight: value 投影权重 [D, D]
        out_proj_weight: 输出投影权重 [D, D]
        key_reduce_weight: key 序列维低秩投影权重 [projection_dim, S]
        value_reduce_weight: value 序列维低秩投影权重 [projection_dim, S]
        n_heads: 注意力头数
        projection_dim: 序列维低秩投影维度

    Returns:
        注意力输出 [B, S, D]
    """
    n_heads = int(n_heads)
    projection_dim = int(projection_dim)
    batch, sequence, d_model = x.shape
    head_dim = d_model // n_heads
    dtype = x.dtype
    linear = torch.nn.functional.linear

    q = linear(x, q_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
    k = linear(x, k_proj_weight).reshape(batch, sequence, n_heads, head_dim).permute(0, 2, 3, 1)
    v = linear(x, v_proj_weight).reshape(batch, sequence, n_heads, head_dim).permute(0, 2, 3, 1)
    k = linear(k, key_reduce_weight).transpose(-2, -1)
    v = linear(v, value_reduce_weight).transpose(-2, -1)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    weights = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().reshape(batch, sequence, d_model)
    return linear(output, out_proj_weight)
