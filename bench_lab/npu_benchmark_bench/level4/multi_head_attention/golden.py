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
MultiHeadAttention 算子 Torch Golden 参考实现

标准多头注意力前向：query/key/value 分别经无偏置线性投影，按头拆分后做缩放点积
注意力，经输出投影得到结果。query 与 key/value 的序列长度允许不同。

公式:
    q = Q query,  k = K key,  v = V value
    P = softmax(q k^T / sqrt(head_dim))
    y = O (P v)
"""


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    多头注意力前向

    公式:
        q = Q query,  k = K key,  v = V value
        P = softmax(q k^T / sqrt(head_dim))
        y = O (P v)

    Args:
        query: query 张量 [B, Lq, D]
        key: key 张量 [B, Lk, D]
        value: value 张量 [B, Lk, D]
        q_proj_weight: query 投影权重 [D, D]
        k_proj_weight: key 投影权重 [D, D]
        v_proj_weight: value 投影权重 [D, D]
        out_proj_weight: 输出投影权重 [D, D]
        n_heads: 注意力头数

    Returns:
        注意力输出 [B, Lq, D]
    """
    n_heads = int(n_heads)
    batch, query_length, d_model = query.shape
    key_length = key.shape[1]
    head_dim = d_model // n_heads
    dtype = query.dtype
    linear = torch.nn.functional.linear

    q = linear(query, q_proj_weight).reshape(
        batch, query_length, n_heads, head_dim).transpose(1, 2)
    k = linear(key, k_proj_weight).reshape(
        batch, key_length, n_heads, head_dim).transpose(1, 2)
    v = linear(value, v_proj_weight).reshape(
        batch, key_length, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().reshape(batch, query_length, d_model)
    return linear(output, out_proj_weight)
