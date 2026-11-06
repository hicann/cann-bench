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
ViTAttention 算子 Torch Golden 参考实现

Vision Transformer 自注意力前向：输入 token 序列分别经 query/key/value 无偏置线性
投影，按头拆分后做缩放点积注意力，再经输出投影得到结果。

公式:
    q = Q x,  k = K x,  v = V x
    P = softmax(q k^T / sqrt(head_dim))
    y = O (P v)
"""


def vit_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    ViT 自注意力前向

    公式:
        q = Q x,  k = K x,  v = V x
        P = softmax(q k^T / sqrt(head_dim))
        y = O (P v)

    Args:
        x: 输入 token 序列 [B, S, D]
        q_proj_weight: query 投影权重 [D, D]
        k_proj_weight: key 投影权重 [D, D]
        v_proj_weight: value 投影权重 [D, D]
        out_proj_weight: 输出投影权重 [D, D]
        n_heads: 注意力头数

    Returns:
        注意力输出 [B, S, D]
    """
    n_heads = int(n_heads)
    batch, sequence, d_model = x.shape
    head_dim = d_model // n_heads
    dtype = x.dtype
    linear = torch.nn.functional.linear

    q = linear(x, q_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
    k = linear(x, k_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
    v = linear(x, v_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().reshape(batch, sequence, d_model)
    return linear(output, out_proj_weight)
