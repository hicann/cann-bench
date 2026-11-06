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
import torch.nn.functional as F

"""
EMSA 算子 Torch Golden 参考实现

高效多头自注意力（Efficient Multi-Head Self-Attention）。
ratio > 1 时先对 query 做分组空间缩减（Conv2d + LayerNorm）得到 key/value 输入；
apply_transform 为真时在 softmax 前对注意力图做 1x1 卷积变换并在之后做 instance norm。
公式:
    q = W_q @ queries^T + b_q
    ratio > 1 时: kv_in = LayerNorm(Conv2d(queries, sr_weight, stride=ratio, groups=d_model))
    A = softmax(q @ k^T / sqrt(d_k))
    y = W_out @ (A @ v)^T + b_out
"""


def emsa(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    sr_weight: torch.Tensor,
    sr_bias: torch.Tensor,
    sr_norm_weight: torch.Tensor,
    sr_norm_bias: torch.Tensor,
    transform_weight: torch.Tensor,
    transform_bias: torch.Tensor,
    n_heads: int,
    height: int,
    width: int,
    ratio: int,
    d_k: int,
    d_v: int,
    apply_transform: bool,
) -> torch.Tensor:
    """
    Efficient Multi-Head Self-Attention

    公式:
        q = W_q @ queries^T + b_q
        ratio > 1 时: kv_in = LayerNorm(Conv2d(queries, sr_weight, stride=ratio, groups=d_model))
        A = softmax(q @ k^T / sqrt(d_k))
        y = W_out @ (A @ v)^T + b_out

    Args:
        queries: query 张量 [B, Lq, d_model]
        keys: key 张量 [B, Lk, d_model]
        values: value 张量 [B, Lk, d_model]
        q_weight/q_bias: query 线性投影 [H*d_k, d_model] / [H*d_k]
        k_weight/k_bias: key 线性投影 [H*d_k, d_model] / [H*d_k]
        v_weight/v_bias: value 线性投影 [H*d_v, d_model] / [H*d_v]
        out_weight/out_bias: 输出线性投影 [d_model, H*d_v] / [d_model]
        sr_weight/sr_bias: 空间缩减分组卷积 [d_model, 1, ratio+1, ratio+1] / [d_model]
        sr_norm_weight/sr_norm_bias: 空间缩减后 LayerNorm [d_model]
        transform_weight/transform_bias: 注意力图 1x1 卷积 [H, H, 1, 1] / [H]
        n_heads: 注意力头数
        height: query 空间高度，须满足 height * width == Lq
        width: query 空间宽度
        ratio: 空间缩减倍率，<= 1 时不做缩减
        d_k: 每个头的 key/query 维度
        d_v: 每个头的 value 维度
        apply_transform: 是否对注意力图做 1x1 卷积变换 + instance norm

    Returns:
        输出张量 [B, Lq, H*d_v]，dtype 与 queries 相同
    """
    n_heads = int(n_heads)
    height = int(height)
    width = int(width)
    ratio = int(ratio)
    d_k = int(d_k)
    d_v = int(d_v)

    batch, query_length, d_model = queries.shape
    key_length = keys.shape[1]
    if query_length != height * width:
        raise ValueError("height * width must match the query length")

    orig_dtype = queries.dtype
    queries_f = queries.float()
    keys_f = keys.float()
    values_f = values.float()
    q_weight_f = q_weight.float()
    q_bias_f = q_bias.float()
    k_weight_f = k_weight.float()
    k_bias_f = k_bias.float()
    v_weight_f = v_weight.float()
    v_bias_f = v_bias.float()
    out_weight_f = out_weight.float()
    out_bias_f = out_bias.float()

    q = F.linear(queries_f, q_weight_f, q_bias_f)
    q = q.view(batch, query_length, n_heads, d_k).permute(0, 2, 1, 3)

    if ratio > 1:
        reduced = queries_f.permute(0, 2, 1).reshape(batch, d_model, height, width)
        reduced = F.conv2d(
            reduced,
            sr_weight.float(),
            sr_bias.float(),
            stride=ratio,
            padding=ratio // 2,
            groups=d_model,
        )
        reduced = reduced.flatten(2).transpose(1, 2)
        reduced = F.layer_norm(
            reduced,
            (d_model,),
            sr_norm_weight.float(),
            sr_norm_bias.float(),
        )
        key_input = reduced
        value_input = reduced
        key_length = reduced.shape[1]
    else:
        key_input = keys_f
        value_input = values_f

    k = F.linear(key_input, k_weight_f, k_bias_f)
    k = k.view(batch, key_length, n_heads, d_k).permute(0, 2, 3, 1)
    v = F.linear(value_input, v_weight_f, v_bias_f)
    v = v.view(batch, key_length, n_heads, d_v).permute(0, 2, 1, 3)

    attention = torch.matmul(q, k) / math.sqrt(d_k)
    if apply_transform and n_heads > 1:
        attention = F.conv2d(attention, transform_weight.float(), transform_bias.float())
        attention = F.softmax(attention, dim=-1)
        attention = F.instance_norm(attention)
        output = torch.matmul(attention, v)
    else:
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)

    output = output.permute(0, 2, 1, 3).contiguous()
    output = output.view(batch, query_length, n_heads * d_v)
    return F.linear(output, out_weight_f, out_bias_f).to(orig_dtype)
