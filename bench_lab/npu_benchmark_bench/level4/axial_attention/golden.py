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
import torch.nn.functional as F

"""
AxialAttention 算子 Torch Golden 参考实现

Axial Attention：沿高度轴与宽度轴分别做多头自注意力，两路输出相加。
    y_H = out_proj(softmax(q_H k_H^T / sqrt(d_h)) v_H)
    y_W = out_proj(softmax(q_W k_W^T / sqrt(d_h)) v_W)
    y   = reshape(y_H + y_W)  -> [B, C, H, W]
其中 q/kv 由共享的 to_q / to_kv 线性层得到，源实现的 to_q / to_kv 无 bias。
"""


def axial_attention(
    x: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_kv_weight: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    dim: int,
    heads: int,
    dim_heads: int,
) -> torch.Tensor:
    """
    Axial Attention（H 轴 + W 轴自注意力相加）

    公式:
        y_H = out_proj(softmax(q_H k_H^T / sqrt(d_h)) v_H)
        y_W = out_proj(softmax(q_W k_W^T / sqrt(d_h)) v_W)
        y   = reshape(y_H + y_W)  -> [B, C, H, W]

    Args:
        x: 输入特征图 [B, C, H, W]
        to_q_weight: query 投影权重 [heads * d_h, C]
        to_kv_weight: key/value 投影权重 [2 * heads * d_h, C]
        to_out_weight: 输出投影权重 [C, heads * d_h]
        to_out_bias: 输出投影偏置 [C]
        dim: 特征维（须等于 C）
        heads: 注意力头数
        dim_heads: 每个头的维度 d_h

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    effective_dim = int(dim)
    effective_heads = int(heads)
    head_dim = int(dim_heads) if dim_heads is not None else effective_dim // effective_heads

    x_perm = x.permute(0, 2, 3, 1).contiguous().float()  # [B, H, W, C]
    w_q = to_q_weight.float()
    w_kv = to_kv_weight.float()
    w_out = to_out_weight.float()
    b_out = to_out_bias.float()

    # Height axis: 每条序列长 H，共 B*W 条
    seq_h = x_perm.permute(0, 2, 1, 3).contiguous().view(b * w, h, c)
    q_h = F.linear(seq_h, w_q)
    kv_h = F.linear(seq_h, w_kv)
    k_h, v_h = kv_h.chunk(2, dim=-1)
    q_h = q_h.view(q_h.shape[0], q_h.shape[1], effective_heads, -1).transpose(1, 2)
    k_h = k_h.view(k_h.shape[0], k_h.shape[1], effective_heads, -1).transpose(1, 2)
    v_h = v_h.view(v_h.shape[0], v_h.shape[1], effective_heads, -1).transpose(1, 2)
    attn_h = F.softmax(torch.matmul(q_h, k_h.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1)
    out_h = torch.matmul(attn_h, v_h).transpose(1, 2).contiguous().view(seq_h.shape[0], seq_h.shape[1], -1)
    out_h = F.linear(out_h, w_out, b_out).view(b, w, h, c).permute(0, 2, 1, 3)

    # Width axis: 每条序列长 W，共 B*H 条
    seq_w = x_perm.view(b * h, w, c)
    q_w = F.linear(seq_w, w_q)
    kv_w = F.linear(seq_w, w_kv)
    k_w, v_w = kv_w.chunk(2, dim=-1)
    q_w = q_w.view(q_w.shape[0], q_w.shape[1], effective_heads, -1).transpose(1, 2)
    k_w = k_w.view(k_w.shape[0], k_w.shape[1], effective_heads, -1).transpose(1, 2)
    v_w = v_w.view(v_w.shape[0], v_w.shape[1], effective_heads, -1).transpose(1, 2)
    attn_w = F.softmax(torch.matmul(q_w, k_w.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1)
    out_w = torch.matmul(attn_w, v_w).transpose(1, 2).contiguous().view(seq_w.shape[0], seq_w.shape[1], -1)
    out_w = F.linear(out_w, w_out, b_out).view(b, h, w, c)

    out = out_h + out_w
    return out.permute(0, 3, 1, 2).contiguous().to(x.dtype)
