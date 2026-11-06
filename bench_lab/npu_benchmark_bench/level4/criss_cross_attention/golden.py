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
CrissCrossAttention 算子 Torch Golden 参考实现

在特征图的每一行（横向）与每一列（纵向）上分别计算十字交叉注意力：
对每个空间位置，收集同一行的 W 个 key 与同一列的 H 个 key，拼接后 softmax，
再分别对 value 的同行/同列元素加权求和。两路结果相加后按标量 gamma 缩放并残差。
公式:
    Q = Conv1x1_q(x), K = Conv1x1_k(x), V = Conv1x1_v(x)
    e_H[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, j, x]，j == y 时置 -inf
    e_W[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, y, j]
    a = softmax(cat([e_H, e_W], dim=-1))
    o_H[b, c, y, x] = sum_j V[b, c, j, x] * a_H[b, y, x, j]
    o_W[b, c, y, x] = sum_j V[b, c, y, j] * a_W[b, y, x, j]
    y = gamma * (o_H + o_W) + x
"""


def criss_cross_attention(
    x: torch.Tensor,
    conv_q_weight: torch.Tensor,
    conv_q_bias: torch.Tensor,
    conv_k_weight: torch.Tensor,
    conv_k_bias: torch.Tensor,
    conv_v_weight: torch.Tensor,
    conv_v_bias: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Criss-Cross 注意力前向

    公式:
        Q = Conv1x1_q(x), K = Conv1x1_k(x), V = Conv1x1_v(x)
        e_H[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, j, x]，j == y 时置 -inf
        e_W[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, y, j]
        a = softmax(cat([e_H, e_W], dim=-1))
        o_H[b, c, y, x] = sum_j V[b, c, j, x] * a_H[b, y, x, j]
        o_W[b, c, y, x] = sum_j V[b, c, y, j] * a_W[b, y, x, j]
        y = gamma * (o_H + o_W) + x

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_q_weight: query 1x1 投影卷积权重 [C//8, C, 1, 1]
        conv_q_bias: query 1x1 投影卷积偏置 [C//8]
        conv_k_weight: key 1x1 投影卷积权重 [C//8, C, 1, 1]
        conv_k_bias: key 1x1 投影卷积偏置 [C//8]
        conv_v_weight: value 1x1 投影卷积权重 [C, C, 1, 1]
        conv_v_bias: value 1x1 投影卷积偏置 [C]
        gamma: 注意力支路的缩放标量 [1]

    Returns:
        Criss-Cross 注意力输出 [B, C, H, W]
    """
    x_f = x.float()
    b, c, h, w = x_f.shape

    proj_q = F.conv2d(x_f, conv_q_weight.float(), conv_q_bias.float())
    proj_k = F.conv2d(x_f, conv_k_weight.float(), conv_k_bias.float())
    proj_v = F.conv2d(x_f, conv_v_weight.float(), conv_v_bias.float())

    # 横向能量：固定列，对所有行两两打分；纵向能量：固定行，对所有列两两打分。
    # 直接用 einsum 在 [B, C, H, W] 上完成，避免逐列/逐行的 permute+bmm 展开。
    energy_h = torch.einsum("bcyx,bcjx->byxj", proj_q, proj_k)
    energy_w = torch.einsum("bcyx,bcyj->byxj", proj_q, proj_k)

    # 横向自注意力位置（行下标相等）置 -inf，与源实现一致
    diag_mask = torch.eye(h, dtype=torch.bool, device=x_f.device).view(1, h, 1, h)
    energy_h = energy_h.masked_fill(diag_mask, float("-inf"))

    attn = F.softmax(torch.cat([energy_h, energy_w], dim=-1), dim=-1)
    attn_h = attn[..., :h]
    attn_w = attn[..., h:]

    out_h = torch.einsum("bcjx,byxj->bcyx", proj_v, attn_h)
    out_w = torch.einsum("bcyj,byxj->bcyx", proj_v, attn_w)
    out = gamma.float() * (out_h + out_w) + x_f
    return out.to(dtype=x.dtype)
