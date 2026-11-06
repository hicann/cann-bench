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
Sequential Polarized Self-Attention 算子 Torch Golden 参考实现

依次执行通道极化注意力与空间极化注意力，并在通道注意力的门控上使用 LayerNorm：
    ch_wv_out = conv1x1_v(x).view(B, C2, HW)
    ch_wq_out = softmax(conv1x1_q(x).view(B, HW, 1), dim=1)
    ch_z      = conv1x1_z(ch_wv_out @ ch_wq_out)
    ch_weight = sigmoid(LayerNorm(ch_z)).view(B, C, 1, 1)
    channel_out = ch_weight * x
    sp_wv_out = conv1x1_v2(channel_out).view(B, C2, HW)
    sp_wq_out = softmax(adaptive_avg_pool2d(conv1x1_q2(channel_out), 1).view(B, 1, C2), dim=-1)
    sp_weight = sigmoid(sp_wq_out @ sp_wv_out).view(B, 1, H, W)
    y = sp_weight * channel_out

源实现全程保持输入 dtype（内部不升精度）。mix_weight / use_channel / scale_factor
仅为接口兼容参数，源 forward 未使用，本 golden 同样忽略。
"""


def sequential_polarized_self_attention(
    x: torch.Tensor,
    ch_wv_weight: torch.Tensor,
    ch_wv_bias: torch.Tensor,
    ch_wq_weight: torch.Tensor,
    ch_wq_bias: torch.Tensor,
    ch_wz_weight: torch.Tensor,
    ch_wz_bias: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    sp_wv_weight: torch.Tensor,
    sp_wv_bias: torch.Tensor,
    sp_wq_weight: torch.Tensor,
    sp_wq_bias: torch.Tensor,
    mix_weight: float = None,
    use_channel: bool = None,
    scale_factor: float = None,
) -> torch.Tensor:
    """
    Sequential Polarized Self-Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        ch_wv_weight: 通道注意力 value 1x1 卷积权重 [C2, C, 1, 1]
        ch_wv_bias: 通道注意力 value 1x1 卷积偏置 [C2]
        ch_wq_weight: 通道注意力 query 1x1 卷积权重 [1, C, 1, 1]
        ch_wq_bias: 通道注意力 query 1x1 卷积偏置 [1]
        ch_wz_weight: 通道注意力 output 1x1 卷积权重 [C, C2, 1, 1]
        ch_wz_bias: 通道注意力 output 1x1 卷积偏置 [C]
        ln_weight: LayerNorm 仿射权重 [C]
        ln_bias: LayerNorm 仿射偏置 [C]
        sp_wv_weight: 空间注意力 value 1x1 卷积权重 [C2, C, 1, 1]
        sp_wv_bias: 空间注意力 value 1x1 卷积偏置 [C2]
        sp_wq_weight: 空间注意力 query 1x1 卷积权重 [C2, C, 1, 1]
        sp_wq_bias: 空间注意力 query 1x1 卷积偏置 [C2]
        mix_weight: 接口兼容参数，源实现未使用
        use_channel: 接口兼容参数，源实现未使用
        scale_factor: 接口兼容参数，源实现未使用

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    c2 = max(1, c // 2)

    # ---------- 通道极化注意力 ----------
    ch_wv_out = F.conv2d(x, ch_wv_weight, ch_wv_bias).view(b, c2, -1)
    ch_wq_out = F.conv2d(x, ch_wq_weight, ch_wq_bias).view(b, -1, 1)
    ch_wq_out = F.softmax(ch_wq_out, dim=1)
    ch_wz_out = torch.matmul(ch_wv_out, ch_wq_out).unsqueeze(-1)
    ch_wz_out = F.conv2d(ch_wz_out, ch_wz_weight, ch_wz_bias).view(b, c, 1).permute(0, 2, 1)
    ch_weight = torch.sigmoid(F.layer_norm(ch_wz_out, (c,), ln_weight, ln_bias))
    ch_weight = ch_weight.permute(0, 2, 1).view(b, c, 1, 1)
    channel_out = ch_weight * x

    # ---------- 空间极化注意力 ----------
    sp_wv_out = F.conv2d(channel_out, sp_wv_weight, sp_wv_bias).view(b, c2, -1)
    sp_wq_out = F.conv2d(channel_out, sp_wq_weight, sp_wq_bias)
    sp_wq_out = F.adaptive_avg_pool2d(sp_wq_out, (1, 1)).view(b, -1, 1)
    sp_wq_out = F.softmax(sp_wq_out.permute(0, 2, 1), dim=-1)
    sp_wz_out = torch.matmul(sp_wq_out, sp_wv_out).view(b, 1, h, w)
    sp_weight = torch.sigmoid(sp_wz_out)

    return sp_weight * channel_out
