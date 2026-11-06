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
Parallel Polarized Self-Attention 算子 Torch Golden 参考实现

通道支路与空间支路并行计算：

通道支路:
    ch_wv = conv1x1_{C->C/2}(x).flatten(2)          # [B, C/2, HW]
    ch_wq = softmax(conv1x1_{C->1}(x).flatten(2), dim=1)  # [B, HW, 1]
    ch_wz = conv1x1_{C/2->C}(ch_wv @ ch_wq)         # [B, C, 1, 1]
    ch_w  = sigmoid(LayerNorm_C(ch_wz))             # [B, C, 1, 1]
空间支路:
    sp_wv = conv1x1_{C->C/2}(x).flatten(2)          # [B, C/2, HW]
    sp_wq = softmax(avgpool(conv1x1_{C->C/2}(x)), dim=-1)  # [B, 1, C/2]
    sp_w  = sigmoid(sp_wq @ sp_wv)                  # [B, 1, H, W]
输出:
    y = sp_w * x + ch_w * x

源实现全程保持输入 dtype（不做 fp32 升精度）。
注意：源 Model.forward 只接收 x 与 scale_factor，且 scale_factor 在函数体内未被使用；
use_channel / use_spatial / mix_weight 仅出现在源 json 中，源 forward 未接收、未使用。
因此本 golden 中这四个属性均为接口兼容占位，不参与计算。
"""


def parallel_polarized_self_attention(
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
    use_channel: bool = False,
    use_spatial: bool = False,
    mix_weight: float = 0.5,
    scale_factor: float = 0.1,
) -> torch.Tensor:
    """
    Parallel Polarized Self-Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        ch_wv_weight: 通道支路 value 1x1 卷积权重 [C/2, C, 1, 1]
        ch_wv_bias: 通道支路 value 1x1 卷积偏置 [C/2]
        ch_wq_weight: 通道支路 query 1x1 卷积权重 [1, C, 1, 1]
        ch_wq_bias: 通道支路 query 1x1 卷积偏置 [1]
        ch_wz_weight: 通道支路输出 1x1 卷积权重 [C, C/2, 1, 1]
        ch_wz_bias: 通道支路输出 1x1 卷积偏置 [C]
        ln_weight: 通道支路 LayerNorm 仿射权重 [C]
        ln_bias: 通道支路 LayerNorm 仿射偏置 [C]
        sp_wv_weight: 空间支路 value 1x1 卷积权重 [C/2, C, 1, 1]
        sp_wv_bias: 空间支路 value 1x1 卷积偏置 [C/2]
        sp_wq_weight: 空间支路 query 1x1 卷积权重 [C/2, C, 1, 1]
        sp_wq_bias: 空间支路 query 1x1 卷积偏置 [C/2]
        use_channel: 源 forward 未使用（接口兼容占位）
        use_spatial: 源 forward 未使用（接口兼容占位）
        mix_weight: 源 forward 未使用（接口兼容占位）
        scale_factor: 源 forward 未使用（接口兼容占位）

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x
    c2 = max(1, c // 2)

    # ---------- 通道支路 ----------
    ch_wv_out = F.conv2d(x, ch_wv_weight, ch_wv_bias).view(b, c2, -1)
    ch_wq_out = F.conv2d(x, ch_wq_weight, ch_wq_bias).view(b, -1, 1)
    ch_wq_out = F.softmax(ch_wq_out, dim=1)
    ch_wz_out = torch.matmul(ch_wv_out, ch_wq_out).unsqueeze(-1)
    ch_wz_out = F.conv2d(ch_wz_out, ch_wz_weight, ch_wz_bias).view(b, c, 1).permute(0, 2, 1)
    ch_weight = torch.sigmoid(
        F.layer_norm(ch_wz_out, (c,), ln_weight, ln_bias)
    ).permute(0, 2, 1).view(b, c, 1, 1)
    channel_out = ch_weight * x

    # ---------- 空间支路 ----------
    sp_wv_out = F.conv2d(x, sp_wv_weight, sp_wv_bias).view(b, c2, -1)
    sp_wq_out = F.adaptive_avg_pool2d(
        F.conv2d(x, sp_wq_weight, sp_wq_bias), (1, 1)
    ).view(b, -1, 1)  # B, c2, 1
    sp_wq_out = F.softmax(sp_wq_out.permute(0, 2, 1), dim=-1)  # B, 1, c2
    sp_wz_out = torch.matmul(sp_wq_out, sp_wv_out).view(b, 1, h, w)
    sp_weight = torch.sigmoid(sp_wz_out)
    spatial_out = sp_weight * x

    return spatial_out + channel_out
