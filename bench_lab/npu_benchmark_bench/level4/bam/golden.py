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
BAM 算子 Torch Golden 参考实现

Bottleneck Attention Module (BAM) 前向：通道注意力（全局平均池化 + 两层 MLP）
与空间注意力（1x1 降维卷积 + 两层扩张卷积 + 1x1 升维卷积）相加后经 sigmoid
得到注意力权重，并做残差连接。

公式:
    c_attn = MLP(avgpool(x))
    s_attn = Conv1x1(Conv_kd(ReLU(Conv_kd(ReLU(Conv1x1(x))))))
    s_attn = adaptive_avg_pool2d(s_attn, (H, W))
    y = x + x * sigmoid(c_attn + s_attn)
"""


def bam(
    x: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2a_weight: torch.Tensor,
    conv2a_bias: torch.Tensor,
    conv2b_weight: torch.Tensor,
    conv2b_bias: torch.Tensor,
    conv3_weight: torch.Tensor,
    conv3_bias: torch.Tensor,
    channel: int,
    reduction: int = 16,
    kernel_size: int = 3,
    dilation_val: int = 4,
) -> torch.Tensor:
    """
    BAM 瓶颈注意力模块前向

    公式:
        c_attn = MLP(avgpool(x))
        s_attn = Conv1x1(Conv_kd(ReLU(Conv_kd(ReLU(Conv1x1(x))))))
        s_attn = adaptive_avg_pool2d(s_attn, (H, W))
        y = x + x * sigmoid(c_attn + s_attn)

    Args:
        x: 输入特征图 [B, C, H, W]
        mlp_fc1_weight: 通道 MLP 第一层权重 [C // reduction, C]
        mlp_fc1_bias: 通道 MLP 第一层偏置 [C // reduction]
        mlp_fc2_weight: 通道 MLP 第二层权重 [C, C // reduction]
        mlp_fc2_bias: 通道 MLP 第二层偏置 [C]
        conv1_weight: 空间支路 1x1 降维卷积权重 [C // reduction, C, 1, 1]
        conv1_bias: 空间支路 1x1 降维卷积偏置 [C // reduction]
        conv2a_weight: 第一层扩张卷积权重 [C // reduction, C // reduction, keff, keff]
        conv2a_bias: 第一层扩张卷积偏置 [C // reduction]
        conv2b_weight: 第二层扩张卷积权重 [C // reduction, C // reduction, keff, keff]
        conv2b_bias: 第二层扩张卷积偏置 [C // reduction]
        conv3_weight: 空间支路 1x1 升维卷积权重 [1, C // reduction, 1, 1]
        conv3_bias: 空间支路 1x1 升维卷积偏置 [1]
        channel: 输入通道数，保留用于接口兼容，须等于 x.shape[1]
        reduction: 通道瓶颈降维比
        kernel_size: 扩张卷积核大小
        dilation_val: 扩张卷积的扩张率

    Returns:
        注意力细化后的特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape

    # 与源实现一致：扩张卷积的等效核超出最小空间维时收缩核与扩张率
    eff = dilation_val * (kernel_size - 1) + 1
    min_size = min(h, w)
    if eff > min_size:
        if min_size <= 1:
            safe_dilation = 1
            safe_kernel = 1
        else:
            safe_dilation = min(dilation_val, min_size)
            safe_kernel = min(kernel_size, (min_size - 1) // safe_dilation + 1)
            if safe_kernel > 1 and safe_kernel % 2 == 0:
                safe_kernel -= 1
        kernel_size = safe_kernel
        dilation_val = safe_dilation

    # ---------- 通道注意力 ----------
    y_ch = F.adaptive_avg_pool2d(x, 1).reshape(b, c)
    y_ch = F.linear(y_ch, mlp_fc1_weight, mlp_fc1_bias)
    y_ch = F.relu(y_ch)
    y_ch = F.linear(y_ch, mlp_fc2_weight, mlp_fc2_bias).view(b, c, 1, 1)

    # ---------- 空间注意力 ----------
    y_sp = F.conv2d(x, conv1_weight, conv1_bias)
    y_sp = F.relu(F.conv2d(y_sp, conv2a_weight, conv2a_bias, padding=dilation_val, dilation=dilation_val))
    y_sp = F.relu(F.conv2d(y_sp, conv2b_weight, conv2b_bias, padding=dilation_val, dilation=dilation_val))
    y_sp = F.conv2d(y_sp, conv3_weight, conv3_bias)
    y_sp = F.adaptive_avg_pool2d(y_sp, (h, w))

    # ---------- 融合 + 残差 ----------
    attn = torch.sigmoid(y_ch + y_sp)
    return x + x * attn
