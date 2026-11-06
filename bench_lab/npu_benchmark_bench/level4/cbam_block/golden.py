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
CBAMBlock 算子 Torch Golden 参考实现

Convolutional Block Attention Module (CBAM) 前向：通道注意力（avg-pool 与
max-pool 两路共享 MLP）与空间注意力（通道 mean/max 拼接后 2->1 卷积）顺序
加权，最后加残差。

公式:
    ca = sigmoid(MLP(avgpool(x)) + MLP(maxpool(x)))
    out = x * ca
    sa = sigmoid(Conv2d(cat([max_c(out), mean_c(out)])))
    y = out * sa + x
"""


def cbam_block(
    x: torch.Tensor,
    ca_fc1_weight: torch.Tensor,
    ca_fc2_weight: torch.Tensor,
    sa_conv_weight: torch.Tensor,
    sa_conv_bias: torch.Tensor,
    channel: int,
    reduction: int = 16,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    CBAM 卷积块注意力模块前向

    公式:
        ca = sigmoid(MLP(avgpool(x)) + MLP(maxpool(x)))
        out = x * ca
        sa = sigmoid(Conv2d(cat([max_c(out), mean_c(out)])))
        y = out * sa + x

    Args:
        x: 输入特征图 [B, C, H, W]
        ca_fc1_weight: 通道注意力第一层 1x1 卷积权重 [C // reduction, C, 1, 1]（无 bias）
        ca_fc2_weight: 通道注意力第二层 1x1 卷积权重 [C, C // reduction, 1, 1]（无 bias）
        sa_conv_weight: 空间注意力 2->1 卷积权重 [1, 2, kernel_size, kernel_size]
        sa_conv_bias: 空间注意力 2->1 卷积偏置 [1]
        channel: 输入通道数，保留用于接口兼容，实际通道数取自 x.shape[1]
        reduction: 通道注意力 MLP 的瓶颈降维比
        kernel_size: 空间注意力卷积核大小

    Returns:
        注意力细化后的特征图 [B, C, H, W]
    """
    residual = x

    # ---------- 通道注意力 ----------
    avg_p = F.adaptive_avg_pool2d(x, 1)
    max_p = F.adaptive_max_pool2d(x, 1)

    def shared_mlp(v):
        v = F.conv2d(v, ca_fc1_weight)
        v = F.relu(v)
        return F.conv2d(v, ca_fc2_weight)

    channel_att = torch.sigmoid(shared_mlp(avg_p) + shared_mlp(max_p))
    out = x * channel_att

    # ---------- 空间注意力 ----------
    avg_out = torch.mean(out, dim=1, keepdim=True)
    max_out = torch.max(out, dim=1, keepdim=True)[0]
    spatial_in = torch.cat([max_out, avg_out], dim=1)
    spatial_att = torch.sigmoid(
        F.conv2d(spatial_in, sa_conv_weight, sa_conv_bias, padding=kernel_size // 2)
    )
    out = out * spatial_att

    # ---------- 残差 ----------
    return out + residual
