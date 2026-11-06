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
Shuffle Attention 算子 Torch Golden 参考实现

将通道按 G 组切分，每组再一分为二：前半做通道注意力（全局平均池化 + 1x1 个
Parameter 仿射 + sigmoid），后半做空间注意力（GroupNorm + 1x1 个 Parameter
仿射 + sigmoid），拼接后按 2 组做 channel shuffle。

源实现全程保持输入 dtype（内部不升精度）。GroupNorm 的组数取 channel // (2 * G)，
与每个子特征的通道数一致。
"""


def shuffle_attention(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    channel_weight: torch.Tensor,
    channel_bias: torch.Tensor,
    spatial_weight: torch.Tensor,
    spatial_bias: torch.Tensor,
    channel: int,
    reduction: int = 16,
    G: int = 8,
) -> torch.Tensor:
    """
    Shuffle Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        gn_weight: GroupNorm 仿射权重 [C // (2 * G)]
        gn_bias: GroupNorm 仿射偏置 [C // (2 * G)]
        channel_weight: 通道注意力缩放参数 [1, C // (2 * G), 1, 1]
        channel_bias: 通道注意力偏移参数 [1, C // (2 * G), 1, 1]
        spatial_weight: 空间注意力缩放参数 [1, C // (2 * G), 1, 1]
        spatial_bias: 空间注意力偏移参数 [1, C // (2 * G), 1, 1]
        channel: 通道数属性（接口兼容，实际通道数取自 x.shape[1]）
        reduction: 接口兼容参数，源实现未使用
        G: 分组数，须整除 C 且 C // G 为偶数

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    gn_groups = channel // (2 * G)

    # 分组：每组的通道再一分为二
    x = x.view(b * G, -1, h, w)
    x_0, x_1 = x.chunk(2, dim=1)

    # ---------- 通道注意力分支 ----------
    avg_pool = F.adaptive_avg_pool2d(x_0, 1)
    x_channel = channel_weight * avg_pool + channel_bias
    x_channel = x_0 * torch.sigmoid(x_channel)

    # ---------- 空间注意力分支 ----------
    x_spatial = F.group_norm(x_1, gn_groups, gn_weight, gn_bias, eps=1e-5)
    x_spatial = spatial_weight * x_spatial + spatial_bias
    x_spatial = x_1 * torch.sigmoid(x_spatial)

    # 拼接并按 2 组做 channel shuffle
    out = torch.cat([x_channel, x_spatial], dim=1)
    out = out.contiguous().view(b, -1, h, w)
    out = out.reshape(b, 2, -1, h, w)
    out = out.permute(0, 2, 1, 3, 4)
    return out.reshape(b, -1, h, w)
