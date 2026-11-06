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
CoordAtt 算子 Torch Golden 参考实现

Coordinate Attention 前向：分别沿 H、W 方向做全局池化，拼接后经 1x1 卷积 +
BatchNorm（batch 统计）+ h_swish 得到中间特征，再 split 成 H/W 两路各经 1x1
卷积 + sigmoid 得到坐标注意力权重，与输入逐元素相乘。

公式:
    x_h = adaptive_avg_pool2d(x, (None, 1))
    x_w = adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
    y = h_swish(BN(Conv1x1(cat([x_h, x_w], dim=2))))
    x_h, x_w = split(y, [H, W], dim=2)
    a_h = sigmoid(Conv1x1(x_h))
    a_w = sigmoid(Conv1x1(x_w.permute(0, 1, 3, 2)))
    y = x * a_w * a_h
"""


def _conv1x1(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """1x1 卷积的显式 einsum 实现，与源参考实现一致。"""
    weight = weight.squeeze(-1).squeeze(-1)
    out = torch.einsum("oc,bchw->bohw", weight, x)
    return out + bias.view(1, -1, 1, 1)


def coord_att(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    conv_h_weight: torch.Tensor,
    conv_h_bias: torch.Tensor,
    conv_w_weight: torch.Tensor,
    conv_w_bias: torch.Tensor,
    inp: int,
    oup: int,
    reduction: int = 32,
) -> torch.Tensor:
    """
    Coordinate Attention 前向

    公式:
        x_h = adaptive_avg_pool2d(x, (None, 1))
        x_w = adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        y = h_swish(BN(Conv1x1(cat([x_h, x_w], dim=2))))
        x_h, x_w = split(y, [H, W], dim=2)
        a_h = sigmoid(Conv1x1(x_h))
        a_w = sigmoid(Conv1x1(x_w.permute(0, 1, 3, 2)))
        y = x * a_w * a_h

    Args:
        x: 输入特征图 [B, C, H, W]
        conv1_weight: 融合支路 1x1 卷积权重 [mip, C, 1, 1]
        conv1_bias: 融合支路 1x1 卷积偏置 [mip]
        bn_weight: BatchNorm 缩放参数 [mip]
        bn_bias: BatchNorm 偏移参数 [mip]
        conv_h_weight: H 方向注意力 1x1 卷积权重 [C, mip, 1, 1]
        conv_h_bias: H 方向注意力 1x1 卷积偏置 [C]
        conv_w_weight: W 方向注意力 1x1 卷积权重 [C, mip, 1, 1]
        conv_w_bias: W 方向注意力 1x1 卷积偏置 [C]
        inp: 输入通道数，保留用于接口兼容，实际通道数由 x.shape[1] 推导
        oup: 输出通道数，保留用于接口兼容，不参与计算
        reduction: 中间通道降维比，mip = max(8, C // reduction)

    Returns:
        坐标注意力加权后的特征图 [B, C, H, W]
    """
    identity = x
    b, c, h, w = x.shape

    x_h = F.adaptive_avg_pool2d(x, (None, 1))
    x_w = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)

    y = torch.cat([x_h, x_w], dim=2)
    y = _conv1x1(y, conv1_weight, conv1_bias)
    y = F.batch_norm(
        y,
        torch.zeros_like(bn_weight),
        torch.ones_like(bn_weight),
        bn_weight,
        bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
    )
    y = y * (F.relu6(y + 3) / 6)

    x_h, x_w = torch.split(y, [h, w], dim=2)
    x_w = x_w.permute(0, 1, 3, 2)

    a_h = torch.sigmoid(_conv1x1(x_h, conv_h_weight, conv_h_bias))
    a_w = torch.sigmoid(_conv1x1(x_w, conv_w_weight, conv_w_bias))

    return identity * a_w * a_h
