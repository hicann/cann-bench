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
from typing import List

"""
PSA 算子 Torch Golden 参考实现

Pyramid Squeeze Attention (PSA)：
    - SPC：把通道均分为 S 组，第 g 组用核大小 2(g+1)+1、padding g+1 的卷积
      （每组内 ci = channel // S 个通道独立卷积，相当于 depthwise 分组卷积）
    - SPA：每组经全局平均池化 + 两层 1x1 卷积（ReLU -> Sigmoid）得到通道权重，
      在组维度 softmax 后对各组特征加权求和

说明：
- 源实现把 S 组 conv / SE 权重缓存在 _cache 中；本 golden 用 List[Tensor]
  逐组传入（每个 TensorList 长度 = S），数学上完全等价。
- 全程使用输入 dtype 计算，输出 dtype 与输入一致。
"""


def psa(
    x: torch.Tensor,
    conv_w: List[torch.Tensor],
    conv_b: List[torch.Tensor],
    se_w1: List[torch.Tensor],
    se_w2: List[torch.Tensor],
    channel: int,
    reduction: int,
    S: int,
) -> torch.Tensor:
    """
    Pyramid Squeeze Attention (PSA)

    Args:
        x: 输入特征图 [B, C, H, W]，C = S * ci
        conv_w: S 组分组卷积权重，第 g 个为 [ci, ci, 2(g+1)+1, 2(g+1)+1]
        conv_b: S 组分组卷积偏置，每个 [ci]
        se_w1: S 组 SE 第一层 1x1 卷积权重（无 bias），每个 [ci//reduction, ci, 1, 1]
        se_w2: S 组 SE 第二层 1x1 卷积权重（无 bias），每个 [ci, ci//reduction, 1, 1]
        channel: 输入通道数参数（ci = channel // S）
        reduction: SE 通道缩减比
        S: 分组数，同时是各 TensorList 的长度

    Returns:
        输出特征图 [B, C, H, W]，dtype 与 x 相同
    """
    b, c, h, w = x.shape
    # 第 g 组的卷积核大小与 padding 由组号决定：k = 2*(g+1)+1, padding = g+1
    SPC_out = x.view(b, S, -1, h, w).clone()  # (b, S, ci, h, w)
    for idx in range(S):
        SPC_out[:, idx, :, :, :] = F.conv2d(
            SPC_out[:, idx, :, :, :], conv_w[idx], conv_b[idx], padding=idx + 1
        )

    # 每组 SE：全局平均池化 -> 1x1 conv -> ReLU -> 1x1 conv -> Sigmoid
    se_outs = []
    for idx in range(S):
        group = SPC_out[:, idx, :, :, :]          # (b, ci, h, w)
        pooled = F.adaptive_avg_pool2d(group, 1)  # (b, ci, 1, 1)
        t = F.conv2d(pooled, se_w1[idx])
        t = F.relu(t, inplace=True)
        t = F.conv2d(t, se_w2[idx])
        se_outs.append(torch.sigmoid(t))
    SE_out = torch.stack(se_outs, dim=1)          # (b, S, ci, 1, 1)
    SE_out = SE_out.expand_as(SPC_out)            # (b, S, ci, h, w)

    # 组维度 softmax + 加权求和
    softmax_out = F.softmax(SE_out, dim=1)
    PSA_out = SPC_out * softmax_out
    return PSA_out.view(b, -1, h, w)
