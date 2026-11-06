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
GCModule 算子 Torch Golden 参考实现

GCNet Global Context 模块：
    attn    = conv1x1(x)                    # (B, 1, H, W) raw logits
    context = x_flat @ attn^T               # (B, C, 1, 1)
    y       = x + conv1x1(ReLU(LN(conv1x1(context))))

注意：源实现 __init__ 中定义了 softmax，但 context_modeling 从未调用，
本 golden 严格遵循源行为，不对 attn 做 softmax 归一化。
"""


def gc_module(
    x: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    t_w1: torch.Tensor,
    t_b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    t_w2: torch.Tensor,
    t_b2: torch.Tensor,
    channel: int,
    reduction: int,
) -> torch.Tensor:
    """
    GCNet Global Context 模块

    Args:
        x: 输入特征图 [B, C, H, W]
        conv1_w: 注意力 1x1 卷积权重 [1, C, 1, 1]
        conv1_b: 注意力 1x1 卷积偏置 [1]
        t_w1: Transform 第一层 1x1 卷积权重 [C//reduction, C, 1, 1]
        t_b1: Transform 第一层 1x1 卷积偏置 [C//reduction]
        ln_w: LayerNorm 仿射权重 [C//reduction, 1, 1]
        ln_b: LayerNorm 仿射偏置 [C//reduction, 1, 1]
        t_w2: Transform 第二层 1x1 卷积权重 [C, C//reduction, 1, 1]
        t_b2: Transform 第二层 1x1 卷积偏置 [C]
        channel: 输入通道数（须等于 C）
        reduction: bottleneck 缩减比

    Returns:
        输出特征图 [B, C, H, W]
    """
    batch, channels, height, width = x.shape

    attn = F.conv2d(x, conv1_w, conv1_b)
    b_att, _, h_att, w_att = attn.shape
    attn = attn.view(b_att, 1, h_att * w_att)

    x_flat = x.view(batch, channels, -1)
    context = torch.bmm(x_flat, attn.transpose(1, 2)).unsqueeze(-1)

    y = F.conv2d(context, t_w1, t_b1)
    y = F.layer_norm(y, y.shape[1:], ln_w, ln_b)
    y = F.relu(y, inplace=True)
    y = F.conv2d(y, t_w2, t_b2)

    return x + y
