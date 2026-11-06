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
SKAttention 算子 Torch Golden 参考实现

Selective Kernel Attention (SK Attention)：
    U_i = ReLU(BN_i(Conv2d(x, k_i, padding=k_i//2, groups=group)))   # K 个分支
    U   = sum_i U_i
    S   = mean(U, dim=(-1, -2))                                       # [B, C]
    Z   = fc(S)                                                       # [B, d]
    w_i = softmax_i(fc_i(Z))                                          # 沿分支维 softmax
    y   = sum_i (w_i * U_i)

说明：
- 源实现把 K 组 conv/BN 与 fc/fcs 缓存在 _cache 中；本 golden 用 List[Tensor]
  逐分支传入（TensorList 长度 = len(kernels)），数学上完全等价。
- BatchNorm 使用训练态批统计（源 Module 默认 training=True），
  momentum=0.1、eps=1e-5，与 nn.BatchNorm2d 默认值一致。
- 全程使用输入 dtype 计算，输出 dtype 与输入一致。
"""


def sk_attention(
    x: torch.Tensor,
    conv_w: List[torch.Tensor],
    conv_b: List[torch.Tensor],
    bn_w: List[torch.Tensor],
    bn_b: List[torch.Tensor],
    fc_w: torch.Tensor,
    fc_b: torch.Tensor,
    fcs_w: List[torch.Tensor],
    fcs_b: List[torch.Tensor],
    channel: int,
    kernels: list,
    reduction: int,
    group: int,
    L: int,
) -> torch.Tensor:
    """
    Selective Kernel Attention (SK Attention)

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_w: K 个卷积分支权重，第 i 个 [C, C//group, k_i, k_i]
        conv_b: K 个卷积分支偏置，每个 [C]
        bn_w/bn_b: K 个 BatchNorm 仿射权重/偏置，每个 [C]
        fc_w/fc_b: 降维全连接权重 [d, C] / 偏置 [d]，d = max(L, C//reduction)
        fcs_w/fcs_b: K 个升维全连接权重 [C, d] / 偏置 [C]
        channel: 输入通道数
        kernels: K 个卷积分支的核大小列表
        reduction: 全连接瓶颈通道缩减比
        group: 卷积分组数
        L: 瓶颈维度下界

    Returns:
        输出特征图 [B, C, H, W]，dtype 与 x 相同
    """
    b, c, h, w = x.shape
    if channel is None:
        channel = c
    d = max(L, channel // reduction)
    num_k = len(kernels)

    # ---------- Split ----------
    conv_outs = []
    for i in range(num_k):
        y = F.conv2d(x, conv_w[i], conv_b[i], padding=kernels[i] // 2, groups=group)
        y = F.batch_norm(
            y, None, None, bn_w[i], bn_b[i],
            training=True, momentum=0.1, eps=1e-5,
        )
        y = F.relu(y)
        conv_outs.append(y)
    feats = torch.stack(conv_outs, 0)  # (K, b, c, h, w)

    # ---------- Fuse ----------
    U = sum(conv_outs)  # (b, c, h, w)

    # ---------- Reduction ----------
    S = U.mean(-1).mean(-1)  # (b, c)
    Z = F.linear(S, fc_w, fc_b)  # (b, d)

    # ---------- Attention weights ----------
    weights = []
    for i in range(num_k):
        w = F.linear(Z, fcs_w[i], fcs_b[i]).view(b, c, 1, 1)  # (b, c, 1, 1)
        weights.append(w)
    attention_weights = torch.stack(weights, 0)  # (K, b, c, 1, 1)
    attention_weights = F.softmax(attention_weights, dim=0)  # 沿分支维 softmax

    # ---------- Fuse ----------
    V = (attention_weights * feats).sum(0)  # (b, c, h, w)
    return V
