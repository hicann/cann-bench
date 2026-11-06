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
Triplet Attention 算子 Torch Golden 参考实现

三个分支分别在 (C,H)、(C,W)、(H,W) 维度上做 Z-Pool（通道维 mean/max 拼接）与
k x k 卷积，BatchNorm 使用训练态（batch 统计，momentum=0.1，eps=1e-5），
经 ReLU 与 sigmoid 得到门控乘回分支输入，最后三分支求平均。

源实现全程保持输入 dtype（内部不升精度）。源 forward 的 gate_weights 为嵌套
结构，本 golden 将其拆成 12 个独立张量输入（每分支 conv 权重/偏置 + BN
仿射权重/偏置）。
"""


def triplet_attention(
    x: torch.Tensor,
    conv_ch_weight: torch.Tensor,
    conv_ch_bias: torch.Tensor,
    bn_ch_weight: torch.Tensor,
    bn_ch_bias: torch.Tensor,
    conv_cw_weight: torch.Tensor,
    conv_cw_bias: torch.Tensor,
    bn_cw_weight: torch.Tensor,
    bn_cw_bias: torch.Tensor,
    conv_hw_weight: torch.Tensor,
    conv_hw_bias: torch.Tensor,
    bn_hw_weight: torch.Tensor,
    bn_hw_bias: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Triplet Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_ch_weight: (C,H) 分支卷积权重 [1, 2, kernel_size, kernel_size]
        conv_ch_bias: (C,H) 分支卷积偏置 [1]
        bn_ch_weight: (C,H) 分支 BatchNorm 仿射权重 [1]
        bn_ch_bias: (C,H) 分支 BatchNorm 仿射偏置 [1]
        conv_cw_weight: (C,W) 分支卷积权重 [1, 2, kernel_size, kernel_size]
        conv_cw_bias: (C,W) 分支卷积偏置 [1]
        bn_cw_weight: (C,W) 分支 BatchNorm 仿射权重 [1]
        bn_cw_bias: (C,W) 分支 BatchNorm 仿射偏置 [1]
        conv_hw_weight: (H,W) 分支卷积权重 [1, 2, kernel_size, kernel_size]
        conv_hw_bias: (H,W) 分支卷积偏置 [1]
        bn_hw_weight: (H,W) 分支 BatchNorm 仿射权重 [1]
        bn_hw_bias: (H,W) 分支 BatchNorm 仿射偏置 [1]
        kernel_size: 卷积核大小（奇数），padding = (kernel_size - 1) // 2

    Returns:
        输出特征图 [B, C, H, W]
    """
    padding = (kernel_size - 1) // 2

    def gate_fn(t, conv_w, conv_b, bn_w, bn_b, permute=None, permute_back=None):
        if permute is not None:
            t = t.permute(permute)
        z = torch.cat([t.mean(dim=1, keepdim=True),
                       t.max(dim=1, keepdim=True)[0]], dim=1)
        y = F.conv2d(z, conv_w, conv_b, padding=padding)
        y = F.batch_norm(
            y, torch.zeros_like(bn_w), torch.ones_like(bn_w),
            bn_w, bn_b, training=True, momentum=0.1, eps=1e-5
        )
        y = F.relu(y)
        attn = torch.sigmoid(y)
        out = t * attn
        if permute_back is not None:
            out = out.permute(permute_back)
        return out

    x_ch = gate_fn(
        x, conv_ch_weight, conv_ch_bias, bn_ch_weight, bn_ch_bias,
        permute=(0, 3, 1, 2), permute_back=(0, 2, 3, 1)
    )
    x_cw = gate_fn(
        x, conv_cw_weight, conv_cw_bias, bn_cw_weight, bn_cw_bias,
        permute=(0, 2, 1, 3), permute_back=(0, 2, 1, 3)
    )
    x_hw = gate_fn(x, conv_hw_weight, conv_hw_bias, bn_hw_weight, bn_hw_bias)

    return (x_ch + x_cw + x_hw) / 3.0
