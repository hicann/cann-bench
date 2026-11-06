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
DAModule 算子 Torch Golden 参考实现

Dual Attention Module：位置注意力分支与通道注意力分支并行计算后逐元素相加。

    y_pos    = conv_pos(x).view(B, C, -1).permute(0, 2, 1)           # (B, N, C)
    attn_pos = softmax(bmm(y_pos.float(), y_pos.float()^T) * C^-0.5)
    out_pos  = bmm(attn_pos, y_pos.float()) -> (B, C, H, W) -> x.dtype

    y_ch     = conv_ch(x).view(B, C, -1)                             # (B, C, N)
    attn_ch  = softmax(bmm(y_ch.float(), y_ch.float()^T) * (H*W)^-0.5)
    out_ch   = bmm(attn_ch, y_ch.float()) -> (B, C, H, W) -> x.dtype

    y = out_pos + out_ch

dtype 策略与源实现一致：两个分支的注意力分数都先上转 fp32 计算，
结果转回输入 dtype 后再相加；所有输入输出使用同一 dtype。
"""


def da_module(
    x: torch.Tensor,
    conv_pos_w: torch.Tensor,
    conv_pos_b: torch.Tensor,
    conv_ch_w: torch.Tensor,
    conv_ch_b: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Dual Attention Module 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_pos_w: 位置注意力分支卷积权重 [C, C, K, K]
        conv_pos_b: 位置注意力分支卷积偏置 [C]
        conv_ch_w: 通道注意力分支卷积权重 [C, C, K, K]
        conv_ch_b: 通道注意力分支卷积偏置 [C]
        kernel_size: 卷积核大小，padding 取 kernel_size // 2

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    padding = kernel_size // 2

    # Position attention
    y_pos = F.conv2d(x, conv_pos_w, conv_pos_b, padding=padding).view(b, c, -1).permute(0, 2, 1)
    y_pos_f = y_pos.float()
    attn_pos = F.softmax(torch.bmm(y_pos_f, y_pos_f.transpose(1, 2)) * (c ** -0.5), dim=-1)
    out_pos = torch.bmm(attn_pos, y_pos_f).permute(0, 2, 1).view(b, c, h, w).to(x.dtype)

    # Channel attention
    y_ch = F.conv2d(x, conv_ch_w, conv_ch_b, padding=padding).view(b, c, -1)
    y_ch_f = y_ch.float()
    attn_ch = F.softmax(torch.bmm(y_ch_f, y_ch_f.transpose(1, 2)) * ((h * w) ** -0.5), dim=-1)
    out_ch = torch.bmm(attn_ch, y_ch_f).view(b, c, h, w).to(x.dtype)

    return out_pos + out_ch
