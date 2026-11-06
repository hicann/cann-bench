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
DoubleAttention 算子 Torch Golden 参考实现

Double Attention Network：用全局描述子做二阶特征交互。

    A  = convA(x).view(B, C, N)                       # C = c_m
    B  = softmax(convB(x).view(B, c_n, N), dim=-1)    # c_n = max(1, C // 8)
    V  = softmax(convV(x).view(B, c_n, N), dim=-1)
    G  = bmm(A, B^T)                                  # (B, C, c_n)
    Z  = bmm(G, V).view(B, C, H, W)
    y  = convZ(Z)

源实现 forward 的 a / v 外部参数实际未参与计算，本 golden 省略；
源实现全程使用输入 dtype 计算（softmax 也不做 fp32 上转），本 golden 保持一致。
b / c / h / w 任一为 0 时源实现直接返回输入，本 golden 同样提前返回。
"""


def double_attention(
    x: torch.Tensor,
    conv_a_w: torch.Tensor,
    conv_a_b: torch.Tensor,
    conv_b_w: torch.Tensor,
    conv_b_b: torch.Tensor,
    conv_v_w: torch.Tensor,
    conv_v_b: torch.Tensor,
    conv_z_w: torch.Tensor,
    conv_z_b: torch.Tensor,
) -> torch.Tensor:
    """
    Double Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_a_w: 汇聚卷积 A 权重 [C, C, 1, 1]
        conv_a_b: 汇聚卷积 A 偏置 [C]
        conv_b_w: 注意力卷积 B 权重 [c_n, C, 1, 1]
        conv_b_b: 注意力卷积 B 偏置 [c_n]
        conv_v_w: 注意力卷积 V 权重 [c_n, C, 1, 1]
        conv_v_b: 注意力卷积 V 偏置 [c_n]
        conv_z_w: 重建卷积 Z 权重 [C, C, 1, 1]
        conv_z_b: 重建卷积 Z 偏置 [C]

    Returns:
        输出特征图 [B, C, H, W]；若输入任一维为 0 则原样返回输入
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x

    c_m = c
    c_n = max(1, c // 8)

    a = F.conv2d(x, conv_a_w, conv_a_b).view(b, c_m, -1)
    b_mat = F.softmax(F.conv2d(x, conv_b_w, conv_b_b).view(b, c_n, -1), dim=-1)
    v = F.softmax(F.conv2d(x, conv_v_w, conv_v_b).view(b, c_n, -1), dim=-1)

    global_descriptors = torch.bmm(a, b_mat.permute(0, 2, 1))
    tmp_z = torch.bmm(global_descriptors, v)
    tmp_z = tmp_z.view(b, c_m, h, w)
    return F.conv2d(tmp_z, conv_z_w, conv_z_b)
