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
Col2Im 算子 Torch Golden 参考实现

滑窗列重排回图像（col2im / fold）：对每个输出像素，累加所有覆盖它的滑窗块贡献，
未被覆盖的位置补 0。
公式:
    y[n, c, h, w] = sum_{kh, kw} x[n, c * kH * kW + kh * kW + kw, l]
    l = h_col * Lw + w_col
    h = h_col * strideH - padH + kh * dilH
    w = w_col * strideW - padW + kw * dilW
"""


def col2im(
    x: torch.Tensor,
    output_size: list,
    kernel_size: list,
    dilation: list = None,
    padding: list = None,
    stride: list = None,
) -> torch.Tensor:
    """
    滑窗列重排回图像（col2im / fold）

    公式:
        y[n, c, h, w] = sum_{kh, kw} x[n, c * kH * kW + kh * kW + kw, l]
        l = h_col * Lw + w_col
        h = h_col * strideH - padH + kh * dilH
        w = w_col * strideW - padW + kw * dilW

    Args:
        x: 滑窗列张量 [N, C*kH*kW, L]
        output_size: 输出图像尺寸 [H_out, W_out]
        kernel_size: 滑窗核尺寸 [kH, kW]
        dilation: 膨胀系数 [dilH, dilW]
        padding: 填充大小 [padH, padW]
        stride: 滑窗步长 [strideH, strideW]

    Returns:
        输出张量 [N, C, H_out, W_out]
    """
    if dilation is None:
        dilation = [1, 1]
    if padding is None:
        padding = [0, 0]
    if stride is None:
        stride = [1, 1]
    return F.fold(
        x,
        tuple(output_size),
        tuple(kernel_size),
        dilation=tuple(dilation),
        padding=tuple(padding),
        stride=tuple(stride),
    )
