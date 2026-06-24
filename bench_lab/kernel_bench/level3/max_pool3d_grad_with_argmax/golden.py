#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
MaxPool3dGradWithArgmax 算子 Torch Golden 参考实现。

基于 PyTorch aten 接口：torch.ops.aten.max_pool3d_with_indices_backward

说明：
- 框架会随机生成 argmax，无法直接用于 max_pool3d_with_indices_backward。
- 通过 get_input() 钩子重新运行正向 maxpool 生成合法的 grad 与 argmax 索引，
  确保 golden 与 NPU 使用同一套有效索引。
"""

import torch
import torch.nn.functional as F
from typing import List


def get_input(
    x: torch.Tensor,
    grad: torch.Tensor,
    argmax: torch.Tensor,
    ksize: List[int],
    strides: List[int],
    pads: List[int],
    dilation: List[int] = None,
    ceil_mode: bool = False,
    **kwargs,
):
    """
    根据正向 maxpool 生成合法的 grad 与 argmax。

    Args:
        x: 正向输入（框架生成，可能为 fp16/bf16/fp32）。
        grad: 框架生成的占位梯度，会被替换。
        argmax: 框架生成的占位索引，会被替换。
        ksize, strides, pads, dilation, ceil_mode: 池化属性。

    Returns:
        (x, grad, argmax) 三元组，其中 grad/argmax 的 shape 与正向输出一致，
        argmax 为合法的最大值索引。
    """
    if dilation is None:
        dilation = [1, 1, 1]

    # CPU 上 fp16/bf16 的 max_pool3d 可能不支持，使用 fp32 计算索引
    x_for_indices = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    with torch.no_grad():
        _, indices = F.max_pool3d(
            x_for_indices,
            kernel_size=ksize,
            stride=strides,
            padding=pads,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )

    # 生成与正向输出同 shape 的随机梯度
    grad = torch.randn_like(indices, dtype=x.dtype)
    return x, grad, indices


def max_pool3d_grad_with_argmax(
    x: torch.Tensor,
    grad: torch.Tensor,
    argmax: torch.Tensor,
    ksize: List[int],
    strides: List[int],
    pads: List[int],
    dilation: List[int] = None,
    ceil_mode: bool = False,
) -> torch.Tensor:
    """
    MaxPool3dGradWithArgmax 算子 Golden 参考实现。

    Args:
        x: 正向输入，5D NCDHW。
        grad: 正向输出的梯度，shape 与正向输出一致。
        argmax: 正向输入中最大元素的索引，shape 与 grad 一致。
        ksize: 池化窗口大小 [kD, kH, kW]。
        strides: 池化步长 [sD, sH, sW]。
        pads: 在 D、H、W 方向上的补零层数 [pD, pH, pW]。
        dilation: 窗口内元素步幅，默认 [1, 1, 1]。
        ceil_mode: 是否向上取整计算输出形状。

    Returns:
        y: 输入 x 的梯度，shape 与 x 一致。
    """
    if dilation is None:
        dilation = [1, 1, 1]

    # 低精度累加容易溢出/精度损失，参考官方 golden 提升到 fp32 计算再 cast 回输入 dtype
    orig_dtype = x.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
        grad = grad.float()

    y = torch.ops.aten.max_pool3d_with_indices_backward(
        grad,
        x,
        kernel_size=ksize,
        stride=strides,
        padding=pads,
        dilation=dilation,
        ceil_mode=ceil_mode,
        indices=argmax,
    )

    if orig_dtype in (torch.float16, torch.bfloat16):
        return y.to(orig_dtype)
    return y
