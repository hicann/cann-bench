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
Conv3DBackpropFilter算子Torch Golden参考实现

Conv3D的filter梯度
公式: y = conv3d_filter_grad(x, grad, filter_size)
"""
def conv3_d_backprop_filter(
    x: torch.Tensor, grad: torch.Tensor, strides: list, pads: list, dilations: list, groups: int = 1, filter_size: list = None
) -> torch.Tensor:
    """
    Conv3D的filter梯度

    公式: y = conv3d_filter_grad(x, grad, filter_size)

    Args:
        x: 输入特征图，shape为[N, C_in, D, H, W]
        grad: 输出梯度，shape为[N, C_out, D_out, H_out, W_out]
        strides: 步长，3元素 [stride_d, stride_h, stride_w]
        pads: 填充，6元素 [D_front, D_back, H_top, H_bottom, W_left, W_right]，对称时取front/top/left
        dilations: 膨胀率，3元素 [dilation_d, dilation_h, dilation_w]
        groups: 分组数
        filter_size: filter的shape [C_out, C_in/groups, K_d, K_h, K_w]

    Returns:
        filter梯度，shape与filter_size相同
    """

    # pads 是 6 元素格式，对称 padding 时取 (D_front, H_top, W_left)
    # 即 pads[0], pads[2], pads[4]
    padding = (pads[0], pads[2], pads[4])
    stride = (strides[0], strides[1], strides[2])
    dilation = (dilations[0], dilations[1], dilations[2])

    # 使用 torch.nn.grad.conv3d_weight 计算 filter 梯度
    y = F.grad.conv3d_weight(x, tuple(filter_size), grad, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y