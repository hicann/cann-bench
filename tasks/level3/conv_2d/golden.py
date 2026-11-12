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

"""
Conv2D算子Torch Golden参考实现

计算2D卷积
公式: y = CONV(x, filter) + bias
"""


def conv_2d(
    x: torch.Tensor,
    filter: torch.Tensor,
    bias: torch.Tensor,
    strides: list,
    pads: list,
    dilations: list = [1, 1],
) -> torch.Tensor:
    """
    计算2D卷积

    公式: y = CONV(x, filter) + bias

    Args:
        x: 输入特征图
        filter: 卷积核
        bias: 偏置
        strides: 步长
        pads: 填充
        dilations: 膨胀率

    Returns:
        输出特征图
    """

    # pads 格式: [pad_top, pad_bottom, pad_left, pad_right]
    # PyTorch conv2d padding 格式: (left, right, top, bottom) 或 (pad_h, pad_w) 对称模式
    # 检查是否对称填充
    if pads[0] == pads[1] and pads[2] == pads[3]:
        # 对称模式: (pad_height, pad_width)
        padding = (pads[0], pads[2])
    else:
        # 非对称模式: (left, right, top, bottom)
        x = torch.nn.functional.pad(x, (pads[2], pads[3], pads[0], pads[1]))
        padding = 0
    stride = (strides[0], strides[1])
    dilation = (dilations[0], dilations[1])

    y = torch.nn.functional.conv2d(
        x, filter, bias, stride=stride, padding=padding, dilation=dilation
    )
    return y
