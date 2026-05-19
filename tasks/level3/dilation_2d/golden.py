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
Dilation2D 算子 Torch Golden 参考实现

2D形态学膨胀操作，对每个位置在膨胀窗口内取 input + filter 的最大值
公式: y[b, y, x, c] = max_{dy,dx} (x[b, y*stride_h + rate_h*dy, x*stride_w + rate_w*dx, c] + filter[dy, dx, c])
"""
def dilation_2d(
    x: torch.Tensor, filter: torch.Tensor, strides: list, rates: list,
    padding_mode: str = 'SAME', pads: list = [0, 0, 0, 0],
    ceil_mode: bool = False, data_format: str = 'NHWC'
) -> torch.Tensor:
    """
    2D形态学膨胀操作，对每个位置在膨胀窗口内取 input + filter 的最大值

    公式: y[b, y, x, c] = max_{dy,dx} (x[b, y*stride_h + rate_h*dy, x*stride_w + rate_w*dx, c] + filter[dy, dx, c])

    Args:
        x: 输入图像，shape 为 [batch, height, width, depth] (NHWC) 或 [batch, depth, height, width] (NCHW)
        filter: 结构元素/卷积核，shape 为 [filter_h, filter_w, depth]
        strides: 步长 [1, stride_h, stride_w, 1]，首尾固定为1
        rates: 膨胀率 [1, rate_h, rate_w, 1]，首尾固定为1
        padding_mode: 填充模式：'SAME' 或 'VALID'
        pads: 填充值 [pad_top, pad_bottom, pad_left, pad_right]
        ceil_mode: 是否向上取整计算输出尺寸
        data_format: 数据格式，'NHWC' 或 'NCHW'

    Returns:
        膨胀后的图像
    """

    if data_format == 'NHWC':
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        filter = filter.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

    batch, channels, in_h, in_w = x.shape
    filter_h, filter_w = filter.shape[1], filter.shape[2]
    stride_h, stride_w = strides[1], strides[2]
    rate_h, rate_w = rates[1], rates[2]

    effective_filter_h = (filter_h - 1) * rate_h + 1
    effective_filter_w = (filter_w - 1) * rate_w + 1

    if padding_mode == 'SAME':
        # SAME 模式的输出尺寸由 TF/TensorFlow 语义固定为 ceil(in/stride)，与 ceil_mode 参数无关；
        # 原代码在 SAME 下叠加 ceil_mode 会导致 (in - 1) % stride != 0 时输出多 1
        # (F205 P0)。ceil_mode 仅在 VALID / else 分支生效。
        out_h = (in_h + stride_h - 1) // stride_h
        out_w = (in_w + stride_w - 1) // stride_w
        pad_h = max((out_h - 1) * stride_h + effective_filter_h - in_h, 0)
        pad_w = max((out_w - 1) * stride_w + effective_filter_w - in_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # 形态学膨胀：padding 区域填充负无穷大，使 max 操作忽略这些位置
        x = torch.nn.functional.pad(x, [pad_left, pad_right, pad_top, pad_bottom], value=float('-inf'))
    elif padding_mode == 'VALID':
        # VALID 模式不需要 padding（或使用指定 pads）
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]], value=float('-inf'))
        out_h = (in_h - effective_filter_h + stride_h) // stride_h
        out_w = (in_w - effective_filter_w + stride_w) // stride_w
    else:
        if pads and sum(pads) > 0:
            x = torch.nn.functional.pad(x, [pads[2], pads[3], pads[0], pads[1]], value=float('-inf'))
        out_h = (x.shape[2] - effective_filter_h + stride_h) // stride_h
        out_w = (x.shape[3] - effective_filter_w + stride_w) // stride_w
        if ceil_mode:
            out_h = (x.shape[2] - effective_filter_h + stride_h - 1) // stride_h + 1
            out_w = (x.shape[3] - effective_filter_w + stride_w - 1) // stride_w + 1

    # 形态学膨胀: 使用 unfold 获取 patches
    # unfold 的 dilation 参数会自动按 rate 步长采样
    # kernel_size 使用实际的 filter 尺寸，而不是 effective 尺寸
    patches = torch.nn.functional.unfold(
        x,
        kernel_size=(filter_h, filter_w),
        dilation=(rate_h, rate_w),
        stride=(stride_h, stride_w)
    )

    # patches shape: [batch, channels * filter_h * filter_w, out_h * out_w]
    patches = patches.view(batch, channels, filter_h, filter_w, out_h, out_w)

    # 形态学膨胀：input_patch + filter，然后取最大值
    # filter shape: [C, H, W] -> expand to [batch, C, filter_h, filter_w, out_h, out_w]
    filter_expanded = filter.unsqueeze(0).unsqueeze(4).unsqueeze(5).expand(batch, -1, -1, -1, out_h, out_w)

    # 对每个 patch 位置，计算 input + filter，然后取最大值
    # patches shape: [batch, C, filter_h, filter_w, out_h, out_w]
    # 需要在 filter_h (dim=2) 和 filter_w (dim=3) 维度上取 max
    y = (patches + filter_expanded).amax(dim=(2, 3))  # [batch, C, out_h, out_w]

    if data_format == 'NHWC':
        y = y.permute(0, 2, 3, 1)  # NCHW -> NHWC

    return y