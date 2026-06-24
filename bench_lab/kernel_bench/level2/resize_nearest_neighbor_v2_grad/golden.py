#!/usr/bin/env python3
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
ResizeNearestNeighborV2Grad 算子 Torch Golden 参考实现。

基于 PyTorch aten 接口：
  - half_pixel_centers=false: torch.ops.aten.upsample_nearest2d_backward
  - half_pixel_centers=true : torch.ops.aten._upsample_nearest_exact2d_backward

说明：
  cases.yaml 中 size 输入使用占位符 value_range=[0,0]，真实原始输入尺寸通过
  attrs['input_size'] 给出。get_input 与本 golden 函数均据此构造正确的 size tensor。
"""

import torch
from typing import List


def _make_size_tensor(size_or_input_size):
    """把占位 size 或 attrs 中的 input_size 转换为 INT32 [H, W] tensor。"""
    if size_or_input_size is None:
        return None
    if isinstance(size_or_input_size, torch.Tensor):
        # 若传入的是占位 tensor（元素全 0 或长度不对），上游调用方应已替换为真实值。
        return size_or_input_size.to(torch.int32)
    return torch.tensor(size_or_input_size, dtype=torch.int32)


def resize_nearest_neighbor_v2_grad(
    grads: torch.Tensor,
    size: torch.Tensor,
    half_pixel_centers: bool = False,
    scales: List[float] = None,
) -> torch.Tensor:
    """
    ResizeNearestNeighborV2Grad 算子 Golden 参考实现。

    说明：本实现仅覆盖 PyTorch aten 最近邻反向接口语义，即 align_corners=false。
          函数签名中不再提供 align_corners 参数。

    Args:
        grads: 正向输出梯度，4D NCHW。
        size: 原始输入图像尺寸，INT32 1D Tensor [H_in, W_in]；必须由调用方提供有效值，
              不能为占位 [0,0]；若缺失有效 size 且 scales 无效，将显式报错而非静默兜底。
        half_pixel_centers: 是否使用半像素中心。
        scales: 空间尺寸乘数 [scaleH, scaleW]，默认 [0.0, 0.0]。

    Returns:
        y: 输入端梯度，4D NCHW，dtype 与 grads 一致。
    """
    if grads.dim() != 4:
        raise ValueError(f"ResizeNearestNeighborV2Grad supports 4D input, got {grads.dim()}D")

    if scales is None:
        scales = [0.0, 0.0]

    orig_dtype = grads.dtype
    # 低精度累加容易溢出，提升到 fp32 计算
    if orig_dtype in (torch.float16, torch.bfloat16):
        grads = grads.float()

    n, c, h_out, w_out = grads.shape
    size = _make_size_tensor(size)
    if size is not None and size.numel() == 2 and not (size == 0).all():
        h_in, w_in = int(size[0].item()), int(size[1].item())
    else:
        # 没有有效 size 时，只能依赖有效 scales
        scales_h, scales_w = float(scales[0]), float(scales[1])
        if scales_h > 0 and scales_w > 0:
            h_in = int(round(h_out * scales_h))
            w_in = int(round(w_out * scales_w))
        else:
            raise ValueError(
                "ResizeNearestNeighborV2Grad: cannot determine original input size. "
                "Either 'size' must be a valid [H, W] tensor, or 'scales' must be positive."
            )

    input_size_4d = [n, c, h_in, w_in]
    output_size = [h_out, w_out]
    scales_h, scales_w = float(scales[0]), float(scales[1])

    # 当输入输出尺寸相同时直接返回
    if h_out == h_in and w_out == w_in:
        return grads.to(orig_dtype)

    if half_pixel_centers:
        y = torch.ops.aten._upsample_nearest_exact2d_backward(
            grads, output_size, input_size_4d, scales_h=scales_h, scales_w=scales_w
        )
    else:
        y = torch.ops.aten.upsample_nearest2d_backward(
            grads, output_size, input_size_4d, scales_h=scales_h, scales_w=scales_w
        )

    return y.to(orig_dtype)


def get_input(
    grads: torch.Tensor,
    size: torch.Tensor,
    half_pixel_centers: bool = False,
    scales: List[float] = None,
    input_size: List[int] = None,
    **kwargs,
):
    """
    框架输入预处理函数。

    cases.yaml 中 size 使用占位符生成，这里用 attrs['input_size'] 替换为真实尺寸。
    返回顺序与 golden 函数签名一致：grads, size, ...

    Raises:
        ValueError: 当 attrs 未正确转发 'input_size'，或 input_size 不是有效的 [H, W] 时。
    """
    if input_size is None:
        raise ValueError(
            "ResizeNearestNeighborV2Grad.get_input: 'input_size' is required "
            "and must be forwarded from case attrs. The placeholder 'size' tensor "
            "alone cannot determine the original input spatial shape."
        )
    size_tensor = torch.tensor(input_size, dtype=torch.int32)
    if size_tensor.numel() != 2:
        raise ValueError(
            f"ResizeNearestNeighborV2Grad.get_input: 'input_size' must have exactly 2 elements [H, W], "
            f"got {size_tensor.numel()} elements: {input_size}"
        )
    return grads, size_tensor
