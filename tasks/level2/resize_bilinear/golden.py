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
from typing import List, Optional

"""
ResizeBilinear 算子 Torch Golden 参考实现

使用双线性插值调整 4D 图像 (N, C, H, W) 的空间维度大小
公式: y = resize_bilinear(x, size)

参考 PyTorch API: torch.nn.functional.interpolate (mode='bilinear')
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
"""


def resize_bilinear(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    使用双线性插值调整图像大小（仅 4D 输入）。

    Args:
        x: 输入张量，形状为 (N, C, H, W)
        output_size: 输出尺寸 [output_height, output_width]，与 scale_factor 互斥
        align_corners: 是否对齐角点
        scale_factor: 缩放因子 [scale_height, scale_width]，与 output_size 互斥

    Returns:
        输出张量 (N, C, H_out, W_out)，dtype 与 x 一致
    """
    if x.dim() != 4:
        raise ValueError(f"ResizeBilinear requires 4D input (N, C, H, W), got {x.dim()}D")
    return torch.nn.functional.interpolate(
        x,
        size=output_size,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=align_corners,
    )
