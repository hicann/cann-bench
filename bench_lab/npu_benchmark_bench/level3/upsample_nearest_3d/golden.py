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
UpsampleNearest3D 算子 Torch Golden 参考实现

最近邻 3D 上采样，按目标尺寸逐元素取最近邻源坐标。
公式: y[n, c, i, j, k] = input[n, c, floor(i * D_in / D_out), floor(j * H_in / H_out),
                              floor(k * W_in / W_out)]
"""


def upsample_nearest_3d(input: torch.Tensor, output_size: list) -> torch.Tensor:
    """
    最近邻 3D 上采样

    公式: y[n, c, i, j, k] = input[n, c, floor(i * D_in / D_out), floor(j * H_in / H_out),
                                  floor(k * W_in / W_out)]

    Args:
        input: 输入 5D 张量 [N, C, D_in, H_in, W_in]
        output_size: 目标输出尺寸 (D_out, H_out, W_out)

    Returns:
        输出张量 [N, C, D_out, H_out, W_out]
    """
    size = (int(output_size[0]), int(output_size[1]), int(output_size[2]))
    return F.interpolate(input, size=size, mode="nearest")
