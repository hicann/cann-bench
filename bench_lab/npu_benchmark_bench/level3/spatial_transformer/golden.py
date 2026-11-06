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
SpatialTransformer 算子 Torch Golden 参考实现

STN 仿射空间变换：用 2x3 仿射矩阵 theta 把输出网格映射到输入坐标，再做双线性采样。
公式: grid = affine_grid(theta, [B, C, H_out, W_out], align_corners)
      y = grid_sample(x, grid, align_corners)
"""


def spatial_transformer(
    x: torch.Tensor,
    theta: torch.Tensor,
    output_size: list,
    align_corners: bool = False,
    default_theta: list = None,
    use_default_theta: list = None,
) -> torch.Tensor:
    """
    STN 仿射空间变换（affine_grid + grid_sample）

    公式: grid = affine_grid(theta, [B, C, H_out, W_out], align_corners)
          y = grid_sample(x, grid, align_corners)

    Args:
        x: 输入特征图 [B, C, H, W]
        theta: 仿射变换矩阵 [B, 2, 3] 或 [2, 3]（[2, 3] 时广播到 batch）
        output_size: 目标输出尺寸 (out_h, out_w)，元素 <= 0 表示沿用输入尺寸
        align_corners: 角像素中心对齐标志
        default_theta: 默认仿射参数（6 个）
        use_default_theta: 6 个 0/1 标志，1 表示对应位置取 default_theta

    Returns:
        输出张量 [B, C, out_h, out_w]
    """
    b, c, h, w = x.shape
    if theta.dim() == 2:                          # [2, 3] -> [B, 2, 3]
        theta = theta.unsqueeze(0).expand(b, -1, -1).contiguous()

    if use_default_theta:
        mask = torch.tensor(use_default_theta, dtype=torch.float32,
                            device=theta.device)
        default = torch.tensor(default_theta, dtype=theta.dtype,
                               device=theta.device)
        theta_flat = torch.where(mask[None, :] > 0,
                                 default[None, :].expand(b, 6),
                                 theta.reshape(b, 6))
        theta = theta_flat.reshape(b, 2, 3)

    out_h = h if output_size[0] <= 0 else int(output_size[0])
    out_w = w if output_size[1] <= 0 else int(output_size[1])

    grid = F.affine_grid(theta, (b, c, out_h, out_w),
                         align_corners=align_corners)
    # aclnnGridSampler2D 要求 grid 与 input dtype 一致
    grid = grid.to(x.dtype)
    return F.grid_sample(x, grid, align_corners=align_corners)
