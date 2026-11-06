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
GlobalFilter 算子 Torch Golden 参考实现

Global Filter Network：在频域对 token 做逐元素复乘滤波后逆变换回空间域。

    x_s    = x.view(B, H, W, C).float()                 # 内部上转 fp32
    X_f    = rfft2(x_s, dim=(1, 2), norm='ortho')       # (B, H, W//2+1, C) complex64
    W_c    = view_as_complex(weight)                    # weight: (H, W//2+1, C, 2) fp32
    Y_f    = X_f * W_c
    y      = irfft2(Y_f, s=(H, W), dim=(1, 2), norm='ortho').reshape(B, N, C).to(input_dtype)

源实现的滤波器参数始终以 float32 存储（与输入 dtype 无关），因此本 golden 的
weight 输入固定为 float32；其余计算在 fp32 复数域完成，最终转回输入 dtype。
dim 属性仅用于源实现的缓存键，不参与数学计算。
"""


def global_filter(
    x: torch.Tensor,
    weight: torch.Tensor,
    dim: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Global Filter 前向

    Args:
        x: 输入 token 序列 [B, N, C]，N = h * w
        weight: 频域滤波器实部/虚部堆叠 [h, w // 2 + 1, C, 2]，float32
        dim: 通道维大小（源实现仅用于缓存键，不参与计算）
        h: 空间网格高度
        w: 空间网格宽度

    Returns:
        输出 token 序列 [B, N, C]，dtype 与 x 一致
    """
    batch, n, c = x.shape
    if h * w != n:
        raise ValueError(f"GlobalFilter: h*w ({h}*{w}={h * w}) must equal N ({n})")

    dtype = x.dtype
    x = x.view(batch, h, w, c).float()
    x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
    weight_c = torch.view_as_complex(weight)
    x_fft = x_fft * weight_c
    x_ifft = torch.fft.irfft2(x_fft, s=(h, w), dim=(1, 2), norm="ortho")
    return x_ifft.reshape(batch, n, c).to(dtype)
