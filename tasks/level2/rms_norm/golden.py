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
RmsNorm 算子 Torch Golden 参考实现

计算 RMS (均方根) 归一化

公式:
    y = x / sqrt(mean(x^2) + eps) * gamma

参考 PyTorch API: torch.nn.functional.rms_norm
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html

参考论文: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467

Parameters:
    - x: (..., D) 输入张量，最后一维为归一化维度
    - gamma: (D,) 缩放参数
    - epsilon: float, 默认 1e-6 - 数值稳定性参数
"""


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算 RMS (均方根) 归一化

    Args:
        x: 输入张量，shape (..., D)
           最后一维 D 为归一化维度
        gamma: 缩放参数，shape (D,)
               与输入最后一维大小相同
        epsilon: 数值稳定性参数，防止除零
                 默认值 1e-6

    Returns:
        RMS 归一化后的张量，shape 与输入相同

    Examples:
        >>> x = torch.randn(32, 128, 4096)
        >>> gamma = torch.ones(4096)
        >>> y = rms_norm(x, gamma, epsilon=1e-6)
    """
    # 直接调用 PyTorch 原生 RMSNorm 实现；fp16/bf16 输入由 F.rms_norm 内部
    # 自动以 fp32 累加，避免 |x|>256 时 x^2 上溢。
    return torch.nn.functional.rms_norm(
        x, normalized_shape=gamma.shape, weight=gamma, eps=epsilon
    )
