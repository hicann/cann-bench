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
Sigmoid PyPTO selected-case Torch Golden 参考实现

计算输入张量的 Sigmoid 激活:
y = 1 / (1 + e^(-x))
"""
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量的 Sigmoid 激活。

    Args:
        x: float32 2D 输入张量

    Returns:
        Sigmoid 激活结果
    """

    return torch.sigmoid(x)
