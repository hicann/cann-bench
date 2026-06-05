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
Add算子Torch Golden参考实现（评测流水线 fixture）

逐元素加法运算
公式: z = x + y
"""
def add(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    逐元素加法运算

    公式: z = x + y

    Args:
        x: 第1个输入张量
        y: 第2个输入张量（与x同shape）

    Returns:
        输出张量，逐元素加法结果
    """
    return torch.add(x, y)