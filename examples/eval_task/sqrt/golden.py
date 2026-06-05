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
Sqrt算子Torch Golden参考实现（评测流水线 fixture）

逐元素平方根运算
公式: y = sqrt(x)
"""
def sqrt(
    x: torch.Tensor
) -> torch.Tensor:
    """
    逐元素平方根运算

    公式: y = sqrt(x)

    Args:
        x: 输入张量（值应非负以保证数值稳定）

    Returns:
        输出张量，逐元素平方根结果
    """
    return torch.sqrt(x)