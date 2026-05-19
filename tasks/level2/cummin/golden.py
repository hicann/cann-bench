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


def cummin(input: torch.Tensor, dim: int):
    """Cummin 算子 Torch Golden 参考实现 (P1 op, 对齐 torch.cummin).

    公式: values[i] = min(input[0], input[1], ..., input[i]) 沿指定轴
          indices[i] = argmin(input[0:i+1]) 沿指定轴

    Args:
        input: 输入张量
        dim: 计算累积最小值的轴

    Returns:
        torch.return_types.cummin (named tuple): (values, indices)
          • values: 同 input 形状/dtype 的累积最小值张量
          • indices: 同 input 形状的 int64 索引张量
    """
    return torch.cummin(input, dim=dim)
