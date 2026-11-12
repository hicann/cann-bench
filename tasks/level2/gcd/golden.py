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
Gcd算子Torch Golden参考实现

计算两个整数的最大公约数
公式: y = gcd(x1, x2)
"""


def gcd(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    计算两个整数的最大公约数

    公式: y = gcd(x1, x2)

    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量

    Returns:
        输出张量，最大公约数（dtype 与输入一致）
    """
    # NPU 的 int16 GCD 并行实现存在内存/线程同步问题
    # 使用 int32 计算避免 NPU 和 CPU 结果不一致
    if x1.dtype == torch.int16:
        x1_32 = x1.int()
        x2_32 = x2.int()
        result = torch.gcd(x1_32, x2_32)
        return result.to(torch.int16)

    # NPU broadcast GCD 存在 bug，先显式广播再计算
    x1_bc, x2_bc = torch.broadcast_tensors(x1, x2)
    # CPU torch.gcd overflows on int32 min (-2**31); compute in int64
    # and cast back so results match the mathematically correct value.
    if x1_bc.dtype == torch.int32:
        return torch.gcd(x1_bc.long(), x2_bc.long()).to(torch.int32)
    y = torch.gcd(x1_bc, x2_bc)
    return y
