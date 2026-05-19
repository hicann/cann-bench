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
WeightQuantBatchMatmul 算子 Torch Golden 参考实现

权重量化批量矩阵乘法算子
公式: y = x @ ANTIQUANT(weight) + bias
      ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
"""
def weight_quant_batch_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor = None,
    bias: torch.Tensor = None
) -> torch.Tensor:
    """
    权重量化批量矩阵乘法算子

    公式: y = x @ ANTIQUANT(weight) + bias
          ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale

    Args:
        x: 左输入矩阵，shape 为 [M, K]，dtype 为 float16/bfloat16
        weight: 右输入矩阵（量化权重），shape 为 [K, N]，dtype 为 int8
        antiquantScale: 反量化scale参数，shape 为 [N] 或 [1, N]
        antiquantOffset: 反量化offset参数（可选），shape 与 antiquantScale 相同
        bias: 偏置张量（可选），shape 为 [N] 或 [1, N]

    Returns:
        输出张量，shape 为 [M, N]，dtype 与 x 相同
    """

    # 反量化 weight: (weight + antiquantOffset) * antiquantScale
    # weight 是 int8，需要转换为浮点类型进行计算
    weight_float = weight.float()  # [K, N]

    # antiquantScale shape: [N] 或 [1, N]，需要 broadcast 到 [K, N]
    scale_float = antiquantScale.float()  # [N] 或 [1, N]

    # 计算 ANTIQUANT(weight)
    if antiquantOffset is not None:
        offset_float = antiquantOffset.float()  # [N] 或 [1, N]
        weight_dequant = (weight_float + offset_float) * scale_float
    else:
        weight_dequant = weight_float * scale_float

    # weight_dequant shape: [K, N]
    # x shape: [M, K]
    # matmul: [M, K] @ [K, N] = [M, N]

    # x 转换为浮点类型
    x_float = x.float()  # [M, K]

    # 矩阵乘法
    y_float = torch.matmul(x_float, weight_dequant)  # [M, N]

    # 加偏置（可选）
    if bias is not None:
        bias_float = bias.float()  # [N] 或 [1, N]
        y_float = y_float + bias_float

    # 转换回输入类型
    y = y_float.to(x.dtype)

    return y