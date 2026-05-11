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
from typing import List, Optional

"""
GroupedMatmul 算子 Torch Golden 参考实现

分组矩阵乘法算子，每组矩阵乘的维度大小可以不同
公式: y_i = x_i @ weight_i + bias_i (for each group i)
"""
def grouped_matmul(
    x: List[torch.Tensor],
    weight: List[torch.Tensor],
    bias: Optional[List[torch.Tensor]] = None,
    split_item: int = 0,
    transpose_weight: bool = False
) -> List[torch.Tensor]:
    """
    分组矩阵乘法算子

    对每个分组执行独立的矩阵乘法：y_i = x_i @ weight_i + bias_i

    Args:
        x: 输入矩阵TensorList，每个tensor shape为[m_i, k_i]
        weight: 权重矩阵TensorList，每个tensor shape为[k_i, n_i]（transpose_weight=false）
               或[n_i, k_i]（transpose_weight=true）
        bias: 偏置TensorList（可选），每个tensor shape为[n_i]
        split_item: 输出切分模式
                   - 0/1: 输出多tensor（每组独立），返回TensorList
                   - 2/3: 输出单tensor（结果连续存放），返回合并后的单tensor
        transpose_weight: 是否转置权重
                         - false: weight shape为[k_i, n_i]，matmul为x[m,k] @ weight[k,n]
                         - true: weight shape为[n_i, k_i]，matmul为x[m,k] @ weight[n,k]^T

    Returns:
        输出TensorList（split_item=0/1）或单tensor（split_item=2/3）
        每组输出shape为[m_i, n_i]
    """
    num_groups = len(weight)
    results = []

    for i in range(num_groups):
        x_i = x[i].float()  # [m_i, k_i]
        weight_i = weight[i].float()

        if transpose_weight:
            # weight shape: [n_i, k_i]
            # 需要转置: [n_i, k_i]^T = [k_i, n_i]
            # matmul: [m_i, k_i] @ [k_i, n_i] = [m_i, n_i]
            y_i = torch.matmul(x_i, weight_i.transpose(-2, -1))
        else:
            # weight shape: [k_i, n_i]
            # matmul: [m_i, k_i] @ [k_i, n_i] = [m_i, n_i]
            y_i = torch.matmul(x_i, weight_i)

        # 加偏置（可选）
        if bias is not None and bias[i] is not None:
            bias_i = bias[i].float()  # [n_i]
            y_i = y_i + bias_i.unsqueeze(0)  # broadcast to [m_i, n_i]

        # 转换回输入dtype
        y_i = y_i.to(x[i].dtype)
        results.append(y_i)

    # 根据 split_item 决定输出格式
    if split_item in [0, 1]:
        # 输出多tensor（TensorList）
        return results
    else:
        # split_item in [2, 3]: 输出单tensor（连续存放）
        # 将所有结果沿 M 轴合并
        return [torch.cat(results, dim=0)]