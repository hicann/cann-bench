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
    """权重量化批量矩阵乘法算子 —— 同时充当同精度参考 (bench b) 与 golden-npu-mock ST 候选。

    公式: y = x @ ANTIQUANT(weight) + bias，ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale。

    A16W8 weight-only 量化的标准约定 (与 torchao / Marlin 等 GPU 库一致): int8 权重反量化到
    **输出精度 T = x.dtype** (fp16/bf16)，再在 **fp32 累加器** 上做 matmul，输出舍回 T。这是一个
    正确 A16W8 kernel 在该输出精度下应有的误差下限 (库算子 meet-or-exceed 它)，供 checker 的
    小值域/相消同精度对照，使 |b - oracle| 不再恒为 0。数学真值 (fp64) 见 `_oracle`；oracle/bench
    golden 拆分的约定见 docs/guide/contributing.md。

    Args:
        x: 左输入矩阵，shape 为 [M, K]，dtype 为 float16/bfloat16
        weight: 右输入矩阵 (量化权重)，shape 为 [K, N]，dtype 为 int8
        antiquantScale: 反量化 scale 参数，shape 为 [N] 或 [1, N]
        antiquantOffset: 反量化 offset 参数 (可选)，shape 与 antiquantScale 相同
        bias: 偏置张量 (可选)，shape 为 [N] 或 [1, N]

    Returns:
        输出张量，shape 为 [M, N]，dtype 与 x 相同
    """
    # 反量化跟随输出精度 T：int8 权重升到 T 后 (weight + offset) * scale 在 T 上算，保留硬件同款舍入
    T = x.dtype
    weight_dequant = weight.to(T)
    scale = antiquantScale.to(T)
    if antiquantOffset is not None:
        weight_dequant = (weight_dequant + antiquantOffset.to(T)) * scale
    else:
        weight_dequant = weight_dequant * scale

    # fp32 累加器 (tensor-core 约定)：T 操作数升 fp32 相乘累加，不改变操作数已有的 T 舍入
    y = torch.matmul(x.float(), weight_dequant.float())
    if bias is not None:
        y = y + bias.float()
    return y.to(T)


def weight_quant_batch_matmul_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor = None,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Oracle: A16W8 的数学真值 (g)。唯一的近似是 int8 权重量化本身；反量化
    (weight + offset) * scale 与 matmul 全程跟随输入精度、不硬编码 .float()/.double() —— 在
    golden_precision=fp64_cpu 下 x 为 fp64，故整条在 fp64 计算，是精确反量化的 fp64 真值上界
    (不再把 fp64 下采成 fp32)。int8 weight 用 .to(x.dtype) 反量化跟随计算精度。
    """
    cdt = x.dtype
    weight_dq = weight.to(cdt)
    scale_c = antiquantScale.to(cdt)
    if antiquantOffset is not None:
        weight_dq = (weight_dq + antiquantOffset.to(cdt)) * scale_c
    else:
        weight_dq = weight_dq * scale_c
    y = torch.matmul(x, weight_dq)
    if bias is not None:
        y = y + bias.to(cdt)
    return y.to(x.dtype)
