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
AddRmsNormDynamicQuant 算子 Torch Golden 参考实现

Add、RMSNorm 和动态量化的融合
公式：y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)
"""
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
    dst_type: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add、RMSNorm 和动态量化的融合

    公式：y, xOut, scaleOut = quantize(rmsnorm(x1 + x2) * gamma)

    Args:
        x1: 第 1 个输入张量
        x2: 第 2 个输入张量
        gamma: 缩放参数
        epsilon: epsilon 值
        dst_type: 目标数据类型 (0:DT_INT8, 1:DT_INT4)

    Returns:
        y: 量化后的输出张量
        xOut: Add 结果，x1 + x2
        scaleOut: 量化使用的 scale 值
    """

    out_dtype = x1.dtype

    # Promote bf16/fp16 inputs to fp32 for golden computation.
    # This aligns the golden's scale calculation with the NPU's native
    # precision, avoiding fp64-vs-bf16 quantization rounding gaps.
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    gamma = gamma.to(torch.float32)

    # Add 操作
    xOut = x1 + x2

    # RMSNorm
    variance = xOut.pow(2).mean(-1, keepdim=True)
    rms = torch.sqrt(variance + epsilon)
    normalized = xOut / rms
    y_norm = normalized * gamma

    # 动态量化
    if dst_type == 0:  # INT8
        scale = (127.0 / y_norm.abs().max()).to(torch.float32)
        y = torch.clamp((y_norm * scale.item()).round(), -128, 127).to(torch.int8)
    else:  # INT4 (存储为 int8，值范围 [-8, 7])
        scale = (7.0 / y_norm.abs().max()).to(torch.float32)
        y = torch.clamp((y_norm * scale.item()).round(), -8, 7).to(torch.int8)

    return y, xOut.to(out_dtype), scale
