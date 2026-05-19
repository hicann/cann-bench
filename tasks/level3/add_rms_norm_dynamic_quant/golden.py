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

Add、RMSNorm 和 per-token 对称动态量化的融合，对齐 torch_npu.npu_add_rms_norm_dynamic_quant：
- scale 为 per-token 反量化系数 (max/127)，shape = x1.shape[:-1]，dtype = fp32
- 下游算子拿 scale 还原浮点：x_fp = y_int8 * scale
"""
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add、RMSNorm 和 per-token 对称动态量化的融合

    公式:
        xOut     = x1 + x2
        y_norm   = xOut / sqrt(mean(xOut^2) + eps) * gamma
        scaleOut = row_max(abs(y_norm)) / 127      # 反量化系数, shape = x1.shape[:-1]
        yOut     = round(y_norm / scaleOut)         # int8, shape = x1.shape

    Args:
        x1: 第 1 个输入张量
        x2: 第 2 个输入张量
        gamma: 缩放参数
        epsilon: epsilon 值

    Returns:
        y:        量化后的输出张量 (int8, shape = x1.shape)
        xOut:     Add 结果，x1 + x2
        scaleOut: per-token 反量化系数 (float32, shape = x1.shape[:-1])
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

    # 动态量化 (per-token INT8)
    # scale = row_max(abs(y_norm)) / 127 — 反量化系数, 与 NPU API 一致。
    # clamp(min=1e-12) 防止全零输入触发 0/0 = NaN。
    abs_max = y_norm.abs().amax(dim=-1, keepdim=True)
    scale_out = (abs_max.clamp(min=1e-12) / 127.0).to(torch.float32)
    y = torch.clamp((y_norm / scale_out).round(), -128, 127).to(torch.int8)
    # scale 报给下游：去掉 keepdim 的最后一维
    scale = scale_out.squeeze(-1).to(torch.float32)

    return y, xOut.to(out_dtype), scale
