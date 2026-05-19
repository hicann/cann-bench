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
ApplyAdamW 算子 Torch Golden 参考实现

FP16/BF16 输入升精度到 FP32 计算，FP32/FP64 保持原样计算。
"""


def apply_adam_w(
    var: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    epsilon: float = 1e-8,
    step: int = 1,
    maximize: bool = False,
) -> torch.Tensor:
    # 检测输入 dtype
    input_dtype = var.dtype

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    var = var.to(compute_dtype)
    grad = grad.to(compute_dtype)
    m = m.to(compute_dtype)
    v = v.to(compute_dtype)

    # Adam 偏置校正——按 step 取指数。step=1 时与旧公式 (1 - beta) 完全等价，
    # 现有 cases 不传 step 时走 default=1 保持向后兼容。
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1 ** step)
    v_hat = v_new / (1 - beta2 ** step)
    update = m_hat / (v_hat.sqrt() + epsilon)
    if weight_decay != 0:
        update = update + var * weight_decay
    result = var + lr * update if maximize else var - lr * update

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return result.to(input_dtype)
    return result
