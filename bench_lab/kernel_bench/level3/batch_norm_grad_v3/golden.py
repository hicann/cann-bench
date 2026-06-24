#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
BatchNormGradV3 算子 Torch Golden 参考实现。

基于 PyTorch aten 接口：torch.ops.aten.native_batch_norm_backward
"""

import torch
from typing import Tuple


def batch_norm_grad_v3(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    save_mean: torch.Tensor,
    save_invstd: torch.Tensor,
    is_training: bool = True,
    epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BatchNormGradV3 算子 Golden 参考实现。

    Args:
        grad_out: 正向输出梯度，shape 与 x 相同。
        x: 正向输入。
        weight: 缩放权重，1D，长度等于 channel 数。
        running_mean: 训练累积均值，1D，长度等于 channel 数。
        running_var: 训练累积方差，1D，长度等于 channel 数。
        save_mean: 前向保存均值，1D，长度等于 channel 数。
        save_invstd: 前向保存标准差倒数，1D，长度等于 channel 数。
        is_training: 是否训练场景。
        epsilon: 防止除零的极小值。

    Returns:
        (dx, dweight, dbias)。dx 的 dtype 与输入 x 一致；dweight/dbias 在 x 为 fp16/bf16 时返回 fp32，
        与 NPU 行为保持一致。
    """
    if x.dim() < 2 or x.dim() > 8:
        raise ValueError(f"BatchNormGradV3 supports 2-8D input, got {x.dim()}D")

    # fp16/bf16 累加容易溢出，提升到 fp32 计算
    orig_dtype = x.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        grad_out = grad_out.float()
        x = x.float()
        weight = weight.float()
        running_mean = running_mean.float()
        running_var = running_var.float()
        save_mean = save_mean.float()
        save_invstd = save_invstd.float()

    dx, dweight, dbias = torch.ops.aten.native_batch_norm_backward(
        grad_out,
        x,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        is_training,
        epsilon,
        output_mask=[True, True, True],
    )

    # NPU 对 fp16/bf16 输入返回 fp32 的 dweight/dbias，golden 与之对齐
    if orig_dtype in (torch.float16, torch.bfloat16):
        return dx.to(orig_dtype), dweight, dbias
    return dx, dweight, dbias
