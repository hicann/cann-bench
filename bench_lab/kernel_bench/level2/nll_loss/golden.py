#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as F


def nll_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
             reduction: str = "mean", ignore_index: int = -100):
    """NLLLoss 的 PyTorch 参考实现。

    使用 torch.nn.functional.nll_loss 计算损失（支持 1D/2D/高维输入），
    并额外返回 total_weight 以与 NPU 输出结构对齐。

    Args:
        input: log-probabilities 张量
        target: 类别索引张量
        weight: 类别权重张量，shape (C,)
        reduction: "none" | "mean" | "sum"
        ignore_index: 忽略的目标值

    Returns:
        (out, total_weight) 元组
    """
    out = F.nll_loss(
        input,
        target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
    )

    # total_weight: 有效 target 对应 weight 之和
    # reduction='none' 时无意义，但为了输出结构与 NPU 一致，返回标量 0
    if reduction == "none":
        total_weight = torch.zeros((), dtype=input.dtype, device=input.device)
    else:
        valid_mask = target != ignore_index
        if valid_mask.any():
            valid_targets = target[valid_mask].flatten()
            total_weight = weight.index_select(0, valid_targets).sum().to(input.dtype)
        else:
            total_weight = torch.zeros((), dtype=input.dtype, device=input.device)

    return out, total_weight
