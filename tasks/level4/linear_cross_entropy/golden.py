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
import torch.nn.functional as F

"""
LinearCrossEntropy算子Torch Golden参考实现

融合 LM-head 线性投影与交叉熵损失：logits = hidden @ weight^T，
loss = CrossEntropy(logits, labels)，支持 reduction (mean/sum) 与 ignore_index。
kernel 侧应分块计算、在线 softmax，不物化完整 [T, V] logits。
plain golden 以 fp32 计算 logits 与损失（bench 语义）；
linear_cross_entropy_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _linear_cross_entropy_core(
    hidden, weight, labels, reduction, ignore_index, compute_dtype
):
    """Token-chunked reference: retain full vocabulary and global valid-token reduction."""
    if hidden.shape[0] == 0 or reduction not in ("mean", "sum"):
        # Retain legacy edge behavior without extending the declared task domain.
        logits = torch.matmul(hidden.to(compute_dtype), weight.to(compute_dtype).t())
        return F.cross_entropy(
            logits, labels.long(), reduction=reduction, ignore_index=ignore_index
        ).reshape(1)
    weight_c = weight.to(compute_dtype)
    labels_c = labels.long()
    losses = []
    # This controls logits materialization only, not the already-promoted weight.
    for start in range(0, hidden.shape[0], 128):
        logits = torch.matmul(
            hidden[start : start + 128].to(compute_dtype), weight_c.t()
        )
        losses.append(
            F.cross_entropy(
                logits,
                labels_c[start : start + 128],
                reduction="none",
                ignore_index=ignore_index,
            )
        )
    total = torch.cat(losses).sum()
    if reduction == "mean":
        total = total / (labels_c != ignore_index).sum()
    return total.reshape(1)


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    融合 LM-head 投影 + 交叉熵损失（plain golden = bench：fp32 计算）

    Args:
        hidden: [T, H] 最后一层 hidden states, bfloat16/float16/float32
        weight: [V, H] LM-head 投影权重, dtype 与 hidden 一致
        labels: [T] 目标 token id (int32), 取值 [0, V-1] 或等于 ignore_index
        reduction: 聚合方式, "mean"（按有效 token 数归一）或 "sum"
        ignore_index: 忽略的标签值, 默认 -100

    Returns:
        loss: [1] 聚合后的交叉熵损失, float32
    """
    return _linear_cross_entropy_core(
        hidden, weight, labels, reduction, ignore_index, torch.float32
    )


def linear_cross_entropy_oracle(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _linear_cross_entropy_core(
        hidden, weight, labels, reduction, ignore_index, hidden.dtype
    )
