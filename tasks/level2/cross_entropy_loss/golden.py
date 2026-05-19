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


def cross_entropy_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """CrossEntropyLoss Torch Golden 参考实现 (P1 op, 对齐 torch.nn.functional.cross_entropy).

    公式: L = -log(softmax(input)[target])  ('mean' / 'sum' / 'none' reduction)

    本 spec 只暴露 reduction + ignore_index 两个 attr,torch 的其他 optional
    kwargs (weight / label_smoothing / size_average / reduce) 不在本评测范围。

    Args:
        input: logits 张量,shape (N, C) 或 (N, C, d1, ...) (channel_first)
        target: 硬标签 (N,) 或 (N, d1, ...) ,或 软标签 (N, C) 等
        reduction: 'mean' | 'sum' | 'none',默认 'mean'
        ignore_index: 忽略的硬标签索引,默认 -100

    Returns:
        loss: reduction='none' 时 shape 与 target 一致 (去除 C 维),否则标量
    """
    return F.cross_entropy(
        input=input,
        target=target,
        reduction=reduction,
        ignore_index=ignore_index,
    )
