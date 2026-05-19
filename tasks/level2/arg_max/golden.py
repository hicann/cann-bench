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


def arg_max(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """ArgMax 算子 Torch Golden 参考实现 (P1 op, 对齐 torch.argmax).

    公式: indices = argmax(input, dim=dim)
    返回最大值索引张量;若 keepdim=True 则保留 reduce 轴为 size 1,否则去掉该轴.

    Args:
        input: 输入张量
        dim: 计算 argmax 的维度
        keepdim: 是否保留约简维度,默认 False

    Returns:
        indices (int64): 最大值索引张量
    """
    return torch.argmax(input, dim=dim, keepdim=keepdim)
