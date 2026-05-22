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
from typing import Optional, List

"""
QuantMatmul 算子 Torch Golden 参考实现

输入 int8，输出 float16/bfloat16。

计算公式：
    无bias:             out = x1 @ x2 * scale
    int32 bias:         out = (x1 @ x2 + bias) * scale
    浮点bias(post-scale): out = x1 @ x2 * scale + bias
    pertoken:           out = (x1 @ x2 * scale) * pertoken_scale
"""


def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
    group_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    量化矩阵乘法

    Args:
        x1: [..., m, k] int8 左矩阵
        x2: [..., k, n] int8 右矩阵
        scale: [t] 反量化 scale (float32 / bfloat16)
        pertoken_scale: [m] per-token scale (float32)
        bias: [n] 或 [batch, 1, n] 偏置，int32 走 pre-scale，浮点走 post-scale
        output_dtype: 输出类型 "float16"（默认）或 "bfloat16"
        group_sizes: 分组量化粒度 [group_m, group_n, group_k]

    Returns:
        out: [..., m, n] float16 或 bfloat16
    """
    # 矩阵乘（int8 用 float32 等效计算）
    mm = torch.matmul(x1.float(), x2.float())

    # int32 bias 在反量化前累加 (pre-scale)
    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.float()

    # 反量化 scale
    y = mm * scale.float()

    # pertoken_scale 沿 m 维广播
    if pertoken_scale is not None:
        y = y * pertoken_scale.float().unsqueeze(-1)

    # 浮点 bias 在反量化后相加 (post-scale)
    if bias is not None and bias.dtype != torch.int32:
        y = y + bias.float()

    # 输出 dtype，默认 float16
    if output_dtype is None or output_dtype == "float16":
        return y.to(torch.float16)
    elif output_dtype == "bfloat16":
        return y.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")
