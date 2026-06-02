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
from typing import Optional

"""
QuantMatmul 算子 Torch Golden 参考实现

输入 int8，输出 float16/bfloat16。

计算公式：
    无bias:             out = x1 @ x2 * scale
    int32 bias:         out = (x1 @ x2 + bias) * scale
    浮点bias(post-scale): out = x1 @ x2 * scale + bias
    pertoken:           out = (x1 @ x2 * scale) * pertoken_scale
    offset(非对称量化):  out = x1 @ x2 * scale + offset
"""


def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
) -> torch.Tensor:
    """
    量化矩阵乘法

    Args:
        x1: [..., m, k] int8 左矩阵
        x2: [..., k, n] int8 右矩阵
        scale: [t] 反量化 scale (float32 / bfloat16)
        offset: [t] 非对称量化偏移 (float32, t=1 或 n)，反量化时 out += offset；
                对称量化时为 None
        pertoken_scale: [m] per-token scale (float32)
        bias: [n] 或 [batch, 1, n] 偏置，int32 走 pre-scale，浮点走 post-scale
        output_dtype: 输出类型 "float16"（默认）或 "bfloat16"

    Returns:
        out: [..., m, n] float16 或 bfloat16
    """
    # 矩阵乘：int8/int32 输入直接走 fp64 matmul。
    # PR-001: int8 × int8 单乘积 ≤ 127² = 16129；K 个累加最大 |mm| = 16129·K，
    # 即便 K=65535 也只到 ~1.06e9，远小于 fp64 整数精确上界 2^53 (≈9e15)，
    # 故 fp64 matmul 对该量程是 *精确* 的（与 int64 累加逐位相等）。
    # 不用 int64 matmul：CPU 上 int64 GEMM 无 BLAS，朴素实现比 fp64 慢 ~1000x
    # （大 shape 单次可达 ~60s），会令评测在 golden 阶段超时。fp64 走 BLAS，
    # 精度不变而速度恢复。fp32 不可用（24-bit 尾数，K>1024 会溢出）。
    if x1.dtype in (torch.int8, torch.int32) and x2.dtype in (torch.int8, torch.int32):
        mm = torch.matmul(x1.double(), x2.double())
    else:
        # bf16/fp16 输入路径维持原 fp32 等效计算
        mm = torch.matmul(x1.float(), x2.float()).double()

    # int32 bias 在反量化前累加 (pre-scale)
    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.double()

    # 反量化 scale
    y = mm * scale.double()

    # 非对称量化偏移 (zero-point 校正)：out = mm*scale + offset
    # offset 与 scale 配对（NPU 侧由 npu_trans_quant_param(scale, offset) 打包）；
    # 对称量化时 offset=None，此步为 no-op。
    if offset is not None:
        y = y + offset.double()

    # pertoken_scale 沿 m 维广播
    if pertoken_scale is not None:
        y = y * pertoken_scale.double().unsqueeze(-1)

    # 浮点 bias 在反量化后相加 (post-scale)
    if bias is not None and bias.dtype != torch.int32:
        y = y + bias.double()

    # 输出 dtype，默认 float16
    if output_dtype is None or output_dtype == "float16":
        return y.to(torch.float16)
    elif output_dtype == "bfloat16":
        return y.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")
