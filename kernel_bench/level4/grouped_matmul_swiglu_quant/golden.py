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
from typing import Tuple

"""
GroupedMatmulSwigluQuant 算子 Torch Golden 参考实现

计算流程：
    y, y_scale = Quant( SwiGLU( Dequant( GroupedMatmul(x, weight) ) ) )

语义约定：
  - GroupedMatmul 按 group_list 把 x 的行切成 E 组，每组与 weight[g] 做 matmul
  - group_list 采用 cumsum 语义（累计和）
  - Dequant 使用 x_scale (per-token) × weight_scale (per-channel)
  - SwiGLU 沿最后一维对半拆分: left, right -> SiLU(left) * right
  - 输出按 per-token 量化为 int8
"""


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分组矩阵乘法 + SwiGLU + 量化 融合算子的 Golden 实现。

    Args:
        x:            [M, K] 激活矩阵（已 per-token 量化），int8
        weight:       [E, K, N] 权重，int8
        weight_scale: [E, N] 权重反量化因子（per-channel），float32
        x_scale:      [M] 激活 per-token 反量化因子，float32
        group_list:   [E] cumsum 语义的累计 token 数，int32

    Returns:
        y:       [M, N/2] int8，SwiGLU 后的 per-token int8 量化结果
        y_scale: [M] float32，per-token 反量化因子
    """
    assert x.dim() == 2 and weight.dim() == 3, "x must be 2D, weight must be 3D [E, K, N]"
    M, K = x.shape
    E, Kw, N = weight.shape
    assert K == Kw, f"K mismatch: x has {K}, weight has {Kw}"
    assert N % 2 == 0, "N must be even so SwiGLU can split the last dim in half"
    N_out = N // 2

    ends = group_list.to(torch.int64).tolist()
    assert len(ends) == E and ends[-1] <= M, "group_list must be cumsum of length E and not exceed M"
    starts = [0] + ends[:-1]

    dequant = torch.empty((M, N), dtype=torch.float32, device=x.device)
    x_scale_f = x_scale.float()
    for g in range(E):
        s, e = starts[g], ends[g]
        if s == e:
            continue
        mm = torch.matmul(x[s:e].float(), weight[g].float())
        xs = x_scale_f[s:e].unsqueeze(1)
        ws = weight_scale[g].float().unsqueeze(0)
        dequant[s:e] = mm * xs * ws

    x_left = dequant[..., :N_out]
    x_right = dequant[..., N_out:]
    activated = torch.nn.functional.silu(x_left) * x_right  # [M, N_out]

    eps = torch.finfo(torch.float32).tiny
    amax = activated.abs().amax(dim=-1).clamp_min(eps)  # [M]
    y_scale = amax / 127.0
    y = torch.clamp(torch.round(activated / y_scale.unsqueeze(1)), -128, 127).to(torch.int8)

    return y, y_scale.to(torch.float32)
