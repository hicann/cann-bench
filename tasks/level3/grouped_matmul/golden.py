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
from typing import List, Optional

"""
GroupedMatmul 算子 Torch Golden 参考实现

分组矩阵乘法算子，x 沿 M 轴合并、weight 按 expert 维堆叠。
公式：对每个专家 g ∈ [0, E)，根据 group_list（cumsum）取属于该组的 token 行 rows_g：
        y[rows_g] = x[rows_g] @ weight[g] (+ bias[g])
"""


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list=None,  # List[int]，cumsum 语义；不加 type annotation 避免 param_builder 误判
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    """
    分组矩阵乘法算子

    Args:
        x: 激活矩阵 [M, K]，所有组沿 M 轴合并
        weight: 专家权重，按 expert 维堆叠
                - transpose_weight=false: [E, K, N]，直接 matmul
                - transpose_weight=true:  [E, N, K]，需在最后两维 transpose 后 matmul
        bias: 偏置（可选） [E, N]
        group_list: 累计 token 数列表（长度 E），cumsum 语义；最后一个值等于 M
        split_item: 输出切分模式
                    - 0/1: 输出 List[Tensor] 长度 E，按 group_list 切回每组 [m_i, N]
                    - 2/3: 输出单 tensor [M, N]
        transpose_weight: 是否转置权重（见上）

    Returns:
        split_item ∈ {0, 1}: List[Tensor] 长度 E，每个 [m_i, N]
        split_item ∈ {2, 3}: List[Tensor] 长度 1，单 tensor [M, N]
    """
    assert x.dim() == 2, "x must be 2D [M, K]"
    assert weight.dim() == 3, "weight must be 3D [E, K, N] or [E, N, K]"

    M, K = x.shape
    E = weight.shape[0]
    if transpose_weight:
        # weight: [E, N, K]
        assert weight.shape[2] == K, f"K mismatch: x has {K}, weight (transposed) has {weight.shape[2]}"
        N = weight.shape[1]
    else:
        # weight: [E, K, N]
        assert weight.shape[1] == K, f"K mismatch: x has {K}, weight has {weight.shape[1]}"
        N = weight.shape[2]

    if isinstance(group_list, torch.Tensor):
        ends = group_list.to(torch.int64).tolist()
    else:
        ends = list(group_list)
    assert len(ends) == E, f"group_list length {len(ends)} != E {E}"
    assert ends[-1] == M, f"group_list last value {ends[-1]} must equal M {M}"
    starts = [0] + ends[:-1]

    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    x_f = x.float()
    for g in range(E):
        s, e = starts[g], ends[g]
        if s == e:
            continue
        w_g = weight[g].float()
        if transpose_weight:
            # [m_i, K] @ [N, K]^T = [m_i, N]
            mm = torch.matmul(x_f[s:e], w_g.transpose(-2, -1))
        else:
            # [m_i, K] @ [K, N] = [m_i, N]
            mm = torch.matmul(x_f[s:e], w_g)
        if bias is not None:
            mm = mm + bias[g].float().unsqueeze(0)
        y[s:e] = mm.to(x.dtype)

    if split_item in (0, 1):
        return [y[starts[g]:ends[g]] for g in range(E)]
    return [y]


def grouped_matmul_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list=None,
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    """Oracle: A16W16 分组矩阵乘的数学真值 (g)。dtype-agnostic —— 不硬编码 .float()
    (=float32,会把 fp64 下采成 fp32),整条 matmul 跟随输入精度;在 golden_precision=fp64_cpu
    下即真 fp64 oracle。

    注:本算子权重是 fp16/bf16(**非量化**),plain ``grouped_matmul`` 的 .float()(= fp16/bf16
    操作数无损升 fp32 + fp32 累加,再舍回 T)**本身就是标准 A16W16 tensor-core 的同精度参考 (b)**
    —— evaluator 的 bench 路径(``get_bench_function(rel_path) or golden_func``)缺 _bench 时回退到
    plain golden 即正确,故本算子**默认让 plain golden 兼任 bench,只补 oracle**(仅当 bench 与
    plain golden 不同时才需显式 _bench,如 weight_quant 的 bf16-dequant)。唯一病理是 plain golden
    的 .float()(=float32)也把 fp64 oracle 封成 fp32,导致 |b−oracle|=0、相消/小值域严格分支误杀;
    修正 oracle 为真 fp64 后,plain-golden bench 的 fp32 累加在相消 cell 相对 fp64 有非零误差,
    严格分支放松。
    """
    assert x.dim() == 2, "x must be 2D [M, K]"
    assert weight.dim() == 3, "weight must be 3D [E, K, N] or [E, N, K]"

    M, K = x.shape
    E = weight.shape[0]
    if transpose_weight:
        assert weight.shape[2] == K, f"K mismatch: x has {K}, weight (transposed) has {weight.shape[2]}"
        N = weight.shape[1]
    else:
        assert weight.shape[1] == K, f"K mismatch: x has {K}, weight has {weight.shape[1]}"
        N = weight.shape[2]

    if isinstance(group_list, torch.Tensor):
        ends = group_list.to(torch.int64).tolist()
    else:
        ends = list(group_list)
    assert len(ends) == E, f"group_list length {len(ends)} != E {E}"
    assert ends[-1] == M, f"group_list last value {ends[-1]} must equal M {M}"
    starts = [0] + ends[:-1]

    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    for g in range(E):
        s, e = starts[g], ends[g]
        if s == e:
            continue
        w_g = weight[g]  # dtype-agnostic:不 .float(),跟随输入精度(oracle 下为 fp64)
        if transpose_weight:
            mm = torch.matmul(x[s:e], w_g.transpose(-2, -1))
        else:
            mm = torch.matmul(x[s:e], w_g)
        if bias is not None:
            mm = mm + bias[g].to(mm.dtype).unsqueeze(0)
        y[s:e] = mm.to(x.dtype)

    if split_item in (0, 1):
        return [y[starts[g]:ends[g]] for g in range(E)]
    return [y]
