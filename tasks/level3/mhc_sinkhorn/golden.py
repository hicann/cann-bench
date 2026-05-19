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
MhcSinkhorn 算子 Torch Golden 参考实现 — linear-domain Sinkhorn-Knopp

来源对齐 DeepSeek-V4 mHC (Manifold-Constrained Hyper-Connections) 模块中
hyper-connection stream 混合矩阵的双随机化投影 kernel (详见 arXiv:2512.24880)：
第 1 轮做 row-softmax (数值稳定地把 raw 输入转成概率分布) + 加 eps 偏移，再
column normalize；剩余 iter_step - 1 轮做交替 row / column 归一化，直到
row_sum 和 col_sum 都趋近 1 (投影到 Birkhoff Polytope 流形)。

输入输出均为 fp32 [B, hc_mult, hc_mult] 方阵，inner 两维必须相等。
DSv4 实际取 hc_mult=4，即 4×4 的小方阵；B 可达 16384。
"""


def mhc_sinkhorn(
    comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """
    mHC Sinkhorn — linear-domain doubly-stochastic projection on hc_mult × hc_mult matrices

    Args:
        comb: 输入 hyper-connection 混合矩阵，shape [B, hc_mult, hc_mult] fp32，inner 两维必须相等
        iter_step: Sinkhorn 总迭代轮数 (≥ 1)；DSv4 production 默认 20
        eps: 数值稳定项，所有除法分母上加该常量

    Returns:
        comb_out: 双随机化后的方阵，shape 与输入完全一致 (fp32)
    """
    assert comb.dtype == torch.float32, (
        f"mhc_sinkhorn 仅支持 fp32 输入；mHC 迭代对低精度敏感会破坏双随机性质 (got {comb.dtype})"
    )
    # First iter: row-softmax + eps, then column normalize
    # 等价于把 raw logits 转成 row-probability 分布的起点
    row_max = comb.amax(dim=-1, keepdim=True)               # numerically-stable shift
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps                              # row-softmax + eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    # 后续 iter_step - 1 轮：linear-domain row + column normalize
    for _ in range(iter_step - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)

    return comb
