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
import numpy as np
from typing import Optional

"""
MoeReRouting 算子 Torch/Numpy Golden 参考实现

MoE 网络中，将 token 按照专家顺序重新排列
公式：SrcOffset/DstOffset 双重求和计算位置映射
"""

def moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0
):
    """
    MoeReRouting 算子 Torch/Numpy Golden 参考实现

    MoE 网络中，将 token 按照专家顺序重新排列

    Args:
        tokens: 待重新排布的 token，shape (A, H)
        expert_token_num_per_rank: 每张卡上各个专家处理的 token 数，shape (N, E)
        per_token_scales: 每个 token 对应的 scale，shape (A)，可选
        expert_token_num_type: 输出 expert_token_num 的模式，0=cumsum, 1=count，当前只支持 1
        idx_type: 输出 permute_token_idx 的索引类型，0=gather, 1=scatter，当前只支持 0

    Returns:
        (permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num)
    """
    # 判断输入类型
    is_torch = isinstance(tokens, torch.Tensor)

    # 获取参数
    if is_torch:
        N, E = expert_token_num_per_rank.shape
        A, H = tokens.shape
        dtype = tokens.dtype
        device = tokens.device
        int_dtype = expert_token_num_per_rank.dtype
    else:
        N, E = expert_token_num_per_rank.shape
        A, H = tokens.shape
        dtype = tokens.dtype
        int_dtype = expert_token_num_per_rank.dtype

    # 确保总和匹配
    if is_torch:
        total_tokens = expert_token_num_per_rank.sum().item()
    else:
        total_tokens = expert_token_num_per_rank.sum()
    assert total_tokens == A, f"Sum of expert_token_num_per_rank ({total_tokens}) must equal A ({A})"

    # 构建 src_offset 和 dst_offset 映射
    # 计算 SrcOffset：按 rank 和 expert 的顺序累加
    src_offsets = {}  # (rank, expert) -> src_offset
    dst_offsets = {}  # (rank, expert) -> dst_offset

    # 计算 SrcOffset：按 rank 和 expert 的顺序累加
    src_acc = 0
    for i in range(N):  # cur_rank
        for j in range(E):  # cur_expert
            src_offsets[(i, j)] = src_acc
            if is_torch:
                src_acc += expert_token_num_per_rank[i, j].item()
            else:
                src_acc += expert_token_num_per_rank[i, j]

    # 计算 DstOffset：按 expert 和 rank 的顺序累加
    dst_acc = 0
    for j in range(E):  # cur_expert
        for i in range(N):  # cur_rank
            dst_offsets[(i, j)] = dst_acc
            if is_torch:
                dst_acc += expert_token_num_per_rank[i, j].item()
            else:
                dst_acc += expert_token_num_per_rank[i, j]

    # 构建重排映射：src_pos -> dst_pos
    src_to_dst = {}
    for i in range(N):
        for j in range(E):
            if is_torch:
                num_tokens = expert_token_num_per_rank[i, j].item()
            else:
                num_tokens = expert_token_num_per_rank[i, j]
            src_start = src_offsets[(i, j)]
            dst_start = dst_offsets[(i, j)]
            for k in range(int(num_tokens)):
                src_to_dst[src_start + k] = dst_start + k

    # 构建反向映射用于 gather 索引
    dst_to_src = {v: k for k, v in src_to_dst.items()}

    # 生成 permute_token_idx (gather 索引)
    if is_torch:
        permute_token_idx = torch.zeros(A, dtype=torch.int32, device=device)
        for dst_pos in range(A):
            permute_token_idx[dst_pos] = dst_to_src[dst_pos]
    else:
        permute_token_idx = np.zeros(A, dtype=np.int32)
        for dst_pos in range(A):
            permute_token_idx[dst_pos] = dst_to_src[dst_pos]

    # 重排 tokens
    if is_torch:
        permute_tokens = tokens[permute_token_idx]
    else:
        permute_tokens = tokens[permute_token_idx]

    # 重排 per_token_scales（如果存在）
    if per_token_scales is not None:
        if is_torch:
            permute_per_token_scales = per_token_scales[permute_token_idx]
        else:
            permute_per_token_scales = per_token_scales[permute_token_idx]
    else:
        if is_torch:
            permute_per_token_scales = torch.zeros(A, dtype=torch.float32, device=device)
        else:
            permute_per_token_scales = np.zeros(A, dtype=np.float32)

    # 计算 expert_token_num (count 模式)
    if expert_token_num_type == 1:
        if is_torch:
            expert_token_num = expert_token_num_per_rank.sum(dim=0)
        else:
            expert_token_num = expert_token_num_per_rank.sum(axis=0)
    else:
        # cumsum 模式（暂不支持）
        if is_torch:
            expert_token_num = torch.zeros(E, dtype=int_dtype, device=device)
        else:
            expert_token_num = np.zeros(E, dtype=int_dtype)

    return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num


def get_input(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0
):
    """
    输入数据预处理函数

    调整 expert_token_num_per_rank 使其总和等于 tokens 数量 (A)

    Args:
        tokens: 待重新排布的 token，shape (A, H)
        expert_token_num_per_rank: 每张卡上各个专家处理的 token 数，shape (N, E)
        per_token_scales: 每个 token 对应的 scale，shape (A)，可选
        expert_token_num_type: 输出模式
        idx_type: 索引类型

    Returns:
        处理后的输入数据列表 [tokens, expert_token_num_per_rank, per_token_scales]
    """
    A = tokens.shape[0]
    N, E = expert_token_num_per_rank.shape
    total_cells = N * E

    # 计算每个位置的基础值，确保总和等于 A
    base_value = A // total_cells
    remainder = A % total_cells

    # 生成新的 expert_token_num_per_rank
    if isinstance(expert_token_num_per_rank, torch.Tensor):
        new_expert_token_num = torch.full((N, E), base_value, dtype=expert_token_num_per_rank.dtype)
        # 将剩余的 token 分配到最后一个位置
        new_expert_token_num[-1, -1] += remainder
    else:
        new_expert_token_num = np.full((N, E), base_value, dtype=expert_token_num_per_rank.dtype)
        new_expert_token_num[-1, -1] += remainder

    # 调整 per_token_scales 的形状（如果需要）
    if per_token_scales is not None and per_token_scales.shape[0] != A:
        if isinstance(per_token_scales, torch.Tensor):
            new_scales = torch.randn(A, dtype=per_token_scales.dtype)
        else:
            new_scales = np.random.randn(A).astype(per_token_scales.dtype)
        per_token_scales = new_scales

    return [tokens, new_expert_token_num, per_token_scales]