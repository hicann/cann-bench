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
from copy import deepcopy

from typing import Optional

"""
MoeFinalizeRouting 算子 Torch Golden 参考实现

在 MoE 计算的最后，合并 MoE FFN 的输出结果
公式：out = skip1 + skip2 + Σ(scales * (expanded_permuted_rows + bias))
"""

def moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    expanded_src_to_dst_row: Optional[torch.Tensor] = None,
    skip1: Optional[torch.Tensor] = None,
    skip2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_for_source_row: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
):
    """
    MoE Finalize Routing 算子 Torch Golden 参考实现

    在 MoE 计算的最后，合并 MoE FFN 的输出结果

    Args:
        expanded_permuted_rows: MoE FFN 输出，shape 为 (NUM_ROWS * K, H) 或 (E, C, H)
        expanded_src_to_dst_row: 行索引映射，shape 为 (NUM_ROWS * K)
        skip1: 共享专家1，shape 为 (NUM_ROWS, H)
        skip2: 共享专家2，shape 为 (NUM_ROWS, H)
        bias: 专家偏置，shape 为 (E, H)
        scales: 路由权重，shape 为 (NUM_ROWS, K)
        expert_for_source_row: 专家索引，shape 为 (NUM_ROWS, K)
        drop_pad_mode: 模式选择，取值范围 [0, 3]
            0: drop less, 按列排列
            1: drop pad, 按列排列
            2: drop less, 按行排列
            3: drop pad, 按行排列

    Returns:
        输出张量，shape 为 (NUM_ROWS, H)
    """
    # 确定输入类型（numpy 或 torch）
    is_torch = isinstance(expanded_permuted_rows, torch.Tensor)

    # 低精度类型升到 fp32，避免累积循环中的舍入误差与 fp64 参考值产生偏差
    original_dtype = None
    if is_torch:
        original_dtype = expanded_permuted_rows.dtype
        _low_prec = original_dtype in (torch.float16, torch.bfloat16)
        if _low_prec:
            expanded_permuted_rows = expanded_permuted_rows.float()
            if skip1 is not None:
                skip1 = skip1.float()
            if skip2 is not None:
                skip2 = skip2.float()
            if bias is not None:
                bias = bias.float()
            if scales is not None:
                scales = scales.float()

    # 确定 K 和 num_rows
    if expanded_src_to_dst_row is None:
        # 如果没有提供索引映射，使用默认值
        NK = expanded_permuted_rows.shape[0]
        K = 1
    else:
        NK = expanded_src_to_dst_row.shape[0]
        K = 1
        if scales is not None:
            K = scales.shape[1]
    num_rows = NK // K
    H = expanded_permuted_rows.shape[-1]

    # 将 expanded_permuted_rows reshape 为 2D
    if is_torch:
        expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)
        dtype = expanded_permuted_rows.dtype
        device = expanded_permuted_rows.device
    else:
        expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)
        dtype = expanded_permuted_rows.dtype

    # 初始化输出：skip1 + skip2
    if (skip1 is not None) and (skip2 is not None):
        if is_torch:
            out = skip1.clone() + skip2
        else:
            out = skip1.copy() + skip2
    elif (skip2 is not None) and (skip1 is None):
        if is_torch:
            out = skip2.clone()
        else:
            out = deepcopy(skip2)
    elif (skip2 is None) and (skip1 is not None):
        if is_torch:
            out = skip1.clone()
        else:
            out = deepcopy(skip1)
    else:
        if is_torch:
            out = torch.zeros(num_rows, H, dtype=dtype, device=device)
        else:
            out = np.zeros([num_rows, H], dtype=dtype)

    # 核心计算循环
    for i in range(num_rows):
        for k in range(K):
            # 根据 drop_pad_mode 获取索引位置
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                # 按列排列
                index_pos = k * num_rows + i
            else:
                # 按行排列 (drop_pad_mode == 2 or 3)
                index_pos = i * K + k

            # 获取行索引值
            if expanded_src_to_dst_row is None:
                # 如果没有提供索引映射，直接使用当前位置
                value = index_pos
            elif is_torch:
                value = expanded_src_to_dst_row[index_pos].item()
            else:
                value = expanded_src_to_dst_row[index_pos]

            # drop pad 模式：索引为 -1 时贡献为 0
            if value == -1:
                if is_torch:
                    dst_row = torch.zeros(H, dtype=dtype, device=device)
                else:
                    dst_row = 0
            else:
                dst_row = expanded_permuted_rows[value, :]

            # 获取缩放因子
            scale_val = 1.0
            if scales is not None:
                if is_torch:
                    scale_val = scales[i, k].item() if scales.dtype in [torch.float16, torch.bfloat16] else scales[i, k]
                else:
                    scale_val = scales[i, k]

            # 获取专家 ID 和 bias
            if bias is not None and expert_for_source_row is not None:
                if is_torch:
                    expert_id = expert_for_source_row[i, k].item()
                else:
                    expert_id = expert_for_source_row[i, k]
                out[i, :] += scale_val * (dst_row + bias[expert_id, :])
            else:
                out[i, :] += scale_val * dst_row

    if is_torch and original_dtype is not None and _low_prec:
        out = out.to(original_dtype)

    return out


def generate_moe_finalize_routing_inputs(
    expert_num=16,
    hidden_dim=512,
    topk=8,
    num_rows=1024,
    dtype="float16",
    use_skip2=True,
    use_bias=True,
    use_scales=True,
    drop_pad_mode=0,
    expert_capacity=None,
    seed=42
):
    """
    生成 MoeFinalizeRouting 算子的测试输入数据

    Args:
        expert_num: 专家数量 E
        hidden_dim: 隐藏层维度 H
        topk: 每个token选择的专家数 K
        num_rows: token 数量 NUM_ROWS
        dtype: 数据类型，支持 float16, float32, bfloat16
        use_skip2: 是否使用 skip2
        use_bias: 是否使用 bias
        use_scales: 是否使用 scales
        drop_pad_mode: 模式选择 [0, 3]
        expert_capacity: drop pad 模式下的专家容量 C
        seed: 随机种子

    Returns:
        包含所有输入参数的字典
    """
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 数据类型映射
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float32  # numpy不支持bfloat16，用float32代替
    }.get(dtype, np.float16)

    # 根据 drop_pad_mode 生成不同 shape 的 expanded_permuted_rows
    if drop_pad_mode == 0 or drop_pad_mode == 2:
        # drop less 模式：2D shape
        expanded_permuted_rows = torch.randn(num_rows * topk, hidden_dim, dtype=torch_dtype)
        expanded_permuted_rows_np = np.random.randn(num_rows * topk, hidden_dim).astype(np_dtype)
        # 索引范围 [0, NUM_ROWS * K - 1]
        expanded_src_to_dst_row = torch.randint(0, num_rows * topk, (num_rows * topk,), dtype=torch.int32)
        expanded_src_to_dst_row_np = np.arange(num_rows * topk).astype(np.int32)
        np.random.shuffle(expanded_src_to_dst_row_np)
    else:
        # drop pad 模式：3D shape
        if expert_capacity is None:
            expert_capacity = num_rows // expert_num + 10
        expanded_permuted_rows = torch.randn(expert_num, expert_capacity, hidden_dim, dtype=torch_dtype)
        expanded_permuted_rows_np = np.random.randn(expert_num, expert_capacity, hidden_dim).astype(np_dtype)
        # 索引范围 [-1, E * C - 1]
        expanded_src_to_dst_row = torch.randint(-1, expert_num * expert_capacity - 1, (num_rows * topk,), dtype=torch.int32)
        expanded_src_to_dst_row_np = np.random.randint(-1, expert_num * expert_capacity - 1, num_rows * topk).astype(np.int32)

    # skip1
    skip1 = torch.randn(num_rows, hidden_dim, dtype=torch_dtype)
    skip1_np = np.random.randn(num_rows, hidden_dim).astype(np_dtype)

    # skip2
    if use_skip2:
        skip2 = torch.randn(num_rows, hidden_dim, dtype=torch_dtype)
        skip2_np = np.random.randn(num_rows, hidden_dim).astype(np_dtype)
    else:
        skip2 = None
        skip2_np = None

    # bias
    if use_bias:
        bias = torch.randn(expert_num, hidden_dim, dtype=torch_dtype)
        bias_np = np.random.randn(expert_num, hidden_dim).astype(np_dtype)
        # expert_for_source_row 必须存在
        expert_for_source_row = torch.randint(0, expert_num, (num_rows, topk), dtype=torch.int32)
        expert_for_source_row_np = np.random.randint(0, expert_num, size=(num_rows, topk)).astype(np.int32)
    else:
        bias = None
        bias_np = None
        expert_for_source_row = None
        expert_for_source_row_np = None

    # scales
    if use_scales:
        scales = torch.randn(num_rows, topk, dtype=torch_dtype)
        scales_np = np.random.randn(num_rows, topk).astype(np_dtype)
    else:
        scales = None
        scales_np = None

    return {
        "torch": {
            "expanded_permuted_rows": expanded_permuted_rows,
            "skip1": skip1,
            "skip2": skip2,
            "bias": bias,
            "scales": scales,
            "expanded_src_to_dst_row": expanded_src_to_dst_row,
            "expert_for_source_row": expert_for_source_row,
            "drop_pad_mode": drop_pad_mode
        },
        "numpy": {
            "expanded_permuted_rows": expanded_permuted_rows_np,
            "skip1": skip1_np,
            "skip2": skip2_np,
            "bias": bias_np,
            "scales": scales_np,
            "expanded_src_to_dst_row": expanded_src_to_dst_row_np,
            "expert_for_source_row": expert_for_source_row_np,
            "drop_pad_mode": drop_pad_mode
        }
    }


if __name__ == "__main__":
    # 测试示例
    inputs = generate_moe_finalize_routing_inputs(
        expert_num=16, hidden_dim=512, topk=8, num_rows=1024,
        dtype="float16", drop_pad_mode=0
    )

    # 使用 numpy 计算 golden
    golden_np = moe_finalize_routing(**inputs["numpy"])
    print(f"Golden output shape: {golden_np.shape}")
    print(f"Golden output dtype: {golden_np.dtype}")
    print(f"Golden output sample: {golden_np[0, :5]}")

    # 使用 torch 计算 golden
    golden_torch = moe_finalize_routing(**inputs["torch"])
    print(f"Torch golden output shape: {golden_torch.shape}")
    print(f"Torch golden output dtype: {golden_torch.dtype}")
    print(f"Torch golden output sample: {golden_torch[0, :5]}")