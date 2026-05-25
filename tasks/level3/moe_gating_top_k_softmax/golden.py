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

"""
MoeGatingTopKSoftmax算子Torch Golden参考实现

公式:
  softmaxOut = softmax(x, axis=-1)
  yOut, expertIdxOut = topK(softmaxOut, k)
  rowIdxRange = arange(expertIdxOut.shape[0] * expertIdxOut.shape[1])
  rowIdxOut = rowIdxRange.reshape([expertIdxOut.shape[1], expertIdxOut.shape[0]]).transpose(1, 0)

注意: row_idx是展平后的全局位置索引，而非行号
"""
def moe_gating_top_k_softmax(
    x: torch.Tensor,
    finished: torch.Tensor = None,
    k: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MoE门控网络中Softmax和TopK的融合

    Args:
        x: 输入张量，shape (..., E)
        finished: 可选，标记哪些行参与计算，bool类型，shape x_shape[:-1]
        k: topK数量

    Returns:
        y: topK值，shape (..., k)
        expert_idx: topK索引（专家序号），shape (..., k)，int32
        row_idx: 行位置索引（展平后的全局位置），shape (..., k)，int32
    """
    # Softmax沿最后一维
    softmax_out = torch.nn.functional.softmax(x, dim=-1)

    # TopK沿最后一维
    values, indices = torch.topk(softmax_out, k, dim=-1)

    # 计算row_idx，严格对标op-plugin测试代码
    # 公式: rowIdxRange = arange(shape[0] * shape[1])
    #       rowIdxOut = rowIdxRange.reshape([shape[1], shape[0]]).transpose(1, 0)
    # 注意: row_idx是展平后的全局位置索引
    output_shape = indices.shape

    if len(output_shape) == 2:
        # 2D: (N, k)
        # row_idx = arange(N*k).reshape(k, N).transpose(1, 0) -> (N, k)
        row_idx_range = torch.arange(output_shape[0] * output_shape[1], dtype=torch.int32)
        row_idx = row_idx_range.reshape(output_shape[1], output_shape[0]).transpose(0, 1)
    else:
        # 3D: (B, N, k)
        # 先把(B, N)看作整体，计算展平后的索引
        # row_idx_range = arange(B*N*k)
        # reshape成(k, B*N)，transpose成(B*N, k)
        # 再reshape成(B, N, k)
        row_idx_range = torch.arange(output_shape[0] * output_shape[1] * output_shape[2], dtype=torch.int32)
        row_idx = row_idx_range.reshape(output_shape[2], output_shape[0] * output_shape[1]).transpose(0, 1)
        row_idx = row_idx.reshape(output_shape)

    # 处理finished参数
    # F510: `num_expert` (== E) is an out-of-range SENTINEL marking finished
    # tokens — downstream consumers MUST treat `expert_id == num_expert` as
    # "skip routing", NOT as a valid expert index. `moe_finalize_routing`
    # golden bounds-checks before `bias[expert_id, :]` to honor this contract;
    # any non-golden consumer of `indices` must do the same.
    if finished is not None:
        num_expert = x.shape[-1]
        finished_expanded = finished.unsqueeze(-1).expand_as(indices)
        indices = torch.where(finished_expanded, num_expert, indices)

    return values, indices.to(torch.int32), row_idx.to(torch.int32)