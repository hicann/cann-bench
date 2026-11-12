#!/usr/bin/env python# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import torch
import random
import numpy as np
from typing import Optional


def make_noncontig(tensor: torch.Tensor, pad: int, dim: int, device: str):
    """在指定维度后 padding，使张量在该维度上非连续。"""
    shape = list(tensor.shape)
    shape[dim] += pad
    padded = torch.zeros(shape, dtype=tensor.dtype, device=device)
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(0, tensor.shape[dim])
    padded[tuple(slices)] = tensor
    return padded[tuple(slices)]


def get_input(input, indices, update, **kwargs):
    """
    对 indices 做规则化构造，把 T 个 update 按行优先映射到 input 的 (bn, bs) 位置。
    若 case 名称包含 noncontiguous，则将 input 在第 1 维构造为非连续张量。
    """
    bn, bs, D = input.shape
    T, _ = indices.shape

    # 构造规则索引：第0列按 bs 分块，第1列在块内循环
    indices_flat = torch.arange(T, dtype=torch.int32, device=input.device)
    col1 = indices_flat // bs
    col2 = indices_flat % bs
    indices = torch.stack([col1, col2], dim=1)

    # 处理非连续输入场景
    case_name = kwargs.get("case_name", "")
    case_idx = kwargs.get("case_idx", 0)
    if "noncontiguous" in case_name:
        random.seed(case_idx)
        pad = random.randint(1, 128)
        input = make_noncontig(input, pad=pad, dim=1, device=str(input.device))

    return (input, indices, update)


def scatter_block_update(
    input: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
):
    """
    ScatterBlockUpdate算子Torch Golden参考实现

    公式: scatter_block_update(...)
    """
    # Make a copy of input to avoid modifying it
    output = input.clone()
    if indices.shape[0] == 0:
        return output

    # The task requires distinct destination blocks. Assign those blocks in one
    # operation instead of dispatching a Python slice assignment for every row.
    # Keep the row-wise behavior for malformed update shapes: missing rows
    # raise, extra rows are ignored, and a flat update is not broadcast.
    if (
        indices.dtype in (torch.int32, torch.int64)
        and update.ndim == 2
        and update.shape[0] == indices.shape[0]
    ):
        rows, cols = indices.to(torch.int64).unbind(dim=1)
        # Normalize negative indices for duplicate detection only. Use the
        # original indices for assignment so out-of-bounds indices still fail.
        canonical_rows = torch.where(rows < 0, rows + input.shape[0], rows)
        canonical_cols = torch.where(cols < 0, cols + input.shape[1], cols)
        destinations = canonical_rows * input.shape[1] + canonical_cols
        if torch.unique(destinations).numel() == indices.shape[0]:
            output[rows, cols, :] = update.to(dtype=output.dtype)
            return output

    # Preserve the historical last-write-wins behavior for duplicate indices,
    # even though such inputs are outside the task's distinct-index contract.
    for k in range(indices.shape[0]):
        idx0 = indices[k, 0].item()
        idx1 = indices[k, 1].item()
        output[idx0, idx1, :] = update[k, :]
    return output
