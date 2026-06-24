#!/usr/bin/env python3
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

"""
GroupedDynamicBlockQuant 算子 Golden 参考实现（CPU）

基于 numpy 实现分组块量化逻辑，输出 y 保持 float32，scale 为 float32。
精度对比时由 cann-bench 框架将 golden y 截断到目标低比特类型后比较。
"""

from typing import Tuple

import numpy as np
import torch


def get_input(x: torch.Tensor, group_list: torch.Tensor, **attrs) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造合法的 group_list 并返回 [x, group_list]。

    cases.yaml 中通过 attrs.group_list 指定具体的 cumsum 数组；
    未指定时默认使用单 group [M]。
    """
    m = x.shape[-2]
    group_list_values = attrs.get("group_list")
    if group_list_values is None:
        group_list_values = [m]
    else:
        group_list_values = list(group_list_values)

    if group_list_values[-1] != m:
        raise ValueError(
            f"group_list 最后一个元素 {group_list_values[-1]} 必须与 x 的 M 轴 {m} 相等"
        )

    group_list = torch.tensor(group_list_values, dtype=torch.int32, device=x.device)
    return x, group_list


def _grouped_block_reshape_to_blocks(
    x_array: np.ndarray,
    group_list: np.ndarray,
    row_block_size: int,
    col_block_size: int,
):
    """按 group 分组、补 pad、reshape 为 block 并取每个 block 的 max(abs(x))。"""
    axis = -2
    groups = []
    groups_scale_addr = []
    prev = 0

    for index, end in enumerate(group_list):
        slices = [slice(None)] * x_array.ndim
        slices[axis] = slice(prev, end)
        group = x_array[tuple(slices)]
        groups.append(group)
        groups_scale_addr.append(end // row_block_size + index + 1)
        prev = end

    padded_groups = []
    padded_group_row_blocks = []
    for group in groups:
        batch, rows, cols = group.shape
        pad_rows = (row_block_size - rows % row_block_size) % row_block_size
        pad_cols = (col_block_size - cols % col_block_size) % col_block_size
        padded_group = np.pad(
            group,
            ((0, 0), (0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=0,
        )
        padded_groups.append(padded_group)
        padded_group_row_blocks.append(int((rows + pad_rows) / row_block_size))

    padded_array = np.concatenate(padded_groups, axis=axis)
    padded_batch, padded_rows, padded_cols = padded_array.shape
    row_blocks = padded_rows // row_block_size
    col_blocks = padded_cols // col_block_size

    result = np.zeros((padded_batch, row_blocks, col_blocks), dtype=padded_array.dtype)
    for k in range(padded_batch):
        for i in range(row_blocks):
            for j in range(col_blocks):
                block = padded_array[
                    k,
                    i * row_block_size : (i + 1) * row_block_size,
                    j * col_block_size : (j + 1) * col_block_size,
                ]
                result[k, i, j] = np.max(block)

    return result, groups_scale_addr, padded_group_row_blocks


def _reshape_scale_array_to_paded_group(
    scale_array: np.ndarray, groups_scale_addr: list, padded_group_row_blocks: list
):
    """将 per-block scale 插入 group 分隔用的零行，匹配 NPU 输出 layout。"""
    batch, rows, cols = scale_array.shape
    group_num = len(padded_group_row_blocks)
    stacked_scale_array_list = []

    zero_row = np.full((1, cols), 0, dtype=scale_array.dtype)
    for i in range(batch):
        sub_array = scale_array[i]
        sub_array_pre = 0
        sub_sub_array_pre = 0
        sub_sub_array = np.array([], dtype=scale_array.dtype).reshape(0, cols)
        for j in range(group_num):
            group_row_blocks = int(padded_group_row_blocks[j])
            scale_addr = groups_scale_addr[j]
            sub_sub_array = np.vstack(
                [sub_sub_array, sub_array[sub_array_pre : sub_array_pre + group_row_blocks, :]]
            )
            sub_array_pre += group_row_blocks
            diff = scale_addr - group_row_blocks - sub_sub_array_pre
            sub_sub_array_pre += group_row_blocks + diff
            for _ in range(diff):
                sub_sub_array = np.vstack([sub_sub_array, zero_row])
        stacked_scale_array_list.append(sub_sub_array)

    return np.stack(stacked_scale_array_list, axis=0)


def _get_dtype_range(y_dtype: str):
    """获取目标类型的最大有限值（用于 clip）。"""
    if "bfloat16" in y_dtype:
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "float8_e5m2" in y_dtype:
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in y_dtype:
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in y_dtype:
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    np_dtype = np.dtype(y_dtype)
    if np_dtype.kind in "iu":
        info = np.iinfo(np_dtype)
    else:
        info = np.finfo(np_dtype)
    return info.min, info.max


def grouped_dynamic_block_quant(
    x: torch.Tensor,
    group_list: torch.Tensor,
    min_scale: float = 0.0,
    round_mode: str = "rint",
    dst_type: int = 35,
    row_block_size: int = 1,
    col_block_size: int = 128,
    group_list_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GroupedDynamicBlockQuant CPU 参考实现。

    Args:
        x: 输入张量，shape [M, N] 或 [B, M, N]，dtype float16/bfloat16/float32/float64
        group_list: int32 cumsum 数组
        min_scale: 最小 scale 值
        round_mode: 仅用于对齐接口，golden 内统一按 rint 语义处理
        dst_type: 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN
        row_block_size: M 轴 block 大小
        col_block_size: N 轴 block 大小
        group_list_type: 当前仅支持 0

    Returns:
        (y, scale): y 为 float32，scale 为 float32
    """
    dst_type_str = {34: "hifloat8", 35: "float8_e5m2", 36: "float8_e4m3fn"}.get(
        dst_type, "float8_e5m2"
    )

    x_array = x.detach().cpu().to(torch.float32).numpy()
    group_list_array = group_list.detach().cpu().to(torch.int64).numpy().astype(np.int32)

    if group_list_array.ndim != 1:
        raise RuntimeError(f"group_list 必须是 1D 数组，当前 ndim={group_list_array.ndim}")
    if group_list_array[-1] != x_array.shape[-2]:
        raise RuntimeError(
            f"group_list 最后一个元素 {group_list_array[-1]} 必须等于 x M 轴 {x_array.shape[-2]}"
        )
    if not np.all(np.diff(group_list_array) >= 0):
        raise RuntimeError("group_list 必须非递减")

    expend_flag = False
    if x_array.ndim == 2:
        expend_flag = True
        x_array = np.expand_dims(x_array, axis=0)
    if x_array.ndim != 3:
        raise RuntimeError(f"x 维度必须为 2 或 3，当前 ndim={x_array.ndim}")

    # 按目标类型最大有限值做 scale 参考
    if dst_type_str == "hifloat8":
        max_value = 2.0 ** 15
    elif dst_type_str == "float8_e5m2":
        max_value = (2.0 - 2.0 ** -2) * (2.0 ** 15)
    elif dst_type_str == "float8_e4m3fn":
        max_value = (2.0 - 2.0 ** -2) * (2.0 ** 8)
    else:
        max_value = _get_dtype_range(dst_type_str)[1]

    x_array_abs = np.abs(x_array)
    block_max, groups_scale_addr, padded_group_row_blocks = _grouped_block_reshape_to_blocks(
        x_array_abs, group_list_array, row_block_size, col_block_size
    )
    block_max_f32 = block_max.astype(np.float32)

    if min_scale != 0:
        is_finite = np.isfinite(block_max_f32)
        scale_all = block_max_f32 / max_value
        scale = np.minimum(scale_all, 1.0 / min_scale)
        scale = np.where(is_finite, scale, scale_all)
    else:
        scale = block_max_f32 / max_value

    min_normal_f32 = np.finfo(np.float32).tiny
    scale = np.where(scale < min_normal_f32, 0.0, scale)

    # 将 scale 扩展回 x 的 shape，用于逐元素除法
    scale_expanded = np.zeros_like(x_array, dtype=np.float32)
    for k in range(scale.shape[0]):
        scale_offset_ids = 0
        scale_offset = 0
        nxt_scale_offset = group_list_array[0]
        for i in range(scale.shape[1]):
            for j in range(scale.shape[2]):
                value = scale[k, i, j]
                scale_expanded[
                    k,
                    scale_offset : min(scale_offset + row_block_size, nxt_scale_offset),
                    j * col_block_size : (j + 1) * col_block_size,
                ] = value
            scale_offset += row_block_size
            if scale_offset >= nxt_scale_offset:
                scale_offset = nxt_scale_offset
                if scale_offset_ids + 1 < len(group_list_array):
                    nxt_scale_offset = group_list_array[scale_offset_ids + 1]
                    scale_offset_ids += 1

    x_f32 = x_array.astype(np.float32)
    # 避免除零：scale 为 0 的位置输出也置 0（与 NPU 一致）
    with np.errstate(divide="ignore", invalid="ignore"):
        out_f32 = np.where(scale_expanded == 0, 0.0, x_f32 / scale_expanded)

    # scale 输出 layout：按 group 插入零行
    reshape_paded_scale = _reshape_scale_array_to_paded_group(
        scale, groups_scale_addr, padded_group_row_blocks
    )
    output_scale = reshape_paded_scale.astype(np.float32)

    max_norm = _get_dtype_range(dst_type_str)[1]
    np.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)
    round_data = np.round(out_f32, decimals=8)
    round_data = np.nan_to_num(round_data, nan=0.0, copy=False)

    if expend_flag and round_data.shape[0] == 1 and output_scale.shape[0] == 1:
        round_data = np.squeeze(round_data, axis=0)
        output_scale = np.squeeze(output_scale, axis=0)

    return torch.from_numpy(round_data), torch.from_numpy(output_scale)
