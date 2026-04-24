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

"""
测试数据生成器

职责：
1. 根据shape和dtype生成输入张量
2. 根据value_range填充数据
"""

import torch
from typing import List, Any, Union

from ..utils.dtype_mapper import str_to_torch_dtype, is_float_dtype, is_int_dtype


class DataGenerator:
    """测试数据生成器"""

    def __init__(self, seed: int = None):
        if seed:
            torch.manual_seed(seed)

    def generate_input_tensor(self, shape: List[int], dtype: str, value_range: Any = None) -> torch.Tensor:
        """生成输入张量"""
        torch_dtype = str_to_torch_dtype(dtype)
        min_val, max_val = self._parse_range(value_range, dtype)

        if is_float_dtype(dtype):
            if min_val == max_val:
                return torch.full(shape, float(min_val), dtype=torch_dtype)
            return self._gen_float(shape, torch_dtype, float(min_val), float(max_val))
        elif is_int_dtype(dtype):
            if min_val == max_val:
                return torch.full(shape, int(min_val), dtype=torch_dtype)
            return torch.randint(int(min_val), int(max_val) + 1, shape, dtype=torch_dtype)
        else:
            return torch.zeros(shape, dtype=torch_dtype)

    def generate_input_tensors_from_case(self, input_shapes: List, dtypes: List, value_ranges: List) -> List:
        """根据用例信息生成输入数据"""
        input_tensors = []
        num_inputs = len(input_shapes)

        # 扩展dtypes
        if len(dtypes) < num_inputs:
            dtypes = dtypes + [dtypes[-1]] * (num_inputs - len(dtypes)) if dtypes else ['float32'] * num_inputs

        # 规范化 value_ranges: 区分单输入 [min, max] 和多输入 [[min1, max1], [min2, max2], ...]
        value_ranges = self._normalize_value_ranges(value_ranges, num_inputs)

        for i in range(num_inputs):
            shape_item = input_shapes[i]
            dtype_item = dtypes[i] if isinstance(dtypes[i], str) else (dtypes[i][0] if dtypes[i] else 'float32')
            value_range_item = value_ranges[i] if i < len(value_ranges) else None

            # 检查是否为张量列表
            if self._is_tensor_list(shape_item):
                tensors = [self.generate_input_tensor(s, dtype_item, value_range_item) for s in shape_item]
                input_tensors.append(tensors)
            else:
                input_tensors.append(self.generate_input_tensor(shape_item, dtype_item, value_range_item))

        return input_tensors

    def _normalize_value_ranges(self, value_ranges: List, num_inputs: int) -> List:
        """规范化 value_ranges 为每个输入一个范围"""
        if not value_ranges:
            return [None] * num_inputs

        # 判断是单输入 [min, max] 还是多输入 [[min1, max1], ...]
        first_item = value_ranges[0]
        is_single_range = not isinstance(first_item, list)

        if is_single_range:
            # 单输入算子: value_range 就是 [min, max]
            return [value_ranges] + [None] * (num_inputs - 1)
        else:
            # 多输入算子: value_ranges 是 [[min1, max1], [min2, max2], ...]
            if len(value_ranges) < num_inputs:
                value_ranges = value_ranges + [value_ranges[-1]] * (num_inputs - len(value_ranges))
            return value_ranges

    def _parse_range(self, value_range: Any, dtype: str) -> tuple:
        """解析值范围"""
        if value_range is None:
            return (0.0, 1.0) if is_float_dtype(dtype) else (0, 100)

        if isinstance(value_range, list):
            if len(value_range) >= 2:
                return (value_range[0], value_range[1])
            elif len(value_range) == 1:
                return (value_range[0], value_range[0])

        return (value_range, value_range)

    def _is_tensor_list(self, shape_item: Any) -> bool:
        """判断是否为张量列表"""
        return (isinstance(shape_item, list) and shape_item and
                isinstance(shape_item[0], list) and shape_item[0] and
                isinstance(shape_item[0][0], int))

    def _gen_float(self, shape: List[int], dtype: torch.dtype, min_val: float, max_val: float) -> torch.Tensor:
        """生成浮点张量"""
        # 处理特殊值
        if isinstance(min_val, str) or isinstance(max_val, str):
            return self._gen_special(shape, dtype, min_val, max_val)

        # 使用 torch.finfo 获取 dtype 的精确范围
        finfo = torch.finfo(dtype)
        dmin = finfo.min
        dmax = finfo.max

        # 裁剪到 dtype 可表示范围
        min_val = max(min_val, dmin)
        max_val = min(max_val, dmax)

        # 当范围过大时，uniform_ 会因 max-min 溢出而失败
        # 改用 float64 中间计算避免溢出
        range_val = max_val - min_val
        if range_val > dmax:
            rand_f64 = torch.rand(shape, dtype=torch.float64)
            tensor_f64 = rand_f64 * (max_val - min_val) + min_val
            # clamp 确保值严格在 dtype 范围内，避免转换溢出
            tensor_f64 = torch.clamp(tensor_f64, dmin, dmax)
            return tensor_f64.to(dtype)

        tensor = torch.empty(shape, dtype=dtype)
        tensor.uniform_(min_val, max_val)
        return tensor

    def _gen_special(self, shape: List[int], dtype: torch.dtype, min_val: Any, max_val: Any) -> torch.Tensor:
        """生成包含特殊值的张量"""
        tensor = torch.randn(shape, dtype=torch.float32).to(dtype)

        # 转换特殊值字符串
        def to_float(v):
            if v == 'inf': return float('inf')
            if v == '-inf': return float('-inf')
            if v == 'nan': return float('nan')
            return float(v)

        min_f = to_float(min_val) if isinstance(min_val, str) else min_val
        max_f = to_float(max_val) if isinstance(max_val, str) else max_val

        # 在边界填充特殊值
        flat = tensor.flatten()
        n = max(1, len(flat) // 20)
        if min_f == float('-inf') or max_f == float('inf'):
            flat[:n] = float('-inf')
            flat[-n:] = float('inf')

        return tensor