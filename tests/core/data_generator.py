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

import json
import re
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
        # 处理 CaseLoader 包装: 单元素列表包含字符串型嵌套结构
        # 例如 dtypes = ['[[float16, float16], ...]']
        if isinstance(dtypes, list) and len(dtypes) == 1 and isinstance(dtypes[0], str) and dtypes[0].startswith('['):
            dtypes = self._maybe_parse_json(dtypes[0])
        if isinstance(input_shapes, list) and len(input_shapes) == 1 and isinstance(input_shapes[0], str) and input_shapes[0].startswith('['):
            input_shapes = self._maybe_parse_json(input_shapes[0])
        if isinstance(value_ranges, list) and len(value_ranges) == 1 and isinstance(value_ranges[0], str) and value_ranges[0].startswith('['):
            value_ranges = self._maybe_parse_json(value_ranges[0])

        # 再处理普通字符串
        input_shapes = self._maybe_parse_json(input_shapes)
        dtypes = self._maybe_parse_json(dtypes)
        value_ranges = self._maybe_parse_json(value_ranges)

        input_tensors = []
        num_inputs = self._count_inputs(input_shapes)

        # 扩展dtypes
        if not isinstance(dtypes, list) or (dtypes and isinstance(dtypes[0], str) and len(dtypes) < num_inputs):
            if isinstance(dtypes, str):
                dtypes = [dtypes]
            elif not isinstance(dtypes, list):
                dtypes = ['float32']
        # 如果 dtypes 是嵌套结构与 input_shapes 维度匹配，直接使用
        # 否则补齐长度
        if self._structure_depth(dtypes) != self._structure_depth(input_shapes):
            if isinstance(dtypes, list) and len(dtypes) < num_inputs:
                dtypes = dtypes + [dtypes[-1]] * (num_inputs - len(dtypes)) if dtypes else ['float32'] * num_inputs

        # 规范化 value_ranges: 区分单输入 [min, max] 和多输入 [[min1, max1], [min2, max2], ...]
        value_ranges = self._normalize_value_ranges(value_ranges, num_inputs)

        self._generate_tensors(input_shapes, dtypes, value_ranges, input_tensors)

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

    # ---- 嵌套结构支持（GroupedMatmul 等算子） ----

    def _maybe_parse_json(self, value: Any) -> Any:
        """如果是 JSON 格式的字符串，尝试解析为 Python 对象"""
        if isinstance(value, str) and value.startswith('['):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            # 处理未加引号的标识符: [float16, float32] → ["float16", "float32"]
            try:
                # 将不是数字、布尔值、null 的标识符加上双引号
                quoted = re.sub(r'(?<![\"\'\w])([a-zA-Z_]\w*)(?![\"\'\w])', r'"\1"', value)
                return json.loads(quoted)
            except (json.JSONDecodeError, ValueError):
                pass
        return value

    def _count_inputs(self, shapes: Any) -> int:
        """统计输入张量总数（顶层列表长度）"""
        if isinstance(sizes := shapes, list):
            return len(sizes)
        return 1

    def _structure_depth(self, obj: Any) -> int:
        """估算嵌套结构深度：1=扁平, 2=一层嵌套, 3=两层嵌套"""
        if not isinstance(obj, list):
            return 0
        if not obj:
            return 1
        first = obj[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            return 3  # [[[...]]]
        if isinstance(first, list):
            return 2  # [[...]]
        return 1  # [...]

    def _generate_tensors(self, shapes, dtypes, value_ranges, output: list):
        """递归生成张量，支持嵌套结构"""
        if not isinstance(shapes, list):
            return

        # 单个形状 [N, C, H, W] —— 直接生成单个张量
        if shapes and isinstance(shapes[0], int):
            dtype_item = dtypes if isinstance(dtypes, str) else (dtypes[0] if dtypes else 'float32')
            vr_item = value_ranges if isinstance(value_ranges, list) else None
            output.append(self.generate_input_tensor(shapes, dtype_item, vr_item))
            return

        # 判断是嵌套列表 [[shape, ...], ...] 还是扁平形状列表 [shape1, shape2, ...]
        # 关键区别：嵌套列表的每个子元素内部元素是 int（形状），而扁平列表的元素本身就是 int
        is_nested = (shapes and isinstance(shapes[0], list) and
                     shapes[0] and isinstance(shapes[0][0], list))

        if is_nested:
            # 嵌套列表 [[shape, shape], [shape, ...], ...] —— 递归到每个子组
            for i, s in enumerate(shapes):
                d = dtypes[i] if isinstance(dtypes, list) and i < len(dtypes) else dtypes
                vr = value_ranges[i] if isinstance(value_ranges, list) and i < len(value_ranges) else value_ranges
                sub = []
                self._generate_tensors(s, d, vr, sub)
                if sub and len(sub) == 1:
                    output.append(sub[0])
                else:
                    output.append(sub)
        else:
            # 扁平形状列表 [[N,C,H,W], [N,C,H,W], ...] —— 每个元素是一个输入张量
            for i, s in enumerate(shapes):
                d = dtypes[i] if isinstance(dtypes, list) and i < len(dtypes) else dtypes
                vr = value_ranges[i] if isinstance(value_ranges, list) and i < len(value_ranges) else None
                sub = []
                self._generate_tensors(s, d, vr, sub)
                output.extend(sub)