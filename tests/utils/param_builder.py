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
参数构建器

职责：
1. 解析golden函数签名
2. 根据用例数据构建调用参数
3. 处理 input_shape 中 null 位置，正确映射 tensor 参数
"""

from typing import Dict, List, Any, Callable, Optional
from inspect import signature, Parameter


class ParamBuilder:
    """参数构建器"""

    def __init__(self, importer=None):
        self.importer = importer

    def build_call_params(self, golden_func: Callable, case: Any, input_tensors: List) -> Dict[str, Any]:
        """构建golden函数调用参数

        Args:
            golden_func: Golden 函数
            case: 测试用例信息（包含 input_shapes, attrs 等）
            input_tensors: 已生成的输入张量列表（不含 null 位置的 tensor）

        Returns:
            参数字典，用于调用 golden_func
        """
        sig = signature(golden_func)
        params = {}

        # 分类参数：tensor 参数、tensor list 参数、属性参数
        # 同时区分 optional 和必需参数
        tensor_params = []
        tensor_list_params = []
        optional_tensor_params = []
        optional_tensor_list_params = []
        attr_params = []

        for name, param in sig.parameters.items():
            annotation = str(param.annotation) if param.annotation != Parameter.empty else ""
            is_optional = 'Optional' in annotation or 'NoneType' in annotation or param.default != Parameter.empty

            if 'List[' in annotation and 'Tensor' in annotation:
                if is_optional:
                    optional_tensor_list_params.append(name)
                else:
                    tensor_list_params.append(name)
            elif 'Tensor' in annotation:
                if is_optional:
                    optional_tensor_params.append(name)
                else:
                    tensor_params.append(name)
            else:
                attr_params.append(name)

        # 获取原始 input_shapes（包含 null 信息）
        original_shapes = getattr(case, 'input_shapes', None)

        # 构建 tensor 参数映射表：位置 -> 参数名
        # 优先映射必需参数，再映射 optional 参数
        tensor_param_map = self._build_tensor_param_map_with_optional(
            tensor_params, tensor_list_params,
            optional_tensor_params, optional_tensor_list_params,
            original_shapes
        )

        # 匹配张量参数
        tensor_idx = 0
        for position, param_name in tensor_param_map.items():
            if tensor_idx < len(input_tensors):
                val = input_tensors[tensor_idx]
                all_tensor_list_params = tensor_list_params + optional_tensor_list_params
                # 如果值是列表且参数期望单个张量，展开列表到后续参数
                if isinstance(val, list) and param_name in tensor_params:
                    # 当前参数取第一个元素
                    params[param_name] = val[0] if val else None
                    tensor_idx += 1
                    # 将剩余元素作为独立的输入项
                    if len(val) > 1:
                        input_tensors = input_tensors[:tensor_idx] + [v for v in val[1:]] + input_tensors[tensor_idx:]
                elif param_name in all_tensor_list_params:
                    params[param_name] = val if isinstance(val, list) else [val]
                    tensor_idx += 1
                else:
                    params[param_name] = val
                    tensor_idx += 1

        # 处理属性参数
        attrs = getattr(case, 'attrs', None) or {}
        for name in attr_params:
            if name in attrs:
                params[name] = self._convert_value(attrs[name])
            elif sig.parameters[name].default != Parameter.empty:
                params[name] = sig.parameters[name].default

        return params

    def _build_tensor_param_map_with_optional(
            self,
            tensor_params: List[str],
            tensor_list_params: List[str],
            optional_tensor_params: List[str],
            optional_tensor_list_params: List[str],
            original_shapes: Optional[List]
        ) -> Dict[int, str]:
        """构建 tensor 参数位置映射表，按函数签名顺序映射

        Args:
            tensor_params: 必需 tensor 参数名列表
            tensor_list_params: 必需 tensor list 参数名列表
            optional_tensor_params: optional tensor 参数名列表
            optional_tensor_list_params: optional tensor list 参数名列表
            original_shapes: 原始 input_shapes

        Returns:
            字典 {tensor_index: param_name}
        """
        # 所有 tensor 参数，按签名顺序排列
        # 必需参数在前，optional 参数在后（但需保持签名顺序）
        # 注意：签名顺序是 tensor_params, tensor_list_params, optional_tensor_params, optional_tensor_list_params
        # 但实际签名顺序可能混合，这里简化处理：先必需再 optional

        # 简化方案：按必需/optional 分组，每组内按参数类型顺序排列
        required_params = tensor_params + tensor_list_params  # ['x', 'weight_ih', 'weight_hh']
        # optional 参数按签名顺序：先 tensor_list 类型的 optional，再 tensor 类型的 optional
        optional_params = optional_tensor_list_params + optional_tensor_params  # ['bias_ih', 'bias_hh', 'h0']

        all_params = required_params + optional_params

        if original_shapes is None:
            # 没有 shapes 信息，优先映射必需参数
            result = {}
            idx = 0
            for name in required_params:
                result[idx] = name
                idx += 1
            return result

        # 处理 input_shapes 格式
        shapes_to_check = self._normalize_input_shapes(original_shapes)
        if shapes_to_check is None:
            result = {}
            idx = 0
            for name in required_params:
                result[idx] = name
                idx += 1
            return result

        # 统计有效 shapes 数量（不含 null）
        valid_shapes_count = sum(1 for s in shapes_to_check if s is not None)

        # 按顺序映射：先必需参数，再 optional 参数
        param_map = {}
        tensor_idx = 0
        param_idx = 0

        for position, shape_item in enumerate(shapes_to_check):
            is_null = shape_item is None

            # 判断是否为嵌套结构
            is_nested = (isinstance(shape_item, list) and shape_item and
                         isinstance(shape_item[0], list) and shape_item[0] and
                         not isinstance(shape_item[0][0], int))

            if is_nested:
                # 嵌套结构，递归处理
                remaining_required = required_params[param_idx:] if param_idx < len(required_params) else []
                remaining_optional = optional_params if param_idx >= len(required_params) else []
                sub_map = self._build_tensor_param_map_with_optional(
                    remaining_required, [],
                    [], remaining_optional,
                    shape_item
                )
                for t_idx, p_name in sub_map.items():
                    param_map[tensor_idx + t_idx] = p_name
                tensor_idx += len(sub_map)
                param_idx = min(param_idx + len(sub_map), len(all_params))
            elif is_null:
                # null 位置：跳过对应的参数
                if param_idx < len(all_params):
                    param_idx += 1
            else:
                # 有效 shape
                if param_idx < len(all_params):
                    param_map[tensor_idx] = all_params[param_idx]
                    tensor_idx += 1
                    param_idx += 1

        return param_map

    def _build_tensor_param_map(self, tensor_params: List[str], tensor_list_params: List[str],
                                 original_shapes: Optional[List]) -> Dict[int, str]:
        """构建 tensor 参数位置映射表

        根据 original_shapes 中的 null 位置，确定每个 tensor 应映射到哪个参数。

        Args:
            tensor_params: tensor 参数名列表（按函数签名顺序）
            tensor_list_params: tensor list 参数名列表（按函数签名顺序）
            original_shapes: 原始 input_shapes（包含 null）

        Returns:
            字典 {tensor_index: param_name}，表示第几个 tensor 应传给哪个参数
        """
        # 合并所有 tensor 参数（tensor 和 tensor_list 按签名顺序）
        all_tensor_params = []
        for name in tensor_params + tensor_list_params:
            all_tensor_params.append(name)

        if original_shapes is None:
            # 没有 original_shapes 信息，按顺序映射
            return {i: name for i, name in enumerate(all_tensor_params)}

        # 处理 input_shapes 格式
        shapes_to_check = self._normalize_input_shapes(original_shapes)
        if shapes_to_check is None:
            return {i: name for i, name in enumerate(all_tensor_params)}

        # 遍历 shapes，跳过 null 位置
        param_map = {}
        tensor_idx = 0
        param_idx = 0

        for position, shape_item in enumerate(shapes_to_check):
            is_null = shape_item is None

            # 判断是否为嵌套结构：元素是 list 且第一个元素也是 list（不是 int）
            is_nested = (isinstance(shape_item, list) and shape_item and
                         isinstance(shape_item[0], list) and shape_item[0] and
                         not isinstance(shape_item[0][0], int))

            if is_nested:
                # 嵌套结构（如 [[shape1], [shape2]]）：递归处理
                sub_map = self._build_tensor_param_map(
                    all_tensor_params[param_idx:] if param_idx < len(all_tensor_params) else [],
                    [],
                    shape_item
                )
                for t_idx, p_name in sub_map.items():
                    if p_name in all_tensor_params:
                        param_map[tensor_idx + t_idx] = p_name
                        param_idx = all_tensor_params.index(p_name) + 1
                tensor_idx += len(sub_map)
            elif is_null:
                # null 位置：跳过对应的参数（这个参数不传入 tensor）
                if param_idx < len(all_tensor_params):
                    param_idx += 1
            else:
                # 有效 shape（包括 [N, C, H, W] 这种 shape）：映射当前 tensor 到当前参数
                if param_idx < len(all_tensor_params):
                    param_map[tensor_idx] = all_tensor_params[param_idx]
                    tensor_idx += 1
                    param_idx += 1

        return param_map

    def _normalize_input_shapes(self, original_shapes: List) -> Optional[List]:
        """规范化 input_shapes 格式

        处理 CaseLoader 包装的格式，返回统一的扁平 shapes 列表。
        支持以下格式：
        1. 扁平格式: [shape1, shape2, ...]
        2. 两层包装: [[shape1, shape2, ...]] -> [shape1, shape2, ...]
        3. 三层包装: [[[shape1]], [[shape2]], ...] -> [[shape1], [shape2], ...]
        """
        if not isinstance(original_shapes, list) or not original_shapes:
            return None

        first = original_shapes[0]

        # 判断是否是三层嵌套：[[[shape1]], [[shape2]], ...]
        # 特征：first 是 list，且 first[0] 是 list，且 first[0][0] 是 int（shape 的元素）
        if isinstance(first, list) and first:
            inner_first = first[0] if first else None
            if isinstance(inner_first, list) and inner_first:
                # 检查 inner_first 是否是 shape（元素是 int）
                if isinstance(inner_first[0], int):
                    # 三层嵌套，展平每个子列表
                    # [[[shape1]], [[shape2]], ...] -> [[shape1], [shape2], ...]
                    return [item[0] if isinstance(item, list) and len(item) == 1 else item for item in original_shapes]
                # inner_first 是 list 但不是 shape，说明是两层包装
                # [[shape1, shape2, ...]] -> [shape1, shape2, ...]
                return first
            elif inner_first is None:
                # null 位置，保持原样
                return first

        # 已经是扁平格式：[shape1, shape2, ...]
        return original_shapes

    def _convert_value(self, value: Any) -> Any:
        """转换特殊值"""
        if isinstance(value, str):
            if value == 'inf':
                return float('inf')
            elif value == '-inf':
                return float('-inf')
            elif value == 'nan':
                return float('nan')
            try:
                return float(value)
            except ValueError:
                pass
        return value