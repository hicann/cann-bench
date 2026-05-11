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
4. 按 proto.yaml inputs 顺序直接映射（规范格式）
"""

from typing import Dict, List, Any, Callable, Optional
from inspect import signature, Parameter


class ParamBuilder:
    """参数构建器"""

    def __init__(self, importer=None):
        self.importer = importer

    def build_params_by_proto_order(
        self,
        input_tensors: List[Any],
        proto_inputs: List[Any],
        case_attrs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """按 proto.yaml inputs 顺序直接构建参数（规范格式）

        适用场景：cases.yaml 已规范化，使用 null 占位符

        Args:
            input_tensors: 已生成的输入张量列表（包含 None 占位）
            proto_inputs: proto.yaml 中定义的 inputs 列表
            case_attrs: 用例属性字典

        Returns:
            参数字典，按 proto_inputs 顺序映射

        规范要求：
            - input_tensors 长度必须与 proto_inputs 长度一致
            - 省略的 optional 参数用 None 占位
        """
        params = {}
        case_attrs = case_attrs or {}

        # 检查长度一致性
        if len(input_tensors) != len(proto_inputs):
            # 兼容旧格式：长度不一致时按实际 tensor 数量映射
            return self._build_params_legacy(input_tensors, proto_inputs, case_attrs)

        # 按 proto.yaml inputs 顺序直接映射
        for i, input_info in enumerate(proto_inputs):
            input_name = input_info.name
            tensor = input_tensors[i]

            # None 表示 optional 参数省略
            if tensor is None:
                params[input_name] = None
            else:
                params[input_name] = tensor

        # 添加属性参数
        for attr_key, attr_val in case_attrs.items():
            if attr_key not in params:
                params[attr_key] = attr_val

        return params

    def _build_params_legacy(
        self,
        input_tensors: List[Any],
        proto_inputs: List[Any],
        case_attrs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """兼容旧格式的参数构建（cases.yaml 无 null 占位）"""
        params = {}

        # 统计实际 tensor 数量
        actual_tensors = [t for t in input_tensors if t is not None]
        proto_tensor_count = len([i for i in proto_inputs if 'Tensor' in str(i.dtype) or isinstance(i.dtype, list)])

        # 如果长度一致，直接映射
        if len(actual_tensors) == proto_tensor_count:
            tensor_idx = 0
            for input_info in proto_inputs:
                if 'Tensor' in str(input_info.dtype) or isinstance(input_info.dtype, list):
                    if tensor_idx < len(input_tensors):
                        params[input_info.name] = input_tensors[tensor_idx]
                        tensor_idx += 1
                    else:
                        params[input_info.name] = None
                # optional 参数未提供
                elif getattr(input_info, 'optional', False):
                    params[input_info.name] = None
        else:
            # 复杂映射：按签名顺序处理
            tensor_idx = 0
            for input_info in proto_inputs:
                if tensor_idx < len(input_tensors):
                    params[input_info.name] = input_tensors[tensor_idx]
                    tensor_idx += 1
                else:
                    params[input_info.name] = None

        # 添加属性参数
        for attr_key, attr_val in case_attrs.items():
            if attr_key not in params:
                params[attr_key] = attr_val

        return params

    def build_call_params(self, golden_func: Callable, case: Any, input_tensors: List, override_shapes: List = None) -> Dict[str, Any]:
        """构建golden函数调用参数

        Args:
            golden_func: Golden 函数
            case: 测试用例信息（包含 input_shapes, attrs 等）
            input_tensors: 已生成的输入张量列表（不含 null 位置的 tensor）
            override_shapes: 可选的形状覆盖（用于 get_input 后重新排序的情况）

        Returns:
            参数字典，用于调用 golden_func
        """
        sig = signature(golden_func)
        params = {}

        # 分类参数：tensor 参数、tensor list 参数、属性参数
        # 同时记录签名顺序的完整 tensor 参数列表
        tensor_params = []
        tensor_list_params = []
        optional_tensor_params = []
        optional_tensor_list_params = []
        attr_params = []
        # 按签名顺序记录所有 tensor 参数
        all_tensor_params_in_order = []

        for name, param in sig.parameters.items():
            annotation = str(param.annotation) if param.annotation != Parameter.empty else ""
            is_optional = 'Optional' in annotation or 'NoneType' in annotation or param.default != Parameter.empty

            if 'List[' in annotation and 'Tensor' in annotation:
                all_tensor_params_in_order.append(name)
                if is_optional:
                    optional_tensor_list_params.append(name)
                else:
                    tensor_list_params.append(name)
            elif 'Tensor' in annotation:
                all_tensor_params_in_order.append(name)
                if is_optional:
                    optional_tensor_params.append(name)
                else:
                    tensor_params.append(name)
            else:
                attr_params.append(name)

        # 使用 override_shapes 或原始 input_shapes
        original_shapes = override_shapes if override_shapes is not None else getattr(case, 'input_shapes', None)

        # 构建 tensor 参数映射表：位置 -> 参数名
        # 按签名顺序映射
        tensor_param_map = self._build_tensor_param_map(
            tensor_params, tensor_list_params,
            original_shapes,
            optional_tensor_params, optional_tensor_list_params,
            all_tensor_params_in_order  # 传入签名顺序
        )

        # 匹配张量参数
        tensor_idx = 0
        for position, param_name in sorted(tensor_param_map.items(), key=lambda x: x[0]):
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

    def _build_tensor_param_map(
            self,
            tensor_params: List[str],
            tensor_list_params: List[str],
            original_shapes: Optional[List],
            optional_tensor_params: List[str] = None,
            optional_tensor_list_params: List[str] = None,
            all_tensor_params_in_order: List[str] = None
        ) -> Dict[int, str]:
        """构建 tensor 参数位置映射表，按函数签名顺序映射

        Args:
            tensor_params: 必需 tensor 参数名列表
            tensor_list_params: 必需 tensor list 参数名列表
            original_shapes: 原始 input_shapes
            optional_tensor_params: optional tensor 参数名列表（可选）
            optional_tensor_list_params: optional tensor list 参数名列表（可选）
            all_tensor_params_in_order: 按签名顺序的所有 tensor 参数名列表

        Returns:
            字典 {tensor_index: param_name}
        """
        # 处理 optional 参数默认值
        optional_tensor_params = optional_tensor_params or []
        optional_tensor_list_params = optional_tensor_list_params or []
        all_tensor_params_in_order = all_tensor_params_in_order or []

        # 所有 tensor 参数，按签名顺序排列
        required_params = tensor_params + tensor_list_params
        optional_params = optional_tensor_params + optional_tensor_list_params

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

        # 按签名顺序映射：直接用 all_tensor_params_in_order
        param_map = {}
        for i, shape_item in enumerate(shapes_to_check):
            if i < len(all_tensor_params_in_order):
                param_map[i] = all_tensor_params_in_order[i]

        return param_map

    def _normalize_input_shapes(self, original_shapes: List) -> Optional[List]:
        """规范化 input_shapes 格式

        处理 CaseLoader 包装的格式，返回统一的扁平 shapes 列表。

        关键判断：
        - 如果 original_shapes 只有1个元素且内部是多个shape，可能是包装结构，需要展开
        - 如果 original_shapes 有多个元素，外层结构是真实的参数结构，不应该展开
        """
        if not isinstance(original_shapes, list) or not original_shapes:
            return None

        # 只有当 original_shapes 只有1个元素时，才可能是包装结构
        if len(original_shapes) == 1:
            first = original_shapes[0]
            if isinstance(first, list) and first:
                # 检查 first 的第一个元素是否是 list（shape）或 None
                inner_first = first[0] if first else None
                if isinstance(inner_first, list) or inner_first is None:
                    # 被包装，展开
                    return first

        # 已经是扁平格式或多参数结构：保持原样
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