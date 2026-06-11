#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
算子匹配器

职责：
1. 加载 AI 算子函数（从 torch.ops.cann_bench 或 cann_bench 模块）
2. 查找算子定义信息
3. 通过 snake_case 名称反查算子

继承 OperatorMatcherBase，适配 cann-bench 评测体系的 AI 算子来源。
"""

from typing import Callable, Dict, List, Optional

from .cann_loader import CannTaskLoader
from .cann_spec import CannTaskSpec
from ..utils.naming import snake_case_candidates
from ..base.matcher import OperatorMatcherBase


class OperatorMatcher(OperatorMatcherBase):
    """算子匹配器（cann-bench 评测体系）

    从 torch.ops.cann_bench 或 cann_bench 模块加载 AI 算子函数。
    """

    def __init__(self, operator_loader: CannTaskLoader):
        self.operator_loader = operator_loader
        self._ai_op_cache: Dict[str, Callable] = {}

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载AI生成的算子函数

        查找顺序：
        1. torch.ops.cann_bench（golden whl 注册位置）
        2. cann_bench 模块（submission whl）

        Args:
            operator_name: 算子名称（PascalCase）

        Returns:
            算子函数

        Raises:
            AttributeError: 无法找到算子
        """
        cache_key = operator_name.lower()
        if cache_key in self._ai_op_cache:
            return self._ai_op_cache[cache_key]

        candidates = snake_case_candidates(operator_name) + [
            operator_name.lower(),
            operator_name,
        ]
        schema_fn = self._get_schema_function_name(operator_name)
        if schema_fn and schema_fn not in candidates:
            candidates.append(schema_fn)

        # 1. 先尝试 torch.ops.cann_bench（golden whl）
        try:
            import torch
            if hasattr(torch.ops, 'cann_bench'):
                for name in candidates:
                    if hasattr(torch.ops.cann_bench, name):
                        func = getattr(torch.ops.cann_bench, name)
                        self._ai_op_cache[cache_key] = func
                        return func
        except (AttributeError, ImportError):
            # 仅吞掉"模块未安装"或"属性不存在"——其它异常应继续向外冒泡。
            pass

        # 2. 再尝试 cann_bench 模块（submission whl）
        try:
            import cann_bench
            for name in candidates:
                if hasattr(cann_bench, name):
                    func = getattr(cann_bench, name)
                    self._ai_op_cache[cache_key] = func
                    return func
        except ImportError:
            pass

        raise AttributeError(f"无法找到算子 {operator_name}（已检查 torch.ops.cann_bench 和 cann_bench 模块）")

    def find_operator_info(self, operator_name: str) -> Optional[CannTaskSpec]:
        """查找算子定义信息

        Args:
            operator_name: 算子名称

        Returns:
            CannTaskSpec 或 None
        """
        operators = self.operator_loader.list_operators()
        for op_info in operators:
            if op_info.name == operator_name:
                return op_info
        return None

    def find_operator_info_by_snake(self, snake_name: str) -> Optional[CannTaskSpec]:
        """通过 snake_case 名称反查 CannTaskSpec

        与 load_ai_operator 的 CamelCase→snake_case 规则保持一致。

        Args:
            snake_name: snake_case 形式的算子名（build_submission 里的 op 目录名）

        Returns:
            CannTaskSpec 或 None
        """
        target = snake_name.lower()
        operators = self.operator_loader.list_operators()
        for op_info in operators:
            if target in snake_case_candidates(op_info.name):
                return op_info
        return None

    def _get_schema_function_name(self, operator_name: str) -> Optional[str]:
        op_info = self.find_operator_info(operator_name)
        if op_info is not None:
            fn = op_info.get_function_name()
            if fn:
                return fn
        return None

    def clear_cache(self):
        """清空算子缓存"""
        self._ai_op_cache.clear()
