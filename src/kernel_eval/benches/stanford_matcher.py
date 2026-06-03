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
StanfordBench 算子匹配器

职责：
1. 从 source_dir/ai_op.py 加载 AI 算子（Model/ModelNew 类）
2. 查找算子定义信息
3. 支持 source_dir 动态设置

继承 OperatorMatcherBase，适配 StanfordBench 评测体系的 AI 算子来源。
"""

import importlib.util
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from ..base.matcher import OperatorMatcherBase
from ..base.models import TaskSpec
from .stanford_loader import StanfordTaskLoader, StanfordGoldenLoader


class StanfordMatcher(OperatorMatcherBase):
    """StanfordBench 算子匹配器

    从 source_dir/ai_op.py 加载 AI 算子函数。
    支持 Model 和 ModelNew 两种类名。
    """

    def __init__(
        self,
        operator_loader: StanfordTaskLoader = None,
        source_dir: str = None,
        random_seed: int = 42
    ):
        self.operator_loader = operator_loader
        self.source_dir = source_dir
        self._random_seed = random_seed
        self._cache: Dict[str, Callable] = {}

    def set_source_dir(self, source_dir: str):
        """设置 AI 算子源码目录"""
        self.source_dir = source_dir
        self._cache.clear()

    def clear_cache(self):
        """清空算子缓存"""
        self._cache.clear()

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载 AI 算子函数

        加载策略：
        1. 若 source_dir 已设置：从 source_dir/ai_op.py 加载 ModelNew/Model 类
        2. 若 source_dir 未设置：回退到 KernelBench 原始 Model（自验证模式）

        自验证模式下，AI 算子与 golden 使用同一个 Model，
        精度应为 100%，用于验证评测框架本身是否正常工作。

        Args:
            operator_name: 算子名称（PascalCase，如 Softmax）

        Returns:
            AI 算子函数（device wrapper）

        Raises:
            AttributeError: 无法找到 AI 算子
        """
        torch.manual_seed(self._random_seed)

        cache_key = operator_name.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.source_dir:
            func = self._load_from_source_dir(operator_name)
        else:
            func = self._load_from_kernelbench(operator_name)

        self._cache[cache_key] = func
        return func

    def _load_from_source_dir(self, operator_name: str) -> Callable:
        """从 source_dir/ai_op.py 加载 AI 算子（用户提交的优化实现）"""
        ai_op_path = Path(self.source_dir) / "ai_op.py"
        if not ai_op_path.exists():
            raise AttributeError(f"ai_op.py 不存在: {ai_op_path}")

        spec = importlib.util.spec_from_file_location(
            f"ai_op.{operator_name}",
            ai_op_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_cls = getattr(module, 'ModelNew', None) or getattr(module, 'Model', None)
        if not model_cls:
            raise AttributeError(f"ai_op.py 缺少 Model/ModelNew 类: {ai_op_path}")

        init_inputs = self._load_task_init_inputs(operator_name)
        if init_inputs is None:
            init_inputs = []
            if hasattr(module, 'get_init_inputs'):
                init_inputs = module.get_init_inputs()

        if isinstance(init_inputs, list) and init_inputs:
            model = model_cls(*init_inputs)
        else:
            model = model_cls()

        return StanfordGoldenLoader._make_device_wrapper(model)

    def _load_task_init_inputs(self, operator_name: str):
        """从匹配到的 Stanford task 读取 Model 构造参数。

        source_dir/ai_op.py 只描述提交实现；构造参数属于 benchmark task。
        若当前 matcher 没有 task loader，保留 None 让旧的自包含 ai_op.py 路径兜底。
        """
        if not self.operator_loader:
            return None

        task_spec = self.operator_loader.get_operator_by_name(operator_name)
        if not task_spec:
            return None

        golden_loader = StanfordGoldenLoader(
            bench_root=str(self.operator_loader.bench_root),
            random_seed=self._random_seed,
        )
        return golden_loader.get_init_inputs(task_spec.rel_path)

    def _load_from_kernelbench(self, operator_name: str) -> Callable:
        """从 KernelBench 原始 .py 文件加载 Model（自验证模式）

        自验证模式下，AI 算子与 golden 使用同一个 Model 类，
        精度应为 100%，用于验证评测框架本身是否正常工作。
        """
        if not self.operator_loader:
            raise AttributeError(
                "未设置 source_dir 且 operator_loader 不可用，无法加载 AI 算子"
            )

        task_spec = self.operator_loader.get_operator_by_name(operator_name)
        if not task_spec:
            raise AttributeError(f"未找到算子 {operator_name} 的定义")

        golden_loader = StanfordGoldenLoader(
            bench_root=str(self.operator_loader.bench_root),
            random_seed=self._random_seed,
        )
        return golden_loader.get_golden_function(task_spec.rel_path)

    def find_operator_info(self, operator_name: str) -> Optional[TaskSpec]:
        """查找算子定义信息

        Args:
            operator_name: 算子名称

        Returns:
            TaskSpec 或 None
        """
        if self.operator_loader:
            return self.operator_loader.get_operator_by_name(operator_name)
        return None

    def find_operator_info_by_snake(self, snake_name: str) -> Optional[TaskSpec]:
        """通过 snake_case 名称反查 TaskSpec

        Args:
            snake_name: snake_case 形式的算子名

        Returns:
            TaskSpec 或 None
        """
        # StanfordBench 算子名直接是 PascalCase，snake_case 需要转换
        # 简化处理：直接查找
        return self.find_operator_info(snake_name)
