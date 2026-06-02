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
评测集配置注册表

BenchRegistry: BenchConfig 注册表
BenchConfig: 评测集配置数据类

Why: 简化 CLI 使用，一个 --bench-name 参数即可完成所有组件配置
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base.loaders import TaskLoader, CaseLoader
from ..base.scoring import ScoringScheme
from ..base.checker import CorrectnessChecker
from .base import BaseRegistry
from .loader_registry import get_task_loader, get_case_loader
from .golden_registry import get_golden_loader
from .matcher_registry import get_operator_matcher
from .checker_registry import get_correctness_checker
from .scoring_registry import get_scoring_scheme
from .case_spec_registry import CaseSpecRegistry


@dataclass
class BenchConfig:
    """评测集配置

    定义一个评测集所需的全部组件配置。
    """
    name: str = ""
    task_loader: str = ""
    case_loader: str = ""
    golden_loader: str = "cann"
    operator_matcher: str = "cann"
    scoring_scheme: str = ""
    checker: str = "relative_error"
    case_spec_cls: str = "cann"                    # CaseSpec 子类标识
    # Golden 参考输出的精度策略：
    #   fp64_cpu（默认）: 升精度到 float64 + CPU 计算，避免 NPU 溢出污染
    #   native_cpu: 保持原始精度在 CPU 上计算
    #   native_npu: 保持原始精度在 NPU 上计算
    golden_precision: str = "fp64_cpu"
    precision_thresholds: Dict[str, float] = field(default_factory=dict)
    default_tasks_root: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_task_loader(self, **kwargs) -> TaskLoader:
        """获取 TaskLoader 实例"""
        return get_task_loader(self.task_loader, **kwargs)

    def get_case_loader(self, **kwargs) -> CaseLoader:
        """获取 CaseLoader 实例"""
        return get_case_loader(self.case_loader, **kwargs)

    def get_golden_loader(self, **kwargs):
        """获取 GoldenLoader 实例"""
        return get_golden_loader(self.golden_loader, **kwargs)

    def get_operator_matcher(self, operator_loader=None):
        """获取 OperatorMatcher 实例"""
        return get_operator_matcher(self.operator_matcher, operator_loader)

    def get_scoring_scheme(self) -> Optional[ScoringScheme]:
        """获取评分方案实例"""
        return get_scoring_scheme(self.scoring_scheme)

    def get_checker(self) -> Optional[CorrectnessChecker]:
        """获取精度判断器实例"""
        return get_correctness_checker(self.checker)

    def get_case_spec_cls(self):
        """获取 CaseSpec 子类"""
        return CaseSpecRegistry.get(self.case_spec_cls)

    def get_precision_thresholds(self) -> Dict[str, float]:
        """获取精度阈值表"""
        if self.precision_thresholds:
            return self.precision_thresholds
        from ..utils.thresholds import PRECISION_THRESHOLDS
        return dict(PRECISION_THRESHOLDS)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'task_loader': self.task_loader,
            'case_loader': self.case_loader,
            'golden_loader': self.golden_loader,
            'operator_matcher': self.operator_matcher,
            'scoring_scheme': self.scoring_scheme,
            'checker': self.checker,
            'case_spec_cls': self.case_spec_cls,
            'golden_precision': self.golden_precision,
            'precision_thresholds': self.precision_thresholds,
            'default_tasks_root': self.default_tasks_root,
            'description': self.description,
            'metadata': self.metadata,
        }


class BenchRegistry(BaseRegistry[BenchConfig]):
    """评测集注册表"""

    _items: Dict[str, BenchConfig] = {}

    @classmethod
    def register(cls, name: str, config: BenchConfig) -> None:
        """注册评测集配置"""
        if name in cls._items:
            raise ValueError(f"评测集 '{name}' 已注册")
        config.name = name
        cls._items[name] = config

    @classmethod
    def get(cls, name: str) -> Optional[BenchConfig]:
        """获取评测集配置"""
        return cls._items.get(name)

    @classmethod
    def list_benches(cls) -> List[str]:
        """列出已注册的评测集"""
        return list(cls._items.keys())

    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册名称"""
        return cls.list_benches()

    @classmethod
    def get_default(cls) -> Optional[BenchConfig]:
        """获取默认评测集配置"""
        return cls._items.get('cann')

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查评测集是否已注册"""
        return name in cls._items

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._items.clear()


def get_bench_config(bench_name: str = 'cann') -> BenchConfig:
    """获取评测集配置"""
    config = BenchRegistry.get(bench_name)
    if config is None:
        registered = BenchRegistry.list_benches()
        raise ValueError(f"评测集 '{bench_name}' 未注册，已注册: {registered}")
    return config


def get_bench_components(bench_name: str = 'cann', tasks_root: str = None) -> Dict[str, Any]:
    """获取评测集的所有组件实例"""
    config = get_bench_config(bench_name)

    loader_kwargs = {}
    if tasks_root:
        loader_kwargs['tasks_root'] = tasks_root
    elif config.default_tasks_root:
        loader_kwargs['tasks_root'] = config.default_tasks_root

    golden_kwargs = {}
    if tasks_root:
        golden_kwargs['bench_root'] = tasks_root
    elif config.default_tasks_root:
        golden_kwargs['bench_root'] = config.default_tasks_root

    operator_loader = config.get_task_loader(**loader_kwargs)

    return {
        'task_loader': config.get_task_loader(**loader_kwargs),
        'case_loader': config.get_case_loader(**loader_kwargs),
        'golden_loader': config.get_golden_loader(**golden_kwargs),
        'operator_matcher': config.get_operator_matcher(operator_loader),
        'scoring_scheme': config.get_scoring_scheme(),
        'checker': config.get_checker(),
        'precision_thresholds': config.get_precision_thresholds(),
    }