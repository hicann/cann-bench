#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
CANN 评测集特化实现

包含：
- Loader: CannTaskLoader, CannCaseLoader, GoldenLoader
- Models: CannTaskSpec, CannCaseSpec, CannInputSpec, CannOutputSpec, CannSolutionSpec
- Checker: RelativeErrorChecker
- Matcher: OperatorMatcher
- Scoring: CannScoringScheme, SimpleComparisonScheme, RecordingOnlyScheme

使用方式：
    import kernel_eval.benches  # 自动注册所有组件
    from kernel_eval.benches.cann import CannTaskLoader, CannTaskSpec
"""

# === CANN 特化组件 ===
from .cann_loader import CannTaskLoader, CannCaseLoader, GoldenLoader
from .cann_spec import CannTaskSpec, CannCaseSpec, CannInputSpec, CannOutputSpec
from .cann_solution import CannSolutionSpec
from ..checkers.relative_error_checker import RelativeErrorChecker, RelativeErrorOutputResult
from .cann_matcher import OperatorMatcher
from .cann_scoring import (
    CannScoringScheme,
    SimpleComparisonScheme,
    RecordingOnlyScheme,
    ScoringCalculator,
    OperatorScoreInfo,
    per_case_sol_score,
    aggregate_eq4,
)

__all__ = [
    # Loader
    "CannTaskLoader",
    "CannCaseLoader",
    "GoldenLoader",
    # Models
    "CannTaskSpec",
    "CannCaseSpec",
    "CannInputSpec",
    "CannOutputSpec",
    "CannSolutionSpec",
    # Checker
    "RelativeErrorChecker",
    "RelativeErrorOutputResult",
    # Matcher
    "OperatorMatcher",
    # Scoring
    "CannScoringScheme",
    "SimpleComparisonScheme",
    "RecordingOnlyScheme",
    "ScoringCalculator",
    "OperatorScoreInfo",
    "per_case_sol_score",
    "aggregate_eq4",
]


# === 注册到 Registry ===

_CANN_REGISTERED = False

def _register_cann_components():
    """注册 CANN 特化组件到 Registry（幂等）"""
    global _CANN_REGISTERED
    if _CANN_REGISTERED:
        return

    from ..registry.loader_registry import LoaderRegistry
    from ..registry.golden_registry import GoldenLoaderRegistry
    from ..registry.matcher_registry import OperatorMatcherRegistry
    from ..registry.checker_registry import CheckerRegistry
    from ..registry.scoring_registry import ScoringSchemeRegistry
    from ..registry.bench_registry import BenchRegistry, BenchConfig
    from ..registry.case_spec_registry import CaseSpecRegistry
    from ..utils.thresholds import PRECISION_THRESHOLDS

    # 注册 Loader
    if 'cann' not in LoaderRegistry._task_loaders:
        LoaderRegistry.register_task_loader('cann', CannTaskLoader)
    if 'cann' not in LoaderRegistry._case_loaders:
        LoaderRegistry.register_case_loader('cann', CannCaseLoader)

    # 注册 GoldenLoader
    if 'cann' not in GoldenLoaderRegistry._items:
        GoldenLoaderRegistry.register('cann', GoldenLoader)

    # 注册 OperatorMatcher
    if 'cann' not in OperatorMatcherRegistry._matchers:
        OperatorMatcherRegistry.register('cann', OperatorMatcher)

    # 注册 Checker
    if 'relative_error' not in CheckerRegistry.get_all():
        checker = RelativeErrorChecker()
        CheckerRegistry.register('relative_error', checker)
        CheckerRegistry.register('cann_default', checker)  # 兼容旧名

    # 注册 ScoringScheme
    if 'cann' not in ScoringSchemeRegistry._items:
        ScoringSchemeRegistry.register('cann', CannScoringScheme())
    if 'simple_comparison' not in ScoringSchemeRegistry._items:
        ScoringSchemeRegistry.register('simple_comparison', SimpleComparisonScheme())
    if 'recording_only' not in ScoringSchemeRegistry._items:
        ScoringSchemeRegistry.register('recording_only', RecordingOnlyScheme())

    # 注册 BenchConfig
    if 'cann' not in BenchRegistry._items:
        # 注册 CaseSpec 子类
        if 'cann' not in CaseSpecRegistry._items:
            CaseSpecRegistry.register('cann', CannCaseSpec)

        BenchRegistry.register('cann', BenchConfig(
            task_loader='cann',
            case_loader='cann',
            golden_loader='cann',
            operator_matcher='cann',
            scoring_scheme='cann',
            checker='relative_error',
            case_spec_cls='cann',
            precision_thresholds=dict(PRECISION_THRESHOLDS),
            default_tasks_root='tasks',
            description='CANN NPU 算子评测集（默认）',
            metadata={
                'backend': 'npu',
                'profiler': 'torch_npu.profiler',
            },
        ))

    _CANN_REGISTERED = True


# 执行注册
_register_cann_components()