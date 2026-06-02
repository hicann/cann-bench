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
StanfordBench 评测集实现

包含：
- Loader: StanfordTaskLoader, StanfordCaseLoader, StanfordGoldenLoader
- Checker: AllCloseChecker
- Matcher: StanfordMatcher
- Scoring: StanfordScoringScheme

使用方式：
    import kernel_eval.benches  # 自动注册所有组件
    from kernel_eval.benches.stanford import StanfordTaskLoader
"""

# === Stanford 特化组件 ===
from .stanford_loader import StanfordTaskLoader, StanfordCaseLoader, StanfordGoldenLoader
from ..checkers.allclose_checker import AllCloseChecker, AllCloseOutputResult
from .stanford_matcher import StanfordMatcher
from .stanford_scoring import StanfordScoringScheme

__all__ = [
    # Loader
    "StanfordTaskLoader",
    "StanfordCaseLoader",
    "StanfordGoldenLoader",
    # Checker
    "AllCloseChecker",
    "AllCloseOutputResult",
    # Matcher
    "StanfordMatcher",
    # Scoring
    "StanfordScoringScheme",
]


# === 注册到 Registry ===

_STANFORD_REGISTERED = False

def _register_stanford_components():
    """注册 StanfordBench 特化组件到 Registry（幂等）"""
    global _STANFORD_REGISTERED
    if _STANFORD_REGISTERED:
        return

    from ..registry.loader_registry import LoaderRegistry
    from ..registry.golden_registry import GoldenLoaderRegistry
    from ..registry.matcher_registry import OperatorMatcherRegistry
    from ..registry.checker_registry import CheckerRegistry
    from ..registry.scoring_registry import ScoringSchemeRegistry
    from ..registry.bench_registry import BenchRegistry, BenchConfig
    from ..registry.case_spec_registry import CaseSpecRegistry

    # 注册 Loader
    if 'stanford' not in LoaderRegistry._task_loaders:
        LoaderRegistry.register_task_loader('stanford', StanfordTaskLoader)
    if 'stanford' not in LoaderRegistry._case_loaders:
        LoaderRegistry.register_case_loader('stanford', StanfordCaseLoader)

    # 注册 GoldenLoader
    if 'stanford' not in GoldenLoaderRegistry._items:
        GoldenLoaderRegistry.register('stanford', StanfordGoldenLoader)

    # 注册 OperatorMatcher
    if 'stanford' not in OperatorMatcherRegistry._matchers:
        OperatorMatcherRegistry.register('stanford', StanfordMatcher)

    # 注册 Checker
    if 'allclose' not in CheckerRegistry.get_all():
        CheckerRegistry.register('allclose', AllCloseChecker())

    # 注册 ScoringScheme
    if 'stanford' not in ScoringSchemeRegistry._items:
        ScoringSchemeRegistry.register('stanford', StanfordScoringScheme())

    # 注册 BenchConfig
    if 'stanford' not in BenchRegistry._items:
        # 注册 CaseSpec（Stanford 用基类）
        if 'stanford' not in CaseSpecRegistry._items:
            from ..base.models import CaseSpec
            CaseSpecRegistry.register('stanford', CaseSpec)

        BenchRegistry.register('stanford', BenchConfig(
            task_loader='stanford',
            case_loader='stanford',
            golden_loader='stanford',
            operator_matcher='stanford',
            scoring_scheme='stanford',
            checker='allclose',
            case_spec_cls='stanford',
            golden_precision='native_npu',
            precision_thresholds={'atol': 0.01, 'rtol': 0.01},
            default_tasks_root='thirdparty/KernelBench/KernelBench',
            description='StanfordBench 评测集 - Scaling Intelligence',
            metadata={
                'solution_file': 'ai_op.py',
                'backend': 'npu',
            },
        ))

    _STANFORD_REGISTERED = True


# 执行注册
_register_stanford_components()