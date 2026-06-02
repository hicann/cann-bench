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
评测集插件模块

职责：
- 聚合各评测集的特化实现
- 自动加载所有已安装的评测集
- 提供统一的导入入口

目录设计：
- benches/cann.py: CANN 评测集特化（导出 + 注册）
- benches/cann_*.py: CANN 特化组件文件
- checkers/relative_error_checker.py: 相对误差判断器（通用标准）
- checkers/allclose_checker.py: AllClose 判断器（通用标准）

使用方式：
    import kernel_eval.benches  # 自动加载所有评测集并注册
    from kernel_eval.benches.cann import CannTaskLoader  # 推荐
"""

# 导入所有评测集模块，触发注册
from . import cann
from . import stanford

# 重新导出 CANN 组件（便于直接从 benches 导入）
from .cann import (
    # Loader
    CannTaskLoader,
    CannCaseLoader,
    GoldenLoader,
    # Models
    CannTaskSpec,
    CannCaseSpec,
    CannInputSpec,
    CannOutputSpec,
    CannSolutionSpec,
    # Matcher
    OperatorMatcher,
    # Scoring
    CannScoringScheme,
    SimpleComparisonScheme,
    RecordingOnlyScheme,
    ScoringCalculator,
    OperatorScoreInfo,
    per_case_sol_score,
    aggregate_eq4,
)

# 重新导出 Checker + OutputResult（通用标准，不与评测集绑定）
from ..checkers.relative_error_checker import RelativeErrorChecker, RelativeErrorOutputResult
from ..checkers.allclose_checker import AllCloseChecker, AllCloseOutputResult

# 重新导出 Stanford 组件
from .stanford import (
    # Loader
    StanfordTaskLoader,
    StanfordCaseLoader,
    StanfordGoldenLoader,
    # Matcher
    StanfordMatcher,
    # Scoring
    StanfordScoringScheme,
)

__all__ = [
    'cann',
    'stanford',
    # CANN Loader
    "CannTaskLoader",
    "CannCaseLoader",
    "GoldenLoader",
    # CANN Models
    "CannTaskSpec",
    "CannCaseSpec",
    "CannInputSpec",
    "CannOutputSpec",
    "CannSolutionSpec",
    # CANN Matcher
    "OperatorMatcher",
    # CANN Scoring
    "CannScoringScheme",
    "SimpleComparisonScheme",
    "RecordingOnlyScheme",
    "ScoringCalculator",
    "OperatorScoreInfo",
    "per_case_sol_score",
    "aggregate_eq4",
    # Checker + OutputResult（通用标准）
    "RelativeErrorChecker",
    "RelativeErrorOutputResult",
    "AllCloseChecker",
    "AllCloseOutputResult",
    # Stanford Loader
    "StanfordTaskLoader",
    "StanfordCaseLoader",
    "StanfordGoldenLoader",
    # Stanford Matcher
    "StanfordMatcher",
    # Stanford Scoring
    "StanfordScoringScheme",
]