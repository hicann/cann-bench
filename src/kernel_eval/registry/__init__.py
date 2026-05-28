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
注册层模块

职责：
- 提供所有组件的注册表
- 定义统一的注册接口
- 提供便捷获取函数

目录设计：
- registry/ 目录只包含注册表定义
- 无任何特化类 import（注册由 benches/ 执行）
"""

# === Base Registry ===
from .base import BaseRegistry

# === Loader Registry ===
from .loader_registry import LoaderRegistry, get_task_loader, get_case_loader

# === Golden Loader Registry ===
from .golden_registry import GoldenLoaderRegistry, get_golden_loader

# === Operator Matcher Registry ===
from .matcher_registry import OperatorMatcherRegistry, get_operator_matcher

# === Checker Registry ===
from .checker_registry import CheckerRegistry, get_correctness_checker, register_correctness_checker

# === Scoring Registry ===
from .scoring_registry import ScoringSchemeRegistry, get_scoring_scheme

# === CaseSpec Registry ===
from .case_spec_registry import CaseSpecRegistry

# === Bench Registry ===
from .bench_registry import BenchRegistry, BenchConfig, get_bench_config, get_bench_components

__all__ = [
    # Base Registry
    "BaseRegistry",
    # Loader Registry
    "LoaderRegistry",
    "get_task_loader",
    "get_case_loader",
    # Golden Loader Registry
    "GoldenLoaderRegistry",
    "get_golden_loader",
    # Operator Matcher Registry
    "OperatorMatcherRegistry",
    "get_operator_matcher",
    # Checker Registry
    "CheckerRegistry",
    "get_correctness_checker",
    "register_correctness_checker",
    # Scoring Registry
    "ScoringSchemeRegistry",
    "get_scoring_scheme",
    # CaseSpec Registry
    "CaseSpecRegistry",
    # Bench Registry
    "BenchRegistry",
    "BenchConfig",
    "get_bench_config",
    "get_bench_components",
]