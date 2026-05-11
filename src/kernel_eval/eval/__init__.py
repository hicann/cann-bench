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
评测层模块

职责：
1. 算子执行器（设备迁移、函数执行）
2. 精度评测（Golden对比验证、CPU fp64、二次验证）
3. 性能评测（Profiler kernel-only、升频清cache）
4. 输入池管理（防缓存攻击）
5. 综合评测调度（协调精度和性能评测）
6. 子进程执行（算子级进程隔离）
7. 失败结果合成（编译/安全/子进程失败）
8. 算子匹配（名称查找、AI算子加载）
9. 结果统计公共函数
"""

from .op_runner import OpRunner, OpRunResult
from .accuracy_eval import AccuracyEvaluator, AccuracyResult
from .perf_eval import PerfEvaluator, PerfResult
from .input_pool import InputPool, InputPoolConfig, create_input_pool
from .results import EvalCaseResult, EvalOperatorResult, EvalSessionResult, summarize_case_results, CaseResultSummary
from .failure_synthesizer import FailureSynthesizer
from .operator_matcher import OperatorMatcher
from .subprocess_runner import SubprocessRunner
from .evaluator import Evaluator

__all__ = [
    "OpRunner", "OpRunResult",
    "AccuracyEvaluator", "AccuracyResult",
    "PerfEvaluator", "PerfResult",
    "InputPool", "InputPoolConfig", "create_input_pool",
    "EvalCaseResult", "EvalOperatorResult", "EvalSessionResult",
    "summarize_case_results", "CaseResultSummary",
    "FailureSynthesizer",
    "OperatorMatcher",
    "SubprocessRunner",
    "Evaluator",
]