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
报告层模块

职责：
1. 评测报告生成（JSON + Markdown）
2. Summary生成（几何平均加速比）
3. 评分计算（功能得分 + 性能得分）
"""

from .report_generator import ReportGenerator, EvalResult
from .scoring import ScoringCalculator, ScoreInfo
from .summary_generator import (
    EvaluationSummary, OperatorSummary,
    calculate_geometric_mean, generate_summary, render_summary_markdown, save_summary,
)

__all__ = [
    "ReportGenerator", "EvalResult",
    "ScoringCalculator", "ScoreInfo",
    "EvaluationSummary", "OperatorSummary",
    "calculate_geometric_mean", "generate_summary", "render_summary_markdown", "save_summary",
]