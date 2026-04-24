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
Summary生成模块

职责：
1. 生成易读的Summary报告（Markdown格式）
2. 计算几何平均加速比
3. 统计通过率和综合得分

参考evaluation/tools/summarize.py
"""

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OperatorSummary:
    """算子评测摘要"""
    operator: str
    level: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    geometric_mean_speedup: float
    mere_avg: float
    mare_avg: float
    # 可选字段：当算子跑不起来时的原因（编译失败 / 子进程超时或崩溃）。
    # 为 None 时渲染普通的 case 表；非空时在小节头下方优先渲染原因块。
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None


@dataclass
class EvaluationSummary:
    """评测总摘要"""
    eval_code: str
    hardware: str
    total_operators: int
    total_cases: int
    total_passed: int
    overall_pass_rate: float
    overall_geometric_mean_speedup: float
    operators: List[OperatorSummary]
    timestamp: str


def calculate_geometric_mean(values: List[float]) -> float:
    """
    计算几何平均值

    Args:
        values: 数值列表（如加速比）

    Returns:
        几何平均值
    """
    if not values:
        return 0.0

    # 过滤无效值
    valid = [v for v in values if v > 0]
    if not valid:
        return 0.0

    # 几何平均 = exp(mean(log(x)))
    return math.exp(sum(math.log(max(v, 1e-9)) for v in valid) / len(valid))


def calculate_operator_summary(op_result: Dict[str, Any]) -> OperatorSummary:
    """
    从算子结果计算摘要

    Args:
        op_result: 算子评测结果（字典格式）

    Returns:
        OperatorSummary
    """
    operator = op_result.get("operator", "")
    level = op_result.get("level", 0)
    total_cases = op_result.get("total_cases", 0)
    passed_cases = op_result.get("passed_cases", 0)
    failed_cases = total_cases - passed_cases
    pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0

    # 计算几何平均加速比
    speedups = []
    meres = []
    mares = []
    for case in op_result.get("results", []):
        if case.get("speedup") and case["speedup"] > 0:
            speedups.append(case["speedup"])
        if case.get("mere"):
            meres.append(case["mere"])
        if case.get("mare"):
            mares.append(case["mare"])

    geometric_mean_speedup = calculate_geometric_mean(speedups)
    mere_avg = sum(meres) / len(meres) if meres else 0.0
    mare_avg = sum(mares) / len(mares) if mares else 0.0

    return OperatorSummary(
        operator=operator,
        level=level,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        geometric_mean_speedup=geometric_mean_speedup,
        mere_avg=mere_avg,
        mare_avg=mare_avg,
        compilation_error=op_result.get("compilation_error"),
        subprocess_failure_reason=op_result.get("subprocess_failure_reason"),
    )


def generate_summary(
    evaluation_results: Dict[str, Any],
    eval_code: str = None,
    hardware: str = "unknown"
) -> EvaluationSummary:
    """
    生成评测总摘要

    Args:
        evaluation_results: 评测结果（JSON格式）
        eval_code: 评测代号
        hardware: 硬件名称

    Returns:
        EvaluationSummary
    """
    if eval_code is None:
        eval_code = evaluation_results.get("eval_code", "")

    operators = []
    for op_result in evaluation_results.get("operators", []):
        operators.append(calculate_operator_summary(op_result))

    total_operators = len(operators)
    total_cases = sum(op.total_cases for op in operators)
    total_passed = sum(op.passed_cases for op in operators)
    overall_pass_rate = total_passed / total_cases if total_cases > 0 else 0.0

    # 计算整体几何平均加速比
    all_speedups = [op.geometric_mean_speedup for op in operators if op.geometric_mean_speedup > 0]
    overall_geometric_mean_speedup = calculate_geometric_mean(all_speedups)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return EvaluationSummary(
        eval_code=eval_code,
        hardware=hardware,
        total_operators=total_operators,
        total_cases=total_cases,
        total_passed=total_passed,
        overall_pass_rate=overall_pass_rate,
        overall_geometric_mean_speedup=overall_geometric_mean_speedup,
        operators=operators,
        timestamp=timestamp
    )


def render_summary_markdown(summary: EvaluationSummary) -> str:
    """
    渲染Summary为Markdown格式

    Args:
        summary: 评测摘要

    Returns:
        Markdown文本
    """
    lines = []

    # 标题
    lines.append("# 算子评测报告")
    lines.append("")
    lines.append(f"**评测代号**: {summary.eval_code}")
    lines.append(f"**硬件**: {summary.hardware}")
    lines.append(f"**时间**: {summary.timestamp}")
    lines.append("")

    # 总体摘要
    lines.append("## 总体结果")
    lines.append("")
    lines.append(f"- **总算子数**: {summary.total_operators}")
    lines.append(f"- **总用例数**: {summary.total_cases}")
    lines.append(f"- **通过用例**: {summary.total_passed}")
    lines.append(f"- **通过率**: {summary.overall_pass_rate:.2%}")
    lines.append(f"- **几何平均加速比**: {summary.overall_geometric_mean_speedup:.3f}x")
    lines.append("")

    # 各算子结果
    lines.append("## 算子详情")
    lines.append("")
    lines.append("| 算子 | Level | 用例数 | 通过 | 失败 | 通过率 | 几何平均加速比 | 平均MERE | 平均MARE |")
    lines.append("|------|-------|--------|------|------|--------|---------------|----------|----------|")

    for op in summary.operators:
        lines.append(
            f"| {op.operator} | L{op.level} | {op.total_cases} | {op.passed_cases} | "
            f"{op.failed_cases} | {op.pass_rate:.2%} | {op.geometric_mean_speedup:.3f}x | "
            f"{op.mere_avg:.2e} | {op.mare_avg:.2e} |"
        )

    lines.append("")

    # 列出需要用户关注的异常：编译失败的算子（贴出错误摘要），以及子进程
    # 超时 / 崩溃的算子（贴出原因）。跑完但精度/性能不过的算子不在这里出现
    # —— 那些 case 级细节由 report_generator 写到 markdown 报告里。
    compile_failed = [op for op in summary.operators if op.compilation_error]
    subprocess_failed = [op for op in summary.operators if op.subprocess_failure_reason]

    if compile_failed:
        lines.append("## 编译失败的算子")
        lines.append("")
        lines.append(f"共 {len(compile_failed)} 个算子在 `build.sh` 阶段失败，未进入评测。")
        lines.append("错误摘要（完整日志见 `logs/compile_round_*.log` 或 `build_errors.json`）：")
        lines.append("")
        for op in compile_failed:
            lines.append(f"### {op.operator}（L{op.level}）")
            lines.append("")
            lines.append("```")
            lines.append(op.compilation_error.strip())
            lines.append("```")
            lines.append("")

    if subprocess_failed:
        lines.append("## 子进程失败的算子")
        lines.append("")
        lines.append(f"共 {len(subprocess_failed)} 个算子在子进程隔离评测下异常：")
        lines.append("")
        for op in subprocess_failed:
            lines.append(f"- **{op.operator}** (L{op.level}): {op.subprocess_failure_reason}")
        lines.append("")

    # 结论
    if summary.overall_pass_rate >= 0.9:
        lines.append("## 结论")
        lines.append("")
        lines.append(f"评测通过率高（{summary.overall_pass_rate:.2%}），算子质量良好。")
        if summary.overall_geometric_mean_speedup > 1.0:
            lines.append(f"性能加速比 {summary.overall_geometric_mean_speedup:.3f}x，有性能优化空间。")
    elif summary.overall_pass_rate >= 0.7:
        lines.append("## 结论")
        lines.append("")
        lines.append(f"评测通过率中等（{summary.overall_pass_rate:.2%}），需要改进部分用例。")
    else:
        lines.append("## 结论")
        lines.append("")
        lines.append(f"评测通过率较低（{summary.overall_pass_rate:.2%}），需要重点排查精度问题。")

    return "\n".join(lines)


def save_summary(summary: EvaluationSummary, output_path: str) -> None:
    """
    保存Summary到文件

    Args:
        summary: 评测摘要
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    markdown = render_summary_markdown(summary)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)