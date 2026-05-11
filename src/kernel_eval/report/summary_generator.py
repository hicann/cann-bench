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

from .scoring import (
    WEIGHT_COMPILATION,
    WEIGHT_FUNCTION,
    WEIGHT_PERFORMANCE,
    per_case_sol_score,
)


@dataclass
class OperatorSummary:
    """算子评测摘要"""
    operator: str = ""
    rel_path: str = ""
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    geometric_mean_speedup: float = 0.0
    mere_avg: float = 0.0
    mare_avg: float = 0.0
    # bench.tex 三轴得分（0-100 量纲）
    compile_passed: bool = False
    compilation_score: float = 0.0
    function_score: float = 0.0
    performance_score: float = 0.0
    composite_score: float = 0.0  # EachOperatorScore，[0, 100]
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
    # bench.tex Eq. 5: Σ EachOperatorScore (跨算子求和) 与各 Level 的小计
    benchmark_total_score: float = 0.0
    level_scores: Dict[str, float] = None  # type: ignore[assignment]


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


def _composite_score_from_dict(op_result: Dict[str, Any]) -> Dict[str, float]:
    """从 op_result 字典直接计算 bench.tex Eq. 4 的三轴得分与综合得分。

    权重与单用例公式从 scoring 模块导入，确保单一事实来源。
    支持两种 JSON 形状：
      - EvalCaseResult.to_dict 形状：perf={'elapsed_us', 'perf_score', ...}（嵌套）
      - EvalResult.to_dict 形状：elapsed_us / perf_score 直接放在 case 顶层
    """
    w_c, w_f, w_p = WEIGHT_COMPILATION, WEIGHT_FUNCTION, WEIGHT_PERFORMANCE
    compile_passed = op_result.get("compile_passed",
                                   op_result.get("compilation_error") is None)
    delta_pass = 1 if compile_passed else 0
    # 同时识别两种 case 列表字段名：results (EvalOperatorResult) / cases (OperatorReport)
    cases = op_result.get("results") or op_result.get("cases") or []
    total_cases = max(op_result.get("total_cases", len(cases)), 1)

    n_func_pass = 0
    perf_score_sum = 0.0
    if compile_passed:
        for case in cases:
            success = case.get("success", case.get("status") == "success")
            if not success:
                continue
            n_func_pass += 1
            perf = case.get("perf") or {}
            score_i = perf.get("perf_score", case.get("perf_score"))
            if score_i is None:
                t_cand = perf.get("elapsed_us") or case.get("elapsed_us") or 0
                t_base = case.get("baseline_perf_us") or 0
                t_hw = case.get("t_hw_us") or 0
                score_i = per_case_sol_score(t_base, t_cand, t_hw)
            perf_score_sum += score_i if score_i is not None else 0.0

    compilation_score = w_c * delta_pass * 100.0
    function_score = (n_func_pass * w_f / total_cases) * 100.0
    performance_score = (perf_score_sum * w_p / total_cases) * 100.0
    composite = compilation_score + function_score + performance_score
    return {
        "compile_passed": compile_passed,
        "compilation_score": compilation_score,
        "function_score": function_score,
        "performance_score": performance_score,
        "composite_score": composite,
    }


def calculate_operator_summary(op_result: Dict[str, Any]) -> OperatorSummary:
    """
    从算子结果计算摘要

    Args:
        op_result: 算子评测结果（字典格式）

    Returns:
        OperatorSummary
    """
    operator = op_result.get("operator", "")
    rel_path = op_result.get("rel_path", "")
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

    scores = _composite_score_from_dict(op_result)

    return OperatorSummary(
        operator=operator,
        rel_path=rel_path,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        geometric_mean_speedup=geometric_mean_speedup,
        mere_avg=mere_avg,
        mare_avg=mare_avg,
        compile_passed=scores["compile_passed"],
        compilation_score=scores["compilation_score"],
        function_score=scores["function_score"],
        performance_score=scores["performance_score"],
        composite_score=scores["composite_score"],
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

    # bench.tex Eq. 5: benchmark / level 总分 = Σ EachOperatorScore
    benchmark_total_score = sum(op.composite_score for op in operators)
    level_scores: Dict[str, float] = {}
    for op in operators:
        # 取 rel_path 第一段作为 level 标签 (e.g. "level1/exp" -> "level1")
        level_key = op.rel_path.split('/', 1)[0] if op.rel_path else "unknown"
        level_scores[level_key] = level_scores.get(level_key, 0.0) + op.composite_score

    return EvaluationSummary(
        eval_code=eval_code,
        hardware=hardware,
        total_operators=total_operators,
        total_cases=total_cases,
        total_passed=total_passed,
        overall_pass_rate=overall_pass_rate,
        overall_geometric_mean_speedup=overall_geometric_mean_speedup,
        operators=operators,
        timestamp=timestamp,
        benchmark_total_score=benchmark_total_score,
        level_scores=level_scores,
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
    lines.append(f"- **几何平均加速比 (诊断)**: {summary.overall_geometric_mean_speedup:.3f}x")
    lines.append(f"- **Benchmark 总分** (Σ EachOperatorScore): {summary.benchmark_total_score:.2f}")
    if summary.level_scores:
        for level in sorted(summary.level_scores.keys()):
            lines.append(f"  - {level}: {summary.level_scores[level]:.2f}")
    lines.append("")

    # 各算子结果
    lines.append("## 算子详情")
    lines.append("")
    lines.append("| 算子 | 路径 | 用例数 | 通过 | 失败 | 通过率 | 综合得分 | 编译 | 功能 | 性能 | 几何加速比 |")
    lines.append("|------|------|--------|------|------|--------|----------|------|------|------|-----------|")

    for op in summary.operators:
        lines.append(
            f"| {op.operator} | {op.rel_path} | {op.total_cases} | {op.passed_cases} | "
            f"{op.failed_cases} | {op.pass_rate:.2%} | {op.composite_score:.2f} | "
            f"{op.compilation_score:.2f} | {op.function_score:.2f} | {op.performance_score:.2f} | "
            f"{op.geometric_mean_speedup:.3f}x |"
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
            lines.append(f"### {op.operator}（{op.rel_path}）")
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
            lines.append(f"- **{op.operator}** ({op.rel_path}): {op.subprocess_failure_reason}")
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