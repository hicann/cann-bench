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
    aggregate_eq4,
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
    composite_score: float = 0.0  # EachOperatorScore，常规区间 [0, 100]；T_cand<T_HW 时可 >100（不截断）
    # 可选字段：当算子跑不起来时的原因（编译失败 / 子进程超时或崩溃）。
    # 为 None 时渲染普通的 case 表；非空时在小节头下方优先渲染原因块。
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None
    score_error_code: Optional[str] = None
    score_error: Optional[str] = None


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
    # F023: 归一化总分——每算子满分约 100（T_cand<T_HW 时可 >100），total / num_operators 跨算子集可比。
    # Σ 总分会随算子数线性膨胀，跨 benchmark 版本/不同 Level 子集不可比；
    # 平均分作为额外维度让"系统变好" vs "系统变大"能被区分。
    benchmark_avg_score: float = 0.0
    level_avg_scores: Dict[str, float] = None  # type: ignore[assignment]


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
    """从 op_result 字典计算 Eq.4 三轴得分与综合得分。

    本函数仅负责把 dict 解析成 (success, score_i) 元组列表，实际 Eq.4 聚合
    委托给 scoring.aggregate_eq4——单一事实来源。

    支持两种 JSON 形状：
      - EvalCaseResult.to_dict（嵌套）：perf={'elapsed_us', 'perf_score', ...}
      - EvalResult.to_dict（扁平）：elapsed_us / perf_score 在 case 顶层
    """
    compile_passed = op_result.get(
        "compile_passed",
        op_result.get("compilation_error") is None
        and op_result.get("subprocess_failure_reason") is None,
    )
    # rel_path 用于 F058/F061 的 warn dedup key + 错误溯源
    rel_path = op_result.get("rel_path") or op_result.get("operator") or None
    # 同时识别两种 case 列表字段名：results (EvalOperatorResult) / cases (OperatorReport)
    cases = op_result.get("results") or op_result.get("cases") or []
    total_cases = max(op_result.get("total_cases", 0), len(cases), 1)

    case_scores: List = []
    n_no_perf_pass = 0
    for case in cases:
        success = case.get("success", case.get("status") == "success")
        if not success:
            case_scores.append((False, None))
            continue
        perf = case.get("perf") or {}
        score_i = perf.get("perf_score", case.get("perf_score"))
        if score_i is None:
            t_cand = perf.get("elapsed_us") or case.get("elapsed_us") or 0
            if t_cand <= 0:
                n_no_perf_pass += 1
            t_base = case.get("baseline_perf_us") or 0
            t_hw = case.get("t_hw_us") or 0
            score_i = per_case_sol_score(t_base, t_cand, t_hw, rel_path=rel_path)
        case_scores.append((True, score_i))

    agg = aggregate_eq4(
        compile_passed=compile_passed,
        total_cases=total_cases,
        case_scores=case_scores,
        rel_path=rel_path,
        n_no_perf_pass=n_no_perf_pass,
    )
    return {
        "compile_passed": compile_passed,
        "compilation_score": agg["compilation_score"],
        "function_score": agg["function_score"],
        "performance_score": agg["performance_score"],
        "composite_score": agg["total_score"],
        "score_error_code": agg.get("score_error_code"),
        "score_error": agg.get("score_error"),
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
    cases = op_result.get("results") or op_result.get("cases") or []
    total_cases = op_result.get("total_cases", len(cases))
    passed_cases = op_result.get("passed_cases", 0)
    # 优先使用上游统计字段（与 results.py summarize_case_results 口径一致），
    # 缺失时再退化为 total - passed（仅当 skipped/failed 字段都不可用时）。
    failed_cases = op_result.get("failed_cases")
    if failed_cases is None:
        failed_cases = max(total_cases - passed_cases, 0)
    pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0

    # 计算几何平均加速比 / MERE / MARE。
    # mere/mare 存在 accuracy.metadata 中（RelativeErrorChecker 产生），
    # AllCloseChecker 不产生这些指标。兼容旧格式（top-level）和新格式（metadata 嵌套）。
    speedups = []
    meres = []
    mares = []
    for case in cases:
        perf = case.get("perf") or {}
        speedup = perf.get("speedup", case.get("speedup"))
        if speedup and speedup > 0:
            speedups.append(speedup)
        accuracy = case.get("accuracy") or {}
        acc_meta = accuracy.get("metadata") or {}
        mere = acc_meta.get("mere") or accuracy.get("mere") or case.get("mere")
        if mere:
            meres.append(mere)
        mare = acc_meta.get("mare") or accuracy.get("mare") or case.get("mare")
        if mare:
            mares.append(mare)

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
        score_error_code=scores.get("score_error_code"),
        score_error=scores.get("score_error"),
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
    level_op_count: Dict[str, int] = {}
    for op in operators:
        # 取 rel_path 第一段作为 level 标签 (e.g. "level1/exp" -> "level1")
        level_key = op.rel_path.split('/', 1)[0] if op.rel_path else "unknown"
        level_scores[level_key] = level_scores.get(level_key, 0.0) + op.composite_score
        level_op_count[level_key] = level_op_count.get(level_key, 0) + 1

    # F023: 归一化分——按算子数除回到 [0, 100] 量纲，让跨算子集可比
    benchmark_avg_score = (
        benchmark_total_score / total_operators if total_operators > 0 else 0.0
    )
    level_avg_scores: Dict[str, float] = {
        lvl: (level_scores[lvl] / level_op_count[lvl]) if level_op_count[lvl] > 0 else 0.0
        for lvl in level_scores
    }

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
        benchmark_avg_score=benchmark_avg_score,
        level_avg_scores=level_avg_scores,
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
    lines.append(f"- **Benchmark 总分** (Σ EachOperatorScore，随算子数线性膨胀): {summary.benchmark_total_score:.2f}")
    lines.append(f"- **Benchmark 归一化分** (avg，每算子满分 100，跨算子集可比): {summary.benchmark_avg_score:.2f} / 100")
    if summary.level_scores:
        for level in sorted(summary.level_scores.keys()):
            total = summary.level_scores[level]
            avg = summary.level_avg_scores.get(level, 0.0) if summary.level_avg_scores else 0.0
            lines.append(f"  - {level}: total={total:.2f}, avg={avg:.2f}/100")
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

    # 结论：综合考量通过率与综合得分（pass_rate 高但性能 0 不应被评为"良好"）。
    # 每算子满分=100，故 avg_composite >= 70 视为整体性能也达标。
    avg_composite = (
        summary.benchmark_total_score / summary.total_operators
        if summary.total_operators > 0 else 0.0
    )
    lines.append("## 结论")
    lines.append("")
    lines.append(
        f"- 通过率：{summary.overall_pass_rate:.2%}"
        f"  |  平均综合得分：{avg_composite:.2f}/100"
        f"  |  几何平均加速比：{summary.overall_geometric_mean_speedup:.3f}x"
    )
    if summary.overall_pass_rate >= 0.9 and avg_composite >= 70:
        lines.append("评测整体表现良好：通过率高且综合得分达标。")
    elif summary.overall_pass_rate >= 0.9 and avg_composite < 70:
        lines.append(
            f"通过率高但综合得分偏低（{avg_composite:.2f}/100）——"
            "多数算子精度通过但性能/编译扣分严重，建议排查性能数据。"
        )
    elif summary.overall_pass_rate >= 0.7:
        lines.append("评测中等：建议改进未通过的用例并复核性能采集。")
    else:
        lines.append("评测不达标：通过率偏低，需重点排查精度问题。")

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
