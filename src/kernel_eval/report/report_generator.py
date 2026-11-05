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
评测报告生成器

职责：
1. 记录执行状态和性能数据
2. 输出JSON格式报告
3. 输出Markdown格式报告
"""

import json
from html import escape
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field

from ..config import Config, get_config
from .._version import FRAMEWORK_VERSION, read_tasks_version
from ..base.perf_metric import resolve_perf_metric
from ..base.score_weights import score_weights_context
from ..eval.evaluator import EvalCaseResult, EvalOperatorResult
from .scoring import NO_NPU_PERF_ERROR, ScoringCalculator, OperatorScoreInfo
from .setup_info import collect_setup_info
from ..base.result import FAILURE_TYPE_REFERENCE_INVALID


def _format_score(value):
    return "不可判定（参考失效）" if value is None else f"{value:.2f}"


def format_baseline_source(source):
    """Describe the selected origin, without asserting it is a measurement."""
    origin = source.get("source", "来源未记录")
    if origin == "runtime_reference":
        return "本轮参考路径计时（runtime_reference）"
    requested = source.get("requested_hardware", "未记录")
    actual = source.get("source_hardware", "未记录")
    label = "跨平台复用" if source.get("fallback_used") else "同平台数据"
    return f"{label}：请求 {requested}，来源 {actual}（{origin}）"


@dataclass
class EvalResult:
    """单用例评测结果"""

    rel_path: str = ""
    operator: str = ""
    case_id: str = ""
    status: str = ""  # success / failed / skipped
    elapsed_us: Optional[float] = 0  # None 表示未采集性能(--no-perf / 非 profiler 路径)
    op_times: Optional[Dict[str, Dict[str, float]]] = None
    error_msg: Optional[str] = None
    performance_error_msg: Optional[str] = None
    device: str = ""
    timestamp: str = ""
    accuracy: Optional[Dict] = None
    speedup: float = 0.0
    baseline_perf_us: float = 0.0
    t_hw_us: float = 0.0
    perf_score: Optional[float] = (
        None  # bench.tex Eq. 3: per-case hardware-anchored score
    )
    _perf_result: Any = None
    # 失败类型标注：区分真实失败与级联失败
    # None / "genuine"   — 真实的精度/执行失败
    # "cascade_device"   — 因 NPU 设备损坏级联失败
    # "skipped"          — 因设备不可恢复而跳过
    failure_type: Optional[str] = None
    perf_recheck: Optional[Dict[str, Any]] = None
    memory_peak_mb: Optional[float] = None
    memory_baseline_mb: Optional[float] = None
    baseline_source: Dict[str, Any] = field(default_factory=dict)
    perf_collection_failed: bool = False

    def resolve_profiling(self):
        if self._perf_result is not None:
            pr = self._perf_result
            self.elapsed_us = pr.elapsed_us
            self.op_times = pr.op_times
            if pr.error:
                self.error_msg = pr.error
            self._perf_result = None
        if self.op_times is None:
            self.op_times = {}

    @classmethod
    def from_eval_case_result(cls, result: EvalCaseResult) -> "EvalResult":
        """从EvalCaseResult创建"""
        # failure_type 影响状态标签：级联失败/跳过标注为 "cascade" / "skipped"
        status = "success" if result.success else "failed"
        if not result.success:
            if result.failure_type == "cascade_device":
                status = "cascade"
            elif result.failure_type == "skipped":
                status = "skipped"
            elif result.failure_type == FAILURE_TYPE_REFERENCE_INVALID:
                status = FAILURE_TYPE_REFERENCE_INVALID

        performance_error_msg = None
        if result.success and result.perf_result is None and result.perf_recheck:
            performance_error_msg = result.perf_recheck.get(
                "error_msg"
            ) or result.perf_recheck.get("note")

        return cls(
            rel_path=result.rel_path,
            operator=result.operator,
            case_id=result.case_id,
            status=status,
            elapsed_us=result.perf_result.elapsed_us if result.perf_result else None,
            op_times=result.perf_result.op_times if result.perf_result else {},
            error_msg=result.error_msg,
            performance_error_msg=performance_error_msg,
            accuracy=result.accuracy_result.to_dict()
            if result.accuracy_result
            else None,
            speedup=result.get_speedup(),
            baseline_perf_us=result.baseline_perf_us,
            t_hw_us=result.t_hw_us,
            baseline_source=result.baseline_source,
            perf_score=result.get_perf_score(),
            timestamp=datetime.now().isoformat(),
            failure_type=result.failure_type,
            perf_recheck=result.perf_recheck,
            memory_peak_mb=result.memory_peak_mb,
            memory_baseline_mb=result.memory_baseline_mb,
            perf_collection_failed=bool(
                result.perf_result is None
                or result.perf_result.metadata.get("perf_collection_failed", False)
            ),
        )


@dataclass
class OperatorReport:
    """算子报告"""

    rel_path: str = ""
    operator: str = ""
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    avg_speedup: float = 0.0
    score: Optional[float] = 0.0
    compilation_score: Optional[float] = 0.0
    compile_runtime_score: Optional[float] = 0.0
    compile_runtime_fail_cases: int = 0
    function_score: Optional[float] = 0.0
    performance_score: Optional[float] = 0.0
    cases: List[EvalResult] = field(default_factory=list)
    # 透传 EvalOperatorResult 上的诊断字段到最终报告；summary_generator
    # 读这两个字段渲染"编译失败"和"子进程失败"的分组。
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None
    score_error_code: Optional[str] = None
    score_error: Optional[str] = None
    reference_invalid_cases: int = 0

    @classmethod
    def from_eval_operator_result(
        cls,
        result: EvalOperatorResult,
        score: Optional[float],
        score_info: Optional[OperatorScoreInfo] = None,
    ) -> "OperatorReport":
        """从EvalOperatorResult创建"""
        cases = [EvalResult.from_eval_case_result(r) for r in result.results]
        if score_info is not None and score_info.zeroed_by_no_npu_perf:
            for case in cases:
                if case.status == "success" and (
                    case.elapsed_us is None or case.elapsed_us <= 0
                ):
                    case.performance_error_msg = NO_NPU_PERF_ERROR
        return cls(
            rel_path=result.rel_path,
            operator=result.operator,
            total_cases=result.total_cases,
            passed_cases=result.passed_cases,
            failed_cases=result.failed_cases,
            pass_rate=result.pass_rate,
            avg_speedup=result.avg_speedup,
            score=score,
            compilation_score=score_info.compilation_score if score_info else 0.0,
            compile_runtime_score=score_info.compilation_score if score_info else 0.0,
            compile_runtime_fail_cases=score_info.compile_runtime_fail_cases
            if score_info
            else 0,
            function_score=score_info.function_score if score_info else 0.0,
            performance_score=score_info.performance_score if score_info else 0.0,
            cases=cases,
            compilation_error=getattr(result, "compilation_error", None),
            subprocess_failure_reason=getattr(
                result, "subprocess_failure_reason", None
            ),
            score_error_code=score_info.score_error_code if score_info else None,
            score_error=score_info.score_error if score_info else None,
            reference_invalid_cases=result.reference_invalid_cases,
        )


@dataclass
class EvalReport:
    """完整评测报告"""

    framework_version: str
    tasks_version: str
    eval_code: str
    timestamp: str
    device: str
    total_operators: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_score: Optional[float]
    operators: List[OperatorReport]
    summary: Dict[str, Any]
    setup_info: Dict[str, Any] = field(default_factory=dict)
    score_weight_profile: Optional[str] = None
    perf_metric: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_weight_profile": self.score_weight_profile,
            "perf_metric": self.perf_metric,
            "framework_version": self.framework_version,
            "tasks_version": self.tasks_version,
            "eval_code": self.eval_code,
            "timestamp": self.timestamp,
            "device": self.device,
            "total_operators": self.total_operators,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "setup_info": self.setup_info,
            "operators": [asdict(op) for op in self.operators],
        }


class ReportGenerator:
    """报告生成器"""

    def __init__(
        self,
        output_dir: str = None,
        eval_code: str = None,
        semantic_prefix: str = "",
        config: Config = None,
    ):
        self.config = config or get_config()
        self.output_dir = Path(output_dir or self.config.reports_dir)
        self.semantic_prefix = semantic_prefix
        self.eval_code = eval_code or self._generate_eval_code()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scoring_calculator = ScoringCalculator()
        self.operator_results: List[EvalOperatorResult] = []

    def _generate_eval_code(self) -> str:
        """生成评测代号

        语义前缀优先级:
        - semantic_prefix 非空时: {prefix}_eval_{timestamp}
        - 否则: eval_{timestamp}（旧格式，向后兼容）
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.semantic_prefix:
            return f"{self.semantic_prefix}_eval_{ts}"
        return f"eval_{ts}"

    def _get_device_info(self) -> str:
        """获取设备信息"""
        device_type = self.config.device_type
        device_id = self.config.device_id
        return f"{device_type}:{device_id}" if device_type == "npu" else device_type

    def add_operator_result(self, result: EvalOperatorResult):
        """添加算子评测结果"""
        self.operator_results.append(result)

    def generate(self) -> "EvalReport":
        """生成完整评测报告"""
        with score_weights_context(self.config.tasks_root) as profile:
            metric = resolve_perf_metric()
            return self._generate(profile, metric)

    def _generate(self, profile: str, metric: str) -> "EvalReport":
        # 计算每个算子得分
        operator_reports = []
        score_infos = []
        total_cases = 0
        passed_cases = 0
        failed_cases = 0
        cascade_cases = 0  # 级联失败（设备异常导致的失败）

        for op_result in self.operator_results:
            score_info = self.scoring_calculator.calculate_operator_score(op_result)
            score_infos.append(score_info)
            op_report = OperatorReport.from_eval_operator_result(
                op_result,
                score_info.total_score,
                score_info=score_info,
            )
            operator_reports.append(op_report)

            total_cases += op_result.total_cases
            passed_cases += op_result.passed_cases
            # 区分真实失败和级联失败
            genuine_failed = sum(
                1
                for r in op_result.results
                if not r.success
                and r.failure_type
                not in ("cascade_device", "skipped", FAILURE_TYPE_REFERENCE_INVALID)
            )
            cascade_failed = sum(
                1
                for r in op_result.results
                if not r.success and r.failure_type in ("cascade_device", "skipped")
            )
            failed_cases += genuine_failed
            cascade_cases += cascade_failed

        # 计算综合得分
        overall_score = self.scoring_calculator.calculate_overall_score(score_infos)
        reference_invalid = sum(
            op.reference_invalid_cases for op in self.operator_results
        )
        valid_cases = total_cases - reference_invalid
        genuine_cases = valid_cases - cascade_cases

        # 构建摘要
        summary = {
            "total_operators": len(self.operator_results),
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "cascade_cases": cascade_cases,
            "reference_invalid_cases": reference_invalid,
            "score_valid": not reference_invalid,
            "pass_rate": passed_cases / valid_cases if valid_cases > 0 else 0.0,
            "genuine_pass_rate": passed_cases / genuine_cases
            if genuine_cases > 0
            else 0.0,
            "overall_score": overall_score,
        }
        if self.config.enable_profiler:
            measured = sum(
                1
                for op in self.operator_results
                for case in op.results
                if case.success
                and case.perf_result is not None
                and case.perf_result.elapsed_us > 0
            )
            summary["performance_measured_cases"] = measured
            summary["performance_unmeasured_cases"] = total_cases - measured

        try:
            tasks_version = read_tasks_version(self.config.tasks_root)
        except ValueError:
            # 显式环境覆盖允许没有 VERSION 的旧数据复算；不能冒用安装包版本。
            tasks_version = "unknown"

        return EvalReport(
            score_weight_profile=profile,
            perf_metric=metric,
            framework_version=FRAMEWORK_VERSION,
            tasks_version=tasks_version,
            eval_code=self.eval_code,
            timestamp=datetime.now().isoformat(),
            device=self._get_device_info(),
            total_operators=len(self.operator_results),
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            overall_score=overall_score,
            operators=operator_reports,
            summary=summary,
            setup_info=collect_setup_info(self.config),
        )

    def save_json(self, report: EvalReport, filename: str = None) -> Path:
        """保存JSON报告"""
        filename = filename or f"{self.eval_code}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[INFO] JSON报告已保存到: {output_path}")
        return output_path

    def save_markdown(self, report: EvalReport, filename: str = None) -> Path:
        """保存Markdown报告"""
        filename = filename or f"{self.eval_code}.md"
        output_path = self.output_dir / filename

        content = self._generate_markdown_content(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[INFO] Markdown报告已保存到: {output_path}")
        return output_path

    def _generate_markdown_content(self, report: EvalReport) -> str:
        """生成Markdown内容"""
        lines = [
            "# 算子评测报告",
            "",
            f"**评测代号**: {report.eval_code}",
            f"**评测时间**: {report.timestamp}",
            f"**设备**: {report.device}",
            f"**框架版本**: V{report.framework_version}",
            f"**评测集版本**: tasks-v{report.tasks_version}",
            "",
            "## 概览",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 评测算子数 | {report.total_operators} |",
            f"| 总用例数 | {report.total_cases} |",
            f"| 通过用例数 | {report.passed_cases} |",
            f"| 失败用例数 | {report.failed_cases} |",
            f"| 通过率 | {report.summary['pass_rate']:.2%} |",
            f"| 参考失效用例 | {report.summary.get('reference_invalid_cases', 0)} |",
            f"| 综合得分 | {_format_score(report.overall_score)} |",
            "",
            "## 算子详情",
            "",
        ]

        for op_report in report.operators:
            if (
                "performance_measured_cases" in report.summary
                and len(report.operators) == 1
            ):
                lines.extend(
                    [
                        f"性能有效计时：{report.summary['performance_measured_cases']} / {report.total_cases}；"
                        f"性能未完成：{report.summary['performance_unmeasured_cases']}。",
                        "",
                    ]
                )
            lines.extend(
                [
                    f"### {op_report.operator}（{op_report.rel_path}）",
                    "",
                    "| 指标 | 数值 |",
                    "|------|------|",
                    f"| 用例数 | {op_report.total_cases} |",
                    f"| 通过数 | {op_report.passed_cases} |",
                    f"| 失败数 | {op_report.failed_cases} |",
                    f"| 通过率 | {op_report.pass_rate:.2%} |",
                    f"| 平均加速比 | {op_report.avg_speedup:.2f}x |",
                    f"| 参考失效用例 | {op_report.reference_invalid_cases} |",
                    f"| 编译/运行得分 | {_format_score(op_report.compile_runtime_score)} |",
                    f"| 精度得分 | {_format_score(op_report.function_score)} |",
                    f"| 性能得分 | {_format_score(op_report.performance_score)} |",
                    f"| 得分 | {_format_score(op_report.score)} |",
                    "",
                ]
            )

            # 用例详情表格
            if op_report.cases:
                lines.extend(
                    [
                        "| 用例ID | 状态 | 耗时(μs) | 加速比 | 精度误差 |",
                        "|--------|------|----------|--------|----------|",
                    ]
                )
                for case in op_report.cases:
                    # 区分真实失败、级联失败、跳过
                    if case.status == "success":
                        status_icon = "✅"
                        if case.performance_error_msg:
                            status_icon = "✅精度 / ⚠️性能未完成"
                    elif case.status == "cascade":
                        status_icon = "⚠️级联"
                    elif case.status == "skipped":
                        status_icon = "⏭️跳过"
                    elif case.status == FAILURE_TYPE_REFERENCE_INVALID:
                        status_icon = "⚠️参考失效"
                    else:
                        status_icon = "❌"
                    speedup_str = f"{case.speedup:.2f}x" if case.speedup > 0 else "N/A"
                    accuracy_str = ""
                    if case.accuracy:
                        # max_diff 在 metadata 子字典中（新格式），兼容旧格式（顶层）
                        acc_meta = case.accuracy.get("metadata") or {}
                        max_diff = acc_meta.get(
                            "max_diff", case.accuracy.get("max_diff", 0)
                        )
                        accuracy_str = f"{max_diff:.6f}"
                        if acc_meta.get("checker_name") == "mc2_three_way":
                            accuracy_str = (
                                "三方精度通过"
                                if case.accuracy.get("passed")
                                else "三方精度失败"
                            )
                    else:
                        accuracy_str = case.error_msg or "N/A"
                    if case.status == FAILURE_TYPE_REFERENCE_INVALID:
                        accuracy_str = "参考失效，候选精度不可判定"
                    # elapsed_us 为 None 表示未做性能采集(--no-perf / 非 profiler 路径)
                    elapsed_str = (
                        "N/A" if case.elapsed_us is None else f"{case.elapsed_us:.2f}"
                    )
                    lines.append(
                        f"| {case.case_id} | {status_icon} | {elapsed_str} "
                        f"| {speedup_str} | {accuracy_str} |"
                    )
                    if case.performance_error_msg:
                        reason = (
                            str(case.performance_error_msg)
                            .replace("\n", " ")
                            .replace("|", "&#124;")
                        )
                        lines.append(f"| ↳ 性能原因 | {reason} | | | |")
                lines.append("")

            if any(case.baseline_source for case in op_report.cases):
                lines.extend(
                    [
                        "基线与理论下界来源（不改变原评分规则）：",
                        "",
                        "| 用例ID | 字段 | 来源 |",
                        "|---|---|---|",
                    ]
                )
                for case in op_report.cases:
                    for name, source in case.baseline_source.items():
                        cells = (case.case_id, name, format_baseline_source(source))
                        cells = [
                            escape(str(cell)).replace("|", "&#124;").replace("\n", " ")
                            for cell in cells
                        ]
                        lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

        return "\n".join(lines)

    def save_all(self, report: EvalReport) -> Dict[str, Path]:
        """保存所有格式报告"""
        return {
            "json": self.save_json(report),
            "markdown": self.save_markdown(report),
            "html": self.save_html(report),
        }

    def save_html(self, report: EvalReport, filename: str = None) -> Path:
        """保存 HTML 报告"""
        filename = filename or f"{self.eval_code}.html"
        output_path = self.output_dir / filename

        from .html_generator import render_html_report
        from ..config import get_project_root

        # Determine description path
        index_path = get_project_root() / "tasks" / "description.html"

        html_content = render_html_report(report, report.setup_info, str(index_path))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[INFO] HTML报告已保存到: {output_path}")
        return output_path

    def print_summary(self, report: EvalReport):
        """打印摘要"""
        cascade = report.summary.get("cascade_cases", 0)
        genuine_rate = report.summary.get(
            "genuine_pass_rate", report.summary["pass_rate"]
        )
        print("\n" + "=" * 60)
        print("评测结果摘要")
        print("=" * 60)
        print(f"评测代号: {report.eval_code}")
        print(f"评测算子数: {report.total_operators}")
        print(f"总用例数: {report.total_cases}")
        print(f"通过用例数: {report.passed_cases}")
        print(f"失败用例数: {report.failed_cases}")
        if cascade > 0:
            print(f"级联失败（设备异常）: {cascade}")
            print(f"真实通过率（排除级联）: {genuine_rate:.2%}")
        print(f"通过率: {report.summary['pass_rate']:.2%}")
        if report.summary.get("reference_invalid_cases"):
            print(f"参考失效用例: {report.summary['reference_invalid_cases']}")
        print(f"综合得分: {_format_score(report.overall_score)}")
        print("=" * 60 + "\n")
