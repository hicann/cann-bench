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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field

from ..config import Config, get_config
from ..eval.evaluator import EvalCaseResult, EvalOperatorResult
from .scoring import ScoringCalculator, ScoreInfo


@dataclass
class EvalResult:
    """单用例评测结果"""
    rel_path: str = ""
    operator: str = ""
    case_id: int = 0
    status: str = ""  # success / failed / skipped
    elapsed_us: float = 0
    op_times: Optional[Dict[str, Dict[str, float]]] = None
    error_msg: Optional[str] = None
    device: str = ""
    timestamp: str = ""
    accuracy: Optional[Dict] = None
    speedup: float = 0.0
    baseline_perf_us: float = 0.0
    t_hw_us: float = 0.0
    perf_score: Optional[float] = None  # bench.tex Eq. 3: per-case SOL score
    _perf_result: Any = None

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
        return cls(
            rel_path=result.rel_path,
            operator=result.operator,
            case_id=result.case_num,
            status="success" if result.success else "failed",
            elapsed_us=result.perf_result.elapsed_us if result.perf_result else 0,
            op_times=result.perf_result.op_times if result.perf_result else {},
            error_msg=result.error_msg,
            accuracy=result.accuracy_result.to_dict() if result.accuracy_result else None,
            speedup=result.get_speedup(),
            baseline_perf_us=result.baseline_perf_us,
            t_hw_us=result.t_hw_us,
            perf_score=result.get_perf_score(),
            timestamp=datetime.now().isoformat(),
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
    score: float = 0.0
    cases: List[EvalResult] = field(default_factory=list)
    # 透传 EvalOperatorResult 上的诊断字段到最终报告；summary_generator
    # 读这两个字段渲染"编译失败"和"子进程失败"的分组。
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None

    @classmethod
    def from_eval_operator_result(cls, result: EvalOperatorResult, score: float) -> "OperatorReport":
        """从EvalOperatorResult创建"""
        cases = [EvalResult.from_eval_case_result(r) for r in result.results]
        return cls(
            rel_path=result.rel_path,
            operator=result.operator,
            total_cases=result.total_cases,
            passed_cases=result.passed_cases,
            failed_cases=result.failed_cases,
            pass_rate=result.pass_rate,
            avg_speedup=result.avg_speedup,
            score=score,
            cases=cases,
            compilation_error=getattr(result, 'compilation_error', None),
            subprocess_failure_reason=getattr(result, 'subprocess_failure_reason', None),
        )


@dataclass
class EvalReport:
    """完整评测报告"""
    version: str
    eval_code: str
    timestamp: str
    device: str
    total_operators: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_score: float
    operators: List[OperatorReport]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'eval_code': self.eval_code,
            'timestamp': self.timestamp,
            'device': self.device,
            'total_operators': self.total_operators,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'overall_score': self.overall_score,
            'summary': self.summary,
            'operators': [asdict(op) for op in self.operators],
        }


class ReportGenerator:
    """报告生成器"""

    VERSION = "1.0"

    def __init__(self, output_dir: str = None, eval_code: str = None, config: Config = None):
        self.config = config or get_config()
        self.output_dir = Path(output_dir or self.config.reports_dir)
        self.eval_code = eval_code or self._generate_eval_code()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scoring_calculator = ScoringCalculator()
        self.operator_results: List[EvalOperatorResult] = []

    def _generate_eval_code(self) -> str:
        """生成评测代号"""
        return datetime.now().strftime("eval_%Y%m%d_%H%M%S")

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
        # 计算每个算子得分
        operator_reports = []
        score_infos = []
        total_cases = 0
        passed_cases = 0
        failed_cases = 0

        for op_result in self.operator_results:
            score_info = self.scoring_calculator.calculate_operator_score(op_result)
            score_infos.append(score_info)
            op_report = OperatorReport.from_eval_operator_result(op_result, score_info.total_score)
            operator_reports.append(op_report)

            total_cases += op_result.total_cases
            passed_cases += op_result.passed_cases
            failed_cases += op_result.failed_cases

        # 计算综合得分
        overall_score = self.scoring_calculator.calculate_overall_score(score_infos)

        # 构建摘要
        summary = {
            'total_operators': len(self.operator_results),
            'total_cases': total_cases,
            'passed_cases': passed_cases,
            'failed_cases': failed_cases,
            'pass_rate': passed_cases / total_cases if total_cases > 0 else 0.0,
            'overall_score': overall_score,
        }

        return EvalReport(
            version=self.VERSION,
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
        )

    def save_json(self, report: EvalReport, filename: str = None) -> Path:
        """保存JSON报告"""
        filename = filename or f"{self.eval_code}.json"
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[INFO] JSON报告已保存到: {output_path}")
        return output_path

    def save_markdown(self, report: EvalReport, filename: str = None) -> Path:
        """保存Markdown报告"""
        filename = filename or f"{self.eval_code}.md"
        output_path = self.output_dir / filename

        content = self._generate_markdown_content(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[INFO] Markdown报告已保存到: {output_path}")
        return output_path

    def _generate_markdown_content(self, report: EvalReport) -> str:
        """生成Markdown内容"""
        lines = [
            f"# 算子评测报告",
            f"",
            f"**评测代号**: {report.eval_code}",
            f"**评测时间**: {report.timestamp}",
            f"**设备**: {report.device}",
            f"**版本**: {report.version}",
            f"",
            f"## 概览",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 评测算子数 | {report.total_operators} |",
            f"| 总用例数 | {report.total_cases} |",
            f"| 通过用例数 | {report.passed_cases} |",
            f"| 失败用例数 | {report.failed_cases} |",
            f"| 通过率 | {report.summary['pass_rate']:.2%} |",
            f"| 综合得分 | {report.overall_score:.2f} |",
            f"",
            f"## 算子详情",
            f"",
        ]

        for op_report in report.operators:
            lines.extend([
                f"### {op_report.operator}（{op_report.rel_path}）",
                f"",
                f"| 指标 | 数值 |",
                f"|------|------|",
                f"| 用例数 | {op_report.total_cases} |",
                f"| 通过数 | {op_report.passed_cases} |",
                f"| 失败数 | {op_report.failed_cases} |",
                f"| 通过率 | {op_report.pass_rate:.2%} |",
                f"| 平均加速比 | {op_report.avg_speedup:.2f}x |",
                f"| 得分 | {op_report.score:.2f} |",
                f"",
            ])

            # 用例详情表格
            if op_report.cases:
                lines.extend([
                    f"| 用例ID | 状态 | 耗时(μs) | 加速比 | 精度误差 |",
                    f"|--------|------|----------|--------|----------|",
                ])
                for case in op_report.cases:
                    status_icon = "✅" if case.status == "success" else "❌"
                    speedup_str = f"{case.speedup:.2f}x" if case.speedup > 0 else "N/A"
                    accuracy_str = ""
                    if case.accuracy:
                        max_diff = case.accuracy.get('max_diff', 0)
                        accuracy_str = f"{max_diff:.6f}"
                    else:
                        accuracy_str = case.error_msg or "N/A"
                    lines.append(
                        f"| {case.case_id} | {status_icon} | {case.elapsed_us:.2f} "
                        f"| {speedup_str} | {accuracy_str} |"
                    )
                lines.append(f"")

        return "\n".join(lines)

    def save_all(self, report: EvalReport) -> Dict[str, Path]:
        """保存所有格式报告"""
        return {
            'json': self.save_json(report),
            'markdown': self.save_markdown(report),
        }

    def print_summary(self, report: EvalReport):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("评测结果摘要")
        print("=" * 60)
        print(f"评测代号: {report.eval_code}")
        print(f"评测算子数: {report.total_operators}")
        print(f"总用例数: {report.total_cases}")
        print(f"通过用例数: {report.passed_cases}")
        print(f"失败用例数: {report.failed_cases}")
        print(f"通过率: {report.summary['pass_rate']:.2%}")
        print(f"综合得分: {report.overall_score:.2f}")
        print("=" * 60 + "\n")