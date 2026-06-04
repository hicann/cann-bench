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
评测结果数据类

职责：
1. 定义评测结果数据结构
2. 提供 to_dict 序列化方法
3. 提供结果统计公共函数

从 evaluator.py 拆分出来，避免循环导入。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class CaseResultSummary:
    """用例结果统计摘要"""
    passed: int
    failed: int
    skipped: int
    avg_speedup: float
    pass_rate: float


def summarize_case_results(case_results: List[Any]) -> CaseResultSummary:
    """统计用例结果

    Args:
        case_results: EvalCaseResult 列表

    Returns:
        CaseResultSummary 统计摘要

    Note:
        - passed: success=True 的用例数
        - failed: success=False 且 accuracy_result 不为 None 的用例数（精度失败）
        - skipped: success=False 且 accuracy_result 为 None 的用例数（执行失败/跳过）
        - avg_speedup: 成功用例的平均加速比
        - pass_rate: 通过率
    """
    passed = sum(1 for r in case_results if r.success)
    failed = sum(1 for r in case_results if not r.success and r.accuracy_result is not None)
    skipped = sum(1 for r in case_results if not r.success and r.accuracy_result is None)
    speedups = [r.get_speedup() for r in case_results if r.success and r.get_speedup() > 0]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
    total = len(case_results)
    pass_rate = passed / total if total > 0 else 0.0

    return CaseResultSummary(
        passed=passed,
        failed=failed,
        skipped=skipped,
        avg_speedup=avg_speedup,
        pass_rate=pass_rate,
    )


from .op_runner import OpRunResult
from .accuracy_eval import AccuracyResult
from .perf_eval import PerfResult
from ..data.package_manager import PackageInfo


@dataclass
class EvalCaseResult:
    """单用例评测结果"""
    case_id: str
    rel_path: str           # 替代 level，使用相对路径
    operator: str
    case_num: int
    success: bool
    accuracy_result: Optional[AccuracyResult] = None
    perf_result: Optional[PerfResult] = None
    golden_run_result: Optional[OpRunResult] = None
    ai_run_result: Optional[OpRunResult] = None
    error_msg: Optional[str] = None
    baseline_perf_us: float = 0.0
    t_hw_us: float = 0.0  # 硬件下界 T_HW
    # 失败类型标注：区分真实失败与级联失败
    # None / "genuine"   — 真实的精度/执行失败（case 本身有问题）
    # "cascade_device"   — 因 NPU 设备损坏级联失败
    # "skipped"          — 因设备不可恢复而跳过
    failure_type: Optional[str] = None

    def get_speedup(self) -> float:
        """计算加速比（保留为诊断指标）

        baseline_perf_us 缺失时走 fallback 代理基线 max(t_hw*3, 10)，
        与 per_case_sol_score 的 fallback 规则一致。
        """
        if not self.perf_result or self.perf_result.elapsed_us <= 0:
            return 0.0
        if self.baseline_perf_us > 0:
            return self.baseline_perf_us / self.perf_result.elapsed_us
        # Fallback: baseline 缺失，使用代理基线
        if self.t_hw_us > 0:
            from ..report.scoring import _fallback_baseline_from_hw
            proxy_baseline = _fallback_baseline_from_hw(self.t_hw_us)
            return proxy_baseline / self.perf_result.elapsed_us
        return 0.0

    def get_perf_score(self) -> Optional[float]:
        """单用例 hardware-anchored 性能得分（bench.tex Eq. 3）。

        薄封装：实际公式与边界处理见 scoring.per_case_sol_score——单一事实来源。
        """
        if not self.perf_result or self.perf_result.elapsed_us <= 0:
            return None
        # 局部 import 避免 evaluator/results <-> scoring 循环依赖。
        from ..report.scoring import per_case_sol_score
        return per_case_sol_score(
            float(self.baseline_perf_us),
            float(self.perf_result.elapsed_us),
            float(self.t_hw_us),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'rel_path': self.rel_path,
            'operator': self.operator,
            'case_num': self.case_num,
            'success': self.success,
            'accuracy': self.accuracy_result.to_dict() if self.accuracy_result else None,
            'perf': {
                'elapsed_us': self.perf_result.elapsed_us if self.perf_result else 0,
                'speedup': self.get_speedup(),
                'perf_score': self.get_perf_score(),
                'op_times': self.perf_result.op_times if self.perf_result else {},
            } if self.perf_result else None,
            'golden_elapsed_us': self.golden_run_result.elapsed_us if self.golden_run_result else 0,
            'ai_elapsed_us': self.ai_run_result.elapsed_us if self.ai_run_result else 0,
            'error_msg': self.error_msg,
            'baseline_perf_us': self.baseline_perf_us,
            't_hw_us': self.t_hw_us,
            'failure_type': self.failure_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalCaseResult':
        """从字典反序列化"""
        perf_data = data.get('perf')
        perf_result = None
        if perf_data:
            perf_result = PerfResult(
                elapsed_us=perf_data.get('elapsed_us', 0),
                op_times=perf_data.get('op_times', {}),
                error_msg=perf_data.get('error_msg'),
                metadata={
                    'case_id': data.get('case_id', ''),
                    'baseline_us': perf_data.get('baseline_us', 0),
                    't_hw_us': perf_data.get('t_hw_us', 0),
                }
            )

        accuracy_data = data.get('accuracy')
        accuracy_result = None
        if accuracy_data:
            # 优先使用新格式（metadata 为嵌套子 dict）
            # 兼容旧格式（聚合指标在 accuracy 顶层，无 metadata 子 dict）
            metadata = accuracy_data.get('metadata', {})
            if not metadata:
                # 旧格式：聚合指标散落在 accuracy 顶层，收入 metadata
                metadata = {
                    'dtype': accuracy_data.get('dtype', 'float32'),
                    'mare': accuracy_data.get('mare', 0.0),
                    'mere': accuracy_data.get('mere', 0.0),
                    'max_diff': accuracy_data.get('max_diff', 0.0),
                    'mean_diff': accuracy_data.get('mean_diff', 0.0),
                    'mismatch_count': accuracy_data.get('mismatch_count', 0),
                    'total_count': accuracy_data.get('total_count', 0),
                    'mismatch_ratio': accuracy_data.get('mismatch_ratio', 0.0),
                    'small_value_error_count': accuracy_data.get('small_value_error_count', 0),
                    'small_value_cpu_error_count': accuracy_data.get('small_value_cpu_error_count', 0),
                    'small_value_total_count': accuracy_data.get('small_value_total_count', 0),
                    'cancel_error_count': accuracy_data.get('cancel_error_count', 0),
                    'cancel_cpu_error_count': accuracy_data.get('cancel_cpu_error_count', 0),
                    'cancel_total_count': accuracy_data.get('cancel_total_count', 0),
                }
            if 'trial' not in metadata:
                metadata['trial'] = accuracy_data.get('trial', 1)

            # 反序列化 output_results：通过注册表按 checker_name 查找子类
            output_results_raw = accuracy_data.get('output_results', [])
            output_results = []
            if output_results_raw:
                checker_name = metadata.get('checker_name', '')
                # 触发 checker 模块注册（import 会执行 register_output_result）
                from ..base.result import get_output_result_cls
                if checker_name:
                    _checker_modules = {
                        'relative_error': '..checkers.relative_error_checker',
                        'cann_default': '..checkers.relative_error_checker',  # 兼容旧名
                        'allclose': '..checkers.allclose_checker',
                    }
                    mod_path = _checker_modules.get(checker_name)
                    if mod_path:
                        import importlib
                        importlib.import_module(mod_path, __package__)
                output_cls = get_output_result_cls(checker_name)
                if output_cls is not None:
                    output_results = [output_cls.from_dict(item) for item in output_results_raw]

            accuracy_result = AccuracyResult(
                passed=accuracy_data.get('passed', True),
                threshold=accuracy_data.get('threshold'),
                error_msg=accuracy_data.get('error_msg'),
                output_results=output_results,
                metadata=metadata,
            )

        return cls(
            case_id=data.get('case_id', ''),
            rel_path=data.get('rel_path', ''),
            operator=data.get('operator', ''),
            case_num=data.get('case_num', 0),
            success=data.get('success', False),
            accuracy_result=accuracy_result,
            perf_result=perf_result,
            error_msg=data.get('error_msg'),
            baseline_perf_us=data.get('baseline_perf_us', 0.0),
            t_hw_us=data.get('t_hw_us', 0.0),
            failure_type=data.get('failure_type'),
        )


@dataclass
class EvalOperatorResult:
    """算子评测结果"""
    rel_path: str           # 替代 level，使用相对路径
    operator: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    results: List[EvalCaseResult]
    pass_rate: float
    avg_speedup: float
    # 当算子跑不起来时附带的诊断信息
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None

    @property
    def compile_passed(self) -> bool:
        """δ_pass：编译通过标记。无 compilation_error 且无 subprocess_failure_reason 视为通过。"""
        return self.compilation_error is None and self.subprocess_failure_reason is None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'rel_path': self.rel_path,
            'operator': self.operator,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'skipped_cases': self.skipped_cases,
            'pass_rate': self.pass_rate,
            'avg_speedup': self.avg_speedup,
            'compile_passed': self.compile_passed,
            'results': [r.to_dict() for r in self.results],
        }
        if self.compilation_error:
            d['compilation_error'] = self.compilation_error
        if self.subprocess_failure_reason:
            d['subprocess_failure_reason'] = self.subprocess_failure_reason
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalOperatorResult':
        """从字典反序列化"""
        results_data = data.get('results', [])
        results = [EvalCaseResult.from_dict(r) for r in results_data]

        return cls(
            rel_path=data.get('rel_path', ''),
            operator=data.get('operator', ''),
            total_cases=data.get('total_cases', 0),
            passed_cases=data.get('passed_cases', 0),
            failed_cases=data.get('failed_cases', 0),
            skipped_cases=data.get('skipped_cases', 0),
            results=results,
            pass_rate=data.get('pass_rate', 0.0),
            avg_speedup=data.get('avg_speedup', 0.0),
            compilation_error=data.get('compilation_error'),
            subprocess_failure_reason=data.get('subprocess_failure_reason'),
        )


@dataclass
class EvalSessionResult:
    """评测会话结果"""
    operators: List[EvalOperatorResult]
    package_info: Optional[PackageInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operators': [op.to_dict() for op in self.operators],
            'package_info': {
                'source_dir': self.package_info.source_dir if self.package_info else '',
                'whl_path': self.package_info.whl_path if self.package_info else '',
                'run_path': self.package_info.run_path if self.package_info else '',
            } if self.package_info else None,
        }