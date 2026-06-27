#!/usr/bin/python3
# coding=utf-8

from kernel_eval.base.result import AccuracyResult, PerfResult
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult
from kernel_eval.staged_eval import _case_num_from_value, _merge_results


def _case(case_num, *, success, failure_type=None, accuracy_result=None, perf_result=None):
    return EvalCaseResult(
        case_id=f"level2/dynamic_quant_{case_num}",
        rel_path="level2/dynamic_quant",
        operator="DynamicQuant",
        case_num=case_num,
        success=success,
        accuracy_result=accuracy_result,
        perf_result=perf_result,
        baseline_perf_us=100.0,
        t_hw_us=10.0,
        failure_type=failure_type,
    )


def _op(cases):
    passed = sum(1 for case in cases if case.success)
    return EvalOperatorResult(
        rel_path="level2/dynamic_quant",
        operator="DynamicQuant",
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=len(cases) - passed,
        skipped_cases=0,
        results=cases,
        pass_rate=passed / len(cases) if cases else 0.0,
        avg_speedup=0.0,
    )


def test_case_num_parses_full_case_id_suffix():
    assert _case_num_from_value("level2/dynamic_quant_17") == 17
    assert _case_num_from_value(18) == 18
    assert _case_num_from_value(None) == 0


def test_merge_results_matches_string_case_id_and_recounts_failures():
    correctness_ops = [
        _op([
            _case("level2/dynamic_quant_17", success=True),
            _case(
                "level2/dynamic_quant_9",
                success=False,
                failure_type="precision_mismatch",
                accuracy_result=AccuracyResult(passed=False),
            ),
            _case(
                "level2/dynamic_quant_6",
                success=False,
                failure_type="compile_runtime_error",
            ),
        ])
    ]
    # Reproduce the old aggregate behavior where runtime failures with no
    # accuracy_result were not counted in failed_cases.
    correctness_ops[0].failed_cases = 1

    performance_ops = [
        _op([
            _case(17, success=True, perf_result=PerfResult(elapsed_us=20.0)),
        ])
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]

    assert merged.passed_cases == 1
    assert merged.failed_cases == 2
    assert merged.skipped_cases == 0
    assert merged.pass_rate == 1 / 3
    assert merged.results[0].perf_result is not None
    assert merged.results[0].perf_result.elapsed_us == 20.0
