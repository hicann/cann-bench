# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Reference failures must invalidate benchmarking, never penalize a candidate."""

import json
import pytest
import torch

from kernel_eval.base.result import (
    FAILURE_TYPE_REFERENCE_INVALID,
    is_compile_runtime_case_failure,
)
from kernel_eval.base.score_weights import SCORE_WEIGHTS_ENV, score_weights_context
from kernel_eval.checkers.mc2_three_way_checker import (
    check_mc2_references,
    compare_mc2_outputs,
)
from kernel_eval.config import Config
from kernel_eval.eval.results import (
    EvalCaseResult,
    EvalOperatorResult,
    summarize_case_results,
)
from kernel_eval.report.html_generator import render_html_report
from kernel_eval.report.report_generator import ReportGenerator
from kernel_eval.report.scoring import ScoringCalculator
from kernel_eval.report.summary_generator import (
    generate_summary,
    render_summary_markdown,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_guard_uses_native_rounding_despite_fp32_comparison(dtype):
    cpu = torch.tensor([31.23456, 0.00123456, -40.321], dtype=torch.float32)
    native = cpu.to(dtype)
    result = compare_mc2_outputs(native, native, cpu, 0, comparison_dtype="float32")[0]
    assert result.passed
    assert result.metadata["comparison_dtype"] == "float32"
    assert result.metadata["reference_check"]["dtype"] == str(dtype).removeprefix(
        "torch."
    )


def test_reference_guard_rejects_bad_reference_even_when_candidate_is_identical():
    cpu = torch.ones(8)
    native = cpu + 0.1
    result = compare_mc2_outputs(native, native, cpu, 0)[0]
    assert not result.passed
    assert result.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    assert not result.to_dict()["passed"]
    assert result.metadata["accepted_by"] is None
    assert "candidate_metrics" not in result.metadata
    assert "candidate unscored" in result.format_summary()
    json.dumps(result.to_dict(), allow_nan=False)


def test_independent_staged_health_accepts_sparse_gemm_cancellation_without_hiding_candidate_error():
    # Eight near-zero entries in a large GEMM are enough to trip the generic
    # small-value absolute-count gate, although the independent CPU cascade
    # and the NPU reference have essentially identical global errors.
    cpu = torch.linspace(1, 64, 8192, dtype=torch.float32) + 0.1234
    cpu[:8] = 0.0002
    staged = cpu.bfloat16().float()
    native = staged.bfloat16()
    native[:8] = 0.00022
    assert not check_mc2_references(native, cpu, 0)[0].passed

    result = check_mc2_references(native, cpu, 0, cascade_golden=staged)[0]
    assert result.passed, result.error_msg
    health = result.metadata["reference_check"]
    assert health["policy"] == "independent_cpu_staged_mc2_ratio"
    assert health["reference"] == "cpu_staged_cascade"
    assert health["cpu_stage_relative_rmse"] <= health["cpu_stage_relative_rmse_limit"]
    assert health["npu_stage_relative_rmse"] <= health["npu_stage_relative_rmse_limit"]
    assert (
        compare_mc2_outputs(native + 10, native, cpu, 0, cascade_golden=staged)[
            0
        ].passed
        is False
    )


def test_staged_health_rejects_systematic_error_despite_mc2_ratio():
    # B/G error is exactly twice H/G, so the upstream ratio alone accepts B.
    # B and the independent CPU stage disagree by 1.5% of the FP32 scale.
    cpu = torch.full((8192,), 100.0, dtype=torch.float32)
    staged = torch.full_like(cpu, 100.5)
    wrong_npu = torch.full((8192,), 99.0, dtype=torch.bfloat16)
    result = check_mc2_references(wrong_npu, cpu, 0, cascade_golden=staged)[0]
    assert not result.passed
    assert (
        "npu_vs_cpu_staged.relative_rmse"
        in result.metadata["reference_check"]["failed_metrics"]
    )


def test_staged_health_keeps_absolute_tolerance_for_zero_reference():
    cpu = torch.zeros(8192)
    native = torch.full((8192,), 2e-6, dtype=torch.bfloat16)
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert result.passed, result.error_msg


def test_staged_health_rejects_broken_reference_and_broken_cpu_stage():
    cpu = torch.linspace(1, 64, 8192, dtype=torch.float32) + 0.1234
    staged = cpu.bfloat16().float()
    wrong_npu = (staged + 10).bfloat16()
    result = check_mc2_references(wrong_npu, cpu, 0, cascade_golden=staged)[0]
    assert not result.passed
    assert result.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    assert result.metadata["reference_check"]["failed_metrics"]

    wrong_stage = staged * 2
    result = check_mc2_references(
        wrong_stage.bfloat16(), cpu, 0, cascade_golden=wrong_stage
    )[0]
    assert not result.passed
    assert (
        "cpu_staged_cascade.relative_rmse"
        in result.metadata["reference_check"]["failed_metrics"]
    )


def test_nonfinite_cpu_reference_is_invalid_even_when_cascade_matches_it():
    cpu = torch.tensor([1.0, float("nan")])
    native = cpu.bfloat16()
    result = check_mc2_references(native, cpu, 0)[0]
    assert not result.passed
    assert result.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID


def test_fixed_reference_diagnostics_show_physical_dtype_threshold():
    cpu = torch.ones(8)
    native = cpu.bfloat16()
    result = check_mc2_references(native, cpu, 0)[0]
    assert result.metadata["reference_check"]["threshold"] == 2**-7


def test_existing_cancellation_allowance_is_preserved():
    cpu = torch.tensor([1.0, 2.0, 0.001])
    native = torch.tensor([1.0, 2.0, 0.0])
    assert compare_mc2_outputs(native, native, cpu, 0)[0].passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_reference_mismatch_is_never_rescued(value):
    result = compare_mc2_outputs(
        torch.ones(1), torch.tensor([value]), torch.ones(1), 0
    )[0]
    assert result.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    assert not result.passed


def test_fp8_auxiliary_uses_fp32_guard_and_tolerance():
    cpu = torch.ones(4)
    good = cpu.to(torch.float8_e4m3fn)
    bad = torch.full((4,), 1.125).to(torch.float8_e4m3fn)
    assert compare_mc2_outputs(good, good, cpu, 0)[0].passed
    candidate_bad = compare_mc2_outputs(bad, good, cpu, 0)[0]
    assert not candidate_bad.passed
    assert candidate_bad.metadata["relative_tolerance"] == 2**-23
    reference_bad = compare_mc2_outputs(bad, bad, cpu, 0)[0]
    assert reference_bad.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID


def test_promoted_low_precision_output_does_not_receive_loose_fallback():
    cpu = torch.ones(4)
    native = cpu.half()
    candidate = (cpu + 2**-10).half()
    result = compare_mc2_outputs(candidate, native, cpu, 0, comparison_dtype="float32")[
        0
    ]
    assert not result.passed


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        pytest.param({"success": True, "accuracy": None}, False, id="explicit-success"),
        pytest.param(
            {"status": "success", "accuracy": None}, False, id="status-success"
        ),
        pytest.param(
            {
                "success": False,
                "status": "success",
                "failure_type": "compile_runtime_error",
            },
            True,
            id="explicit-failure-takes-precedence",
        ),
        pytest.param(
            {"success": False, "failure_type": FAILURE_TYPE_REFERENCE_INVALID},
            False,
            id="invalid-reference-is-not-runtime-failure",
        ),
    ],
)
def test_serialized_case_compile_runtime_classification(case, expected):
    assert is_compile_runtime_case_failure(case) is expected


@pytest.mark.parametrize("profile", ["1", "2"])
def test_reference_invalid_round_trip_scoring_and_rendering(
    tmp_path, monkeypatch, profile
):
    monkeypatch.delenv(SCORE_WEIGHTS_ENV, raising=False)
    bench_root = tmp_path / "tasks"
    (bench_root / "metadata").mkdir(parents=True)
    (bench_root / "metadata" / "VERSION").write_text(
        f"{profile}.0.0\n", encoding="utf-8"
    )
    case = EvalCaseResult(
        "op_1",
        "level4/op",
        "Op",
        1,
        False,
        error_msg="broken CPU/cascade reference",
        failure_type=FAILURE_TYPE_REFERENCE_INVALID,
        memory_peak_mb=1234.0,
        memory_baseline_mb=512.0,
    )
    case = EvalCaseResult.from_dict(case.to_dict())
    assert (case.memory_peak_mb, case.memory_baseline_mb) == (1234.0, 512.0)
    assert not is_compile_runtime_case_failure(case)
    assert not is_compile_runtime_case_failure(case.to_dict())
    counts = summarize_case_results([case])
    assert (counts.failed, counts.reference_invalid) == (0, 1)
    op = EvalOperatorResult("level4/op", "Op", 1, 0, 0, 0, [case], 0.0, 0.0)
    score = ScoringCalculator(weights=profile).calculate_operator_score(op)
    assert score.total_score is None and score.compilation_score is None
    assert score.bonus_score is None
    assert score.compile_runtime_fail_cases == 0
    assert ScoringCalculator().calculate_overall_score([score]) is None
    config = Config(tasks_root=str(bench_root), device_type="cpu")
    monkeypatch.setattr(
        "kernel_eval.report.report_generator.collect_setup_info", lambda config: {}
    )
    generator = ReportGenerator(str(tmp_path), config=config)
    generator.add_operator_result(op)
    report = generator.generate()
    assert report.score_weight_profile == profile
    assert report.overall_score is None
    assert report.summary["reference_invalid_cases"] == 1
    assert report.failed_cases == 0
    assert report.operators[0].cases[0].status == FAILURE_TYPE_REFERENCE_INVALID
    assert report.operators[0].cases[0].memory_peak_mb == 1234.0
    assert report.operators[0].cases[0].memory_baseline_mb == 512.0
    assert "不可判定" in generator._generate_markdown_content(report)
    assert "参考失效" in render_html_report(report, {}, str(tmp_path / "absent.html"))
    with score_weights_context(bench_root):
        summary = generate_summary(report.to_dict())
    assert summary.benchmark_total_score is None
    assert summary.operators[0].bonus_score is None
    assert "不可判定" in render_summary_markdown(summary)
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "statuses,expected_rate,expected_genuine_rate,expected_failed",
    [
        (["success", "reference_invalid"], 1.0, 1.0, 0),
        (["reference_invalid", "reference_invalid"], 0.0, 0.0, 0),
        (
            ["success", "reference_invalid", "cascade_device", "skipped", "genuine"],
            0.25,
            0.5,
            1,
        ),
        (["reference_invalid", "cascade_device"], 0.0, 0.0, 0),
        (["success", "cascade_device", "genuine"], 1 / 3, 0.5, 1),
    ],
)
def test_reference_valid_pass_rates_match_report_formats_and_summary(
    tmp_path,
    monkeypatch,
    capsys,
    statuses,
    expected_rate,
    expected_genuine_rate,
    expected_failed,
):
    from kernel_eval.base.result import PerfResult

    monkeypatch.delenv(SCORE_WEIGHTS_ENV, raising=False)
    bench_root = tmp_path / "tasks"
    (bench_root / "metadata").mkdir(parents=True)
    (bench_root / "metadata" / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    cases = [
        EvalCaseResult(
            f"op_{index}",
            "level4/op",
            "Op",
            index,
            status == "success",
            failure_type=None if status == "success" else status,
            perf_result=PerfResult(elapsed_us=10.0) if status == "success" else None,
            baseline_perf_us=20.0,
            t_hw_us=1.0,
        )
        for index, status in enumerate(statuses, 1)
    ]
    counts = summarize_case_results(cases)
    op = EvalOperatorResult(
        "level4/op",
        "Op",
        len(cases),
        counts.passed,
        counts.failed,
        counts.skipped,
        cases,
        counts.pass_rate,
        counts.avg_speedup,
    )
    monkeypatch.setattr(
        "kernel_eval.report.report_generator.collect_setup_info", lambda config: {}
    )
    generator = ReportGenerator(
        str(tmp_path), config=Config(tasks_root=str(bench_root), device_type="cpu")
    )
    generator.add_operator_result(op)
    report = generator.generate()
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["total_cases"] == len(cases)
    assert payload["failed_cases"] == expected_failed
    assert payload["summary"]["pass_rate"] == pytest.approx(expected_rate)
    assert payload["summary"]["genuine_pass_rate"] == pytest.approx(
        expected_genuine_rate
    )
    assert payload["operators"][0]["pass_rate"] == pytest.approx(expected_rate)
    assert f"| 通过率 | {expected_rate:.2%} |" in generator._generate_markdown_content(
        report
    )
    generator.print_summary(report)
    console = capsys.readouterr().out
    assert f"通过率: {expected_rate:.2%}" in console
    if counts.skipped:
        assert f"真实通过率（排除级联）: {expected_genuine_rate:.2%}" in console
    if counts.reference_invalid:
        assert report.overall_score is None

    # Both serialized shapes may come from a report written before the explicit
    # reference-invalid count was introduced. Never add that count twice.
    for serialized in (op.to_dict(), payload["operators"][0]):
        for explicit_count in (True, False):
            entry = dict(serialized)
            if not explicit_count:
                entry.pop("reference_invalid_cases")
            with score_weights_context(bench_root):
                summary = generate_summary({"operators": [entry]})
            assert summary.total_cases == len(cases)
            assert summary.overall_pass_rate == pytest.approx(expected_rate)
            assert summary.operators[0].pass_rate == pytest.approx(expected_rate)
            assert summary.operators[0].failed_cases == expected_failed
            assert (
                summary.operators[0].reference_invalid_cases == counts.reference_invalid
            )
            if counts.reference_invalid:
                assert summary.benchmark_total_score is None


@pytest.mark.parametrize(
    "statuses,expected_rate,expected_failed",
    [
        (["success", "reference_invalid"], 1.0, 0),
        (["reference_invalid", "reference_invalid"], 0.0, 0),
        (
            ["success", "reference_invalid", "cascade_device", "skipped", "genuine"],
            0.25,
            1,
        ),
        (["success", "genuine"], 0.5, 1),
        ([], 0.0, 0),
    ],
)
def test_evaluator_run_cases_excludes_reference_invalid_from_pass_rate(
    statuses, expected_rate, expected_failed
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from kernel_eval.eval.evaluator import Evaluator

    cases = [
        SimpleNamespace(attrs={}, get_case_id_str=Mock(return_value=f"op_{index}"))
        for index in range(1, len(statuses) + 1)
    ]
    results = [
        EvalCaseResult(
            f"op_{index}",
            "level4/op",
            "Op",
            index,
            status == "success",
            failure_type=None if status == "success" else status,
        )
        for index, status in enumerate(statuses, 1)
    ]
    evaluator = object.__new__(Evaluator)
    evaluator.config = Config(device_type="cpu", perf_batch_cases=False)
    evaluator.operator_matcher = Mock()
    evaluator.incremental_output_path = None
    evaluator.device_manager = Mock()
    evaluator.device_manager.health_check.return_value = True
    evaluator._cleanup_memory = Mock()
    evaluator.evaluate_case = Mock(side_effect=results)

    result = evaluator.run_cases(cases, "Op", "level4/op")

    assert evaluator.evaluate_case.call_count == len(cases)
    assert result.total_cases == len(cases)
    assert result.passed_cases == statuses.count("success")
    assert result.failed_cases == expected_failed
    assert result.results == results
    assert result.pass_rate == pytest.approx(expected_rate)
    assert result.to_dict()["pass_rate"] == pytest.approx(expected_rate)


def test_summary_combines_reference_valid_denominators_across_operators(monkeypatch):
    monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
    summary = generate_summary(
        {
            "operators": [
                {
                    "operator": "Mixed",
                    "total_cases": 3,
                    "passed_cases": 1,
                    "failed_cases": 1,
                    "results": [
                        {"success": True, "perf": {"perf_score": 1.0}},
                        {"success": False, "failure_type": "genuine"},
                        {"success": False, "failure_type": "reference_invalid"},
                    ],
                },
                {
                    "operator": "AllInvalid",
                    "total_cases": 2,
                    "passed_cases": 0,
                    "failed_cases": 0,
                    "reference_invalid_cases": 2,
                    "cases": [
                        {
                            "status": "reference_invalid",
                            "failure_type": "reference_invalid",
                        },
                        {
                            "status": "reference_invalid",
                            "failure_type": "reference_invalid",
                        },
                    ],
                },
            ]
        }
    )
    assert summary.total_cases == 5
    assert summary.total_passed == 1
    assert summary.overall_pass_rate == 0.5
    assert [op.pass_rate for op in summary.operators] == [0.5, 0.0]
    assert summary.benchmark_total_score is None


@pytest.mark.parametrize(
    "module_name", ["kernel_eval.report.scoring", "kernel_eval.benches.cann_scoring"]
)
@pytest.mark.parametrize("profile", ["1", "2"])
def test_public_scoring_entrypoints_preserve_unavailable_score(module_name, profile):
    import importlib

    scoring = importlib.import_module(module_name)
    invalid = EvalCaseResult(
        "op_1", "level4/op", "Op", 1, False, failure_type=FAILURE_TYPE_REFERENCE_INVALID
    )
    operator = EvalOperatorResult("level4/op", "Op", 1, 0, 0, 0, [invalid], 0.0, 0.0)
    calculator = scoring.ScoringCalculator(weights=profile)
    score = calculator.calculate_operator_score(operator)
    assert score.reference_invalid_cases == 1
    assert score.compile_runtime_fail_cases == 0
    assert score.to_dict()["total_score"] is None
    assert score.to_dict()["bonus_score"] is None
    assert score.score_error_code == FAILURE_TYPE_REFERENCE_INVALID
    normal = scoring.OperatorScoreInfo(
        operator="Other", rel_path="level1/other", total_score=50.0
    )
    assert calculator.calculate_overall_score([normal, score]) is None
    assert calculator.calculate_average_score([normal, score]) is None
    assert calculator.calculate_level_score([normal, score], "level4") is None
    assert calculator.calculate_level_score([normal, score], "level1") == 50.0
    assert [
        entry["operator"] for entry in calculator.calculate_ranking([normal, score])
    ] == ["Other"]
    invalid_score = scoring.aggregate_eq4(
        True, 1, [(False, None)], weights=profile, n_reference_invalid=1
    )
    assert invalid_score["total_score"] is None
    assert invalid_score["bonus_score"] is None
    assert (
        scoring.aggregate_eq4(True, 1, [(True, 1.0)], weights=profile)["total_score"]
        == 100.0
    )


def test_public_cann_scheme_reference_metadata_invalidates_score():
    from kernel_eval.base.result import PerfResult
    from kernel_eval.base.scoring import CaseScoreInfo
    from kernel_eval.benches.cann import CannScoringScheme

    scheme = CannScoringScheme()
    metadata = {"failure_type": FAILURE_TYPE_REFERENCE_INVALID, "t_hw_us": 10.0}
    assert (
        scheme.calculate_case_score(
            PerfResult(elapsed_us=20.0, metadata=metadata), 100.0
        )
        is None
    )
    assert scheme.aggregate_operator_scores([CaseScoreInfo(metadata=metadata)]) is None
    assert (
        scheme.aggregate_operator_scores([CaseScoreInfo(passed=True, score=1.0)])
        == 100.0
    )


def test_generic_html_unavailable_score_does_not_claim_reference_invalid():
    from kernel_eval.report.report_generator import EvalReport

    report = EvalReport(
        "1",
        "1",
        "no_accuracy",
        "",
        "cpu",
        0,
        0,
        0,
        0,
        None,
        [],
        {"pass_rate": None, "genuine_pass_rate": None, "cascade_cases": None},
    )
    html = render_html_report(report)
    assert "N/A" in html
    assert "参考失效" not in html
