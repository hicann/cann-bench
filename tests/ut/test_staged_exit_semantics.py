# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Exit status describes completion of the requested evaluation stages."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kernel_eval import cli, staged_eval
from kernel_eval.base.result import PerfResult
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult


def result(success=True, failure_type=None, elapsed=2.0):
    return EvalCaseResult(
        case_id="mixed_1",
        operator="Mixed",
        rel_path="mixed",
        case_num=1,
        success=success,
        failure_type=failure_type,
        error_msg="diagnostic",
        perf_result=PerfResult(elapsed_us=elapsed) if elapsed is not None else None,
    )


def operator(case):
    failed = not case.success and case.failure_type not in (
        "reference_invalid",
        "cascade_device",
        "skipped",
    )
    return EvalOperatorResult(
        operator="Mixed",
        rel_path="mixed",
        total_cases=1,
        passed_cases=int(case.success),
        failed_cases=int(failed),
        skipped_cases=0,
        results=[case],
        pass_rate=float(case.success),
        avg_speedup=0.0,
    )


@pytest.mark.parametrize(
    "perf_kind, expected, expected_success",
    [
        ("success", 0, True),
        ("missing", 1, True),
        ("timeout", 1, True),
        ("compile_runtime_error", 1, True),
        ("zero", 1, True),
        ("precision_mismatch", 1, False),
        ("reference_invalid", 1, False),
    ],
)
def test_requested_performance_must_complete(
    monkeypatch, tmp_path, perf_kind, expected, expected_success
):
    if perf_kind == "missing":
        performance = []
    elif perf_kind == "zero":
        performance = [operator(result(elapsed=0.0))]
    elif perf_kind == "success":
        performance = [operator(result())]
    else:
        performance = [operator(result(False, perf_kind, elapsed=None))]
    args, saved, evaluate = setup_run(
        monkeypatch, tmp_path, [operator(result(elapsed=None))], performance
    )
    assert staged_eval.run(args) == expected
    final_case = saved[-1][1][0].results[0]
    assert final_case.success is expected_success
    assert evaluate.call_count == 3
    if perf_kind in ("missing", "timeout", "compile_runtime_error", "zero"):
        assert final_case.perf_recheck["status"] == "perf_unmeasured"
    if perf_kind == "reference_invalid":
        assert final_case.failure_type == "reference_invalid"
        assert saved[-1][1][0].failed_cases == 0


@pytest.mark.parametrize(
    "failure_type, expected",
    [(None, 0), ("precision_mismatch", 1), ("reference_invalid", 1)],
)
def test_no_perf_requires_no_timing_but_valid_correctness(
    monkeypatch, tmp_path, failure_type, expected
):
    correct = result(failure_type is None, failure_type, elapsed=None)
    args, saved, evaluate = setup_run(monkeypatch, tmp_path, [operator(correct)], [])
    args.no_perf = True
    assert staged_eval.run(args) == expected
    assert evaluate.call_count == 1
    assert saved[-1][0] == "correctness"


def test_correctness_reference_invalid_remains_nonzero_in_final(monkeypatch, tmp_path):
    args, saved, _ = setup_run(
        monkeypatch, tmp_path, [operator(result(False, "reference_invalid", None))], []
    )
    assert staged_eval.run(args) == 1
    assert saved[-1][1][0].reference_invalid_cases == 1


def test_compile_failure_stops_later_stages(monkeypatch, tmp_path):
    args, _, evaluate = setup_run(monkeypatch, tmp_path, [operator(result())], [])
    monkeypatch.setattr(staged_eval, "_compile", lambda *args: 1)
    assert staged_eval.run(args) == 1
    evaluate.assert_not_called()


@pytest.mark.parametrize("no_perf", [False, True])
@pytest.mark.parametrize("from_source", [False, True])
def test_requested_cases_without_matching_interfaces_fail(
    monkeypatch, tmp_path, capsys, no_perf, from_source
):
    from kernel_eval.data.package_manager import (
        InterfaceInfo,
        PackageInfo,
        PackageManager,
    )
    from kernel_eval.security.api_guard import APIGuard

    tasks_root = Path(__file__).resolve().parents[2] / "tasks"
    args = staged_eval.create_parser().parse_args(
        [
            "--task-dir",
            str(tasks_root),
            "--operator",
            "Exp",
            "--case-id",
            "1",
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    args.no_perf = no_perf
    args.source_dir = str(tmp_path / "candidate") if from_source else None
    monkeypatch.setattr(staged_eval, "detect_execution_hardware", lambda *args: "950pr")
    # Exercise real case selection and package/interface matching; no NPU or
    # package installation is needed to reproduce the zero-case success.
    selected = staged_eval._load_cases(args, str(tasks_root), None, filter_prefix="")
    assert [(case.operator, case.case_num) for case in selected] == [("Exp", 1)]
    monkeypatch.setattr(
        PackageManager,
        "scan_interfaces",
        lambda self: [InterfaceInfo("misspelled_exp", None, "misspelled_exp(x)")],
    )
    package = PackageInfo(source_dir=args.source_dir or "", whl_path="candidate.whl")
    monkeypatch.setattr(PackageManager, "build_packages", lambda *a, **kw: package)
    monkeypatch.setattr(PackageManager, "scan_source_dir", lambda *a: package)
    monkeypatch.setattr(PackageManager, "install_packages", lambda *a: True)
    monkeypatch.setattr(APIGuard, "snapshot", lambda self: None)
    monkeypatch.setattr(APIGuard, "verify", lambda self: None)
    monkeypatch.setattr(staged_eval, "_check_build_cann_consistency", lambda *a: None)
    evaluate = Mock(return_value=[])
    save = Mock(return_value=({"failed_cases": 0}, None))
    monkeypatch.setattr(staged_eval, "_evaluate_cases", evaluate)
    monkeypatch.setattr(staged_eval, "_save_report", save)

    assert staged_eval.run(args) == 1
    assert "interfaces do not match any requested cases" in capsys.readouterr().err
    evaluate.assert_not_called()
    save.assert_not_called()


def test_cli_reference_invalid_is_nonzero_and_capped():
    report = SimpleNamespace(
        failed_cases=0,
        operators=[SimpleNamespace(compilation_error=None, reference_invalid_cases=1)],
    )
    assert cli._compute_exit_code(report) == 1
    report.operators[0].reference_invalid_cases = 300
    assert cli._compute_exit_code(report) == 255


def setup_run(monkeypatch, tmp_path, correctness, performance):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    args = staged_eval.create_parser().parse_args(
        ["--task-dir", str(tmp_path), "--reports-dir", str(tmp_path)]
    )
    monkeypatch.setattr(staged_eval, "detect_execution_hardware", lambda *args: "950pr")
    monkeypatch.setattr(
        staged_eval, "_operator_rel_paths", lambda *args, **kwargs: ["mixed"]
    )
    case_spec = SimpleNamespace(rel_path="mixed", case_num=1)
    monkeypatch.setattr(staged_eval, "_load_cases", lambda *args, **kwargs: [case_spec])
    monkeypatch.setattr(staged_eval, "_compile", lambda *args: 0)
    monkeypatch.setattr(
        staged_eval, "_check_build_cann_consistency", lambda *args: None
    )
    monkeypatch.setattr(staged_eval, "_install_or_scan", lambda *args: ["Mixed"])
    # Missing RSS samples take the official fail-closed individual-profiler path.
    evaluate = Mock(side_effect=[correctness, [], performance])
    monkeypatch.setattr(staged_eval, "_evaluate_cases", evaluate)
    saved = []

    def save(args, cfg, ops, *, stage, contains_performance):
        saved.append((stage, ops))
        return {"failed_cases": sum(op.failed_cases for op in ops)}, None

    monkeypatch.setattr(staged_eval, "_save_report", save)
    return args, saved, evaluate


@pytest.mark.parametrize("profile", ["1", "2"])
def test_hardware_is_detected_before_loading_cases_with_score_profile(
    monkeypatch, tmp_path, profile
):
    import os
    from kernel_eval.base.score_weights import SCORE_WEIGHTS_ENV

    monkeypatch.delenv(SCORE_WEIGHTS_ENV, raising=False)
    args, _, evaluate = setup_run(monkeypatch, tmp_path, [], [])
    (tmp_path / "metadata" / "VERSION").write_text(f"{profile}.0.0\n", encoding="utf-8")
    events = []

    def detect(device, device_id):
        assert os.environ[SCORE_WEIGHTS_ENV] == profile
        events.append("hardware")
        return "950pr"

    def load(*args, **kwargs):
        assert events == ["hardware"]
        assert os.environ[SCORE_WEIGHTS_ENV] == profile
        events.append("cases")
        return [SimpleNamespace(rel_path="mixed", case_num=1)]

    monkeypatch.setattr(staged_eval, "detect_execution_hardware", detect)
    monkeypatch.setattr(staged_eval, "_load_cases", load)
    monkeypatch.setattr(staged_eval, "_compile", lambda *args: 1)
    assert staged_eval.run(args) == 1
    assert events == ["hardware", "cases"]
    assert SCORE_WEIGHTS_ENV not in os.environ
    evaluate.assert_not_called()
