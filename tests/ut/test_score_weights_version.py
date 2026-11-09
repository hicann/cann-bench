#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""题集大版本解析、评测入口和报告溯源回归测试。"""

import pytest

from kernel_eval.base.perf_metric import (
    PERF_METRIC_ENV,
)
from kernel_eval.base.score_weights import (
    SCORE_WEIGHTS_ENV,
    WEIGHT_PROFILES,
    resolve_score_weights,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(SCORE_WEIGHTS_ENV, raising=False)
    monkeypatch.delenv(PERF_METRIC_ENV, raising=False)


def _agg_fns():
    from kernel_eval.benches.cann_scoring import aggregate_eq4 as bench_agg
    from kernel_eval.report.scoring import aggregate_eq4 as report_agg

    return bench_agg, report_agg


def _bench_root(tmp_path, version):
    root = tmp_path / version.splitlines()[0].replace("/", "_")
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "VERSION").write_text(version, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "version,profile",
    [
        ("1.1.2\n", "1"),
        ("1.9.99\n", "1"),
        ("2.0.0\n# requires-framework: >=1.0.0\n", "2"),
        ("2.9.99\n", "2"),
    ],
)
def test_content_default(tmp_path, version, profile):
    from kernel_eval.base.score_weights import score_weights_context

    root = _bench_root(tmp_path, version)
    with score_weights_context(root) as actual:
        assert actual == profile
        assert resolve_score_weights() == WEIGHT_PROFILES[profile]
        for aggregate in _agg_fns():
            result = aggregate(
                compile_passed=True, total_cases=1, case_scores=[(True, 0.5)]
            )
            expected = WEIGHT_PROFILES[profile]
            assert result["compilation_score"] == expected.wc * 100
            assert result["function_score"] == expected.wf * 100
            assert result["performance_score"] == expected.wp * 0.5 * 100
            assert result["bonus_score"] == expected.wb * 100
    with pytest.raises(ValueError):
        resolve_score_weights()


@pytest.mark.parametrize(
    "version",
    [
        None,
        "",
        "\n2.0.0",
        "# comment\n2.0.0",
        "2",
        "2.0",
        "2.0.0garbage",
        "2.0.0.1",
        "02.0.0",
        "-2.0.0",
        "3.0.0",
    ],
)
def test_invalid_content_fails(tmp_path, version):
    from kernel_eval.base.score_weights import score_weights_context

    (tmp_path / "metadata").mkdir()
    if version is not None:
        (tmp_path / "metadata" / "VERSION").write_text(version)
    with pytest.raises(ValueError):
        with score_weights_context(tmp_path):
            pytest.fail("invalid version must not start evaluation")


@pytest.mark.parametrize("profile", ["season1", "season2", "3", "", "2.0.0", "banana"])
def test_invalid_profile_rejected(profile):
    with pytest.raises(ValueError):
        resolve_score_weights(profile)


def test_override_priority(tmp_path, monkeypatch):
    from kernel_eval.base.score_weights import (
        resolve_score_weight_profile,
        score_weights_context,
    )

    root = _bench_root(tmp_path, "2.0.0\n")
    monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
    with score_weights_context(root) as profile:
        assert profile == "1"
        assert resolve_score_weights() == WEIGHT_PROFILES["1"]
        assert resolve_score_weights("2") == WEIGHT_PROFILES["2"]
        for aggregate in _agg_fns():
            assert (
                aggregate(
                    compile_passed=True,
                    total_cases=1,
                    case_scores=[(True, 0.0)],
                    weights="2",
                )["compilation_score"]
                == 5
            )
    assert resolve_score_weight_profile(bench_root=tmp_path / "missing") == "1"
    monkeypatch.setenv(SCORE_WEIGHTS_ENV, "invalid")
    assert resolve_score_weight_profile("2", bench_root=root) == "2"
    with pytest.raises(ValueError):
        resolve_score_weight_profile(bench_root=root)


def test_sequential_evaluations_and_exception_restore_environment(tmp_path):
    import os
    from kernel_eval.base.score_weights import score_weights_context

    for version, expected in [("1.1.2", "1"), ("2.0.0", "2"), ("1.9.0", "1")]:
        root = _bench_root(tmp_path, version)
        with pytest.raises(RuntimeError):
            with score_weights_context(root):
                assert os.environ[SCORE_WEIGHTS_ENV] == expected
                with score_weights_context(root):
                    assert resolve_score_weights() == WEIGHT_PROFILES[expected]
                raise RuntimeError("evaluation failed")
        assert SCORE_WEIGHTS_ENV not in os.environ


def test_children_inherit_content_profile(tmp_path):
    import os
    import subprocess
    import sys
    from kernel_eval.base.score_weights import score_weights_context

    with score_weights_context(_bench_root(tmp_path, "1.1.2")):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import os; print(os.environ["CANN_BENCH_SCORE_WEIGHTS"])',
            ],
            check=True,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
    assert result.stdout.strip() == "1"


@pytest.mark.parametrize(
    "version,profile,compilation", [("1.1.2", "1", 20), ("2.0.0", "2", 5)]
)
@pytest.mark.parametrize("metric", ["eq3", "eq3_log"])
def test_report_records_effective_scoring(
    tmp_path, monkeypatch, version, profile, compilation, metric
):
    # External bundle version differs from the framework's bundled tasks version.
    import json
    from kernel_eval.config import Config
    from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult
    from kernel_eval.report.report_generator import ReportGenerator

    monkeypatch.setenv(PERF_METRIC_ENV, metric)
    monkeypatch.setattr(
        "kernel_eval.report.report_generator.collect_setup_info", lambda config: {}
    )
    root = _bench_root(tmp_path, version + "\n# requires-framework: >=1.0.0\n")
    generator = ReportGenerator(
        output_dir=str(tmp_path / "reports"),
        config=Config(tasks_root=str(root), device_type="cpu"),
    )
    generator.add_operator_result(
        EvalOperatorResult(
            rel_path="level1/test",
            operator="test",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            skipped_cases=0,
            pass_rate=0,
            avg_speedup=0,
            results=[
                EvalCaseResult(
                    case_id="level1/test_0",
                    case_num=0,
                    operator="test",
                    rel_path="level1/test",
                    success=False,
                    failure_type="precision",
                )
            ],
        )
    )
    report = generator.generate()
    payload = json.loads(generator.save_json(report).read_text())
    assert payload["score_weight_profile"] == profile
    assert payload["perf_metric"] == metric
    assert payload["tasks_version"] == version
    assert payload["operators"][0]["compilation_score"] == compilation
    # Metadata is a snapshot; serializing later must not re-read environment.
    monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2" if profile == "1" else "1")
    monkeypatch.setenv(PERF_METRIC_ENV, "eq3" if metric == "eq3_log" else "eq3_log")
    assert report.to_dict()["score_weight_profile"] == profile
    assert report.to_dict()["perf_metric"] == metric


@pytest.mark.parametrize("entry", ["cli", "staged"])
@pytest.mark.parametrize("version,profile", [("1.1.2", "1"), ("2.0.0", "2")])
def test_entry_uses_actual_bench_root(tmp_path, monkeypatch, entry, version, profile):
    import os
    from argparse import Namespace
    from kernel_eval import cli, staged_eval

    root = _bench_root(tmp_path, version)

    def resolver(*args):
        return str(root), None

    monkeypatch.setattr(cli, "resolve_task_dir", resolver)
    monkeypatch.setattr("kernel_eval.utils.path_resolver.resolve_task_dir", resolver)
    args = Namespace(task_dir=str(root), bench_name="cann", device="npu")

    def evaluate(args, bench_root, filter_prefix):
        assert bench_root == str(root)
        assert os.environ[SCORE_WEIGHTS_ENV] == profile
        return 17

    if entry == "cli":
        monkeypatch.setattr(cli, "_eval_with_score_profile", evaluate)
        assert cli.cmd_eval(args) == 17
    else:
        monkeypatch.setattr(staged_eval, "_run_with_score_profile", evaluate)
        assert staged_eval.run(args) == 17
    assert SCORE_WEIGHTS_ENV not in os.environ


@pytest.mark.parametrize("entry", ["cli", "staged"])
def test_entry_fails_before_evaluation(tmp_path, monkeypatch, entry):
    from argparse import Namespace
    from kernel_eval import cli, staged_eval

    root = _bench_root(tmp_path, "3.0.0")

    def resolver(*args):
        return str(root), None

    monkeypatch.setattr(cli, "resolve_task_dir", resolver)
    monkeypatch.setattr("kernel_eval.utils.path_resolver.resolve_task_dir", resolver)
    args = Namespace(task_dir=str(root), bench_name="cann", device="npu")
    with pytest.raises(ValueError, match="unknown score weight profile"):
        (cli.cmd_eval if entry == "cli" else staged_eval.run)(args)


@pytest.mark.parametrize("version", [None, "3.0.0", "invalid"])
def test_environment_can_override_unavailable_content(tmp_path, monkeypatch, version):
    from kernel_eval.base.score_weights import score_weights_context

    if version is not None:
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "VERSION").write_text(version)
    monkeypatch.setenv(SCORE_WEIGHTS_ENV, "1")
    with score_weights_context(tmp_path):
        assert resolve_score_weights() == WEIGHT_PROFILES["1"]
    assert resolve_score_weights() == WEIGHT_PROFILES["1"]


@pytest.mark.parametrize(
    "stage", ["compile_failure", "correctness", "performance", "final"]
)
def test_staged_reports_keep_scoring_metadata(tmp_path, monkeypatch, stage):
    import json
    from argparse import Namespace
    from kernel_eval.config import Config
    from kernel_eval.staged_eval import _save_report

    root = _bench_root(tmp_path, "1.1.2\n# requires-framework: >=1.0.0\n")
    monkeypatch.setattr(
        "kernel_eval.report.report_generator.collect_setup_info", lambda config: {}
    )
    monkeypatch.setenv(SCORE_WEIGHTS_ENV, "2")
    monkeypatch.setenv(PERF_METRIC_ENV, "eq3_log")
    args = Namespace(
        reports_dir=str(tmp_path / "reports"), eval_code="test", bench_name="cann"
    )
    payload, _ = _save_report(
        args,
        Config(tasks_root=str(root), device_type="cpu"),
        [],
        stage=stage,
        contains_performance=stage in ("performance", "final"),
    )
    saved = json.loads((tmp_path / "reports" / f"test_{stage}.json").read_text())
    for report in (payload, saved):
        assert report["score_weight_profile"] == "2"
        assert report["perf_metric"] == "eq3_log"
        assert report["tasks_version"] == "1.1.2"
        assert report["result_stage"] == stage


def test_complete_explicit_weights_work_without_default():
    from kernel_eval.benches.cann_scoring import ScoringCalculator as BenchCalc
    from kernel_eval.report.scoring import ScoringCalculator as ReportCalc

    explicit = dict(wc=0.2, wf=0.3, wp=0.5, wb=0.0)
    for aggregate in _agg_fns():
        result = aggregate(
            compile_passed=True, total_cases=1, case_scores=[(True, 0.5)], **explicit
        )
        assert result["total_score"] == 75.0
        # Invalid named profiles must still fail even with numeric overrides.
        with pytest.raises(ValueError):
            aggregate(
                compile_passed=True,
                total_cases=1,
                case_scores=[(True, 0.5)],
                weights="3",
                **explicit,
            )
    for calculator in (BenchCalc, ReportCalc):
        assert calculator(**explicit)._effective_weights() == (0.2, 0.3, 0.5, 0.0)


def test_example_fixture_has_supported_default_profile():
    from pathlib import Path
    from kernel_eval.base.score_weights import score_weights_context

    root = Path(__file__).resolve().parents[2] / "examples" / "tasks"
    with score_weights_context(root) as profile:
        assert profile == "1"
        assert resolve_score_weights() == WEIGHT_PROFILES[profile]


def test_unknown_profile_explains_override(tmp_path):
    from kernel_eval.base.score_weights import score_weights_context

    root = _bench_root(tmp_path, "0.1.0")
    with pytest.raises(ValueError, match=SCORE_WEIGHTS_ENV):
        with score_weights_context(root):
            pytest.fail("unsupported version must not start evaluation")
