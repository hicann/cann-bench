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


import json
import os
from argparse import Namespace
from pathlib import Path
from unittest import mock

from kernel_eval.base.result import AccuracyResult, PerfResult
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult
from kernel_eval.staged_eval import (
    _case_num_from_value,
    _cann_home_dirname,
    _check_build_cann_consistency,
    _compile_failure_operator_filter,
    _detect_build_cann_dirname,
    _merge_results,
    _partition_performance_case_keys,
    create_parser,
)


def _case(
    case_num,
    *,
    success,
    failure_type=None,
    accuracy_result=None,
    perf_result=None,
    error_msg=None,
    memory_peak_mb=None,
):
    return EvalCaseResult(
        case_id=f"level2/dynamic_quant_{case_num}",
        rel_path="level2/dynamic_quant",
        operator="DynamicQuant",
        case_num=case_num,
        success=success,
        accuracy_result=accuracy_result,
        perf_result=perf_result,
        error_msg=error_msg,
        baseline_perf_us=100.0,
        t_hw_us=10.0,
        failure_type=failure_type,
        memory_peak_mb=memory_peak_mb,
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


def test_baseline_source_survives_merge_transport_and_reports(tmp_path):
    from kernel_eval.report.report_generator import (
        EvalReport,
        EvalResult,
        OperatorReport,
        ReportGenerator,
    )
    from kernel_eval.report.html_generator import render_html_report

    correct = _case(1, success=True)
    perf = _case(1, success=True, perf_result=PerfResult(elapsed_us=20))
    perf.baseline_source = {
        "baseline_perf_us": {
            "requested_hardware": "910c",
            "source_hardware": "910b2",
            "fallback_used": True,
            "source": "metadata/910b2.json<script>",
        }
    }
    merged = _merge_results([_op([correct])], [_op([perf])])[0].results[0]
    restored = EvalCaseResult.from_dict(merged.to_dict())
    assert restored.baseline_source == perf.baseline_source
    assert restored.get_speedup() == 5
    case = EvalResult.from_eval_case_result(restored)
    assert case.baseline_source == perf.baseline_source
    report = EvalReport(
        "1",
        "1",
        "source",
        "",
        "npu:0",
        1,
        1,
        1,
        0,
        50,
        [
            OperatorReport(
                rel_path=case.rel_path, total_cases=1, passed_cases=1, cases=[case]
            )
        ],
        {"pass_rate": 1.0},
    )
    assert (
        report.to_dict()["operators"][0]["cases"][0]["baseline_source"]
        == perf.baseline_source
    )
    html = render_html_report(report)
    markdown = ReportGenerator(output_dir=str(tmp_path))._generate_markdown_content(
        report
    )
    for text in (html, markdown):
        assert "跨平台复用" in text
        assert "910c" in text and "910b2" in text
        assert "metadata/910b2.json&lt;script&gt;" in text
        assert "metadata/910b2.json<script>" not in text
    assert EvalCaseResult.from_dict({"case_id": "old"}).baseline_source == {}
    report.summary["reference_invalid_cases"] = 1
    invalid_html = render_html_report(report)
    assert "跨平台复用" in invalid_html
    assert "metadata/910b2.json&lt;script&gt;" in invalid_html


def test_compile_failure_filter_uses_selected_operators():
    args = Namespace(operator=None, selected_operators=["conv_3d_backprop_filter"])
    assert _compile_failure_operator_filter(args) == ["conv_3d_backprop_filter"]

    args.operator = "Conv3DBackpropFilter"
    assert _compile_failure_operator_filter(args) == ["Conv3DBackpropFilter"]


def test_perf_batch_cases_defaults_on_and_can_be_disabled():
    parser = create_parser()

    assert parser.parse_args([]).perf_batch_cases is True
    assert parser.parse_args(["--no-perf-batch-cases"]).perf_batch_cases is False


def test_perf_batch_memory_limit_defaults_and_can_be_overridden():
    parser = create_parser()

    assert parser.parse_args([]).perf_batch_case_memory_limit_mb == 32 * 1024
    assert (
        parser.parse_args(
            [
                "--perf-batch-case-memory-limit-mb",
                "24576",
            ]
        ).perf_batch_case_memory_limit_mb
        == 24576
    )


def test_partition_performance_cases_by_correctness_peak_memory():
    operator_results = [
        _op(
            [
                _case(1, success=True, memory_peak_mb=1024),
                _case(2, success=True, memory_peak_mb=4096),
                _case(3, success=True, memory_peak_mb=None),
                _case(4, success=False, memory_peak_mb=128),
            ]
        )
    ]

    batched, individual = _partition_performance_case_keys(
        operator_results, batch_enabled=True, memory_limit_mb=2048
    )

    assert batched == {("level2/dynamic_quant", 1)}
    assert individual == {
        ("level2/dynamic_quant", 2),
        ("level2/dynamic_quant", 3),
    }


def test_partition_memory_gate_can_be_disabled():
    operator_results = [
        _op(
            [
                _case(1, success=True, memory_peak_mb=None),
                _case(2, success=True, memory_peak_mb=999999),
            ]
        )
    ]

    batched, individual = _partition_performance_case_keys(
        operator_results, batch_enabled=True, memory_limit_mb=0
    )

    assert len(batched) == 2
    assert individual == set()


def test_case_memory_measurement_survives_result_and_report_conversion():
    case = _case(1, success=True, memory_peak_mb=1536.5)
    case.memory_baseline_mb = 256.0

    roundtrip = EvalCaseResult.from_dict(case.to_dict())
    assert roundtrip.memory_peak_mb == 1536.5
    assert roundtrip.memory_baseline_mb == 256.0

    from src.kernel_eval.report.report_generator import EvalResult

    report_case = EvalResult.from_eval_case_result(roundtrip)
    assert report_case.memory_peak_mb == 1536.5
    assert report_case.memory_baseline_mb == 256.0


def test_max_cases_per_task_unit_defaults_and_can_be_overridden():
    parser = create_parser()

    assert parser.parse_args([]).max_cases_per_task_unit == 64
    assert (
        parser.parse_args(
            [
                "--max-cases-per-task-unit",
                "32",
            ]
        ).max_cases_per_task_unit
        == 32
    )


def test_merge_results_matches_string_case_id_and_recounts_failures():
    correctness_ops = [
        _op(
            [
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
            ]
        )
    ]
    # Reproduce the old aggregate behavior where runtime failures with no
    # accuracy_result were not counted in failed_cases.
    correctness_ops[0].failed_cases = 1

    performance_ops = [
        _op(
            [
                _case(17, success=True, perf_result=PerfResult(elapsed_us=20.0)),
            ]
        )
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]

    assert merged.passed_cases == 1
    assert merged.failed_cases == 2
    assert merged.skipped_cases == 0
    assert merged.pass_rate == 1 / 3
    assert merged.results[0].perf_result is not None
    assert merged.results[0].perf_result.elapsed_us == 20.0


def test_merge_flags_perf_stage_precision_flip_as_precision_failure():
    """correctness 过、performance 精度复检翻车 → 视为该 case 精度错误 + 打标签。

    输入与 golden 两阶段一致，唯一变量是 NPU kernel 输出，故这类翻车等价于
    非确定性算子，应扣精度分（precision_mismatch）、无性能分，并在 results.json
    的 perf_recheck 字段中留痕。
    """
    correctness_ops = [
        _op(
            [
                _case(1, success=True, accuracy_result=AccuracyResult(passed=True)),
                _case(2, success=True, accuracy_result=AccuracyResult(passed=True)),
            ]
        )
    ]
    performance_ops = [
        _op(
            [
                _case(
                    1,
                    success=True,
                    perf_result=PerfResult(elapsed_us=20.0),
                    accuracy_result=AccuracyResult(passed=True),
                ),
                # 同一 case 在性能阶段精度复检失败
                _case(
                    2,
                    success=False,
                    failure_type="precision_mismatch",
                    accuracy_result=AccuracyResult(passed=False, threshold=1e-3),
                    perf_result=None,
                ),
            ]
        )
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]
    by = {case.case_num: case for case in merged.results}

    # case 1: 两阶段都过 → 保留时延
    assert by[1].success is True
    assert by[1].perf_result is not None and by[1].perf_result.elapsed_us == 20.0
    assert by[1].perf_recheck is None

    # case 2: 精度翻车 → 判失败、扣精度分、无性能分、accuracy 反映失败、打标签
    assert by[2].success is False
    assert by[2].failure_type == "precision_mismatch"
    assert by[2].perf_result is None
    assert by[2].accuracy_result is not None and by[2].accuracy_result.passed is False
    assert by[2].perf_recheck is not None
    assert by[2].perf_recheck["status"] == "precision_unstable"
    assert by[2].perf_recheck["correctness_passed"] is True

    # 计数：1 通过 / 1 失败
    assert merged.passed_cases == 1
    assert merged.failed_cases == 1
    assert merged.pass_rate == 1 / 2

    # perf_recheck 必须能 to_dict → from_dict 往返（下游读 results.json 才看得到）
    roundtrip = EvalCaseResult.from_dict(by[2].to_dict())
    assert roundtrip.perf_recheck == by[2].perf_recheck
    assert roundtrip.perf_recheck["status"] == "precision_unstable"


def test_merge_excludes_reference_invalid_from_pass_rate():
    passed = _case(1, success=True)
    invalid = _case(2, success=False, failure_type="reference_invalid")
    performance = _case(1, success=True, perf_result=PerfResult(elapsed_us=20))
    merged = _merge_results([_op([passed, invalid])], [_op([performance])])[0]
    assert merged.total_cases == 2
    assert merged.passed_cases == merged.reference_invalid_cases == 1
    assert merged.failed_cases == 0
    assert merged.pass_rate == 1.0

    # Missing worker results remain in the denominator.
    incomplete = _op([passed, invalid])
    incomplete.total_cases = 3
    merged = _merge_results([incomplete], [_op([performance])])[0]
    assert merged.total_cases == 3
    assert merged.pass_rate == 0.5

    # The performance recheck may invalidate a previously passing reference.
    merged = _merge_results(
        [_op([_case(1, success=True), _case(2, success=True)])],
        [_op([performance, invalid])],
    )[0]
    assert merged.passed_cases == merged.reference_invalid_cases == 1
    assert merged.failed_cases == 0
    assert merged.pass_rate == 1.0

    all_invalid = _merge_results([_op([invalid])], [])[0]
    assert all_invalid.pass_rate == 0.0


def test_merge_tags_perf_unmeasured_without_failing_case():
    """性能阶段非精度失败（timeout/runtime）→ 沿用 correctness 通过判定，仅标注缺失。"""
    correctness_ops = [
        _op([_case(1, success=True, accuracy_result=AccuracyResult(passed=True))])
    ]
    performance_ops = [
        _op(
            [
                _case(
                    1,
                    success=False,
                    failure_type="subprocess_failure",
                    perf_result=None,
                    error_msg="重试后仍异常退出 rc=-11",
                )
            ]
        )
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]
    case = merged.results[0]

    assert case.success is True  # 非精度问题，不改判定
    assert case.perf_result is None  # 但没有有效性能数据
    assert case.perf_recheck is not None
    assert case.perf_recheck["status"] == "perf_unmeasured"
    assert case.perf_recheck["perf_failure_type"] == "subprocess_failure"
    assert case.perf_recheck["error_msg"] == "重试后仍异常退出 rc=-11"
    assert "rc=-11" in case.perf_recheck["note"]
    assert merged.passed_cases == 1
    assert merged.failed_cases == 0

    from src.kernel_eval.report.report_generator import EvalResult

    report_case = EvalResult.from_eval_case_result(case)
    assert report_case.performance_error_msg == "重试后仍异常退出 rc=-11"
    assert report_case.perf_recheck == case.perf_recheck


class TestCannHomeDirname:
    def test_plain_cann_dir(self):
        assert _cann_home_dirname("/usr/local/Ascend/cann-9.1.0") == "cann-9.1.0"

    def test_latest_symlink_resolved(self, tmp_path):
        real = tmp_path / "cann-9.1.0-beta.1"
        real.mkdir()
        latest = tmp_path / "latest"
        latest.symlink_to(real)
        assert _cann_home_dirname(str(latest)) == "cann-9.1.0-beta.1"

    def test_non_cann_path_returns_none(self):
        assert _cann_home_dirname("/usr/local/Ascend/ascend-toolkit") is None
        assert _cann_home_dirname("") is None


class TestDetectBuildCannDirname:
    def _write_compile_commands(self, source_dir: Path, commands):
        build = source_dir / "build"
        build.mkdir(parents=True, exist_ok=True)
        (build / "compile_commands.json").write_text(
            json.dumps([{"command": c} for c in commands])
        )

    def test_from_compile_commands_kernel_line(self, tmp_path):
        self._write_compile_commands(
            tmp_path,
            [
                "/usr/local/Ascend/cann-9.1.0-beta.1/x86_64-linux/ccec_compiler/bin/bisheng "
                "-I/usr/local/Ascend/cann-9.1.0-beta.1/include -c kernel.cpp",
                "gcc -I/usr/local/Ascend/cann-9.1.0/include -c host.cpp",
            ],
        )
        # kernel (bisheng) 行优先于普通 gcc 行
        assert _detect_build_cann_dirname(tmp_path) == "cann-9.1.0-beta.1"

    def test_from_compile_commands_host_only(self, tmp_path):
        self._write_compile_commands(
            tmp_path,
            [
                "gcc -I/usr/local/Ascend/cann-9.1.0/include -c host.cpp",
            ],
        )
        assert _detect_build_cann_dirname(tmp_path) == "cann-9.1.0"

    def test_from_compile_log_majority(self, tmp_path):
        (tmp_path / "_compile.log").write_text(
            "/usr/local/Ascend/cann-9.1.0-beta.1/include\n"
            "/usr/local/Ascend/cann-9.1.0-beta.1/include\n"
            "/usr/local/Ascend/cann-9.1.0/include\n"
        )
        assert _detect_build_cann_dirname(tmp_path) == "cann-9.1.0-beta.1"

    def test_no_evidence_returns_none(self, tmp_path):
        assert _detect_build_cann_dirname(tmp_path) is None


class TestCheckBuildCannConsistency:
    def _args(self, source_dir: Path) -> Namespace:
        return Namespace(source_dir=str(source_dir))

    def test_mismatch_detected(self, tmp_path):
        (tmp_path / "_compile.log").write_text(
            "/usr/local/Ascend/cann-9.1.0-beta.1/include\n"
        )
        with mock.patch.dict(
            os.environ, {"ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.1.0"}
        ):
            msg = _check_build_cann_consistency(self._args(tmp_path))
        assert msg is not None
        assert "cann-9.1.0-beta.1" in msg
        assert "cann-9.1.0" in msg

    def test_consistent_passes(self, tmp_path):
        (tmp_path / "_compile.log").write_text("/usr/local/Ascend/cann-9.1.0/include\n")
        with mock.patch.dict(
            os.environ, {"ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.1.0"}
        ):
            assert _check_build_cann_consistency(self._args(tmp_path)) is None

    def test_undetectable_passes(self, tmp_path):
        with mock.patch.dict(
            os.environ, {"ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.1.0"}
        ):
            assert _check_build_cann_consistency(self._args(tmp_path)) is None

    def test_missing_run_env_passes(self, tmp_path):
        (tmp_path / "_compile.log").write_text(
            "/usr/local/Ascend/cann-9.1.0-beta.1/include\n"
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _check_build_cann_consistency(self._args(tmp_path)) is None

    def test_no_source_dir_passes(self):
        with mock.patch.dict(
            os.environ, {"ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.1.0"}
        ):
            assert _check_build_cann_consistency(Namespace(source_dir=None)) is None
