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

from types import SimpleNamespace

import pytest

import kernel_eval.cli as cli


@pytest.mark.parametrize(
    ("value", "bounds", "expected"),
    [
        (None, None, None),
        (0, None, {"case_id": 0}),
        (1, None, {"case_id": 1}),
        (None, "1-2", {"case_id_range": (1, 2)}),
    ],
)
def test_cpu_case_id_filter_preserves_zero(
    monkeypatch, tmp_path, value, bounds, expected
):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "VERSION").write_text("2.0.0\n")
    command = ["eval", "--device", "cpu"]
    if value is not None:
        command += ["--case-id", str(value)]
    if bounds is not None:
        command += ["--case-id-range", bounds]
    args = cli.create_parser().parse_args(command)
    captured = {}

    class DummyReport:
        operators = []
        failed_cases = 0

    class DummyReportGenerator:
        def __init__(self, **unused):
            pass

        def generate(self):
            return DummyReport()

        def save_all(self, report):
            pass

        def print_summary(self, report):
            pass

    monkeypatch.setattr(cli, "get_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(
        cli, "resolve_task_dir", lambda task_dir, project_root: (str(tmp_path), None)
    )
    monkeypatch.setattr(
        cli,
        "_create_config_from_args",
        lambda args, bench_root: SimpleNamespace(reports_dir="/tmp/reports"),
    )
    monkeypatch.setattr(cli, "ReportGenerator", DummyReportGenerator)

    def fake_simulate(
        config, bench_name, operator_filter, case_filter, report_generator
    ):
        captured["case_filter"] = case_filter

    monkeypatch.setattr("kernel_eval.simulation.simulate", fake_simulate)

    assert cli.cmd_eval(args) == 0
    assert captured["case_filter"] == expected


@pytest.mark.parametrize(
    ("value", "bounds", "expected_case_nums"),
    [(None, None, [1, 2]), (0, None, []), (1, None, [1]), (None, (2, 2), [2])],
)
def test_npu_case_id_filter_preserves_zero(
    monkeypatch, value, bounds, expected_case_nums
):
    cases = [
        SimpleNamespace(case_num=1, rel_path="level1/dummy", operator="Dummy"),
        SimpleNamespace(case_num=2, rel_path="level1/dummy", operator="Dummy"),
    ]
    captured = {}

    class DummyLoader:
        def scan_all(self):
            return list(cases)

    class DummyCoordinator:
        card_count = 1
        total_processes = 1

        def __init__(self, **unused):
            pass

        def evaluate_task_units(self, task_units):
            captured["case_nums"] = [
                case.case_num for unit in task_units for case in unit.cases
            ]
            return []

        def shutdown(self):
            pass

    args = SimpleNamespace(
        bench_name="cann",
        device_id=0,
        processes_per_card=1,
        max_cases_per_task_unit=64,
        timeout_per_operator=300,
        source_dir=None,
        skip_install=False,
        level=None,
        operator=None,
        case_id=value,
        case_id_range=bounds,
        no_perf=True,
        reports_dir=None,
        warmup=1,
        repeat=1,
    )

    def load_cases(bench_name, tasks_root, target_hardware):
        assert target_hardware == "950dt"
        return DummyLoader()

    monkeypatch.setattr(
        cli, "detect_execution_hardware", lambda device_type, device_id: "950dt"
    )
    monkeypatch.setattr("kernel_eval.registry.get_case_loader", load_cases)
    monkeypatch.setattr(
        "kernel_eval.eval.process_pool.ProcessPoolCoordinator", DummyCoordinator
    )
    monkeypatch.setattr(
        "kernel_eval.eval.subprocess_utils.detect_pypto_pro_submission", lambda: False
    )

    config = SimpleNamespace(device_type="npu")
    report_generator = SimpleNamespace(add_operator_result=lambda result: None)

    assert cli._cmd_eval_npu(args, "/tmp/tasks", None, config, report_generator) == 0
    assert captured.get("case_nums", []) == expected_case_nums


def test_case_id_zero_and_range_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        cli.create_parser().parse_args(
            ["eval", "--case-id", "0", "--case-id-range", "1-2"]
        )
    assert exc.value.code == 2
