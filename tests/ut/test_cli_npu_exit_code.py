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


class _Report:
    def __init__(self, operators):
        self.operators = operators
        self.failed_cases = sum(op.failed_cases for op in operators)


class _ReportGenerator:
    def __init__(self, **unused):
        self.operators = []

    def add_operator_result(self, result):
        self.operators.append(result)

    def generate(self):
        return _Report(self.operators)

    def save_all(self, report):
        pass

    def print_summary(self, report):
        pass


@pytest.mark.parametrize(("card_count", "expected_exit_code"), [(0, 1), (1, 0)])
def test_eval_propagates_npu_setup_failure(
    monkeypatch, tmp_path, card_count, expected_exit_code
):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "VERSION").write_text("2.0.0\n")
    args = cli.create_parser().parse_args(["eval", "--device", "npu", "--no-perf"])
    case = SimpleNamespace(
        case_id="level1/dummy_1",
        case_num=1,
        operator="Dummy",
        rel_path="level1/dummy",
    )
    monkeypatch.setattr(cli, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        cli, "resolve_task_dir", lambda task_dir, project_root: (str(tmp_path), None)
    )
    monkeypatch.setattr(
        cli,
        "_create_config_from_args",
        lambda unused_args, unused_root: SimpleNamespace(
            reports_dir=str(tmp_path), device_type="npu"
        ),
    )
    monkeypatch.setattr(cli, "ReportGenerator", _ReportGenerator)

    class _Loader:
        def scan_all(self):
            return [case]

    monkeypatch.setattr(
        "kernel_eval.registry.get_case_loader",
        lambda bench_name, tasks_root, target_hardware: _Loader(),
    )
    monkeypatch.setattr(
        cli, "detect_execution_hardware", lambda device_type, device_id: "950dt"
    )
    monkeypatch.setattr(
        "kernel_eval.eval.subprocess_utils.detect_pypto_pro_submission",
        lambda: False,
    )

    class _Coordinator:
        def __init__(self, **unused):
            self.card_count = card_count
            self.total_processes = 1

        def evaluate_task_units(self, task_units):
            assert len(task_units) == 1
            passed_case = SimpleNamespace(success=True)
            return [passed_case]

        def shutdown(self):
            pass

    def aggregate(results):
        assert len(results) == 1
        return [
            SimpleNamespace(
                operator="Dummy",
                rel_path="level1/dummy",
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
                compilation_error=None,
            )
        ]

    monkeypatch.setattr(
        "kernel_eval.eval.process_pool.ProcessPoolCoordinator", _Coordinator
    )
    monkeypatch.setattr(
        "kernel_eval.eval.process_pool.aggregate_by_operator", aggregate
    )

    assert cli.cmd_eval(args) == expected_exit_code


def test_eval_propagates_cpu_simulation_failure(monkeypatch, tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "VERSION").write_text("2.0.0\n")
    args = cli.create_parser().parse_args(["eval", "--device", "cpu", "--no-perf"])
    monkeypatch.setattr(cli, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        cli, "resolve_task_dir", lambda task_dir, project_root: (str(tmp_path), None)
    )
    monkeypatch.setattr(
        cli,
        "_create_config_from_args",
        lambda unused_args, unused_root: SimpleNamespace(reports_dir=str(tmp_path)),
    )
    monkeypatch.setattr(cli, "ReportGenerator", _ReportGenerator)
    monkeypatch.setattr(
        "kernel_eval.simulation.simulate", lambda *unused, **unused_kwargs: 1
    )

    assert cli.cmd_eval(args) == 1
