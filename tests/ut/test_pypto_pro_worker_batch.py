# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for bounded Pro worker reuse and failure isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_eval.base.result import AccuracyResult, PerfResult
from kernel_eval.benches import CannCaseSpec
from kernel_eval.config import Config
from kernel_eval.eval.evaluator import Evaluator
from kernel_eval.eval.op_runner import OpRunResult
from kernel_eval.eval.process_pool import (
    ProcessConfig,
    ProcessPoolCoordinator,
    TaskUnit,
    build_task_units,
)
from kernel_eval.eval.results import EvalCaseResult


def case(number):
    return CannCaseSpec(
        case_id=f"level1/add_{number}",
        case_num=number,
        rel_path="level1/add",
        operator="Add",
        input_shapes=[[1]],
        dtypes=["float32"],
        attrs={},
        value_ranges=[{"min": -1, "max": 1}],
        metadata={},
    )


def verdict(number, failure=None):
    return EvalCaseResult(
        case_id=f"level1/add_{number}",
        case_num=number,
        rel_path="level1/add",
        operator="Add",
        success=failure is None,
        failure_type=failure,
        skipped_before_execution=failure == "cascade_device",
        accuracy_result=AccuracyResult(passed=True) if failure is None else None,
    )


@pytest.fixture
def coordinator(monkeypatch):
    monkeypatch.setattr(ProcessPoolCoordinator, "_detect_cards", lambda self: 2)
    return ProcessPoolCoordinator(
        Config(pypto_pro_outer_case_isolation=True),
        ProcessConfig(),
    )


@pytest.mark.parametrize("cards, cap", [(1, 64), (2, 64), (4, 64), (2, 8)])
def test_grouping_matches_common_policy(coordinator, cards, cap):
    coordinator.card_count = cards
    coordinator.base_config.max_cases_per_task_unit = cap
    cases = [case(i) for i in range(20)]
    tasks = build_task_units({"Add": cases}, cards, isolate_each_case=True)
    grouped = coordinator._group_pro_tasks(tasks)
    assert grouped == build_task_units(
        {"Add": cases}, cards, max_cases_per_task_unit=cap
    )
    coordinator.base_config.max_cases_per_task_unit = 1
    assert coordinator._group_pro_tasks(tasks) == tasks
    coordinator.base_config.pypto_pro_outer_case_isolation = False
    assert coordinator._group_pro_tasks(tasks) == tasks


def test_grouping_does_not_combine_different_operators_or_retry_tasks(coordinator):
    tasks = [
        TaskUnit("Add", "level1/add", [case(1)], 0),
        TaskUnit("Other", "level1/other", [case(2)], 0),
    ]
    assert len(coordinator._group_pro_tasks(tasks)) == 2
    tasks[0].retry_count = 1
    assert coordinator._group_pro_tasks(tasks) == tasks


def test_pro_child_keeps_per_case_profiler(coordinator):
    task = TaskUnit("Add", "level1/add", [case(1), case(2)], 0)
    command = coordinator._build_child_cmd(task, "/tmp/cases.json", "/tmp/out.json")
    assert "--no-perf-batch-cases" in command
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = coordinator.base_config
    assert not evaluator._should_batch_perf(task.cases)


def test_recorded_failure_is_not_retried(coordinator):
    task = TaskUnit("Add", "level1/add", [case(1), case(2)], 0)
    outcome = coordinator._pypto_batch_failure(
        task, "worker failed", True, [verdict(1, "precision_mismatch")]
    )
    assert outcome[2] is True
    assert [
        row.case_id for row in outcome[1] if row.failure_type == "precision_mismatch"
    ] == ["level1/add_1"]
    assert [item.case_num for item in outcome[4]] == [2]


def test_skip_marker_round_trip_cannot_hide_execution(coordinator):
    assert "skipped_before_execution" not in verdict(1).to_dict()
    skipped = verdict(2, "cascade_device")
    restored = EvalCaseResult.from_dict(skipped.to_dict())
    assert coordinator._is_unexecuted_pro_case(restored)
    skipped.ai_run_result = OpRunResult(success=False, error="execution failed")
    assert "skipped_before_execution" not in skipped.to_dict()
    assert coordinator._is_terminal_pro_verdict(
        EvalCaseResult.from_dict(skipped.to_dict())
    )


@pytest.mark.parametrize("performance_failure", [False, True])
def test_unhealthy_worker_skips_remaining_cases(performance_failure):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = Config(
        device_type="cpu", enable_profiler=False, pypto_pro_outer_case_isolation=True
    )
    evaluator._pypto_pro_jit_isolation = False
    evaluator.perf_evaluator = None
    evaluator.operator_matcher = SimpleNamespace(clear_cache=lambda: None)
    evaluator.incremental_output_path = None
    evaluator._cleanup_memory = lambda: None
    evaluator.device_manager = SimpleNamespace(
        health_check=lambda: False, is_npu_mode=lambda: False
    )
    called = []

    def evaluate(item):
        called.append(item.case_num)
        result = verdict(
            item.case_num, None if performance_failure else "compile_runtime_error"
        )
        if performance_failure:
            result.perf_result = PerfResult(error_msg="profiling failed")
        return result

    evaluator.evaluate_case = evaluate
    results = evaluator.run_cases([case(1), case(2)], "Add", "level1/add").results
    assert called == [1]
    assert results[1].failure_type == "cascade_device"
    assert results[1].skipped_before_execution


def test_bad_result_row_does_not_erase_a_recorded_failure(coordinator, tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "case_results": [
                    verdict(1, "precision_mismatch").to_dict(),
                    {"case_id": 12, "success": "yes"},
                ]
            }
        )
    )
    rows = coordinator._read_pro_results(str(path))
    assert len(rows) == 1 and rows[0].failure_type == "precision_mismatch"
    assert coordinator._read_pro_results(str(path), strict=True) == []


@pytest.mark.parametrize("pro", [False, True])
def test_retry_metadata_distinguishes_worker_and_profiler(
    coordinator, monkeypatch, pro
):
    coordinator.base_config.pypto_pro_outer_case_isolation = pro
    result = verdict(1)
    result.perf_result = PerfResult(elapsed_us=10)

    def popen(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"case_results": [result.to_dict()]})
        )
        return SimpleNamespace(pid=-1, returncode=0, communicate=lambda **kw: ("", ""))

    monkeypatch.setattr("kernel_eval.eval.process_pool.subprocess.Popen", popen)
    task = TaskUnit(
        "Add",
        "level1/add",
        [case(1)],
        0,
        parent_task_id="Add@Card0",
        perf_batch_cases=False,
    )
    rows = coordinator._process_retry_queue([task], {}, 1)
    assert len(rows) == 1 and rows[0].perf_result.elapsed_us == 10
    assert bool(rows[0].perf_result.metadata.get("perf_batch_attempted")) is not pro
