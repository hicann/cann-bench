# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MC2 partitioning must not recursively compare case payloads."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kernel_eval.config import Config
from kernel_eval.eval import process_pool
from kernel_eval.eval.process_pool import (
    ProcessConfig,
    ProcessPoolCoordinator,
    TaskUnit,
)


def test_mixed_tasks_partition_by_identity_without_equality(monkeypatch):
    def task(mc2):
        return TaskUnit(
            operator="Mixed",
            rel_path="mixed",
            device_id=0,
            cases=[SimpleNamespace(attrs={"mc2_distributed": mc2})],
        )

    regular, first, second = task(False), task(True), task(True)
    monkeypatch.setattr(
        TaskUnit,
        "__eq__",
        lambda *args: pytest.fail("task equality must not be invoked"),
    )
    coordinator = object.__new__(ProcessPoolCoordinator)
    coordinator.card_count = 2
    coordinator.evaluate_task_units = Mock(
        side_effect=lambda units: [id(unit) for unit in units]
    )
    results = ProcessPoolCoordinator.evaluate_task_units(
        coordinator, [first, regular, second]
    )
    assert results == [id(regular), id(first), id(second)]
    assert [
        [id(unit) for unit in call.args[0]]
        for call in coordinator.evaluate_task_units.call_args_list
    ] == [
        [id(regular)],
        [id(first)],
        [id(second)],
    ]


def test_outer_pro_groups_only_regular_tasks_before_exclusive_mc2(monkeypatch):
    def task(mc2):
        return TaskUnit(
            operator="Mixed",
            rel_path="level4/mixed",
            device_id=0,
            cases=[
                SimpleNamespace(
                    rel_path="level4/mixed",
                    attrs={"mc2_distributed": mc2, "world_size": 2},
                )
            ],
        )

    regular = [task(False) for _ in range(4)]
    first, second = task(True), task(True)
    monkeypatch.setattr(
        TaskUnit,
        "__eq__",
        lambda *args: pytest.fail("task equality must not be invoked"),
    )
    monkeypatch.setattr(ProcessPoolCoordinator, "_detect_cards", lambda self: 2)
    coordinator = ProcessPoolCoordinator(
        Config(pypto_pro_outer_case_isolation=True),
        ProcessConfig(enable_profiler=False, max_retries=0),
    )
    coordinator._build_env = Mock(return_value={})
    group_pro_tasks = coordinator._group_pro_tasks

    def group_regular(units):
        assert not any(
            case.attrs["mc2_distributed"] for unit in units for case in unit.cases
        )
        return group_pro_tasks(units)

    coordinator._group_pro_tasks = Mock(side_effect=group_regular)
    submitted = []
    worker_limits = []

    class CompletedExecutor:
        def __init__(self, max_workers):
            worker_limits.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, payload):
            _, unit = payload
            submitted.append(unit)
            results = [
                SimpleNamespace(success=True, task_identity=id(unit))
                for _ in unit.cases
            ]
            future = Future()
            future.set_result((unit, results, False, None, []))
            return future

    monkeypatch.setattr(process_pool, "ThreadPoolExecutor", CompletedExecutor)

    results = coordinator.evaluate_task_units([first, regular[0], second, *regular[1:]])

    coordinator._group_pro_tasks.assert_called_once()
    grouped_input = coordinator._group_pro_tasks.call_args.args[0]
    assert [id(unit) for unit in grouped_input] == [id(unit) for unit in regular]
    assert [len(unit.cases) for unit in submitted[:2]] == [2, 2]
    assert [id(unit) for unit in submitted[2:]] == [id(first), id(second)]
    assert [result.task_identity for result in results[-2:]] == [id(first), id(second)]
    assert worker_limits == [4, 1, 1]
