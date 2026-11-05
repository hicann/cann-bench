# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MC2 parent-side aggregation regressions; no NPU or subprocesses required."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kernel_eval.base.result import (
    FAILURE_TYPE_CASCADE_DEVICE,
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    FAILURE_TYPE_PRECISION_MISMATCH,
    FAILURE_TYPE_REFERENCE_INVALID,
    FAILURE_TYPE_SKIPPED,
)
from kernel_eval.checkers.mc2_three_way_checker import compare_mc2_outputs
from kernel_eval.config import Config
from kernel_eval.eval import mc2_distributed_runner as runner
from kernel_eval.eval.evaluator import Evaluator
from kernel_eval.eval.perf_eval import PerfResult
from kernel_eval.eval.results import EvalCaseResult, summarize_case_results


@pytest.fixture
def evaluate_rank_results(monkeypatch, tmp_path):
    """Deliver serialized rank results through the real parent evaluator."""

    class FinishedProcess:
        exitcode = 0

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout):
            pass

    queue = Mock()
    context = SimpleNamespace(
        Queue=lambda: queue, Process=lambda **kwargs: FinishedProcess()
    )
    monkeypatch.setattr(runner.torch.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(runner, "get_golden_loader", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(runner, "_find_free_port", lambda: 29500)
    config = SimpleNamespace(
        tasks_root=tmp_path,
        reports_dir=tmp_path,
        warmup=1,
        repeat=1,
        enable_profiler=True,
        timeout_per_operator=90,
    )

    def evaluate(payloads, mode, baseline_perf_us=0, baseline_source=None):
        queue.get.side_effect = payloads
        attrs = {"world_size": 2}
        if mode is not None:
            attrs["mc2_accuracy_mode"] = mode
        case = SimpleNamespace(
            attrs=attrs,
            rel_path="level4/moe_distribute_dispatch_v2",
            operator="MoeDistributeDispatchV2",
            case_num=1,
            input_shapes=[],
            dtypes=["float32"],
            value_ranges=[],
            baseline_perf_us=baseline_perf_us,
            t_hw_us=10,
            metadata={"baseline_source": baseline_source or {}},
            get_case_id_str=lambda: "level4/moe_distribute_dispatch_v2_1",
        )
        return runner.MC2DistributedEvaluator(config).evaluate_case(case)

    return evaluate


@pytest.mark.parametrize("failed_rank", [0, 1])
@pytest.mark.parametrize("structural", [False, True])
def test_three_way_moe_failure_survives_parent_aggregation(
    evaluate_rank_results,
    failed_rank,
    structural,
):
    payloads = []
    for rank in range(2):
        reference = torch.ones(4)
        candidate = reference
        if rank == failed_rank:
            candidate = torch.ones(3) if structural else reference + 0.25
        outputs = compare_mc2_outputs(candidate, reference, reference, rank)
        payloads.append(
            {
                "rank": rank,
                "passed": all(output.passed for output in outputs),
                "dtype": "float32",
                "threshold": 0.0,
                "compare_error": "; ".join(
                    output.error_msg for output in outputs if output.error_msg
                ),
                "output_results": [output.to_dict() for output in outputs],
                "elapsed_us": 100,
                "golden_elapsed_us": 200,
            }
        )

    result = evaluate_rank_results(payloads, "three_way")

    assert not result.success
    assert not result.accuracy_result.passed
    assert result.perf_result is None
    assert result.accuracy_result.threshold is None
    assert result.accuracy_result.metadata["checker_name"] == "mc2_three_way"
    assert "mismatch_count" not in result.accuracy_result.metadata
    assert result.failure_type == (
        FAILURE_TYPE_COMPILE_RUNTIME_ERROR
        if structural
        else FAILURE_TYPE_PRECISION_MISMATCH
    )
    assert [output.passed for output in result.accuracy_result.output_results] == [
        rank != failed_rank for rank in range(2)
    ]
    assert f"rank {failed_rank}:" in result.error_msg


@pytest.mark.parametrize("mode", [None, "legacy"])
@pytest.mark.parametrize("max_diff", [0.0, 0.25])
def test_legacy_moe_zero_error_compatibility_is_preserved(
    evaluate_rank_results, mode, max_diff
):
    payloads = [
        {
            "rank": rank,
            "passed": False,
            "dtype": "float32",
            "threshold": 0.001,
            "mismatch_count": 0,
            "mere": 0.0,
            "mare": 0.0,
            "max_diff": max_diff,
            "mean_diff": 0.0,
            "elapsed_us": 100,
            "golden_elapsed_us": 200,
        }
        for rank in range(2)
    ]

    result = evaluate_rank_results(payloads, mode)

    assert result.success == (max_diff == 0.0)
    assert result.accuracy_result.passed == result.success
    assert result.accuracy_result.metadata["checker_name"] == "relative_error"
    assert result.accuracy_result.threshold == 0.001
    assert result.failure_type == (
        None if result.success else FAILURE_TYPE_PRECISION_MISMATCH
    )


@pytest.mark.parametrize("failed_rank", [0, 1])
def test_reference_failure_survives_parent_aggregation(
    evaluate_rank_results, failed_rank
):
    payloads = []
    for rank in range(2):
        cpu = torch.ones(4)
        native = cpu + 0.25 if rank == failed_rank else cpu
        outputs = compare_mc2_outputs(native, native, cpu, rank)
        payloads.append(
            {
                "rank": rank,
                "passed": all(output.passed for output in outputs),
                "dtype": "float32",
                "threshold": 0.0,
                "compare_error": "; ".join(output.error_msg for output in outputs),
                "output_results": [output.to_dict() for output in outputs],
                "elapsed_us": 100,
                "golden_elapsed_us": 200,
            }
        )
    result = evaluate_rank_results(payloads, "three_way")
    assert not result.success
    assert result.failure_type == FAILURE_TYPE_REFERENCE_INVALID
    assert result.perf_result is None
    restored = EvalCaseResult.from_dict(result.to_dict())
    assert restored.failure_type == FAILURE_TYPE_REFERENCE_INVALID
    assert summarize_case_results([restored]).failed == 0


def test_new_run_serializes_measurement_provenance(evaluate_rank_results):
    payloads = []
    for rank in range(2):
        cpu = torch.ones(4)
        outputs = compare_mc2_outputs(cpu, cpu, cpu, rank)
        payloads.append(
            {
                "rank": rank,
                "passed": True,
                "dtype": "float32",
                "threshold": 0.0,
                "compare_error": "",
                "output_results": [output.to_dict() for output in outputs],
                "elapsed_us": 100 + rank,
                "golden_elapsed_us": 200 + rank,
            }
        )
    result = evaluate_rank_results(payloads, "three_way")
    assert result.success
    data = result.to_dict()
    measurement = data["accuracy"]["metadata"]["measurement"]
    assert measurement["warmup"] == 1
    assert measurement["repeat"] == 1
    assert (
        measurement["timing_statistic"] == "maximum_across_ranks_of_mean_wall_time_us"
    )
    assert measurement["cold_start_excluded"]
    assert measurement["includes_final_barrier"]
    assert measurement["measured"]
    assert all(
        data["perf"]["metadata"][key] == value for key, value in measurement.items()
    )
    assert EvalCaseResult.from_dict(data).perf_result.metadata["repeat"] == 1
    assert data["baseline_source"]["baseline_perf_us"]["source"] == "runtime_reference"
    assert result.baseline_perf_us == 201

    fixed_source = {
        "baseline_perf_us": {
            "requested_hardware": "950pr",
            "source_hardware": "950pr",
            "fallback_used": False,
            "source": "metadata/950pr.json",
        }
    }
    fixed = evaluate_rank_results(
        payloads, "three_way", baseline_perf_us=150, baseline_source=fixed_source
    )
    assert fixed.baseline_perf_us == 150
    assert fixed.baseline_source == fixed_source


def test_summary_counts_only_explicit_skips_and_keeps_runtime_failures():
    failure_types = [
        FAILURE_TYPE_SKIPPED,
        FAILURE_TYPE_CASCADE_DEVICE,
        FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
        FAILURE_TYPE_PRECISION_MISMATCH,
        "timeout",
        "oom_killed",
        None,
        "unknown",
    ]
    cases = [
        EvalCaseResult(
            case_id=f"case_{index}",
            rel_path="level4/moe_distribute_dispatch_v2",
            operator="MoeDistributeDispatchV2",
            case_num=index,
            success=False,
            failure_type=failure_type,
            perf_result=PerfResult(elapsed_us=1),
            baseline_perf_us=100,
        )
        for index, failure_type in enumerate(failure_types)
    ]
    cases.append(
        EvalCaseResult(
            case_id="passed",
            rel_path="level4/moe_distribute_dispatch_v2",
            operator="MoeDistributeDispatchV2",
            case_num=10,
            success=True,
            perf_result=PerfResult(elapsed_us=100),
            baseline_perf_us=200,
        )
    )

    summary = summarize_case_results(cases)

    assert (summary.passed, summary.failed, summary.skipped) == (1, 6, 2)
    assert summary.pass_rate == pytest.approx(1 / 9)
    assert summary.avg_speedup == 2


@pytest.mark.parametrize("reference_invalid", [False, True])
def test_reused_pro_worker_preserves_reference_failure_boundary(reference_invalid):
    evaluator = object.__new__(Evaluator)
    evaluator.config = Config(
        device_type="cpu",
        enable_profiler=False,
        perf_batch_cases=False,
        pypto_pro_outer_case_isolation=True,
    )
    evaluator._pypto_pro_jit_isolation = False
    evaluator.perf_evaluator = None
    evaluator.operator_matcher = Mock()
    evaluator.incremental_output_path = None
    evaluator._cleanup_memory = Mock()
    evaluator.device_manager = Mock()
    evaluator.device_manager.health_check.return_value = False
    evaluator.device_manager.is_npu_mode.return_value = False
    cases = [
        SimpleNamespace(
            attrs={},
            rel_path="level4/test",
            operator="Test",
            case_id=number,
            get_case_id_str=Mock(return_value=f"test_{number}"),
        )
        for number in (1, 2)
    ]
    first = EvalCaseResult(
        case_id="test_1",
        rel_path="level4/test",
        operator="Test",
        case_num=1,
        success=not reference_invalid,
        failure_type=FAILURE_TYPE_REFERENCE_INVALID if reference_invalid else None,
        perf_result=None
        if reference_invalid
        else PerfResult(error_msg="profiling failed"),
    )
    second = EvalCaseResult(
        case_id="test_2",
        rel_path="level4/test",
        operator="Test",
        case_num=2,
        success=True,
    )
    evaluator.evaluate_case = Mock(side_effect=[first, second])

    result = evaluator.run_cases(cases, "Test", "level4/test")

    evaluator.device_manager.recover_light.assert_not_called()
    evaluator.device_manager.recover_full.assert_not_called()
    assert result.results[0] is first
    if reference_invalid:
        assert evaluator.evaluate_case.call_count == 2
        assert result.results[1] is second
        evaluator._cleanup_memory.assert_not_called()
        evaluator.device_manager.health_check.assert_not_called()
        assert result.reference_invalid_cases == 1
        assert result.failed_cases == 0
        assert result.pass_rate == 1.0
    else:
        evaluator.evaluate_case.assert_called_once_with(cases[0])
        evaluator._cleanup_memory.assert_called_once()
        evaluator.device_manager.health_check.assert_called_once()
        assert result.results[1].failure_type == FAILURE_TYPE_CASCADE_DEVICE
        assert result.results[1].skipped_before_execution
