# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""CPU accuracy remains usable while restored NPU profiling fails closed."""

from unittest.mock import Mock

import pytest
import torch

from kernel_eval.base.result import PerfResult
from kernel_eval.benches.cann_spec import CannCaseSpec
from kernel_eval.config import Config
from kernel_eval.eval.evaluator import Evaluator
from kernel_eval.eval.op_runner import OpRunner
from kernel_eval.eval.perf_eval import PerfEvaluator
from kernel_eval.utils.device_manager import DeviceConfig, DeviceManager


@pytest.mark.parametrize("enable_profiler", [True, False])
@pytest.mark.parametrize("correct", [True, False])
def test_cpu_evaluator_checks_accuracy_with_default_restoration(
    tmp_path, monkeypatch, enable_profiler, correct
):
    task = tmp_path / "tasks" / "level1" / "cpu_add_one"
    task.mkdir(parents=True)
    (task / "golden.py").write_text(
        "import torch\n"
        "def cpu_add_one(x: torch.Tensor) -> torch.Tensor:\n"
        "    return x + 1\n", encoding="utf-8"
    )
    (task / "proto.yaml").write_text(
        "operator:\n  name: CpuAddOne\n  difficulty: L1\n"
        "  inputs:\n  - name: x\n    dtype: [float32]\n"
        "  outputs:\n  - name: y\n    dtype: [float32]\n"
        "  schema: cpu_add_one(Tensor x) -> Tensor y\n", encoding="utf-8"
    )
    (task / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    case = CannCaseSpec(
        case_id="level1/cpu_add_one_1", case_num=1, operator="CpuAddOne",
        rel_path="level1/cpu_add_one", input_shapes=[[4]], dtypes=["float32"],
        value_ranges=[[1, 4]], attrs={},
    )
    config = Config(
        device_type="cpu", tasks_root=str(tmp_path / "tasks"),
        reports_dir=str(tmp_path / "reports"), torch_op_guard_mode="off",
        perf_freq_boost=False,
    )
    assert config.enable_profiler and config.perf_restore_inputs
    config.enable_profiler = enable_profiler
    evaluator = Evaluator(config)
    profile = Mock(side_effect=AssertionError("CPU accuracy must not profile"))
    monkeypatch.setattr(evaluator.perf_evaluator, "run_profiled", profile)
    calls = []

    def candidate(x):
        calls.append(x.device.type)
        return x + (1 if correct else 8)

    result = evaluator.evaluate_case(case, ai_op_func=candidate)

    assert calls == ["cpu"], result.error_msg
    assert result.ai_run_result.success, result.error_msg
    assert result.accuracy_result is not None, result.error_msg
    assert result.accuracy_result.passed is correct
    assert result.success is correct
    assert result.perf_result is None
    assert result.ai_run_result.perf_result is None
    assert result.ai_run_result.elapsed_us == 0
    profile.assert_not_called()


@pytest.mark.parametrize("partial_view", [True, False])
def test_cpu_runner_does_not_apply_npu_restored_layout_restrictions(
    tmp_path, partial_view
):
    config = Config(device_type="cpu", reports_dir=str(tmp_path))
    device = DeviceManager(DeviceConfig(type="cpu"))
    perf = PerfEvaluator(config, device_manager=device, freq_boost=False)
    runner = OpRunner(device, perf)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x = x[1:] if partial_view else x.t()

    result = runner.run(lambda x: x + 1, {"x": x}, "cpu_view_1", [x],
                        enable_profiler=True)

    assert result.success, result.error
    assert torch.equal(result.outputs, x + 1)
    assert result.perf_result is None
    assert result.elapsed_us == 0


@pytest.mark.parametrize("profiler_enabled", [False, True])
def test_npu_restored_protocol_failure_never_falls_back(
    tmp_path, monkeypatch, profiler_enabled
):
    config = Config(enable_profiler=profiler_enabled, reports_dir=str(tmp_path))
    device = DeviceManager(DeviceConfig(type="cpu"))
    # Exercise NPU dispatch without allocating device tensors.
    monkeypatch.setattr(device, "is_npu_mode", lambda: True)
    perf = PerfEvaluator(config, device_manager=device, freq_boost=False)
    failure = PerfResult(
        error_msg="restored capture failed",
        metadata={"perf_integrity_failed": True},
    )
    profile = Mock(return_value=(None, failure))
    monkeypatch.setattr(perf, "run_profiled", profile)
    runner = OpRunner(device, perf)
    x = torch.ones(2)
    candidate = Mock(return_value=x + 1)

    result = runner.run(candidate, {"x": x}, "npu_failure_1", [x],
                        enable_profiler=True)

    assert not result.success
    candidate.assert_not_called()
    if profiler_enabled:
        profile.assert_called_once()
        assert "restored capture failed" in result.error
    else:
        profile.assert_not_called()
        assert "requires NPU profiler" in result.error
