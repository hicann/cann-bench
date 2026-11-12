# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Strict inputs remain pristine through reference computation and migration."""

from types import SimpleNamespace

import pytest
import torch

from kernel_eval.base.models import InputSpec, OutputSpec
from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
from kernel_eval.eval.evaluator import Evaluator
from kernel_eval.eval.op_runner import OpRunner
from kernel_eval.eval.restored_inputs import (
    PerfIntegrityError,
    clone_reference_inputs,
    migrate_call_inputs,
)
from tests.ut.test_restored_inputs import mock_evaluator


class CopyingDevice:
    def __init__(self):
        self.moves = 0

    def to_device(self, tensor):
        self.moves += 1
        return tensor.clone()  # CPU control emulates a real cross-device copy.

    def to_device_batch(self, tensors):
        return [self.to_device(tensor) for tensor in tensors]

    def is_npu_mode(self):
        return True

    def get_device(self):
        return "cpu"

    def synchronize(self):
        pass


def evaluator_control(tmp_path, golden, params, *, strategy="fp64_cpu", get_input=None):
    perf, _ = mock_evaluator(tmp_path)
    config = perf.config
    config.warmup = perf.warmup = 0
    config.repeat = perf.repeat = 2
    config.torch_op_guard_mode = "off"
    perf._prepare_warmup_tensors = lambda: None
    perf._boost_freq_and_clear_cache = lambda: None
    device = CopyingDevice()
    runner = OpRunner(device, perf)
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = config
    evaluator.bench_name = "cann"
    evaluator.bench_config = SimpleNamespace(
        golden_precision=strategy, dtype_tolerance_map=None
    )
    evaluator._pypto_pro_jit_isolation = False
    evaluator._cleanup_memory = lambda: None
    evaluator.perf_evaluator = perf
    evaluator.op_runner = runner
    evaluator.golden_loader = SimpleNamespace(
        get_golden_function=lambda _: golden,
        get_oracle_function=lambda _: None,
        get_bench_function=lambda _: None,
        get_input_function=lambda _: get_input,
        get_output_function=lambda _: None,
    )
    values = list(params.values())
    dtype = str(values[0].dtype).removeprefix("torch.")
    evaluator.data_generator = SimpleNamespace(
        generate_input_tensors_from_case=lambda **kwargs: values
    )
    evaluator.param_builder = SimpleNamespace(
        build_call_params=lambda func, case, inputs: dict(zip(params, inputs))
    )
    op_info = SimpleNamespace(
        inputs=[InputSpec(name=name) for name in params],
        outputs=[OutputSpec(name="out", dtype=dtype)],
        precision_thresholds={},
    )
    evaluator.operator_loader = SimpleNamespace(get_operator=lambda _: op_info)
    evaluator.accuracy_evaluator = AccuracyEvaluator()
    case = SimpleNamespace(
        rel_path="level1/test",
        operator="Test",
        case_num=1,
        case_id=1,
        input_shapes=[list(v.shape) for v in values],
        dtypes=[dtype] * len(values),
        value_ranges=[[1, 1]] * len(values),
        attrs={},
        metadata={},
        baseline_perf_us=1,
        t_hw_us=1,
        get_case_id_str=lambda: "level1/test_1",
    )
    return evaluator, case, device


def test_pr1_only_mutating_int64_golden_does_not_contaminate_candidate(tmp_path):
    x = torch.ones(1, dtype=torch.int64)

    def golden(x):
        x.add_(1)
        return x

    evaluator, case, _ = evaluator_control(tmp_path, golden, {"x": x})
    seen = []

    def candidate(x):
        seen.append(x.clone())
        return x + 1

    result = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert result.success, result.error_msg
    assert len(seen) == 3
    assert all(torch.equal(value, torch.ones(1, dtype=torch.int64)) for value in seen)
    assert torch.equal(x, torch.ones(1, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cummin_structured_return_reaches_accuracy_after_restored_profiling(
    tmp_path, dtype
):
    x = torch.tensor([4.0, 1.0, 3.0, 2.0], dtype=dtype)

    def golden(x):
        return torch.cummin(x, dim=0)

    evaluator, case, _ = evaluator_control(tmp_path, golden, {"x": x})
    op_info = evaluator.operator_loader.get_operator(case.rel_path)
    op_info.outputs = [
        OutputSpec(name="values", dtype=str(dtype).removeprefix("torch.")),
        OutputSpec(name="indices", dtype="int64"),
    ]
    result = evaluator.evaluate_case(case, ai_op_func=golden)
    assert result.success, result.error_msg
    assert result.accuracy_result is not None and result.accuracy_result.passed
    assert result.ai_run_result.perf_result.elapsed_us > 0
    assert torch.equal(x, torch.tensor([4.0, 1.0, 3.0, 2.0], dtype=dtype))


@pytest.mark.parametrize(
    "strategy,dtype", [("native_cpu", torch.int64), ("fp64_cpu", torch.float32)]
)
def test_repeated_tensor_alias_preserved_in_references_and_h2d(
    tmp_path, strategy, dtype
):
    x = torch.ones(1, dtype=dtype)

    def inplace(x, y):
        assert x is y
        x.add_(1)
        return y

    evaluator, case, device = evaluator_control(
        tmp_path, inplace, {"x": x, "y": x}, strategy=strategy
    )
    result = evaluator.evaluate_case(case, ai_op_func=inplace)
    assert result.success, result.error_msg
    assert device.moves == 1
    assert torch.equal(x, torch.ones(1, dtype=dtype))


def test_get_input_reordering_and_shape_changes_remain_bound(tmp_path):
    x, y = torch.ones(2), torch.full((2,), 3.0)

    def get_input(x, y, **kwargs):
        return y.clone(), x.clone().reshape(1, 2)

    def golden(y: torch.Tensor, x: torch.Tensor):
        x.add_(2)
        return y + x.reshape(2)

    evaluator, case, _ = evaluator_control(
        tmp_path, golden, {"x": x, "y": y}, get_input=get_input
    )

    def candidate(y, x):
        assert x.shape == (1, 2)
        assert torch.equal(y, torch.full((2,), 3.0))
        return y + x.reshape(2) + 2

    result = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert result.success, result.error_msg


def test_nested_migration_preserves_identity_and_validates_before_any_copy():
    x = torch.ones(3)
    shared = {"x": [x]}
    moves = []

    def move(value):
        moves.append(value)
        return value.clone()

    args, kwargs, _ = migrate_call_inputs((shared,), {"nested": shared, "x": x}, move)
    assert len(moves) == 1
    assert args[0] is kwargs["nested"]
    assert args[0]["x"][0] is kwargs["x"]
    moves.clear()
    with pytest.raises(PerfIntegrityError):
        migrate_call_inputs((x, x.view_as(x)), {}, move)
    assert not moves


def test_reference_conversion_maps_duplicate_tensor_once():
    x = torch.ones(3)
    inputs, params = clone_reference_inputs(
        [x, x], {"x": x, "y": x}, lambda value: value.double()
    )
    assert inputs[0] is inputs[1] is params["x"] is params["y"]
    assert inputs[0].dtype == torch.float64
    inputs[0].add_(5)
    assert torch.equal(x, torch.ones(3))
