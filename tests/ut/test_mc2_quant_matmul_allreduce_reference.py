# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Independent CPU checks for the two-rank INT8 MatmulAllReduce cascade."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


TASK = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/kernel_bench/level4/matmul_all_reduce/golden.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "quant_matmul_allreduce_reference_test", TASK
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _inputs(dtype, ndim, transpose, bias, residual):
    generator = torch.Generator().manual_seed(3491)
    peers = []
    scale_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    for rank in range(2):
        x1 = torch.randint(-3, 4, (4, 3), generator=generator, dtype=torch.int8)
        weight = torch.randint(-3, 4, (3, 5), generator=generator, dtype=torch.int8)
        x3 = (
            (torch.randn((4, 5), generator=generator) * 0.1).to(dtype)
            if residual
            else None
        )
        if ndim == 3:
            x1 = x1.reshape(2, 2, 3)
            x3 = x3.reshape(2, 2, 5) if x3 is not None else None
        peers.append(
            {
                "x1": x1,
                "x2": weight.t().contiguous() if transpose else weight,
                "bias": torch.tensor([3, -2, 5, 1, -4], dtype=torch.int32) + rank
                if bias
                else None,
                "dequant_scale": torch.tensor(
                    [0.0317, 0.0673, 0.0259, 0.0921, 0.0153], dtype=scale_dtype
                ),
                "pertoken_scale": torch.tensor([0.217, 0.431, 0.683, 0.917])
                + rank * 0.013,
                "x3": x3,
            }
        )
    assert not torch.equal(peers[0]["x1"], peers[1]["x1"])
    return peers


def _local_oracles(inputs, attrs):
    """Use INT64 accumulation, independent of the task's INT32 implementation."""
    weight = inputs["x2"].t() if attrs["is_trans_b"] else inputs["x2"]
    acc = inputs["x1"].long() @ weight.long()
    if inputs["bias"] is not None:
        acc = acc + inputs["bias"].long()
    shape = acc.shape
    token = inputs["pertoken_scale"].float().reshape(-1, 1)
    channel = inputs["dequant_scale"].float().reshape(1, -1)
    mathematical = ((acc.float().reshape(-1, shape[-1]) * token) * channel).reshape(
        shape
    )
    output_dtype = getattr(torch, attrs["out_dtype"])
    staged = (
        ((acc.float().reshape(-1, shape[-1]) * channel) * token)
        .reshape(shape)
        .to(output_dtype)
    )
    if inputs["x3"] is not None:
        mathematical = mathematical + inputs["x3"].float()
        staged = (staged.float() + inputs["x3"].float()).to(output_dtype)
    return mathematical, staged.float()


def _cpu_context(peers, attrs, rank):
    local = [_local_oracles(item, attrs) for item in peers]
    calls = []

    def all_reduce(tensor, op):
        index = len(calls)
        assert index < 2, "math and stage must each communicate exactly once"
        assert op == "sum"
        assert tensor.dtype == torch.float32
        _exact(tensor, local[rank][index])
        tensor.copy_(local[0][index] + local[1][index])
        calls.append(index)

    ctx = {
        "rank": rank,
        "world_size": 2,
        "device": torch.device("cpu"),
        "dist": SimpleNamespace(
            all_reduce=all_reduce, ReduceOp=SimpleNamespace(SUM="sum")
        ),
    }
    expected_math = local[0][0] + local[1][0]
    expected_stage = (
        (local[0][1] + local[1][1]).to(getattr(torch, attrs["out_dtype"])).float()
    )
    return ctx, calls, expected_math, expected_stage


def _native_result(module, peers, attrs, rank):
    """Run the unchanged baseline with a separately implemented CPU quant primitive."""
    inputs = peers[rank]
    dtype = getattr(torch, attrs["out_dtype"])
    weight = inputs["x2"].t() if attrs["is_trans_b"] else inputs["x2"]
    stages = [_local_oracles(item, attrs)[1] for item in peers]
    quant_calls, comm_calls = [], []

    def quant_matmul(x1, x2, scale, *, pertoken_scale, bias, output_dtype):
        assert not quant_calls
        _exact(x1, inputs["x1"].reshape(-1, inputs["x1"].shape[-1]))
        _exact(x2, weight)
        _exact(scale, inputs["dequant_scale"].reshape(-1))
        _exact(pertoken_scale, inputs["pertoken_scale"].reshape(-1))
        assert bias is inputs["bias"]
        assert output_dtype == dtype
        acc = x1.long() @ x2.long()
        if bias is not None:
            acc += bias.long()
        result = acc.float() * scale.float().reshape(1, -1)
        result = result * pertoken_scale.float().reshape(-1, 1)
        quant_calls.append(True)
        return result.to(output_dtype)

    def all_reduce(tensor, op):
        assert not comm_calls
        assert quant_calls == [True] and op == "sum"
        assert tensor.dtype == dtype
        _exact(tensor, stages[rank].to(dtype))
        tensor.copy_((stages[0] + stages[1]).to(dtype))
        comm_calls.append(True)

    ctx = {
        "rank": rank,
        "world_size": 2,
        "torch_npu": SimpleNamespace(npu_quant_matmul=quant_matmul),
        "dist": SimpleNamespace(
            all_reduce=all_reduce, ReduceOp=SimpleNamespace(SUM="sum")
        ),
    }
    output = module.mc2_distributed_golden(ctx, inputs, attrs)
    assert quant_calls == comm_calls == [True]
    return output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_quantized_two_rank_math_and_staged_contract(
    dtype, ndim, transpose, bias, residual, rank
):
    module = _load()
    peers = _inputs(dtype, ndim, transpose, bias, residual)
    attrs = {
        "quantized": True,
        "is_trans_b": transpose,
        "out_dtype": str(dtype).removeprefix("torch."),
        "reduce_op": "sum",
        "quant_type": "per_token_channel",
        "comm_quant_mode": 0,
    }
    ctx, calls, expected_math, expected_stage = _cpu_context(peers, attrs, rank)

    mathematical, staged = module.mc2_cpu_references(ctx, peers[rank], attrs)

    assert staged is not None, (
        "two-rank INT8 outputs need an independent stage-rounded reference"
    )
    assert calls == [0, 1]
    assert mathematical.dtype == staged.dtype == torch.float32
    _exact(mathematical, expected_math)
    _exact(staged, expected_stage)
    native = _native_result(module, peers, attrs, rank)
    assert native.dtype == dtype
    _exact(native.float(), expected_stage)


@pytest.mark.parametrize(
    "dtype,token0,token1",
    [
        (torch.float16, 1.0004, 1.0003),
        (torch.bfloat16, 1.003, 1.002),
    ],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_quantized_cancellation_needs_stage_but_does_not_hide_wrong_results(
    dtype, token0, token1, rank
):
    from kernel_eval.base.result import FAILURE_TYPE_REFERENCE_INVALID
    from kernel_eval.checkers.mc2_three_way_checker import (
        _reference_health,
        check_mc2_references,
        compare_mc2_outputs,
    )

    # Eight entries cancel to a nonzero FP32 value, but independent local
    # quantization rounds both rank contributions to +/-1 before the sum.
    # The remaining entries keep the global rounding error within fixed bounds.
    scale_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    remote_scale = torch.full((129,), 0.5, dtype=scale_dtype)
    remote_scale[:8] = 1
    peers = [
        {
            "x1": torch.tensor([[sign]], dtype=torch.int8),
            "x2": torch.ones((1, 129), dtype=torch.int8),
            "bias": None,
            "x3": None,
            "dequant_scale": scale,
            "pertoken_scale": torch.tensor([token]),
        }
        for sign, token, scale in (
            (1, token0, torch.ones(129, dtype=scale_dtype)),
            (-1, token1, remote_scale),
        )
    ]
    attrs = {
        "quantized": True,
        "is_trans_b": False,
        "out_dtype": str(dtype).removeprefix("torch."),
        "reduce_op": "sum",
        "quant_type": "per_token_channel",
        "comm_quant_mode": 0,
    }
    module = _load()
    ctx, calls, expected_math, expected_stage = _cpu_context(peers, attrs, rank)
    mathematical, staged = module.mc2_cpu_references(ctx, peers[rank], attrs)
    assert staged is not None
    assert calls == [0, 1]
    _exact(mathematical, expected_math)
    _exact(staged, expected_stage)
    native = _native_result(module, peers, attrs, rank)
    _exact(native.float(), staged)
    assert torch.count_nonzero(mathematical[0, :8]) == 8
    assert torch.count_nonzero(native[0, :8]) == 0

    old_health = _reference_health(native, mathematical)
    assert old_health["passed"] is False
    assert old_health["small_value_error_count"] == 8
    assert _reference_health(native, mathematical, staged)["passed"] is True
    old_result = check_mc2_references(native, mathematical, rank)[0]
    assert old_result.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    good = compare_mc2_outputs(
        native, native, mathematical, rank, cascade_golden=staged
    )[0]
    assert good.passed
    assert good.metadata["reference_valid"] is True
    # Candidate errors are still measured against FP32 G, not against H.
    assert good.metadata["cascade_metrics"]["diff_big_max"] > 0

    wrong = (native.float() + 1).to(dtype)
    bad_reference = check_mc2_references(
        wrong, mathematical, rank, cascade_golden=staged
    )[0]
    assert not bad_reference.passed
    assert bad_reference.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    bad_candidate = compare_mc2_outputs(
        wrong, native, mathematical, rank, cascade_golden=staged
    )[0]
    assert not bad_candidate.passed
    assert bad_candidate.metadata["reference_valid"] is True
    # Nor can a corrupted independent stage bless an equally corrupted baseline.
    bad_stage = check_mc2_references(
        wrong, mathematical, rank, cascade_golden=wrong.float()
    )[0]
    assert not bad_stage.passed


def test_stage_scaling_order_at_fp16_rounding_boundary():
    # These FP32 scale values cross an FP16 midpoint when multiplication order
    # changes. This locks the CPU stage model; it is not an NPU execution test.
    peers = [
        {
            "x1": torch.tensor([[1 if rank == 0 else 0]], dtype=torch.int8),
            "x2": torch.ones((1, 1), dtype=torch.int8),
            "bias": torch.tensor([12660 if rank == 0 else 0], dtype=torch.int32),
            "x3": None,
            "dequant_scale": torch.tensor([0.00909915380179882]),
            "pertoken_scale": torch.tensor([0.863953709602356]),
        }
        for rank in range(2)
    ]
    attrs = {
        "quantized": True,
        "is_trans_b": False,
        "out_dtype": "float16",
        "reduce_op": "sum",
    }
    ctx, calls, expected_math, expected_stage = _cpu_context(peers, attrs, rank=0)
    mathematical, staged = _load().mc2_cpu_references(ctx, peers[0], attrs)
    assert calls == [0, 1]
    _exact(mathematical, expected_math)
    _exact(staged, expected_stage)
    assert mathematical.half().item() == 99.5
    assert staged.item() == 99.5625


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "fault",
    [
        "missing_all_reduce",
        "missing_bias",
        "missing_x3",
        "channel_scale",
        "token_scale",
    ],
)
def test_staged_reference_rejects_structural_quantized_errors(dtype, fault):
    from kernel_eval.base.result import FAILURE_TYPE_REFERENCE_INVALID
    from kernel_eval.checkers.mc2_three_way_checker import (
        check_mc2_references,
        compare_mc2_outputs,
    )

    # Exact, positive values isolate semantic errors from rounding/cancellation.
    # Different rank contributions make a missing collective observable.
    peers = [
        {
            "x1": torch.full((4, 3), 4 * (rank + 1), dtype=torch.int8),
            "x2": torch.full((3, 129), 4, dtype=torch.int8),
            "bias": torch.full((129,), 1024, dtype=torch.int32),
            "x3": torch.full((4, 129), 64, dtype=dtype),
            "dequant_scale": torch.full((129,), 0.25, dtype=torch.float32),
            "pertoken_scale": torch.ones(4),
        }
        for rank in range(2)
    ]
    attrs = {
        "quantized": True,
        "is_trans_b": False,
        "out_dtype": str(dtype).removeprefix("torch."),
        "reduce_op": "sum",
        "quant_type": "per_token_channel",
        "comm_quant_mode": 0,
    }
    ctx, calls, expected_math, expected_stage = _cpu_context(peers, attrs, rank=0)
    mathematical, staged = _load().mc2_cpu_references(ctx, peers[0], attrs)
    assert calls == [0, 1]
    _exact(mathematical, expected_math)
    _exact(staged, expected_stage)
    _exact(staged, mathematical)
    native = staged.to(dtype)
    assert check_mc2_references(native, mathematical, 0, cascade_golden=staged)[
        0
    ].passed

    damaged = [{key: value.clone() for key, value in item.items()} for item in peers]
    if fault in ("missing_bias", "missing_x3"):
        key = "bias" if fault == "missing_bias" else "x3"
        for item in damaged:
            item[key] = None
    elif fault in ("channel_scale", "token_scale"):
        key = "dequant_scale" if fault == "channel_scale" else "pertoken_scale"
        for item in damaged:
            item[key] *= 0.5
    local = [_local_oracles(item, attrs)[1] for item in damaged]
    wrong = (local[0] if fault == "missing_all_reduce" else local[0] + local[1]).to(
        dtype
    )
    assert not torch.equal(wrong.float(), staged)

    bad_reference = check_mc2_references(wrong, mathematical, 0, cascade_golden=staged)[
        0
    ]
    assert not bad_reference.passed
    assert bad_reference.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    bad_candidate = compare_mc2_outputs(
        wrong, native, mathematical, 0, cascade_golden=staged
    )[0]
    assert not bad_candidate.passed
    assert bad_candidate.metadata["reference_valid"] is True


@pytest.mark.parametrize("case_id", [21, 22])
def test_published_case_subset_reproduces_reference_failure_on_cpu(case_id):
    """Original seeded inputs, full K; this is a CPU subset test, not an NPU pass."""
    from kernel_eval.checkers.mc2_three_way_checker import (
        _reference_health,
        compare_mc2_outputs,
    )

    cases = yaml.safe_load(TASK.with_name("cases.yaml").read_text(encoding="utf-8"))[
        "cases"
    ]
    case = next(item for item in cases if item["case_id"] == case_id)
    attrs = case["attrs"]
    assert attrs["world_size"] == 2 and attrs["quantized"] and attrs["is_trans_b"]
    assert attrs["out_dtype"] == ("bfloat16" if case_id == 21 else "float16")
    assert attrs["is_bias"] == attrs["is_x3"] == (case_id == 22)
    payload = {
        "input_shapes": case["input_shape"],
        "dtypes": case["dtype"],
        "value_ranges": case["value_range"],
        "attrs": attrs,
    }
    module = _load()
    peers = []
    for rank in range(2):
        # Generate the full tensors before selecting rows/columns: changing the
        # requested RNG tensor shapes would not preserve the published x3 data.
        with torch.random.fork_rng(devices=[]):
            full = module.mc2_make_rank_inputs({"rank": rank, "device": "cpu"}, payload)
        peers.append(
            {
                "x1": full["x1"][:64, :].clone(),
                "x2": full["x2"][:256, :].clone(),
                "bias": full["bias"][:256].clone()
                if full["bias"] is not None
                else None,
                "dequant_scale": full["dequant_scale"][:256].clone(),
                "pertoken_scale": full["pertoken_scale"][:64].clone(),
                "x3": full["x3"][:64, :256].clone() if full["x3"] is not None else None,
            }
        )
        assert peers[-1]["x1"].shape == (64, case["input_shape"][0][-1])
        assert peers[-1]["x2"].shape == (256, case["input_shape"][1][-1])
    assert not torch.equal(peers[0]["x1"], peers[1]["x1"])

    for rank in range(2):
        ctx, calls, expected_math, expected_stage = _cpu_context(peers, attrs, rank)
        mathematical, staged = module.mc2_cpu_references(ctx, peers[rank], attrs)
        assert calls == [0, 1]
        _exact(mathematical, expected_math)
        _exact(staged, expected_stage)

        # The old mathematical entry point must stay bit-identical. Its single
        # collective receives the same local G as the new hook's first call.
        legacy_ctx, legacy_calls, _, _ = _cpu_context(peers, attrs, rank)
        legacy = module.mc2_cpu_golden(legacy_ctx, peers[rank], attrs)
        assert legacy_calls == [0]
        _exact(mathematical, legacy)

        native_cpu = _native_result(module, peers, attrs, rank)
        _exact(native_cpu.float(), staged)
        assert _reference_health(native_cpu, mathematical)["passed"] is False
        assert _reference_health(native_cpu, mathematical, staged)["passed"] is True
        checked = compare_mc2_outputs(
            native_cpu, native_cpu, mathematical, rank, cascade_golden=staged
        )[0]
        assert checked.passed and checked.metadata["reference_valid"]
