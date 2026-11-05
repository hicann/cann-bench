# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""AllGather reference health models rounding without hiding candidate errors."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kernel_eval.checkers.mc2_three_way_checker import (
    check_mc2_references,
    compare_mc2_outputs,
)


TASK = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/kernel_bench/level4/all_gather_matmul"
)


def load_task():
    spec = importlib.util.spec_from_file_location(
        "allgather_staged_test", TASK / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(shards, rank):
    calls = []

    def gather(output, local):
        assert local.device.type == "cpu" and local.dtype == torch.float32
        torch.testing.assert_close(local, shards[rank].float(), rtol=0, atol=0)
        output.copy_(torch.cat([shard.float() for shard in shards]))
        calls.append(1)

    return {
        "rank": rank,
        "world_size": len(shards),
        "dist": SimpleNamespace(_all_gather_base=gather),
    }, calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("gather_output", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_distinct_rank_inputs_and_output_contract(
    dtype, transpose, with_bias, gather_output, rank
):
    generator = torch.Generator().manual_seed(841)
    shards = [torch.randn((3, 64), generator=generator).to(dtype) for _ in range(2)]
    # Deliberately use different weights and bias on each rank.
    weights = [torch.randn((64, 7), generator=generator).to(dtype) for _ in range(2)]
    biases = [torch.randn((7,), generator=generator).to(dtype) for _ in range(2)]
    inputs = {
        "x1": shards[rank],
        "x2": weights[rank].t().contiguous() if transpose else weights[rank],
        "bias": biases[rank] if with_bias else None,
    }
    attrs = {"is_trans_b": transpose, "gather_output": gather_output}
    ctx, calls = context(shards, rank)
    module = load_task()
    math, staged = module.mc2_cpu_references(ctx, inputs, attrs)
    assert len(calls) == 1
    original = module.mc2_cpu_golden(ctx, inputs, attrs)
    gathered = torch.cat(shards).float()
    # An independent FP64 oracle avoids requiring CPU BLAS implementations
    # with different operand strides to produce bit-identical FP32 sums.
    expected = gathered.double() @ weights[rank].double()
    magnitude = gathered.double().abs() @ weights[rank].double().abs()
    operations = 2 * gathered.shape[1] + int(with_bias)
    eps = torch.finfo(torch.float32).eps
    bound = operations * eps / (1 - operations * eps) * magnitude
    # Rounding-stage checks, unlike the mathematical oracle, keep the effective
    # operand layout fixed so a BLAS reduction change cannot cross a cast tie.
    effective_weight = inputs["x2"].float()
    if transpose:
        effective_weight = effective_weight.t()
    expected_stage = (gathered @ effective_weight).to(dtype).float()
    if with_bias:
        expected += biases[rank].double()
        bound += eps * (magnitude + biases[rank].double().abs())
        expected_stage = (expected_stage + biases[rank].float()).to(dtype).float()
    if gather_output:
        for result in (math, staged, original):
            torch.testing.assert_close(result[1], gathered, rtol=0, atol=0)
        math, staged, original = math[0], staged[0], original[0]
    assert torch.all((math.double() - expected).abs() <= bound)
    for actual, wanted in ((math, original), (staged, expected_stage)):
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,value", [(torch.float16, 1.015625), (torch.bfloat16, 1.0625)]
)
def test_sparse_small_value_roundoff_preserves_reference_health_and_scoring(
    dtype, value
):
    shards = [torch.full((64, 1), value, dtype=dtype) for _ in range(2)]
    weight = torch.full((1, 64), value, dtype=dtype)
    weight[0, 0] = 0.0002 / value
    ctx, _ = context(shards, 0)
    mathematical, staged = load_task().mc2_cpu_references(
        ctx, {"x1": shards[0], "x2": weight, "bias": None}, {}
    )
    native = staged.to(dtype)
    # Synthetic sparse device roundoff: this is a checker regression, not an
    # assertion that CPU emulation has reproduced a particular NPU accumulator.
    native[:8, 0] = (mathematical[:8, 0] + 0.00002).to(dtype)
    assert not check_mc2_references(native, mathematical, 0)[0].passed
    kwargs = {"cascade_golden": staged}
    good = compare_mc2_outputs(native, native, mathematical, 0, **kwargs)[0]
    assert good.passed, good.error_msg
    assert good.metadata["reference_check"]["reference"] == "cpu_staged_cascade"
    assert good.metadata["candidate_metrics"]["diff_rmse"] > 0
    for broken in (native * 2, torch.zeros_like(native)):
        bad_candidate = compare_mc2_outputs(broken, native, mathematical, 0, **kwargs)[
            0
        ]
        assert not bad_candidate.passed and bad_candidate.metadata["reference_valid"]
        bad_reference = check_mc2_references(broken, mathematical, 0, **kwargs)[0]
        assert not bad_reference.passed
        assert bad_reference.metadata["failure_type"] == "reference_invalid"


@pytest.mark.parametrize("world_size,dtype", [(4, torch.float16), (2, torch.float32)])
def test_outside_two_rank_low_precision_scope_keeps_existing_reference(
    world_size, dtype
):
    shards = [torch.ones((2, 3), dtype=dtype) for _ in range(world_size)]
    ctx, calls = context(shards, 0)
    mathematical, staged = load_task().mc2_cpu_references(
        ctx, {"x1": shards[0], "x2": torch.ones((3, 4), dtype=dtype)}, {}
    )
    assert len(calls) == 1 and staged is None
    torch.testing.assert_close(mathematical, torch.full((world_size * 2, 4), 3.0))


@pytest.mark.parametrize("case_id", [25, 35])
@pytest.mark.parametrize("rank", [0, 1])
def test_real_case_seed_and_k_produce_expected_fp16_overflow(case_id, rank):
    from kernel_eval.eval.mc2_distributed_runner import _make_rank_inputs

    with (TASK / "cases.csv").open(encoding="utf-8", newline="") as stream:
        row = next(
            row for row in csv.DictReader(stream) if int(row["case_id"]) == case_id
        )
    payload = {
        "attrs": json.loads(row["attrs"]),
        "input_shapes": json.loads(row["input_shape"]),
        "dtypes": json.loads(row["dtype"]),
        "value_ranges": json.loads(row["value_range"]),
    }
    # Preserve real K, seed, ranges, dtype, transpose and first rows of the RNG
    # stream. Reduce M/N only, avoiding a multi-billion-op CPU regression test.
    payload["input_shapes"][0][0] = 4
    payload["input_shapes"][1][0] = 8
    if payload["input_shapes"][2] is not None:
        payload["input_shapes"][2][0] = 8
    inputs = [_make_rank_inputs(payload, r, torch.device("cpu")) for r in (0, 1)]
    shards = [item["x1"] for item in inputs]
    ctx, _ = context(shards, rank)
    mathematical, staged = load_task().mc2_cpu_references(
        ctx, inputs[rank], payload["attrs"]
    )
    gathered = torch.cat(shards)
    # Exercise real CPU low-precision matmul and add separately, not native=staged.
    output = gathered @ inputs[rank]["x2"].t()
    if inputs[rank]["bias"] is not None:
        output = output + inputs[rank]["bias"]
    native = (output, gathered)
    assert mathematical[0][0, 0] > torch.finfo(torch.float16).max
    assert torch.isfinite(mathematical[0]).all()
    assert torch.isposinf(staged[0][0, 0]) and torch.isposinf(output[0, 0])
    results = compare_mc2_outputs(
        native, native, mathematical, rank, cascade_golden=staged
    )
    assert all(result.passed for result in results), [
        result.error_msg for result in results
    ]
    assert results[0].metadata["reference_check"]["expected_overflow_count"] >= 4
    for result in results:
        json.dumps(result.to_dict(), allow_nan=False)
