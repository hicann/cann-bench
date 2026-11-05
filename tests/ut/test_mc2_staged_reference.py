# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Two-rank low-precision cascade health must model intermediate rounding."""

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
from kernel_eval.utils.mc2_compare import compare_three_way


TASK = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/kernel_bench/level4/matmul_reduce_scatter_v2"
)


def load_task():
    spec = importlib.util.spec_from_file_location("mrs_v2_staged", TASK / "golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def references(module, inputs, attrs, rank):
    partials, rounded = [], []
    for item in inputs:
        weight = item["x2"].t() if attrs.get("is_trans_b") else item["x2"]
        mm = item["x1"].float() @ weight.float()
        staged = mm.to(item["x1"].dtype)
        if item.get("bias") is not None:
            mm = mm + item["bias"].float()
            staged = staged + item["bias"]
        partials.append(mm)
        rounded.append(staged.float())
    mathematical = partials[0] + partials[1]
    staged_sum = rounded[0] + rounded[1]
    calls = []

    def all_reduce(tensor, op):
        # CPU collectives remain FP32; emulate two distinct ranks, not local * 2.
        index = len(calls)
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, (partials if index == 0 else rounded)[rank])
        tensor.copy_(mathematical if index == 0 else staged_sum)
        calls.append(index)

    ctx = {
        "rank": rank,
        "world_size": 2,
        "device": torch.device("cpu"),
        "dist": SimpleNamespace(
            ReduceOp=SimpleNamespace(SUM="sum"), all_reduce=all_reduce
        ),
    }
    hook = getattr(module, "mc2_cpu_references", None)
    if hook is None:  # Exercise the pre-fix path and its exact false rejection.
        cpu, health = module.mc2_cpu_golden(ctx, inputs[rank], attrs), None
    else:
        cpu, health = hook(ctx, inputs[rank], attrs)
        assert len(calls) == 2
    chunk = mathematical.shape[0] // 2
    selected = slice(rank * chunk, (rank + 1) * chunk)
    torch.testing.assert_close(cpu, mathematical[selected], rtol=0, atol=0)
    native = staged_sum.to(inputs[rank]["x1"].dtype)[selected]
    if health is not None:
        assert health.dtype == torch.float32
        torch.testing.assert_close(health, native.float(), rtol=0, atol=0)
    return cpu, native, health, rounded[rank][selected].to(native.dtype)


@pytest.mark.parametrize("case_id", [29, 34])
@pytest.mark.parametrize("rank", [0, 1])
def test_real_bf16_cancellation_is_not_reference_failure(case_id, rank):
    module = load_task()
    with (TASK / "cases.csv").open(encoding="utf-8", newline="") as stream:
        row = next(
            row for row in csv.DictReader(stream) if int(row["case_id"]) == case_id
        )
    attrs = json.loads(row["attrs"])
    payload = {
        "attrs": attrs,
        "input_shapes": json.loads(row["input_shape"]),
        "dtypes": json.loads(row["dtype"]),
        "value_ranges": json.loads(row["value_range"]),
    }
    inputs = [
        module.mc2_make_rank_inputs({"rank": r, "device": torch.device("cpu")}, payload)
        for r in (0, 1)
    ]
    cpu, native, health, local_only = references(module, inputs, attrs, rank)
    # Demonstrate why the old endpoint-only gate is not suitable for this cascade.
    assert not check_mc2_references(native, cpu, rank)[0].passed
    kwargs = {"cascade_golden": health} if health is not None else {}
    result = compare_mc2_outputs(native, native, cpu, rank, **kwargs)[0]
    assert result.passed, result.error_msg
    expected = compare_three_way(
        native.float().numpy(), native.float().numpy(), cpu.numpy(), "bfloat16"
    )
    assert result.metadata["cascade_metrics"] == expected["cascade_metrics"]
    assert result.metadata["candidate_metrics"] == expected["candidate_metrics"]
    assert result.metadata["reference_check"]["reference"] == "cpu_staged_cascade"
    assert result.metadata["cascade_metrics"]["diff_rmse"] > 0
    # Neither candidate errors nor broken reference communication may be rescued.
    for broken in (native * 2, local_only, torch.zeros_like(native)):
        candidate_result = compare_mc2_outputs(broken, native, cpu, rank, **kwargs)[0]
        assert not candidate_result.passed
        assert candidate_result.metadata["reference_valid"]
        assert "candidate_metrics" in candidate_result.metadata
        reference_result = compare_mc2_outputs(broken, broken, cpu, rank, **kwargs)[0]
        assert not reference_result.passed
        assert reference_result.metadata["failure_type"] == "reference_invalid"
        assert "candidate_metrics" not in reference_result.metadata


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_stages_preserve_bias_transpose_and_rank_slices(dtype, transpose, bias, rank):
    generator = torch.Generator().manual_seed(31)
    inputs = []
    for _ in (0, 1):
        weight = torch.randn((32, 7), generator=generator).to(dtype)
        inputs.append(
            {
                "x1": torch.randn((8, 32), generator=generator).to(dtype),
                "x2": weight.t() if transpose else weight,
                "bias": torch.randn((7,), generator=generator).to(dtype)
                if bias
                else None,
            }
        )
    attrs = {
        "x1_dtype": str(dtype).removeprefix("torch."),
        "x2_dtype": str(dtype).removeprefix("torch."),
        "is_trans_b": transpose,
    }
    cpu, native, health, _ = references(load_task(), inputs, attrs, rank)
    assert health is not None
    assert check_mc2_references(native, cpu, rank, cascade_golden=health)[0].passed


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "inf", "none", "structure"])
def test_malformed_staged_reference_fails_closed(kind):
    cpu = torch.ones(4)
    native = cpu.bfloat16()
    health = {
        "shape": torch.ones(1),
        "dtype": native,
        "nan": torch.full((4,), float("nan")),
        "inf": torch.full((4,), float("inf")),
        "none": [None],
        "structure": [cpu],
    }[kind]
    result = check_mc2_references(native, cpu, 0, cascade_golden=health)[0]
    assert not result.passed and result.metadata["failure_type"] == "reference_invalid"
    assert "candidate_metrics" not in result.metadata


@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_other_world_sizes_keep_original_reference_policy(world_size):
    module = load_task()
    calls = []

    def all_reduce(tensor, op):
        calls.append(tensor.clone())
        tensor.mul_(world_size)

    ctx = {
        "rank": 0,
        "world_size": world_size,
        "dist": SimpleNamespace(
            ReduceOp=SimpleNamespace(SUM="sum"), all_reduce=all_reduce
        ),
    }
    inputs = {
        "x1": torch.ones((8, 4), dtype=torch.float16),
        "x2": torch.ones((4, 3), dtype=torch.float16),
        "bias": None,
    }
    cpu, health = module.mc2_cpu_references(
        ctx, inputs, {"x1_dtype": "fp16", "x2_dtype": "fp16"}
    )
    assert len(calls) == 1 and health is None
    torch.testing.assert_close(
        cpu, torch.full((8 // world_size, 3), float(4 * world_size))
    )


def test_optional_and_multiple_staged_outputs_keep_contract():
    cpu = (torch.ones(3), None)
    native = (cpu[0].half(), None)
    assert all(
        r.passed for r in check_mc2_references(native, cpu, 0, cascade_golden=cpu)
    )
    for broken in ((None, None), (cpu[0], torch.ones(1)), (cpu[0],)):
        assert any(
            not r.passed
            for r in check_mc2_references(native, cpu, 0, cascade_golden=broken)
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_staged_reference_never_rescues_nonfinite_math_mismatch(value):
    cpu, native, staged = torch.ones(4), torch.ones(4).half(), torch.ones(4)
    for bad_native, bad_cpu in (
        (torch.full_like(native, value), cpu),
        (native, torch.full_like(cpu, value)),
    ):
        result = check_mc2_references(bad_native, bad_cpu, 0, cascade_golden=staged)[0]
        assert (
            not result.passed and result.metadata["failure_type"] == "reference_invalid"
        )
