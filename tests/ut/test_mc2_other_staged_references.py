# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU-only checks for two-rank MC2 stage-rounded reference hooks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK_ROOT = Path(__file__).resolve().parents[2] / "bench_lab/kernel_bench/level4"
OPS = ("matmul_all_reduce", "allto_all_matmul", "matmul_allto_all")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_stage_test", TASK_ROOT / name / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _inputs(op, dtype, transposed, with_bias):
    generator = torch.Generator().manual_seed(743)
    weight_rows = 10 if op == "allto_all_matmul" else 5
    weight = torch.randn((weight_rows, 6), generator=generator).to(dtype)
    result = []
    for _ in range(2):
        rows = 4 if op == "allto_all_matmul" else 3
        result.append(
            {
                "x1": torch.randn((rows, 5), generator=generator).to(dtype),
                "x2": weight.t().contiguous() if transposed else weight,
                "bias": torch.randn((6,), generator=generator) if with_bias else None,
            }
        )
    return result


def _matmul(item, transposed):
    weight = item["x2"].t() if transposed else item["x2"]
    return item["x1"].float() @ weight.float()


def _alltoall_dist(rank, sends):
    calls = []

    def all_to_all_single(received, sent):
        call = len(calls)
        assert call < len(sends)
        _assert_exact(sent, sends[call][rank])
        received.copy_(torch.stack([sends[call][src][rank] for src in range(2)]))
        calls.append(call)

    return SimpleNamespace(all_to_all_single=all_to_all_single), calls


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_two_rank_math_and_rounding(op, dtype, transposed, with_bias, rank):
    module = _load(op)
    inputs = _inputs(op, dtype, transposed, with_bias)
    attrs = {
        "is_trans_b": transposed,
        "is_trans_x2": transposed,
        "is_bias": with_bias,
        "alltoall_out_flag": with_bias,
        "reduce_op": "sum",
    }

    if op == "matmul_all_reduce":
        # addmm fuses bias before the first low-precision cast.
        local = [_matmul(item, transposed) for item in inputs]
        if with_bias:
            local = [value + item["bias"] for value, item in zip(local, inputs)]
        staged = [value.to(dtype).float() for value in local]
        expected_math = local[0] + local[1]
        expected_stage = (staged[0] + staged[1]).to(dtype).float()
        calls = []

        def all_reduce(tensor, op):
            index = len(calls)
            source = local if index == 0 else staged
            _assert_exact(tensor, source[rank])
            tensor.copy_(source[0] + source[1])
            calls.append(index)

        dist = SimpleNamespace(
            ReduceOp=SimpleNamespace(SUM="sum"), all_reduce=all_reduce
        )
    elif op == "allto_all_matmul":
        # All-to-all x1 swaps row blocks, then interleaves sender columns.
        sends = [[item["x1"].float().reshape(2, 2, 5) for item in inputs]]
        dist, calls = _alltoall_dist(rank, sends)
        gathered = torch.stack([sends[0][src][rank] for src in range(2)])
        gathered = gathered.permute(1, 0, 2).contiguous().reshape(2, 10)
        weight = inputs[rank]["x2"]
        weight = weight.t() if transposed else weight
        expected_math = gathered @ weight.float()
        expected_stage = expected_math.to(dtype).float()
        if with_bias:
            expected_math = expected_math + inputs[rank]["bias"]
            expected_stage = (expected_stage + inputs[rank]["bias"]).to(dtype).float()
        expected_math = (expected_math, gathered if with_bias else None)
        expected_stage = (expected_stage, gathered if with_bias else None)
    else:
        # Matmul rounds before the optional bias, then all-to-all permutes.
        local = [_matmul(item, transposed) for item in inputs]
        staged = [value.to(dtype).float() for value in local]
        if with_bias:
            local = [value + item["bias"] for value, item in zip(local, inputs)]
            staged = [
                (value + item["bias"]).to(dtype).float()
                for value, item in zip(staged, inputs)
            ]
        sends = [
            [value.reshape(3, 2, 3).permute(1, 0, 2).contiguous() for value in source]
            for source in (local, staged)
        ]
        dist, calls = _alltoall_dist(rank, sends)
        expected_math = torch.stack([sends[0][src][rank] for src in range(2)]).reshape(
            6, 3
        )
        expected_stage = torch.stack([sends[1][src][rank] for src in range(2)]).reshape(
            6, 3
        )

    ctx = {"rank": rank, "world_size": 2, "device": torch.device("cpu"), "dist": dist}
    mathematical, health = module.mc2_cpu_references(ctx, inputs[rank], attrs)
    assert len(calls) == (1 if op == "allto_all_matmul" else 2)
    if op == "allto_all_matmul":
        for actual, expected in zip(mathematical, expected_math):
            if expected is None:
                assert actual is None
            else:
                _assert_exact(actual, expected)
        for actual, expected in zip(health, expected_stage):
            if expected is None:
                assert actual is None
            else:
                _assert_exact(actual, expected)
    else:
        _assert_exact(mathematical, expected_math)
        _assert_exact(health, expected_stage)
    if op == "matmul_all_reduce":
        assert health.dtype == torch.float32


@pytest.mark.parametrize("op", OPS)
def test_non_two_rank_nonquant_paths_keep_legacy_golden(monkeypatch, op):
    module = _load(op)
    sentinel = object()
    monkeypatch.setattr(module, "mc2_cpu_golden", lambda *_: sentinel)
    for attrs, world_size in (({}, 4),):
        math, health = module.mc2_cpu_references({"world_size": world_size}, {}, attrs)
        assert math is sentinel and health is None


@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_non_two_rank_quant_matmul_allreduce_keeps_legacy_golden(
    monkeypatch, world_size
):
    module = _load("matmul_all_reduce")
    sentinel = object()
    monkeypatch.setattr(module, "mc2_cpu_golden", lambda *_: sentinel)
    math, health = module.mc2_cpu_references(
        {"world_size": world_size}, {}, {"quantized": True}
    )
    assert math is sentinel and health is None


@pytest.mark.parametrize(
    "input_dtype,output_dtype", [(torch.int8, "float32"), (torch.float16, "float16")]
)
def test_unmodeled_quant_matmul_allreduce_keeps_legacy_golden(
    monkeypatch, input_dtype, output_dtype
):
    module = _load("matmul_all_reduce")
    sentinel = object()
    monkeypatch.setattr(module, "mc2_cpu_golden", lambda *_: sentinel)
    inputs = {name: torch.ones((1, 1), dtype=input_dtype) for name in ("x1", "x2")}
    math, health = module.mc2_cpu_references(
        {"world_size": 2}, inputs, {"quantized": True, "out_dtype": output_dtype}
    )
    assert math is sentinel and health is None
