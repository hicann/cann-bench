# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""The FP8 cascade must implement accumulator bias, not output bias.

The fake suboperator distinguishes the two documented npu_quant_matmul modes;
it is an API-contract test, not a substitute for an Ascend runtime test.
"""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kernel_eval.checkers.mc2_three_way_checker import compare_mc2_outputs


TASK = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/kernel_bench/level4/all_gather_matmul_v2"
)


def load_task():
    spec = importlib.util.spec_from_file_location("agmv2_reference", TASK / "golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuantMatmulContract:
    def __init__(self):
        self.calls = []

    def npu_trans_quant_param(self, scale):
        self.calls.append(("encode", scale.clone()))
        bits = scale.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFE000
        return bits | (1 << 46)

    def npu_quant_matmul(
        self, x1, x2, scale, *, offset, pertoken_scale, bias, output_dtype, group_sizes
    ):
        self.calls.append(("mm", scale.clone(), pertoken_scale))
        value = x1.float() @ x2.float()
        if scale.dtype == torch.int64:
            assert pertoken_scale is None
            decoded = (scale & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
            if bias is not None:
                value = value + bias.float()
            value = value * decoded
        else:
            # FP8 + two FP32 scales has post-dequant FP32 bias.
            product = scale * pertoken_scale
            product = (product.contiguous().view(torch.int32) & -8192).view(
                torch.float32
            )
            value = value * product
            if bias is not None:
                value = value + bias.float()
        return value.to(output_dtype)


@pytest.mark.parametrize("out_dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("gather_output", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_cascade_bias_is_scaled_like_cpu_golden(
    out_dtype, transpose, gather_output, bias
):
    module = load_task()
    weight = torch.tensor([[1.0, 2.0, -1.0], [2.0, -1.0, 0.5]]).to(torch.float8_e5m2)
    inputs = {
        "x1": torch.tensor([[1.0, 2.0]]).to(torch.float8_e4m3fn),
        "x2": weight.t() if transpose else weight,
        "bias": torch.tensor([0.25, -0.125, 0.5]) if bias else None,
        "x1_scale": torch.tensor([0.3333]),
        "x2_scale": torch.tensor([0.2222]),
    }
    snapshot = {
        key: value.clone() for key, value in inputs.items() if value is not None
    }
    # Two different ranks: duplicating the local input must not pass.
    gathered = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    calls = []

    def gather(out, local):
        calls.append(tuple(local.shape))
        out.copy_(gathered.to(out.dtype))

    native = QuantMatmulContract()
    ctx = {
        "world_size": 2,
        "dist": SimpleNamespace(_all_gather_base=gather),
        "torch_npu": native,
    }
    attrs = {
        "mc2_accuracy_mode": "three_way",
        "x1_dtype": "fp8_e4m3fn",
        "x2_dtype": "fp8_e5m2",
        "out_dtype": out_dtype,
        "is_trans_b": transpose,
        "gather_output": gather_output,
        "block_size": 0,
        "group_sizes": [],
    }
    cpu = module.mc2_cpu_golden(ctx, inputs, attrs)
    mathematical, staged = module.mc2_cpu_references(ctx, inputs, attrs)
    result = module.mc2_distributed_golden(ctx, inputs, attrs)
    cpu_y, actual_y = (cpu[0], result[0]) if gather_output else (cpu, result)
    math_y, stage_y = (
        (mathematical[0], staged[0]) if gather_output else (mathematical, staged)
    )
    torch.testing.assert_close(math_y, cpu_y, rtol=0, atol=0)
    torch.testing.assert_close(
        stage_y, cpu_y.to(getattr(torch, out_dtype)).float(), rtol=0, atol=0
    )
    assert math_y.dtype == stage_y.dtype == torch.float32
    torch.testing.assert_close(
        actual_y, cpu_y.to(getattr(torch, out_dtype)), rtol=0, atol=0
    )
    assert [call[0] for call in native.calls] == ["encode", "mm"]
    assert native.calls[-1][1].dtype == torch.int64 and native.calls[-1][2] is None
    assert calls == [(1, 2)] * 3  # scales must not be gathered for per-tensor
    if gather_output:
        assert result[1].dtype == inputs["x1"].dtype
        torch.testing.assert_close(result[1].float(), gathered, rtol=0, atol=0)
        torch.testing.assert_close(staged[1], gathered, rtol=0, atol=0)
    for key, saved in snapshot.items():
        torch.testing.assert_close(inputs[key].float(), saved.float(), rtol=0, atol=0)


def test_legacy_int8_cascade_keeps_row_and_column_scales():
    module = load_task()
    x1 = torch.ones((2, 2), dtype=torch.int8)
    inputs = {
        "x1": x1,
        "x2": x1.clone(),
        "bias": None,
        "x1_scale": torch.tensor([0.1, 0.2]),
        "x2_scale": torch.tensor([0.3, 0.4]),
    }
    calls = []

    def gather(output, local):
        calls.append(local.shape)
        output.copy_(torch.cat([local, local]))

    def quant_mm(a, b, scale, **kwargs):
        assert scale is inputs["x2_scale"]
        torch.testing.assert_close(
            kwargs["pertoken_scale"], torch.tensor([0.1, 0.2, 0.1, 0.2])
        )
        return torch.ones((4, 2), dtype=kwargs["output_dtype"])

    ctx = {
        "world_size": 2,
        "dist": SimpleNamespace(_all_gather_base=gather),
        "torch_npu": SimpleNamespace(npu_quant_matmul=quant_mm),
    }
    result = module.mc2_distributed_golden(
        ctx, inputs, {"x1_dtype": "int8", "gather_output": False}
    )
    assert result.shape == (4, 2) and len(calls) == 2


@pytest.mark.parametrize("out_dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("transposed", [False, True])
def test_public_fp8_reference_uses_the_same_accumulator_bias_and_scale(
    out_dtype, transposed
):
    module = load_task()
    x1 = torch.tensor([[1.0, 2.0], [-0.5, 0.75]]).to(torch.float8_e4m3fn)
    weight = torch.tensor([[1.0, -1.5, 0.5], [0.5, 2.0, -0.75]]).to(torch.float8_e5m2)
    bias = torch.tensor([1.5, 2.0, -0.5])
    s1, s2 = torch.tensor([0.3333]), torch.tensor([0.2222])
    x2 = weight.t() if transposed else weight
    actual, actual_gathered = module.all_gather_matmul_v2(
        x1,
        x2,
        bias,
        s1,
        s2,
        world_size=2,
        gather_output=True,
        is_trans_b=transposed,
        out_dtype=out_dtype,
    )

    def gather(output, local):
        output.copy_(torch.cat([local, local]))

    ctx = {"world_size": 2, "dist": SimpleNamespace(_all_gather_base=gather)}
    attrs = {"gather_output": True, "is_trans_b": transposed}
    cpu, gathered = module.mc2_cpu_golden(
        ctx, {"x1": x1, "x2": x2, "bias": bias, "x1_scale": s1, "x2_scale": s2}, attrs
    )
    torch.testing.assert_close(
        actual, cpu.to(getattr(torch, out_dtype)), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_gathered.float(), gathered, rtol=0, atol=0)
    # The old smoke formula put bias outside the scale and must differ.
    old = (gathered @ weight.float()) * s1 * s2 + bias
    assert not torch.allclose(actual.float(), old, rtol=0.01, atol=0.01)


def test_public_fp8_reference_rejects_ambiguous_vector_scales():
    module = load_task()
    value = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="per-tensor"):
        module.all_gather_matmul_v2(
            value, value, x1_scale=torch.ones(2), x2_scale=torch.ones(2)
        )


@pytest.mark.parametrize("case_id", [21, 25, 28, 37])
def test_a5_fp32_case_cpu_stage_is_exact_mathematical_reference(case_id):
    """These reported cases need an independent FP32 reference-health bound.

    Keep their real dtype, K, scale, bias, and transpose options, while
    shrinking M/N so this contract check remains quick on a CPU-only host.
    """
    with (TASK / "cases.csv").open(newline="", encoding="utf-8") as stream:
        row = next(
            row for row in csv.DictReader(stream) if int(row["case_id"]) == case_id
        )
    shapes = json.loads(row["input_shape"])
    attrs = json.loads(row["attrs"])
    assert attrs["out_dtype"] == "float32"
    shapes[0][0] = 4
    if attrs["is_trans_b"]:
        shapes[1][0] = 8
    else:
        shapes[1][1] = 8
    if attrs["is_bias"]:
        shapes[4][0] = 8
    payload = {
        "input_shapes": shapes,
        "dtypes": json.loads(row["dtype"]),
        "value_ranges": json.loads(row["value_range"]),
        "attrs": attrs,
    }
    module = load_task()

    def gather(output, local):
        output.copy_(torch.cat([local, local], dim=0))

    ctx = {
        "rank": 0,
        "world_size": 2,
        "device": torch.device("cpu"),
        "dist": SimpleNamespace(_all_gather_base=gather),
    }
    inputs = module.mc2_make_rank_inputs(ctx, payload)
    mathematical, staged = module.mc2_cpu_references(ctx, inputs, attrs)
    math_y = mathematical[0] if isinstance(mathematical, tuple) else mathematical
    stage_y = staged[0] if isinstance(staged, tuple) else staged
    assert math_y.dtype == stage_y.dtype == torch.float32
    assert torch.equal(math_y, stage_y)


@pytest.mark.parametrize("case_id", range(21, 41))
def test_a5_rank_input_generator_produces_distinct_reproducible_shards(case_id):
    with (TASK / "cases.csv").open(newline="", encoding="utf-8") as stream:
        row = next(
            row for row in csv.DictReader(stream) if int(row["case_id"]) == case_id
        )
    shapes, attrs = json.loads(row["input_shape"]), json.loads(row["attrs"])
    # Keep real K, dtypes, ranges and seed, but limit CPU memory and GEMM cost.
    shapes[0][0] = 4
    shapes[1][0 if attrs["is_trans_b"] else 1] = 8
    if attrs["is_bias"]:
        shapes[4][0] = 8
    payload = {
        "input_shapes": shapes,
        "dtypes": json.loads(row["dtype"]),
        "value_ranges": json.loads(row["value_range"]),
        "attrs": attrs,
    }
    module = load_task()
    contexts = [
        {"rank": r, "world_size": 2, "device": torch.device("cpu")} for r in (0, 1)
    ]
    inputs = [module.mc2_make_rank_inputs(ctx, payload) for ctx in contexts]
    assert not torch.equal(inputs[0]["x1"].float(), inputs[1]["x1"].float())
    generator = torch.Generator().manual_seed(attrs["seed"])
    lo, hi = payload["value_ranges"][0]
    original_rank0 = (
        torch.empty(shapes[0])
        .uniform_(lo, hi, generator=generator)
        .to(inputs[0]["x1"].dtype)
    )
    assert torch.equal(inputs[0]["x1"].float(), original_rank0.float())

    def gather(output, local):
        output.copy_(
            torch.cat([item["x1"].float() for item in inputs]).to(output.dtype)
        )

    def repeat_local(output, local):
        output.copy_(torch.cat([local.float(), local.float()]).to(output.dtype))

    for ctx, expected in zip(contexts, inputs):
        repeated = module.mc2_make_rank_inputs(ctx, payload)
        for key, value in expected.items():
            if value is None:
                assert repeated[key] is None
            else:
                assert torch.equal(repeated[key].float(), value.float())
                assert repeated[key].dtype == value.dtype
        # Exercise the real maker through both references and the checker.
        # Only the device/collective boundary is substituted on this CPU host.
        ctx = {
            **ctx,
            "dist": SimpleNamespace(_all_gather_base=gather),
            "torch_npu": QuantMatmulContract(),
        }
        mathematical, staged = module.mc2_cpu_references(ctx, expected, attrs)
        native = module.mc2_distributed_golden(ctx, expected, attrs)
        good = compare_mc2_outputs(
            native, native, mathematical, ctx["rank"], cascade_golden=staged
        )
        assert all(result.passed for result in good), [
            result.error_msg for result in good
        ]
        broken_ctx = {**ctx, "dist": SimpleNamespace(_all_gather_base=repeat_local)}
        broken = module.mc2_distributed_golden(broken_ctx, expected, attrs)
        results = compare_mc2_outputs(
            broken, native, mathematical, ctx["rank"], cascade_golden=staged
        )
        assert all(result.metadata["reference_valid"] for result in results)
        assert not results[0].passed  # Matmul must fail even without gather_output.


def test_legacy_int8_rank_inputs_keep_original_seed():
    with (TASK / "cases.csv").open(newline="", encoding="utf-8") as stream:
        row = next(row for row in csv.DictReader(stream) if int(row["case_id"]) == 1)
    payload = {
        "input_shapes": [[4, 8], [8, 3], [4], [3]],
        "attrs": json.loads(row["attrs"]),
        "dtypes": json.loads(row["dtype"]),
        "value_ranges": json.loads(row["value_range"]),
    }
    generator = torch.Generator().manual_seed(payload["attrs"]["seed"])
    lo, hi = payload["value_ranges"][0]
    expected = torch.randint(lo, hi + 1, (4, 8), dtype=torch.int8, generator=generator)
    module = load_task()
    for rank in (0, 1):
        actual = module.mc2_make_rank_inputs(
            {"rank": rank, "device": torch.device("cpu")}, payload
        )
        assert torch.equal(actual["x1"], expected)
