# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

BENCH = Path(__file__).resolve().parents[2] / "bench_lab/kernel_bench"

# Canonical semantic fingerprints of the published a18e6872 legacy cases and 910B metadata.
LEGACY_FINGERPRINTS = {
    "all_gather_matmul": (
        "3d1d212ed5f6dd311285d1129b4e53c13ff609f0ef9073a1e8dd08960fde0791",
        "131f7e7a4aa8a988685303b6caf6613e25ea8b673eaf4fe154964270edee7dfa",
    ),
    "matmul_all_reduce": (
        "258edb696f9500f58f20bd6f80e71f1aaa603bd54d9a6b709bd4968578817aa7",
        "c45347c8af4b9e1239bb56489e58f06e1cc1ae30f10a60bbe373f65d3e8ce95d",
    ),
}


def golden(op):
    spec = importlib.util.spec_from_file_location(
        op, BENCH / "level4" / op / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("op", ["matmul_all_reduce", "all_gather_matmul"])
def test_specs_and_csv_agree_and_legacy_baselines_are_preserved(op):
    folder = BENCH / "level4" / op
    cases = yaml.safe_load((folder / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    with (folder / "cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for case, row in zip(cases, rows):
        for key in ["input_shape", "dtype", "attrs", "value_range"]:
            assert case[key] == json.loads(row[key])
        if case["case_id"] <= 20:
            assert case["attrs"]["world_size"] == 8
            assert "mc2_accuracy_mode" not in case["attrs"]
            assert case["dtype"][0] in ("float16", "bfloat16")
        else:
            assert case["attrs"]["world_size"] == 2
            assert case["attrs"]["mc2_accuracy_mode"] == "three_way"
    assert len(cases) == len(rows)
    original = json.loads((BENCH / "metadata/910b2.json").read_text(encoding="utf-8"))[
        "level4"
    ][op]
    for value, expected in zip((cases[:20], original), LEGACY_FINGERPRINTS[op]):
        canonical = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == expected
    for path in (BENCH / "metadata").glob("*.json"):
        entries = (
            json.loads(path.read_text(encoding="utf-8")).get("level4", {}).get(op, {})
        )
        for i in range(1, 21):
            entry = entries.get(str(i), {})
            if path.stem == "910b2":
                assert entry["t_hw_us"] > 0
                assert entry["baseline_perf_us"] >= entry["t_hw_us"]
            else:
                assert not entry


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_nonquantized_cpu_allreduce(transpose, bias_enabled):
    x = torch.tensor([[1.5, -0.75]], dtype=torch.bfloat16)
    w = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.bfloat16)
    b = torch.tensor([0.125, 0.25], dtype=torch.bfloat16) if bias_enabled else None
    dist = SimpleNamespace(
        ReduceOp=SimpleNamespace(SUM="sum"), all_reduce=lambda out, op: out.mul_(2)
    )
    out = golden("matmul_all_reduce").mc2_cpu_golden(
        {"dist": dist},
        {"x1": x, "x2": w.T if transpose else w, "bias": b},
        {"is_trans_b": transpose},
    )
    expected = x.float() @ w.float()
    if b is not None:
        expected += b.float()
    torch.testing.assert_close(out, expected * 2)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("gather_output", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_cpu_allgather_uses_real_rank_data(gather_output, transpose):
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    other = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    w = torch.tensor([[2.0], [3.0]], dtype=torch.float16)
    b = torch.tensor([0.25], dtype=torch.float16)

    def collect(out, local):
        out.copy_(torch.cat([local, other]))

    out = golden("all_gather_matmul").mc2_cpu_golden(
        {"world_size": 2, "dist": SimpleNamespace(_all_gather_base=collect)},
        {"x1": x, "x2": w.T if transpose else w, "bias": b},
        {"is_trans_b": transpose, "gather_output": gather_output},
    )
    expected_gather = torch.cat([x.float(), other])
    if gather_output:
        out, gathered = out
        torch.testing.assert_close(gathered, expected_gather)
    torch.testing.assert_close(out, expected_gather @ w.float() + b.float())


def test_allgather_twenty_cases_cover_all_attribute_combinations():
    cases = yaml.safe_load(
        (BENCH / "level4/all_gather_matmul/cases.yaml").read_text(encoding="utf-8")
    )["cases"]
    cases = [
        case for case in cases if case["attrs"].get("mc2_accuracy_mode") == "three_way"
    ]
    assert [c["case_id"] for c in cases] == list(range(21, 41))
    combos = [
        (c["dtype"][0], c["attrs"]["is_bias"], c["attrs"]["gather_output"])
        for c in cases
    ]
    assert len(set(combos)) == 8
    assert sum(dtype == "bfloat16" for dtype, _, _ in combos) == 10
    assert sum(bias for _, bias, _ in combos) == 10
    assert sum(gather for _, _, gather in combos) == 10
    for case in cases:
        n = case["input_shape"][1][0 if case["attrs"]["is_trans_b"] else 1]
        assert case["input_shape"][2] == ([n] if case["attrs"]["is_bias"] else None)


@pytest.mark.parametrize(
    "op",
    ["matmul_all_reduce", "all_gather_matmul", "allto_all_matmul", "matmul_allto_all"],
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_rank_distinct_inputs_reject_broken_communication(op, dtype, rank, auxiliary):
    from kernel_eval.eval.mc2_distributed_runner import _make_rank_inputs
    from kernel_eval.checkers.mc2_three_way_checker import compare_mc2_outputs

    module = golden(op)
    attrs = {
        "seed": 23,
        "world_size": 2,
        "mc2_accuracy_mode": "three_way",
        "is_bias": False,
        "is_trans_b": False,
        "is_trans_x2": False,
        "gather_output": auxiliary,
        "alltoall_out_flag": auxiliary,
    }
    payload = {
        "input_shapes": [[8, 16], [32 if op == "allto_all_matmul" else 16, 12], None],
        # Isolate communication semantics from cancellation/reference health.
        "dtypes": [dtype] * 3,
        "value_ranges": [[0.25, 1.25], [0.25, 1.25], None],
        "attrs": attrs,
    }

    def make(r):
        if hasattr(module, "mc2_make_rank_inputs"):
            return module.mc2_make_rank_inputs(
                {"rank": r, "device": torch.device("cpu")}, payload
            )
        return _make_rank_inputs(payload, r, torch.device("cpu"))

    inputs = [make(0), make(1)]
    assert torch.equal(inputs[0]["x1"], make(0)["x1"])
    assert not torch.equal(inputs[0]["x1"], inputs[1]["x1"])
    xs = [item["x1"] for item in inputs]
    w = inputs[rank]["x2"]
    if op == "matmul_allto_all":
        partials = [(item["x1"] @ item["x2"]).chunk(2, dim=1)[rank] for item in inputs]
        native = torch.cat(partials, dim=0)
        cpu = torch.cat(
            [
                (item["x1"].float() @ item["x2"].float()).chunk(2, dim=1)[rank]
                for item in inputs
            ],
            dim=0,
        )
        wrong = [
            torch.cat(partials[::-1], dim=0),
            torch.cat([partials[rank]] * 2, dim=0),
        ]
    elif op == "matmul_all_reduce":
        partials = [item["x1"] @ item["x2"] for item in inputs]
        native = partials[0] + partials[1]
        cpu = sum(item["x1"].float() @ item["x2"].float() for item in inputs)
        # Both local-repeat and omitted-peer substitutions must fail.
        wrong = [partials[rank] * 2, partials[rank]]
    else:
        if op == "all_gather_matmul":
            parts, axis = xs, 0
            local = xs[rank]
        else:
            parts, axis = [x.chunk(2, dim=0)[rank] for x in xs], 1
            local = parts[rank]
        gathered = torch.cat(parts, dim=axis)

        def output(g, fp32=False):
            result = g.float() @ w.float() if fp32 else g @ w
            aux = g.float() if fp32 else g
            if op == "all_gather_matmul":
                return (result, aux) if auxiliary else result
            return result, aux if auxiliary else None

        native, cpu = output(gathered), output(gathered, fp32=True)
        wrong = [
            output(torch.cat([local, local], dim=axis)),
            output(torch.cat(parts[::-1], dim=axis)),
        ]

    assert all(r.passed for r in compare_mc2_outputs(native, native, cpu, rank))
    for broken in wrong:
        assert not all(r.passed for r in compare_mc2_outputs(broken, native, cpu, rank))


def test_matmul_alltoall_rank_seed_preserves_rank0_and_legacy_inputs():
    module = golden("matmul_allto_all")
    attrs = {"seed": 23, "is_bias": True, "world_size": 2}
    payload = {
        "input_shapes": [[8, 16], [16, 12], [12]],
        "dtypes": ["float16"] * 3,
        "value_ranges": [[-1, 1]] * 3,
        "attrs": attrs,
    }
    legacy = [
        module.mc2_make_rank_inputs({"rank": r, "device": "cpu"}, payload)
        for r in (0, 1)
    ]
    for name in ("x1", "x2", "bias"):
        torch.testing.assert_close(legacy[0][name], legacy[1][name], rtol=0, atol=0)
    attrs["mc2_accuracy_mode"] = "three_way"
    current = [
        module.mc2_make_rank_inputs({"rank": r, "device": "cpu"}, payload)
        for r in (0, 1)
    ]
    for name in ("x1", "x2", "bias"):
        torch.testing.assert_close(current[0][name], legacy[0][name], rtol=0, atol=0)
    for name in ("x2", "bias"):
        torch.testing.assert_close(current[1][name], legacy[1][name], rtol=0, atol=0)
    assert not torch.equal(current[0]["x1"], current[1]["x1"])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("profile", ["1", "2"])
def test_cancelling_matmul_allreduce_reference_invalidates_scoring(dtype, profile):
    from kernel_eval.base.result import FAILURE_TYPE_REFERENCE_INVALID
    from kernel_eval.checkers.mc2_three_way_checker import compare_mc2_outputs
    from kernel_eval.report.scoring import aggregate_eq4

    module = golden("matmul_all_reduce")
    payload = {
        "input_shapes": [[8, 16], [16, 12], None],
        "dtypes": [dtype] * 3,
        "value_ranges": [[-5, 5], [-5, 5], None],
        "attrs": {
            "seed": 23,
            "world_size": 2,
            "mc2_accuracy_mode": "three_way",
            "is_bias": False,
            "is_trans_b": False,
        },
    }
    inputs = [
        module.mc2_make_rank_inputs({"rank": rank, "device": "cpu"}, payload)
        for rank in range(2)
    ]
    partials = [item["x1"] @ item["x2"] for item in inputs]
    native = partials[0] + partials[1]
    cpu = sum(item["x1"].float() @ item["x2"].float() for item in inputs)
    # Multi-stage low-precision rounding can exceed the independent reference
    # limits under cancellation. Candidate == cascade must not bless that B.
    result = compare_mc2_outputs(native, native, cpu, 0)[0]
    assert not result.passed
    assert result.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    assert result.metadata["reference_check"]["normal_error_count"] > 0
    score = aggregate_eq4(
        True, 1, [(False, None)], weights=profile, n_reference_invalid=1
    )
    assert score["total_score"] is None
    assert score["bonus_score"] is None
    assert score["score_error_code"] == FAILURE_TYPE_REFERENCE_INVALID


def test_legacy_generic_input_seed_is_unchanged():
    from kernel_eval.eval.mc2_distributed_runner import _make_rank_inputs

    payload = {
        "input_shapes": [[4, 8], [8, 4], None],
        "dtypes": ["float16"] * 3,
        "value_ranges": [[-5, 5], [-5, 5], None],
        "attrs": {"seed": 4},
    }
    a = _make_rank_inputs(payload, 0, torch.device("cpu"))
    b = _make_rank_inputs(payload, 1, torch.device("cpu"))
    assert torch.equal(a["x1"], b["x1"])


def test_legacy_matmul_allreduce_keeps_identical_rank_inputs():
    module = golden("matmul_all_reduce")
    payload = {
        "input_shapes": [[4, 8], [8, 4], [4]],
        "dtypes": ["float16"] * 3,
        "value_ranges": [[-5, 5]] * 3,
        "attrs": {"seed": 4, "world_size": 8, "is_bias": True},
    }
    first = module.mc2_make_rank_inputs({"rank": 0, "device": "cpu"}, payload)
    for rank in (1, 7):
        other = module.mc2_make_rank_inputs({"rank": rank, "device": "cpu"}, payload)
        assert all(torch.equal(first[name], other[name]) for name in first)
