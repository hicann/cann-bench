# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU-only contracts for AlltoAllQuantMatmul MXFP8 A5 cases."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "bench_lab/kernel_bench/level4/allto_all_matmul"
SPEC = importlib.util.spec_from_file_location(
    "alltoall_quant_golden", TASK / "golden.py"
)
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)


def _cases():
    return yaml.safe_load((TASK / "cases.yaml").read_text(encoding="utf-8"))["cases"]


def _attrs(**updates):
    attrs = dict(
        is_quant=True,
        is_trans_x2=True,
        x1_quant_mode=6,
        x2_quant_mode=6,
        group_size=4295032864,
        scale_dtype=293,
        out_dtype="bfloat16",
        alltoall_out_flag=True,
    )
    attrs.update(updates)
    return attrs


def _rank_inputs():
    ranks = []
    for rank in range(2):
        x = ((torch.arange(8 * 64).reshape(8, 64) % 13 - 6).float() / 2 + rank).to(
            torch.float8_e4m3fn
        )
        weight = ((torch.arange(5 * 128).reshape(5, 128) % 11 - 5).float() / 4).to(
            torch.float8_e5m2
        )
        x_scale = (torch.arange(8 * 2).reshape(8, 1, 2) % 4 + 124 + rank).to(
            torch.uint8
        )
        w_scale = (torch.arange(5 * 4).reshape(5, 2, 2) % 5 + 124).to(torch.uint8)
        ranks.append(
            dict(
                x1=x,
                x2=weight,
                x1_scale=x_scale,
                x2_scale=w_scale,
                bias=torch.linspace(-0.3, 0.3, 5),
            )
        )
    return ranks


class _StaticAllToAll:
    """Two-rank collective with independently prepared sender slices."""

    def __init__(self, ranks, rank):
        self.ranks = ranks
        self.rank = rank
        self.calls = 0

    def all_to_all_single(self, output, source):
        name = ("x1", "x1_scale")[self.calls]
        self.calls += 1
        pieces = []
        for inputs in self.ranks:
            value = inputs[name]
            if source.dtype == torch.uint8 and value.dtype != torch.uint8:
                value = value.view(torch.uint8)
            else:
                value = value.to(source.dtype)
            pieces.append(value.reshape(2, 4, -1)[self.rank])
        output.copy_(torch.stack(pieces))


def _ctx(ranks, rank):
    return dict(
        rank=rank,
        world_size=2,
        device=torch.device("cpu"),
        hcomm_info="test-hccl-group",
        dist=_StaticAllToAll(ranks, rank),
    )


def _independent_reference(ranks, rank):
    chunks, dequant_chunks = [], []
    for inputs in ranks:
        x = inputs["x1"].float()[rank * 4 : (rank + 1) * 4]
        scales = inputs["x1_scale"][rank * 4 : (rank + 1) * 4].reshape(4, 2)
        chunks.append(x)
        dequant_chunks.append(
            torch.cat(
                [
                    x[:, j * 32 : (j + 1) * 32]
                    * (2.0 ** (scales[:, j].float() - 127))[:, None]
                    for j in range(2)
                ],
                dim=1,
            )
        )
    inputs = ranks[rank]
    weight = inputs["x2"].float()
    scales = inputs["x2_scale"].reshape(5, 4)
    weight = torch.cat(
        [
            weight[:, j * 32 : (j + 1) * 32]
            * (2.0 ** (scales[:, j].float() - 127))[:, None]
            for j in range(4)
        ],
        dim=1,
    )
    return torch.cat(dequant_chunks, dim=1) @ weight.t() + inputs["bias"], torch.cat(
        chunks, dim=1
    )


def test_new_quantized_cases_keep_separate_id_range():
    cases = _cases()
    assert [case["case_id"] for case in cases] == list(range(1, 41))
    assert all(not case["attrs"].get("is_quant", False) for case in cases[:20])
    for case in cases[:20]:
        for field in ("input_shape", "dtype", "value_range"):
            assert len(case[field]) == 5
            assert case[field][-2:] == [None, None]
    assert all(case["attrs"]["is_quant"] for case in cases[20:])
    assert (
        len(
            {
                tuple(case["input_shape"][0] + case["input_shape"][1])
                for case in cases[20:]
            }
        )
        == 20
    )


def test_schema_parses_and_preserves_legacy_prefix():
    proto = yaml.safe_load((TASK / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    schema = torch._C.parse_schema(proto["schema"])
    assert [arg.name for arg in schema.arguments[:7]] == [
        "x1",
        "x2",
        "hcomm_info",
        "world_size",
        "bias",
        "is_trans_x2",
        "alltoall_out_flag",
    ]
    assert schema.arguments[-1].default_value == "bfloat16"


def test_mx_case_shapes_scales_and_coverage():
    cases = _cases()[20:]
    for case in cases:
        attrs = case["attrs"]
        m, k = case["input_shape"][0]
        n, effective_k = case["input_shape"][1]
        assert attrs["world_size"] == 2
        assert attrs["supported_devices"] == ["950pr", "950dt"]
        assert m % 2 == 0 and k % 64 == 0 and effective_k == k * 2 <= 65535
        assert attrs["is_trans_x2"]
        assert case["input_shape"][3:] == [[m, k // 64, 2], [n, k // 32, 2]]
        assert case["dtype"][2:] == ["float32", "uint8", "uint8"]
        assert all(0 <= lo < hi < 255 for lo, hi in case["value_range"][3:])
        assert (attrs["x1_quant_mode"], attrs["x2_quant_mode"]) == (6, 6)
        assert attrs["mc2_accuracy_mode"] == "three_way"
    assert {tuple(case["dtype"][:2]) for case in cases} == {
        (a, b)
        for a in ("float8_e4m3fn", "float8_e5m2")
        for b in ("float8_e4m3fn", "float8_e5m2")
    }
    for key in ("is_bias", "alltoall_out_flag"):
        assert {case["attrs"][key] for case in cases} == {False, True}
    assert {case["attrs"]["out_dtype"] for case in cases} == {"float16", "bfloat16"}


def test_csv_matches_quantized_cases_exactly():
    with (TASK / "cases.csv").open(encoding="utf-8", newline="") as stream:
        csv_cases = list(csv.DictReader(stream))
    for case, row in zip(_cases(), csv_cases):
        for key in ("input_shape", "dtype", "attrs", "value_range"):
            assert case[key] == json.loads(row[key])
        assert case["note"] == row["note"]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("output_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_cpu_and_cascade_match_independent_rank_scale_reference(
    rank, output_dtype, with_bias
):
    ranks = _rank_inputs()
    if not with_bias:
        for inputs in ranks:
            inputs["bias"] = torch.zeros(5)
    attrs = _attrs(out_dtype=output_dtype)
    expected, expected_comm = _independent_reference(ranks, rank)
    cpu, cpu_comm = GOLDEN.mc2_cpu_golden(_ctx(ranks, rank), ranks[rank], attrs)
    cascade, cascade_comm = GOLDEN.mc2_distributed_golden(
        _ctx(ranks, rank), ranks[rank], attrs
    )
    (math, math_comm), (staged, stage_comm) = GOLDEN.mc2_cpu_references(
        _ctx(ranks, rank), ranks[rank], attrs
    )
    torch.testing.assert_close(math, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        staged, expected.to(getattr(torch, output_dtype)).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(math_comm, expected_comm, rtol=0, atol=0)
    torch.testing.assert_close(stage_comm, expected_comm, rtol=0, atol=0)
    torch.testing.assert_close(cpu, expected)
    torch.testing.assert_close(cpu_comm, expected_comm)
    torch.testing.assert_close(cascade, expected.to(getattr(torch, output_dtype)))
    torch.testing.assert_close(cascade_comm.float(), expected_comm)
    assert cascade_comm.dtype == torch.float8_e4m3fn
    assert cpu.dtype == torch.float32


def test_optional_output_and_bias_none():
    ranks = _rank_inputs()
    for inputs in ranks:
        inputs["bias"] = None
    out, comm = GOLDEN.mc2_cpu_golden(
        _ctx(ranks, 0), ranks[0], _attrs(alltoall_out_flag=False)
    )
    assert out.shape == (4, 5)
    assert comm is None


def test_rank_inputs_generate_nonconstant_scale_codes():
    case = _cases()[20]
    payload = dict(
        input_shapes=case["input_shape"],
        dtypes=case["dtype"],
        value_ranges=case["value_range"],
        attrs=case["attrs"],
    )
    inputs0 = GOLDEN.mc2_make_rank_inputs(
        dict(rank=0, device="cpu", world_size=2), payload
    )
    inputs1 = GOLDEN.mc2_make_rank_inputs(
        dict(rank=1, device="cpu", world_size=2), payload
    )
    assert torch.unique(inputs0["x1_scale"]).numel() > 1
    assert torch.unique(inputs0["x2_scale"]).numel() > 1
    assert not torch.equal(inputs0["x1"].float(), inputs1["x1"].float())
    assert not torch.equal(inputs0["x1_scale"], inputs1["x1_scale"])
    assert torch.equal(inputs0["x2"].float(), inputs1["x2"].float())


def test_candidate_receives_quantized_tail_without_changing_legacy_call():
    ranks = _rank_inputs()
    recorded = {}

    def candidate(*args, **kwargs):
        recorded["args"], recorded["kwargs"] = args, kwargs
        return "candidate-result"

    ctx = _ctx(ranks, 0)
    assert (
        GOLDEN.mc2_call_candidate(candidate, ctx, ranks[0], _attrs())
        == "candidate-result"
    )
    assert recorded["kwargs"]["is_quant"] is True
    assert recorded["kwargs"]["x1_scale"] is ranks[0]["x1_scale"]
    assert recorded["kwargs"]["x2_scale"] is ranks[0]["x2_scale"]
    assert recorded["kwargs"]["scale_dtype"] == 293
    assert recorded["args"][2:] == ("test-hccl-group", 2)
    GOLDEN.mc2_call_candidate(candidate, ctx, ranks[0], dict(is_trans_x2=True))
    assert set(recorded["kwargs"]) == {"bias", "is_trans_x2", "alltoall_out_flag"}


@pytest.mark.parametrize(
    "changes",
    [
        dict(is_trans_x2=False),
        dict(x1_quant_mode=7),
        dict(group_size=0),
        dict(scale_dtype=0),
    ],
)
def test_unsupported_mx_attributes_are_rejected(changes):
    ranks = _rank_inputs()
    with pytest.raises(ValueError):
        GOLDEN.mc2_cpu_golden(_ctx(ranks, 0), ranks[0], _attrs(**changes))


def test_mismatched_weight_scale_shape_is_rejected():
    ranks = _rank_inputs()
    ranks[0]["x2_scale"] = torch.ones(5, 1, 2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="x2_scale"):
        GOLDEN.mc2_cpu_golden(_ctx(ranks, 0), ranks[0], _attrs())
