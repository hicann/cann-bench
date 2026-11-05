# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""QuantMatmulAlltoAll MXFP8 contracts and independent rank-routing regression."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "bench_lab/kernel_bench/level4/matmul_allto_all"


def golden():
    spec = importlib.util.spec_from_file_location(
        "quant_matmul_alltoall_reference", FOLDER / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    return yaml.safe_load((FOLDER / "cases.yaml").read_text(encoding="utf-8"))["cases"]


def compact(case):
    case = copy.deepcopy(case)
    m, k, n = 3, (65 if case["input_shape"][0][1] % 64 else 64), 10
    case["input_shape"] = [
        [m, k],
        [n, k],
        [n] if case["attrs"]["is_bias"] else None,
        [m, (k + 63) // 64, 2],
        [n, (k + 63) // 64, 2],
    ]
    return case


def inputs_for(module, case, rank):
    return module.mc2_make_rank_inputs(
        {"rank": rank, "world_size": 2, "device": "cpu"},
        {
            "input_shapes": case["input_shape"],
            "dtypes": case["dtype"],
            "value_ranges": case["value_range"],
            "attrs": case["attrs"],
        },
    )


def decode(values, codes):
    """Independent ldexp per 32-element block; do not call golden helpers."""
    result = torch.empty_like(values, dtype=torch.float32)
    for b in range((values.shape[-1] + 31) // 32):
        lo, hi = b * 32, min(values.shape[-1], (b + 1) * 32)
        exponent = codes[:, b // 2, b % 2].int() - 127
        result[:, lo:hi] = torch.ldexp(values[:, lo:hi].float(), exponent[:, None])
    return result


def local_product(inputs):
    result = (
        decode(inputs["x1"], inputs["x1_scale"])
        @ decode(inputs["x2"], inputs["x2_scale"]).t()
    )
    if inputs["bias"] is not None:
        result += inputs["bias"]
    return result


def exchange(products, rank, rounded):
    """Reference N-split / source-rank M-concat, including wire precision."""

    def alltoall(received, sent):
        values = [x.to(rounded) if rounded is not None else x for x in products]
        m, n = values[rank].shape
        expected_sent = torch.stack(
            [values[rank][:, : n // 2], values[rank][:, n // 2 :]]
        )
        torch.testing.assert_close(sent, expected_sent, rtol=0, atol=0)
        received.copy_(
            torch.stack([x[:, rank * (n // 2) : (rank + 1) * (n // 2)] for x in values])
        )

    return SimpleNamespace(all_to_all_single=alltoall)


def test_quant_cases_and_schema():
    module, data = golden(), cases()
    assert len(data) == 40
    with (FOLDER / "cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    proto = yaml.safe_load((FOLDER / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    assert [v["name"] for v in proto["inputs"]] == [
        "x1",
        "x2",
        "bias",
        "x1_scale",
        "x2_scale",
    ]
    declared = {v["name"] for v in proto["attrs"]}
    torch._C.parse_schema(proto["schema"])
    for case, row in zip(data, rows):
        for key, value in case.items():
            assert (
                json.loads(row[key]) if isinstance(value, (dict, list)) else str(value)
            ) == (value if isinstance(value, (dict, list)) else row[key])
        assert set(case["attrs"]) <= declared
    quant = data[20:]
    assert [v["case_id"] for v in quant] == list(range(21, 41))
    assert len({tuple(v["dtype"][:2]) for v in quant}) == 4
    assert {v["attrs"]["out_dtype"] for v in quant} == {"float16", "bfloat16"}
    assert any(v["input_shape"][0][1] % 64 for v in quant)
    assert max(v["input_shape"][0][0] for v in quant) == 8192
    for case in quant:
        attrs, shapes = case["attrs"], case["input_shape"]
        assert attrs["is_quant"] and attrs["is_trans_x2"] and attrs["world_size"] == 2
        assert attrs["supported_devices"] == ["950pr", "950dt"]
        assert attrs["mc2_accuracy_mode"] == "three_way"
        assert attrs["mc2_compare_dtype"] == "float32"
        tensors = {
            name["name"]: torch.empty(shape, dtype=module._dtype(dtype), device="meta")
            if shape is not None
            else None
            for name, shape, dtype in zip(proto["inputs"], shapes, case["dtype"])
        }
        module._validate_quant({"world_size": 2}, tensors, attrs)


@pytest.mark.parametrize("index", range(20))
def test_mx_cpu_device_cascade_and_rank_routing(index):
    module, case = golden(), compact(cases()[20 + index])
    attrs = case["attrs"]
    peers = [inputs_for(module, case, rank) for rank in range(2)]
    assert not torch.equal(peers[0]["x1"].float(), peers[1]["x1"].float())
    products = [local_product(v) for v in peers]
    dtype = module._dtype(attrs["out_dtype"])
    for rank in range(2):
        cpu_ctx = {
            "rank": rank,
            "world_size": 2,
            "dist": exchange(products, rank, None),
        }
        npu_ctx = {
            "rank": rank,
            "world_size": 2,
            "dist": exchange(products, rank, dtype),
        }
        cpu = module.mc2_cpu_golden(cpu_ctx, peers[rank], attrs)
        rounded = module.mc2_distributed_golden(npu_ctx, peers[rank], attrs)
        math, stage = module.mc2_cpu_references(cpu_ctx, peers[rank], attrs)
        want = torch.cat([x[:, rank * 5 : (rank + 1) * 5] for x in products], dim=0)
        assert cpu.dtype == torch.float32 and rounded.dtype == dtype
        torch.testing.assert_close(cpu, want, rtol=0, atol=0)
        torch.testing.assert_close(rounded, want.to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(math, want, rtol=0, atol=0)
        torch.testing.assert_close(stage, want.to(dtype).float(), rtol=0, atol=0)
        assert cpu.shape == rounded.shape == (6, 5)


def test_scale_second_lane_and_k_tail():
    module = golden()
    values = torch.ones(2, 65, dtype=torch.float8_e4m3fn)
    codes = torch.tensor(
        [[[127, 128], [126, 129]], [[129, 127], [128, 126]]], dtype=torch.uint8
    )
    result = module._mx_rows(values, codes)
    torch.testing.assert_close(result, decode(values, codes), rtol=0, atol=0)
    assert result[0, 0] == 1 and result[0, 32] == 2 and result[0, 64] == 0.5
    assert result.shape == (2, 65)


def test_quant_candidate_and_nonquant_compatibility():
    module, case = golden(), compact(cases()[21])
    attrs, calls = case["attrs"], []
    ctx = {"rank": 0, "world_size": 2, "device": "cpu", "hcomm_info": "test_group"}
    tensors = inputs_for(module, case, 0)

    def candidate(
        x1,
        x2,
        hcomm_info,
        world_size,
        bias=None,
        is_trans_x2=False,
        x1_scale=None,
        x2_scale=None,
        x1_quant_mode=0,
        x2_quant_mode=0,
        group_size=0,
        scale_dtype=293,
        out_dtype=None,
    ):
        calls.append(locals())
        return "result"

    assert module.mc2_call_candidate(candidate, ctx, tensors, attrs) == "result"
    assert calls[0]["x1_scale"] is tensors["x1_scale"]
    assert calls[0]["x2_scale"] is tensors["x2_scale"]
    for name in (
        "x1_quant_mode",
        "x2_quant_mode",
        "group_size",
        "scale_dtype",
        "out_dtype",
    ):
        assert calls[0][name] == attrs[name]
    assert calls[0]["is_trans_x2"] is True
    seen = []

    def old_candidate(x1, x2, hcomm_info, world_size, bias=None, is_trans_x2=False):
        seen.append((hcomm_info, world_size, is_trans_x2))
        return x1

    plain = {
        "x1": torch.ones(2, 4, dtype=torch.float16),
        "x2": torch.ones(4, 4, dtype=torch.float16),
        "bias": None,
    }
    assert module.mc2_call_candidate(old_candidate, ctx, plain, {}) is plain["x1"]
    assert seen == [("test_group", 2, False)]


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("is_trans_x2", False, "is_trans_x2"),
        ("x1_quant_mode", 3, "modes"),
        ("group_size", 0, "group_size"),
        ("scale_dtype", 0, "scale_dtype"),
        ("out_dtype", "float32", "out_dtype"),
    ],
)
def test_unsupported_quant_attributes_rejected(key, value, message):
    module, case = golden(), compact(cases()[20])
    tensors = inputs_for(module, case, 0)
    case["attrs"][key] = value
    with pytest.raises(ValueError, match=message):
        module._validate_quant({"world_size": 2}, tensors, case["attrs"])


def test_invalid_scale_bias_and_partition_rejected():
    module, case = golden(), compact(cases()[21])
    tensors = inputs_for(module, case, 0)
    bad = dict(tensors, x1_scale=tensors["x1_scale"][:, :, :1])
    with pytest.raises(ValueError, match="x1_scale"):
        module._validate_quant({"world_size": 2}, bad, case["attrs"])
    bad = dict(tensors, bias=tensors["bias"].half())
    with pytest.raises(ValueError, match="bias"):
        module._validate_quant({"world_size": 2}, bad, case["attrs"])
    bad = dict(
        tensors, x2=tensors["x2"][:9], x2_scale=tensors["x2_scale"][:9], bias=None
    )
    with pytest.raises(ValueError, match="divisible"):
        module._validate_quant({"world_size": 2}, bad, case["attrs"])
