# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""INT8/FP32 quant collective contracts and independent multi-rank references."""

import ast
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
LEVEL4 = ROOT / "bench_lab/kernel_bench/level4"
OPS = ("quant_all_reduce", "quant_reduce_scatter")


def golden(op):
    spec = importlib.util.spec_from_file_location(
        op + "_test_reference", LEVEL4 / op / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(op):
    return yaml.safe_load((LEVEL4 / op / "cases.yaml").read_text(encoding="utf-8"))[
        "cases"
    ]


def payload(case):
    return {
        "input_shapes": case["input_shape"],
        "dtypes": case["dtype"],
        "value_ranges": case["value_range"],
        "attrs": case["attrs"],
    }


def compact(case):
    case = copy.deepcopy(case)
    shape = [4, 1024] if len(case["input_shape"][0]) == 2 else [2, 3, 1152]
    case["input_shape"] = [shape, shape[:-1] + [shape[-1] // 128]]
    return case


def dequantize(inputs):
    # Channel-to-scale indexing is independent of repeat_interleave and group reshaping.
    channel_group = torch.arange(inputs["x"].shape[-1]) // 128
    scales = inputs["scales"].index_select(-1, channel_group)
    return inputs["x"].float() * scales


@pytest.mark.parametrize("op", OPS)
def test_specs_csv_and_source_constraints(op):
    data = cases(op)
    proto = yaml.safe_load((LEVEL4 / op / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    with (LEVEL4 / op / "cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(data) == len(rows) == 20
    assert [c["case_id"] for c in data] == list(range(1, 21))
    declared = {attr["name"] for attr in proto["attrs"]}
    for case, row in zip(data, rows):
        for key, value in case.items():
            assert (
                json.loads(row[key]) if isinstance(value, (dict, list)) else row[key]
            ) == (value if isinstance(value, (dict, list)) else str(value))
        attrs, shapes = case["attrs"], case["input_shape"]
        assert set(attrs) <= declared
        assert attrs["world_size"] == 2 and attrs["supported_devices"] == ["950dt"]
        assert attrs["platform"] == "Ascend 950DT (A5)"
        assert attrs["mc2_distributed"] and attrs["mc2_accuracy_mode"] == "three_way"
        assert attrs["mc2_isolated_candidate_group"] is True
        assert attrs["x_dtype"] == "int8" and attrs["scales_dtype"] == "float32"
        assert attrs["group_size"] == 128 and attrs["reduce_op"] == "sum"
        assert case["dtype"] == ["int8", "float32"]
        assert -128 <= case["value_range"][0][0] <= case["value_range"][0][1] <= 127
        shape, scale_shape = shapes
        assert len(shape) in (2, 3) and all(value > 0 for value in shape)
        assert math.prod(shape[:-1]) % 2 == 0
        assert 1024 <= shape[-1] <= 8192 and shape[-1] % 128 == 0
        assert scale_shape == shape[:-1] + [shape[-1] // 128]
    assert {c["attrs"]["out_dtype"] for c in data} == {"float16", "bfloat16", "float32"}
    assert {len(c["input_shape"][0]) for c in data} == {2, 3}
    assert {c["input_shape"][0][-1] for c in data} >= {1024, 1152, 8192}


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("index", range(20))
def test_cpu_fp32_group_scales_peer_sum_and_single_final_cast(op, index):
    module, case = golden(op), compact(cases(op)[index])
    attrs = case["attrs"]
    peers = [
        module.mc2_make_rank_inputs(
            {"rank": rank, "world_size": 2, "device": "cpu"}, payload(case)
        )
        for rank in range(2)
    ]
    repeated = module.mc2_make_rank_inputs(
        {"rank": 1, "world_size": 2, "device": "cpu"}, payload(case)
    )
    assert torch.equal(repeated["x"], peers[1]["x"])
    assert torch.equal(repeated["scales"], peers[1]["scales"])
    assert not torch.equal(peers[0]["x"], peers[1]["x"])
    if index != 1:  # Deliberate zero scales exercise exact zero output.
        assert not torch.equal(peers[0]["scales"], peers[1]["scales"])
    local_values = [dequantize(peer) for peer in peers]
    total = local_values[0] + local_values[1]
    for rank in range(2):
        calls = []

        def all_reduce(tensor, op):
            assert tensor.dtype == torch.float32 and op == "sum"
            torch.testing.assert_close(
                tensor, local_values[rank].reshape(tensor.shape), rtol=0, atol=0
            )
            tensor.add_(local_values[1 - rank].reshape(tensor.shape))
            calls.append("all_reduce")

        def reduce_scatter(output, tensor, op):
            assert tensor.dtype == output.dtype == torch.float32 and op == "sum"
            torch.testing.assert_close(
                tensor, local_values[rank].reshape(tensor.shape), rtol=0, atol=0
            )
            output.copy_(
                (tensor + local_values[1 - rank].reshape(tensor.shape)).chunk(2, dim=0)[
                    rank
                ]
            )
            calls.append("reduce_scatter")

        dist = SimpleNamespace(
            all_reduce=all_reduce,
            _reduce_scatter_base=reduce_scatter,
            ReduceOp=SimpleNamespace(SUM="sum"),
        )
        ctx = {"rank": rank, "world_size": 2, "dist": dist}
        reference = module.mc2_cpu_golden(ctx, peers[rank], attrs)
        cascade = module.mc2_distributed_golden(ctx, peers[rank], attrs)
        mathematical, staged = module.mc2_cpu_references(ctx, peers[rank], attrs)
        expected = (
            total.reshape(-1, total.shape[-1]).chunk(2, dim=0)[rank]
            if op == "quant_reduce_scatter"
            else total
        )
        out_dtype = getattr(torch, attrs["out_dtype"])
        assert reference.dtype == torch.float32 and cascade.dtype == out_dtype
        assert reference.shape == cascade.shape == expected.shape
        torch.testing.assert_close(reference, expected, rtol=0, atol=0)
        torch.testing.assert_close(cascade, expected.to(out_dtype), rtol=0, atol=0)
        torch.testing.assert_close(mathematical, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            staged, expected.to(out_dtype).float(), rtol=0, atol=0
        )
        assert calls == [
            "all_reduce",
            "reduce_scatter" if op == "quant_reduce_scatter" else "all_reduce",
            "all_reduce",
        ]
        if index != 1 and out_dtype != torch.float32:
            assert torch.any(reference != reference.to(out_dtype).float())


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("endpoint", [-128, 127])
def test_integer_value_range_includes_both_endpoints(op, endpoint):
    case = compact(cases(op)[0])
    case["value_range"][0] = [endpoint, endpoint]
    inputs = golden(op).mc2_make_rank_inputs(
        {"rank": 0, "world_size": 2, "device": "cpu"}, payload(case)
    )
    assert torch.all(inputs["x"] == endpoint)


@pytest.mark.parametrize("op", OPS)
def test_candidate_binding_and_invalid_group_shape_are_explicit(op):
    module, case = golden(op), compact(cases(op)[0])
    ctx = {
        "rank": 0,
        "world_size": 2,
        "device": "cpu",
        "hcomm_info": "dedicated-quant-group",
    }
    inputs = module.mc2_make_rank_inputs(ctx, payload(case))
    seen = []

    def candidate(x, scales, hcomm_info, world_size, out_dtype="float16"):
        seen.append((x, scales, hcomm_info, world_size, out_dtype))
        return "candidate-result"

    assert (
        module.mc2_call_candidate(candidate, ctx, inputs, case["attrs"])
        == "candidate-result"
    )
    assert seen[0][0] is inputs["x"] and seen[0][1] is inputs["scales"]
    assert seen[0][2:] == ("dedicated-quant-group", 2, "float16")
    with pytest.raises(ValueError, match="128 channels"):
        module.mc2_cpu_golden(
            ctx, dict(inputs, scales=inputs["scales"][:, :1]), case["attrs"]
        )
    with pytest.raises(ValueError, match="divisible"):
        bad = {"x": inputs["x"][:3], "scales": inputs["scales"][:3]}
        module.mc2_cpu_golden(ctx, bad, case["attrs"])
    with pytest.raises(ValueError, match="INT8"):
        module.mc2_cpu_golden(ctx, dict(inputs, x=inputs["x"].float()), case["attrs"])
    with pytest.raises(ValueError, match="group_size"):
        module.mc2_cpu_golden(ctx, inputs, dict(case["attrs"], group_size=64))


@pytest.mark.parametrize("op", OPS)
def test_document_exact_reference_and_public_signature(op):
    folder = LEVEL4 / op
    doc = (folder / "desc.md").read_text(encoding="utf-8")
    source = (folder / "golden.py").read_text(encoding="utf-8")
    proto = yaml.safe_load((folder / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    blocks = re.findall(r"```python\n(.*?)\n```", doc, re.S)
    assert len(blocks) == 3
    assert re.sub(r"\s+", "", "cann_bench." + proto["schema"]) == re.sub(
        r"\s+", "", blocks[0]
    )
    nodes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    copied = {
        node.name: node
        for node in ast.parse(blocks[1]).body
        if isinstance(node, ast.FunctionDef)
    }
    assert blocks[1].strip() == source.strip()
    assert set(copied) == set(nodes)
    assert {
        "_dtype",
        "_validate",
        "mc2_cpu_golden",
        "mc2_distributed_golden",
        "mc2_make_rank_inputs",
        "mc2_call_candidate",
    } <= set(copied)
    for name, node in copied.items():
        assert ast.get_source_segment(blocks[1], node) == ast.get_source_segment(
            source, nodes[name]
        )
    assert "同一通信域只允许" in doc


def _gloo_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from kernel_eval.eval.mc2_cpu_collectives import CPUCollectives

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        for op in OPS:
            module = golden(op)
            for index in (0, 13, 19):
                case = compact(cases(op)[index])
                peers = [
                    module.mc2_make_rank_inputs(
                        {"rank": s, "world_size": 2, "device": "cpu"}, payload(case)
                    )
                    for s in range(2)
                ]
                cpu_ctx = {
                    "rank": rank,
                    "world_size": 2,
                    "dist": CPUCollectives(dist, dist.group.WORLD, rank, 2),
                }
                reference = module.mc2_cpu_golden(cpu_ctx, peers[rank], case["attrs"])
                cascade = module.mc2_distributed_golden(
                    {"rank": rank, "world_size": 2, "dist": dist},
                    peers[rank],
                    case["attrs"],
                )
                total = dequantize(peers[0]) + dequantize(peers[1])
                expected = (
                    total.reshape(-1, total.shape[-1]).chunk(2, dim=0)[rank]
                    if op == "quant_reduce_scatter"
                    else total
                )
                torch.testing.assert_close(reference, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    cascade,
                    expected.to(getattr(torch, case["attrs"]["out_dtype"])),
                    rtol=0,
                    atol=0,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo unavailable"
)
def test_actual_gloo_allreduce_and_reduce_scatter_with_distinct_rank_scales(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=2, join=True
    )
