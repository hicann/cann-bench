# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MXFP8 GMM/AlltoAllv contracts, independent scale indexing and real Gloo."""

import ast
import csv
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
FOLDER = (
    Path(__file__).resolve().parents[2]
    / "bench_lab/kernel_bench/level4/quant_grouped_mat_mul_allto_allv"
)


def cases():
    return yaml.safe_load((FOLDER / "cases.yaml").read_text(encoding="utf-8"))["cases"]


def golden():
    spec = importlib.util.spec_from_file_location(
        "quant_gmm_test_golden", FOLDER / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(case):
    return {
        "input_shapes": case["input_shape"],
        "dtypes": case["dtype"],
        "value_ranges": case["value_range"],
        "attrs": case["attrs"],
    }


def decode_x(x, codes):
    # Explicit 32-element group slices avoid the golden's flatten/repeat implementation.
    result = torch.empty_like(x, dtype=torch.float32)
    for k in range(0, x.shape[-1], 32):
        exponent = codes[..., k // 64, (k % 64) // 32].int() - 127
        result[..., k : k + 32] = torch.ldexp(
            x[..., k : k + 32].float(), exponent.unsqueeze(-1)
        )
    return result


def decode_weight(weight, codes, transposed):
    effective = weight.transpose(-2, -1) if transposed else weight
    result = torch.empty_like(effective, dtype=torch.float32)
    for k in range(0, effective.shape[-2], 32):
        if transposed:
            exponent = codes[..., :, k // 64, (k % 64) // 32].int() - 127
        else:
            exponent = codes[..., k // 64, :, (k % 64) // 32].int() - 127
        result[..., k : k + 32, :] = torch.ldexp(
            effective[..., k : k + 32, :].float(), exponent.unsqueeze(-2)
        )
    return result


def oracle_messages(peers, attrs, rounded=False):
    counts, e = attrs["token_counts"], attrs["e"]
    messages, shared = [], []
    dtype = getattr(torch, attrs["out_dtype"]) if rounded else torch.float32
    for source, inputs in enumerate(peers):
        x = decode_x(inputs["gmm_x"], inputs["gmm_x_scale"])
        weight = decode_weight(
            inputs["gmm_weight"], inputs["gmm_weight_scale"], attrs["trans_gmm_weight"]
        )
        blocks, offset = [[], []], 0
        for expert in range(e):
            for destination in range(2):
                rows = counts[source][destination][expert]
                blocks[destination].append(
                    (x[offset : offset + rows] @ weight[expert]).to(dtype)
                )
                offset += rows
        messages.append([torch.cat(items, dim=0) for items in blocks])
        if attrs["mm_out"]:
            a = decode_x(inputs["mm_x"], inputs["mm_x_scale"])
            b = decode_weight(
                inputs["mm_weight"], inputs["mm_weight_scale"], attrs["trans_mm_weight"]
            )
            shared.append((a @ b).to(dtype))
        else:
            shared.append(None)
    return messages, shared


def context(rank, messages):
    def exchange(output, input, output_split_sizes, input_split_sizes):
        assert input_split_sizes == [len(block) for block in messages[rank]]
        assert output_split_sizes == [
            len(messages[source][rank]) for source in range(2)
        ]
        tolerance = (
            1e-5 if input.dtype == torch.float32 else 2 * torch.finfo(input.dtype).eps
        )
        torch.testing.assert_close(
            input, torch.cat(messages[rank], dim=0), rtol=tolerance, atol=1e-4
        )
        output.copy_(torch.cat([messages[source][rank] for source in range(2)], dim=0))

    return {
        "rank": rank,
        "world_size": 2,
        "dist": SimpleNamespace(all_to_all_single=exchange),
    }


def test_twenty_cases_are_native_mx_fp8_with_correct_scale_layouts():
    data = cases()
    assert len(data) == 20 and [c["case_id"] for c in data] == list(range(1, 21))
    for case in data:
        a, s = case["attrs"], case["input_shape"]
        rows, k = s[0]
        n = s[1][1 if a["trans_gmm_weight"] else 2]
        q = (k + 63) // 64
        assert a["supported_devices"] == ["950dt"] and a["world_size"] == 2
        assert a["gmm_x_quant_mode"] == a["gmm_weight_quant_mode"] == 6
        assert (
            a["comm_quant_mode"] == 0
            and a["group_size"] == 4295032864
            and a["scale_dtype"] == 293
        )
        assert s[6] == [rows, q, 2]
        assert s[7] == (
            [a["e"], n, q, 2] if a["trans_gmm_weight"] else [a["e"], q, n, 2]
        )
        for rank in range(2):
            assert sum(sum(edge) for edge in a["token_counts"][rank]) == rows
            assert (
                sum(sum(a["token_counts"][source][rank]) for source in range(2)) == rows
            )
        if a["mm_out"]:
            bs, h2 = s[4]
            n2 = s[5][0 if a["trans_mm_weight"] else 1]
            assert rows % bs == 0 and 2 <= rows // bs <= 8
            assert s[8] == [bs, (h2 + 63) // 64, 2]
            assert s[9] == (
                [n2, (h2 + 63) // 64, 2]
                if a["trans_mm_weight"]
                else [(h2 + 63) // 64, n2, 2]
            )
            assert a["mm_x_quant_mode"] == a["mm_weight_quant_mode"] == 6
        else:
            assert all(s[i] is None for i in (4, 5, 8, 9))
    assert {c["attrs"]["out_dtype"] for c in data} == {"float16", "bfloat16"}
    assert len({tuple(c["dtype"][:2]) for c in data}) == 4
    assert {c["attrs"]["e"] for c in data} == {1, 2, 4}


def test_csv_proto_and_document_match():
    proto = yaml.safe_load((FOLDER / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    with (FOLDER / "cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(cases())
    declared = {a["name"] for a in proto["attrs"]}
    for case, row in zip(cases(), rows):
        assert set(case["attrs"]) <= declared
        for name in ("input_shape", "dtype", "attrs", "value_range"):
            assert json.loads(row[name]) == case[name]
    torch._C.parse_schema(proto["schema"])
    doc = (FOLDER / "desc.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", doc, re.S)
    assert len(blocks) == 3
    assert re.sub(r"\s+", "", blocks[0]) == re.sub(
        r"\s+", "", "cann_bench." + proto["schema"]
    )
    source = (FOLDER / "golden.py").read_text(encoding="utf-8")
    nodes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    for node in ast.parse(blocks[1]).body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == ast.dump(nodes[node.name])


@pytest.mark.parametrize("index", range(20))
def test_rank_routing_scale_transpose_and_output_rounding(index):
    module, case = golden(), cases()[index]
    attrs = case["attrs"]
    peers = [
        module.mc2_make_rank_inputs(
            {"rank": rank, "world_size": 2, "device": "cpu"}, payload(case)
        )
        for rank in range(2)
    ]
    assert not torch.equal(peers[0]["gmm_x"].float(), peers[1]["gmm_x"].float())
    assert not torch.equal(peers[0]["gmm_x_scale"], peers[1]["gmm_x_scale"])
    for rounded in (False, True):
        messages, shared = oracle_messages(peers, attrs, rounded)
        hook = module.mc2_distributed_golden if rounded else module.mc2_cpu_golden
        for rank in range(2):
            y, mm_y = hook(context(rank, messages), peers[rank], attrs)
            assert y.dtype == (
                getattr(torch, attrs["out_dtype"]) if rounded else torch.float32
            )
            torch.testing.assert_close(
                y, torch.cat([messages[source][rank] for source in range(2)], dim=0)
            )
            if shared[rank] is None:
                assert mm_y is None
            else:
                torch.testing.assert_close(mm_y, shared[rank])


def test_scale_codes_mean_powers_of_two_and_do_not_move_with_output():
    module = golden()
    values = torch.ones((1, 64), dtype=torch.float32).to(torch.float8_e4m3fn)
    decoded = module._mx_rows(values, torch.tensor([[[126, 128]]], dtype=torch.uint8))
    torch.testing.assert_close(decoded, torch.tensor([[0.5] * 32 + [2.0] * 32]))
    weight = torch.ones((1, 64, 2), dtype=torch.float32).to(torch.float8_e5m2)
    scale = torch.tensor([[[[126, 127], [128, 129]]]], dtype=torch.uint8)
    expected = torch.tensor([[[0.5, 2.0]] * 32 + [[1.0, 4.0]] * 32])
    torch.testing.assert_close(module._mx_weight(weight, scale, False), expected)
    torch.testing.assert_close(
        module._mx_weight(
            weight.transpose(1, 2).contiguous(),
            scale.transpose(1, 2).contiguous(),
            True,
        ),
        expected,
    )


def test_candidate_receives_native_modes_scales_and_hcomm_order():
    module, case = golden(), cases()[1]
    inputs = module.mc2_make_rank_inputs(
        {"rank": 0, "world_size": 2, "device": "cpu"}, payload(case)
    )
    seen = {}

    def candidate(x, w, xs, ws, hcomm, world, send, recv, **kwargs):
        seen.update(kwargs)
        assert x is inputs["gmm_x"] and w is inputs["gmm_weight"]
        assert xs is inputs["gmm_x_scale"] and ws is inputs["gmm_weight_scale"]
        assert hcomm == "quant_group" and world == 2
        assert send == inputs["send_counts"] and recv == inputs["recv_counts"]
        return "sentinel"

    assert (
        module.mc2_call_candidate(
            candidate,
            {"hcomm_info": "quant_group", "world_size": 2},
            inputs,
            case["attrs"],
        )
        == "sentinel"
    )
    assert seen["gmm_x_quant_mode"] == seen["mm_x_quant_mode"] == 6
    assert seen["scale_dtype"] == 293 and seen["group_size"] == 4295032864
    assert seen["mm_weight_scale"] is inputs["mm_weight_scale"]


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
        module = golden()
        for index in (1, 2, 4):
            case = cases()[index]
            attrs = case["attrs"]
            peers = [
                module.mc2_make_rank_inputs(
                    {"rank": source, "world_size": 2, "device": "cpu"}, payload(case)
                )
                for source in range(2)
            ]
            ctx = {
                "rank": rank,
                "world_size": 2,
                "dist": CPUCollectives(dist, dist.group.WORLD, rank, 2),
            }
            y, mm_y = module.mc2_cpu_golden(ctx, peers[rank], attrs)
            messages, shared = oracle_messages(peers, attrs)
            torch.testing.assert_close(
                y,
                torch.cat([messages[source][rank] for source in range(2)], dim=0),
                rtol=1e-5,
                atol=1e-4,
            )
            if shared[rank] is None:
                assert mm_y is None
            else:
                torch.testing.assert_close(mm_y, shared[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo unavailable"
)
def test_real_two_rank_gloo_routes_quantized_matmul_outputs(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=((tmp_path / "gloo-quant-gmm").as_uri(),),
        nprocs=2,
        join=True,
    )
