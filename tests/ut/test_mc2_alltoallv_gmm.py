# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A5 AlltoAllv-GMM contracts and independent two-rank routing references."""

import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "bench_lab/kernel_bench/level4/allto_allv_grouped_mat_mul"


def golden():
    spec = importlib.util.spec_from_file_location(
        "alltoallv_gmm_reference", FOLDER / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    return yaml.safe_load((FOLDER / "cases.yaml").read_text(encoding="utf-8"))["cases"]


def payload(case):
    return {
        "input_shapes": case["input_shape"],
        "dtypes": case["dtype"],
        "value_ranges": case["value_range"],
        "attrs": case["attrs"],
    }


def compact(case):
    case = copy.deepcopy(case)
    attrs = case["attrs"]
    case["input_shape"][0][1] = 8
    case["input_shape"][1] = (
        [attrs["e"], 5, 8] if attrs["trans_gmm_weight"] else [attrs["e"], 8, 5]
    )
    if attrs["mm_out"]:
        case["input_shape"][4][1] = 6
        case["input_shape"][5] = [3, 6] if attrs["trans_mm_weight"] else [6, 3]
    return case


def route_expected(peers, attrs, rank):
    """Reconstruct destination/expert blocks directly from original peer inputs."""
    c, p, e = attrs["token_counts"], attrs["world_size"], attrs["e"]
    local = peers[rank]
    weight = local["gmm_weight"].float()
    if attrs["trans_gmm_weight"]:
        weight = weight.transpose(1, 2)
    products, blocks = [], []
    for expert in range(e):
        for source in range(p):
            offset = sum(sum(edge) for edge in c[source][:rank]) + sum(
                c[source][rank][:expert]
            )
            rows = c[source][rank][expert]
            block = peers[source]["gmm_x"][offset : offset + rows].float()
            blocks.append(block)
            products.append(block @ weight[expert])
    mm_y = None
    if attrs["mm_out"]:
        mm_weight = local["mm_weight"].float()
        if attrs["trans_mm_weight"]:
            mm_weight = mm_weight.t()
        mm_y = local["mm_x"].float() @ mm_weight
    return (
        torch.cat(products),
        mm_y,
        torch.cat(blocks) if attrs["permute_out_flag"] else None,
    )


def exchange_for(peers, attrs, rank):
    c = attrs["token_counts"]

    def exchange(output, input, output_split_sizes, input_split_sizes):
        assert input_split_sizes == [sum(edge) for edge in c[rank]]
        assert output_split_sizes == [sum(c[source][rank]) for source in range(2)]
        torch.testing.assert_close(input, peers[rank]["gmm_x"].to(input.dtype))
        received = []
        for source in range(2):
            start = sum(sum(edge) for edge in c[source][:rank])
            count = sum(c[source][rank])
            received.append(
                peers[source]["gmm_x"][start : start + count].to(output.dtype)
            )
        output.copy_(torch.cat(received))

    return SimpleNamespace(all_to_all_single=exchange)


def test_cases_and_csv_describe_real_a5_shapes_and_balanced_totals():
    data = cases()
    proto = yaml.safe_load((FOLDER / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    with (FOLDER / "cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(data) == len(rows) == 20
    assert [case["case_id"] for case in data] == list(range(1, 21))
    assert [entry["name"] for entry in proto["inputs"]] == [
        "gmm_x",
        "gmm_weight",
        "send_counts",
        "recv_counts",
        "mm_x",
        "mm_weight",
    ]
    declared = {attr["name"] for attr in proto["attrs"]}
    for case, row in zip(data, rows):
        for key, value in case.items():
            assert (
                json.loads(row[key]) if isinstance(value, (list, dict)) else str(value)
            ) == (value if isinstance(value, (list, dict)) else row[key])
        attrs, shapes = case["attrs"], case["input_shape"]
        assert set(attrs) <= declared
        assert attrs["world_size"] == 2 and attrs["supported_devices"] == ["950dt"]
        assert attrs["platform"] == "Ascend 950DT (A5)"
        assert attrs["mc2_distributed"] and attrs["mc2_accuracy_mode"] == "three_way"
        assert case["dtype"][:4] == [case["dtype"][0]] * 2 + ["int64"] * 2
        assert case["dtype"][0] in ("float16", "bfloat16")
        counts = attrs["token_counts"]
        for rank in range(2):
            assert sum(sum(edge) for edge in counts[rank]) == shapes[0][0]
            assert sum(sum(counts[source][rank]) for source in range(2)) == shapes[0][0]
            assert all(
                len(edge) == attrs["e"] and all(type(v) is int and v >= 0 for v in edge)
                for edge in counts[rank]
            )
        assert shapes[2] == shapes[3] == [2 * attrs["e"]]
        assert shapes[0][1] == shapes[1][2 if attrs["trans_gmm_weight"] else 1]
        if attrs["mm_out"]:
            assert shapes[0][0] // shapes[4][0] == 2
            assert shapes[4][1] == shapes[5][1 if attrs["trans_mm_weight"] else 0]
        else:
            assert shapes[4:] == [None, None]
    assert {c["attrs"]["e"] for c in data} == {1, 2, 4}
    assert {c["attrs"]["permute_out_flag"] for c in data} == {False, True}
    assert {c["attrs"]["mm_out"] for c in data} == {False, True}


@pytest.mark.parametrize("index", range(20))
def test_cpu_and_cascade_preserve_peer_expert_layout_and_rounding(index):
    module, case = golden(), compact(cases()[index])
    attrs = case["attrs"]
    contexts = [{"rank": rank, "world_size": 2, "device": "cpu"} for rank in range(2)]
    peers = [module.mc2_make_rank_inputs(ctx, payload(case)) for ctx in contexts]
    assert not torch.equal(peers[0]["gmm_x"], peers[1]["gmm_x"])
    assert not torch.equal(peers[0]["gmm_weight"], peers[1]["gmm_weight"])
    for rank, ctx in enumerate(contexts):
        ctx["dist"] = exchange_for(peers, attrs, rank)
        expected = route_expected(peers, attrs, rank)
        reference = module.mc2_cpu_golden(ctx, peers[rank], attrs)
        cascade = module.mc2_distributed_golden(ctx, peers[rank], attrs)
        for output_index, (actual, baseline, want) in enumerate(
            zip(reference, cascade, expected)
        ):
            if want is None:
                assert actual is baseline is None
            else:
                assert actual.dtype == torch.float32
                assert baseline.dtype == peers[rank]["gmm_x"].dtype
                torch.testing.assert_close(actual, want, rtol=0, atol=0)
                # Native-dtype expert-wide GEMM can round one ULP differently
                # from per-source FP32 GEMM followed by a cast. Communication
                # and permutation-only outputs must still be bitwise exact.
                rtol = torch.finfo(baseline.dtype).eps if output_index < 2 else 0
                torch.testing.assert_close(
                    baseline, want.to(baseline.dtype), rtol=rtol, atol=0
                )
        # Matrix results must retain FP32 information in the CPU branch.
        assert torch.any(reference[0] != reference[0].to(cascade[0].dtype).float())


def test_candidate_receives_host_counts_and_optional_output_flags():
    module, case = golden(), compact(cases()[4])
    ctx = {"rank": 1, "world_size": 2, "device": "cpu", "hcomm_info": "test_hccl_group"}
    inputs = module.mc2_make_rank_inputs(ctx, payload(case))
    calls = []

    def candidate(
        gmm_x,
        gmm_weight,
        hcomm_info,
        world_size,
        send_counts,
        recv_counts,
        mm_x=None,
        mm_weight=None,
        trans_gmm_weight=False,
        trans_mm_weight=False,
        permute_out_flag=False,
    ):
        calls.append(
            (
                gmm_x,
                gmm_weight,
                hcomm_info,
                world_size,
                send_counts,
                recv_counts,
                mm_x,
                mm_weight,
                trans_gmm_weight,
                trans_mm_weight,
                permute_out_flag,
            )
        )
        return "candidate-output"

    assert (
        module.mc2_call_candidate(candidate, ctx, inputs, case["attrs"])
        == "candidate-output"
    )
    assert calls[0][2:6] == (
        "test_hccl_group",
        2,
        inputs["send_counts"],
        inputs["recv_counts"],
    )
    assert calls[0][6] is inputs["mm_x"] and calls[0][7] is inputs["mm_weight"]
    assert calls[0][8:] == (
        case["attrs"]["trans_gmm_weight"],
        case["attrs"]["trans_mm_weight"],
        case["attrs"]["permute_out_flag"],
    )


def test_invalid_counts_and_partial_shared_inputs_fail_before_communication():
    module, case = golden(), compact(cases()[1])
    ctx = {"rank": 0, "world_size": 2, "device": "cpu"}
    inputs = module.mc2_make_rank_inputs(ctx, payload(case))
    broken = dict(inputs, recv_counts=[0] * len(inputs["recv_counts"]))
    with pytest.raises(ValueError, match="counts disagree"):
        module.mc2_cpu_golden(ctx, broken, case["attrs"])
    with pytest.raises(ValueError, match="both be present"):
        module.mc2_cpu_golden(ctx, dict(inputs, mm_weight=None), case["attrs"])
    attrs = copy.deepcopy(case["attrs"])
    attrs["token_counts"][0][0][0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        module.mc2_cpu_golden(ctx, inputs, attrs)


def test_public_document_has_complete_golden_and_consistent_schema():
    doc = (FOLDER / "desc.md").read_text(encoding="utf-8")
    source = (FOLDER / "golden.py").read_text(encoding="utf-8")
    proto = yaml.safe_load((FOLDER / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    blocks = re.findall(r"```python\n(.*?)\n```", doc, re.S)
    assert len(blocks) == 3
    assert re.sub(r"\s+", "", "cann_bench." + proto["schema"]) == re.sub(
        r"\s+", "", blocks[0]
    )
    source_nodes = {
        n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    }
    excerpt_nodes = {
        n.name: n for n in ast.parse(blocks[1]).body if isinstance(n, ast.FunctionDef)
    }
    assert blocks[1].strip() == source.strip()
    assert set(excerpt_nodes) == set(source_nodes)
    assert {
        "_layout",
        "_validate_inputs",
        "mc2_cpu_golden",
        "mc2_distributed_golden",
        "mc2_make_rank_inputs",
        "mc2_call_candidate",
    } <= set(excerpt_nodes)
    for name, node in excerpt_nodes.items():
        assert ast.get_source_segment(blocks[1], node) == ast.get_source_segment(
            source, source_nodes[name]
        )


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
        for index in (3, 4, 7):
            case = compact(cases()[index])
            attrs = case["attrs"]
            peers = [
                module.mc2_make_rank_inputs(
                    {"rank": s, "world_size": 2, "device": "cpu"}, payload(case)
                )
                for s in range(2)
            ]
            ctx = {
                "rank": rank,
                "world_size": 2,
                "dist": CPUCollectives(dist, dist.group.WORLD, rank, 2),
            }
            actual = module.mc2_cpu_golden(ctx, peers[rank], attrs)
            expected = route_expected(peers, attrs, rank)
            for value, want in zip(actual, expected):
                if want is None:
                    assert value is None
                else:
                    torch.testing.assert_close(value, want, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo unavailable"
)
def test_actual_gloo_alltoallv_with_zero_counts_and_distinct_experts(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=2, join=True
    )
