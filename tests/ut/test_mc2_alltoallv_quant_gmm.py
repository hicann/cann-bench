# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""AlltoAllv MXFP8 GMM metadata, independent routing and scale-layout checks."""

import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from docs.hap_thw_model import compute_mc2_a5 as model


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "bench_lab/kernel_bench/level4/allto_allv_quant_grouped_mat_mul"
NAMES = (
    "gmm_x",
    "gmm_weight",
    "send_counts",
    "recv_counts",
    "mm_x",
    "mm_weight",
    "gmm_x_scale",
    "gmm_weight_scale",
    "mm_x_scale",
    "mm_weight_scale",
)


def golden():
    spec = importlib.util.spec_from_file_location(
        "alltoallv_quant_gmm_reference", FOLDER / "golden.py"
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
    attrs, shapes = case["attrs"], case["input_shape"]
    k, n, e = (65 if shapes[0][1] % 64 else 64), 5, attrs["e"]
    q = (k + 63) // 64
    shapes[0][1] = k
    shapes[1] = [e, n, k] if attrs["trans_gmm_weight"] else [e, k, n]
    shapes[6] = [shapes[0][0], q, 2]
    shapes[7] = [e, n, q, 2] if attrs["trans_gmm_weight"] else [e, q, n, 2]
    if attrs["mm_out"]:
        shapes[4][1] = 96
        shapes[5] = [3, 96] if attrs["trans_mm_weight"] else [96, 3]
        shapes[8] = [shapes[4][0], 2, 2]
        shapes[9] = [3, 2, 2] if attrs["trans_mm_weight"] else [2, 3, 2]
    return case


def decode(values, codes):
    """Independent per-group ldexp implementation, without reference helpers."""
    result = torch.empty_like(values, dtype=torch.float32)
    for group in range((values.shape[-1] + 31) // 32):
        start, end = group * 32, min((group + 1) * 32, values.shape[-1])
        exponent = codes[..., group // 2, group % 2].to(torch.int32) - 127
        result[..., start:end] = torch.ldexp(
            values[..., start:end].float(), exponent.unsqueeze(-1)
        )
    return result


def weight_reference(value, codes, transposed):
    if not transposed:
        value, codes = value.transpose(-2, -1), codes.transpose(-3, -2)
    return decode(value, codes).transpose(-2, -1)


def expected(peers, attrs, rank):
    counts, e, local = attrs["token_counts"], attrs["e"], peers[rank]
    weight = weight_reference(
        local["gmm_weight"], local["gmm_weight_scale"], attrs["trans_gmm_weight"]
    )
    products, raw = [], []
    for expert in range(e):
        for source in range(2):
            begin = sum(sum(edge) for edge in counts[source][:rank]) + sum(
                counts[source][rank][:expert]
            )
            end = begin + counts[source][rank][expert]
            block = peers[source]["gmm_x"][begin:end]
            scales = peers[source]["gmm_x_scale"][begin:end]
            products.append(decode(block, scales) @ weight[expert])
            raw.append(block.float())
    mm = None
    if attrs["mm_out"]:
        mm = decode(local["mm_x"], local["mm_x_scale"]) @ weight_reference(
            local["mm_weight"], local["mm_weight_scale"], attrs["trans_mm_weight"]
        )
    return (
        torch.cat(products),
        mm,
        torch.cat(raw) if attrs["permute_out_flag"] else None,
    )


def exchange_for(peers, attrs, rank):
    calls, counts = [], attrs["token_counts"]

    def exchange(output, input, output_split_sizes, input_split_sizes):
        name = ("gmm_x", "gmm_x_scale")[len(calls) % 2]
        calls.append(name)
        assert input_split_sizes == [sum(edge) for edge in counts[rank]]
        assert output_split_sizes == [sum(counts[source][rank]) for source in range(2)]
        torch.testing.assert_close(input, peers[rank][name].float(), rtol=0, atol=0)
        blocks = []
        for source in range(2):
            start = sum(sum(edge) for edge in counts[source][:rank])
            end = start + sum(counts[source][rank])
            blocks.append(peers[source][name][start:end].float())
        output.copy_(torch.cat(blocks))

    return SimpleNamespace(all_to_all_single=exchange)


def test_metadata_csv_proto_and_published_shapes():
    module, data = golden(), cases()
    proto = yaml.safe_load((FOLDER / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    with (FOLDER / "cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(data) == len(rows) == 20
    assert [case["case_id"] for case in data] == list(range(1, 21))
    assert tuple(v["name"] for v in proto["inputs"]) == NAMES
    assert all(
        "required" in v for v in proto["attrs"] + proto["inputs"] + proto["outputs"]
    )
    declared = {a["name"] for a in proto["attrs"]}
    for case, row in zip(data, rows):
        for key, value in case.items():
            assert (
                json.loads(row[key]) if isinstance(value, (dict, list)) else str(value)
            ) == (value if isinstance(value, (dict, list)) else row[key])
        attrs, shapes = case["attrs"], case["input_shape"]
        assert set(attrs) <= declared
        assert attrs["world_size"] == 2 and attrs["supported_devices"] == ["950dt"]
        assert attrs["mc2_accuracy_mode"] == "three_way" and attrs["mc2_distributed"]
        assert "baseline_perf_us" not in case and "t_hw_us" not in case
        assert shapes[2] == shapes[3] == [2 * attrs["e"]]
        for rank in range(2):
            ctx = {"rank": rank, "world_size": 2}
            _, _, _, _, send, recv = module._layout(ctx, attrs)
            inputs = {
                name: torch.empty(shape, device="meta", dtype=module._dtype(dtype))
                if shape is not None
                else None
                for name, shape, dtype in zip(NAMES, shapes, case["dtype"])
                if name not in NAMES[2:4]
            }
            inputs.update(send_counts=send, recv_counts=recv)
            module._validate_inputs(ctx, inputs, attrs)
            assert sum(send) == sum(recv) == shapes[0][0]
    assert {c["attrs"]["out_dtype"] for c in data} == {"float16", "bfloat16"}
    assert len({tuple(c["dtype"][:2]) for c in data}) == 4
    assert {c["attrs"]["e"] for c in data} == {1, 2, 4, 8, 16}
    assert {(c["attrs"]["mm_out"], c["attrs"]["permute_out_flag"]) for c in data} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }
    assert max(c["input_shape"][0][0] for c in data) == 8192
    assert any(c["input_shape"][0][1] % 64 for c in data)


@pytest.mark.parametrize("index", range(20))
def test_cpu_and_device_cascade_route_raw_fp8_and_scales_before_gmm(index):
    module, case = golden(), compact(cases()[index])
    attrs = case["attrs"]
    contexts = [{"rank": r, "world_size": 2, "device": "cpu"} for r in range(2)]
    peers = [module.mc2_make_rank_inputs(ctx, payload(case)) for ctx in contexts]
    assert not torch.equal(peers[0]["gmm_x"].float(), peers[1]["gmm_x"].float())
    for rank, ctx in enumerate(contexts):
        ctx["dist"] = exchange_for(peers, attrs, rank)
        cpu = module.mc2_cpu_golden(ctx, peers[rank], attrs)
        device = module.mc2_distributed_golden(ctx, peers[rank], attrs)
        math, staged = module.mc2_cpu_references(ctx, peers[rank], attrs)
        want = expected(peers, attrs, rank)
        for i, (actual, rounded, reference) in enumerate(zip(cpu, device, want)):
            if reference is None:
                assert actual is rounded is None
                assert math[i] is staged[i] is None
                continue
            assert actual.dtype == torch.float32
            dtype = (
                peers[rank]["gmm_x"].dtype
                if i == 2
                else module._dtype(attrs["out_dtype"])
            )
            assert rounded.dtype == dtype
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(math[i], reference, rtol=0, atol=0)
            stage_expected = reference if i == 2 else reference.to(dtype).float()
            torch.testing.assert_close(staged[i], stage_expected, rtol=0, atol=0)
            if i == 2:
                torch.testing.assert_close(rounded.float(), reference, rtol=0, atol=0)
            else:
                # Matmul grouping can change FP32 summation at a rounding tie.
                eps = torch.finfo(dtype).eps
                torch.testing.assert_close(
                    rounded.float(), reference, rtol=eps, atol=eps
                )


def test_candidate_contract_and_rejected_shared_transpose():
    module, case = golden(), compact(cases()[1])
    ctx = {"rank": 1, "world_size": 2, "device": "cpu", "hcomm_info": "test_group"}
    inputs = module.mc2_make_rank_inputs(ctx, payload(case))
    calls = []

    def candidate(
        gmm_x,
        gmm_weight,
        gmm_x_scale,
        gmm_weight_scale,
        hcomm_info,
        world_size,
        send_counts,
        recv_counts,
        mm_x=None,
        mm_weight=None,
        mm_x_scale=None,
        mm_weight_scale=None,
        trans_gmm_weight=False,
        trans_mm_weight=False,
        out_dtype="bfloat16",
        gmm_x_quant_mode=6,
        gmm_weight_quant_mode=6,
        mm_x_quant_mode=0,
        mm_weight_quant_mode=0,
        group_size=4295032864,
        scale_dtype=293,
        permute_out_flag=False,
    ):
        calls.append(locals())
        return "output"

    assert module.mc2_call_candidate(candidate, ctx, inputs, case["attrs"]) == "output"
    assert calls[0]["hcomm_info"] == "test_group" and calls[0]["world_size"] == 2
    assert calls[0]["gmm_x_scale"] is inputs["gmm_x_scale"]
    assert calls[0]["send_counts"] == inputs["send_counts"]
    for name in ("out_dtype", "group_size", "scale_dtype", "permute_out_flag"):
        assert calls[0][name] == case["attrs"][name]
    with pytest.raises(ValueError, match="Shared matrices and scales"):
        module._validate_inputs(ctx, dict(inputs, mm_x_scale=None), case["attrs"])
    case = compact(cases()[0])
    inputs = module.mc2_make_rank_inputs(ctx, payload(case))
    with pytest.raises(ValueError, match="trans_mm_weight must be false"):
        module._validate_inputs(ctx, inputs, dict(case["attrs"], trans_mm_weight=True))
    with pytest.raises(ValueError, match="Host counts"):
        module._validate_inputs(ctx, dict(inputs, send_counts=[0, 0]), case["attrs"])


def test_document_matches_schema_and_complete_golden():
    doc = (FOLDER / "desc.md").read_text(encoding="utf-8")
    source = (FOLDER / "golden.py").read_text(encoding="utf-8")
    schema = yaml.safe_load((FOLDER / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]["schema"]
    imports = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(imports) == 1 and isinstance(imports[0], ast.Import)
    assert [name.name for name in imports[0].names] == ["torch"]
    for parameter in schema.split("(", 1)[1].split(")", 1)[0].split(","):
        name = parameter.strip().split("=", 1)[0].split()[-1]
        assert f"| `{name}` |" in doc
    blocks = re.findall(r"```python\n(.*?)\n```", doc, re.S)
    assert len(blocks) == 3
    assert re.sub(r"\s+", "", "cann_bench." + schema) == re.sub(r"\s+", "", blocks[0])
    original = {
        n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    }
    included = {
        n.name: n for n in ast.parse(blocks[1]).body if isinstance(n, ast.FunctionDef)
    }
    assert blocks[1].strip() == source.strip()
    assert set(included) == set(original)
    assert {
        "_layout",
        "_check_mx_rows",
        "_validate_inputs",
        "_mx_rows",
        "_mx_weight",
        "_exchange",
        "_shared",
        "mc2_cpu_golden",
        "mc2_distributed_golden",
        "mc2_make_rank_inputs",
        "mc2_call_candidate",
    } <= set(included)
    for name, node in included.items():
        assert ast.get_source_segment(blocks[1], node) == ast.get_source_segment(
            source, original[name]
        )
    namespace = {}
    exec(compile(blocks[1], "desc_complete_golden", "exec"), namespace)
    assert callable(namespace["mc2_cpu_golden"])


def _gloo_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from kernel_eval.eval.mc2_cpu_collectives import CPUCollectives

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        module = golden()
        for index in (0, 4, 11):
            case = compact(cases()[index])
            attrs = case["attrs"]
            peers = [
                module.mc2_make_rank_inputs(
                    {"rank": r, "world_size": 2, "device": "cpu"}, payload(case)
                )
                for r in range(2)
            ]
            ctx = {
                "rank": rank,
                "world_size": 2,
                "dist": CPUCollectives(dist, dist.group.WORLD, rank, 2),
            }
            for actual, want in zip(
                module.mc2_cpu_golden(ctx, peers[rank], attrs),
                expected(peers, attrs, rank),
            ):
                if want is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(actual, want, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo unavailable"
)
def test_gloo_routes_values_and_scales_with_empty_experts(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=2, join=True
    )


@pytest.mark.parametrize("k", [64, 65])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("empty_expert", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_resource_model_counts_mx_tail_scales_received_experts_and_shared(
    k, transpose, empty_expert, shared
):
    q = (k + 63) // 64
    # Each source uses only one expert, but each destination receives both:
    # counting source-active rather than destination-active weights is wrong.
    counts = [[[4, 0], [4, 0]], [[0, 4], [0, 4]]]
    if empty_expert:
        counts = [[[4, 0], [4, 0]], [[4, 0], [4, 0]]]
    case = {
        "input_shape": [
            [8, k],
            [2, 96, k] if transpose else [2, k, 96],
            [4],
            [4],
            [4, 128] if shared else None,
            ([64, 128] if transpose else [128, 64]) if shared else None,
            [8, q, 2],
            [2, 96, q, 2] if transpose else [2, q, 96, 2],
            [4, 2, 2] if shared else None,
            ([64, 2, 2] if transpose else [2, 64, 2]) if shared else None,
        ],
        "dtype": [
            "float8_e4m3fn",
            "float8_e5m2",
            "int64",
            "int64",
            "float8_e4m3fn",
            "float8_e5m2",
            "uint8",
            "uint8",
            "uint8",
            "uint8",
        ],
        "attrs": {
            "world_size": 2,
            "e": 2,
            "trans_gmm_weight": transpose,
            "trans_mm_weight": transpose if shared else False,
            "out_dtype": "float16",
            "supported_devices": ["950dt"],
            "token_counts": counts,
        },
    }
    # K=64: x 512 + scale 16 + each active weight (6144+192).
    # K=65: x 520 + scale 32 + each active weight (6240+384).
    # Shared: 512+8192+16+256=8976 bytes, 2*4*128*64=65536 ops.
    reads = {(64, False): 13200, (64, True): 6864, (65, False): 13800, (65, True): 7176}
    result = model.resource_counts("allto_allv_quant_grouped_mat_mul", case)
    assert result == {
        "cube_ops": (98304 if k == 64 else 99840) + (65536 if shared else 0),
        "cube_type": "cube8",
        "read_bytes": reads[k, empty_expert] + (8976 if shared else 0),
        # Four remote rows send raw FP8 plus 2 / 4 E8M0 codes, not N*2 output bytes.
        "network_bytes": 264 if k == 64 else 276,
    }
    assert result["network_bytes"] != 4 * 96 * 2
    with pytest.raises(ValueError, match="does not support 950pr"):
        model.compute("allto_allv_quant_grouped_mat_mul", case, "950pr")


def test_candidate_name_mapping_selects_alltoallv_first_quant_entry(monkeypatch):
    from kernel_eval.eval.mc2_distributed_runner import _load_candidate

    def candidate(*args, **kwargs):
        return args, kwargs

    monkeypatch.setitem(
        sys.modules,
        "cann_bench",
        SimpleNamespace(allto_allv_quant_grouped_mat_mul=candidate),
    )
    assert _load_candidate("AlltoAllvQuantGroupedMatMul") is candidate
