# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A5 GMM hardware scope, candidate discovery and independent resource examples."""

import copy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from docs.hap_thw_model import compute_mc2_a5 as model
from kernel_eval.eval import mc2_distributed_runner as runner


def test_quantized_and_nonquantized_subsets_give_fifteen_benchmarks():
    cases = list(model.a5_cases())
    nonquant = {"float16", "bfloat16"}
    assert len({(op, case["dtype"][0] not in nonquant) for op, case in cases}) == 15
    assert len({op for op, _ in cases}) == 12
    assert len(cases) == 300
    subsets = {}
    for op, case in cases:
        subsets.setdefault((op, case["dtype"][0] not in nonquant), []).append(
            case["case_id"]
        )
    assert all(len(ids) == len(set(ids)) == 20 for ids in subsets.values())
    for op in ("allto_all_matmul", "matmul_allto_all"):
        assert subsets[(op, False)] == list(range(1, 21))
        assert subsets[(op, True)] == list(range(21, 41))
    assert subsets[("allto_allv_quant_grouped_mat_mul", True)] == list(range(1, 21))
    assert sum("950pr" in model.supported_platforms(case) for _, case in cases) == 180
    assert sum("950dt" in model.supported_platforms(case) for _, case in cases) == 300
    new_cases = [
        case
        for op, case in cases
        if op in ("grouped_mat_mul_allto_allv", "allto_allv_grouped_mat_mul")
    ]
    assert len(new_cases) == 40
    assert all(case["dtype"][0] in ("float16", "bfloat16") for case in new_cases)
    quant_cases = [case for op, case in cases if op.startswith("quant_")]
    assert len(quant_cases) == 60


@pytest.mark.parametrize(
    "op,network_bytes",
    [
        ("allto_allv_grouped_mat_mul", 128),
        ("grouped_mat_mul_allto_allv", 192),
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_gmm_resource_counts_include_shared_compute_and_remote_tokens_only(
    op, network_bytes, transpose
):
    case = {
        "input_shape": [
            [8, 16],
            [2, 24, 16] if transpose else [2, 16, 24],
            [4],
            [4],
            [4, 8],
            [12, 8] if transpose else [8, 12],
        ],
        "dtype": ["float16", "float16", "int64", "int64", "float16", "float16"],
        "attrs": {
            "world_size": 2,
            "e": 2,
            "trans_gmm_weight": transpose,
            "trans_mm_weight": transpose,
            "supported_devices": ["950dt"],
            "token_counts": [[[3, 1], [0, 4]], [[2, 2], [1, 3]]],
        },
    }
    # Eight rows * 16 * 24 per-rank GEMM, plus a 4 * 8 * 12 shared GEMM.
    assert model.resource_counts(op, case) == {
        "cube_ops": 6912,
        "read_bytes": 2048,
        "network_bytes": network_bytes,
        "cube_type": "cube16",
    }
    with pytest.raises(ValueError, match="does not support 950pr"):
        model.compute(op, case, "950pr")
    broken = copy.deepcopy(case)
    broken["attrs"]["token_counts"][0][1][1] += 1
    with pytest.raises(ValueError, match="row totals"):
        model.resource_counts(op, broken)


@pytest.mark.parametrize(
    "operator,function",
    [
        ("AlltoAllvGroupedMatMul", "allto_allv_grouped_mat_mul"),
        ("GroupedMatMulAlltoAllv", "grouped_mat_mul_allto_allv"),
        ("AlltoAllMatmul", "all_to_all_matmul"),
        ("MatmulAlltoAll", "matmul_all_to_all"),
        ("AlltoAllvQuantGroupedMatMul", "allto_allv_quant_grouped_mat_mul"),
    ],
)
def test_mc2_candidates_resolve_to_public_export(monkeypatch, operator, function):
    implementation = Mock()
    monkeypatch.setitem(
        sys.modules, "cann_bench", SimpleNamespace(**{function: implementation})
    )
    assert runner._load_candidate(operator) is implementation


@pytest.mark.parametrize(
    "op", ["allto_allv_grouped_mat_mul", "grouped_mat_mul_allto_allv"]
)
def test_empty_expert_does_not_require_weight_reads(op):
    case = {
        "input_shape": [[64, 1024], [2, 1024, 512], [4], [4], None, None],
        "dtype": ["bfloat16", "bfloat16", "int64", "int64", "bfloat16", "bfloat16"],
        "attrs": {
            "world_size": 2,
            "e": 2,
            "token_counts": [[[32, 0], [32, 0]], [[32, 0], [32, 0]]],
        },
    }
    assert model.resource_counts(op, case)["read_bytes"] == 2 * (64 * 1024 + 1024 * 512)


@pytest.mark.parametrize("device_name", ["Ascend950PR_9582", "Ascend910_9362"])
def test_dt_only_gmm_rejects_other_devices_before_collectives(monkeypatch, device_name):
    npu = SimpleNamespace(
        device_count=lambda: 2,
        get_device_name=lambda rank: device_name,
        set_device=Mock(),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=npu))
    initialize = Mock()
    monkeypatch.setattr(runner.torch.distributed, "init_process_group", initialize)
    payload = {
        "operator": "AlltoAllvGroupedMatMul",
        "attrs": {"world_size": 2, "supported_devices": ["950dt"]},
    }
    with pytest.raises(RuntimeError, match="supports.*950dt"):
        runner._run_rank(0, 2, 29500, payload, 1, 1, False, {})
    initialize.assert_not_called()
