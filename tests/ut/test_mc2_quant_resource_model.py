# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Independent byte/FLOP examples for the three explicitly named Quant tasks."""

import pytest

from docs.hap_thw_model import compute_mc2_a5 as model


@pytest.mark.parametrize(
    "op,weight_k,read_bytes,network_bytes",
    [
        ("allto_all_matmul", 128, 13584, 264),
        ("matmul_allto_all", 64, 7248, 768),
    ],
)
def test_mx_alltoall_counts_scales_and_correct_communication_dtype(
    op, weight_k, read_bytes, network_bytes
):
    case = {
        "input_shape": [
            [8, 64],
            [96, weight_k],
            [96],
            [8, 1, 2],
            [96, weight_k // 64, 2],
        ],
        "dtype": ["float8_e4m3fn", "float8_e5m2", "float32", "uint8", "uint8"],
        "attrs": {
            "world_size": 2,
            "is_quant": True,
            "is_trans_x2": True,
            "x1_quant_mode": 6,
            "x2_quant_mode": 6,
            "out_dtype": "bfloat16",
        },
    }
    assert model.resource_counts(op, case) == {
        "cube_ops": 98304,
        "cube_type": "cube8",
        "read_bytes": read_bytes,
        "network_bytes": network_bytes,
    }
    case["input_shape"][3] = [8, 2, 2]
    with pytest.raises(ValueError, match="scale shapes"):
        model.resource_counts(op, case)


@pytest.mark.parametrize("shape", [[6, 1280], [2, 3, 1280]])
@pytest.mark.parametrize(
    "op,network_bytes",
    [
        ("quant_all_reduce", 7920),
        ("quant_reduce_scatter", 3960),
    ],
)
def test_quant_collectives_exchange_packed_inputs_and_scales(shape, op, network_bytes):
    case = {
        "input_shape": [shape, shape[:-1] + [10]],
        "dtype": ["int8", "float32"],
        "attrs": {
            "world_size": 2,
            "supported_devices": ["950dt"],
            "out_dtype": "float32",
        },
    }
    assert model.resource_counts(op, case) == {
        "cube_ops": 0,
        "cube_type": "none",
        "read_bytes": 7920,
        "network_bytes": network_bytes,
    }
    assert model.compute(op, case, "950dt")["components_us"]["cube_us"] == 0
    with pytest.raises(ValueError, match="does not support 950pr"):
        model.compute(op, case, "950pr")


@pytest.mark.parametrize(
    "shape,scales",
    [
        ([3, 1024], [3, 8]),  # QAR also requires rows divisible by the rank count.
        ([2, 1000], [2, 8]),
        ([2, 8320], [2, 65]),
        ([2, 1024], [2, 1]),
    ],
)
def test_quant_collective_model_rejects_incompatible_dimensions(shape, scales):
    case = {
        "input_shape": [shape, scales],
        "dtype": ["int8", "float32"],
        "attrs": {"world_size": 2},
    }
    with pytest.raises(ValueError):
        model.resource_counts("quant_all_reduce", case)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("empty_expert", [False, True])
def test_mx_gmm_counts_fp8_reads_e8m0_scales_and_fp16_network(transpose, empty_expert):
    case = {
        "input_shape": [
            [8, 64],
            [2, 96, 64] if transpose else [2, 64, 96],
            [4],
            [4],
            [4, 128],
            [64, 128] if transpose else [128, 64],
            [8, 1, 2],
            [2, 96, 1, 2] if transpose else [2, 1, 96, 2],
            [4, 2, 2],
            [64, 2, 2] if transpose else [2, 64, 2],
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
            "trans_mm_weight": transpose,
            "out_dtype": "float16",
            "token_counts": [[[3, 1], [0, 4]], [[2, 2], [1, 3]]],
        },
    }
    if empty_expert:
        case["attrs"]["token_counts"] = [[[4, 0], [4, 0]], [[4, 0], [4, 0]]]
    # 2*8*64*96 + 2*4*128*64 = 163840 Cube ops.
    # Four remote rows send 96 FP16 outputs each, regardless of FP8 input storage.
    assert model.resource_counts("quant_grouped_mat_mul_allto_allv", case) == {
        "cube_ops": 163840,
        "cube_type": "cube8",
        "read_bytes": 15840 if empty_expert else 22176,
        "network_bytes": 768,
    }
