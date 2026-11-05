#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Ascend 950DT MXFP8 grouped matmul, unpermute and AlltoAllv."""

import torch


def _dtype(name):
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
    }[name]


def _layout(ctx, attrs):
    p, rank, e = int(ctx["world_size"]), int(ctx["rank"]), int(attrs["e"])
    counts = attrs["token_counts"]
    if p != 2 or attrs["world_size"] != p or not 0 <= rank < p or not 1 <= e <= 32:
        raise ValueError(
            "This A5 MXFP8 suite requires two ranks and 1..32 local experts"
        )
    if len(counts) != p or any(len(row) != p for row in counts):
        raise ValueError("token_counts must be C[source][destination][expert]")
    if any(
        len(edge) != e or any(type(v) is not int or v < 0 for v in edge)
        for row in counts
        for edge in row
    ):
        raise ValueError("token_counts must contain nonnegative integer counts")
    send = [v for edge in counts[rank] for v in edge]
    recv = [v for source in range(p) for v in counts[source][rank]]
    if not 0 < sum(send) < 52428800 or not 0 < sum(recv) < 52428800:
        raise ValueError("Each rank must send and receive nonempty token tensors")
    return p, rank, e, counts, send, recv


def mc2_make_rank_inputs(ctx, case_payload):
    attrs = case_payload["attrs"]
    shapes, dtypes, ranges = (
        case_payload["input_shapes"],
        case_payload["dtypes"],
        case_payload["value_ranges"],
    )
    _, rank, _, _, send, recv = _layout(ctx, attrs)
    names = (
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
    result = {"send_counts": send, "recv_counts": recv}
    for index, name in enumerate(names):
        if index in (2, 3):
            continue
        shape = shapes[index]
        if shape is None:
            result[name] = None
            continue
        low, high = ranges[index]
        seed = int(attrs["seed"]) + rank * 1000003 + index * 1009
        if name.endswith("_scale"):
            if not 1 <= low <= high <= 254:
                raise ValueError("Scale codes must be finite normal E8M0 exponents")
            count = 1
            for dimension in shape:
                count *= dimension
            codes = torch.arange(count, dtype=torch.int64)
            value = (
                (int(low) + (codes + seed) % (int(high) - int(low) + 1))
                .reshape(shape)
                .to(torch.uint8)
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = (
                torch.empty(shape, dtype=torch.float32)
                .uniform_(low, high, generator=generator)
                .to(_dtype(dtypes[index]))
            )
        result[name] = value.to(ctx["device"])
    if result["gmm_x"].shape[0] != sum(send):
        raise ValueError("gmm_x row count must equal sum(send_counts)")
    return result


def _mx_rows(values, codes):
    """Values are [...,rows,K]; E8M0 codes are [...,rows,ceil(K/64),2]."""
    k = values.shape[-1]
    if codes.dtype != torch.uint8 or tuple(codes.shape) != tuple(values.shape[:-1]) + (
        (k + 63) // 64,
        2,
    ):
        raise ValueError("MX row scales must be uint8 [...,rows,ceil(K/64),2]")
    scales = (
        torch.exp2(codes.float() - 127.0)
        .flatten(-2)
        .repeat_interleave(32, dim=-1)[..., :k]
    )
    return values.float() * scales


def _mx_weight(weight, scale, transposed):
    # Physical NT: [e,K,N], [e,ceil(K/64),N,2]; T: [e,N,K], [e,N,ceil(K/64),2].
    # The same last-axis rule handles the optional 2D shared weight.
    rows = weight if transposed else weight.transpose(-2, -1)
    codes = scale if transposed else scale.transpose(-3, -2)
    return _mx_rows(rows, codes).transpose(-2, -1)


def _reference(ctx, inputs, attrs, round_output):
    p, rank, e, counts, send, recv = _layout(ctx, attrs)
    if list(inputs["send_counts"]) != send or list(inputs["recv_counts"]) != recv:
        raise ValueError("Local counts must match C[source][destination][expert]")
    if attrs["gmm_x_quant_mode"] != 6 or attrs["gmm_weight_quant_mode"] != 6:
        raise ValueError("This contract supports MX quant mode 6 only")
    if attrs["group_size"] != 4295032864 or attrs["comm_quant_mode"] != 0:
        raise ValueError("MX groups are [1,1,32] and communication is not quantized")
    x = _mx_rows(inputs["gmm_x"], inputs["gmm_x_scale"])
    weight = _mx_weight(
        inputs["gmm_weight"], inputs["gmm_weight_scale"], attrs["trans_gmm_weight"]
    )
    output_dtype = _dtype(attrs["out_dtype"]) if round_output else torch.float32
    by_expert, offset = [], 0
    for expert in range(e):
        rows = sum(counts[rank][destination][expert] for destination in range(p))
        product = torch.matmul(x[offset : offset + rows], weight[expert]).to(
            output_dtype
        )
        by_expert.append(
            list(
                product.split(
                    [counts[rank][destination][expert] for destination in range(p)],
                    dim=0,
                )
            )
        )
        offset += rows
    sent = torch.cat(
        [
            by_expert[expert][destination]
            for destination in range(p)
            for expert in range(e)
        ],
        dim=0,
    ).contiguous()
    y = torch.empty((sum(recv), weight.shape[-1]), dtype=output_dtype, device=x.device)
    ctx["dist"].all_to_all_single(
        y,
        input=sent,
        input_split_sizes=[sum(edge) for edge in counts[rank]],
        output_split_sizes=[sum(counts[source][rank]) for source in range(p)],
    )
    mm_y = None
    if attrs["mm_out"]:
        mm_x = _mx_rows(inputs["mm_x"], inputs["mm_x_scale"])
        mm_weight = _mx_weight(
            inputs["mm_weight"], inputs["mm_weight_scale"], attrs["trans_mm_weight"]
        )
        mm_y = torch.matmul(mm_x, mm_weight).to(output_dtype)
    return y, mm_y


def mc2_distributed_golden(ctx, inputs, attrs):
    """NPU cascade: exact MX dequantization, FP32 matmul, then output rounding before communication."""
    return _reference(ctx, inputs, attrs, round_output=True)


def mc2_cpu_golden(ctx, inputs, attrs):
    """CPU reference communicates FP32 results and does not round the outputs."""
    return _reference(ctx, inputs, attrs, round_output=False)


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["gmm_x"],
        inputs["gmm_weight"],
        inputs["gmm_x_scale"],
        inputs["gmm_weight_scale"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        inputs["send_counts"],
        inputs["recv_counts"],
        mm_x=inputs.get("mm_x"),
        mm_weight=inputs.get("mm_weight"),
        mm_x_scale=inputs.get("mm_x_scale"),
        mm_weight_scale=inputs.get("mm_weight_scale"),
        trans_gmm_weight=attrs["trans_gmm_weight"],
        trans_mm_weight=attrs["trans_mm_weight"],
        out_dtype=attrs["out_dtype"],
        gmm_x_quant_mode=attrs["gmm_x_quant_mode"],
        gmm_weight_quant_mode=attrs["gmm_weight_quant_mode"],
        mm_x_quant_mode=attrs["mm_x_quant_mode"],
        mm_weight_quant_mode=attrs["mm_weight_quant_mode"],
        comm_quant_mode=attrs["comm_quant_mode"],
        group_size=attrs["group_size"],
        scale_dtype=attrs["scale_dtype"],
    )
