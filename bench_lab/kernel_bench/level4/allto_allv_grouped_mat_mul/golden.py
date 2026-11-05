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


"""Ascend 950DT AlltoAllv -> Permute -> GroupedMatMul references."""

import torch


def _layout(ctx, attrs):
    p, rank, e = int(ctx["world_size"]), int(ctx["rank"]), int(attrs["e"])
    counts = attrs["token_counts"]
    if (
        p not in (2, 4, 8, 16, 32, 64)
        or int(attrs["world_size"]) != p
        or not 0 <= rank < p
    ):
        raise ValueError(
            "world_size and rank must match the Ascend 950DT communication group"
        )
    if not 1 <= e <= 32 or p * e > 256:
        raise ValueError(
            "this contract requires 1 <= e <= 32 and world_size * e <= 256"
        )
    if len(counts) != p or any(len(row) != p for row in counts):
        raise ValueError("token_counts must have shape [world_size, world_size, e]")
    for row in counts:
        for edge in row:
            if len(edge) != e or any(
                type(value) is not int or value < 0 for value in edge
            ):
                raise ValueError(
                    "token counts must be nonnegative integers, one per expert"
                )
    send = [value for edge in counts[rank] for value in edge]
    recv = [value for source in range(p) for value in counts[source][rank]]
    if not 0 < sum(send) < 52428800 or not 0 < sum(recv) < 52428800:
        raise ValueError("each rank must send and receive a nonempty token tensor")
    return p, rank, e, counts, send, recv


def _validate_inputs(ctx, inputs, attrs):
    p, rank, e, counts, send, recv = _layout(ctx, attrs)
    x, physical = inputs["gmm_x"], inputs["gmm_weight"]
    weight = (
        physical.transpose(1, 2)
        if bool(attrs.get("trans_gmm_weight", False))
        else physical
    )
    if x.ndim != 2 or weight.ndim != 3 or weight.shape[0] != e:
        raise ValueError("gmm_x must be 2D and gmm_weight must contain e matrices")
    if x.shape[0] != sum(send) or x.shape[1] != weight.shape[1]:
        raise ValueError(
            "gmm_x rows must equal sum(send_counts), with matching contraction dimensions"
        )
    if not 0 < x.shape[1] < 65536 or not 0 < weight.shape[2] < 65536:
        raise ValueError("H1 and N1 must be in [1,65535]")
    if list(inputs["send_counts"]) != send or list(inputs["recv_counts"]) != recv:
        raise ValueError(
            "rank-local counts disagree with token_counts[source][destination][expert]"
        )
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("this contract requires FP16 or BF16 inputs")
    mm_x, mm_weight = inputs.get("mm_x"), inputs.get("mm_weight")
    if (mm_x is not None) != bool(attrs.get("mm_out", False)) or (mm_x is None) != (
        mm_weight is None
    ):
        raise ValueError(
            "mm_x and mm_weight must both be present exactly when mm_out is enabled"
        )
    tensors = [x, physical]
    if mm_x is not None:
        effective_mm = (
            mm_weight.t() if bool(attrs.get("trans_mm_weight", False)) else mm_weight
        )
        if (
            mm_x.ndim != 2
            or effective_mm.ndim != 2
            or mm_x.shape[1] != effective_mm.shape[0]
        ):
            raise ValueError(
                "shared expert matrices must be 2D with matching contraction dimensions"
            )
        bs = mm_x.shape[0]
        if bs <= 0 or x.shape[0] % bs or not 2 <= x.shape[0] // bs <= 8:
            raise ValueError(
                "shared expert BS must divide sent token rows with a ratio in [2,8]"
            )
        if not 0 < mm_x.shape[1] <= 12288 or not 0 < effective_mm.shape[1] < 65536:
            raise ValueError("shared expert H2 and N2 exceed the supported range")
        tensors += [mm_x, mm_weight]
    if any(
        value.dtype != x.dtype or value.device != x.device or not value.is_contiguous()
        for value in tensors
    ):
        raise ValueError(
            "physical inputs must be contiguous tensors of the same dtype on one device"
        )
    return p, rank, e, counts, send, recv


def _make_tensor(shape, dtype, value_range, seed, device):
    if shape is None:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    lo, hi = value_range or [-1.0, 1.0]
    return (
        torch.empty(shape, dtype=torch.float32)
        .uniform_(float(lo), float(hi), generator=generator)
        .to(dtype)
        .to(device)
    )


def mc2_make_rank_inputs(ctx, case_payload):
    attrs, shapes, dtypes = (
        case_payload["attrs"],
        case_payload["input_shapes"],
        case_payload["dtypes"],
    )
    ranges = case_payload.get("value_ranges") or [None] * 6
    _, rank, _, _, send, recv = _layout(ctx, attrs)
    seed = int(attrs.get("seed", 1)) + rank * 1000003
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtypes[0]]
    inputs = {
        "gmm_x": _make_tensor(shapes[0], dtype, ranges[0], seed, ctx["device"]),
        "gmm_weight": _make_tensor(
            shapes[1], dtype, ranges[1], seed + 101, ctx["device"]
        ),
        "send_counts": send,
        "recv_counts": recv,
        "mm_x": _make_tensor(shapes[4], dtype, ranges[4], seed + 211, ctx["device"]),
        "mm_weight": _make_tensor(
            shapes[5], dtype, ranges[5], seed + 307, ctx["device"]
        ),
    }
    _validate_inputs(ctx, inputs, attrs)
    return inputs


def mc2_distributed_golden(ctx, inputs, attrs):
    """Real AlltoAllv followed by a permutation and native-dtype matrix products."""
    p, rank, e, counts, send, recv = _validate_inputs(ctx, inputs, attrs)
    x = inputs["gmm_x"]
    received = x.new_empty((sum(recv), x.shape[1]))
    ctx["dist"].all_to_all_single(
        received,
        input=x,
        output_split_sizes=[sum(counts[s][rank]) for s in range(p)],
        input_split_sizes=[sum(edge) for edge in counts[rank]],
    )
    offsets, offset = [], 0
    for count in recv:
        offsets.append(offset)
        offset += count
    permutation = []
    for expert in range(e):
        for source in range(p):
            block = source * e + expert
            permutation.extend(range(offsets[block], offsets[block] + recv[block]))
    indices = torch.tensor(permutation, dtype=torch.int64, device=x.device)
    permuted = received.index_select(0, indices)
    weight = inputs["gmm_weight"]
    if bool(attrs.get("trans_gmm_weight", False)):
        weight = weight.transpose(1, 2)
    products, start = [], 0
    for expert in range(e):
        rows = sum(counts[source][rank][expert] for source in range(p))
        if rows:
            products.append(
                torch.matmul(permuted[start : start + rows], weight[expert])
            )
        start += rows
    y = torch.cat(products, dim=0)
    mm_y = None
    if inputs.get("mm_x") is not None:
        mm_weight = (
            inputs["mm_weight"].t()
            if bool(attrs.get("trans_mm_weight", False))
            else inputs["mm_weight"]
        )
        mm_y = torch.matmul(inputs["mm_x"], mm_weight)
    return y, mm_y, permuted if bool(attrs.get("permute_out_flag", False)) else None


def mc2_cpu_golden(ctx, inputs, attrs):
    """FP32 source/expert slices define the reference independently of the permutation."""
    p, rank, e, counts, send, recv = _validate_inputs(ctx, inputs, attrs)
    x, weight = inputs["gmm_x"].float(), inputs["gmm_weight"].float()
    if bool(attrs.get("trans_gmm_weight", False)):
        weight = weight.transpose(1, 2)
    received = torch.empty(
        (sum(recv), x.shape[1]), dtype=torch.float32, device=x.device
    )
    ctx["dist"].all_to_all_single(
        received,
        input=x,
        output_split_sizes=[sum(counts[source][rank]) for source in range(p)],
        input_split_sizes=[sum(edge) for edge in counts[rank]],
    )
    by_source = list(
        received.split([sum(counts[source][rank]) for source in range(p)], dim=0)
    )
    products, ordered_inputs = [], []
    for expert in range(e):
        for source in range(p):
            begin = sum(counts[source][rank][:expert])
            end = begin + counts[source][rank][expert]
            block = by_source[source][begin:end]
            ordered_inputs.append(block)
            products.append(torch.matmul(block, weight[expert]))
    mm_y = None
    if inputs.get("mm_x") is not None:
        mm_weight = inputs["mm_weight"].float()
        if bool(attrs.get("trans_mm_weight", False)):
            mm_weight = mm_weight.t()
        mm_y = torch.matmul(inputs["mm_x"].float(), mm_weight)
    permuted = (
        torch.cat(ordered_inputs, dim=0)
        if bool(attrs.get("permute_out_flag", False))
        else None
    )
    return torch.cat(products, dim=0), mm_y, permuted


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["gmm_x"],
        inputs["gmm_weight"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        inputs["send_counts"],
        inputs["recv_counts"],
        mm_x=inputs.get("mm_x"),
        mm_weight=inputs.get("mm_weight"),
        trans_gmm_weight=bool(attrs.get("trans_gmm_weight", False)),
        trans_mm_weight=bool(attrs.get("trans_mm_weight", False)),
        permute_out_flag=bool(attrs.get("permute_out_flag", False)),
    )
