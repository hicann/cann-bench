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


"""Ascend 950DT INT8 per-token/per-group QuantReduceScatter references."""

import torch


def _dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(name).lower()]


def _validate(ctx, inputs, attrs):
    x, scales = inputs["x"], inputs["scales"]
    p = int(ctx["world_size"])
    if p not in (2, 4, 8) or int(attrs.get("world_size", p)) != p:
        raise ValueError("world_size must match the 2-, 4-, or 8-rank group")
    if x.ndim not in (2, 3) or any(dim <= 0 for dim in x.shape):
        raise ValueError("x must be a nonempty 2D or 3D tensor")
    h = x.shape[-1]
    if not 1024 <= h <= 8192 or h % 128 or (x.numel() // h) % p:
        raise ValueError(
            "H must be 128-aligned in [1024,8192]; BS must be divisible by world_size"
        )
    if tuple(scales.shape) != tuple(x.shape[:-1]) + (h // 128,):
        raise ValueError(
            "scales must contain one FP32 value per token and 128 channels"
        )
    if x.dtype != torch.int8 or scales.dtype != torch.float32:
        raise ValueError("this contract requires INT8 x and FP32 scales")
    if not x.is_contiguous() or not scales.is_contiguous() or x.device != scales.device:
        raise ValueError("x and scales must be contiguous on the same device")
    if (
        attrs.get("x_dtype", "int8") != "int8"
        or attrs.get("scales_dtype", "float32") != "float32"
    ):
        raise ValueError("only INT8 x and FP32 scales are supported by this contract")
    if attrs.get("group_size", 128) != 128 or attrs.get("reduce_op", "sum") != "sum":
        raise ValueError("group_size=128 and reduce_op=sum are required")
    _dtype(attrs.get("out_dtype", "float16"))
    return p


def mc2_make_rank_inputs(ctx, case_payload):
    shapes, attrs = case_payload["input_shapes"], case_payload["attrs"]
    ranges = case_payload.get("value_ranges") or [[-128, 127], [0.001, 0.25]]
    seed = int(attrs.get("seed", 1)) + int(ctx["rank"]) * 1000003
    x_generator = torch.Generator(device="cpu").manual_seed(seed)
    scale_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    x = torch.randint(
        int(ranges[0][0]),
        int(ranges[0][1]) + 1,
        shapes[0],
        dtype=torch.int8,
        generator=x_generator,
    )
    scales = torch.empty(shapes[1], dtype=torch.float32).uniform_(
        float(ranges[1][0]), float(ranges[1][1]), generator=scale_generator
    )
    inputs = {"x": x.to(ctx["device"]), "scales": scales.to(ctx["device"])}
    _validate(ctx, inputs, attrs)
    return inputs


def mc2_distributed_golden(ctx, inputs, attrs):
    """Dequantize in FP32, perform a real reduce-scatter, then cast once."""
    p = _validate(ctx, inputs, attrs)
    x, scales = inputs["x"], inputs["scales"]
    local = (x.float() * scales.repeat_interleave(128, dim=-1)).reshape(-1, x.shape[-1])
    output = torch.empty(
        (local.shape[0] // p, local.shape[1]), dtype=torch.float32, device=local.device
    )
    ctx["dist"]._reduce_scatter_base(
        output, local.contiguous(), op=ctx["dist"].ReduceOp.SUM
    )
    return output.to(_dtype(attrs.get("out_dtype", "float16")))


def mc2_cpu_golden(ctx, inputs, attrs):
    """Independent FP32 reference: sum all rows, then select this rank's contiguous slice."""
    p = _validate(ctx, inputs, attrs)
    x, scales = inputs["x"], inputs["scales"]
    groups = x.float().reshape(*scales.shape, 128)
    total = (groups * scales.unsqueeze(-1)).reshape(-1, x.shape[-1])
    ctx["dist"].all_reduce(total, op=ctx["dist"].ReduceOp.SUM)
    rows, rank = total.shape[0] // p, int(ctx["rank"])
    return total[rank * rows : (rank + 1) * rows].contiguous()


def mc2_cpu_references(ctx, inputs, attrs):
    """Model the one final output cast after FP32 dequantization and reduction."""
    mathematical = mc2_cpu_golden(ctx, inputs, attrs)
    return mathematical, mathematical.to(
        _dtype(attrs.get("out_dtype", "float16"))
    ).float()


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["x"],
        inputs["scales"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        out_dtype=attrs.get("out_dtype", "float16"),
    )
