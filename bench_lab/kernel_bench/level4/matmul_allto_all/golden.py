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


"""A5 MatmulAlltoAll distributed golden, following mc2_test semantics."""

import torch


def _dtype(name):
    return {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "uint8": torch.uint8,
    }[str(name).lower()]


def _tensor(shape, dtype_name, value_range, seed, device):
    if shape is None:
        return None
    torch.manual_seed(int(seed))
    lo, hi = value_range or [-1.0, 1.0]
    if dtype_name == "uint8":
        return torch.randint(int(lo), int(hi) + 1, shape, dtype=torch.uint8).to(device)
    return (
        torch.empty(shape, dtype=torch.float32)
        .uniform_(float(lo), float(hi))
        .to(_dtype(dtype_name))
        .to(device)
    )


def mc2_make_rank_inputs(ctx, case_payload):
    shapes, dtypes = case_payload["input_shapes"], case_payload["dtypes"]
    ranges, attrs = case_payload.get("value_ranges") or [], case_payload["attrs"]
    seed, device = int(attrs.get("seed", 1)), ctx["device"]
    if attrs.get("is_quant", False):
        rank_seed = seed + 1009 * int(ctx["rank"])
        names = ("x1", "x2", "bias", "x1_scale", "x2_scale")
        inputs = {
            name: _tensor(shape, dtype, bounds, rank_seed + i * 37, device)
            for i, (name, shape, dtype, bounds) in enumerate(
                zip(names, shapes, dtypes, ranges)
            )
        }
        _validate_quant(ctx, inputs, attrs)
        return inputs
    # Distinct senders expose omitted peers and incorrect receive ordering.
    x1_seed = seed
    if attrs.get("mc2_accuracy_mode") == "three_way":
        x1_seed += int(ctx["rank"]) * 1000003
    bias = None
    if bool(attrs.get("is_bias", False)):
        bias = _tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            seed * 7,
            device,
        )
    return {
        "x1": _tensor(
            shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
        ),
        "x2": _tensor(
            shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device
        ),
        "bias": bias,
    }


def _validate_quant(ctx, inputs, attrs):
    """Validate the supported A5 MXFP8 interface, including packed scale layout."""
    if (attrs.get("x1_quant_mode"), attrs.get("x2_quant_mode")) != (6, 6):
        raise ValueError("QuantMatmulAlltoAll requires MX quantization modes (6, 6)")
    if not attrs.get("is_trans_x2", False):
        raise ValueError("MX quantization requires is_trans_x2=True")
    if attrs.get("group_size") != 4295032864 or attrs.get("scale_dtype") != 293:
        raise ValueError("MX requires group_size=[1,1,32] and E8M0 scale_dtype=293")
    if attrs.get("out_dtype") not in ("float16", "bfloat16"):
        raise ValueError("out_dtype must be float16 or bfloat16")
    x1, x2 = inputs["x1"], inputs["x2"]
    allowed = (torch.float8_e4m3fn, torch.float8_e5m2)
    if x1.dtype not in allowed or x2.dtype not in allowed:
        raise ValueError("MX operands must use FP8 E4M3FN or E5M2")
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("MX operands must be two-dimensional")
    m, k = x1.shape
    n, weight_k = x2.shape
    p = int(ctx["world_size"])
    if p not in (2, 4, 8, 16) or m < 1 or not 1 <= k <= 65535 or weight_k != k:
        raise ValueError("Invalid MX operand dimensions or world_size")
    if n < 2 or n % p or p * m > 2147483647 or n > 2147483647:
        raise ValueError(
            "N must be divisible by world_size and output dimensions must fit INT32"
        )
    q = (k + 63) // 64
    for name, expected in (("x1_scale", (m, q, 2)), ("x2_scale", (n, q, 2))):
        value = inputs.get(name)
        if (
            value is None
            or value.dtype != torch.uint8
            or tuple(value.shape) != expected
        ):
            raise ValueError(f"{name} must be uint8 E8M0 codes of shape {expected}")
    bias = inputs.get("bias")
    if bias is not None and (bias.dtype != torch.float32 or tuple(bias.shape) != (n,)):
        raise ValueError("MX bias must be float32 of shape [N]")


def _mx_rows(values, codes):
    """Decode the two E8M0 bytes per 64 elements, each covering 32 K elements."""
    scales = torch.exp2(codes.float() - 127.0).flatten(-2).repeat_interleave(32, dim=-1)
    return values.float() * scales[..., : values.shape[-1]]


def _alltoall(ctx, out):
    world_size = int(ctx["world_size"])
    m, n = out.shape
    if n % world_size:
        raise ValueError(f"N={n} must be divisible by world_size={world_size}")
    sent = out.reshape(m, world_size, n // world_size).permute(1, 0, 2).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    return received.reshape(m * world_size, n // world_size)


def _quant_matmul(inputs):
    x1 = _mx_rows(inputs["x1"], inputs["x1_scale"])
    weight = _mx_rows(inputs["x2"], inputs["x2_scale"]).t()
    out = torch.matmul(x1, weight)
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
    return out


def mc2_distributed_golden(ctx, inputs, attrs):
    """Local matmul followed by documented N-split/M-concat all-to-all."""
    if attrs.get("is_quant", False):
        _validate_quant(ctx, inputs, attrs)
        return _alltoall(ctx, _quant_matmul(inputs).to(_dtype(attrs["out_dtype"])))
    x2 = inputs["x2"].t() if bool(attrs.get("is_trans_x2", False)) else inputs["x2"]
    out = torch.matmul(inputs["x1"], x2)
    bias = inputs.get("bias")
    if bias is not None:
        out = (
            (out.to(torch.float32) + bias.float()).to(out.dtype)
            if bias.dtype == torch.float32
            else out + bias
        )
    world_size = int(ctx["world_size"])
    m, n = out.shape[-2:]
    if n % world_size:
        raise ValueError(f"N={n} must be divisible by world_size={world_size}")
    sent = out.reshape(m, world_size, n // world_size).permute(1, 0, 2).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    return received.reshape(m * world_size, n // world_size)


def mc2_cpu_golden(ctx, inputs, attrs):
    """Independent FP32 math and CPU communication, with no output rounding."""
    if attrs.get("is_quant", False):
        _validate_quant(ctx, inputs, attrs)
        return _alltoall(ctx, _quant_matmul(inputs))
    fp32_inputs = {
        key: value.float() if value is not None else None
        for key, value in inputs.items()
    }
    return mc2_distributed_golden(ctx, fp32_inputs, attrs)


def mc2_cpu_references(ctx, inputs, attrs):
    """Return FP32 G and a separately communicated low-precision stage model.

    The native cascade rounds matmul, optionally adds bias and rounds again,
    then performs an exact all-to-all permutation. Neither CPU path depends
    on the candidate or the NPU cascade output.
    """
    if bool(attrs.get("is_quant", False)):
        mathematical = mc2_cpu_golden(ctx, inputs, attrs)
        # A final elementwise cast commutes with the exact all-to-all permutation.
        # This preserves the NPU order (cast before exchange) without a second
        # CPU collective or any dependence on the NPU/candidate output.
        return mathematical, mathematical.to(_dtype(attrs["out_dtype"])).float()
    if int(ctx["world_size"]) != 2:
        return mc2_cpu_golden(ctx, inputs, attrs), None

    x2 = inputs["x2"].float()
    x2_eff = x2.t() if bool(attrs.get("is_trans_x2", False)) else x2
    local = torch.matmul(inputs["x1"].float(), x2_eff)
    dtype = inputs["x1"].dtype
    rounded = local.to(dtype).float()
    bias = inputs.get("bias")
    if bias is not None:
        local = local + bias.float()
        rounded = (rounded + bias.float()).to(dtype).float()
    mathematical = _alltoall(ctx, local)
    stage_rounded = _alltoall(ctx, rounded)
    return mathematical, stage_rounded


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    if attrs.get("is_quant", False):
        _validate_quant(ctx, inputs, attrs)
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            int(ctx["world_size"]),
            bias=inputs.get("bias"),
            is_trans_x2=True,
            x1_scale=inputs["x1_scale"],
            x2_scale=inputs["x2_scale"],
            x1_quant_mode=6,
            x2_quant_mode=6,
            group_size=attrs["group_size"],
            scale_dtype=attrs["scale_dtype"],
            out_dtype=attrs["out_dtype"],
        )
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        is_trans_x2=bool(attrs.get("is_trans_x2", False)),
    )
