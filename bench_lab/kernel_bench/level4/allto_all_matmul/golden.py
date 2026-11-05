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


"""A5 AlltoAllMatmul distributed golden, following mc2_test semantics."""

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
    # Different senders must remain distinguishable after the all-to-all permutation.
    x1_seed = seed + int(ctx["rank"]) * 1000003
    bias = None
    if bool(attrs.get("is_bias", False)):
        bias = _tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            seed * 7,
            device,
        )
    inputs = {
        "x1": _tensor(
            shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
        ),
        "x2": _tensor(
            shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device
        ),
        "bias": bias,
    }
    if bool(attrs.get("is_quant", False)):
        for name, index, scale_seed in (
            ("x1_scale", 3, x1_seed + 17),
            ("x2_scale", 4, seed + 29),
        ):
            torch.manual_seed(scale_seed)
            lo, hi = ranges[index]
            inputs[name] = torch.randint(
                int(lo), int(hi) + 1, shapes[index], dtype=torch.uint8
            ).to(device)
        _validate_mx(inputs, attrs, int(ctx["world_size"]))
    return inputs


def _validate_mx(inputs, attrs, world_size):
    x1, x2 = inputs["x1"], inputs["x2"]
    m, k = x1.shape
    if (
        m < world_size
        or m % world_size
        or k % 64
        or not 1 <= k * world_size <= 65535
        or not bool(attrs.get("is_trans_x2", False))
    ):
        raise ValueError(
            "MX requires M divisible by world_size, K divisible by 64 and transposed x2"
        )
    if x2.ndim != 2 or x2.shape[1] != k * world_size:
        raise ValueError("MX x2 must be stored as [N, world_size*K]")
    if (int(attrs.get("x1_quant_mode", 6)), int(attrs.get("x2_quant_mode", 6))) != (
        6,
        6,
    ):
        raise ValueError("Only MX quantization modes (6, 6) are supported")
    if (
        int(attrs.get("group_size", 4295032864)) != 4295032864
        or int(attrs.get("scale_dtype", 293)) != 293
    ):
        raise ValueError("MX requires group_size=[1,1,32] and E8M0 scale encoding 293")
    for name, shape in (
        ("x1_scale", (m, k // 64, 2)),
        ("x2_scale", (x2.shape[0], k * world_size // 64, 2)),
    ):
        scale = inputs.get(name)
        if scale is None or tuple(scale.shape) != shape or scale.dtype != torch.uint8:
            raise ValueError(f"{name} must contain raw E8M0 bytes with shape {shape}")


def _alltoall_columns(ctx, tensor):
    world_size = int(ctx["world_size"])
    m = tensor.shape[0]
    columns = tensor.numel() // m
    # Transport FP8 bit patterns; HCCL/Gloo do not need a native FP8 wire type.
    is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    wire = tensor.contiguous().view(torch.uint8) if is_fp8 else tensor
    sent = wire.reshape(world_size, m // world_size, columns).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    result = (
        received.permute(1, 0, 2)
        .contiguous()
        .reshape(m // world_size, world_size * columns)
    )
    return result.view(tensor.dtype) if is_fp8 else result


def _mx_dequant(tensor, scales):
    # Two E8M0 bytes describe consecutive 32-element K blocks.
    expanded = torch.exp2(scales.float().reshape(scales.shape[0], -1) - 127)
    return tensor.float() * expanded.repeat_interleave(32, dim=-1)


def _mx_golden(ctx, inputs, attrs, round_output):
    _validate_mx(inputs, attrs, int(ctx["world_size"]))
    alltoall_out = _alltoall_columns(ctx, inputs["x1"])
    received_scale = _alltoall_columns(ctx, inputs["x1_scale"])
    left = _mx_dequant(alltoall_out, received_scale)
    right = _mx_dequant(inputs["x2"], inputs["x2_scale"]).t()
    out = torch.matmul(left, right)
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
    if round_output:
        out = out.to(_dtype(attrs.get("out_dtype", "bfloat16")))
    return out, alltoall_out if bool(attrs.get("alltoall_out_flag", False)) else None


def mc2_distributed_golden(ctx, inputs, attrs):
    """All-to-all x1 on NPU, then local matmul, matching the documented formula."""
    if bool(attrs.get("is_quant", False)):
        return _mx_golden(ctx, inputs, attrs, round_output=True)
    x1, x2 = inputs["x1"], inputs["x2"]
    world_size = int(ctx["world_size"])
    m, k = x1.shape[-2:]
    if m % world_size:
        raise ValueError(f"M={m} must be divisible by world_size={world_size}")
    sent = x1.reshape(world_size, m // world_size, k).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    alltoall_out = (
        received.permute(1, 0, 2).contiguous().reshape(m // world_size, k * world_size)
    )
    x2_eff = x2.t() if bool(attrs.get("is_trans_x2", False)) else x2
    out = torch.matmul(alltoall_out, x2_eff)
    bias = inputs.get("bias")
    if bias is not None:
        out = (
            (out.to(torch.float32) + bias.float()).to(out.dtype)
            if bias.dtype == torch.float32
            else out + bias
        )
    return out, alltoall_out if bool(attrs.get("alltoall_out_flag", False)) else None


def mc2_cpu_golden(ctx, inputs, attrs):
    """Independent FP32 math and CPU communication, with no output rounding."""
    if bool(attrs.get("is_quant", False)):
        fp32_inputs = {
            key: value.float()
            if value is not None and key not in ("x1_scale", "x2_scale")
            else value
            for key, value in inputs.items()
        }
        return _mx_golden(ctx, fp32_inputs, attrs, round_output=False)
    fp32_inputs = {
        key: value.float() if value is not None else None
        for key, value in inputs.items()
    }
    return mc2_distributed_golden(ctx, fp32_inputs, attrs)


def mc2_cpu_references(ctx, inputs, attrs):
    """Return mathematical G and a CPU model of the NPU rounding stages.

    All-to-all only permutes the already-rounded input, so one FP32 CPU
    exchange feeds both paths exactly. Matmul rounds before optional bias.
    """
    if bool(attrs.get("is_quant", False)):
        mathematical = mc2_cpu_golden(ctx, inputs, attrs)
        output, exchanged = mathematical
        # MX dequantization and matmul stay FP32; only the final matrix is cast.
        rounded = output.to(_dtype(attrs.get("out_dtype", "bfloat16"))).float()
        return mathematical, (rounded, exchanged)
    if int(ctx["world_size"]) != 2:
        return mc2_cpu_golden(ctx, inputs, attrs), None

    received = _alltoall_columns(ctx, inputs["x1"].float())
    x2 = inputs["x2"].float()
    x2_eff = x2.t() if bool(attrs.get("is_trans_x2", False)) else x2
    mathematical = torch.matmul(received, x2_eff)
    dtype = inputs["x1"].dtype
    rounded = mathematical.to(dtype).float()
    bias = inputs.get("bias")
    if bias is not None:
        mathematical = mathematical + bias.float()
        rounded = (rounded + bias.float()).to(dtype).float()
    exchanged = received if bool(attrs.get("alltoall_out_flag", False)) else None
    return (mathematical, exchanged), (rounded, exchanged)


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    if bool(attrs.get("is_quant", False)):
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            int(ctx["world_size"]),
            bias=inputs.get("bias"),
            is_trans_x2=bool(attrs.get("is_trans_x2", False)),
            alltoall_out_flag=bool(attrs.get("alltoall_out_flag", False)),
            is_quant=True,
            x1_scale=inputs["x1_scale"],
            x2_scale=inputs["x2_scale"],
            x1_quant_mode=int(attrs.get("x1_quant_mode", 6)),
            x2_quant_mode=int(attrs.get("x2_quant_mode", 6)),
            group_size=int(attrs.get("group_size", 4295032864)),
            scale_dtype=int(attrs.get("scale_dtype", 293)),
            out_dtype=attrs.get("out_dtype", "bfloat16"),
        )
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        is_trans_x2=bool(attrs.get("is_trans_x2", False)),
        alltoall_out_flag=bool(attrs.get("alltoall_out_flag", False)),
    )
