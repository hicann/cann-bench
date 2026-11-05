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


from typing import Any, Dict, Optional, Tuple, Union

import torch


def all_gather_matmul_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0).float()
    out = torch.matmul(gathered, x2_eff.float())
    is_fp8 = x1.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8:
        if (
            x1_scale is None
            or x2_scale is None
            or x1_scale.numel() != 1
            or x2_scale.numel() != 1
        ):
            raise ValueError("FP8 AllGatherMatmulV2 requires per-tensor scalar scales")
        # Match the A5 distributed contract: accumulator bias is scaled too.
        if bias is not None:
            out = out + bias.float()
        scale = x1_scale.float() * x2_scale.float()
        scale = (scale.contiguous().view(torch.int32) & -8192).view(torch.float32)
        out = out * scale.reshape(1)
    else:
        if x1_scale is not None:
            scale = (
                torch.cat([x1_scale] * int(world_size), dim=0).float().reshape(-1, 1)
            )
            out = out * scale
        if x2_scale is not None:
            out = out * x2_scale.float().reshape(1, -1)
        if bias is not None:
            out = out + bias.float()
    out = out.to(_torch_dtype(out_dtype))
    if gather_output:
        return out, gathered.to(x1.dtype)
    return out


def mc2_make_rank_inputs(
    ctx: Dict[str, Any], case_payload: Dict[str, Any]
) -> Dict[str, Any]:
    device = ctx["device"]
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    attrs = case_payload["attrs"]
    seed = int(attrs.get("seed", 1))
    weight_same = int(attrs.get("weight_same", 1))
    rank = int(ctx["rank"])

    a5 = attrs.get("mc2_accuracy_mode") == "three_way"
    # Distinct shards make a missing all-gather observable; preserve legacy seeds.
    x1_seed = seed + rank * 1000003 if a5 else seed
    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0], x1_seed, device)
    x2_seed = seed + rank * 2 if weight_same else seed
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1], x2_seed, device)

    bias = None
    if (
        attrs.get("mc2_accuracy_mode") != "three_way"
        and bool(attrs.get("is_bias", False))
        and len(shapes) > 2
        and shapes[2] is not None
    ):
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2], rank * 3, device)

    x1_scale_index = 2 if a5 else (3 if bool(attrs.get("is_bias", False)) else 2)
    x2_scale_index = x1_scale_index + 1
    if a5:
        bias = (
            _make_tensor(shapes[4], dtypes[4], ranges[4], rank * 3, device)
            if attrs.get("is_bias", False)
            else None
        )
    x1_scale = _make_tensor(
        shapes[x1_scale_index],
        dtypes[x1_scale_index],
        ranges[x1_scale_index],
        rank * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        rank * 11,
        device,
    )
    return {
        "x1": x1,
        "x2": x2,
        "bias": bias,
        "x1_scale": x1_scale,
        "x2_scale": x2_scale,
    }


def mc2_call_candidate(
    candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
        gather_output=bool(attrs.get("gather_output", True)),
        is_trans_b=bool(attrs.get("is_trans_b", False)),
        out_dtype=str(attrs.get("out_dtype", "float16")),
        x1_dtype=str(attrs.get("x1_dtype", "int8")),
        x2_dtype=str(attrs.get("x2_dtype", "int8")),
        x1_scale_dtype=str(attrs.get("x1_scale_dtype", "float32")),
        x2_scale_dtype=str(attrs.get("x2_scale_dtype", "float32")),
        block_size=int(attrs.get("block_size", 128)),
        group_sizes=list(attrs.get("group_sizes", [])),
    )


def mc2_distributed_golden(
    ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    """Distributed golden following mc2_test's hccl_mm reference path."""
    torch_npu = ctx["torch_npu"]
    dist = ctx["dist"]
    world_size = int(ctx["world_size"])
    gather_shape = [inputs["x1"].shape[0] * world_size] + list(inputs["x1"].shape[1:])
    gathered = torch.empty(
        gather_shape, dtype=inputs["x1"].dtype, device=inputs["x1"].device
    )
    dist._all_gather_base(gathered, inputs["x1"])
    gathered_x1_scale = inputs.get("x1_scale")
    if gathered_x1_scale is not None and attrs.get("mc2_accuracy_mode") != "three_way":
        scale_shape = [gathered_x1_scale.shape[0] * world_size] + list(
            gathered_x1_scale.shape[1:]
        )
        gathered_scale = torch.empty(
            scale_shape, dtype=gathered_x1_scale.dtype, device=gathered_x1_scale.device
        )
        dist._all_gather_base(gathered_scale, gathered_x1_scale)
        gathered_x1_scale = gathered_scale
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    if str(attrs.get("x1_dtype", "int8")).lower() in (
        "fp16",
        "float16",
        "bf16",
        "bfloat16",
    ):
        output = torch.matmul(gathered, x2_eff)
        if inputs.get("bias") is not None:
            output = output + inputs["bias"]
    else:
        scale = inputs.get("x2_scale")
        if attrs.get("mc2_accuracy_mode") == "three_way":
            # FP8 per-tensor MC2 uses (G @ W + bias) * trunc19(s1*s2).
            # Two FP32 scales select the dynamic quant-matmul path, whose
            # FP32 bias is added AFTER dequantization. Encode the product to
            # select static T-T quantization and keep accumulator-bias semantics.
            scale = torch_npu.npu_trans_quant_param(
                (inputs["x1_scale"].float() * inputs["x2_scale"].float())
                .reshape(1)
                .contiguous()
            )
            gathered_x1_scale = None
        output = torch_npu.npu_quant_matmul(
            gathered,
            x2_eff,
            scale,
            offset=None,
            pertoken_scale=gathered_x1_scale,
            bias=inputs.get("bias"),
            output_dtype=out_dtype,
            group_sizes=list(attrs.get("group_sizes", [])),
        )
    if bool(attrs.get("gather_output", True)):
        return output, gathered
    return output


def mc2_cpu_golden(ctx, inputs, attrs):
    """AllGatherMatmulV2.get_cpu per-tensor FP8 reference."""
    x1 = inputs["x1"].float()
    gathered = torch.empty(
        (x1.shape[0] * ctx["world_size"], x1.shape[1]), dtype=torch.float32
    )
    ctx["dist"]._all_gather_base(gathered, x1)
    x2 = inputs["x2"].t() if attrs.get("is_trans_b", False) else inputs["x2"]
    out = torch.matmul(gathered, x2.float())
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
    # mc2_test scale_generate: truncate the product's low 13 mantissa bits.
    scale = inputs["x1_scale"].float() * inputs["x2_scale"].float()
    scale = (scale.contiguous().view(torch.int32) & -8192).view(torch.float32)
    out = out * scale.reshape(1)
    return (out, gathered) if attrs.get("gather_output", True) else out


def mc2_cpu_references(ctx, inputs, attrs):
    """Keep FP32 G and independently model the quantized cascade's final cast.

    Static per-tensor quant-matmul applies accumulator bias and the truncated
    scale before casting. All-gather itself only transports input values.
    """
    mathematical = mc2_cpu_golden(ctx, inputs, attrs)
    if attrs.get("mc2_accuracy_mode") != "three_way":
        return mathematical, None
    dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    if attrs.get("gather_output", True):
        output, gathered = mathematical
        return mathematical, (output.to(dtype).float(), gathered)
    return mathematical, mathematical.to(dtype).float()


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype = _torch_dtype(dtype_name)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(float(lo), float(hi))
            .to(dtype)
        )
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _torch_dtype(dtype_name: str):
    dtype = str(dtype_name).lower()
    if dtype in ("fp8_e4m3fn", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if dtype in ("fp8_e5m2", "float8_e5m2"):
        return torch.float8_e5m2
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype]
