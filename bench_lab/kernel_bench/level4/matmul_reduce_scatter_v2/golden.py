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


from typing import Any, Dict, Optional

import torch


def matmul_reduce_scatter_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> torch.Tensor:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    if x1_scale is not None or x2_scale is not None:
        out = _apply_dequant_scale(out, x1_scale, x2_scale)
    out = out * int(world_size)
    chunk = out.shape[0] // int(world_size)
    return out[:chunk].to(_torch_dtype(out_dtype))


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
    if not _is_quant_case(attrs):
        # A5 TP slices must differ across ranks; x1/x2 share the rank seed.
        seed += rank

    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0], seed, device)
    x2_seed = seed if weight_same else seed + rank * 2
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1], x2_seed, device)

    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2], rank * 3, device)

    if not _is_quant_case(attrs):
        return {"x1": x1, "x2": x2, "bias": bias, "x1_scale": None, "x2_scale": None}

    x1_scale_index = 3 if bool(attrs.get("is_bias", False)) else 2
    x2_scale_index = x1_scale_index + 1
    x1_scale = _make_tensor(
        shapes[x1_scale_index],
        dtypes[x1_scale_index],
        ranges[x1_scale_index],
        seed * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        seed * 13,
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
        reduce_op=str(attrs.get("reduce_op", "sum")),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
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
    """Distributed CPU-semantic golden following mc2_test's get_cpu path.

    Compute matmul and dequant in float32 for precision, then cast to the
    operator's output dtype before all_reduce so the communication precision
    matches the fused MC2 kernel (which communicates in fp16/bf16).
    """
    dist = ctx["dist"]
    device = ctx["device"]
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    # Compute matmul in float32 on CPU to avoid int8→float32 Cast kernel issues
    if attrs.get("mc2_accuracy_mode") == "three_way":
        if _is_quant_case(attrs):
            raise ValueError(
                "This three-way task currently covers non-quantized A5 cases only"
            )
        # Device cascade must actually execute matmul on NPU, not CPU.
        local = torch.matmul(inputs["x1"], x2_eff)
        if inputs.get("bias") is not None:
            local = local + inputs["bias"]
        chunk = local.shape[0] // int(ctx["world_size"])
        result = torch.empty((chunk, local.shape[1]), dtype=local.dtype, device=device)
        dist._reduce_scatter_base(result, local.contiguous(), op=dist.ReduceOp.SUM)
        return result

    # on machines where the CANN CastAiCore JSON config is not installed.
    x1_cpu = inputs["x1"].cpu().float()
    x2_cpu = x2_eff.cpu().float()
    bias_cpu = (
        inputs.get("bias").cpu().float() if inputs.get("bias") is not None else None
    )
    mm_out = torch.matmul(x1_cpu, x2_cpu)
    if bias_cpu is not None:
        mm_out = mm_out + bias_cpu
    if _is_quant_case(attrs):
        x1_scale_cpu = (
            inputs.get("x1_scale").cpu() if inputs.get("x1_scale") is not None else None
        )
        x2_scale_cpu = (
            inputs.get("x2_scale").cpu() if inputs.get("x2_scale") is not None else None
        )
        mm_out = _apply_dequant_scale(mm_out, x1_scale_cpu, x2_scale_cpu)
    # Cast to output dtype before all_reduce to match candidate's communication precision
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    mm_out = mm_out.to(out_dtype).to(device)
    dist.all_reduce(mm_out, op=dist.ReduceOp.SUM)
    chunk = mm_out.shape[0] // int(ctx["world_size"])
    return mm_out.narrow(0, int(ctx["rank"]) * chunk, chunk)


def mc2_cpu_golden(ctx, inputs, attrs):
    return _cpu_references(ctx, inputs, attrs, staged=False)[0]


def mc2_cpu_references(ctx, inputs, attrs):
    """Return mathematical G and an independent, stage-rounded health reference.

    Only two ranks have a single, unambiguous low-precision reduction step.
    Other world sizes retain the generic health check; do not assume a reduction
    tree/order for them. Neither result depends on NPU or candidate outputs.
    """
    return _cpu_references(ctx, inputs, attrs, staged=int(ctx["world_size"]) == 2)


def _cpu_references(ctx, inputs, attrs, staged):
    if _is_quant_case(attrs):
        raise ValueError(
            "CPU three-way reference currently covers A5 non-quantized cases"
        )
    if attrs.get("reduce_op", "sum") != "sum":
        raise ValueError("CPU three-way reference only supports sum")
    x2 = inputs["x2"].t() if attrs.get("is_trans_b", False) else inputs["x2"]
    out = torch.matmul(inputs["x1"].float(), x2.float())
    # Share the expensive FP32 matmul, not its rounded values, with mathematical G.
    dtype = inputs["x1"].dtype
    rounded = out.to(dtype).float() if staged else None
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
        if staged:
            rounded = (rounded + inputs["bias"].float()).to(dtype).float()
    ctx["dist"].all_reduce(out, op=ctx["dist"].ReduceOp.SUM)
    if staged:
        # Communicate FP32 over Gloo, then emulate the one FP16/BF16 sum rounding.
        ctx["dist"].all_reduce(rounded, op=ctx["dist"].ReduceOp.SUM)
        rounded = rounded.to(dtype).float()
    chunk = out.shape[0] // int(ctx["world_size"])
    start = int(ctx["rank"]) * chunk
    return (
        out.narrow(0, start, chunk),
        rounded.narrow(0, start, chunk) if staged else None,
    )


def _is_quant_case(attrs: Dict[str, Any]) -> bool:
    x1_dtype = str(attrs.get("x1_dtype", "int8")).lower()
    x2_dtype = str(attrs.get("x2_dtype", "int8")).lower()
    non_quant = {"fp16", "float16", "bf16", "bfloat16"}
    return x1_dtype not in non_quant or x2_dtype not in non_quant


def _apply_dequant_scale(
    out: torch.Tensor,
    x1_scale: Optional[torch.Tensor],
    x2_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply MC2 INT8 per-token/per-channel dequant scale.

    mc2_test truncates the float32 scale product with ``0xffffe000`` before it
    multiplies the matmul result.  Keep that behavior here, while leaving final
    dtype rounding to cann-bench's normal compare path.
    """
    if x1_scale is None and x2_scale is None:
        return out

    rows, cols = int(out.shape[-2]), int(out.shape[-1])
    device = out.device
    x1 = _scale_tensor_or_ones(x1_scale, rows, device)
    x2 = _scale_tensor_or_ones(x2_scale, cols, device)

    if x1.numel() == rows and x2.numel() == cols:
        result = out.clone()
        row_scale = x1.float().reshape(rows, 1)
        col_scale = x2.float().reshape(1, cols)
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            scale = _truncate_hw_scale(row_scale[start:end] * col_scale)
            result[start:end] = result[start:end] * scale
        return result

    scale = _broadcast_dequant_scale(x1, x2, rows, cols)
    return out * _truncate_hw_scale(scale)


def _scale_tensor_or_ones(
    scale: Optional[torch.Tensor], size: int, device: torch.device
) -> torch.Tensor:
    if scale is None:
        return torch.ones(size, dtype=torch.float32, device=device)
    return scale.float().to(device)


def _broadcast_dequant_scale(
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    x1 = x1_scale.float()
    x2 = x2_scale.float()
    if x1.numel() == 1 and x2.numel() == 1:
        return x1.reshape(1) * x2.reshape(1)
    if x1.numel() == rows and x2.numel() == 1:
        return x1.reshape(rows, 1) * x2.reshape(1)
    if x1.numel() == 1 and x2.numel() == cols:
        return x1.reshape(1) * x2.reshape(1, cols)
    if list(x1.shape) == [rows, 1] and x2.numel() == cols:
        return x1 * x2.reshape(1, cols)
    if x1.numel() == rows and list(x2.shape) == [1, cols]:
        return x1.reshape(rows, 1) * x2
    if list(x1.shape) == [rows, cols] and x2.numel() == 1:
        return x1 * x2.reshape(1)
    raise ValueError(
        "Unsupported dequant scale shapes: "
        f"x1_scale={tuple(x1_scale.shape)}, x2_scale={tuple(x2_scale.shape)}, output=({rows}, {cols})"
    )


def _truncate_hw_scale(scale: torch.Tensor) -> torch.Tensor:
    bits = scale.float().view(torch.int32)
    bits &= torch.tensor(
        -8192, dtype=torch.int32, device=scale.device
    )  # 0xFFFFE000 as signed int32
    return bits.view(torch.float32)


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
