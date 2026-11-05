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


from typing import Optional

import torch


def _dtype_from_name(dtype_name: str):
    dtype = str(dtype_name).lower()
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp32", "float32"):
        return torch.float32
    if dtype in ("int32",):
        return torch.int32
    if dtype == "int8":
        return torch.int8
    if dtype in ("int64",):
        return torch.int64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    if shape is None:
        return None
    torch.manual_seed(int(seed))
    dtype = _dtype_from_name(dtype_name)
    if value_range is None:
        value_range = [0, 1]
    lo, hi = value_range
    if dtype.is_floating_point:
        tensor = (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(float(lo), float(hi))
            .to(dtype)
        )
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def mc2_make_rank_inputs(ctx, case_payload):
    """Preserve legacy seeds; distinguish A5 rank inputs for communication checks."""
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload.get("value_ranges") or []
    attrs = case_payload["attrs"]
    rank = int(ctx["rank"])
    device = ctx["device"]
    seed = int(attrs.get("seed", 1))
    if attrs.get("quantized", False):
        seed += rank

    x1_seed = seed
    if (
        not attrs.get("quantized", False)
        and attrs.get("mc2_accuracy_mode") == "three_way"
    ):
        x1_seed += rank * 1000003
    x1 = _make_tensor(
        shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
    )
    x2 = _make_tensor(
        shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device
    )
    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            seed * 7,
            device,
        )
    inputs = {"x1": x1, "x2": x2, "bias": bias}
    if attrs.get("quantized", False):
        for index, name in ((3, "dequant_scale"), (4, "pertoken_scale"), (5, "x3")):
            inputs[name] = _make_tensor(
                shapes[index], dtypes[index], ranges[index], seed * (index + 3), device
            )
    return inputs


def matmul_all_reduce(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
) -> torch.Tensor:
    """Single-process CPU reference for the MC2 matmul all-reduce task."""
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    # Single-process fallback assumes all ranks have identical seeded inputs.
    return out * int(world_size)


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed HCCL baseline for MatmulAllReduce.

    Use torch.addmm for fused matmul+bias to match the MC2 kernel's
    cube-unit behavior (bias added in fp32 accumulator before truncation).
    """
    x2 = inputs["x2"]
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    bias = inputs.get("bias")
    if attrs.get("quantized", False):
        shape = inputs["x1"].shape
        out = (
            ctx["torch_npu"]
            .npu_quant_matmul(
                inputs["x1"].reshape(-1, shape[-1]),
                x2_eff,
                inputs["dequant_scale"].reshape(-1),
                pertoken_scale=inputs["pertoken_scale"].reshape(-1),
                bias=bias,
                output_dtype=_dtype_from_name(attrs["out_dtype"]),
            )
            .reshape(*shape[:-1], x2_eff.shape[-1])
        )
        if inputs.get("x3") is not None:
            out = out + inputs["x3"]
        _all_reduce_sum(ctx, out, attrs)
        return out
    if bias is not None:
        out = torch.addmm(bias, inputs["x1"], x2_eff)
    else:
        out = torch.matmul(inputs["x1"], x2_eff)
    _all_reduce_sum(ctx, out, attrs)
    return out


def _all_reduce_sum(ctx, tensor: torch.Tensor, attrs):
    reduce_op = str(attrs.get("reduce_op", "sum")).lower()
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    ctx["dist"].all_reduce(tensor, op=ctx["dist"].ReduceOp.SUM)


def _quant_matmul_int32(inputs, attrs):
    """Accumulate integer bias before dequantization, shared by G and H."""
    x2 = inputs["x2"].t() if attrs.get("is_trans_b", False) else inputs["x2"]
    out = torch.matmul(inputs["x1"].to(torch.int32), x2.to(torch.int32))
    if inputs.get("bias") is not None:
        out = out + inputs["bias"]
    return out


def _quant_math_local(accumulator, inputs):
    """Preserve the mathematical golden's FP32 token-then-channel order."""
    out = accumulator
    shape = out.shape
    out = (
        out.float().reshape(-1, shape[-1])
        * inputs["pertoken_scale"].float().reshape(-1, 1)
    ).reshape(shape)
    out = out * inputs["dequant_scale"].float().reshape(-1)
    if inputs.get("x3") is not None:
        out = out + inputs["x3"].float()
    return out


def mc2_cpu_golden(ctx, inputs, attrs):
    """QuantMatmulAllReduceV4.get_cpu INT8 path: INT32 GEMM then FP32 scales."""
    if not attrs.get("quantized", False):
        fp32 = {
            key: value.float() if value is not None else None
            for key, value in inputs.items()
        }
        return mc2_distributed_golden(ctx, fp32, attrs)
    out = _quant_math_local(_quant_matmul_int32(inputs, attrs), inputs)
    _all_reduce_sum(ctx, out, attrs)
    return out


def mc2_cpu_references(ctx, inputs, attrs):
    """Return FP32 math and an independent two-rank, stage-rounded cascade.

    Native addmm fuses bias before rounding. INT8 quant matmul instead rounds
    after dequantization and again after optional x3 addition. HCCL SUM rounds
    the two low-precision local outputs once more. H is computed only from
    CPU inputs for reference health; candidate scoring still uses unchanged G.
    More than two ranks require a separately verified reduction-tree model.
    """
    if int(ctx["world_size"]) != 2:
        return mc2_cpu_golden(ctx, inputs, attrs), None

    if attrs.get("quantized", False):
        dtype = _dtype_from_name(attrs["out_dtype"])
        if dtype not in (torch.float16, torch.bfloat16) or any(
            inputs[name].dtype != torch.int8 for name in ("x1", "x2")
        ):
            return mc2_cpu_golden(ctx, inputs, attrs), None
        accumulator = _quant_matmul_int32(inputs, attrs)
        local = _quant_math_local(accumulator, inputs)
        # Quant matmul scales channels before tokens and materializes D output.
        shape = accumulator.shape
        rounded = accumulator.float() * inputs["dequant_scale"].float().reshape(-1)
        rounded = (
            rounded.reshape(-1, shape[-1])
            * inputs["pertoken_scale"].float().reshape(-1, 1)
        ).reshape(shape)
        rounded = rounded.to(dtype).float()
        if inputs.get("x3") is not None:
            rounded = (rounded + inputs["x3"].float()).to(dtype).float()
        _all_reduce_sum(ctx, local, attrs)
        _all_reduce_sum(ctx, rounded, attrs)
        return local, rounded.to(dtype).float()

    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    local = torch.matmul(inputs["x1"].float(), x2_eff.float())
    bias = inputs.get("bias")
    if bias is not None:
        local = local + bias.float()
    dtype = inputs["x1"].dtype
    rounded = local.to(dtype).float()
    _all_reduce_sum(ctx, local, attrs)
    _all_reduce_sum(ctx, rounded, attrs)
    return local, rounded.to(dtype).float()


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    """Call task candidate with the MatmulAllReduce argument contract."""
    if attrs.get("quantized", False):
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
            dequant_scale=inputs["dequant_scale"],
            pertoken_scale=inputs["pertoken_scale"],
            x3=inputs.get("x3"),
            out_dtype=attrs["out_dtype"],
        )
    try:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
        )
    except TypeError:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            attrs.get("reduce_op", "sum"),
            inputs.get("bias"),
            bool(attrs.get("is_trans_b", False)),
        )
