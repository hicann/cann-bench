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


import torch
from typing import Optional, Union, Tuple


def all_gather_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process CPU reference for the MC2 all-gather matmul task.

    The real benchmark path for cases with ``mc2_distributed: true`` uses
    ``MC2DistributedEvaluator`` and HCCL. This function keeps the task usable
    for metadata inspection and golden-only smoke checks.
    """
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0)
    out = torch.matmul(gathered.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    out = out.to(x1.dtype)
    if gather_output:
        return out, gathered
    return out


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed golden hook used by MC2DistributedEvaluator."""
    dist = ctx["dist"]
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    bias = inputs.get("bias")
    world_size = int(ctx["world_size"])
    gather_shape = [x1.shape[0] * world_size] + list(x1.shape[1:])
    gathered = torch.empty(gather_shape, dtype=x1.dtype, device=x1.device)
    dist._all_gather_base(gathered, x1)
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    out = torch.matmul(gathered, x2_eff)
    if bias is not None:
        out = out + bias
    if bool(attrs.get("gather_output", False)):
        return out, gathered
    return out


def mc2_cpu_golden(ctx, inputs, attrs):
    """CPU/Gloo FP32 reference, without intermediate output rounding."""
    fp32 = {
        key: value.float() if value is not None else None
        for key, value in inputs.items()
    }
    return mc2_distributed_golden(ctx, fp32, attrs)


def mc2_cpu_references(ctx, inputs, attrs):
    """Return FP32 math and an independent model of the native rounding stages.

    AllGather copies the input exactly. The native cascade rounds the matmul
    output to the input dtype, then rounds again after optional bias addition.
    Only reference health uses this second result; candidate error metrics
    retain the unrounded FP32 mathematical reference.
    """
    dtype = inputs["x1"].dtype
    if int(ctx["world_size"]) != 2 or dtype not in (torch.float16, torch.bfloat16):
        return mc2_cpu_golden(ctx, inputs, attrs), None

    x1 = inputs["x1"].float()
    gathered = torch.empty(
        (x1.shape[0] * int(ctx["world_size"]), x1.shape[1]),
        dtype=torch.float32,
        device=x1.device,
    )
    ctx["dist"]._all_gather_base(gathered, x1)
    weight = inputs["x2"].float()
    if bool(attrs.get("is_trans_b", False)):
        weight = weight.t()
    mathematical = torch.matmul(gathered, weight)
    staged = mathematical.to(dtype).float()
    bias = inputs.get("bias")
    if bias is not None:
        mathematical = mathematical + bias.float()
        staged = (staged + bias.float()).to(dtype).float()
    if bool(attrs.get("gather_output", False)):
        return (mathematical, gathered), (staged, gathered)
    return mathematical, staged
