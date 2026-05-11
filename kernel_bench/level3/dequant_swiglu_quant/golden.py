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

"""DequantSwigluQuant Golden 参考实现（精简版，5 参数，仅动态量化）。

签名是 `torch_npu.npu_dequant_swiglu_quant` 在去掉 group_index / swiglu_mode /
bias / quant_offset 后并固定 quant_mode=1（动态 per-token）的子集：

  dequant_swiglu_quant(x, weight_scale, activation_scale, quant_scale,
                       activate_left) -> (y, scale)

x 支持 int32（W8A8 反量化路径，必配 weight_scale + activation_scale）和
bfloat16（直接 SwiGLU 路径，weight_scale / activation_scale 必须 None）。

固定动态量化的原因：CANN 850 的静态量化路径在实测中表现为 identity scale
且 scale 返回值未初始化（垃圾数据），不具备生产可用性，因此从 kernel_bench
spec 中移除。
"""
from typing import Optional, Tuple
import torch


def dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    activate_left: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: [TokensNum, 2H] int32 / bfloat16 / float16
        weight_scale: [1, 2H] float32；x=int32 时必须；x=bfloat16/float16 时必须 None
        activation_scale: [TokensNum] float32；x=int32 时必须；x=bfloat16/float16 时必须 None
        quant_scale: [1, H] float32；smooth quant 系数（可选）
        activate_left: False = SiLU(B)*A；True = SiLU(A)*B

    Returns:
        y: [TokensNum, H] int8（每行独立 per-token scale 量化）
        scale: [TokensNum] float32（每 token 的 scale）
    """
    # ---- Step 1: 反量化 ----
    if x.dtype == torch.int32:
        assert weight_scale is not None and activation_scale is not None, \
            "x=int32 时必须提供 weight_scale 和 activation_scale"
        # weight_scale 形状 [1, 2H]；activation_scale 形状 [TokensNum]
        dequant_out = x.float() * weight_scale.float()                   # broadcast → [TokensNum, 2H]
        dequant_out = dequant_out * activation_scale.float().unsqueeze(-1)
    elif x.dtype in (torch.bfloat16, torch.float16):
        assert weight_scale is None and activation_scale is None, \
            f"x={x.dtype} 时 weight_scale / activation_scale 必须为 None"
        dequant_out = x.float()
    else:
        raise ValueError(f"x dtype must be int32, bfloat16, or float16, got {x.dtype}")

    # ---- Step 2: SwiGLU 激活 ----
    last_dim = dequant_out.shape[-1]
    assert last_dim % 2 == 0, f"x 最后一维必须为偶数, got {last_dim}"
    half = last_dim // 2
    A = dequant_out[..., :half]    # 左半
    B = dequant_out[..., half:]    # 右半
    silu = torch.nn.functional.silu
    if activate_left:
        # 官方 API: True → swish(split[0]) * split[1] = SiLU(A) * B
        swiglu_out = silu(A) * B
    else:
        # 官方 API: False → swish(split[1]) * split[0] = SiLU(B) * A
        swiglu_out = silu(B) * A

    # ---- Step 3: smooth quant 系数（可选） ----
    if quant_scale is not None:
        swiglu_out = swiglu_out * quant_scale.float()    # broadcast [1, H] → [TokensNum, H]

    # ---- Step 4: 动态量化（per-token，int8） ----
    # 每行独立 max → 每行独立 scale；与 quant_mode=1 一致
    max_per_row = swiglu_out.abs().amax(dim=-1)              # [TokensNum]
    s = (max_per_row.float() / 127.0).clamp_min(1e-12)       # avoid div 0
    y = torch.clamp((swiglu_out.float() / s.unsqueeze(-1)).round(), -128, 127).to(torch.int8)
    scale = s.to(torch.float32)
    return y, scale
