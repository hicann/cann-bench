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

import math
import torch
import torch.nn.functional as F

"""
EngramGateFusion Torch Golden 参考实现

DeepSeek Engram 模块 (arXiv:2601.07372) 的 7 步融合算子：
  1-2. 双路 RMSNorm
  3.   缩放点积门控
  4.   sqrt + sigmoid 非线性门控
  5.   门控广播乘法
  6.   ShortConv（RMSNorm → 深度可分离扩张 Conv1d → SiLU），支持 Decode 状态缓存
  7.   残差相加

精度策略：
  - 标杆计算时（测试框架传入 FP64 输入）：全程 FP64 计算，输出截断到 BF16
  - 算子计算时（NPU 上 BF16 输入）：升精度到 FP32 计算，输出截断到 BF16
  - 计算精度不低于 FP32（确保数值稳定）

返回 (output, conv_state_out)。
"""


def engram_gate_fusion(
    keys: torch.Tensor,
    hidden_states: torch.Tensor,
    value: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor = None,
    hc_mult: int = 4,
    hidden_size: int = 1024,
    kernel_size: int = 4,
    dilation: int = 3,
    norm_eps: float = 1e-5,
):
    """EngramGateFusion: dual RMSNorm gate + broadcast multiply + ShortConv + residual.

    Args:
        keys:             [B, L, HC, D]
        hidden_states:    [B, L, HC, D]
        value:            [B, L, D]
        norm1_weight:     [HC, D], float32
        norm2_weight:     [HC, D], float32
        conv_norm_weight: [HC, D], float32
        conv_weight:      [HC*D, 1, K], float32
        conv_state:       [B, HC*D, (K-1)*dilation], 输入精度 or None

    Returns:
        output: [B, L, HC, D]
        conv_state_out: [B, HC*D, (K-1)*dilation]
    """
    B, L, HC, D = keys.shape
    state_len = (kernel_size - 1) * dilation

    assert HC == hc_mult and D == hidden_size
    assert hidden_states.shape == (B, L, HC, D)
    assert value.shape == (B, L, D)
    assert norm1_weight.shape == (HC, D)
    assert norm2_weight.shape == (HC, D)
    assert conv_norm_weight.shape == (HC, D)
    assert conv_weight.shape == (HC * D, 1, kernel_size)
    if conv_state is not None:
        assert conv_state.shape == (B, HC * D, state_len)

    # 计算精度：所有路径下都用 FP32 做内部计算（消除半精度累积误差）；
    # 输出再转回 input_dtype。
    input_dtype = keys.dtype
    output_dtype = input_dtype
    compute_dtype = torch.float32

    def rms_norm(x, w, eps):
        """RMSNorm：输入升精度，全程高精度计算"""
        x_hp = x.to(compute_dtype)
        w_view = w.view(1, 1, HC, D).to(compute_dtype)
        rms = x_hp.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return x_hp / rms * w_view

    # ============================================================
    # Step 1 & 2: 双路 RMSNorm（高精度计算）
    # ============================================================
    normed_keys = rms_norm(keys, norm1_weight, norm_eps)
    normed_qs = rms_norm(hidden_states, norm2_weight, norm_eps)

    # ============================================================
    # Step 3: 缩放点积门控（高精度累加）
    # ============================================================
    raw_gate = (normed_keys * normed_qs).sum(dim=-1) / math.sqrt(D)

    # ============================================================
    # Step 4: sqrt + sigmoid 非线性门控（高精度）
    # ============================================================
    safe_abs = raw_gate.abs().clamp_min(1e-6)
    gate = torch.sigmoid(safe_abs.sqrt() * raw_gate.sign()).unsqueeze(-1)

    # ============================================================
    # Step 5: 门控广播乘法（高精度）
    # ============================================================
    value_hp = value.to(compute_dtype)
    value_gated = gate * value_hp.unsqueeze(2)

    # ============================================================
    # Step 6: ShortConv（高精度）
    # ============================================================
    normed_vg = rms_norm(value_gated, conv_norm_weight, norm_eps)
    x = normed_vg.permute(0, 2, 3, 1).reshape(B, HC * D, L)

    if conv_state is None:
        x_cat = F.pad(x, (state_len, 0))
    else:
        x_cat = torch.cat([conv_state.to(compute_dtype), x], dim=-1)

    # conv_state_out：高精度计算后截断到 BF16
    conv_state_out_hp = x_cat[:, :, -state_len:].contiguous() if state_len > 0 else x_cat[:, :, :0]

    # Conv1d：权重升精度
    y = F.conv1d(
        x_cat,
        conv_weight.to(compute_dtype),
        dilation=dilation,
        groups=HC * D,
    )
    y = F.silu(y)
    conv_out = y.reshape(B, HC, D, L).permute(0, 3, 1, 2)

    # ============================================================
    # Step 7: 残差相加（高精度）
    # ============================================================
    output_hp = value_gated + conv_out

    # ============================================================
    # 输出截断：高精度 → BF16（预期输出精度）
    # ============================================================
    output = output_hp.to(output_dtype)
    conv_state_out = conv_state_out_hp.to(output_dtype)

    return output, conv_state_out