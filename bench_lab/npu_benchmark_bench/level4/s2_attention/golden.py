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
import torch.nn.functional as F

"""
S2 Attention 算子 Torch Golden 参考实现

S2 (Spatial Shift + Split) Attention：先做通道 3 倍扩展投影，再把通道切成 3 份，
其中两份分别做两种方向的 spatial shift，第三份保持不变；三份分支经 Split Attention
（全局平均 -> MLP -> GELU -> MLP -> softmax 权重）加权融合，最后经通道投影回原维度。

    1. x -> [B, W, H, C] -> Linear(C, 3C)
    2. x1 = shift1(x[..., :C]); x2 = shift2(x[..., C:2C]); x3 = x[..., 2C:]
    3. a = softmax(Linear(3C->C) -> GELU -> Linear(C->3C))   # 每个分支每个通道的权重
    4. y = Linear(C->C)(sum_k a_k * x_k)

注意：源实现中全局平均与 softmax 会先升 fp32 再 cast 回输入 dtype，本 golden 保持一致；
其余计算保持输入 dtype（源实现不做整体 fp32 升精度）。
"""


def _spatial_shift1(x: torch.Tensor) -> torch.Tensor:
    """沿 W/H 两个方向分别平移部分通道（源实现 spatial_shift1）。"""
    b, w, h, c = x.shape
    out = x.clone()
    out[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    out[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    out[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    out[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return out


def _spatial_shift2(x: torch.Tensor) -> torch.Tensor:
    """沿 H/W 两个方向分别平移部分通道（源实现 spatial_shift2）。"""
    b, w, h, c = x.shape
    out = x.clone()
    out[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    out[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    out[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    out[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return out


def s2_attention(
    x: torch.Tensor,
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    mlp3_weight: torch.Tensor,
    mlp3_bias: torch.Tensor,
    mlp4_weight: torch.Tensor,
    mlp4_bias: torch.Tensor,
) -> torch.Tensor:
    """
    S2 Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        mlp1_weight: 扩展投影权重 [3C, C]
        mlp1_bias: 扩展投影偏置 [3C]
        mlp2_weight: 输出投影权重 [C, C]
        mlp2_bias: 输出投影偏置 [C]
        mlp3_weight: Split Attention 第一层权重 [C, C]
        mlp3_bias: Split Attention 第一层偏置 [C]
        mlp4_weight: Split Attention 第二层权重 [3C, C]
        mlp4_bias: Split Attention 第二层偏置 [3C]

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x

    x = x.permute(0, 3, 2, 1).contiguous()  # B,C,H,W -> B,W,H,C
    x = F.linear(x, mlp1_weight, mlp1_bias)  # B,W,H,3C
    x1 = _spatial_shift1(x[:, :, :, :c])
    x2 = _spatial_shift2(x[:, :, :, c:2 * c])
    x3 = x[:, :, :, 2 * c:]
    x_all = torch.stack([x1, x2, x3], 1)  # B,3,W,H,C

    # ---------- Split Attention ----------
    B, k, W, H, C_ = x_all.shape
    x_all = x_all.view(B, k, -1, C_)  # B,3,N,C
    # 分支/Token 维平均在 fp32 下进行，避免 fp16 大 N 溢出
    a = x_all.float().mean(dim=(1, 2)).to(x.dtype)  # B,C
    a = F.linear(a, mlp3_weight, mlp3_bias)
    a = F.gelu(a)
    a = F.linear(a, mlp4_weight, mlp4_bias).view(B, 3, C_)  # B,3,C
    a = a.float().softmax(dim=1).to(x.dtype)  # B,3,C
    out = (a.unsqueeze(-2) * x_all).sum(1).view(B, W, H, C_)  # B,W,H,C
    out = F.linear(out, mlp2_weight, mlp2_bias)
    out = out.permute(0, 3, 2, 1)  # B,W,H,C -> B,C,H,W
    return out
