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
Outlook Attention 算子 Torch Golden 参考实现

Outlook Attention 将每个像素到其 k x k 邻域的注意力权重通过 unfold 展开后
由线性层预测，再 fold 回原空间尺寸：

    v      = v_pj(x)                     # [B, attn_dim, H, W]
    v_unf  = Unfold(k, p, s)(v)
    attn   = attn_pj(avgpool(x))         # [B, L, k^2*head, k^2*head]
    attn   = softmax(attn * head_dim^-0.5, dim=-1)
    out    = Fold((attn @ v_unf))        # [B, attn_dim, H, W]
    y      = proj(out)

精度策略（与源实现一致）：权重按输入 dtype 给出，计算链整体升到 fp32 执行，
最后把结果 cast 回输入 dtype。源实现可用环境变量 REF_COMPUTE_DTYPE 覆盖该
compute dtype（默认 float32），本 golden 固定使用 float32。
"""


def outlook_attention(
    x: torch.Tensor,
    v_pj_weight: torch.Tensor,
    v_pj_bias: torch.Tensor,
    attn_pj_weight: torch.Tensor,
    attn_pj_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    num_heads: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 1,
) -> torch.Tensor:
    """
    Outlook Attention 前向

    Args:
        x: 输入特征图 [B, H, W, C]（NHWC）
        v_pj_weight: value 投影权重 [attn_dim, C]
        v_pj_bias: value 投影偏置 [attn_dim]
        attn_pj_weight: 注意力投影权重 [kernel_size**4 * num_heads, C]
        attn_pj_bias: 注意力投影偏置 [kernel_size**4 * num_heads]
        proj_weight: 输出投影权重 [C, attn_dim]
        proj_bias: 输出投影偏置 [C]
        num_heads: 注意力 head 数；attn_dim = (C // num_heads) * num_heads
        kernel_size: 局部窗口核大小（会被 clamp 到 min(H, W)）
        padding: 窗口 padding
        stride: 窗口步长

    Returns:
        输出特征图 [B, H, W, C]，dtype 与输入 x 一致
    """
    B, H, W, C = x.shape
    if B == 0 or H == 0 or W == 0 or C == 0:
        return x

    # 与源实现一致：按实际输入尺寸修正参数
    kernel_size = max(1, kernel_size)
    stride = max(1, stride)
    padding = max(0, padding)
    if kernel_size > H or kernel_size > W:
        kernel_size = min(kernel_size, H, W)
        if kernel_size < 1:
            kernel_size = 1
    if stride > H or stride > W:
        stride = 1
    if padding >= kernel_size:
        padding = kernel_size // 2

    dim = C
    attn_dim = (dim // num_heads) * num_heads
    if attn_dim == 0:
        attn_dim = num_heads

    # ===== fp32 精度升级：权重无损升到 fp32，计算链整体使用 compute dtype =====
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    v_pj_w = v_pj_weight.to(torch.float32)
    v_pj_b = v_pj_bias.to(torch.float32)
    attn_pj_w = attn_pj_weight.to(torch.float32)
    attn_pj_b = attn_pj_bias.to(torch.float32)
    proj_w = proj_weight.to(torch.float32)
    proj_b = proj_bias.to(torch.float32)

    h_grid = (H + 2 * padding - kernel_size) // stride + 1
    w_grid = (W + 2 * padding - kernel_size) // stride + 1
    if h_grid <= 0 or w_grid <= 0:
        return x.to(orig_dtype)

    v = F.linear(x, v_pj_w, v_pj_b).permute(0, 3, 1, 2)  # B, attn_dim, H, W
    v_unfold = F.unfold(v, kernel_size=kernel_size, padding=padding, stride=stride)
    L = v_unfold.shape[-1]
    head_dim = attn_dim // num_heads
    # unfold 产生的真实空间网格
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    if h_out * w_out != L:
        # 与源实现一致的兜底：按 H/W 比例由 L 反推网格
        ratio = H / max(1, W)
        w_out = int((L / ratio) ** 0.5)
        w_out = max(1, min(L, w_out))
        h_out = L // w_out
        while h_out * w_out != L and w_out > 1:
            w_out -= 1
            h_out = L // w_out
    v = v_unfold.reshape(B, num_heads, head_dim, kernel_size * kernel_size, L).permute(0, 1, 4, 3, 2)

    attn = x.permute(0, 3, 1, 2)  # B, C, H, W
    attn = F.avg_pool2d(attn, kernel_size=stride, stride=stride, ceil_mode=False)
    if attn.shape[2] != h_out or attn.shape[3] != w_out:
        attn = F.adaptive_avg_pool2d(attn, (h_out, w_out))
    attn = attn.permute(0, 2, 3, 1)  # B, h_out, w_out, C
    attn = F.linear(attn, attn_pj_w, attn_pj_b).reshape(
        B, L, num_heads, kernel_size * kernel_size, kernel_size * kernel_size
    ).permute(0, 2, 1, 3, 4)
    scale = head_dim ** -0.5
    attn = F.softmax(attn * scale, dim=-1)

    out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, attn_dim * kernel_size * kernel_size, L)
    # F.fold 是 Unfold 的逆：折回原始空间尺寸
    out = F.fold(out, output_size=(H, W), kernel_size=kernel_size,
                 padding=padding, stride=stride)
    out = F.linear(out.permute(0, 2, 3, 1), proj_w, proj_b)  # B, H, W, C
    return out.to(orig_dtype)
