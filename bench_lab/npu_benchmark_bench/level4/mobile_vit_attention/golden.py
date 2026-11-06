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
from typing import List

"""
MobileViTAttention 算子 Torch Golden 参考实现

MobileViT Attention block（原版语义）：
    y0 = Conv2d(C->C, k) -> Conv2d(C->dim, 1x1)              # 无激活的局部表征
    unfold: (b, dim, nh*ph, nw*pw) -> (b, ph*pw, nh*nw, dim)
    DEPTH=3 x [ 残差(PreNorm(LN -> 8 头 attention -> proj))
                残差(PreNorm(LN -> Linear(dim->1024) -> SiLU -> Linear(1024->dim))) ]
    fold 回 (b, dim, h, w)
    y = Conv2d(dim->C, 1x1) -> cat([x, y]) -> Conv2d(2C->C, k)

说明：
- 源码把 DEPTH=3 层的同构权重按 depth 维堆叠后传入；本 golden 用 List[Tensor]
  表达为每层一个张量（列表长度为 3），展开后逐层计算，数学上完全等价。
- 原版 dropout 全为 0.0（eval 恒等），不纳入。
- 全程使用输入 dtype 计算，输出 dtype 与输入一致（源 forward 的降精度出口
  目标就是 case 标称 dtype，与本任务 case 的输入 dtype 相同）。
"""

DEPTH = 3
HEADS = 8
HEAD_DIM = 64
MLP_DIM = 1024
INNER_DIM = HEADS * HEAD_DIM  # 512


def mobile_vit_attention(
    x: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    conv2_w: torch.Tensor,
    conv2_b: torch.Tensor,
    ln_att_w: List[torch.Tensor],
    ln_att_b: List[torch.Tensor],
    qkv_w: List[torch.Tensor],
    out_w: List[torch.Tensor],
    out_b: List[torch.Tensor],
    ln_ffn_w: List[torch.Tensor],
    ln_ffn_b: List[torch.Tensor],
    ff1_w: List[torch.Tensor],
    ff1_b: List[torch.Tensor],
    ff2_w: List[torch.Tensor],
    ff2_b: List[torch.Tensor],
    conv3_w: torch.Tensor,
    conv3_b: torch.Tensor,
    conv4_w: torch.Tensor,
    conv4_b: torch.Tensor,
    patch_size: int,
    kernel_size: int,
    dim: int,
) -> torch.Tensor:
    """
    MobileViT Attention 块

    Args:
        x: 输入特征图 [B, C, H, W]，H/W 可被 patch_size 整除
        conv1_w/conv1_b: 局部表征 kxk 卷积权重 [C, C, k, k] / 偏置 [C]
        conv2_w/conv2_b: 局部表征 1x1 卷积权重 [dim, C, 1, 1] / 偏置 [dim]
        ln_att_w/ln_att_b: 3 层注意力 PreNorm 的 LayerNorm affine，每个 [dim]
        qkv_w: 3 层 qkv 融合线性权重（无 bias），每个 [3*512, dim]
        out_w/out_b: 3 层注意力输出投影权重 [dim, 512] / 偏置 [dim]
        ln_ffn_w/ln_ffn_b: 3 层 FFN PreNorm 的 LayerNorm affine，每个 [dim]
        ff1_w/ff1_b: 3 层 FFN 第一层线性权重 [1024, dim] / 偏置 [1024]
        ff2_w/ff2_b: 3 层 FFN 第二层线性权重 [dim, 1024] / 偏置 [dim]
        conv3_w/conv3_b: Fusion 前 1x1 卷积权重 [C, dim, 1, 1] / 偏置 [C]
        conv4_w/conv4_b: Fusion 融合 kxk 卷积权重 [C, 2C, k, k] / 偏置 [C]
        patch_size: unfold/fold 的 patch 边长
        kernel_size: conv1/conv4 的卷积核大小 k
        dim: Transformer 隐层维度

    Returns:
        输出特征图 [B, C, H, W]，dtype 与 x 相同
    """
    b, c, h, w = x.shape
    dim = conv2_w.shape[0]
    k = conv1_w.shape[-1]
    ph = pw = patch_size
    nh, nw = h // ph, w // pw
    p, n = ph * pw, nh * nw

    # Local Representation（原版无中间激活）
    y = F.conv2d(x, conv1_w, conv1_b, padding=k // 2)
    y = F.conv2d(y, conv2_w, conv2_b)

    # unfold
    y = y.reshape(b, dim, nh, ph, nw, pw).permute(0, 3, 5, 2, 4, 1).reshape(b, p, n, dim)

    # Global Representation: depth=3 x (PreNorm-Attn + PreNorm-FFN), 残差
    for l in range(DEPTH):
        a = F.layer_norm(y, (dim,), ln_att_w[l], ln_att_b[l])
        qkv = F.linear(a, qkv_w[l]).chunk(3, dim=-1)
        q, kk, v = [t.reshape(b, p, n, HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4) for t in qkv]
        attn = torch.softmax(torch.matmul(q, kk.transpose(-1, -2)) * (HEAD_DIM ** -0.5), dim=-1)
        o = torch.matmul(attn, v).permute(0, 1, 3, 2, 4).reshape(b, p, n, INNER_DIM)
        y = y + F.linear(o, out_w[l], out_b[l])

        f = F.layer_norm(y, (dim,), ln_ffn_w[l], ln_ffn_b[l])
        f = F.silu(F.linear(f, ff1_w[l], ff1_b[l]))
        y = y + F.linear(f, ff2_w[l], ff2_b[l])

    # fold
    y = y.reshape(b, ph, pw, nh, nw, dim).permute(0, 5, 3, 1, 4, 2).reshape(b, dim, h, w)

    # Fusion
    y = F.conv2d(y, conv3_w, conv3_b)
    y = torch.cat([x, y], 1)
    out = F.conv2d(y, conv4_w, conv4_b, padding=k // 2)
    return out
