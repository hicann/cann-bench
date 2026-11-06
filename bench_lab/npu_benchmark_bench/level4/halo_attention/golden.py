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
HaloAttention 算子 Torch Golden 参考实现

Halo Attention：基于 block 的局部自注意力，KV 通过 unfold 提取带 halo 的邻域块。

    q_inp = blockify(x)                                      # (B*num_blocks, b*b, C)
    kv_inp = unfold(x, k=b+2*halo, stride=b, padding=halo)    # (B*num_blocks, (b+2*halo)^2, C)
    q = to_q(q_inp);  k, v = chunk(to_kv(kv_inp), 2, dim=-1)
    attn = softmax((q * dim_head^-0.5) @ k^T)
    y = to_out(merge(attn @ v))

源实现全程使用输入 dtype 计算（softmax 不做 fp32 上转），本 golden 保持一致；
b / c / h / w 任一为 0 时源实现直接返回输入，本 golden 同样提前返回。
"""


def halo_attention(
    x: torch.Tensor,
    q_w: torch.Tensor,
    kv_w: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    block_size: int,
    halo_size: int,
    dim_head: int,
    heads: int,
) -> torch.Tensor:
    """
    Halo Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        q_w: query 线性层权重 [dim_head * heads, C]（无偏置）
        kv_w: key/value 线性层权重 [2 * dim_head * heads, C]（无偏置）
        out_w: 输出线性层权重 [C, dim_head * heads]
        out_b: 输出线性层偏置 [C]
        block_size: 分块边长 b
        halo_size: halo 半径 halo
        dim_head: 每个 head 的维度
        heads: head 数

    Returns:
        输出特征图 [B, C, H, W]；若输入任一维为 0 则原样返回输入
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x

    block = block_size
    halo = halo_size

    q_inp = x.reshape(b, c, h // block, block, w // block, block).permute(0, 2, 4, 3, 5, 1).reshape(
        b * (h // block) * (w // block), block * block, c
    )

    kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
    kv_inp = kv_inp.reshape(b, c, -1, kv_inp.shape[-1]).permute(0, 3, 2, 1).reshape(
        b * kv_inp.shape[-1], -1, c
    )

    q = F.linear(q_inp, q_w)
    k, v = F.linear(kv_inp, kv_w).chunk(2, dim=-1)

    q = q.reshape(q.shape[0], q.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(
        q.shape[0] * heads, q.shape[1], -1
    )
    k = k.reshape(k.shape[0], k.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(
        k.shape[0] * heads, k.shape[1], -1
    )
    v = v.reshape(v.shape[0], v.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(
        v.shape[0] * heads, v.shape[1], -1
    )

    scale = dim_head ** -0.5
    q = q * scale
    sim = torch.bmm(q, k.transpose(1, 2))
    # 源实现未对 padding 位置做 mask
    attn = F.softmax(sim, dim=-1)
    out = torch.bmm(attn, v)
    out = out.reshape(-1, heads, out.shape[1], out.shape[2]).permute(0, 2, 1, 3).reshape(
        -1, out.shape[1], heads * out.shape[2]
    )
    out = F.linear(out, out_w, out_b)
    out = out.reshape(b, h // block, w // block, block, block, c).permute(0, 5, 1, 3, 2, 4).reshape(
        b, c, h, w
    )
    return out
