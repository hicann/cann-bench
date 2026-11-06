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
SlaFwd 算子 Torch Golden 参考实现

SparseLinearAttention 前向：块稀疏注意力分支 o_s 与线性注意力分支 o_l 的可训练融合。

公式:
    arg_k = k - mean(k, dim=-2)
    qm = 块均值(q, BLKQ), km = 块均值(arg_k, BLKK)
    score = qm @ km^T, 每个 query 块选 topk_ratio 比例的最高分 key 块
    o_s = softmax(sparse(q @ k^T * scale)) @ v           (块稀疏分支)
    o_l = (phi(q) @ (phi(k)^T @ v)) / (1e-5 + (phi(q) * sum(phi(k))).sum(-1)) @ W_l^T   (线性分支)
    y = o_s + o_l
"""


def _block_mean(x: torch.Tensor, blk: int) -> torch.Tensor:
    """按最后一维之外的时间维分块求均值（尾块按实际 token 数）。"""
    device = x.device
    b, h, length, d = x.shape
    lb = (length + blk - 1) // blk
    pad = lb * blk - length
    xp = F.pad(x.float(), (0, 0, 0, pad)).view(b, h, lb, blk, d)
    cnt = torch.full((lb,), blk, dtype=torch.float32, device=device)
    if pad:
        cnt[-1] = blk - pad
    return (xp.sum(3) / cnt.view(1, 1, lb, 1)).to(x.dtype)


def sla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W_l: torch.Tensor,
    topk: float,
    feature_map: str,
    BLKQ: int,
    BLKK: int,
) -> torch.Tensor:
    """
    SparseLinearAttention 前向

    Args:
        q: query [B, H, L, D]
        k: key [B, H, L, D]
        v: value [B, H, L, D]
        W_l: 线性分支输出投影权重 [D, D]
        topk: key 块选择比例，0 < topk <= 1
        feature_map: 线性分支特征图，"elu" / "relu" / "softmax"
        BLKQ: query 分块大小
        BLKK: key 分块大小

    Returns:
        融合输出 [B, H, L, D]
    """
    device = q.device
    b, h, length, d = q.shape
    assert k.shape == v.shape == q.shape, "SLA 要求 q/k/v 同形状（MHA）"
    assert q.dtype == k.dtype == v.dtype
    assert isinstance(feature_map, str) and feature_map in ("elu", "relu", "softmax")
    assert 0.0 < topk <= 1.0

    dtype = q.dtype

    # ---- 1. 块选择（get_block_map，精度路径镜像官方实现） ----
    arg_k = k - k.float().mean(dim=-2, keepdim=True).to(dtype)
    qm = _block_mean(q, BLKQ)
    km = _block_mean(arg_k, BLKK)
    pooled = (qm.float() @ km.float().transpose(-1, -2)).to(dtype)
    NK = pooled.shape[-1]
    real_topk = min(NK, int(topk * NK))
    assert real_topk >= 1, "real_topk=0 时官方 kernel 输出 NaN，未定义"
    lut = torch.topk(pooled.float(), real_topk, dim=-1, sorted=False).indices
    lut = lut.sort(dim=-1).values                            # [B, H, MQ, real_topk]

    scale = d ** -0.5
    if feature_map == "softmax":
        fmap = lambda x: F.softmax(x, dim=-1)
    elif feature_map == "elu":
        fmap = lambda x: F.elu(x) + 1
    else:
        fmap = lambda x: F.relu(x)

    MQ = pooled.shape[2]
    kf, vf, qf = k.float(), v.float(), q.float()

    # ---- 2. 稀疏分支 o_s（_attn_fwd 精度路径镜像） ----
    tok_idx = lut[..., None] * BLKK + torch.arange(BLKK, device=device)
    S = real_topk * BLKK
    tok_valid = (tok_idx < length).reshape(b, h, MQ, S)
    tok_idx = tok_idx.reshape(b, h, MQ, S)

    k2 = kf.reshape(b * h, length, d)
    v2 = vf.reshape(b * h, length, d)
    idx2 = tok_idx.reshape(b * h, MQ, S).clamp(0, length - 1)
    tv2 = tok_valid.reshape(b * h, MQ, 1, S)
    qpad = F.pad(qf.reshape(b * h, length, d),
                 (0, 0, 0, MQ * BLKQ - length)).reshape(b * h, MQ, BLKQ, d) * scale

    o_s = torch.empty(b * h, MQ, BLKQ, d, dtype=torch.float32, device=device)
    mq_chunk = 16
    for s0 in range(0, MQ, mq_chunk):
        e0 = min(MQ, s0 + mq_chunk)
        gi = idx2[:, s0:e0].unsqueeze(-1).expand(b * h, e0 - s0, S, d)
        k_g = torch.gather(k2.unsqueeze(1).expand(b * h, e0 - s0, length, d), 2, gi)
        v_g = torch.gather(v2.unsqueeze(1).expand(b * h, e0 - s0, length, d), 2, gi)
        sc = torch.einsum("bmqd,bmsd->bmqs", qpad[:, s0:e0], k_g)
        sc = sc.masked_fill(~tv2[:, s0:e0], float("-inf"))
        # kernel: p 以 fp32 求 exp2 与分母 l_i，PV 前舍回 v.dtype，落盘再舍回 v.dtype
        e = torch.exp(sc - sc.amax(dim=-1, keepdim=True))
        l_i = e.sum(-1, keepdim=True)
        acc = torch.einsum("bmqs,bmsd->bmqd", e.to(dtype).float(), v_g)
        o_s[:, s0:e0] = (acc / l_i).to(dtype).float()
    o_s = o_s.reshape(b, h, MQ * BLKQ, d)[:, :, :length]

    # ---- 3. 线性分支 o_l（core.py 官方路径，全程模型 dtype） ----
    c_q = fmap(qf).to(dtype).float()
    c_k = fmap(kf).to(dtype).float()
    kvsum = (c_k.transpose(-1, -2) @ vf).to(dtype).float()
    ksum = c_k.sum(dim=-2, keepdim=True).to(dtype).float()
    num = (c_q @ kvsum).to(dtype).float()
    den = (c_q * ksum).to(dtype).float().sum(-1, keepdim=True).to(dtype).float()
    den = (1e-5 + den).to(dtype).float()
    o_l = (num / den).to(dtype).float()
    o_l = (o_l @ W_l.float().transpose(-1, -2)).to(dtype).float()

    # ---- 4. 融合 ----
    return (o_s + o_l).to(dtype)
