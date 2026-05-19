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

"""
MLA算子Torch Golden参考实现

多头潜在注意力 (Multi-Head Latent Attention)，仅包含注意力计算部分
Q 和 K 均分为 nope 和 rope 两部分传入，内部拼接后计算注意力
支持 BSND 和 BNSD 两种输入 layout

语义约束:
    v 与 k_nope 在数值上完全相同 (共享同一份 latent KV cache)。
    算子接口为兼容通用 attention API 保留独立入参；调用方需保证两者
    一致，本 Golden 实现不做强制检查。

公式:
    Q = concat(Q_nope, Q_rope)   dim: d_nope + d_rope
    K = concat(K_nope, K_rope)   dim: d_nope + d_rope
    V = K_nope                    dim: d_nope     (语义上等价)
    y = softmax(Q @ K^T * scaleValue) @ V
"""


def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """
    多头潜在注意力 (Multi-Head Latent Attention)

    Args:
        q_nope: query 的 nope 部分，BSND: [B, S, N_q, d_nope]，BNSD: [B, N_q, S, d_nope]
        q_rope: query 的 rope 部分，BSND: [B, S, N_q, d_rope]，BNSD: [B, N_q, S, d_rope]
        k_nope: key 的 nope 部分，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]
        k_rope: key 的 rope 部分，BSND: [B, S_kv, N_kv, d_rope]，BNSD: [B, N_kv, S_kv, d_rope]
        v: 值张量，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]
        numKVHeads: KV 头数
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(d_nope + d_rope)
        inputLayout: 输入 layout，"BSND" 或 "BNSD"
        is_causal: 是否启用因果掩码（右下角对齐），True 时 scores[..., i, j] 满足
            j > i + (S_kv - S) 的位置在 softmax 前置为 -inf。要求 S <= S_kv。

    Returns:
        输出张量，与输入 layout 一致，head dim 为 d_nope
    """
    # 统一转为 BSND 内部计算
    if inputLayout == "BNSD":
        q_nope = q_nope.permute(0, 2, 1, 3)
        q_rope = q_rope.permute(0, 2, 1, 3)
        k_nope = k_nope.permute(0, 2, 1, 3)
        k_rope = k_rope.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

    B, S, N_q, d_nope = q_nope.shape
    d_rope = q_rope.shape[-1]
    D_qk = d_nope + d_rope
    S_kv = k_nope.shape[1]
    N_kv = numKVHeads

    if scaleValue <= 0:
        scaleValue = 1.0 / (D_qk ** 0.5)

    # 拼接 Q = [Q_nope, Q_rope]: [B, S, N_q, d_nope + d_rope]
    q = torch.cat([q_nope, q_rope], dim=-1)

    # 拼接 K = [K_nope, K_rope]: [B, S_kv, N_kv, d_nope + d_rope]
    k = torch.cat([k_nope, k_rope], dim=-1)

    # GQA 扩展: 每个 KV head 复制 N_q // N_kv 次
    G = N_q // N_kv
    if G > 1:
        k = k.unsqueeze(3).expand(B, S_kv, N_kv, G, D_qk).reshape(B, S_kv, N_q, D_qk)
        v = v.unsqueeze(3).expand(B, S_kv, N_kv, G, d_nope).reshape(B, S_kv, N_q, d_nope)

    # 转置为 [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    if is_causal:
        i = torch.arange(S, device=scores.device).unsqueeze(-1)
        j = torch.arange(S_kv, device=scores.device).unsqueeze(0)
        causal_mask = j > (i + (S_kv - S))  # 右下角对齐：上三角置 -inf
        scores = scores.masked_fill(causal_mask, float('-inf'))
    # F217: 全 mask 行 (整行 = -inf) 在 softmax 时得 0/0 = NaN，对齐
    # sparse_flash_attention 加显式保护 → 全 mask 行权重置 0。
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_weights = torch.where(all_masked, torch.zeros_like(attn_weights), attn_weights)
    out = torch.matmul(attn_weights, v)  # [B, N_q, S, d_nope]

    # 转回 BSND: [B, S, N_q, d_nope]
    out = out.transpose(1, 2)

    # 按输入 layout 输出
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)

    return out
