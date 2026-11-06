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

"""
FlexAttentionFwd 算子 Torch Golden 参考实现

FlexAttention 前向的数学子集（score_mod 为空、可选因果掩码与 GQA）：
    S = Q @ K^T * (1 / sqrt(D))
    if is_causal: S[i, j] = -inf，当 j > i + (S_k - S_q)
    P = exp(S - logsumexp(S, dim=-1))
    O = P @ V
enable_gqa 时先把 KV 头按 H // H_kv 复制到 Q 头数。
"""


def flex_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    FlexAttention 前向

    公式:
        S = Q @ K^T * (1 / sqrt(D))
        if is_causal: S[i, j] = -inf，当 j > i + (S_k - S_q)
        P = exp(S - logsumexp(S, dim=-1))
        O = P @ V

    Args:
        query: query 张量 [B, H, S_q, D]
        key: key 张量 [B, H_kv, S_k, D]
        value: value 张量 [B, H_kv, S_k, D]
        is_causal: 是否使用右下角对齐的因果掩码
        enable_gqa: 是否启用 GQA

    Returns:
        注意力输出 [B, H, S_q, D]
    """
    batch, heads, s_q, dim = query.shape
    h_kv = key.shape[1]
    s_k = key.shape[2]
    scale = 1.0 / math.sqrt(dim)

    query_f = query.float()
    key_f = key.float()
    value_f = value.float()

    if enable_gqa and heads != h_kv:
        rep = heads // h_kv
        key_f = key_f.repeat_interleave(rep, dim=1)
        value_f = value_f.repeat_interleave(rep, dim=1)

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, dtype=torch.bool), diagonal=s_k - s_q + 1
        )

    # 按 query 分块，避免一次物化 [B, H, S_q, S_k]
    out = torch.empty(batch, heads, s_q, value_f.shape[-1], dtype=torch.float32)
    budget = 2 ** 24
    blk = max(1, min(s_q, budget // max(1, batch * heads * s_k)))
    for start in range(0, s_q, blk):
        end = min(start + blk, s_q)
        scores = torch.matmul(query_f[:, :, start:end], key_f.transpose(-2, -1)) * scale
        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask[start:end].view(1, 1, end - start, s_k), float("-inf")
            )
        logsumexp = torch.logsumexp(scores, dim=-1)
        attn_probs = torch.exp(scores - logsumexp.unsqueeze(-1))
        out[:, :, start:end] = torch.matmul(attn_probs, value_f)

    return out.to(dtype=query.dtype)
