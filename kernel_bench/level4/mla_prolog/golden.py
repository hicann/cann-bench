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
from typing import Tuple

"""
MlaProlog算子Torch Golden参考实现

Multi-Head Latent Attention前处理，融合 Query/Key 投影、RMSNorm 和 RoPE
"""


def rms_norm(x, gamma, epsilon):
    """
    RMSNorm: gamma * x / sqrt(mean(x^2) + epsilon).
    """
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f ** 2, dim=-1, keepdim=True) + epsilon)
    return (gamma.float() * x_f / rms).to(x.dtype)


def apply_rope(x, rope_cos, rope_sin):
    """
    Apply RoPE with pre-indexed sin/cos.
    """
    cos = rope_cos.float()
    sin = rope_sin.float()
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rotated * sin).to(x.dtype)


def mla_prolog(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-Head Latent Attention 前处理

    Args:
        token_x: [B, S, He] 输入 hidden states, bfloat16
        w_dq: [He, Hcq] query 下投影权重, bfloat16
        w_uq_qr: [Hcq, N*(D+Dr)] query 上投影+RoPE 权重(合并), bfloat16
        w_uk: [N, D, Hckv] key 上投影权重(吸收到 query 侧), bfloat16
        w_dkv_kr: [He, Hckv+Dr] KV 下投影+Key RoPE 权重(合并), bfloat16
        rmsnorm_gamma_cq: [Hcq] c_q 的 RMSNorm gamma, bfloat16
        rmsnorm_gamma_ckv: [Hckv] c_kv 的 RMSNorm gamma, bfloat16
        rope_sin: [B, S, Dr] RoPE 正弦(已按位置索引), bfloat16
        rope_cos: [B, S, Dr] RoPE 余弦(已按位置索引), bfloat16
        n_heads: 注意力头数 N
        rmsnorm_epsilon_cq: c_q RMSNorm epsilon
        rmsnorm_epsilon_ckv: c_kv RMSNorm epsilon

    Returns:
        query: [B, S, N, Hckv] 吸收 W_UK 后的 query, bfloat16
        query_rope: [B, S, N, Dr] query 位置编码, bfloat16
        c_kv: [B, S, Hckv] 归一化后的压缩 KV, bfloat16
        k_rope: [B, S, Dr] key 位置编码, bfloat16
    """
    original_dtype = token_x.dtype
    _low_prec = original_dtype in (torch.float16, torch.bfloat16)
    if _low_prec:
        token_x = token_x.float()
        w_dq = w_dq.float()
        w_uq_qr = w_uq_qr.float()
        w_uk = w_uk.float()
        w_dkv_kr = w_dkv_kr.float()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.float()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.float()
        rope_sin = rope_sin.float()
        rope_cos = rope_cos.float()

    B, S, He = token_x.shape
    N = n_heads
    Hckv = w_uk.shape[2]
    D = w_uk.shape[1]
    Dr = rope_sin.shape[-1]

    # === Query Path ===
    c_q_raw = torch.matmul(token_x, w_dq)
    c_q = rms_norm(c_q_raw, rmsnorm_gamma_cq, rmsnorm_epsilon_cq)
    qr = torch.matmul(c_q, w_uq_qr)
    qr = qr.reshape(B, S, N, D + Dr)
    q_c = qr[..., :D]
    q_r_raw = qr[..., D:]
    query = torch.einsum('bsnd,ndh->bsnh', q_c, w_uk)
    cos_exp = rope_cos.unsqueeze(2).expand(-1, -1, N, -1)
    sin_exp = rope_sin.unsqueeze(2).expand(-1, -1, N, -1)
    query_rope = apply_rope(q_r_raw, cos_exp, sin_exp)

    # === Key Path ===
    dkv_kr = torch.matmul(token_x, w_dkv_kr)
    ckv_raw = dkv_kr[..., :Hckv]
    kr_raw = dkv_kr[..., Hckv:]
    c_kv = rms_norm(ckv_raw, rmsnorm_gamma_ckv, rmsnorm_epsilon_ckv)
    k_rope = apply_rope(kr_raw, rope_cos, rope_sin)

    if _low_prec:
        query = query.to(original_dtype)
        query_rope = query_rope.to(original_dtype)
        c_kv = c_kv.to(original_dtype)
        k_rope = k_rope.to(original_dtype)

    return query, query_rope, c_kv, k_rope
