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
FlexAttentionBwd 算子 Torch Golden 参考实现

FlexAttention 反向（template-style，纯反向逻辑）。由前向保存的 attn_output 与
logsumexp 重建 softmax 概率，计算 query / key / value 梯度，数学与官方 Inductor
backward template（torch/_inductor/kernel/flex/flex_attention.py）一致：

    P   = exp(scale * Q K^T - logsumexp)          # causal 时上三角为 0
    dv  = P^T dO
    dP  = dO V^T
    D   = rowsum(dO ⊙ O)
    dS  = P ⊙ (dP - D)
    dq  = dS K · scale
    dk  = dS^T Q · scale

支持可选因果掩码与 GQA（KV 头 expand 到 Q 头，梯度在组内求和恢复 KV 头数）。
全程 fp32 计算，输出回铸 query/key/value 的 dtype。
"""


def flex_attention_bwd(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> tuple:
    """
    FlexAttention 反向

    Args:
        grad_output: 输出梯度 dO [B, H, S_q, D]
        query: 前向 query [B, H, S_q, D]
        key: 前向 key [B, H_kv, S_k, D]（enable_gqa 时 H_kv < H）
        value: 前向 value [B, H_kv, S_k, D]
        attn_output: 前向注意力输出 O [B, H, S_q, D]
        logsumexp: 前向 softmax 的 logsumexp [B, H, S_q]
        is_causal: 是否使用因果掩码
        enable_gqa: 是否启用 GQA（KV 头扩展 + 梯度组内求和）

    Returns:
        (grad_query, grad_key, grad_value)，shape 分别与 query/key/value 一致
    """
    B, H, S_q, D = query.shape
    H_kv = key.shape[1]
    S_k = key.shape[2]
    scale = 1.0 / math.sqrt(D)

    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    grad_output_f = grad_output.float()

    # GQA: expand KV heads to Q heads
    if enable_gqa and H != H_kv:
        rep = H // H_kv
        key_f = key_f.repeat_interleave(rep, dim=1)
        value_f = value_f.repeat_interleave(rep, dim=1)

    attn_weight = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=query.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1)
        attn_weight.masked_fill_(causal_mask, float('-inf'))

    attn_probs = torch.exp(attn_weight - logsumexp.unsqueeze(-1).float())

    grad_value = torch.matmul(attn_probs.transpose(-2, -1), grad_output_f)
    grad_attn_probs = torch.matmul(grad_output_f, value_f.transpose(-2, -1))
    D_term = (grad_output_f * attn_output.float()).sum(dim=-1, keepdim=True)
    grad_attn_weight = attn_probs * (grad_attn_probs - D_term)

    grad_query = torch.matmul(grad_attn_weight, key_f) * scale
    grad_key = torch.matmul(grad_attn_weight.transpose(-2, -1), query_f) * scale

    # GQA: sum expanded-head grads back to KV heads
    if enable_gqa and H != H_kv:
        rep = H // H_kv
        grad_key = grad_key.view(B, H_kv, rep, S_k, D).sum(dim=2)
        grad_value = grad_value.view(B, H_kv, rep, S_k, D).sum(dim=2)

    return (grad_query.to(query.dtype),
            grad_key.to(key.dtype),
            grad_value.to(value.dtype))


def get_input(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
    **kwargs,
) -> list:
    """按前向公式重算 attn_output 与 logsumexp，保证反向输入自洽。

    单区间 value_range 无法表达「saved tensor 是 q/k/v 的确定性函数」这一约束，
    这里用 fp32 重放前向（含 causal 掩码与 GQA 头扩展）后返回；q/k/v/dO 原样保留。

    Returns:
        [grad_output, query, key, value, attn_output, logsumexp]
    """
    q_f = query.float()
    k_f = key.float()
    v_f = value.float()
    B, H, S_q, D = q_f.shape
    H_kv = k_f.shape[1]
    S_k = k_f.shape[2]
    scale = 1.0 / math.sqrt(D)

    if enable_gqa and H != H_kv:
        rep = H // H_kv
        k_f = k_f.repeat_interleave(rep, dim=1)
        v_f = v_f.repeat_interleave(rep, dim=1)

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=q_f.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    attn_probs = torch.exp(scores - lse.unsqueeze(-1))
    new_attn_output = torch.matmul(attn_probs, v_f).to(attn_output.dtype)
    new_logsumexp = lse.to(logsumexp.dtype)
    return [grad_output, query, key, value, new_attn_output, new_logsumexp]
