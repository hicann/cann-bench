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


def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """BSND GQA; bottom-right causal alignment; output has query dtype.

    Half/bfloat16 values are already quantized by the generator. Promote them to
    FP32 for CPU matmul/softmax; preserve FP64 input for the judge's oracle path.
    Query tiling changes workspace, never the softmax/key reduction domain.
    """
    dtype = query.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    q, k, v = (t.to(work_dtype) for t in (query, key, value))
    B, S, Nq, D = q.shape
    Sk, Nk = k.shape[1:3]
    if Nq % Nk:
        raise ValueError("N_q must be divisible by N_kv")
    if is_causal and S > Sk:
        raise ValueError("causal GQA requires S <= S_kv")
    scale = 1.0 / D**0.5 if scaleValue <= 0 else scaleValue
    G = Nq // Nk
    out = torch.empty((B, S, Nq, D), dtype=work_dtype, device=q.device)
    # No expanded K/V. Bound scores to ~4M entries even for large batches.
    batch_tile = min(B, max(1, (4 * 1024**2) // max(1, Nq * Sk)))
    for b0 in range(0, B, batch_tile):
        b1 = min(B, b0 + batch_tile)
        nb = b1 - b0
        q_step = max(1, min(S, (4 * 1024**2) // (nb * Nq * Sk)))
        kb = k[b0:b1].permute(0, 2, 1, 3)
        vb = v[b0:b1].permute(0, 2, 1, 3)
        for i0 in range(0, S, q_step):
            i1 = min(S, i0 + q_step)
            si = i1 - i0
            qb = (
                q[b0:b1, i0:i1]
                .reshape(nb, si, Nk, G, D)
                .permute(0, 2, 3, 1, 4)
                .reshape(nb, Nk, G * si, D)
            )
            scores = torch.matmul(qb, kb.transpose(-1, -2))
            scores.mul_(scale)
            scores = scores.reshape(nb, Nk, G, si, Sk)
            if is_causal:
                rows = torch.arange(i0, i1, device=q.device).unsqueeze(-1)
                cols = torch.arange(Sk, device=q.device).unsqueeze(0)
                scores.masked_fill_(cols > rows + (Sk - S), float("-inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1).reshape(
                nb, Nk, G * si, Sk
            )
            block = torch.matmul(weights, vb)
            out[b0:b1, i0:i1] = (
                block.reshape(nb, Nk, G, si, D)
                .permute(0, 3, 1, 2, 4)
                .reshape(nb, si, Nq, D)
            )
    return out.to(dtype)


# Native reference only: round stage outputs to the input dtype, as in the
# published Torch reference. The public Golden above remains FP32/FP64.
_NATIVE_SCORE_ELEMENTS = 4 * 1024**2


def _native_attention(q, k, v, scale, is_causal):
    """BSND grouped attention; full KV softmax, bounded query-row workspace."""
    batch, query_length, query_heads, width = q.shape
    key_length, kv_heads = k.shape[1:3]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if is_causal and query_length > key_length:
        raise ValueError("right-bottom causal attention requires S <= S_kv")
    group = query_heads // kv_heads
    value_width = v.shape[-1]
    out = torch.empty(
        (batch, query_length, query_heads, value_width), dtype=q.dtype, device=q.device
    )
    row_tile = max(1, min(group * query_length, _NATIVE_SCORE_ELEMENTS // key_length))
    keys = torch.arange(key_length, device=q.device)
    for b in range(batch):
        for h in range(kv_heads):
            rows = (
                q[b, :, h * group : (h + 1) * group].transpose(0, 1).reshape(-1, width)
            )
            destination = torch.empty(
                (group * query_length, value_width), dtype=q.dtype, device=q.device
            )
            for start in range(0, group * query_length, row_tile):
                end = min(group * query_length, start + row_tile)
                # Each operation stores an input-dtype result; do not replace
                # this path with the promoted public Golden.
                scores = torch.matmul(rows[start:end], k[b, :, h].transpose(0, 1))
                scores = scores * scale
                if is_causal:
                    query_rows = (
                        torch.arange(start, end, device=q.device) % query_length
                    )
                    scores = scores.masked_fill(
                        keys[None] > query_rows[:, None] + key_length - query_length,
                        -torch.inf,
                    )
                probabilities = torch.nn.functional.softmax(scores, dim=-1)
                destination[start:end] = torch.matmul(probabilities, v[b, :, h])
            out[b, :, h * group : (h + 1) * group] = destination.reshape(
                group, query_length, value_width
            ).transpose(0, 1)
    return out


def gqa_bench(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Independent input-dtype reference; not the high-precision oracle."""
    scale = query.shape[-1] ** -0.5 if scaleValue <= 0 else scaleValue
    return _native_attention(query, key, value, scale, is_causal)
