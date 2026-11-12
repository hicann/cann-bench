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


def get_input(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> list:
    """Preserve the shared-latent contract for both first and fresh-input attempts."""
    return [q_nope, q_rope, k_nope, k_rope, k_nope.clone()]


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
    """Attention-only MLA; bounded score storage and full-KV-axis softmax.

    Input quantization is retained. FP16/BF16 CPU arithmetic uses FP32; a
    FP64 input remains FP64 for the judge oracle. Only the final output is
    converted back to the input dtype. Query tiling does not approximate or
    truncate the attention matrix, and KV heads are never expanded to N_q.
    """
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    output_dtype = q_nope.dtype
    compute_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    tensors = (q_nope, q_rope, k_nope, k_rope, v)
    qn, qr, kn, kr, vv = [x.to(compute_dtype) for x in tensors]
    if inputLayout == "BNSD":
        qn, qr, kn, kr, vv = [x.permute(0, 2, 1, 3) for x in (qn, qr, kn, kr, vv)]
    B, S, Nq, Dv = qn.shape
    Sk, Nkv = kn.shape[1:3]
    if numKVHeads != Nkv or Nq % Nkv:
        raise ValueError("numKVHeads must match KV shape and divide query heads")
    if is_causal and S > Sk:
        raise ValueError("right-bottom causal attention requires S <= S_kv")
    G = Nq // Nkv
    Dqk = Dv + qr.shape[-1]
    scale = Dqk**-0.5 if scaleValue <= 0 else scaleValue
    q = torch.cat((qn, qr), -1)
    k = torch.cat((kn, kr), -1)
    out = torch.empty((B, S, Nq, Dv), dtype=compute_dtype, device=q.device)
    # A tile stores at most 4M logits, independent of B, Nq and query length.
    row_tile = max(1, min(G * S, 4194304 // Sk))
    keys = torch.arange(Sk, device=q.device)
    for b in range(B):
        for h in range(Nkv):
            rows = q[b, :, h * G : (h + 1) * G].transpose(0, 1).reshape(G * S, Dqk)
            key = k[b, :, h]
            value = vv[b, :, h]
            dest = torch.empty((G * S, Dv), dtype=compute_dtype, device=q.device)
            for start in range(0, G * S, row_tile):
                end = min(G * S, start + row_tile)
                scores = torch.matmul(rows[start:end], key.transpose(-2, -1)) * scale
                if is_causal:
                    query_rows = torch.arange(start, end, device=q.device) % S
                    scores = scores.masked_fill(
                        keys[None] > query_rows[:, None] + Sk - S, -torch.inf
                    )
                probs = torch.nn.functional.softmax(scores, dim=-1)
                dest[start:end] = torch.matmul(probs, value)
            out[b, :, h * G : (h + 1) * G] = dest.reshape(G, S, Dv).transpose(0, 1)
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)
    return out.to(output_dtype).contiguous()


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


def mla_bench(
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
    """Independent native-stage reference; preserves the shared-latent inputs."""
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    tensors = (q_nope, q_rope, k_nope, k_rope, v)
    if inputLayout == "BNSD":
        tensors = tuple(t.permute(0, 2, 1, 3) for t in tensors)
    qn, qr, kn, kr, value = tensors
    if numKVHeads != kn.shape[2]:
        raise ValueError("numKVHeads must match KV shape")
    q, k = torch.cat((qn, qr), -1), torch.cat((kn, kr), -1)
    scale = q.shape[-1] ** -0.5 if scaleValue <= 0 else scaleValue
    out = _native_attention(q, k, value, scale, is_causal)
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)
    return out.contiguous()
