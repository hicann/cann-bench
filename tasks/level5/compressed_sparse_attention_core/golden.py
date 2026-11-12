#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""DeepSeek-V4 CSA core benchmark, not a complete model attention layer.

Original reference implementation of the mathematical contract in desc.md.
Compressed entries and raw-window entries share K=V, but are distinct entries
in ONE sink-augmented softmax. Compression, index search and RoPE are upstream.
"""

import math

import torch


COMPRESSION_RATIO = 4
WINDOW_SIZE = 128


def _validate(
    query,
    compressed_kv,
    window_kv,
    compressed_indices,
    attn_sink,
    query_start,
    scaleValue,
):
    if query.ndim != 4 or compressed_kv.ndim != 3 or window_kv.ndim != 3:
        raise ValueError("Expected query[B,S,H,D] and two KV tensors[B,N,D]")
    b, s, h, d = query.shape
    if min(b, s, h, d) < 1 or query_start < 0:
        raise ValueError("Nonempty query and nonnegative query_start required")
    start = max(0, query_start - WINDOW_SIZE + 1)
    if window_kv.shape != (b, query_start + s - start, d):
        raise ValueError("window_kv must cover [max(0,query_start-127),query_start+S)")
    if compressed_kv.shape != (b, (query_start + s) // COMPRESSION_RATIO, d):
        raise ValueError("compressed_kv must contain all completed prefix blocks")
    if compressed_indices.ndim != 3 or compressed_indices.shape[:2] != (b, s):
        raise ValueError("compressed_indices must have shape [B,S,K]")
    if compressed_indices.shape[-1] < 1 or attn_sink.shape != (h,):
        raise ValueError("K must be positive and attn_sink must have shape [H]")
    if compressed_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Indices must be integer tensors")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("query must use a supported floating-point dtype")
    if compressed_kv.dtype != query.dtype or window_kv.dtype != query.dtype:
        # The internal FP32 working query is passed only to _compute, not here.
        raise TypeError("Query and both KV inputs must share dtype")
    if attn_sink.dtype not in (torch.float32, torch.float64):
        raise TypeError("attn_sink must be FP32, or FP64 for the oracle")
    if isinstance(query_start, bool) or not isinstance(query_start, int):
        raise TypeError("query_start must be an integer")
    if (
        isinstance(scaleValue, bool)
        or not isinstance(scaleValue, (float, int))
        or not math.isfinite(scaleValue)
    ):
        raise ValueError("scaleValue must be finite")
    if any(
        x.device != query.device
        for x in (compressed_kv, window_kv, compressed_indices, attn_sink)
    ):
        raise ValueError("All inputs must be on the same device")


def _fp32_query_key(query, entries):
    """Explicit FP32 products and reductions; bound the temporary by two tiles.

    FP32 public outputs face the strictest checker. Large finite logits make
    BLAS-dependent dot accumulation visible after softmax and cancellation.
    The reduction stays FP32 and does not change any input values or threshold.
    """
    b, s, h, _ = query.shape
    count = entries.shape[2]
    scores = query.new_empty((b, s, h, count))
    for head in range(0, h, 8):
        for entry in range(0, count, 64):
            products = (
                query[:, :, head : head + 8, None, :]
                * entries[:, :, None, entry : entry + 64, :]
            )
            scores[:, :, head : head + 8, entry : entry + 64] = products.sum(dim=-1)
    return scores


def _compute(
    query,
    compressed_kv,
    window_kv,
    compressed_indices,
    attn_sink,
    query_start,
    scaleValue,
    explicit_fp32_dot=False,
):
    """Bound temporary memory by query tiling; arithmetic follows query dtype."""
    b, s, h, d = query.shape
    c = compressed_kv.shape[1]
    scale = d**-0.5 if scaleValue <= 0 else scaleValue
    start = max(0, query_start - WINDOW_SIZE + 1)
    batch = torch.arange(b, device=query.device)[:, None, None]
    offsets = torch.arange(WINDOW_SIZE, device=query.device)
    outputs = []
    for lo in range(0, s, 16):
        hi = min(lo + 16, s)
        positions = query_start + torch.arange(lo, hi, device=query.device)
        # A fixed-width window is padded on the left near the start of a sequence.
        raw_positions = positions[:, None] - WINDOW_SIZE + 1 + offsets[None, :]
        raw_valid = raw_positions >= 0
        raw_ids = (raw_positions - start).clamp(min=0)
        raw = window_kv[batch, raw_ids[None, :, :], :].to(query.dtype)
        ids = compressed_indices[:, lo:hi, :].long()
        valid = (ids >= 0) & (ids < c)
        valid = valid & ((ids + 1) * COMPRESSION_RATIO <= positions[None, :, None] + 1)
        if c:
            comp = compressed_kv[batch, ids.clamp(min=0, max=c - 1), :].to(query.dtype)
        else:
            comp = torch.zeros((*ids.shape, d), dtype=query.dtype, device=query.device)
        entries = torch.cat((raw, comp), dim=2)
        mask = torch.cat((raw_valid[None, :, :].expand(b, -1, -1), valid), dim=2)
        scores = (
            _fp32_query_key(query[:, lo:hi], entries)
            if explicit_fp32_dot
            else torch.einsum("bshd,bsed->bshe", query[:, lo:hi], entries)
        ) * scale
        scores = scores.masked_fill(~mask[:, :, None, :], -torch.inf)
        sink = attn_sink.to(query.dtype)[None, None, :, None]
        maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        weights = torch.exp(scores - maximum)
        denominator = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
        outputs.append(torch.einsum("bshe,bsed->bshd", weights / denominator, entries))
    return torch.cat(outputs, dim=1).contiguous()


def compressed_sparse_attention_core(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """Return [B,S,H,D] in query dtype; use FP32, or genuine FP64, arithmetic.

    window_kv has absolute origin max(0,query_start-127); compressed block j has
    completion position 4*j+3. Valid indices are unique per row; -1 is padding.
    Inputs have already received any model projection/normalization/RoPE.
    """
    _validate(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )
    accumulation_dtype = (
        torch.float64 if query.dtype == torch.float64 else torch.float32
    )
    return _compute(
        query.to(accumulation_dtype),
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
        explicit_fp32_dot=query.dtype == torch.float32,
    ).to(query.dtype)


def compressed_sparse_attention_core_oracle(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """Dtype-preserving hook: evaluator FP64 inputs yield a genuine FP64 result."""
    _validate(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )
    return _compute(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )


def compressed_sparse_attention_core_bench(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """Same-precision correctness reference; NOT a measured performance baseline."""
    return compressed_sparse_attention_core(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )


def get_input(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
    **kwargs,
):
    """Make causal, unique index rows from seeded generator inputs, for ALL paths.

    An all--1 source row stays all padding. Other rows include the latest completed
    block and block zero when space permits, followed by input-dependent remote
    blocks. Short prefixes pad with -1. No randomness is introduced here and the
    seeded source indices affect selection. This is untimed structural input setup,
    not index search inside the operator. No external files are consulted.
    """
    _validate(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )
    b, s, k = compressed_indices.shape
    c = compressed_kv.shape[1]
    source = compressed_indices.detach().cpu().tolist()
    result = torch.full((b, s, k), -1, dtype=compressed_indices.dtype)
    for bi in range(b):
        for si in range(s):
            limit = min(c, (query_start + si + 1) // COMPRESSION_RATIO)
            row = source[bi][si]
            if not limit or all(x == -1 for x in row):
                continue
            count = min(k, limit)
            choices = [limit - 1, 0] + [abs(x) % limit for x in row if x != -1]
            choices += [(i * limit) // count for i in range(count)]
            seen, selected = set(), []
            for value in choices:
                if value not in seen:
                    seen.add(value)
                    selected.append(value)
                if len(selected) == count:
                    break
            result[bi, si, :count] = torch.tensor(selected, dtype=result.dtype)
    return [
        query,
        compressed_kv,
        window_kv,
        result.to(compressed_indices.device),
        attn_sink,
    ]
