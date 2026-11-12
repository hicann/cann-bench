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

"""Sparse attention reference with bounded sparse gathers.

The public reference accumulates float16/bfloat16 inputs in float32 and returns
the query dtype. Float64 inputs remain float64 throughout the reference, so the
CPU evaluator can obtain a genuine float64 oracle after converting its inputs.
There is no dense S1-by-S2 score tensor or materialized GQA head repetition.
"""

import hashlib
import math

import torch


def _as_bnsd(query, key, value, sparse_indices, layout):
    if layout == "BSND":
        return tuple(x.permute(0, 2, 1, 3) for x in (query, key, value, sparse_indices))
    if layout == "BNSD":
        return query, key, value, sparse_indices
    raise ValueError("inputLayout must be BSND or BNSD")


def _validate(q, k, v, indices, scale, causal):
    if any(x.ndim != 4 for x in (q, k, v, indices)):
        raise ValueError("all four inputs must have rank four")
    b, n1, s1, dk = q.shape
    if min(q.shape) <= 0 or min(k.shape) <= 0 or min(v.shape) <= 0:
        raise ValueError("tensor dimensions must be positive")
    n2, s2, dv = k.shape[1], k.shape[2], v.shape[-1]
    if b > 64 or s1 > 2048 or s2 > 16384 or n1 > 256 or n2 > 64:
        raise ValueError("B/S1/S2/N1/N2 exceed the documented support range")
    if k.shape[0] != b or k.shape[-1] != dk or v.shape[:3] != k.shape[:3]:
        raise ValueError("query/key/value shapes are inconsistent")
    if n1 % n2 or dv > dk or dk % 64 or dv % 64:
        raise ValueError(
            "N1 must be divisible by N2; Dk/Dv must be 64-aligned, Dv <= Dk"
        )
    if indices.shape[:3] != (b, n2, s1) or not 1 <= indices.shape[-1] <= min(2048, s2):
        raise ValueError(
            "sparseIndices must have shape [B,N2,S1,topK], 1 <= topK <= min(2048,S2)"
        )
    if causal and s1 > s2:
        raise ValueError("causal attention requires S1 <= S2")
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scaleValue must be positive and finite")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("query/key/value dtypes must match")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("sparseIndices must be integer")
    if bool((indices < 0).any()) or bool((indices >= s2).any()):
        raise ValueError("sparseIndices entries must lie in [0,S2)")


def _attention(
    query,
    key,
    value,
    sparse_indices,
    scale,
    layout,
    causal,
    force_fp64=False,
    native_precision=False,
):
    q, k, v, indices = _as_bnsd(query, key, value, sparse_indices, layout)
    _validate(q, k, v, indices, scale, causal)
    b, n1, s1, dk = q.shape
    n2, s2, dv, topk = k.shape[1], k.shape[2], v.shape[-1], indices.shape[-1]
    group = n1 // n2
    if force_fp64 or q.dtype == torch.float64:
        work_dtype = torch.float64
    else:
        # The compatibility bench stores each arithmetic stage in the input
        # dtype, as the historical reference did. Public/FP64 paths keep their
        # existing arithmetic policy; all paths retain sparse bounded gathers.
        work_dtype = q.dtype if native_precision else torch.float32
    out = torch.empty((b, n1, s1, dv), device=q.device, dtype=work_dtype)
    # Bound selected-KV tensors plus scores to about 16 MiB. Tile size follows
    # shape/dtype only and never changes the attention definition.
    bytes_per_row = (topk * (dk + dv) + group * (dk + topk + dv)) * torch.empty(
        (), dtype=work_dtype
    ).element_size()
    rows_per_tile = max(1, min(64, (16 * 1024 * 1024) // bytes_per_row))
    for batch in range(b):
        for head in range(n2):
            # Convert each KV source head once, rather than repeating it G times.
            key_head = k[batch, head].to(work_dtype)
            value_head = v[batch, head].to(work_dtype)
            head_start = head * group
            for start in range(0, s1, rows_per_tile):
                stop = min(s1, start + rows_per_tile)
                ids = indices[batch, head, start:stop].long()
                selected_key = key_head.index_select(0, ids.reshape(-1)).reshape(
                    stop - start, topk, dk
                )
                selected_value = value_head.index_select(0, ids.reshape(-1)).reshape(
                    stop - start, topk, dv
                )
                query_tile = (
                    q[batch, head_start : head_start + group, start:stop]
                    .transpose(0, 1)
                    .to(work_dtype)
                )
                scores = torch.bmm(query_tile, selected_key.transpose(1, 2)) * scale
                if causal:
                    limit = (
                        torch.arange(start, stop, device=q.device).unsqueeze(-1)
                        + s2
                        - s1
                    )
                    scores.masked_fill_(~(ids <= limit).unsqueeze(1), float("-inf"))
                # Avoid -inf - -inf and zero division for fully masked rows.
                row_max = scores.amax(-1, keepdim=True)
                empty = torch.isneginf(row_max)
                scores.sub_(
                    torch.where(empty, torch.zeros_like(row_max), row_max)
                ).exp_()
                denominator = scores.sum(-1, keepdim=True)
                scores.div_(
                    torch.where(empty, torch.ones_like(denominator), denominator)
                )
                output_tile = torch.bmm(scores, selected_value)
                out[batch, head_start : head_start + group, start:stop] = (
                    output_tile.transpose(0, 1)
                )
    if layout == "BSND":
        out = out.permute(0, 2, 1, 3)
    return out.to(torch.float64 if force_fp64 else query.dtype).contiguous()


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Sparse GQA/MLA attention; value must equal key[..., :Dv].

    Indices must be unique within each (batch, KV head, query row). Causal
    masking is bottom-right aligned; fully masked rows return exact zeros.
    The output has the input layout, query dtype, and contiguous storage.
    """
    return _attention(
        query, key, value, sparseIndices, scaleValue, inputLayout, is_causal
    )


def sparse_flash_attention_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """CPU audit entry point: float64 arithmetic and float64 output."""
    return _attention(
        query,
        key,
        value,
        sparseIndices,
        scaleValue,
        inputLayout,
        is_causal,
        force_fp64=True,
    )


def sparse_flash_attention_bench(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Historical native-stage precision, with bounded sparse storage.

    This is a compatibility reference for the existing three-input checker,
    not a requirement that NPU implementations use an identical reduction
    tree. It does not change thresholds or the FP64 mathematical oracle.
    """
    return _attention(
        query,
        key,
        value,
        sparseIndices,
        scaleValue,
        inputLayout,
        is_causal,
        native_precision=True,
    )


def _structural_seed(query, key, sparse_indices):
    """Hash bounded samples, independent of the caller's global RNG state.

    Only 256 source indices and 32 Q/K elements each are copied to CPU. The
    coordinates address the original tensor directly, including strided input,
    so sampling never flattens/copies a complete noncontiguous payload.
    """
    digest = hashlib.sha256()
    for tensor, sample_count in ((sparse_indices, 256), (query, 32), (key, 32)):
        count = min(sample_count, tensor.numel())
        positions = torch.arange(count, device=tensor.device, dtype=torch.int64)
        if count > 1:
            positions = positions * (tensor.numel() - 1) // (count - 1)
        coordinates = []
        for size in reversed(tensor.shape):
            coordinates.insert(0, positions.remainder(size))
            positions = positions.div(size, rounding_mode="floor")
        sample = tensor[tuple(coordinates)].detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(bytes(sample.view(torch.uint8).tolist()))
    return int.from_bytes(digest.digest()[:8], "little") % (2**63)


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float = 1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
    **kwargs,
):
    """Construct V=K prefix and reproducible, unique sparse index sets.

    A stable hash of bounded input samples controls the structural input. A
    local generator avoids global RNG dependence or consuming its stream; a
    bounded random-priority block uses topk rather than sorting all groups.
    No duplicate padding is permitted when topK exceeds S2: that is invalid.
    """
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    s2 = int(key.shape[1] if inputLayout == "BSND" else key.shape[2])
    topk = int(sparseIndices.shape[-1])
    if not 1 <= topk <= min(2048, s2):
        raise ValueError("unique sparse indices require 1 <= topK <= min(2048,S2)")
    if value.shape[-1] > key.shape[-1]:
        raise ValueError("Dv must not exceed Dk")
    new_value = (
        key[..., : value.shape[-1]].clone().to(dtype=value.dtype, device=value.device)
    )
    groups = math.prod(sparseIndices.shape[:-1])
    selection = torch.empty((groups, topk), dtype=sparseIndices.dtype, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(
        _structural_seed(query, key, sparseIndices)
    )
    groups_per_block = max(1, (1024 * 1024) // s2)
    for start in range(0, groups, groups_per_block):
        stop = min(groups, start + groups_per_block)
        priorities = torch.rand((stop - start, s2), generator=generator)
        # Every returned column index is distinct, even when priorities tie.
        selection[start:stop] = priorities.topk(
            topk, dim=-1, largest=False, sorted=False
        ).indices
    new_indices = selection.reshape(sparseIndices.shape).to(sparseIndices.device)
    return [query, key, new_value, new_indices]
