#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from typing import Optional, Sequence, Tuple

import torch



def _ceil(value):
    ivalue = int(value)
    return ivalue if float(value) == float(ivalue) else ivalue + 1


class _Math:
    @staticmethod
    def ceil(value):
        return _ceil(value)


math = _Math()
F = torch.nn.functional

def _default_lengths(total: Optional[int], batch: Optional[int], default_mode: Optional[str]):
    if total is None:
        return []
    batch = max(int(batch or 1), 1)
    total = max(int(total), 0)
    if default_mode is None or default_mode == "split_total":
        base = total // batch
        out = [base] * batch
        for idx in range(total - base * batch):
            out[idx] += 1
        return out
    if default_mode == "per_batch_full":
        return [total] * batch
    raise ValueError(f"unsupported default_mode: {default_mode}")


def as_int_list(
    value,
    total: Optional[int] = None,
    batch: Optional[int] = None,
    is_tnd: bool = True,
    *,
    default_mode: Optional[str] = None,
    allow_scalar: bool = True,
    empty_uses_default: bool = True,
):
    if value is None:
        return _default_lengths(total, batch, default_mode) if empty_uses_default else []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif allow_scalar and isinstance(value, (int, float)):
        value = [value]
    out = [int(x) for x in value]
    if not out:
        return _default_lengths(total, batch, default_mode) if empty_uses_default else []
    if not is_tnd or total is None:
        return out
    if all(a <= b for a, b in zip(out, out[1:])) and out[-1] == int(total):
        prev = 0
        lengths = []
        for item in out:
            lengths.append(max(int(item) - prev, 0))
            prev = int(item)
        return lengths
    if sum(out) == int(total):
        return out
    positive = [max(x, 0) for x in out]
    if sum(positive) == 0:
        positive = [1] * len(out)
    raw = [int(total) * x / sum(positive) for x in positive]
    lengths = [int(x) for x in raw]
    for idx in sorted(range(len(raw)), key=lambda i: raw[i] - lengths[i], reverse=True)[: int(total) - sum(lengths)]:
        lengths[idx] += 1
    return lengths


def prefix_from_lengths(lengths: Sequence[int]):
    out = []
    total = 0
    for item in lengths:
        total += int(item)
        out.append(total)
    return out


def starts_from_lengths(lengths: Sequence[int]):
    starts = []
    cur = 0
    for item in lengths:
        starts.append(cur)
        cur += int(item)
    return starts


def page_to_bsnd(x: torch.Tensor, block_table: Optional[torch.Tensor], lengths: Sequence[int]):
    x = x.detach().cpu()
    if block_table is None:
        return x
    block_num, block_size, n, d = x.shape
    table = block_table.detach().cpu().to(torch.int64)
    max_len = max([int(v) for v in lengths], default=0)
    out = torch.zeros((len(lengths), max_len, n, d), dtype=x.dtype)
    for b, seq_len in enumerate(lengths):
        if b >= table.shape[0]:
            continue
        for block_idx in range(min(_ceil(seq_len / block_size), table.shape[1])):
            src = int(table[b, block_idx]) % max(block_num, 1)
            begin = block_idx * block_size
            end = min(begin + block_size, int(seq_len), max_len)
            if end > begin:
                out[b, begin:end] = x[src, : end - begin]
    return out


def page_to_bsnd_device(x: torch.Tensor, block_table: Optional[torch.Tensor], lengths: Sequence[int]):
    if block_table is None:
        return x
    block_num, block_size, n, d = x.shape
    table = block_table.detach().to(torch.int64).to(x.device)
    max_len = max([int(v) for v in lengths], default=0)
    out = torch.zeros((len(lengths), max_len, n, d), dtype=x.dtype, device=x.device)
    for b, seq_len in enumerate(lengths):
        if b >= table.shape[0]:
            continue
        for block_idx in range(min(_ceil(seq_len / block_size), table.shape[1])):
            src = int(table[b, block_idx].item()) % max(block_num, 1)
            begin = block_idx * block_size
            end = min(begin + block_size, int(seq_len), max_len)
            if end > begin:
                out[b, begin:end] = x[src, : end - begin]
    return out


def _to_bsnd_core(x: torch.Tensor, layout: str, lengths: Sequence[int], preserve_grad: bool):
    x = x.cpu() if preserve_grad and x.requires_grad else x.detach().cpu()
    if layout == "TND":
        max_len = max(lengths, default=0)
        if preserve_grad and x.requires_grad:
            parts = []
            start = 0
            for seq_len in lengths:
                part = x[start : start + seq_len]
                start += seq_len
                if int(seq_len) < max_len:
                    pad_shape = (max_len - int(seq_len), *part.shape[1:])
                    part = torch.cat([part, part.new_zeros(pad_shape)], dim=0)
                parts.append(part)
            return torch.stack(parts, dim=0) if parts else x.new_empty((0, max_len, x.shape[-2], x.shape[-1]))
        out = torch.zeros((len(lengths), max_len, x.shape[-2], x.shape[-1]), dtype=x.dtype)
        start = 0
        for b, seq_len in enumerate(lengths):
            out[b, :seq_len] = x[start : start + seq_len]
            start += seq_len
        return out
    return x


def to_bsnd_device(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    if layout == "TND":
        max_len = max(lengths, default=0)
        out = torch.zeros((len(lengths), max_len, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)
        start = 0
        for b, seq_len in enumerate(lengths):
            out[b, :seq_len] = x[start : start + seq_len]
            start += seq_len
        return out
    return x


def to_bsnd_detached(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    return _to_bsnd_core(x, layout, lengths, preserve_grad=False)


def to_bsnd_grad(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    return _to_bsnd_core(x, layout, lengths, preserve_grad=True)


def to_bsnd(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    return to_bsnd_grad(x, layout, lengths)


def from_bsnd(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    if layout == "TND":
        parts = [x[b, : int(seq_len)] for b, seq_len in enumerate(lengths)]
        return torch.cat(parts, dim=0) if parts else x.new_empty((0, x.shape[-2], x.shape[-1]))
    return x


def broadcast_kv(x: torch.Tensor, n1: int):
    if x.shape[-2] == n1:
        return x
    group = max(n1 // x.shape[-2], 1)
    return x.repeat_interleave(group, dim=-2)


def slice_optional(x: Optional[torch.Tensor], b: int, q_len: int, kv_len: int):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() <= 1:
        return None
    x = x.detach().cpu()
    if x.dim() == 2:
        return x[-q_len:, -kv_len:]
    if x.dim() >= 4:
        b_idx = min(b, x.shape[0] - 1)
        return x[b_idx : b_idx + 1, :, -q_len:, -kv_len:]
    return x


def slice_optional_device(x: Optional[torch.Tensor], b: int, q_len: int, kv_len: int):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() <= 1:
        return None
    if x.dim() == 2:
        return x[-q_len:, -kv_len:]
    if x.dim() >= 4:
        b_idx = min(b, x.shape[0] - 1)
        return x[b_idx : b_idx + 1, :, -q_len:, -kv_len:]
    return x

_slice_optional = slice_optional
_slice_optional_device = slice_optional_device

def _seq_lengths(lengths, total: int, batch: int, is_tnd: bool):
    return as_int_list(
        lengths,
        total,
        batch,
        is_tnd,
        default_mode="split_total" if is_tnd else "per_batch_full",
    )


def _tnd_to_bsnd_torch(x: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    batch = len(lengths)
    max_s = max(lengths) if lengths else 0
    out = torch.zeros((batch, max_s, x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
    total_len = x.shape[0]
    packed_len = sum(lengths)
    stride = total_len // batch if batch and total_len % batch == 0 else 0
    padded = stride >= max_s and packed_len != total_len
    start = 0
    for b, seq_len in enumerate(lengths):
        src_start = b * stride if padded else start
        out[b, :seq_len] = x[src_start:src_start + seq_len]
        if not padded:
            start += seq_len
    return out


def _bnsd_to_tnd_torch(x: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    parts = []
    for b, seq_len in enumerate(lengths):
        parts.append(x[b, :, :seq_len, :].permute(1, 0, 2))
    if parts:
        return torch.cat(parts, dim=0)
    return torch.empty((0, x.shape[1], x.shape[3]), dtype=x.dtype, device=x.device)


def _effective_pa_lengths(lengths: Sequence[int], block_table: torch.Tensor, block_size: int) -> Sequence[int]:
    table = block_table.detach().cpu()
    capacity = int(table.shape[1]) * max(int(block_size), 1) if table.dim() >= 2 else 0
    return [min(max(int(length), 0), capacity) for length in lengths]


def _gather_kv_positions(indices, block_size: int, kv_len: int, device) -> torch.Tensor:
    positions = []
    for sparse_id in indices.detach().cpu().tolist():
        sparse_id = int(sparse_id)
        if sparse_id == -1:
            break
        begin = sparse_id * block_size
        end = min(begin + block_size, kv_len)
        if begin < kv_len:
            positions.extend(range(begin, end))
    return torch.tensor(positions, dtype=torch.long, device=device)


def _apply_right_down_causal_mask_torch(scores: torch.Tensor, q_len: int, kv_len: int, indices, q_idx: int,
                                        block_size: int) -> Tuple[torch.Tensor, bool]:
    if scores.shape[-1] == 0:
        return scores, True
    tail_block = _ceil(kv_len / block_size)
    tail_len = kv_len % block_size or block_size
    threshold = kv_len - q_len + q_idx + 1
    offset = 0
    masked = 0
    for sparse_id in indices.detach().cpu().tolist():
        sparse_id = int(sparse_id)
        if sparse_id == -1:
            break
        begin = sparse_id * block_size
        block_len = block_size if sparse_id != tail_block - 1 else tail_len
        end = begin + block_len
        if begin < threshold and end <= threshold:
            offset += block_len
            continue
        if end > threshold:
            local = 0 if threshold <= begin else threshold - begin
            scores[:, offset + local:offset + block_len] = torch.finfo(scores.dtype).min
            masked += block_len - local
        offset += block_len
    return scores, masked == scores.shape[-1]


def _page_to_bsnd_torch(x: torch.Tensor, block_table: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    block_num, block_size, n, d = x.shape
    table = block_table.detach().to(torch.int64).to(x.device)
    max_len = max(lengths) if lengths else 0
    out = torch.zeros((len(lengths), max_len, n, d), dtype=x.dtype, device=x.device)
    for b, seq_len in enumerate(lengths):
        if b >= table.shape[0]:
            continue
        for block_idx in range(min(_ceil(seq_len / block_size), table.shape[1])):
            src = int(table[b, block_idx].item()) % max(block_num, 1)
            begin = block_idx * block_size
            end = min(begin + block_size, int(seq_len), max_len)
            if end > begin:
                out[b, begin:end] = x[src, : end - begin]
    return out


def _ai_infra_sparse_flash_attention_gqa_native_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    sparse_block_size: int,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    block_table: Optional[torch.Tensor],
    num_key_value_heads: int,
    input_layout: str,
    input_layout_kv: str,
    sparse_mode: int,
    return_softmax_lse: bool,
):
    layout_q = input_layout
    layout_kv = input_layout_kv
    batch = len(actual_seq_lengths_query) if actual_seq_lengths_query is not None else (
        query.shape[0] if layout_q == "BSND" else 1
    )
    q_lens = _seq_lengths(
        actual_seq_lengths_query,
        query.shape[0] if layout_q == "TND" else query.shape[1],
        batch,
        layout_q == "TND",
    )
    kv_lens = _seq_lengths(
        actual_seq_lengths_kv,
        key.shape[0] if layout_kv == "TND" else key.shape[1],
        batch,
        layout_kv == "TND",
    )

    if layout_q == "TND":
        q_bsnd = _tnd_to_bsnd_torch(query, q_lens)
        sparse_bsnd = _tnd_to_bsnd_torch(sparse_indices, q_lens)
    else:
        q_bsnd = query
        sparse_bsnd = sparse_indices

    if layout_kv.startswith("PA_"):
        if block_table is None:
            raise ValueError("block_table is required for PA layouts")
        k_bsnd = _page_to_bsnd_torch(key, block_table, kv_lens)
        v_bsnd = _page_to_bsnd_torch(value, block_table, kv_lens)
        kv_lens = _effective_pa_lengths(kv_lens, block_table, key.shape[1])
    elif layout_kv == "TND":
        k_bsnd = _tnd_to_bsnd_torch(key, kv_lens)
        v_bsnd = _tnd_to_bsnd_torch(value, kv_lens)
    else:
        k_bsnd = key
        v_bsnd = value

    q_bnsd = q_bsnd.permute(0, 2, 1, 3).contiguous()
    k_bnsd = k_bsnd.permute(0, 2, 1, 3).contiguous()
    v_bnsd = v_bsnd.permute(0, 2, 1, 3).contiguous()
    idx_bns = sparse_bsnd.permute(0, 2, 1, 3).contiguous()
    num_heads = q_bnsd.shape[1]
    num_kv_heads = k_bnsd.shape[1] if num_key_value_heads == 0 else num_key_value_heads
    group = num_heads // num_kv_heads
    out = torch.zeros_like(q_bnsd)
    max_q = max(q_lens) if q_lens else 0
    lse = torch.zeros((batch, num_heads, max_q, 1), dtype=torch.float32, device=query.device)

    scale = torch.tensor(scale_value, dtype=query.dtype, device=query.device)
    for b in range(batch):
        for kv_head in range(num_kv_heads):
            for q_idx in range(q_lens[b]):
                heads = slice(kv_head * group, (kv_head + 1) * group)
                indices = idx_bns[b, kv_head, q_idx]
                positions = _gather_kv_positions(indices, sparse_block_size, kv_lens[b], query.device)
                if positions.numel() == 0:
                    lse[b, heads, q_idx, 0] = torch.inf
                    continue
                q_cur = q_bnsd[b, heads, q_idx, :]
                k_sparse = k_bnsd[b, kv_head].index_select(0, positions)
                v_sparse = v_bnsd[b, kv_head].index_select(0, positions)
                scores = torch.matmul(q_cur, k_sparse.transpose(0, 1)) * scale
                invalid = False
                if sparse_mode == 3:
                    scores, invalid = _apply_right_down_causal_mask_torch(
                        scores, q_lens[b], kv_lens[b], indices, q_idx, sparse_block_size
                    )
                if invalid:
                    probs = torch.zeros_like(scores)
                    lse[b, heads, q_idx, 0] = torch.inf
                else:
                    probs = torch.softmax(scores, dim=-1)
                    lse[b, heads, q_idx, 0] = torch.logsumexp(scores.to(torch.float32), dim=-1)
                if return_softmax_lse:
                    attn_probs = probs
                    attn_value = v_sparse
                    out[b, heads, q_idx, :] = torch.matmul(attn_probs, attn_value)
                else:
                    out[b, heads, q_idx, :] = torch.matmul(probs, v_sparse)

    if layout_q == "TND":
        attention = _bnsd_to_tnd_torch(out, q_lens)
        lse_out = _bnsd_to_tnd_torch(lse, q_lens)
    else:
        attention = out.permute(0, 2, 1, 3).contiguous()
        lse_out = lse
    return (attention, lse_out) if return_softmax_lse else (attention, torch.empty(0, device=query.device))


def _legalize_block_table(block_table: Optional[torch.Tensor], block_num: int) -> Optional[torch.Tensor]:
    if block_table is None:
        return None
    return torch.remainder(block_table.detach().cpu().to(torch.int64), max(int(block_num), 1)).to(torch.int32)


def _zero_attention(query: torch.Tensor, value_dim: int, q_lens: Sequence[int], layout_q: str, batch: int,
                    num_heads: int, return_softmax_lse: bool):
    if layout_q == "TND":
        attention = torch.zeros((query.shape[0], num_heads, value_dim), dtype=query.dtype)
        lse = torch.zeros((query.shape[0], num_heads, 1), dtype=torch.float32)
    else:
        attention = torch.zeros((query.shape[0], query.shape[1], num_heads, value_dim), dtype=query.dtype)
        lse = torch.zeros((batch, num_heads, max(q_lens, default=0), 1), dtype=torch.float32)
    return (attention, lse) if return_softmax_lse else (attention, torch.empty(0))


def _too_large_for_exact(q_lens: Sequence[int], num_heads: int, sparse_count: int, block_size: int,
                         limit: int = 25_000_000) -> bool:
    work = int(sum(q_lens)) * max(int(num_heads), 1) * max(int(sparse_count), 1) * max(int(block_size), 1)
    return work > limit

def _should_use_native_torch_path(query: torch.Tensor) -> bool:
    return isinstance(query, torch.Tensor)


def ai_infra_sparse_flash_attention_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    sparse_block_size: int = 1,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    block_table: Optional[torch.Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    input_layout: str = "BSND",
    input_layout_kv: str = "BSND",
    sparse_mode: int = 0,
    block_size: int = 0,
    return_softmax_lse: bool = False,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    sparse_block_count: int = 0,
    **kwargs,
):
    del num_query_heads, block_size, pre_tokens, next_tokens, sparse_block_count, kwargs
    return _ai_infra_sparse_flash_attention_gqa_native_torch(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        sparse_block_size,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        block_table,
        num_key_value_heads,
        input_layout,
        input_layout_kv,
        sparse_mode,
        return_softmax_lse,
    )


def _get_input_unused_1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    input_layout_kv: str = "BSND",
    **kwargs,
):
    del actual_seq_lengths_query, actual_seq_lengths_kv, kwargs
    real_block_table = block_table if input_layout_kv.startswith("PA_") else None
    return [query, key, value, sparse_indices, real_block_table]

def _get_input_unused_2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    input_layout_kv: str = "BSND",
    **kwargs,
):
    del actual_seq_lengths_query, actual_seq_lengths_kv, kwargs
    real_block_table = block_table if input_layout_kv.startswith("PA_") else None
    return [query, key, value, sparse_indices, real_block_table]


def _empty_outputs(query: torch.Tensor):
    return torch.empty(0, dtype=torch.float32, device=query.device)


def _split_nope_rope(tensor: torch.Tensor, base_dim: int, rope: Optional[torch.Tensor] = None):
    if rope is not None:
        if tensor.shape[-1] > base_dim:
            tensor = tensor[..., :base_dim]
        return tensor, rope
    if tensor.shape[-1] > base_dim:
        return tensor[..., :base_dim], tensor[..., base_dim:]
    return tensor, None


def _ai_infra_sparse_flash_attention_pioneer_native_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    block_table: Optional[torch.Tensor],
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    query_rope: Optional[torch.Tensor],
    key_rope: Optional[torch.Tensor],
    key_sink: Optional[torch.Tensor],
    value_sink: Optional[torch.Tensor],
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
    return_softmax_lse: bool,
):
    batch = len(actual_seq_lengths_query) if actual_seq_lengths_query is not None else (
        query.shape[0] if layout_query == "BSND" else 1
    )
    q_lens = _seq_lengths(
        actual_seq_lengths_query,
        query.shape[0] if layout_query == "TND" else query.shape[1],
        batch,
        layout_query == "TND",
    )
    kv_lens = _seq_lengths(
        actual_seq_lengths_kv,
        key.shape[0] if layout_kv == "TND" else key.shape[1],
        batch,
        layout_kv == "TND",
    )

    value_dim = int(value.shape[-1])
    q_base, q_rope = _split_nope_rope(query, value_dim, query_rope)
    k_base, k_rope = _split_nope_rope(key, q_base.shape[-1], key_rope)

    if layout_query == "TND":
        q_base = _tnd_to_bsnd_torch(q_base, q_lens)
        q_rope = _tnd_to_bsnd_torch(q_rope, q_lens) if q_rope is not None else None
        sparse_bsnd = _tnd_to_bsnd_torch(sparse_indices, q_lens)
    else:
        sparse_bsnd = sparse_indices

    if layout_kv.startswith("PA_"):
        if block_table is None:
            raise ValueError("block_table is required for PA layouts")
        k_base = _page_to_bsnd_torch(k_base, block_table, kv_lens)
        k_rope = _page_to_bsnd_torch(k_rope, block_table, kv_lens) if k_rope is not None else None
        v_bsnd = _page_to_bsnd_torch(value, block_table, kv_lens)
        kv_lens = _effective_pa_lengths(kv_lens, block_table, value.shape[1])
    elif layout_kv == "TND":
        k_base = _tnd_to_bsnd_torch(k_base, kv_lens)
        k_rope = _tnd_to_bsnd_torch(k_rope, kv_lens) if k_rope is not None else None
        v_bsnd = _tnd_to_bsnd_torch(value, kv_lens)
    else:
        v_bsnd = value

    if q_rope is not None and k_rope is not None:
        q_used = torch.cat([q_base, q_rope], dim=-1)
        k_used = torch.cat([k_base, k_rope], dim=-1)
    else:
        q_used = q_base
        k_used = k_base

    q_bnsd = q_used.permute(0, 2, 1, 3).contiguous()
    k_bnsd = k_used.permute(0, 2, 1, 3).contiguous()
    v_bnsd = v_bsnd[..., :value_dim].permute(0, 2, 1, 3).contiguous()
    idx_bns = sparse_bsnd.permute(0, 2, 1, 3).contiguous()

    if key_sink is not None:
        key_sink_used = key_sink[..., : q_used.shape[-1]]
        if value_sink is None:
            value_sink = key_sink[..., :value_dim]
        value_sink = value_sink[..., :value_dim]
    else:
        key_sink_used = None

    num_heads = q_bnsd.shape[1]
    num_kv_heads = k_bnsd.shape[1]
    group = num_heads // num_kv_heads
    max_q = max(q_lens) if q_lens else 0
    out = torch.zeros((q_bnsd.shape[0], num_heads, q_bnsd.shape[2], value_dim), dtype=query.dtype, device=query.device)
    softmax_max = torch.full((batch, num_heads, max_q, 1), -torch.inf, dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros((batch, num_heads, max_q, 1), dtype=torch.float32, device=query.device)
    scale = torch.tensor(scale_value, dtype=query.dtype, device=query.device)

    for b in range(batch):
        for kv_head in range(num_kv_heads):
            for q_idx in range(q_lens[b]):
                heads = slice(kv_head * group, (kv_head + 1) * group)
                indices = idx_bns[b, kv_head, q_idx]
                positions = _gather_kv_positions(indices, sparse_block_size, kv_lens[b], query.device)

                q_cur = q_bnsd[b, heads, q_idx, :]
                if positions.numel() == 0:
                    scores = q_cur.new_empty((q_cur.shape[0], 0))
                    v_sparse = v_bnsd[b, kv_head].new_empty((0, value_dim))
                else:
                    k_sparse = k_bnsd[b, kv_head].index_select(0, positions)
                    v_sparse = v_bnsd[b, kv_head].index_select(0, positions)
                    scores = torch.matmul(q_cur, k_sparse.transpose(0, 1)) * scale
                    if sparse_mode == 3:
                        scores, _ = _apply_right_down_causal_mask_torch(
                            scores, q_lens[b], kv_lens[b], indices, q_idx, sparse_block_size
                        )

                if key_sink_used is not None:
                    sink_scores = torch.matmul(q_cur, key_sink_used[:, kv_head, :].transpose(0, 1)) * scale
                    scores = torch.cat([sink_scores, scores], dim=-1)
                    v_sparse = torch.cat([value_sink[:, kv_head, :], v_sparse], dim=0)

                if scores.shape[-1] == 0:
                    continue
                probs = torch.softmax(scores, dim=-1)
                scores_fp32 = scores.to(torch.float32)
                score_max = torch.amax(scores_fp32, dim=-1, keepdim=True)
                exp_scores = torch.exp(scores_fp32 - score_max)
                exp_scores = torch.where(torch.isfinite(scores_fp32), exp_scores, torch.zeros_like(exp_scores))
                softmax_max[b, heads, q_idx, 0] = score_max.squeeze(-1)
                softmax_sum[b, heads, q_idx, 0] = torch.sum(exp_scores, dim=-1)
                out[b, heads, q_idx, :] = torch.matmul(probs, v_sparse)

    if layout_query == "TND":
        attention = _bnsd_to_tnd_torch(out, q_lens)
        softmax_max_out = _bnsd_to_tnd_torch(softmax_max, q_lens)
        softmax_sum_out = _bnsd_to_tnd_torch(softmax_sum, q_lens)
    else:
        attention = out.permute(0, 2, 1, 3).contiguous()
        softmax_max_out = softmax_max
        softmax_sum_out = softmax_sum
    if not return_softmax_lse:
        softmax_max_out = _empty_outputs(query)
        softmax_sum_out = _empty_outputs(query)
    return attention, softmax_max_out, softmax_sum_out


def _pioneer_lengths(value, total, batch, is_tnd):
    return _seq_lengths(value, total, batch, is_tnd)


def _build_st_block_table(block_table, key, actual_seq_lengths_kv):
    if block_table is None:
        return None
    block_num = int(key.shape[0])
    block_size = int(key.shape[1])
    batch = int(block_table.shape[0]) if block_table.dim() >= 2 else len(actual_seq_lengths_kv or [block_num * block_size])
    if actual_seq_lengths_kv is None:
        blocks_per_batch = int(block_table.shape[1]) if block_table.dim() >= 2 else block_num
        kv_lens = [blocks_per_batch * block_size] * batch
    else:
        kv_lens = _pioneer_lengths(actual_seq_lengths_kv, block_num * block_size, batch, False)
    max_blocks = int(block_table.shape[1]) if block_table.dim() >= 2 else max((int(_ceil(x / block_size)) for x in kv_lens), default=0)
    out = torch.zeros((batch, max_blocks), dtype=torch.int32)
    cursor = 0
    for b in range(batch):
        need = min(max_blocks, max(int(_ceil(kv_lens[min(b, len(kv_lens) - 1)] / block_size)), 0))
        if need > 0:
            out[b, :need] = torch.arange(cursor, cursor + need, dtype=torch.int32) % max(block_num, 1)
            cursor += need
    return out


def _legalize_sparse_indices_torch(sparse_indices, actual_seq_lengths_kv, sparse_block_size):
    if sparse_indices is None:
        return None
    out = sparse_indices.clone()
    if actual_seq_lengths_kv is None:
        return out
    batch = out.shape[0] if out.dim() == 4 else 1
    kv_lens = _pioneer_lengths(actual_seq_lengths_kv, max(actual_seq_lengths_kv), batch, False)
    for b in range(batch):
        max_blocks = max(int(_ceil(kv_lens[min(b, len(kv_lens) - 1)] / max(int(sparse_block_size), 1))), 1)
        target = out[b] if out.dim() == 4 else out
        valid = target >= 0
        target[valid] = torch.remainder(target[valid], max_blocks)
    return out


def _append_legacy_zero_rope(query, key, value, rope_dim=64):
    if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
        return query, key
    if query.shape[-1] != value.shape[-1] or key.shape[-1] != value.shape[-1]:
        return query, key
    if value.shape[-1] != 512:
        return query, key
    q_rope = torch.zeros(*query.shape[:-1], rope_dim, dtype=query.dtype, device=query.device)
    k_rope = torch.zeros(*key.shape[:-1], rope_dim, dtype=key.dtype, device=key.device)
    return torch.cat([query, q_rope], dim=-1), torch.cat([key, k_rope], dim=-1)


def _get_input_unused_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    key_sink: Optional[torch.Tensor] = None,
    value_sink: Optional[torch.Tensor] = None,
    layout_kv: str = "PA_BSND",
    sparse_block_size: int = 1,
    **kwargs,
):
    del actual_seq_lengths_query, kwargs
    query, key = _append_legacy_zero_rope(query, key, value)
    real_block_table = _build_st_block_table(block_table, key, actual_seq_lengths_kv) if layout_kv.startswith("PA_") else None
    sparse_indices = _legalize_sparse_indices_torch(sparse_indices, actual_seq_lengths_kv, sparse_block_size)
    if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor) and key.shape[-1] >= value.shape[-1]:
        value = key[..., : value.shape[-1]].contiguous()
    tensors = [item for item in (query_rope, key_rope, key_sink, value_sink) if isinstance(item, torch.Tensor)]
    real_query_rope = None
    real_key_rope = None
    real_key_sink = None
    real_value_sink = None
    for item in tensors:
        if item.dim() < 3:
            continue
        if item.shape[0] == query.shape[0] and item.shape[-1] < query.shape[-1] and real_query_rope is None:
            real_query_rope = item
        elif item.shape[0] == key.shape[0] and item.shape[-1] < key.shape[-1] and real_key_rope is None:
            real_key_rope = item
        elif item.shape[-1] == value.shape[-1] and real_value_sink is None:
            real_value_sink = item
        elif real_key_sink is None:
            real_key_sink = item
    if real_key_sink is not None and real_value_sink is not None and real_key_sink.shape[-1] >= real_value_sink.shape[-1]:
        real_value_sink = real_key_sink[..., : real_value_sink.shape[-1]].contiguous()
    return [
        query, key, value, sparse_indices, real_block_table,
        real_query_rope, real_key_rope, real_key_sink, real_value_sink,
    ]

def ai_infra_sparse_flash_attention_pioneer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    key_sink: Optional[torch.Tensor] = None,
    value_sink: Optional[torch.Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "PA_BSND",
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    attention_mode: int = 0,
    return_softmax_lse: bool = False,
    block_size: int = 0,
    sparse_block_count: int = 0,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    **kwargs,
):
    del pre_tokens, next_tokens, attention_mode, block_size, sparse_block_count, num_query_heads, num_key_value_heads, kwargs
    return _ai_infra_sparse_flash_attention_pioneer_native_torch(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        key_sink,
        value_sink,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        return_softmax_lse,
    )

def _get_input_unused_4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    input_layout_kv: str = "BSND",
    **kwargs,
):
    del actual_seq_lengths_query, actual_seq_lengths_kv, kwargs
    real_block_table = block_table if input_layout_kv.startswith("PA_") else None
    return [query, key, value, sparse_indices, real_block_table]


def _get_input_unused_5(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    key_sink: Optional[torch.Tensor] = None,
    value_sink: Optional[torch.Tensor] = None,
    layout_kv: str = "PA_BSND",
    sparse_block_size: int = 1,
    **kwargs,
):
    del actual_seq_lengths_query, kwargs
    query, key = _append_legacy_zero_rope(query, key, value)
    real_block_table = _build_st_block_table(block_table, key, actual_seq_lengths_kv) if layout_kv.startswith("PA_") else None
    sparse_indices = _legalize_sparse_indices_torch(sparse_indices, actual_seq_lengths_kv, sparse_block_size)
    if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor) and key.shape[-1] >= value.shape[-1]:
        value = key[..., : value.shape[-1]].contiguous()
    tensors = [item for item in (query_rope, key_rope, key_sink, value_sink) if isinstance(item, torch.Tensor)]
    real_query_rope = None
    real_key_rope = None
    real_key_sink = None
    real_value_sink = None
    for item in tensors:
        if item.dim() < 3:
            continue
        if item.shape[0] == query.shape[0] and item.shape[-1] < query.shape[-1] and real_query_rope is None:
            real_query_rope = item
        elif item.shape[0] == key.shape[0] and item.shape[-1] < key.shape[-1] and real_key_rope is None:
            real_key_rope = item
        elif item.shape[-1] == value.shape[-1] and real_value_sink is None:
            real_value_sink = item
        elif real_key_sink is None:
            real_key_sink = item
    if real_key_sink is not None and real_value_sink is not None and real_key_sink.shape[-1] >= real_value_sink.shape[-1]:
        real_value_sink = real_key_sink[..., : real_value_sink.shape[-1]].contiguous()
    return [
        query, key, value, sparse_indices, real_block_table,
        real_query_rope, real_key_rope, real_key_sink, real_value_sink,
    ]


def _cpu_tensor(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def _none_if_empty_tensor(x):
    if isinstance(x, torch.Tensor) and x.numel() == 0:
        return None
    return x


def _numel(value) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def _legalize_sparse_indices_for_layout(
    sparse_indices: torch.Tensor,
    key: torch.Tensor,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
) -> torch.Tensor:
    if sparse_indices is None:
        return None
    out = sparse_indices.clone()
    block_size = max(int(sparse_block_size), 1)
    batch = _numel(actual_seq_lengths_query) or _numel(actual_seq_lengths_kv)
    if batch <= 0:
        batch = int(out.shape[0]) if out.dim() == 4 else 1

    q_total = int(out.shape[0]) if layout_query == "TND" else int(out.shape[1])
    q_lens = _pioneer_lengths(actual_seq_lengths_query, q_total, batch, layout_query == "TND")
    if layout_kv == "TND":
        kv_total = int(key.shape[0])
    elif layout_kv.startswith("PA_"):
        kv_total = int(key.shape[0]) * int(key.shape[1])
    else:
        kv_total = int(key.shape[1])
    kv_lens = _pioneer_lengths(actual_seq_lengths_kv, kv_total, batch, layout_kv == "TND")

    if layout_query == "TND":
        start = 0
        for b, q_len in enumerate(q_lens):
            target = out[start : start + int(q_len)]
            start += int(q_len)
            if target.numel() == 0:
                continue
            max_blocks = max(int((int(kv_lens[min(b, len(kv_lens) - 1)]) + block_size - 1) // block_size), 1)
            valid = target >= 0
            target[valid] = torch.remainder(target[valid], max_blocks)
    else:
        for b in range(min(int(out.shape[0]), len(kv_lens))):
            target = out[b]
            max_blocks = max(int((int(kv_lens[b]) + block_size - 1) // block_size), 1)
            valid = target >= 0
            target[valid] = torch.remainder(target[valid], max_blocks)
    return out


def sparse_flash_attention_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    attention_mode: int = 0,
    return_softmax_lse: bool = False,
    block_size: int = 0,
    sparse_block_count: int = 0,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    **kwargs,
):
    """Torch golden for npu_sparse_flash_attention_enhance."""
    del block_size, sparse_block_count, num_query_heads, num_key_value_heads, kwargs
    block_table = _none_if_empty_tensor(block_table)
    return ai_infra_sparse_flash_attention_pioneer(
        _cpu_tensor(query),
        _cpu_tensor(key),
        _cpu_tensor(value),
        _cpu_tensor(sparse_indices),
        scale_value,
        block_table=_cpu_tensor(block_table),
        actual_seq_lengths_query=_cpu_tensor(actual_seq_lengths_query),
        actual_seq_lengths_kv=_cpu_tensor(actual_seq_lengths_kv),
        query_rope=_cpu_tensor(query_rope),
        key_rope=_cpu_tensor(key_rope),
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    *extra_tensors,
    **kwargs,
):
    layout_query = kwargs.get("layout_query", "BSND")
    layout_kv = kwargs.get("layout_kv", "BSND")
    sparse_block_size = int(kwargs.get("sparse_block_size", 1))
    actual_seq_lengths_query = kwargs.get("actual_seq_lengths_query")
    actual_seq_lengths_kv = kwargs.get("actual_seq_lengths_kv")
    block_table = extra_tensors[0] if len(extra_tensors) > 0 else None
    query_rope = extra_tensors[3] if len(extra_tensors) > 3 else None
    key_rope = extra_tensors[4] if len(extra_tensors) > 4 else None
    if getattr(query, "is_meta", False):
        real_block_table = block_table if layout_kv.startswith("PA_") else torch.empty(0, dtype=torch.int32, device=query.device)
        return [query, key, value, sparse_indices, real_block_table, actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope]
    # Legacy cases omit null placeholders for optional actual_seq tensors, so
    # proto-order mapping may place query_rope/key_rope in those slots.
    legacy_query_rope = None
    legacy_key_rope = None
    if query_rope is None and isinstance(actual_seq_lengths_query, torch.Tensor) and actual_seq_lengths_query.is_floating_point():
        legacy_query_rope = actual_seq_lengths_query
    if key_rope is None and isinstance(actual_seq_lengths_kv, torch.Tensor) and actual_seq_lengths_kv.is_floating_point():
        legacy_key_rope = actual_seq_lengths_kv

    if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor) and key.shape[-1] >= value.shape[-1]:
        value = key[..., : value.shape[-1]].contiguous()

    real_actual_q = actual_seq_lengths_query
    real_actual_kv = actual_seq_lengths_kv
    if isinstance(real_actual_q, torch.Tensor) and real_actual_q.is_floating_point():
        real_actual_q = None
    if isinstance(real_actual_kv, torch.Tensor) and real_actual_kv.is_floating_point():
        real_actual_kv = None
    if real_actual_q is not None and not isinstance(real_actual_q, torch.Tensor):
        real_actual_q = torch.tensor(real_actual_q, dtype=torch.int32)
    if real_actual_kv is not None and not isinstance(real_actual_kv, torch.Tensor):
        real_actual_kv = torch.tensor(real_actual_kv, dtype=torch.int32)

    sparse_indices = _legalize_sparse_indices_for_layout(
        sparse_indices,
        key,
        real_actual_q,
        real_actual_kv,
        sparse_block_size,
        layout_query,
        layout_kv,
    )
    if layout_kv.startswith("PA_"):
        real_block_table = _build_st_block_table(block_table, key, real_actual_kv)
        real_query_rope = query_rope if query_rope is not None else legacy_query_rope
        real_key_rope = key_rope if key_rope is not None else legacy_key_rope
    else:
        real_block_table = torch.empty(0, dtype=torch.int32)
        real_query_rope = query_rope if query_rope is not None else (
            block_table if isinstance(block_table, torch.Tensor) and block_table.is_floating_point() else legacy_query_rope
        )
        real_key_rope = key_rope if key_rope is not None else legacy_key_rope
    return [
        query, key, value, sparse_indices,
        real_block_table, real_actual_q, real_actual_kv,
        real_query_rope, real_key_rope,
    ]
