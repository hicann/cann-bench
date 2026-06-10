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


def get_input(
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
