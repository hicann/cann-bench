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

from typing import Optional, Sequence

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


class _NumpyAlias:
    ndarray = torch.Tensor
    dtype = torch.dtype
    float32 = torch.float32
    float64 = torch.float64
    int32 = torch.int32
    int64 = torch.int64
    inf = float("inf")


np = _NumpyAlias()

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
        for block_idx in range(min(math.ceil(seq_len / block_size), table.shape[1])):
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
        for block_idx in range(min(math.ceil(seq_len / block_size), table.shape[1])):
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

def _has_seq_lengths(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, (int, float)):
        return True
    return len(value) > 0


def _seq_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (int, float)):
        return 1
    return len(value)


def lightning_indexer(
    query,
    key,
    weights,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=2048,
    sparse_mode=3,
):
    batch = (
        _seq_count(actual_seq_lengths_query)
        if _has_seq_lengths(actual_seq_lengths_query)
        else (query.shape[0] if layout_query == "BSND" else 1)
    )
    q_total = query.shape[0] if layout_query == "TND" else query.shape[1]
    q_lens = as_int_list(
        actual_seq_lengths_query,
        q_total,
        batch,
        layout_query == "TND",
        default_mode="split_total" if layout_query == "TND" else "per_batch_full",
    )

    if layout_key == "TND":
        k_total = key.shape[0]
    elif layout_key == "BSND":
        k_total = key.shape[1]
    else:
        k_total = max(q_lens, default=0)
    k_lens = as_int_list(
        actual_seq_lengths_key,
        k_total,
        batch,
        layout_key == "TND",
        default_mode="split_total" if layout_key == "TND" else "per_batch_full",
    )

    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    if native:
        q_b = to_bsnd_device(query, layout_query, q_lens)
    else:
        q_b = to_bsnd_detached(query, layout_query, q_lens).to(torch.float32)
    if layout_query == "TND":
        w_b = (
            to_bsnd_device(weights.unsqueeze(-1), layout_query, q_lens).squeeze(-1)
            if native else
            to_bsnd_detached(weights.unsqueeze(-1), layout_query, q_lens).squeeze(-1).to(torch.float32)
        )
    else:
        w_b = weights.detach() if native else weights.detach().cpu().to(torch.float32)

    if layout_key.startswith("PA_"):
        k_b = page_to_bsnd_device(key, block_table, k_lens) if native else page_to_bsnd(key, block_table, k_lens).to(torch.float32)
    else:
        k_b = to_bsnd_device(key, layout_key, k_lens) if native else to_bsnd_detached(key, layout_key, k_lens).to(torch.float32)

    n2 = k_b.shape[-2]
    max_q = q_b.shape[1]
    indices = torch.full((len(q_lens), max_q, n2, int(sparse_count)), -1, dtype=torch.int32, device=q_b.device)
    values = torch.zeros(indices.shape, dtype=q_b.dtype, device=q_b.device)

    for b, q_len in enumerate(q_lens):
        k_len = k_lens[min(b, len(k_lens) - 1)] if k_lens else 0
        if q_len <= 0 or k_len <= 0:
            continue

        q = q_b[b, :q_len]
        k = k_b[b, :k_len]
        w = w_b[b, :q_len].unsqueeze(-1)
        group = max(q.shape[1] // n2, 1)
        q_grouped = q.reshape(q_len, n2, group, q.shape[-1])
        w_grouped = w.reshape(q_len, n2, group, 1)
        score = torch.einsum("sngd,tnd->sngt", q_grouped, k)
        score = (torch.relu(score) * w_grouped).sum(dim=2)

        if sparse_mode == 3:
            for s in range(q_len):
                valid_k = min(max(k_len - q_len + s + 1, 0), k_len)
                k_top = min(int(sparse_count), valid_k)
                if k_top <= 0:
                    continue
                vals, idx = torch.topk(score[s, :, :valid_k], k=k_top, dim=-1)
                indices[b, s, :, :k_top] = idx.to(torch.int32)
                values[b, s, :, :k_top] = vals
        else:
            k_top = min(int(sparse_count), k_len)
            vals, idx = torch.topk(score, k=k_top, dim=-1)
            indices[b, :q_len, :, :k_top] = idx.to(torch.int32)
            values[b, :q_len, :, :k_top] = vals

    return from_bsnd(indices, layout_query, q_lens), from_bsnd(values, layout_query, q_lens)


def lightning_indexer_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    return_value: bool = False,
    sparse_block_size: int = 1,
    sparse_block_mode: int = 0,
):
    del pre_tokens, next_tokens, sparse_block_size, sparse_block_mode
    sparse_indices, sparse_values = lightning_indexer(
        query,
        key,
        weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )
    if return_value:
        return sparse_indices, sparse_values
    return sparse_indices


def get_input(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, block_table=None, **kwargs):
    del kwargs
    return [query, key, weights, block_table]
