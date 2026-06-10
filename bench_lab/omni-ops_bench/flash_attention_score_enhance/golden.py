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

def _logical_atten_mask_for_forward(atten_mask, query, key, **kwargs):
    sparse_mode = int(kwargs.get("sparse_mode", 0))
    if sparse_mode not in (2, 3, 4):
        return atten_mask
    layout = kwargs.get("input_layout", "TND")
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    sink_len = int(kwargs.get("sink_num", 0) or 0) * 64 if layout == "TND" else 0
    if layout == "TND":
        kv_total = _seq_total_from_attr(kwargs.get("actual_seq_kvlen"))
        if kv_total is None:
            kv_total = key.shape[0] - sink_len if sink_len and key.shape[0] > sink_len else key.shape[0]
    else:
        kv_total = key.shape[1]
    actual_seq_qlen = kwargs.get("actual_seq_qlen")
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen")
    batch = len(actual_seq_qlen) if actual_seq_qlen is not None else (query.shape[0] if layout == "BSND" else 1)
    q_lens = as_int_list(
        actual_seq_qlen,
        q_total,
        batch,
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    kv_lens = as_int_list(
        actual_seq_kvlen,
        kv_total,
        len(q_lens),
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    max_q = max(q_lens, default=q_total)
    max_kv = max(kv_lens, default=kv_total)
    if sparse_mode == 2:
        return torch.triu(torch.ones((max_q, max_kv), dtype=torch.uint8, device=query.device), diagonal=1)
    if sparse_mode == 3:
        return torch.triu(torch.ones((max_q, max_kv), dtype=torch.uint8, device=query.device), diagonal=max_kv - max_q + 1)
    pre_tokens = int(kwargs.get("pre_tokens", (1 << 31) - 1))
    next_tokens = int(kwargs.get("next_tokens", (1 << 31) - 1))
    upper = torch.triu(torch.ones((max_q, max_kv), dtype=torch.uint8, device=query.device), diagonal=next_tokens + 1 + max_kv - max_q)
    lower = torch.tril(torch.ones((max_q, max_kv), dtype=torch.uint8, device=query.device), diagonal=-pre_tokens - 1 + max_kv - max_q)
    return upper + lower


def _split_sink_prefix(x: torch.Tensor, sink_len: int, main_total=None):
    if sink_len <= 0:
        return None, x
    base = getattr(x, "_base", None)
    if isinstance(base, torch.Tensor) and x.dim() >= 1:
        storage_offset = x.storage_offset()
        prefix_offset = storage_offset - sink_len * x.stride(0)
        if prefix_offset >= 0:
            prefix = base.as_strided((sink_len, *x.shape[1:]), x.stride(), prefix_offset)
            return prefix, x
    if main_total is not None and int(x.shape[0]) == int(main_total):
        return x[:sink_len], x
    return x[:sink_len], x[sink_len:]


def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: Optional[int] = None,
    input_layout: str = "TND",
    pse: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    scale_value: float = None,
    keep_prob: float = 1.0,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode: int = 0,
    pre_tokens: int = (1 << 31) - 1,
    next_tokens: int = (1 << 31) - 1,
    sink_num: int = 0,
):
    layout = input_layout
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    batch = len(actual_seq_qlen) if actual_seq_qlen is not None else (query.shape[0] if layout == "BSND" else 1)
    q_lens = as_int_list(
        actual_seq_qlen,
        q_total,
        batch,
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    sink_len = int(sink_num) * 64 if layout == "TND" else 0
    kv_main_total = _seq_total_from_attr(actual_seq_kvlen) if layout == "TND" else None
    key_sink = value_sink = key_rope_sink = None
    if sink_len > 0:
        key_sink, key = _split_sink_prefix(key, sink_len, kv_main_total)
        value_sink, value = _split_sink_prefix(value, sink_len, kv_main_total)
        if key_rope is not None and key_rope.numel():
            key_rope_sink, key_rope = _split_sink_prefix(key_rope, sink_len, kv_main_total)

    kv_total = kv_main_total if layout == "TND" else key.shape[1]
    if kv_total is None:
        kv_total = key.shape[0] if layout == "TND" else key.shape[1]
    kv_lens = as_int_list(
        actual_seq_kvlen,
        kv_total,
        len(q_lens),
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    q_b = to_bsnd_detached(query, layout, q_lens).to(torch.float32)
    k_b = to_bsnd_detached(key, layout, kv_lens).to(torch.float32)
    v_b = to_bsnd_detached(value, layout, kv_lens).to(torch.float32)
    if query_rope is not None and key_rope is not None and query_rope.numel() and key_rope.numel():
        qr_b = to_bsnd_detached(query_rope, layout, q_lens).to(torch.float32)
        kr_b = to_bsnd_detached(key_rope, layout, kv_lens).to(torch.float32)
        q_b = torch.cat([q_b, qr_b], dim=-1)
        k_b = torch.cat([k_b, kr_b], dim=-1)
        if key_rope_sink is not None:
            key_sink = torch.cat([key_sink.detach().cpu().to(torch.float32), key_rope_sink.detach().cpu().to(torch.float32)], dim=-1)
    if key_sink is not None:
        key_sink = key_sink.detach().cpu().to(torch.float32).permute(1, 0, 2)
        value_sink = value_sink.detach().cpu().to(torch.float32).permute(1, 0, 2)

    n1 = int(head_num or q_b.shape[-2])
    out = torch.zeros((len(q_lens), max(q_lens, default=0), q_b.shape[-2], v_b.shape[-1]), dtype=torch.float32)
    softmax_max = torch.zeros((len(q_lens), n1, max(q_lens, default=0), 8), dtype=torch.float32)
    softmax_sum = torch.zeros_like(softmax_max)

    for b, q_len in enumerate(q_lens):
        kv_len = kv_lens[min(b, len(kv_lens) - 1)] if kv_lens else 0
        if q_len <= 0 or kv_len <= 0:
            continue
        q = q_b[b, :q_len].permute(1, 0, 2)
        k = broadcast_kv(k_b[b, :kv_len], n1).permute(1, 0, 2)
        v = broadcast_kv(v_b[b, :kv_len], n1).permute(1, 0, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * float(scale)
        sink_cols = 0
        if key_sink is not None and value_sink is not None:
            cur_key_sink = broadcast_kv(key_sink.permute(1, 0, 2), n1).permute(1, 0, 2)
            cur_value_sink = broadcast_kv(value_sink.permute(1, 0, 2), n1).permute(1, 0, 2)
            scores = torch.cat([torch.matmul(q, cur_key_sink.transpose(-1, -2)) * float(scale), scores], dim=-1)
            v = torch.cat([cur_value_sink, v], dim=-2)
            sink_cols = cur_key_sink.shape[-2]
        cur_pse = _slice_optional(pse, b, q_len, kv_len)
        if cur_pse is not None:
            cur_pse = cur_pse.reshape(-1, q_len, kv_len)[: scores.shape[0]].to(torch.float32)
            if sink_cols:
                cur_pse = torch.nn.functional.pad(cur_pse, (sink_cols, 0), value=0)
            scores = scores + cur_pse
        cur_mask = _slice_optional(atten_mask, b, q_len, kv_len)
        if cur_mask is not None:
            cur_mask = cur_mask.reshape(-1, q_len, kv_len).bool()[: scores.shape[0]]
            if sink_cols:
                cur_mask = torch.nn.functional.pad(cur_mask, (sink_cols, 0), value=False)
            scores = scores.masked_fill(cur_mask, -40000.0)
        if cur_mask is None and sparse_mode in (2, 3):
            rows = torch.arange(q_len).view(q_len, 1)
            cols = torch.arange(kv_len).view(1, kv_len)
            if sparse_mode == 3:
                mask = cols > (kv_len - q_len + rows)
            else:
                mask = cols > rows
            if sink_cols:
                mask = torch.nn.functional.pad(mask, (sink_cols, 0), value=False)
            scores = scores.masked_fill(mask.unsqueeze(0), -40000.0)
        elif cur_mask is None and (pre_tokens < (1 << 30) or next_tokens < (1 << 30)):
            rows = torch.arange(q_len).view(q_len, 1)
            cols = torch.arange(kv_len).view(1, kv_len)
            center = kv_len - q_len + rows
            mask = (cols < center - int(pre_tokens)) | (cols > center + int(next_tokens))
            if sink_cols:
                mask = torch.nn.functional.pad(mask, (sink_cols, 0), value=False)
            scores = scores.masked_fill(mask.unsqueeze(0), -40000.0)
        probs = torch.softmax(scores, dim=-1)
        if keep_prob <= 0:
            probs = torch.zeros_like(probs)
        y = torch.matmul(probs, v).permute(1, 0, 2)
        out[b, :q_len] = y
        m = scores.max(dim=-1).values
        s = torch.exp(scores - m.unsqueeze(-1)).sum(dim=-1)
        softmax_max[b, :, :q_len] = m.unsqueeze(-1).expand(-1, -1, 8)
        softmax_sum[b, :, :q_len] = s.unsqueeze(-1).expand(-1, -1, 8)
    return from_bsnd(out, layout, q_lens), from_bsnd(softmax_max.permute(0, 2, 1, 3), layout, q_lens), from_bsnd(softmax_sum.permute(0, 2, 1, 3), layout, q_lens)


def attention_forward_native(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: Optional[int] = None,
    input_layout: str = "TND",
    pse: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    scale_value: float = None,
    keep_prob: float = 1.0,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode: int = 0,
    pre_tokens: int = (1 << 31) - 1,
    next_tokens: int = (1 << 31) - 1,
    sink_num: int = 0,
):
    layout = input_layout
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    batch = len(actual_seq_qlen) if actual_seq_qlen is not None else (query.shape[0] if layout == "BSND" else 1)
    q_lens = as_int_list(actual_seq_qlen, q_total, batch, layout == "TND", default_mode="split_total" if layout == "TND" else "per_batch_full")

    sink_len = int(sink_num) * 64 if layout == "TND" else 0
    kv_main_total = _seq_total_from_attr(actual_seq_kvlen) if layout == "TND" else None
    key_sink = value_sink = key_rope_sink = None
    if sink_len > 0:
        key_sink, key = _split_sink_prefix(key, sink_len, kv_main_total)
        value_sink, value = _split_sink_prefix(value, sink_len, kv_main_total)
        if key_rope is not None and key_rope.numel():
            key_rope_sink, key_rope = _split_sink_prefix(key_rope, sink_len, kv_main_total)

    kv_total = kv_main_total if layout == "TND" else key.shape[1]
    if kv_total is None:
        kv_total = key.shape[0] if layout == "TND" else key.shape[1]
    kv_lens = as_int_list(actual_seq_kvlen, kv_total, len(q_lens), layout == "TND", default_mode="split_total" if layout == "TND" else "per_batch_full")
    q_b = to_bsnd_device(query, layout, q_lens)
    k_b = to_bsnd_device(key, layout, kv_lens)
    v_b = to_bsnd_device(value, layout, kv_lens)
    if query_rope is not None and key_rope is not None and query_rope.numel() and key_rope.numel():
        qr_b = to_bsnd_device(query_rope, layout, q_lens)
        kr_b = to_bsnd_device(key_rope, layout, kv_lens)
        q_b = torch.cat([q_b, qr_b], dim=-1)
        k_b = torch.cat([k_b, kr_b], dim=-1)
        if key_rope_sink is not None:
            key_sink = torch.cat([key_sink, key_rope_sink], dim=-1)
    if key_sink is not None:
        key_sink = key_sink.to(query.device)
        value_sink = value_sink.to(query.device)

    n1 = int(head_num or q_b.shape[-2])
    out = torch.zeros((len(q_lens), max(q_lens, default=0), q_b.shape[-2], v_b.shape[-1]), dtype=query.dtype, device=query.device)
    softmax_max = torch.zeros((len(q_lens), n1, max(q_lens, default=0), 8), dtype=torch.float32, device=query.device)
    softmax_sum = torch.zeros_like(softmax_max)

    scale_t = torch.tensor(float(scale), dtype=query.dtype, device=query.device)
    for b, q_len in enumerate(q_lens):
        kv_len = kv_lens[min(b, len(kv_lens) - 1)] if kv_lens else 0
        if q_len <= 0 or kv_len <= 0:
            continue
        q = q_b[b, :q_len].permute(1, 0, 2)
        k = broadcast_kv(k_b[b, :kv_len], n1).permute(1, 0, 2)
        v = broadcast_kv(v_b[b, :kv_len], n1).permute(1, 0, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale_t
        sink_cols = 0
        if key_sink is not None and value_sink is not None:
            cur_key_sink = broadcast_kv(key_sink, n1).permute(1, 0, 2)
            cur_value_sink = broadcast_kv(value_sink, n1).permute(1, 0, 2)
            scores = torch.cat([torch.matmul(q, cur_key_sink.transpose(-1, -2)) * scale_t, scores], dim=-1)
            v = torch.cat([cur_value_sink, v], dim=-2)
            sink_cols = cur_key_sink.shape[-2]
        cur_pse = _slice_optional_device(pse, b, q_len, kv_len)
        if cur_pse is not None:
            cur_pse = cur_pse.reshape(-1, q_len, kv_len)[: scores.shape[0]].to(scores.dtype)
            if sink_cols:
                cur_pse = torch.nn.functional.pad(cur_pse, (sink_cols, 0), value=0)
            scores = scores + cur_pse
        cur_mask = _slice_optional_device(atten_mask, b, q_len, kv_len)
        if cur_mask is not None:
            cur_mask = cur_mask.reshape(-1, q_len, kv_len).bool()[: scores.shape[0]]
            if sink_cols:
                cur_mask = torch.nn.functional.pad(cur_mask, (sink_cols, 0), value=False)
            scores = scores.masked_fill(cur_mask, torch.finfo(scores.dtype).min)
        if cur_mask is None and sparse_mode in (2, 3):
            rows = torch.arange(q_len, device=query.device).view(q_len, 1)
            cols = torch.arange(kv_len, device=query.device).view(1, kv_len)
            mask = cols > (kv_len - q_len + rows) if sparse_mode == 3 else cols > rows
            if sink_cols:
                mask = torch.nn.functional.pad(mask, (sink_cols, 0), value=False)
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        elif cur_mask is None and (pre_tokens < (1 << 30) or next_tokens < (1 << 30)):
            rows = torch.arange(q_len, device=query.device).view(q_len, 1)
            cols = torch.arange(kv_len, device=query.device).view(1, kv_len)
            center = kv_len - q_len + rows
            mask = (cols < center - int(pre_tokens)) | (cols > center + int(next_tokens))
            if sink_cols:
                mask = torch.nn.functional.pad(mask, (sink_cols, 0), value=False)
            scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if keep_prob <= 0:
            probs = torch.zeros_like(probs)
        y = torch.matmul(probs, v).permute(1, 0, 2)
        out[b, :q_len] = y
        m = scores.to(torch.float32).max(dim=-1).values
        s = torch.exp(scores.to(torch.float32) - m.unsqueeze(-1)).sum(dim=-1)
        softmax_max[b, :, :q_len] = m.unsqueeze(-1).expand(-1, -1, 8)
        softmax_sum[b, :, :q_len] = s.unsqueeze(-1).expand(-1, -1, 8)
    return from_bsnd(out, layout, q_lens), from_bsnd(softmax_max.permute(0, 2, 1, 3), layout, q_lens), from_bsnd(softmax_sum.permute(0, 2, 1, 3), layout, q_lens)


def flash_attention_score_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: int,
    input_layout: str = "TND",
    pse: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    sink_tensor: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    scale_value: float = None,
    keep_prob: float = 1.0,
    pre_tokens: int = (1 << 31) - 1,
    next_tokens: int = (1 << 31) - 1,
    inner_precise: int = 0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode: int = 0,
    sink_num: int = 0,
    pse_type: int = 1,
    softmaxOutLayout: str = "",
    q_start_idx=None,
    kv_start_idx=None,
    **kwargs,
):
    del padding_mask, sink_tensor, inner_precise, prefix, pse_type, softmaxOutLayout, q_start_idx, kv_start_idx, kwargs
    if scale_value is not None:
        scale = scale_value
    forward = attention_forward_native if (
        isinstance(query, torch.Tensor)
        and query.device.type == "npu"
        and query.dtype in (torch.float16, torch.bfloat16)
    ) else attention_forward
    forward_atten_mask = _logical_atten_mask_for_forward(
        atten_mask,
        query,
        key,
        input_layout=input_layout,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        sink_num=sink_num,
    )
    out, softmax_max, softmax_sum = forward(
        query,
        key,
        value,
        head_num=head_num,
        input_layout=input_layout,
        pse=pse,
        atten_mask=forward_atten_mask,
        query_rope=query_rope,
        key_rope=key_rope,
        scale=scale,
        keep_prob=keep_prob,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        sink_num=sink_num,
    )
    return out, softmax_max.float(), softmax_sum.float()


def _seq_total_from_attr(seq_lens):
    if seq_lens is None:
        return None
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.detach().cpu().reshape(-1).tolist()
    if isinstance(seq_lens, tuple):
        seq_lens = list(seq_lens)
    if not isinstance(seq_lens, list) or not seq_lens:
        return None
    values = [int(item) for item in seq_lens]
    if all(a < b for a, b in zip(values, values[1:])):
        return values[-1]
    return sum(values)


def _normalize_tnd_seq_lens(seq_lens, total):
    if not isinstance(seq_lens, list) or not seq_lens or total is None:
        return
    if sum(int(x) for x in seq_lens) == int(total) and not (
        all(int(a) <= int(b) for a, b in zip(seq_lens, seq_lens[1:]))
        and int(seq_lens[-1]) == int(total)
    ):
        cur = 0
        for idx, item in enumerate(seq_lens):
            cur += int(item)
            seq_lens[idx] = cur


def get_input(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pse=None, atten_mask=None, query_rope=None, key_rope=None, **kwargs):
    input_layout = kwargs.get("input_layout", "TND")
    sink_num = int(kwargs.get("sink_num", 0) or 0)
    sink_len = sink_num * 64 if input_layout == "TND" else 0
    if input_layout == "TND":
        _normalize_tnd_seq_lens(kwargs.get("actual_seq_qlen"), query.shape[0])
        kv_main_total = _seq_total_from_attr(kwargs.get("actual_seq_kvlen"))
        if kv_main_total is None:
            kv_main_total = key.shape[0] - sink_len if sink_len and key.shape[0] > sink_len else key.shape[0]
        _normalize_tnd_seq_lens(kwargs.get("actual_seq_kvlen"), kv_main_total)
    if (
        query_rope is None
        and key_rope is None
        and isinstance(pse, torch.Tensor)
        and isinstance(atten_mask, torch.Tensor)
        and pse.shape[:-1] == query.shape[:-1]
        and atten_mask.shape[:-1] == key.shape[:-1]
        and pse.dtype.is_floating_point
        and atten_mask.dtype.is_floating_point
    ):
        query_rope, key_rope = pse, atten_mask
        pse = atten_mask = None
    if atten_mask is None and int(kwargs.get("sparse_mode", 0)) in (2, 3, 4):
        atten_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.uint8), diagonal=1)
    if input_layout == "TND" and sink_len > 0:
        kv_main_total = _seq_total_from_attr(kwargs.get("actual_seq_kvlen"))
        if kv_main_total is not None and key.shape[0] == kv_main_total:
            key_with_sink = torch.cat([key[:sink_len].clone(), key], dim=0)
            value_with_sink = torch.cat([value[:sink_len].clone(), value], dim=0)
            key = key_with_sink
            value = value_with_sink
            if key_rope is not None:
                key_rope_with_sink = torch.cat([key_rope[:sink_len].clone(), key_rope], dim=0)
                key_rope = key_rope_with_sink
    return [query, key, value, pse, None, atten_mask, None, query_rope, key_rope]
