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

def _selected_positions(indices: torch.Tensor, block_size: int, kv_len: int):
    positions = []
    for sparse_id in indices.detach().cpu().to(torch.int64).reshape(-1).tolist():
        if sparse_id < 0:
            break
        begin = int(sparse_id) * int(block_size)
        end = min(begin + int(block_size), int(kv_len))
        if begin < kv_len:
            positions.extend(range(begin, end))
    return positions


def _prefix_or_lengths(value, total):
    if value is None:
        return [int(total)]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (int, float)):
        value = [int(value)]
    out = [int(v) for v in value]
    if not out:
        return [int(total)]
    if len(out) > 1 and all(a <= b for a, b in zip(out, out[1:])) and out[-1] == int(total):
        prev = 0
        lens = []
        for item in out:
            lens.append(item - prev)
            prev = item
        return lens
    return out


def _build_st_sparse_indices(topk_indices, q_lens, kv_lens, sparse_mode, sparse_block_size):
    if topk_indices is None:
        return None
    out = torch.full_like(topk_indices, -1)
    topk = int(topk_indices.shape[-1])
    n2 = int(topk_indices.shape[-2])
    t = 0
    for b, q_len in enumerate(q_lens):
        kv_len = kv_lens[min(b, len(kv_lens) - 1)] if kv_lens else 0
        for s in range(int(q_len)):
            if int(sparse_mode) == 3:
                valid = int(kv_len) - int(q_len) + s + 1
                if valid <= 0:
                    valid = int(kv_len)
            else:
                valid = int(kv_len)
            valid_blocks = max((valid + int(sparse_block_size) - 1) // int(sparse_block_size), 0)
            count = min(topk, valid_blocks)
            if count > 0:
                row = torch.arange(count, dtype=out.dtype, device=out.device)
                if out.dim() == 3:
                    out[t, :n2, :count] = row.view(1, count).expand(n2, count)
                else:
                    out[b, s, :n2, :count] = row.view(1, count).expand(n2, count)
            t += 1
    return out


def _sparse_attention_forward_inputs(query, key, value, topk_indices, query_rope=None, key_rope=None,
                                     actual_seq_qlen=None, actual_seq_kvlen=None, scale_value=1.0,
                                     sparse_block_size=1, layout="TND", sparse_mode=3):
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    kv_total = key.shape[0] if layout == "TND" else key.shape[1]
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
    q_b = to_bsnd_grad(query, layout, q_lens).detach().cpu().to(torch.float32)
    k_b = to_bsnd_grad(key, layout, kv_lens).detach().cpu().to(torch.float32)
    v_b = to_bsnd_grad(value, layout, kv_lens).detach().cpu().to(torch.float32)
    idx_b = topk_indices.detach().cpu()
    if layout == "TND":
        idx_b = to_bsnd_grad(idx_b, layout, q_lens)

    qr_b = kr_b = None
    if isinstance(query_rope, torch.Tensor) and isinstance(key_rope, torch.Tensor) and query_rope.numel() and key_rope.numel():
        qr_b = to_bsnd_grad(query_rope, layout, q_lens).detach().cpu().to(torch.float32)
        kr_b = to_bsnd_grad(key_rope, layout, kv_lens).detach().cpu().to(torch.float32)

    n1 = q_b.shape[-2]
    n2 = k_b.shape[-2]
    group = max(n1 // n2, 1)
    out_b = torch.zeros_like(q_b)
    max_b = torch.zeros((len(q_lens), max(q_lens, default=0), n2, group), dtype=torch.float32)
    sum_b = torch.ones_like(max_b)
    for b, q_len in enumerate(q_lens):
        kv_len = kv_lens[min(b, len(kv_lens) - 1)] if kv_lens else 0
        for s_idx in range(q_len):
            for kv_head in range(n2):
                positions = _selected_positions(idx_b[b, s_idx, kv_head], sparse_block_size, kv_len)
                if not positions:
                    continue
                heads = slice(kv_head * group, (kv_head + 1) * group)
                q = q_b[b, s_idx, heads]
                k = k_b[b, positions, kv_head]
                if qr_b is not None and kr_b is not None:
                    q = torch.cat([q, qr_b[b, s_idx, heads]], dim=-1)
                    k = torch.cat([k, kr_b[b, positions, kv_head]], dim=-1)
                scores = torch.matmul(q, k.transpose(-1, -2)) * float(scale_value)
                if sparse_mode == 3:
                    threshold = kv_len - q_len + s_idx + 1
                    mask = torch.tensor(positions, dtype=torch.int64).view(1, -1) >= threshold
                    scores = scores.masked_fill(mask, -40000.0)
                row_max = torch.max(scores, dim=-1, keepdim=True)[0]
                exp_scores = torch.exp(scores - row_max)
                row_sum = exp_scores.sum(dim=-1, keepdim=True)
                probs = exp_scores / row_sum
                out_b[b, s_idx, heads] = torch.matmul(probs, v_b[b, positions, kv_head])
                max_b[b, s_idx, kv_head] = row_max.squeeze(-1)
                sum_b[b, s_idx, kv_head] = row_sum.squeeze(-1)

    attention_out = from_bsnd(out_b, layout, q_lens).to(query.dtype)
    if layout == "TND":
        softmax_max = torch.cat([max_b[b, : int(seq_len)] for b, seq_len in enumerate(q_lens)], dim=0)
        softmax_sum = torch.cat([sum_b[b, : int(seq_len)] for b, seq_len in enumerate(q_lens)], dim=0)
        softmax_max = softmax_max.permute(1, 0, 2).contiguous()
        softmax_sum = softmax_sum.permute(1, 0, 2).contiguous()
    else:
        softmax_max = max_b.permute(0, 2, 1, 3).contiguous()
        softmax_sum = sum_b.permute(0, 2, 1, 3).contiguous()
    return attention_out, softmax_max, softmax_sum


def sparse_attention_grad(query, key, value, topk_indices, dy, attention_out=None, softmax_max=None,
                          softmax_sum=None, query_rope=None, key_rope=None, actual_seq_qlen=None,
                          actual_seq_kvlen=None, scale_value=1.0, sparse_block_size=1,
                          layout="TND", sparse_mode=3):
    with torch.enable_grad():
        return _sparse_attention_grad_impl(
            query,
            key,
            value,
            topk_indices,
            dy,
            attention_out=attention_out,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            scale_value=scale_value,
            sparse_block_size=sparse_block_size,
            layout=layout,
            sparse_mode=sparse_mode,
        )


def _sparse_attention_grad_impl(query, key, value, topk_indices, dy, attention_out=None, softmax_max=None,
                                softmax_sum=None, query_rope=None, key_rope=None, actual_seq_qlen=None,
                                actual_seq_kvlen=None, scale_value=1.0, sparse_block_size=1,
                                layout="TND", sparse_mode=3):
    del attention_out, softmax_max, softmax_sum
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    batch = len(actual_seq_qlen) if actual_seq_qlen is not None else (query.shape[0] if layout == "BSND" else 1)
    q_lens = as_int_list(
        actual_seq_qlen,
        q_total,
        batch,
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    kv_total = key.shape[0] if layout == "TND" else key.shape[1]
    kv_lens = as_int_list(
        actual_seq_kvlen,
        kv_total,
        len(q_lens),
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    calc_device = query.device if native else torch.device("cpu")
    calc_dtype = query.dtype if native else torch.float32
    layout_fn = to_bsnd_device if native else to_bsnd_grad
    q_b = layout_fn(query, layout, q_lens).detach().to(calc_device).to(calc_dtype)
    k_b = layout_fn(key, layout, kv_lens).detach().to(calc_device).to(calc_dtype)
    v_b = layout_fn(value, layout, kv_lens).detach().to(calc_device).to(calc_dtype)
    dy_b = layout_fn(dy, layout, q_lens).detach().to(calc_device).to(calc_dtype)
    idx_b = topk_indices.detach().to(calc_device)
    if layout == "TND":
        idx_b = layout_fn(idx_b, layout, q_lens)
    qr_b = kr_b = None
    if isinstance(query_rope, torch.Tensor) and isinstance(key_rope, torch.Tensor) and query_rope.numel() and key_rope.numel():
        qr_b = layout_fn(query_rope, layout, q_lens).detach().to(calc_device).to(calc_dtype)
        kr_b = layout_fn(key_rope, layout, kv_lens).detach().to(calc_device).to(calc_dtype)
    n1 = q_b.shape[-2]
    n2 = k_b.shape[-2]
    group = max(n1 // n2, 1)
    dq = torch.zeros_like(q_b)
    dk = torch.zeros_like(k_b)
    dv = torch.zeros_like(v_b)
    dqr = torch.zeros_like(qr_b) if qr_b is not None else None
    dkr = torch.zeros_like(kr_b) if kr_b is not None else None
    for b, q_len in enumerate(q_lens):
        kv_len = kv_lens[min(b, len(kv_lens) - 1)] if kv_lens else 0
        for s in range(q_len):
            for kv_head in range(n2):
                positions = _selected_positions(idx_b[b, s, kv_head], sparse_block_size, kv_len)
                if not positions:
                    continue
                heads = slice(kv_head * group, (kv_head + 1) * group)
                q = q_b[b, s, heads]
                k = k_b[b, positions, kv_head]
                if qr_b is not None and kr_b is not None:
                    q_full = torch.cat([q, qr_b[b, s, heads]], dim=-1).requires_grad_(True)
                    k_full = torch.cat([k, kr_b[b, positions, kv_head]], dim=-1).requires_grad_(True)
                    org_dim = q.shape[-1]
                else:
                    q_full = q.clone().requires_grad_(True)
                    k_full = k.clone().requires_grad_(True)
                    org_dim = q.shape[-1]
                v = v_b[b, positions, kv_head].clone().requires_grad_(True)
                scores = torch.matmul(q_full, k_full.transpose(-1, -2)) * float(scale_value)
                if sparse_mode == 3:
                    threshold = kv_len - q_len + s + 1
                    mask = torch.tensor(positions, device=calc_device).view(1, -1) >= threshold
                    scores = scores.masked_fill(mask, -40000.0)
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v)
                out.backward(dy_b[b, s, heads])
                dq[b, s, heads] += q_full.grad[:, :org_dim]
                dk[b, positions, kv_head] += k_full.grad[:, :org_dim]
                dv[b, positions, kv_head] += v.grad
                if dqr is not None and dkr is not None and q_full.shape[-1] > org_dim:
                    dqr[b, s, heads] += q_full.grad[:, org_dim:]
                    dkr[b, positions, kv_head] += k_full.grad[:, org_dim:]
    return (
        from_bsnd(dq, layout, q_lens),
        from_bsnd(dk, layout, kv_lens),
        from_bsnd(dv, layout, kv_lens),
        from_bsnd(dqr, layout, q_lens) if dqr is not None else torch.empty(0, device=calc_device),
        from_bsnd(dkr, layout, kv_lens) if dkr is not None else torch.empty(0, device=calc_device),
    )


def sparse_flash_attention_grad_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    dy: torch.Tensor,
    attention_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    scale_value: float = 1.0,
    sparse_block_size: int = 1,
    layout: str = "TND",
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 31) - 1,
    next_tokens: int = (1 << 31) - 1,
    deterministic: bool = False,
):
    del pre_tokens, next_tokens, deterministic
    return sparse_attention_grad(
        query,
        key,
        value,
        topk_indices,
        dy,
        attention_out=attention_out,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        scale_value=scale_value,
        sparse_block_size=sparse_block_size,
        layout=layout,
        sparse_mode=sparse_mode,
    )


def get_input(
    query: torch.Tensor = None,
    key: torch.Tensor = None,
    value: torch.Tensor = None,
    topk_indices: torch.Tensor = None,
    dy: torch.Tensor = None,
    attention_out: torch.Tensor = None,
    softmax_max: torch.Tensor = None,
    softmax_sum: torch.Tensor = None,
    *extra_tensors,
    **kwargs,
):
    actual_seq_qlen = kwargs.get("actual_seq_qlen")
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen")
    query_rope = extra_tensors[2] if len(extra_tensors) > 2 else None
    key_rope = extra_tensors[3] if len(extra_tensors) > 3 else None
    layout = kwargs.get("layout", "TND")
    sparse_mode = int(kwargs.get("sparse_mode", 3))
    sparse_block_size = int(kwargs.get("sparse_block_size", 1))
    scale_value = float(kwargs.get("scale_value", 1.0))
    if getattr(query, "is_meta", False):
        return [query, key, value, topk_indices, dy, attention_out, softmax_max, softmax_sum, query_rope, key_rope]
    if query is not None and key is not None:
        value = key
    if query is not None and key is not None and topk_indices is not None:
        q_total = query.shape[0] if layout == "TND" else query.shape[1]
        kv_total = key.shape[0] if layout == "TND" else key.shape[1]
        q_lens = _prefix_or_lengths(kwargs.get("actual_seq_qlen", actual_seq_qlen), q_total)
        kv_lens = _prefix_or_lengths(kwargs.get("actual_seq_kvlen", actual_seq_kvlen), kv_total)
        topk_indices = _build_st_sparse_indices(topk_indices, q_lens, kv_lens, sparse_mode, sparse_block_size)
        if value is not None:
            attention_out, softmax_max, softmax_sum = _sparse_attention_forward_inputs(
                query,
                key,
                value,
                topk_indices,
                query_rope=query_rope,
                key_rope=key_rope,
                actual_seq_qlen=q_lens,
                actual_seq_kvlen=kv_lens,
                scale_value=scale_value,
                sparse_block_size=sparse_block_size,
                layout=layout,
                sparse_mode=sparse_mode,
            )
    return [query, key, value, topk_indices, dy, attention_out, softmax_max, softmax_sum, query_rope, key_rope]
