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

def sparse_lightning_kl_grad(query, key, query_index, key_index, weights, topk_index,
                             scale_value=1.0, layout="TND", actual_seq_lengths_query=None,
                             actual_seq_lengths_key=None, sparse_mode=3):
    with torch.enable_grad():
        return _sparse_lightning_kl_grad_impl(
            query,
            key,
            query_index,
            key_index,
            weights,
            topk_index,
            scale_value=scale_value,
            layout=layout,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            sparse_mode=sparse_mode,
        )


def _sparse_lightning_kl_grad_impl(query, key, query_index, key_index, weights, topk_index,
                                   scale_value=1.0, layout="TND", actual_seq_lengths_query=None,
                                   actual_seq_lengths_key=None, sparse_mode=3):
    batch = len(actual_seq_lengths_query) if actual_seq_lengths_query is not None else (query.shape[0] if layout == "BSND" else 1)
    q_lens = as_int_list(
        actual_seq_lengths_query,
        query.shape[0] if layout == "TND" else query.shape[1],
        batch,
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    k_lens = as_int_list(
        actual_seq_lengths_key,
        key.shape[0] if layout == "TND" else key.shape[1],
        len(q_lens),
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    calc_device = query.device if native else torch.device("cpu")
    calc_dtype = query.dtype if native else torch.float32
    layout_fn = to_bsnd_device if native else to_bsnd_grad
    q_b = layout_fn(query, layout, q_lens).detach().to(calc_device).to(calc_dtype)
    k_b = layout_fn(key, layout, k_lens).detach().to(calc_device).to(calc_dtype)
    qi_b = layout_fn(query_index, layout, q_lens).detach().to(calc_device).to(calc_dtype)
    ki_b = layout_fn(key_index, layout, k_lens).detach().to(calc_device).to(calc_dtype)
    w_b = layout_fn(weights.unsqueeze(-1), layout, q_lens).squeeze(-1).detach().to(calc_device).to(calc_dtype) if layout == "TND" else weights.detach().to(calc_device).to(calc_dtype)
    idx_b = topk_index.detach().to(calc_device)
    if layout == "TND":
        idx_b = layout_fn(idx_b, layout, q_lens)
    dqi = torch.zeros_like(qi_b)
    dki = torch.zeros_like(ki_b)
    dw = torch.zeros_like(w_b)
    loss = torch.zeros(1, dtype=torch.float32, device=calc_device)
    n2 = k_b.shape[-2]
    n1 = q_b.shape[-2]
    group = max(n1 // n2, 1)
    for b, q_len in enumerate(q_lens):
        kv_len = k_lens[min(b, len(k_lens) - 1)] if k_lens else 0
        for s in range(q_len):
            for kv_head in range(n2):
                raw_positions = idx_b[b, s, kv_head].to(torch.int64).reshape(-1).tolist()
                positions = []
                for raw in raw_positions:
                    raw = int(raw)
                    if raw < 0:
                        continue
                    if kv_len <= 0:
                        continue
                    positions.append(raw % kv_len)
                if not positions:
                    continue
                heads = slice(kv_head * group, (kv_head + 1) * group)
                q_prob = q_b[b, s, heads]
                k_prob = k_b[b, positions, kv_head]
                p_scores = torch.matmul(q_prob, k_prob.transpose(-1, -2)) * float(scale_value)
                if sparse_mode == 3:
                    threshold = kv_len - q_len + s + 1
                    p_scores = p_scores.masked_fill(torch.tensor(positions, device=calc_device).view(1, -1) >= threshold, -40000.0)
                p = torch.softmax(p_scores, dim=-1).reshape(n2, group, len(positions)).mean(dim=1)[kv_head]
                q_idx = qi_b[b, s, heads].clone().requires_grad_(True)
                k_idx = ki_b[b, positions, kv_head].clone().requires_grad_(True)
                w = w_b[b, s, heads].clone().requires_grad_(True)
                s_scores = torch.relu(torch.matmul(q_idx, k_idx.transpose(-1, -2))) * w.unsqueeze(-1)
                s_reduced = s_scores.sum(dim=0)
                pred = torch.softmax(s_reduced, dim=-1)
                cur_loss = torch.sum(p.clamp_min(1e-8) * (p.clamp_min(1e-8).log() - pred.clamp_min(1e-8).log()))
                cur_loss.backward()
                loss += cur_loss.detach()
                dqi[b, s, heads] += q_idx.grad
                dki[b, positions, kv_head] += k_idx.grad
                dw[b, s, heads] += w.grad
    return from_bsnd(dqi, layout, q_lens), from_bsnd(dki, layout, k_lens), from_bsnd(dw, layout, q_lens), loss



def sparse_lightning_indexer_grad_klloss_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    topk_index: torch.Tensor,
    softmax_max: torch.Tensor = None,
    softmax_sum: torch.Tensor = None,
    query_rope: torch.Tensor = None,
    key_rope: torch.Tensor = None,
    scale_value: float = 1.0,
    layout: str = "TND",
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    sparse_mode: int = 3,
    deterministic: bool = True,
    sparse_block_size: int = 1,
):
    del softmax_max, softmax_sum, query_rope, key_rope, deterministic, sparse_block_size
    return sparse_lightning_kl_grad(
        query,
        key,
        query_index,
        key_index,
        weights,
        topk_index,
        scale_value=scale_value,
        layout=layout,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        sparse_mode=sparse_mode,
    )


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


def _build_st_sparse_indices(topk_index, q_lens, k_lens, sparse_mode, sparse_block_size):
    if topk_index is None:
        return None
    out = torch.full_like(topk_index, -1)
    topk = int(topk_index.shape[-1])
    n2 = int(topk_index.shape[-2])
    t = 0
    for b, q_len in enumerate(q_lens):
        kv_len = k_lens[min(b, len(k_lens) - 1)] if k_lens else 0
        for s in range(int(q_len)):
            if sparse_mode == 3:
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


def _to_bsnd_for_stats(x, layout, lens):
    if layout == "TND":
        return to_bsnd_grad(x, layout, lens).to(torch.float32)
    return x.detach().cpu().to(torch.float32)


def _zero_rope_like(x, rope_dim=64):
    if x is None:
        return None
    shape = list(x.shape)
    if not shape:
        return None
    shape[-1] = int(rope_dim)
    return torch.zeros(shape, dtype=x.dtype, device=x.device)


def _cat_rope_for_stats(x, rope):
    if x is None or rope is None:
        return x
    return torch.cat((x, rope.to(device=x.device, dtype=x.dtype)), dim=-1)


def _build_st_softmax_stats(query, key, topk_index, layout, actual_seq_lengths_query,
                            actual_seq_lengths_key, scale_value, sparse_block_size,
                            query_rope=None, key_rope=None):
    if query is None or key is None or topk_index is None:
        return None, None
    query = _cat_rope_for_stats(query, query_rope)
    key = _cat_rope_for_stats(key, key_rope)
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    k_total = key.shape[0] if layout == "TND" else key.shape[1]
    q_lens = _prefix_or_lengths(actual_seq_lengths_query, q_total)
    k_lens = _prefix_or_lengths(actual_seq_lengths_key, k_total)
    q_b = _to_bsnd_for_stats(query, layout, q_lens)
    k_b = _to_bsnd_for_stats(key, layout, k_lens)
    idx = topk_index.detach().cpu().to(torch.int64)
    if layout == "TND":
        idx = to_bsnd_grad(idx, layout, q_lens)
    if int(sparse_block_size) != 1:
        start = idx * int(sparse_block_size)
        offset = torch.arange(int(sparse_block_size), dtype=idx.dtype).view(1, 1, 1, 1, -1)
        idx = (start.unsqueeze(-1) + offset).reshape(*idx.shape[:-1], idx.shape[-1] * int(sparse_block_size))

    bsz, max_q, n1, dim = q_b.shape
    n2 = k_b.shape[2]
    group = max(n1 // max(n2, 1), 1)
    max_values = []
    sum_values = []
    for b in range(bsz):
        q_len = int(q_lens[min(b, len(q_lens) - 1)])
        kv_len = int(k_lens[min(b, len(k_lens) - 1)])
        if q_len <= 0:
            continue
        q_cur = q_b[b, :q_len]
        k_cur = k_b[b, :kv_len]
        idx_cur = idx[b, :q_len]
        p_rows_max = []
        p_rows_sum = []
        for s in range(q_len):
            head_max = []
            head_sum = []
            for kv_head in range(n2):
                raw = idx_cur[s, kv_head].reshape(-1)
                safe = raw.clamp(min=0, max=max(kv_len - 1, 0))
                key_topk = k_cur[safe, kv_head]
                scores = torch.matmul(q_cur[s, kv_head * group:(kv_head + 1) * group], key_topk.transpose(-1, -2))
                scores = scores * float(scale_value)
                invalid = (raw < 0) | (raw >= kv_len)
                if invalid.any():
                    scores[:, invalid] = -float("inf")
                score_max = torch.max(scores, dim=-1, keepdim=True).values
                score_sum = torch.exp(scores - score_max).sum(dim=-1, keepdim=True)
                head_max.append(score_max)
                head_sum.append(score_sum)
            p_rows_max.append(torch.cat(head_max, dim=0).squeeze(-1))
            p_rows_sum.append(torch.cat(head_sum, dim=0).squeeze(-1))
        max_values.append(torch.stack(p_rows_max, dim=0))
        sum_values.append(torch.stack(p_rows_sum, dim=0))
    if not max_values:
        empty = torch.empty((1, 0, n1), dtype=torch.float32)
        return empty, empty
    if layout == "TND":
        return torch.cat(max_values, dim=0).unsqueeze(0), torch.cat(sum_values, dim=0).unsqueeze(0)
    return torch.stack(max_values, dim=0).unsqueeze(0).unsqueeze(0), torch.stack(sum_values, dim=0).unsqueeze(0).unsqueeze(0)


def get_input(
    query: torch.Tensor = None,
    key: torch.Tensor = None,
    query_index: torch.Tensor = None,
    key_index: torch.Tensor = None,
    weights: torch.Tensor = None,
    topk_index: torch.Tensor = None,
    softmax_max: torch.Tensor = None,
    softmax_sum: torch.Tensor = None,
    query_rope: torch.Tensor = None,
    key_rope: torch.Tensor = None,
    **kwargs,
):
    layout = kwargs.get("layout", "TND")
    sparse_mode = int(kwargs.get("sparse_mode", 3))
    sparse_block_size = int(kwargs.get("sparse_block_size", 1))
    if query_rope is None and query is not None and query.shape[-1] == 512:
        query_rope = _zero_rope_like(query, kwargs.get("rope_head_dim", 64))
    if key_rope is None and key is not None and key.shape[-1] == 512:
        key_rope = _zero_rope_like(key, kwargs.get("rope_head_dim", 64))
    if query is not None and key is not None and topk_index is not None:
        q_total = query.shape[0] if layout == "TND" else query.shape[1]
        k_total = key.shape[0] if layout == "TND" else key.shape[1]
        q_lens = _prefix_or_lengths(kwargs.get("actual_seq_lengths_query"), q_total)
        k_lens = _prefix_or_lengths(kwargs.get("actual_seq_lengths_key"), k_total)
        topk_index = _build_st_sparse_indices(topk_index, q_lens, k_lens, sparse_mode, sparse_block_size)
    if softmax_max is None or softmax_sum is None:
        st_max, st_sum = _build_st_softmax_stats(
            query,
            key,
            topk_index,
            layout,
            kwargs.get("actual_seq_lengths_query"),
            kwargs.get("actual_seq_lengths_key"),
            kwargs.get("scale_value", 1.0),
            sparse_block_size,
            query_rope=query_rope,
            key_rope=key_rope,
        )
        if softmax_max is None:
            softmax_max = st_max
        if softmax_sum is None:
            softmax_sum = st_sum
    return [query, key, query_index, key_index, weights, topk_index, softmax_max, softmax_sum, query_rope, key_rope]
