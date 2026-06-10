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

def _align_heads(x: torch.Tensor, target_heads: int) -> torch.Tensor:
    if x.shape[1] == target_heads:
        return x
    if x.shape[1] > target_heads:
        return x[:, :target_heads]
    repeat = (target_heads + x.shape[1] - 1) // x.shape[1]
    return x.repeat_interleave(repeat, dim=1)[:, :target_heads]


def _lengths(lengths, total: int):
    batch = 1
    if lengths is not None:
        if isinstance(lengths, torch.Tensor):
            batch = max(int(lengths.numel()), 1)
        elif isinstance(lengths, (int, float)):
            batch = 1
        else:
            batch = max(len(lengths), 1)
    return as_int_list(lengths, total, batch, True, default_mode="per_batch_full")


def _stabilize_lengths(lengths, total: int):
    total = max(int(total), 0)
    n = len(lengths)
    if n == 0:
        return []
    weights = [max(int(x), 0) for x in lengths]
    if sum(weights) == 0:
        weights = [1] * n
    denom = float(sum(weights))
    raw = [total * w / denom for w in weights]
    out = [int(x) for x in raw]
    remain = total - sum(out)
    order = sorted(range(n), key=lambda idx: raw[idx] - out[idx], reverse=True)
    for idx in order[:remain]:
        out[idx] += 1
    return out


def _starts(lengths):
    starts = []
    cur = 0
    for length in lengths:
        starts.append(cur)
        cur += int(length)
    return starts


def _int_tensor_or_none(value, device=None):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.int32)
    if isinstance(value, (list, tuple)):
        return torch.tensor([int(item) for item in value], dtype=torch.int32, device=device)
    if isinstance(value, int):
        return torch.tensor([int(value)], dtype=torch.int32, device=device)
    return None


def _flatten_packed_kv(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    x = x.detach()
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-2], x.shape[-1])
    if x.dim() == 4:
        block_num, heads, block_tokens, dim = x.shape
        return x.permute(0, 2, 1, 3).reshape(block_num * block_tokens, heads, dim)
    if x.dim() == 5:
        block_num, heads, dim_tiles, block_tokens, tile = x.shape
        return x.permute(0, 3, 1, 2, 4).reshape(block_num * block_tokens, heads, dim_tiles * tile)
    return x.reshape(-1, num_heads, head_dim)


def _logical_shape(x: torch.Tensor):
    if x.dim() == 3:
        return int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if x.dim() == 4:
        return int(x.shape[0] * x.shape[2]), int(x.shape[1]), int(x.shape[3])
    if x.dim() == 5:
        return int(x.shape[0] * x.shape[3]), int(x.shape[1]), int(x.shape[2] * x.shape[4])
    return int(x.shape[0]), int(x.shape[-2] if x.dim() >= 2 else 1), int(x.shape[-1])


def _logical_dim(x: torch.Tensor) -> int:
    return _logical_shape(x)[2]


def _append_rope(base: torch.Tensor, rope: Optional[torch.Tensor], heads: int) -> torch.Tensor:
    if rope is None:
        return base
    rope_flat = _flatten_packed_kv(rope, heads, rope.shape[-1])
    if rope_flat.shape[0] != base.shape[0]:
        rope_flat = rope_flat[:base.shape[0]]
    if rope_flat.shape[1] != base.shape[1]:
        rope_flat = _align_heads(rope_flat.unsqueeze(0).transpose(1, 2), base.shape[1]).transpose(1, 2).squeeze(0)
    return torch.cat([base, rope_flat.to(base.dtype)], dim=-1)


def _window_mask(scores: torch.Tensor, q_start: int, kv_len: int, pre_tokens: int, next_tokens: int, sink: int):
    if scores.numel() == 0:
        return scores
    q_len = scores.shape[-2]
    total_k = scores.shape[-1]
    local_k = total_k - sink
    if local_k <= 0:
        return scores
    q_pos = torch.arange(q_start, q_start + q_len, dtype=torch.int64, device=scores.device).unsqueeze(1)
    k_pos = torch.arange(kv_len - local_k, kv_len, dtype=torch.int64, device=scores.device).unsqueeze(0)
    left = q_pos - max(int(pre_tokens), 0)
    right = q_pos + max(int(next_tokens), 0)
    mask = (k_pos < left) | (k_pos > right)
    scores[..., sink:] = scores[..., sink:].masked_fill(mask.unsqueeze(0), -torch.inf)
    return scores


def _apply_atten_mask(scores: torch.Tensor, atten_mask: Optional[torch.Tensor], q_offset: int, q_len: int, kv_len: int, sink: int):
    if atten_mask is None:
        return scores
    mask = atten_mask.detach().cpu().bool()
    def _slice_rows(mask_2d: torch.Tensor):
        if mask_2d.shape[0] == 1:
            return mask_2d.expand(q_len, -1)
        if q_offset + q_len <= mask_2d.shape[0]:
            return mask_2d[q_offset:q_offset + q_len]
        if mask_2d.shape[0] >= q_len:
            return mask_2d[:q_len]
        row_ids = torch.arange(q_offset, q_offset + q_len, dtype=torch.int64) % mask_2d.shape[0]
        return mask_2d.index_select(0, row_ids)

    def _slice_cols(mask_2d: torch.Tensor):
        if mask_2d.shape[1] == 1:
            return mask_2d.expand(-1, kv_len)
        if mask_2d.shape[1] >= kv_len:
            return mask_2d[:, :kv_len]
        # Smaller 2-D masks in these FIA cases describe the local trailing KV
        # window; earlier KV positions stay unmasked unless sparse/window rules
        # mask them separately.
        return torch.nn.functional.pad(mask_2d, (kv_len - mask_2d.shape[1], 0), value=False)

    if mask.dim() == 2:
        mask = _slice_cols(_slice_rows(mask))
        if sink:
            mask = torch.nn.functional.pad(mask, (sink, 0), value=False)
    else:
        while mask.dim() > 2:
            mask = mask[0]
        mask = _slice_cols(_slice_rows(mask))
        if sink:
            mask = torch.nn.functional.pad(mask, (sink, 0), value=False)
    if mask.shape[-2] < scores.shape[-2] or mask.shape[-1] < scores.shape[-1]:
        return scores
    mask = mask[:scores.shape[-2], :scores.shape[-1]].to(scores.device)
    while mask.dim() < scores.dim():
        mask = mask.unsqueeze(0)
    return scores.masked_fill(mask, -torch.inf)


def _finite_softmax(scores: torch.Tensor):
    row_max = scores.max(dim=-1, keepdim=True).values
    finite = torch.isfinite(row_max)
    shifted = torch.where(torch.isfinite(scores), scores - row_max, torch.full_like(scores, torch.finfo(scores.dtype).min))
    exp = torch.where(finite, torch.exp(shifted), torch.zeros_like(scores))
    denom = exp.sum(dim=-1, keepdim=True)
    probs = torch.divide(exp, denom.clamp_min(torch.finfo(exp.dtype).tiny))
    probs = torch.where(denom > 0, probs, torch.zeros_like(probs))
    lse = torch.full_like(denom, torch.inf)
    lse = torch.where(denom > 0, row_max + torch.log(denom), lse)
    return probs, lse


def _attention_slice(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    num_query_heads: int,
    num_key_value_heads: int,
    atten_mask: Optional[torch.Tensor],
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    q_abs_start: int,
    sink: int,
):
    native = q.device.type == "npu" and q.dtype in (torch.float16, torch.bfloat16)
    calc_device = q.device if native else torch.device("cpu")
    calc_dtype = q.dtype if native else torch.float32
    if q.shape[0] == 0:
        return torch.empty((0, num_query_heads, v.shape[-1]), dtype=calc_dtype, device=calc_device), torch.empty((0, num_query_heads, 1), dtype=torch.float32, device=calc_device)
    qh = q.detach().to(calc_device).to(calc_dtype).permute(1, 0, 2).unsqueeze(0)
    kh = k.detach().to(calc_device).to(calc_dtype).permute(1, 0, 2).unsqueeze(0)
    vh = v.detach().to(calc_device).to(calc_dtype).permute(1, 0, 2).unsqueeze(0)
    if num_key_value_heads and kh.shape[1] == num_key_value_heads and num_query_heads != num_key_value_heads:
        repeat = max(num_query_heads // max(num_key_value_heads, 1), 1)
        kh = kh.repeat_interleave(repeat, dim=1)
        vh = vh.repeat_interleave(repeat, dim=1)
    kh = _align_heads(kh, qh.shape[1])
    vh = _align_heads(vh, qh.shape[1])
    if qh.shape[-1] != kh.shape[-1]:
        # Random case metadata can omit or misclassify rope tensors. Valid
        # rope paths should reach equal Q/K dims before this compatibility cut.
        common_dim = min(qh.shape[-1], kh.shape[-1])
        qh = qh[..., :common_dim]
        kh = kh[..., :common_dim]
    outs = []
    lses = []
    local_kv_len = kh.shape[-2] - sink
    max_work = 8_000_000
    denom = max(int(kh.shape[-2]) * int(qh.shape[1]), 1)
    chunk = max(1, min(16, max_work // denom))
    for q_begin in range(0, qh.shape[-2], chunk):
        q_end = min(q_begin + chunk, qh.shape[-2])
        qh_chunk = qh[..., q_begin:q_end, :]
        scores = torch.matmul(qh_chunk, kh.transpose(-1, -2)) * scale
        if sparse_mode == 4:
            scores = _window_mask(scores, q_begin, local_kv_len, pre_tokens, next_tokens, sink)
        scores = _apply_atten_mask(scores, atten_mask, q_begin, q_end - q_begin, local_kv_len, sink)
        probs, lse_chunk = _finite_softmax(scores)
        outs.append(torch.matmul(probs, vh))
        lses.append(lse_chunk.squeeze(0).permute(1, 0, 2))
    out = torch.cat(outs, dim=-2) if outs else torch.empty((1, qh.shape[1], 0, v.shape[-1]), dtype=calc_dtype, device=calc_device)
    lse = torch.cat(lses, dim=0) if lses else torch.empty((0, qh.shape[1], 1), dtype=torch.float32, device=calc_device)
    return out.squeeze(0).permute(1, 0, 2), lse


def ai_infra_fused_infer_attention_sink(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    pse_shift: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_kvlen: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    dequant_scale_query: Optional[torch.Tensor] = None,
    dequant_scale_key: Optional[torch.Tensor] = None,
    dequant_offset_key: Optional[torch.Tensor] = None,
    dequant_scale_value: Optional[torch.Tensor] = None,
    dequant_offset_value: Optional[torch.Tensor] = None,
    dequant_scale_key_rope: Optional[torch.Tensor] = None,
    quant_scale_out: Optional[torch.Tensor] = None,
    quant_offset_out: Optional[torch.Tensor] = None,
    meta_data: Optional[torch.Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    softmax_scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "TND",
    sparse_mode: int = 0,
    block_size: int = 0,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    value_quant_mode: int = 0,
    inner_precise: int = 0,
    return_softmax_lse: bool = False,
    sink_number: int = 0,
    key_sink: Optional[torch.Tensor] = None,
    value_sink: Optional[torch.Tensor] = None,
    key_rope_sink: Optional[torch.Tensor] = None,
):
    """Golden for npu_fused_infer_attention_sink, covering the ST TND float path."""
    del pse_shift, block_table, dequant_scale_query
    del dequant_scale_key, dequant_offset_key, dequant_scale_value, dequant_offset_value
    del dequant_scale_key_rope, quant_scale_out, quant_offset_out, meta_data
    del block_size, query_quant_mode, key_quant_mode, value_quant_mode
    q_lens = _lengths(actual_seq_qlen, _logical_shape(query)[0])
    kv_lens = _lengths(actual_seq_kvlen, _logical_shape(key)[0])
    batch = len(q_lens)
    if len(kv_lens) < batch:
        kv_lens = kv_lens + [kv_lens[-1] if kv_lens else 0] * (batch - len(kv_lens))
    q_starts = _starts(q_lens)
    k_starts = _starts(kv_lens)
    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    if native and key_sink is not None and value_sink is not None:
        try:
            import torch_npu

            k_ref = torch.cat([key_sink, key], dim=0)
            v_ref = torch.cat([value_sink, value], dim=0)
            key_rope_ref = key_rope
            if key_rope is not None and key_rope_sink is not None:
                key_rope_ref = torch.cat([key_rope_sink, key_rope], dim=0)
            kv_len_ref = actual_seq_kvlen
            if isinstance(kv_len_ref, torch.Tensor):
                kv_len_ref = kv_len_ref.to(device=query.device) + int(key_sink.shape[0])
            elif kv_len_ref is not None:
                kv_len_ref = torch.tensor([int(x) + int(key_sink.shape[0]) for x in kv_len_ref], device=query.device, dtype=torch.int32)
            out_ref, lse_ref = torch_npu.npu_fused_infer_attention_score(
                query, k_ref, v_ref,
                actual_seq_lengths=actual_seq_qlen.to(device=query.device) if isinstance(actual_seq_qlen, torch.Tensor) else actual_seq_qlen,
                actual_seq_lengths_kv=kv_len_ref,
                query_rope=query_rope,
                key_rope=key_rope_ref,
                atten_mask=atten_mask,
                num_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads or num_query_heads,
                input_layout=input_layout,
                scale=softmax_scale,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                sparse_mode=sparse_mode,
                inner_precise=inner_precise,
                softmax_lse_flag=return_softmax_lse,
            )
            return (out_ref, lse_ref) if return_softmax_lse else (out_ref, torch.empty(1, device=query.device))
        except Exception:
            pass

    q = query.detach() if native else query.detach().cpu()
    query_rope_src = None if query_rope is None else (query_rope.detach() if native else query_rope.detach().cpu())
    key_src = key.detach() if native else key.detach().cpu()
    value_src = value.detach() if native else value.detach().cpu()
    key_rope_src = None if key_rope is None else (key_rope.detach() if native else key_rope.detach().cpu())
    q = _append_rope(q, query_rope_src, q.shape[1]) if query_rope_src is not None else q
    kv_heads = num_key_value_heads or (key.shape[-2] if key.dim() >= 3 else num_query_heads)
    k_base = _flatten_packed_kv(key_src, kv_heads, key.shape[-1])
    v_base = _flatten_packed_kv(value_src, kv_heads, value.shape[-1])
    if key_rope_src is not None:
        k_base = _append_rope(k_base, key_rope_src, kv_heads)
    outputs = []
    lses = []
    for b in range(batch):
        q_len = min(q_lens[b], max(query.shape[0] - q_starts[b], 0))
        kv_len = min(kv_lens[b], max(k_base.shape[0] - k_starts[b], 0))
        q_slice = q[q_starts[b]:q_starts[b] + q_len]
        k_slice = k_base[k_starts[b]:k_starts[b] + kv_len]
        v_slice = v_base[k_starts[b]:k_starts[b] + kv_len]
        sink = 0
        if sink_number and key_sink is not None and value_sink is not None:
            ks = key_sink.detach() if native else key_sink.detach().cpu()
            vs = value_sink.detach() if native else value_sink.detach().cpu()
            if key_rope_sink is not None:
                krs = key_rope_sink.detach() if native else key_rope_sink.detach().cpu()
                ks = torch.cat([ks, krs], dim=-1)
            if ks.shape[1] != k_slice.shape[1] and ks.shape[1] == 1:
                ks = ks.expand(-1, k_slice.shape[1], -1)
            if vs.shape[1] != v_slice.shape[1] and vs.shape[1] == 1:
                vs = vs.expand(-1, v_slice.shape[1], -1)
            k_slice = torch.cat([ks, k_slice], dim=0)
            v_slice = torch.cat([vs, v_slice], dim=0)
            sink = ks.shape[0]
        out_b, lse_b = _attention_slice(
            q_slice, k_slice, v_slice, softmax_scale, num_query_heads,
            num_key_value_heads, atten_mask, sparse_mode, pre_tokens, next_tokens,
            q_starts[b], sink,
        )
        outputs.append(out_b)
        lses.append(lse_b)
    out_dtype = query.dtype if native else torch.float32
    out_device = query.device if native else torch.device("cpu")
    out = torch.cat(outputs, dim=0) if outputs else torch.empty((0, num_query_heads, value.shape[-1]), dtype=out_dtype, device=out_device)
    lse = torch.cat(lses, dim=0) if lses else torch.empty((0, num_query_heads, 1), dtype=torch.float32, device=out_device)
    out = out[..., :value.shape[-1]]
    return (out, lse) if return_softmax_lse else (out, torch.empty(1, device=out_device))


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *extra_tensors,
    **kwargs,
):
    sink_number = int(kwargs.get("sink_number", 0) or 0)
    real_query_rope = None
    real_key_rope = None
    real_atten_mask = None
    real_key_sink = None
    real_value_sink = None
    real_key_rope_sink = None
    real_meta_data = None
    block_table = None
    int_dtypes = (torch.int32, torch.int64, torch.uint32)
    seq_types = (list, tuple, int)
    actual_seq_qlen = kwargs.get("actual_seq_qlen")
    actual_seq_kvlen = kwargs.get("actual_seq_kvlen")
    real_actual_seq_qlen = (
        actual_seq_qlen
        if (
            isinstance(actual_seq_qlen, torch.Tensor)
            and actual_seq_qlen.dtype in int_dtypes
            and actual_seq_qlen.dim() <= 1
        )
        or isinstance(actual_seq_qlen, seq_types)
        else None
    )
    real_actual_seq_kvlen = (
        actual_seq_kvlen
        if (
            isinstance(actual_seq_kvlen, torch.Tensor)
            and actual_seq_kvlen.dtype in int_dtypes
            and actual_seq_kvlen.dim() <= 1
        )
        or isinstance(actual_seq_kvlen, seq_types)
        else None
    )
    real_actual_seq_qlen = _int_tensor_or_none(real_actual_seq_qlen, query.device)
    real_actual_seq_kvlen = _int_tensor_or_none(real_actual_seq_kvlen, query.device)

    candidates = [(f"extra_{idx}", item) for idx, item in enumerate(extra_tensors)]
    tensor_candidates = []
    for name, item in candidates:
        if item is None:
            continue
        if not isinstance(item, torch.Tensor):
            continue
        if item.dtype in (torch.bool, torch.uint8, torch.int8) and item.dim() == 2:
            real_atten_mask = item
            continue
        if item is real_actual_seq_qlen or item is real_actual_seq_kvlen:
            continue
        if item.dtype in int_dtypes:
            real_meta_data = item
            continue
        if item.dim() >= 3:
            tensor_candidates.append((name, item))

    query_tokens, query_heads, query_dim = _logical_shape(query)
    key_tokens, key_heads, key_dim = _logical_shape(key)
    value_dim = _logical_dim(value)
    remaining = []
    for name, item in tensor_candidates:
        item_tokens, item_heads, item_dim = _logical_shape(item)
        if (item_tokens == query_tokens and item_heads in (query_heads, 1)
                and item_dim != query_dim and real_query_rope is None):
            real_query_rope = item
            continue
        if (item_tokens == key_tokens and item_heads in (key_heads, 1)
                and item_dim != key_dim and real_key_rope is None):
            real_key_rope = item
            continue
        remaining.append((name, item))

    key_rope_dim = _logical_dim(real_key_rope) if real_key_rope is not None else None
    for name, item in remaining:
        item_dim = _logical_dim(item)
        if sink_number and item.shape[0] == sink_number:
            if name == "key_sink" and real_key_sink is None:
                real_key_sink = item
            elif name == "value_sink" and real_value_sink is None:
                real_value_sink = item
            elif name == "key_rope_sink" and real_key_rope_sink is None:
                real_key_rope_sink = item
            elif item_dim == value_dim and real_value_sink is None:
                real_value_sink = item
            elif key_rope_dim is not None and item_dim == key_rope_dim and real_key_rope_sink is None:
                real_key_rope_sink = item
            elif item_dim == key_dim and real_key_sink is None:
                real_key_sink = item
            elif item_dim != key_dim and real_key_rope_sink is None:
                real_key_rope_sink = item
            elif real_key_sink is None:
                real_key_sink = item
            continue
        if item_dim == value_dim and real_value_sink is None:
            real_value_sink = item
        elif item_dim == key_dim and real_key_sink is None:
            real_key_sink = item
        elif real_key_rope_sink is None:
            real_key_rope_sink = item

    if (
        real_query_rope is None
        and real_key_rope is None
        and query_dim > value_dim
        and key_dim == query_dim
    ):
        rope_dim = query_dim - value_dim
        if rope_dim > 0:
            real_query_rope = query[..., value_dim:]
            real_key_rope = key[..., value_dim:]
            query = query[..., :value_dim]
            key = key[..., :value_dim]
            if real_key_sink is not None and _logical_dim(real_key_sink) == query_dim:
                if real_key_rope_sink is None:
                    real_key_rope_sink = real_key_sink[..., value_dim:]
                real_key_sink = real_key_sink[..., :value_dim]

    return [
        query, key, value,
        real_query_rope, real_key_rope, None, real_atten_mask,
        real_actual_seq_qlen, real_actual_seq_kvlen, block_table, None, None, None, None, None, None, None, None,
        real_meta_data, real_key_sink, real_value_sink, real_key_rope_sink,
    ]
