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

from typing import Optional

import torch

F = torch.nn.functional
_BENCH_PRECISION = False


def _default_lengths(
    total: Optional[int], batch: Optional[int], default_mode: Optional[str]
):
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
        return (
            _default_lengths(total, batch, default_mode) if empty_uses_default else []
        )
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif allow_scalar and isinstance(value, (int, float)):
        value = [value]
    out = [int(x) for x in value]
    if not out:
        return (
            _default_lengths(total, batch, default_mode) if empty_uses_default else []
        )
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
    raise ValueError(
        "actual_seq list is inconsistent with the packed total: expected a "
        f"non-decreasing prefix ending at {int(total)} or lengths summing to "
        f"{int(total)}, got {out} (sum={sum(out)})"
    )


def slice_optional(x: Optional[torch.Tensor], b: int, q_len: int, kv_len: int, device):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() <= 1:
        return None
    x = x.detach().to(device=device)
    if x.dim() == 2:
        return x[-q_len:, -kv_len:]
    if x.dim() >= 4:
        b_idx = min(b, x.shape[0] - 1)
        return x[b_idx : b_idx + 1, :, -q_len:, -kv_len:]
    return x


def _attention_compute(
    query, key, value, dy=None, *, calc_dtype=torch.float32, calc_device="cpu", **kwargs
):
    """Bounded-memory reference using the softmax-backward equations.

    Query blocks and head blocks bound temporary storage. No dense attention
    graph is retained across segments. FP64 oracle and FP32 bench use the same
    equations; tests cross-check an independently differentiated forward.
    """
    layout = kwargs.get("input_layout", "TND")
    if layout not in ("TND", "BSND"):
        raise ValueError("only TND and BSND are supported")
    if float(kwargs.get("keep_prob", 1.0)) != 1.0:
        raise ValueError("only keep_prob=1.0 is supported")
    hq, hk = query.shape[-2], key.shape[-2]
    if int(kwargs.get("head_num", hq)) != hq or hq % hk:
        raise ValueError("head_num must match query heads and be divisible by KV heads")
    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError("key/value sequence and head dimensions must match")
    q = query.detach().to(device=calc_device, dtype=calc_dtype)
    k = key.detach().to(device=calc_device, dtype=calc_dtype)
    v = value.detach().to(device=calc_device, dtype=calc_dtype)
    qr, kr = kwargs.get("query_rope"), kwargs.get("key_rope")
    paired_rope = (
        isinstance(qr, torch.Tensor)
        and qr.numel()
        and isinstance(kr, torch.Tensor)
        and kr.numel()
    )
    if bool(isinstance(qr, torch.Tensor) and qr.numel()) != bool(
        isinstance(kr, torch.Tensor) and kr.numel()
    ):
        raise ValueError("query_rope and key_rope must be paired")
    if paired_rope:
        q = torch.cat((q, qr.detach().to(device=calc_device, dtype=calc_dtype)), -1)
        k = torch.cat((k, kr.detach().to(device=calc_device, dtype=calc_dtype)), -1)
    sink = int(kwargs.get("sink_num", 0) or 0) * 64 if layout == "TND" else 0
    if sink < 0 or (sink and sink >= k.shape[0]):
        raise ValueError("sink rows must leave a nonempty ordinary KV sequence")
    aql = kwargs.get("actual_seq_qlen")
    batch = (len(aql) if aql is not None else 1) if layout == "TND" else query.shape[0]
    qt = query.shape[0] if layout == "TND" else query.shape[1]
    kt = key.shape[0] - sink if layout == "TND" else key.shape[1]
    qlens = as_int_list(
        kwargs.get("actual_seq_qlen"),
        qt,
        batch or 1,
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    klens = as_int_list(
        kwargs.get("actual_seq_kvlen"),
        kt,
        len(qlens),
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    if len(qlens) != len(klens) or any(x <= 0 for x in qlens + klens):
        raise ValueError("Q/KV must have matching positive sequence lengths")
    if layout == "BSND" and (any(x > qt for x in qlens) or any(x > kt for x in klens)):
        raise ValueError("actual lengths exceed BSND padded extents")
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    output = q.new_zeros((*query.shape[:-1], value.shape[-1]))
    if dy is not None:
        dy = dy.detach().to(device=calc_device, dtype=calc_dtype)
    scale = float(kwargs.get("scale", kwargs.get("scale_value", 1.0)))
    mode = int(kwargs.get("sparse_mode", 0))
    if mode not in (0, 2, 3, 4):
        raise ValueError("sparse_mode must be 0, 2, 3 or 4")
    pre, nxt = (
        int(kwargs.get("pre_tokens", 2**31 - 1)),
        int(kwargs.get("next_tokens", 2**31 - 1)),
    )
    pse, mask = kwargs.get("pse"), kwargs.get("atten_mask")
    qoff, koff = 0, sink
    maxparts, sumparts = [], []
    for b, (ql, kl) in enumerate(zip(qlens, klens)):
        qslice = slice(qoff, qoff + ql) if layout == "TND" else (b, slice(0, ql))
        kslice = slice(koff, koff + kl) if layout == "TND" else (b, slice(0, kl))
        qb, kb, vb = (
            q[qslice].transpose(0, 1),
            k[kslice].transpose(0, 1),
            v[kslice].transpose(0, 1),
        )
        if sink:
            kb = torch.cat((k[:sink].transpose(0, 1), kb), 1)
            vb = torch.cat((v[:sink].transpose(0, 1), vb), 1)
        dqb, dkb, dvb = torch.zeros_like(qb), torch.zeros_like(kb), torch.zeros_like(vb)
        ob = q.new_zeros((hq, ql, value.shape[-1]))
        rowmax = q.new_empty((hq, ql, 1))
        rowsum = torch.empty_like(rowmax)
        dyb = dy[qslice].transpose(0, 1) if dy is not None else None
        bias = slice_optional(pse, b, ql, kl, calc_device)
        # Sparse modes carry a compressed candidate mask. Logical visibility
        # is reconstructed per sequence in the bounded reference.
        explicit = slice_optional(mask, b, ql, kl, calc_device) if mode == 0 else None
        for h0 in range(0, hq, 4):
            h1 = min(h0 + 4, hq)
            headmap = (torch.arange(h0, h1) // (hq // hk)).to(device=calc_device)
            kh, vh = kb[headmap], vb[headmap]
            for q0 in range(0, ql, 256):
                q1 = min(q0 + 256, ql)
                qc = qb[h0:h1, q0:q1]
                score = (qc @ kh.transpose(-1, -2)) * scale
                if bias is not None:
                    bs = bias.reshape(-1, ql, kl)
                    bs = bs if bs.shape[0] == 1 else bs[h0:h1]
                    bs = bs[:, q0:q1].to(calc_dtype)
                    score = score + (F.pad(bs, (sink, 0)) if sink else bs)
                rows = torch.arange(q0, q1, device=calc_device).view(-1, 1)
                cols = torch.arange(kl, device=calc_device).view(1, -1)
                if explicit is not None:
                    blocked = explicit.reshape(-1, ql, kl)
                    blocked = blocked if blocked.shape[0] == 1 else blocked[h0:h1]
                    blocked = blocked[:, q0:q1].bool()
                elif mode == 2:
                    blocked = cols > rows
                elif mode == 3:
                    blocked = cols > kl - ql + rows
                else:
                    center = kl - ql + rows
                    blocked = (cols < center - pre) | (cols > center + nxt)
                if sink:
                    blocked = F.pad(blocked, (sink, 0), value=False)
                score = score.masked_fill(blocked, -40000.0)
                rm = score.max(-1, keepdim=True).values
                ex = torch.exp(score - rm)
                rs = ex.sum(-1, keepdim=True)
                prob = ex / rs
                ob[h0:h1, q0:q1] = prob @ vh
                rowmax[h0:h1, q0:q1], rowsum[h0:h1, q0:q1] = rm, rs
                if dyb is not None:
                    grad = dyb[h0:h1, q0:q1]
                    dp = grad @ vh.transpose(-1, -2)
                    ds = prob * (dp - (dp * prob).sum(-1, keepdim=True))
                    ds.masked_fill_(blocked, 0)
                    dqb[h0:h1, q0:q1] = (ds @ kh) * scale
                    dkb.index_add_(0, headmap, (ds.transpose(-1, -2) @ qc) * scale)
                    dvb.index_add_(0, headmap, prob.transpose(-1, -2) @ grad)
        output[qslice] = ob.transpose(0, 1)
        if dyb is not None:
            dq[qslice] = dqb.transpose(0, 1)
            if sink:
                dk[:sink] += dkb[:, :sink].transpose(0, 1)
                dv[:sink] += dvb[:, :sink].transpose(0, 1)
            dk[kslice] += dkb[:, sink:].transpose(0, 1)
            dv[kslice] += dvb[:, sink:].transpose(0, 1)
        # Packed auxiliary stats address [valid token, head, lane]. Each
        # sequence/batch contributes its valid query tokens in input order.
        # Flattening [head, token, lane] before the final reshape permutes rows.
        stats_max = rowmax.transpose(0, 1)
        stats_sum = rowsum.transpose(0, 1)
        maxparts.append(stats_max.expand(-1, -1, 8).contiguous().reshape(-1))
        sumparts.append(stats_sum.expand(-1, -1, 8).contiguous().reshape(-1))
        qoff, koff = qoff + ql, koff + kl
    softmax_max = torch.cat(maxparts).reshape(-1, hq, 8).float()
    softmax_sum = torch.cat(sumparts).reshape(-1, hq, 8).float()
    if dy is None:
        return output.to(query.dtype), softmax_max, softmax_sum
    # Rope channels affect score, but only gradients of the original Q/K are exposed.
    return (
        dq[..., : query.shape[-1]],
        dk[..., : key.shape[-1]],
        dv,
        dq[..., query.shape[-1] :],
        dk[..., key.shape[-1] :],
    )


def _attention_forward_inputs(
    query,
    key,
    value,
    pse=None,
    atten_mask=None,
    query_rope=None,
    key_rope=None,
    **kwargs,
):
    return _attention_compute(
        query,
        key,
        value,
        calc_dtype=torch.float32,
        pse=pse,
        atten_mask=atten_mask,
        query_rope=query_rope,
        key_rope=key_rope,
        **kwargs,
    )


def attention_grad(query, key, value, dy, **kwargs):
    with torch.enable_grad():
        return _attention_grad_impl(query, key, value, dy, **kwargs)


def _attention_grad_impl(query, key, value, dy, **kwargs):
    # The input hook can leave the original dtype in the CPU oracle call.
    # Select the oracle precision here so that it stays distinct from the bench.
    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    calc_device = query.device if native else torch.device("cpu")
    calc_dtype = query.dtype if native else (
        torch.float32 if _BENCH_PRECISION else torch.float64
    )
    return _attention_compute(
        query, key, value, dy, calc_dtype=calc_dtype, calc_device=calc_device, **kwargs
    )


def flash_attention_score_grad_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    head_num: int,
    input_layout: str = "TND",
    pse: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    softmax_max: Optional[torch.Tensor] = None,
    softmax_sum: Optional[torch.Tensor] = None,
    softmax_in: Optional[torch.Tensor] = None,
    attention_in: Optional[torch.Tensor] = None,
    sink_tensor: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    scale_value: float = 1.0,
    scale: float = None,
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
    softmaxInLayout: str = "",
    q_start_idx=None,
    kv_start_idx=None,
    **kwargs,
):
    del (
        padding_mask,
        softmax_max,
        softmax_sum,
        softmax_in,
        attention_in,
        sink_tensor,
        inner_precise,
    )
    del prefix, pse_type, softmaxInLayout, q_start_idx, kv_start_idx, kwargs
    if scale is not None:
        scale_value = scale
    grads = attention_grad(
        query,
        key,
        value,
        dy,
        head_num=head_num,
        input_layout=input_layout,
        pse=pse,
        atten_mask=atten_mask,
        query_rope=query_rope,
        key_rope=key_rope,
        scale=scale_value,
        keep_prob=keep_prob,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        sink_num=sink_num,
    )
    # ST only asserts dq/dk/dv; the current Python wrapper exposes those three.
    return grads[:3]


def flash_attention_score_grad_enhance_bench(*args, **kwargs):
    """同精度参考钩子（evaluator 的 b，见 cann_loader.get_bench_function）。

    与 golden 同一实现、同一签名，但强制 float32 累加精度计算——即修复前
    golden 的口径——作为"正确同精度实现应有的误差下限"，使小值域/相消/正常
    值域兜底的参考错误计数不再恒为 0（issue #311）。非对某颗硬件的复刻；
    更高内部精度的实现会 meet-or-exceed 此下限、照常通过。
    """
    global _BENCH_PRECISION
    _BENCH_PRECISION = True
    try:
        return flash_attention_score_grad_enhance(*args, **kwargs)
    finally:
        _BENCH_PRECISION = False


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    pse=None,
    atten_mask=None,
    query_rope=None,
    key_rope=None,
    **kwargs,
):
    # Preserve the published Q/K shapes. Rope participates in the score only;
    # sink rows are already part of the case's physical K/V tensors.
    if (
        not getattr(query, "is_meta", False)
        and atten_mask is None
        and int(kwargs.get("sparse_mode", 0)) in (2, 3, 4)
    ):
        # Preserve master's compressed-mask candidate contract, without
        # expanding it to the actual (possibly >2048) sequence lengths.
        atten_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.uint8), diagonal=1)
    attention_in = softmax_max = softmax_sum = None
    if not getattr(query, "is_meta", False):
        attention_in, softmax_max, softmax_sum = _attention_forward_inputs(
            query,
            key,
            value,
            pse=pse,
            atten_mask=atten_mask,
            query_rope=query_rope,
            key_rope=key_rope,
            **kwargs,
        )
    # Preserve supplied mask storage. Sparse2/3/4 decode visibility only in
    # the CPU reference; mode0 keeps the explicit logical mask semantics.
    return [
        query,
        key,
        value,
        dy,
        pse,
        None,
        atten_mask,
        softmax_max,
        softmax_sum,
        None,
        attention_in,
        None,
        query_rope,
        key_rope,
    ]
