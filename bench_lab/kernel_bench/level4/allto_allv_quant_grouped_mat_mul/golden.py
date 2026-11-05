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


"""MXFP8 AlltoAllv, expert permutation and local grouped/shared matmul."""

import torch


def _dtype(name):
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
    }[name]


def _layout(ctx, attrs):
    p, rank, e = int(ctx["world_size"]), int(ctx["rank"]), int(attrs["e"])
    counts = attrs["token_counts"]
    if p != 2 or attrs["world_size"] != p or not 0 <= rank < p or not 1 <= e <= 32:
        raise ValueError(
            "This A5 MXFP8 contract requires two ranks and 1..32 local experts"
        )
    if len(counts) != p or any(len(row) != p for row in counts):
        raise ValueError("token_counts must have shape [source, destination, expert]")
    if any(
        len(edge) != e or any(type(v) is not int or v < 0 for v in edge)
        for row in counts
        for edge in row
    ):
        raise ValueError("token_counts must contain nonnegative integer counts")
    send = [v for edge in counts[rank] for v in edge]
    recv = [v for source in range(p) for v in counts[source][rank]]
    if not 0 < sum(send) < 52428800 or not 0 < sum(recv) < 52428800:
        raise ValueError("Each rank must send and receive nonempty token tensors")
    return p, rank, e, counts, send, recv


def _check_mx_rows(values, codes):
    expected = tuple(values.shape[:-1]) + ((values.shape[-1] + 63) // 64, 2)
    if codes is None or codes.dtype != torch.uint8 or tuple(codes.shape) != expected:
        raise ValueError("MX scales require uint8 [...,rows,ceil(K/64),2]")
    if values.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("MX matrix inputs must use FP8 E4M3FN or E5M2")


def _validate_inputs(ctx, inputs, attrs):
    layout = _layout(ctx, attrs)
    _, _, e, _, send, recv = layout
    x, weight = inputs["gmm_x"], inputs["gmm_weight"]
    trans = bool(attrs["trans_gmm_weight"])
    if x.ndim != 2 or weight.ndim != 3 or weight.shape[0] != e:
        raise ValueError("gmm_x must be 2D and gmm_weight must contain e matrices")
    k, n = weight.shape[-1 if trans else -2], weight.shape[-2 if trans else -1]
    if x.shape != (sum(send), k) or not 0 < k < 65536 or not 0 < n < 65536:
        raise ValueError("gmm_x rows/K and local expert weight dimensions must agree")
    if list(inputs["send_counts"]) != send or list(inputs["recv_counts"]) != recv:
        raise ValueError("Host counts must match token_counts")
    if attrs["gmm_x_quant_mode"] != 6 or attrs["gmm_weight_quant_mode"] != 6:
        raise ValueError("Only MX quantization mode 6 is supported")
    if attrs["group_size"] != 4295032864 or attrs["scale_dtype"] != 293:
        raise ValueError("MX groups require [1,1,32] and E8M0 scale dtype 293")
    if attrs["out_dtype"] not in ("float16", "bfloat16"):
        raise ValueError("Output dtype must be float16 or bfloat16")
    _check_mx_rows(x, inputs["gmm_x_scale"])
    wrows = weight if trans else weight.transpose(-2, -1)
    wscale = inputs["gmm_weight_scale"]
    _check_mx_rows(wrows, wscale if trans else wscale.transpose(-3, -2))
    mm_names = ("mm_x", "mm_weight", "mm_x_scale", "mm_weight_scale")
    present = [inputs.get(name) is not None for name in mm_names]
    if any(present) != bool(attrs["mm_out"]) or (any(present) and not all(present)):
        raise ValueError(
            "Shared matrices and scales must all be present exactly when mm_out is enabled"
        )
    if all(present):
        mx, mw = inputs["mm_x"], inputs["mm_weight"]
        mt = bool(attrs["trans_mm_weight"])
        if mx.ndim != 2 or mw.ndim != 2:
            raise ValueError("Shared matrices must be 2D")
        mk, mn = mw.shape[-1 if mt else -2], mw.shape[-2 if mt else -1]
        bs = mx.shape[0]
        if bs <= 0 or x.shape[0] % bs or not 2 <= x.shape[0] // bs <= 8:
            raise ValueError(
                "Shared BS must divide sent token rows with a ratio in [2,8]"
            )
        if mx.shape[1] != mk or not 0 < mk <= 12288 or not 0 < mn < 65536:
            raise ValueError("Shared K/N dimensions are incompatible")
        if mx.dtype != x.dtype or mw.dtype != weight.dtype:
            raise ValueError(
                "Shared matrix dtypes must match the corresponding grouped matrices"
            )
        if attrs["mm_x_quant_mode"] != 6 or attrs["mm_weight_quant_mode"] != 6:
            raise ValueError("Shared matrices require MX quantization mode 6")
        _check_mx_rows(mx, inputs["mm_x_scale"])
        ms = inputs["mm_weight_scale"]
        _check_mx_rows(mw if mt else mw.t(), ms if mt else ms.transpose(-3, -2))
    elif attrs["trans_mm_weight"]:
        raise ValueError("trans_mm_weight must be false when mm_weight is absent")
    tensors = [v for v in inputs.values() if isinstance(v, torch.Tensor)]
    if any(v.device != x.device or not v.is_contiguous() for v in tensors):
        raise ValueError("All physical tensors must be contiguous on the same device")
    return layout


def mc2_make_rank_inputs(ctx, case_payload):
    attrs = case_payload["attrs"]
    shapes, dtypes, ranges = (
        case_payload["input_shapes"],
        case_payload["dtypes"],
        case_payload["value_ranges"],
    )
    _, rank, _, _, send, recv = _layout(ctx, attrs)
    names = (
        "gmm_x",
        "gmm_weight",
        "send_counts",
        "recv_counts",
        "mm_x",
        "mm_weight",
        "gmm_x_scale",
        "gmm_weight_scale",
        "mm_x_scale",
        "mm_weight_scale",
    )
    inputs = {"send_counts": send, "recv_counts": recv}
    for i, name in enumerate(names):
        if i in (2, 3):
            continue
        if shapes[i] is None:
            inputs[name] = None
            continue
        low, high = ranges[i]
        seed = int(attrs["seed"]) + rank * 1000003 + i * 1009
        if name.endswith("_scale"):
            if not 1 <= low <= high <= 254:
                raise ValueError("Scale codes must be finite normal E8M0 exponents")
            codes = torch.arange(torch.Size(shapes[i]).numel(), dtype=torch.int64)
            value = (
                (int(low) + (codes + seed) % (int(high) - int(low) + 1))
                .reshape(shapes[i])
                .to(torch.uint8)
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = torch.empty(shapes[i], dtype=torch.float32).uniform_(
                low, high, generator=generator
            )
            value = value.to(_dtype(dtypes[i]))
        inputs[name] = value.to(ctx["device"])
    _validate_inputs(ctx, inputs, attrs)
    return inputs


def _mx_rows(values, codes):
    """Decode each E8M0 exponent byte into a scale for 32 adjacent K elements."""
    scales = torch.exp2(codes.float() - 127.0).flatten(-2).repeat_interleave(32, dim=-1)
    return values.float() * scales[..., : values.shape[-1]]


def _mx_weight(weight, scale, transposed):
    rows = weight if transposed else weight.transpose(-2, -1)
    codes = scale if transposed else scale.transpose(-3, -2)
    return _mx_rows(rows, codes).transpose(-2, -1)


def _exchange(ctx, inputs, layout):
    p, rank, _, counts, _, recv = layout
    received = []
    # FP32 preserves all FP8 values and exponent codes, and avoids requiring
    # native FP8 collectives from either HCCL or the Gloo CPU adapter.
    for name in ("gmm_x", "gmm_x_scale"):
        value = inputs[name].float()
        output = value.new_empty((sum(recv),) + tuple(value.shape[1:]))
        ctx["dist"].all_to_all_single(
            output,
            input=value,
            input_split_sizes=[sum(edge) for edge in counts[rank]],
            output_split_sizes=[sum(counts[source][rank]) for source in range(p)],
        )
        received.append(output)
    return received


def _shared(inputs, attrs):
    if not attrs["mm_out"]:
        return None
    x = _mx_rows(inputs["mm_x"], inputs["mm_x_scale"])
    w = _mx_weight(
        inputs["mm_weight"], inputs["mm_weight_scale"], attrs["trans_mm_weight"]
    )
    return torch.matmul(x, w)


def mc2_distributed_golden(ctx, inputs, attrs):
    """Device cascade: communicate values/scales, permute, dequantize and multiply."""
    layout = _validate_inputs(ctx, inputs, attrs)
    p, rank, e, counts, _, recv = layout
    received, scales = _exchange(ctx, inputs, layout)
    offsets, offset = [], 0
    for rows in recv:
        offsets.append(offset)
        offset += rows
    indices = [
        i
        for expert in range(e)
        for source in range(p)
        for i in range(
            offsets[source * e + expert],
            offsets[source * e + expert] + recv[source * e + expert],
        )
    ]
    indices = torch.tensor(indices, dtype=torch.int64, device=received.device)
    permuted = received.index_select(0, indices)
    x = _mx_rows(permuted, scales.index_select(0, indices))
    weight = _mx_weight(
        inputs["gmm_weight"], inputs["gmm_weight_scale"], attrs["trans_gmm_weight"]
    )
    out_dtype = _dtype(attrs["out_dtype"])
    products, start = [], 0
    for expert in range(e):
        rows = sum(counts[source][rank][expert] for source in range(p))
        if rows:
            products.append(torch.matmul(x[start : start + rows], weight[expert]))
        start += rows
    mm = _shared(inputs, attrs)
    return (
        torch.cat(products).to(out_dtype),
        None if mm is None else mm.to(out_dtype),
        permuted.to(inputs["gmm_x"].dtype) if attrs["permute_out_flag"] else None,
    )


def mc2_cpu_golden(ctx, inputs, attrs):
    """Unrounded FP32 products from source/expert slices; raw FP8 permute output."""
    layout = _validate_inputs(ctx, inputs, attrs)
    p, rank, e, counts, _, _ = layout
    received, scales = _exchange(ctx, inputs, layout)
    lengths = [sum(counts[source][rank]) for source in range(p)]
    by_source, by_scale = received.split(lengths), scales.split(lengths)
    weight = _mx_weight(
        inputs["gmm_weight"], inputs["gmm_weight_scale"], attrs["trans_gmm_weight"]
    )
    products, raw_blocks = [], []
    for expert in range(e):
        for source in range(p):
            begin = sum(counts[source][rank][:expert])
            end = begin + counts[source][rank][expert]
            raw = by_source[source][begin:end]
            decoded = _mx_rows(raw, by_scale[source][begin:end])
            raw_blocks.append(raw)
            products.append(torch.matmul(decoded, weight[expert]))
    return (
        torch.cat(products),
        _shared(inputs, attrs),
        torch.cat(raw_blocks) if attrs["permute_out_flag"] else None,
    )


def mc2_cpu_references(ctx, inputs, attrs):
    """Independent FP32 products and the cascade's final matrix output casts."""
    mathematical = mc2_cpu_golden(ctx, inputs, attrs)
    grouped, shared, permuted = mathematical
    dtype = _dtype(attrs["out_dtype"])
    staged = (
        grouped.to(dtype).float(),
        None if shared is None else shared.to(dtype).float(),
        permuted,
    )
    return mathematical, staged


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["gmm_x"],
        inputs["gmm_weight"],
        inputs["gmm_x_scale"],
        inputs["gmm_weight_scale"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        inputs["send_counts"],
        inputs["recv_counts"],
        mm_x=inputs.get("mm_x"),
        mm_weight=inputs.get("mm_weight"),
        mm_x_scale=inputs.get("mm_x_scale"),
        mm_weight_scale=inputs.get("mm_weight_scale"),
        trans_gmm_weight=attrs["trans_gmm_weight"],
        trans_mm_weight=attrs["trans_mm_weight"],
        out_dtype=attrs["out_dtype"],
        gmm_x_quant_mode=attrs["gmm_x_quant_mode"],
        gmm_weight_quant_mode=attrs["gmm_weight_quant_mode"],
        mm_x_quant_mode=attrs["mm_x_quant_mode"],
        mm_weight_quant_mode=attrs["mm_weight_quant_mode"],
        group_size=attrs["group_size"],
        scale_dtype=attrs["scale_dtype"],
        permute_out_flag=attrs["permute_out_flag"],
    )
