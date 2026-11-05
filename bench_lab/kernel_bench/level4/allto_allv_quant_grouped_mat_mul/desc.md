# AlltoAllvQuantGroupedMatMul 算子 API 描述

## 1. 算子简介

融合可变长度 AlltoAllv 通信、专家重排与 MXFP8 分组矩阵乘，可并行计算本地共享专家矩阵乘。先通信后计算，每个接收 rank 使用自己的专家权重。以下定义 Ascend 950DT 上的 MXFP8 接口，对应 `aclnnAlltoAllvQuantGroupedMatMul`。

## 2. 算子定义

令 `C[s][r][j]` 为源 rank `s` 发给目标 rank `r` 的专家 `j` 的 token 行数。源端输入按 `[destination][expert][token]` 排列，接收后按 `[expert][source][token]` 排列；同行的量化 scale 随 token 一起路由。

E8M0 使用一个指数码 `b` 表示 `2**(b-127)`，每个码作用于 K 维连续 32 个元素。`[ceil(K/64),2]` 的两个末维码依次对应相邻的两个 32 元素组，超过 K 的填充不参与计算。

```text
Xhat[s] = MXDequant(gmm_x[s], gmm_x_scale[s])
What[r,j] = MXDequant(gmm_weight[r,j], gmm_weight_scale[r,j])
gmm_y[r] = Concat_j(Concat_s(Xhat[s -> r,j]) @ What[r,j])
mm_y[r] = MXDequant(mm_x[r], mm_x_scale[r]) @ MXDequant(mm_weight[r], mm_weight_scale[r])
```

`trans_gmm_weight`、`trans_mm_weight` 同时决定相应物理权重和 scale 的解释方式。结果只在最终输出时舍入为指定 FP16/BF16。共享专家不参与通信。`permute_out` 是重排后的原始 FP8 值，**不乘 scale**。

## 3. 接口规范

### 算子原型

```python
cann_bench.allto_allv_quant_grouped_mat_mul(Tensor gmm_x, Tensor gmm_weight, Tensor gmm_x_scale, Tensor gmm_weight_scale, str hcomm_info, int world_size, int[] send_counts, int[] recv_counts, Tensor? mm_x=None, Tensor? mm_weight=None, Tensor? mm_x_scale=None, Tensor? mm_weight_scale=None, bool trans_gmm_weight=False, bool trans_mm_weight=False, str out_dtype="bfloat16", int gmm_x_quant_mode=6, int gmm_weight_quant_mode=6, int mm_x_quant_mode=0, int mm_weight_quant_mode=0, int group_size=4295032864, int scale_dtype=293, bool permute_out_flag=False) -> (Tensor gmm_y, Tensor? mm_y, Tensor? permute_out)
```

### 输入、属性与输出

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `gmm_x` | FP8 E4M3FN/E5M2，`[T,H1]` | 发送 token，`T=sum(send_counts)`。 |
| `gmm_weight` | FP8 E4M3FN/E5M2，`[e,H1,N1]` 或 `[e,N1,H1]` | 本地专家权重；可与 x 使用不同 FP8 格式。 |
| `gmm_x_scale` | uint8，`[T,ceil(H1/64),2]` | E8M0 原始指数码，通过 `scale_dtype=293` 解释。 |
| `gmm_weight_scale` | uint8，`[e,ceil(H1/64),N1,2]` 或 `[e,N1,ceil(H1/64),2]` | 其转置状态与权重一致。 |
| `hcomm_info` | str | 已初始化 HCCL 通信域名。 |
| `world_size` | int | HCCL 通信域大小。 |
| `send_counts` | `list[int]`，长度 `world_size*e` | 发给各目标 rank、专家的 token 行数，rank 优先、专家次之，不是累加前缀。 |
| `recv_counts` | `list[int]`，长度 `world_size*e` | 从各源 rank、专家接收的 token 行数，rank 优先、专家次之，不是累加前缀。 |
| `mm_x` | 可选 FP8，`[BS,H2]` | 共享专家左矩阵，与 gmm_x 类型一致，默认 None。 |
| `mm_weight` | 可选 FP8，`[H2,N2]` 或 `[N2,H2]` | 共享专家右矩阵，与 gmm_weight 类型一致，默认 None。 |
| `mm_x_scale` | 可选 uint8，`[BS,ceil(H2/64),2]` | 共享左矩阵的 E8M0 指数码，与共享矩阵同时提供或同时为 None。 |
| `mm_weight_scale` | 可选 uint8，`[ceil(H2/64),N2,2]` 或 `[N2,ceil(H2/64),2]` | 共享右矩阵的 E8M0 指数码，转置状态与 mm_weight 一致，默认 None。 |
| `trans_gmm_weight` | bool，默认 False | 转置专家权重的最后两个逻辑矩阵维。 |
| `trans_mm_weight` | bool，默认 False | 转置共享专家权重的最后两个逻辑矩阵维。 |
| `out_dtype` | str，默认 bfloat16 | 支持 float16、bfloat16；不改变 permute_out 类型。 |
| `gmm_x_quant_mode` | int，默认6 | 专家左矩阵仅支持 MX 模式。 |
| `gmm_weight_quant_mode` | int，默认6 | 专家右矩阵仅支持 MX 模式。 |
| `mm_x_quant_mode` | int，默认0 | 共享左矩阵存在时为6，否则不参与计算。 |
| `mm_weight_quant_mode` | int，默认0 | 共享右矩阵存在时为6，否则不参与计算。 |
| `group_size` | int，4295032864 | 固定 `[1,1,32]`，即 `32 | (1<<16) | (1<<32)`。 |
| `scale_dtype` | int，默认293 | 将 uint8 尺度存储解释为 FLOAT8_E8M0 指数码。 |
| `permute_out_flag` | bool，默认 False | 是否返回重排后的原始 FP8 token。 |
| `gmm_y` | FP16/BF16，`[A,N1]` | `A=sum(recv_counts)`，专家优先排列。 |
| `mm_y` | FP16/BF16，`[BS,N2]` 或 None | 共享专家结果。 |
| `permute_out` | 与 gmm_x 同类型，`[A,H1]` 或 None | 原始 FP8 重排结果。 |

### 规则与约束

- 上游 Ascend 950DT 支持通信域大小 2、4、8、16、32、64、128、256；本接口采用 `world_size=2`。`1<=e<=32` 且 `world_size*e<=256`。
- 各 rank 的专家数、H1/N1 和类型配置一致；每条发送边对应的接收计数必须相等。允许零通信边或空专家，但每张卡的总发送/接收行数必须为正。
- `0<T,A<52428800`，`1<=H1,N1<=65535`；非空输入均采用同设备上的连续 ND 物理存储。
- 共享输入存在时 `BS>0`、`T%BS=0`、`2<=T/BS<=8`、`1<=H2<=12288`、`1<=N2<=65535`；共享权重为空时 `trans_mm_weight` 必须为 False。
- 本接口选取 MXFP8 和正常有限 E8M0 码，不覆盖上游 HiFloat8 per-tensor、FP4、counts Tensor 或偏置等其他接口组合。

规格依据：`ops-transformer/mc2/allto_allv_quant_grouped_mat_mul/docs/aclnnAlltoAllvQuantGroupedMatMul.md` 及 `op_host/op_tiling/arch35/allto_allv_mx_quant_grouped_mat_mul_tiling.cpp`。

## 4. 精度要求

所有 rank 的输出均须满足形状、dtype 和可选输出契约。启用三方比较时，以相同输入的 CPU FP32 结果为参考：候选的大值域最大误差不超过设备级联的 10 倍，平均误差、误差总和及全量 RMSE 不超过级联的 2 倍。

另允许有限值逐元素满足 `abs(candidate-golden) <= atol + rtol*abs(golden)`，避免零级联误差退化为逐位相等要求。有效比较类型 FP16/BF16/FP32 的 `rtol` 分别为 `2^-10`、`2^-7`、`2^-23`，`atol` 分别为 `1e-5`、`1e-5`、`1e-9`。显式 FP32 比较和 FP8 辅助输出使用 FP32 容差，不能按输入 FP8 类型放宽。

FP16/BF16 的小值域分界为 `1e-3`，FP32 为 `1e-6`。小值元素不少于 100000 个时，候选的小值超差数不得超过级联的 2 倍；计数边界及绝对误差阈值保持不变。结构、NaN/正负无穷掩码及有限误差统计检查始终有效，容差分支不能绕过这些检查。

级联参考先接受按真实输出类型的独立固定精度校验；参考失效时评分不可用，不归为候选编译错误。未启用三方模式的历史用例保持原相对误差流程。完整判定与比较类型规则见[三方精度指南](../../../../docs/guide/mc2_accuracy.md)。

## 5. 标准 Golden 代码

```python
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
```

## 6. 算子调用示例

假设 HCCL 域已初始化，`device` 为当前 rank 的 NPU，所有 rank 执行相同调用：

```python
import torch
import cann_bench

x = torch.randn(128, 1024, device=device).to(torch.float8_e4m3fn)
w = torch.randn(2, 1024, 512, device=device).to(torch.float8_e5m2)
xs = torch.full((128, 16, 2), 127, dtype=torch.uint8, device=device)
ws = torch.full((2, 16, 512, 2), 127, dtype=torch.uint8, device=device)
y, mm_y, permuted = cann_bench.allto_allv_quant_grouped_mat_mul(
    x, w, xs, ws, hcomm_info, 2, [32, 32, 32, 32], [32, 32, 32, 32],
    out_dtype="bfloat16", gmm_x_quant_mode=6, gmm_weight_quant_mode=6,
    group_size=4295032864, scale_dtype=293, permute_out_flag=True,
)
# y: [128,512], mm_y: None, permuted: [128,1024] FP8 E4M3FN
```
