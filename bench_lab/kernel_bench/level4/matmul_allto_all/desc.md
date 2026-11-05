# MatmulAlltoAll 算子 API 描述

## 1. 算子简介

MatmulAlltoAll 将矩阵乘与 AllToAll 集合通信融合执行：先使用本地输入和权重进行矩阵乘，再把结果按列切分并分发到目标 rank，接收端沿行维拼接。适用于张量并行中的线性变换与数据重分布。支持 FP16/BF16 非量化和 MXFP8 量化、可选偏置及权重转置。

非量化计算语义对应 `aclnnMatmulAlltoAll`，MXFP8 量化对应 `aclnnQuantMatmulAlltoAll`，以下描述 Ascend 950PR/950DT（A5）上的 Python 接口。

## 2. 算子定义

设通信域包含 `p = world_size` 个 rank，`r` 为当前 rank，`s` 为源 rank。每个 rank 的输入 `X_s` 为 `[M, K]`，有效权重 `W_s` 为 `[K, N]`。

$$
Z_s = X_s W_s + b_s
$$

$$
Y_r = \operatorname{Concat}_{\mathrm{dim}=0}\left(
Z_0[:,rN/p:(r+1)N/p],\ldots,Z_{p-1}[:,rN/p:(r+1)N/p]
\right)
$$

本地矩阵乘结果 `Z_s` 为 `[M, N]`。每个源 rank 沿 `N` 维等分结果，第 `r` 块发送到 rank `r`；接收端按源 rank 顺序沿行维拼接，得到 `[p*M, N/p]` 的 `Y_r`。

`is_trans_x2=True` 时 `W_s = x2_s.T`，否则 `W_s = x2_s`。可选偏置在通信前按列加入本地矩阵乘结果，随结果一起分发；不能在通信后只加当前 rank 的完整偏置。AllToAll 不执行求和归约。

MXFP8 模式以 `x1_quant_mode=x2_quant_mode=6` 表示。每连续 32 个 K 元素共享一个 E8M0 尺度，编码 `e` 对应 `2**(e-127)`；先按各自尺度反量化 `x1`、`x2` 后矩阵乘并加 FP32 偏置，再转为 `out_dtype` 进行 AllToAll。权重必须按 `[N,K]` 存储并转置使用。非量化模式的 Dequant 为恒等变换。

## 3. 接口规范

### 算子原型

```python
cann_bench.matmul_all_to_all(
    Tensor x1, Tensor x2, str hcomm_info, int world_size,
    Tensor? bias=None, bool is_trans_x2=False,
    Tensor? x1_scale=None, Tensor? x2_scale=None,
    int x1_quant_mode=0, int x2_quant_mode=0, int group_size=0,
    int scale_dtype=293, str? out_dtype=None
) -> Tensor y
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x1` | Tensor，`[M, K]` | 本地矩阵乘左输入，FP16/BF16 或 FP8 E4M3FN/E5M2。 |
| `x2` | Tensor，`[K, N]` 或 `[N, K]` | 本地权重，非量化类型与 `x1` 相同；MX 可独立选择 FP8 E4M3FN/E5M2，且必须使用后一形状。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 通信域的 rank 数，必须与实际通信域大小一致。 |
| `bias` | 可选 Tensor，`[N]` | 本地列偏置，默认 `None`；类型约束见下文。 |
| `is_trans_x2` | bool | 是否在矩阵乘中转置 `x2`，默认 `False`；MX 必须为 `True`。 |
| `x1_scale` | 可选 uint8 Tensor，`[M,ceil(K/64),2]` | MX 左输入 E8M0 原始指数码，默认 `None`。 |
| `x2_scale` | 可选 uint8 Tensor，`[N,ceil(K/64),2]` | MX 转置权重 E8M0 原始指数码，默认 `None`。 |
| `x1_quant_mode` | int | 默认 0 表示非量化，6 表示 MX。 |
| `x2_quant_mode` | int | 默认 0 表示非量化，MX 与左输入同时设为 6。 |
| `group_size` | int | 默认 0；MX 使用 4295032864，即 `32 + (1<<16) + (1<<32)`，对应 [1,1,32]。 |
| `scale_dtype` | int | 默认 293，将 uint8 尺度解释为 ACL FLOAT8_E8M0；不是数值尺度。 |
| `out_dtype` | str | 默认 `None` 沿用非量化输入类型；MX 必须为 `float16` 或 `bfloat16`。 |

### 输出

| 输出 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| `y` | `[p*M, N/p]` | 非量化与 `x1` 相同；MX 由 `out_dtype` 指定 | 当前 rank 接收并按源 rank 顺序拼接的矩阵乘结果。 |

### 数据类型

| 输入 | 支持类型 |
| --- | --- |
| `x1`、`x2` | float16、bfloat16、float8_e4m3fn、float8_e5m2 |
| `bias` | float16、bfloat16、float32；MX 仅 float32 |
| `x1_scale`、`x2_scale` | uint8（E8M0 原始指数码） |
| `y` | float16、bfloat16 |

### 规则与约束

- 此接口的量化分支为 MXFP8，不包含 K-C、FP4 或低比特通信；量化模式必须成对配置为 (6,6)，并同时提供两侧尺度。
- MX 输入不允许空张量，`M >= 1`。两侧 FP8 格式可不同，偏置仅支持 FP32；尺度的最后一维为 2，每个元素对应 32 个 K 元素，不足 64 的尾组仍保留两个尺度槽位。
- MX 仅支持 `is_trans_x2=True`，输出在通信前转换到 FP16/BF16；通信中不再进行量化或归约。

- 输入为二维 ND 张量。`x1` 不转置，`x2` 的有效收缩维必须等于 `K`。
- `N` 必须能被 `p` 整除；`M` 不要求能被 `p` 整除。通信前沿列切分，通信后沿行拼接。
- 各 rank 的输入形状、数据类型和通信属性必须一致；所有 rank 必须以相同顺序参与集合通信。
- Ascend 950PR/950DT 上：`1 <= K <= 65535`，`2 <= N <= INT32_MAX`，`0 <= p*M <= INT32_MAX`；仅允许 `M=0` 的空张量场景。
- 非量化模式的 `x1`、`x2` 与 `y` 数据类型一致。Ascend 950PR/950DT 上，FP16 输入的偏置可为 FP16/FP32，BF16 输入的偏置可为 BF16/FP32；仅 `x2` 支持转置形成的非连续布局。
- Ascend 950PR/950DT 支持 2、4、8、16 卡，使用 AI_CPU 通信，仅支持超节点内通信。原生接口不支持通算融合算子并发调用。
- Atlas A2/A3 的偏置类型、连续性及形状限制不同，不能直接套用上述 A5 约束。

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


"""A5 MatmulAlltoAll distributed golden, following mc2_test semantics."""

import torch


def _dtype(name):
    return {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "uint8": torch.uint8,
    }[str(name).lower()]


def _tensor(shape, dtype_name, value_range, seed, device):
    if shape is None:
        return None
    torch.manual_seed(int(seed))
    lo, hi = value_range or [-1.0, 1.0]
    if dtype_name == "uint8":
        return torch.randint(int(lo), int(hi) + 1, shape, dtype=torch.uint8).to(device)
    return (
        torch.empty(shape, dtype=torch.float32)
        .uniform_(float(lo), float(hi))
        .to(_dtype(dtype_name))
        .to(device)
    )


def mc2_make_rank_inputs(ctx, case_payload):
    shapes, dtypes = case_payload["input_shapes"], case_payload["dtypes"]
    ranges, attrs = case_payload.get("value_ranges") or [], case_payload["attrs"]
    seed, device = int(attrs.get("seed", 1)), ctx["device"]
    if attrs.get("is_quant", False):
        rank_seed = seed + 1009 * int(ctx["rank"])
        names = ("x1", "x2", "bias", "x1_scale", "x2_scale")
        inputs = {
            name: _tensor(shape, dtype, bounds, rank_seed + i * 37, device)
            for i, (name, shape, dtype, bounds) in enumerate(
                zip(names, shapes, dtypes, ranges)
            )
        }
        _validate_quant(ctx, inputs, attrs)
        return inputs
    # Distinct senders expose omitted peers and incorrect receive ordering.
    x1_seed = seed
    if attrs.get("mc2_accuracy_mode") == "three_way":
        x1_seed += int(ctx["rank"]) * 1000003
    bias = None
    if bool(attrs.get("is_bias", False)):
        bias = _tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            seed * 7,
            device,
        )
    return {
        "x1": _tensor(
            shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
        ),
        "x2": _tensor(
            shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device
        ),
        "bias": bias,
    }


def _validate_quant(ctx, inputs, attrs):
    """Validate the supported A5 MXFP8 interface, including packed scale layout."""
    if (attrs.get("x1_quant_mode"), attrs.get("x2_quant_mode")) != (6, 6):
        raise ValueError("QuantMatmulAlltoAll requires MX quantization modes (6, 6)")
    if not attrs.get("is_trans_x2", False):
        raise ValueError("MX quantization requires is_trans_x2=True")
    if attrs.get("group_size") != 4295032864 or attrs.get("scale_dtype") != 293:
        raise ValueError("MX requires group_size=[1,1,32] and E8M0 scale_dtype=293")
    if attrs.get("out_dtype") not in ("float16", "bfloat16"):
        raise ValueError("out_dtype must be float16 or bfloat16")
    x1, x2 = inputs["x1"], inputs["x2"]
    allowed = (torch.float8_e4m3fn, torch.float8_e5m2)
    if x1.dtype not in allowed or x2.dtype not in allowed:
        raise ValueError("MX operands must use FP8 E4M3FN or E5M2")
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("MX operands must be two-dimensional")
    m, k = x1.shape
    n, weight_k = x2.shape
    p = int(ctx["world_size"])
    if p not in (2, 4, 8, 16) or m < 1 or not 1 <= k <= 65535 or weight_k != k:
        raise ValueError("Invalid MX operand dimensions or world_size")
    if n < 2 or n % p or p * m > 2147483647 or n > 2147483647:
        raise ValueError(
            "N must be divisible by world_size and output dimensions must fit INT32"
        )
    q = (k + 63) // 64
    for name, expected in (("x1_scale", (m, q, 2)), ("x2_scale", (n, q, 2))):
        value = inputs.get(name)
        if (
            value is None
            or value.dtype != torch.uint8
            or tuple(value.shape) != expected
        ):
            raise ValueError(f"{name} must be uint8 E8M0 codes of shape {expected}")
    bias = inputs.get("bias")
    if bias is not None and (bias.dtype != torch.float32 or tuple(bias.shape) != (n,)):
        raise ValueError("MX bias must be float32 of shape [N]")


def _mx_rows(values, codes):
    """Decode the two E8M0 bytes per 64 elements, each covering 32 K elements."""
    scales = torch.exp2(codes.float() - 127.0).flatten(-2).repeat_interleave(32, dim=-1)
    return values.float() * scales[..., : values.shape[-1]]


def _alltoall(ctx, out):
    world_size = int(ctx["world_size"])
    m, n = out.shape
    if n % world_size:
        raise ValueError(f"N={n} must be divisible by world_size={world_size}")
    sent = out.reshape(m, world_size, n // world_size).permute(1, 0, 2).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    return received.reshape(m * world_size, n // world_size)


def _quant_matmul(inputs):
    x1 = _mx_rows(inputs["x1"], inputs["x1_scale"])
    weight = _mx_rows(inputs["x2"], inputs["x2_scale"]).t()
    out = torch.matmul(x1, weight)
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
    return out


def mc2_distributed_golden(ctx, inputs, attrs):
    """Local matmul followed by documented N-split/M-concat all-to-all."""
    if attrs.get("is_quant", False):
        _validate_quant(ctx, inputs, attrs)
        return _alltoall(ctx, _quant_matmul(inputs).to(_dtype(attrs["out_dtype"])))
    x2 = inputs["x2"].t() if bool(attrs.get("is_trans_x2", False)) else inputs["x2"]
    out = torch.matmul(inputs["x1"], x2)
    bias = inputs.get("bias")
    if bias is not None:
        out = (
            (out.to(torch.float32) + bias.float()).to(out.dtype)
            if bias.dtype == torch.float32
            else out + bias
        )
    world_size = int(ctx["world_size"])
    m, n = out.shape[-2:]
    if n % world_size:
        raise ValueError(f"N={n} must be divisible by world_size={world_size}")
    sent = out.reshape(m, world_size, n // world_size).permute(1, 0, 2).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    return received.reshape(m * world_size, n // world_size)


def mc2_cpu_golden(ctx, inputs, attrs):
    """Independent FP32 math and CPU communication, with no output rounding."""
    if attrs.get("is_quant", False):
        _validate_quant(ctx, inputs, attrs)
        return _alltoall(ctx, _quant_matmul(inputs))
    fp32_inputs = {
        key: value.float() if value is not None else None
        for key, value in inputs.items()
    }
    return mc2_distributed_golden(ctx, fp32_inputs, attrs)


def mc2_cpu_references(ctx, inputs, attrs):
    """Return FP32 G and a separately communicated low-precision stage model.

    The native cascade rounds matmul, optionally adds bias and rounds again,
    then performs an exact all-to-all permutation. Neither CPU path depends
    on the candidate or the NPU cascade output.
    """
    if bool(attrs.get("is_quant", False)):
        mathematical = mc2_cpu_golden(ctx, inputs, attrs)
        # A final elementwise cast commutes with the exact all-to-all permutation.
        # This preserves the NPU order (cast before exchange) without a second
        # CPU collective or any dependence on the NPU/candidate output.
        return mathematical, mathematical.to(_dtype(attrs["out_dtype"])).float()
    if int(ctx["world_size"]) != 2:
        return mc2_cpu_golden(ctx, inputs, attrs), None

    x2 = inputs["x2"].float()
    x2_eff = x2.t() if bool(attrs.get("is_trans_x2", False)) else x2
    local = torch.matmul(inputs["x1"].float(), x2_eff)
    dtype = inputs["x1"].dtype
    rounded = local.to(dtype).float()
    bias = inputs.get("bias")
    if bias is not None:
        local = local + bias.float()
        rounded = (rounded + bias.float()).to(dtype).float()
    mathematical = _alltoall(ctx, local)
    stage_rounded = _alltoall(ctx, rounded)
    return mathematical, stage_rounded


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    if attrs.get("is_quant", False):
        _validate_quant(ctx, inputs, attrs)
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            int(ctx["world_size"]),
            bias=inputs.get("bias"),
            is_trans_x2=True,
            x1_scale=inputs["x1_scale"],
            x2_scale=inputs["x2_scale"],
            x1_quant_mode=6,
            x2_quant_mode=6,
            group_size=attrs["group_size"],
            scale_dtype=attrs["scale_dtype"],
            out_dtype=attrs["out_dtype"],
        )
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        is_trans_x2=bool(attrs.get("is_trans_x2", False)),
    )
```

## 6. 算子调用示例

以下示例在 Ascend 950PR/950DT 上调用。假设通信域已初始化，`device` 为当前 rank 的 NPU 设备，`hcomm_info` 为该域名称，`world_size=2`；所有 rank 使用相同形状和调用顺序。

```python
import torch
import cann_bench

x1 = torch.randn(16, 256, dtype=torch.float16, device=device)
x2 = torch.randn(256, 128, dtype=torch.float16, device=device)
y = cann_bench.matmul_all_to_all(
    x1, x2, hcomm_info, world_size, bias=None, is_trans_x2=False,
)
# y: [world_size * 16, 128 // world_size]
```
