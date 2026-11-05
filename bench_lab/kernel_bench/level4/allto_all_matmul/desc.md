# AlltoAllMatmul 算子 API 描述

## 1. 算子简介

AlltoAllMatmul 将 AllToAll 集合通信与矩阵乘融合执行：先把按行切分的输入分发到目标 rank，再沿列维拼接收到的数据，最后与本地权重相乘。支持 FP16/BF16 非量化矩阵乘和 MXFP8 量化矩阵乘、可选偏置、权重转置以及 AllToAll 中间结果输出。

计算语义对应 `aclnnAlltoAllMatmul` 和 `aclnnAlltoAllQuantMatmul`，以下描述 Ascend 950PR/950DT（A5）上的 Python 接口。

## 2. 算子定义

设通信域包含 `p = world_size` 个 rank，`r` 为当前 rank，`s` 为源 rank。每个 rank 的输入 `X_s` 为 `[M, K]`，有效权重 `W_r` 为 `[p*K, N]`。

$$
A_r = \operatorname{Concat}_{\mathrm{dim}=1}\left(
X_0[rM/p:(r+1)M/p,:],\ldots,X_{p-1}[rM/p:(r+1)M/p,:]
\right)
$$

$$
Y_r = A_r W_r + b_r
$$

每个源 rank 沿 `M` 维等分输入，第 `r` 块发送到 rank `r`；接收端按源 rank 顺序沿 `K` 维拼接。`A_r` 为 `[M/p, p*K]`，`Y_r` 为 `[M/p, N]`。

`is_trans_x2=True` 时 `W_r = x2_r.T`，否则 `W_r = x2_r`。可选偏置 `b_r` 在本地矩阵乘后按列相加，不参与 AllToAll。这里的通信是数据重分布，不是求和归约。

MXFP8 模式下，`x1_scale` 随输入按同样的源 rank 顺序通信和拼接。每个 E8M0 字节 `e` 表示缩放因子 `2**(e-127)`，沿 K 维每连续 32 个元素共用一个因子；形状末维 2 存放相邻的两个块。将 `A_r` 和转置存放的权重分别按各自尺度反量化到 FP32 后矩阵相乘，再加 FP32 偏置，最后转换为 `out_dtype`。`alltoall_y` 返回通信后的原始 FP8 数值，不是反量化结果。

## 3. 接口规范

### 算子原型

```python
cann_bench.all_to_all_matmul(
    Tensor x1, Tensor x2, str hcomm_info, int world_size,
    Tensor? bias=None, bool is_trans_x2=False, bool alltoall_out_flag=False,
    bool is_quant=False, Tensor? x1_scale=None, Tensor? x2_scale=None,
    int x1_quant_mode=6, int x2_quant_mode=6, int group_size=4295032864,
    int scale_dtype=293, str out_dtype="bfloat16"
) -> (Tensor y, Tensor? alltoall_y)
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x1` | Tensor，`[M, K]` | AllToAll 的本地输入；非量化为 FP16/BF16，MX 为 FP8 E4M3FN/E5M2。 |
| `x2` | Tensor，`[p*K, N]` 或 `[N, p*K]` | 矩阵乘权重；非量化类型与 `x1` 相同，MX 可独立选择 FP8 E4M3FN/E5M2；后一形状用于转置模式。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 通信域的 rank 数，必须与实际通信域大小一致。 |
| `bias` | 可选 Tensor，`[N]` | 当前 rank 的列偏置，默认 `None`；类型约束见下文。 |
| `is_trans_x2` | bool | 是否在矩阵乘中转置 `x2`，默认 `False`。 |
| `alltoall_out_flag` | bool | 是否返回 AllToAll 中间结果，默认 `False`。 |
| `is_quant` | bool | 默认 `False`；为 `True` 时执行 MXFP8 AlltoAllQuantMatmul。 |
| `x1_scale` | 可选 uint8 Tensor，`[M,K/64,2]` | 原始 E8M0 左输入尺度，默认 `None`；MX 时必需。 |
| `x2_scale` | 可选 uint8 Tensor，`[N,p*K/64,2]` | 转置权重的原始 E8M0 尺度，默认 `None`；MX 时必需。 |
| `x1_quant_mode` | int | MX 左输入量化模式，默认且固定为 `6`。 |
| `x2_quant_mode` | int | MX 权重量化模式，默认且固定为 `6`。 |
| `group_size` | int | 默认 `4295032864`，表示分组大小 `[1,1,32]`。 |
| `scale_dtype` | int | 默认 `293`，适配层将 uint8 尺度解释为 E8M0。 |
| `out_dtype` | str | MX 输出类型为 `float16` 或 `bfloat16`，默认 `bfloat16`；非量化时忽略。 |

### 输出

| 输出 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| `y` | `[M/p, N]` | 非量化与 `x1` 相同，MX 由 `out_dtype` 指定 | 当前 rank 的矩阵乘结果。 |
| `alltoall_y` | `[M/p, p*K]` | 与 `x1` 相同 | AllToAll 重分布结果，未经过矩阵乘。 |

始终返回二元组。`alltoall_out_flag=False` 时第二项为 `None`，为 `True` 时第二项为 `alltoall_y`。

### 规则与约束

- 输入为二维 ND 张量，`M` 必须能被 `p` 整除；权重的有效收缩维为 `p*K`，不是 `K`。
- 各 rank 的输入形状、数据类型和通信属性必须一致；所有 rank 必须以相同顺序参与集合通信。
- 非量化时 `x1`、`x2`、`y` 和非空的 `alltoall_y` 数据类型一致；MX 时左右矩阵可分别为 `float8_e4m3fn` 或 `float8_e5m2`，偏置为 `float32`，输出为 `float16`/`bfloat16`，中间结果与 `x1` 同类型。
- 非量化时：`2 <= p*K <= 65535`，`0 <= M <= INT32_MAX`，`1 <= N <= INT32_MAX`；仅允许 `M=0` 的空张量场景。
- MX 时：`M>=2` 且可被 `p` 整除、`K` 可被 64 整除、`p*K<=65535`，不支持空张量；必须令 `is_trans_x2=True`，权重按 `[N,p*K]` 连续存放，左右尺度分别匹配 token 行和输出列。尺度码不能为表示 NaN 的 `255`。
- MX 的量化模式固定为 `(6,6)`，通信附加量化关闭；不传通信尺度或量化 offset。
- Ascend 950PR/950DT 上：FP16 输入的偏置可为 FP16/FP32，BF16 输入的偏置可为 BF16/FP32。`x1` 和偏置须连续，`x2` 支持转置形成的非连续布局。
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


"""A5 AlltoAllMatmul distributed golden, following mc2_test semantics."""

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
    # Different senders must remain distinguishable after the all-to-all permutation.
    x1_seed = seed + int(ctx["rank"]) * 1000003
    bias = None
    if bool(attrs.get("is_bias", False)):
        bias = _tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            seed * 7,
            device,
        )
    inputs = {
        "x1": _tensor(
            shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
        ),
        "x2": _tensor(
            shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device
        ),
        "bias": bias,
    }
    if bool(attrs.get("is_quant", False)):
        for name, index, scale_seed in (
            ("x1_scale", 3, x1_seed + 17),
            ("x2_scale", 4, seed + 29),
        ):
            torch.manual_seed(scale_seed)
            lo, hi = ranges[index]
            inputs[name] = torch.randint(
                int(lo), int(hi) + 1, shapes[index], dtype=torch.uint8
            ).to(device)
        _validate_mx(inputs, attrs, int(ctx["world_size"]))
    return inputs


def _validate_mx(inputs, attrs, world_size):
    x1, x2 = inputs["x1"], inputs["x2"]
    m, k = x1.shape
    if (
        m < world_size
        or m % world_size
        or k % 64
        or not 1 <= k * world_size <= 65535
        or not bool(attrs.get("is_trans_x2", False))
    ):
        raise ValueError(
            "MX requires M divisible by world_size, K divisible by 64 and transposed x2"
        )
    if x2.ndim != 2 or x2.shape[1] != k * world_size:
        raise ValueError("MX x2 must be stored as [N, world_size*K]")
    if (int(attrs.get("x1_quant_mode", 6)), int(attrs.get("x2_quant_mode", 6))) != (
        6,
        6,
    ):
        raise ValueError("Only MX quantization modes (6, 6) are supported")
    if (
        int(attrs.get("group_size", 4295032864)) != 4295032864
        or int(attrs.get("scale_dtype", 293)) != 293
    ):
        raise ValueError("MX requires group_size=[1,1,32] and E8M0 scale encoding 293")
    for name, shape in (
        ("x1_scale", (m, k // 64, 2)),
        ("x2_scale", (x2.shape[0], k * world_size // 64, 2)),
    ):
        scale = inputs.get(name)
        if scale is None or tuple(scale.shape) != shape or scale.dtype != torch.uint8:
            raise ValueError(f"{name} must contain raw E8M0 bytes with shape {shape}")


def _alltoall_columns(ctx, tensor):
    world_size = int(ctx["world_size"])
    m = tensor.shape[0]
    columns = tensor.numel() // m
    # Transport FP8 bit patterns; HCCL/Gloo do not need a native FP8 wire type.
    is_fp8 = tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    wire = tensor.contiguous().view(torch.uint8) if is_fp8 else tensor
    sent = wire.reshape(world_size, m // world_size, columns).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    result = (
        received.permute(1, 0, 2)
        .contiguous()
        .reshape(m // world_size, world_size * columns)
    )
    return result.view(tensor.dtype) if is_fp8 else result


def _mx_dequant(tensor, scales):
    # Two E8M0 bytes describe consecutive 32-element K blocks.
    expanded = torch.exp2(scales.float().reshape(scales.shape[0], -1) - 127)
    return tensor.float() * expanded.repeat_interleave(32, dim=-1)


def _mx_golden(ctx, inputs, attrs, round_output):
    _validate_mx(inputs, attrs, int(ctx["world_size"]))
    alltoall_out = _alltoall_columns(ctx, inputs["x1"])
    received_scale = _alltoall_columns(ctx, inputs["x1_scale"])
    left = _mx_dequant(alltoall_out, received_scale)
    right = _mx_dequant(inputs["x2"], inputs["x2_scale"]).t()
    out = torch.matmul(left, right)
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
    if round_output:
        out = out.to(_dtype(attrs.get("out_dtype", "bfloat16")))
    return out, alltoall_out if bool(attrs.get("alltoall_out_flag", False)) else None


def mc2_distributed_golden(ctx, inputs, attrs):
    """All-to-all x1 on NPU, then local matmul, matching the documented formula."""
    if bool(attrs.get("is_quant", False)):
        return _mx_golden(ctx, inputs, attrs, round_output=True)
    x1, x2 = inputs["x1"], inputs["x2"]
    world_size = int(ctx["world_size"])
    m, k = x1.shape[-2:]
    if m % world_size:
        raise ValueError(f"M={m} must be divisible by world_size={world_size}")
    sent = x1.reshape(world_size, m // world_size, k).contiguous()
    received = torch.empty_like(sent)
    ctx["dist"].all_to_all_single(received, sent)
    alltoall_out = (
        received.permute(1, 0, 2).contiguous().reshape(m // world_size, k * world_size)
    )
    x2_eff = x2.t() if bool(attrs.get("is_trans_x2", False)) else x2
    out = torch.matmul(alltoall_out, x2_eff)
    bias = inputs.get("bias")
    if bias is not None:
        out = (
            (out.to(torch.float32) + bias.float()).to(out.dtype)
            if bias.dtype == torch.float32
            else out + bias
        )
    return out, alltoall_out if bool(attrs.get("alltoall_out_flag", False)) else None


def mc2_cpu_golden(ctx, inputs, attrs):
    """Independent FP32 math and CPU communication, with no output rounding."""
    if bool(attrs.get("is_quant", False)):
        fp32_inputs = {
            key: value.float()
            if value is not None and key not in ("x1_scale", "x2_scale")
            else value
            for key, value in inputs.items()
        }
        return _mx_golden(ctx, fp32_inputs, attrs, round_output=False)
    fp32_inputs = {
        key: value.float() if value is not None else None
        for key, value in inputs.items()
    }
    return mc2_distributed_golden(ctx, fp32_inputs, attrs)


def mc2_cpu_references(ctx, inputs, attrs):
    """Return mathematical G and a CPU model of the NPU rounding stages.

    All-to-all only permutes the already-rounded input, so one FP32 CPU
    exchange feeds both paths exactly. Matmul rounds before optional bias.
    """
    if bool(attrs.get("is_quant", False)):
        mathematical = mc2_cpu_golden(ctx, inputs, attrs)
        output, exchanged = mathematical
        # MX dequantization and matmul stay FP32; only the final matrix is cast.
        rounded = output.to(_dtype(attrs.get("out_dtype", "bfloat16"))).float()
        return mathematical, (rounded, exchanged)
    if int(ctx["world_size"]) != 2:
        return mc2_cpu_golden(ctx, inputs, attrs), None

    received = _alltoall_columns(ctx, inputs["x1"].float())
    x2 = inputs["x2"].float()
    x2_eff = x2.t() if bool(attrs.get("is_trans_x2", False)) else x2
    mathematical = torch.matmul(received, x2_eff)
    dtype = inputs["x1"].dtype
    rounded = mathematical.to(dtype).float()
    bias = inputs.get("bias")
    if bias is not None:
        mathematical = mathematical + bias.float()
        rounded = (rounded + bias.float()).to(dtype).float()
    exchanged = received if bool(attrs.get("alltoall_out_flag", False)) else None
    return (mathematical, exchanged), (rounded, exchanged)


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    if bool(attrs.get("is_quant", False)):
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            int(ctx["world_size"]),
            bias=inputs.get("bias"),
            is_trans_x2=bool(attrs.get("is_trans_x2", False)),
            alltoall_out_flag=bool(attrs.get("alltoall_out_flag", False)),
            is_quant=True,
            x1_scale=inputs["x1_scale"],
            x2_scale=inputs["x2_scale"],
            x1_quant_mode=int(attrs.get("x1_quant_mode", 6)),
            x2_quant_mode=int(attrs.get("x2_quant_mode", 6)),
            group_size=int(attrs.get("group_size", 4295032864)),
            scale_dtype=int(attrs.get("scale_dtype", 293)),
            out_dtype=attrs.get("out_dtype", "bfloat16"),
        )
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        is_trans_x2=bool(attrs.get("is_trans_x2", False)),
        alltoall_out_flag=bool(attrs.get("alltoall_out_flag", False)),
    )
```

## 6. 算子调用示例

以下示例在 Ascend 950PR/950DT 上调用。假设通信域已初始化，`device` 为当前 rank 的 NPU 设备，`hcomm_info` 为该域名称，`world_size=2`；所有 rank 使用相同形状和调用顺序。

```python
import torch
import cann_bench

x1 = torch.randn(16, 256, dtype=torch.float16, device=device)
x2 = torch.randn(world_size * 256, 128, dtype=torch.float16, device=device)
y, alltoall_y = cann_bench.all_to_all_matmul(
    x1, x2, hcomm_info, world_size,
    bias=None, is_trans_x2=False, alltoall_out_flag=True,
)
# y: [16 // world_size, 128]
# alltoall_y: [16 // world_size, world_size * 256]
```
