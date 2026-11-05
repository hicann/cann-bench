# QuantAllReduce 算子 API 描述

## 1. 算子简介

QuantAllReduce 在通信过程中对 INT8 数据反量化，并对各 rank 的结果求和，每个 rank 得到完整结果。同一个 token 中，每连续 128 个通道共享一个 FP32 系数。

计算语义对应 `aclnnQuantAllReduce` 的 pertoken-pergroup 模式，以下定义 Ascend 950DT（A5、arch35）上的 INT8/FP32 接口。

## 2. 算子定义

设通信域大小为 `p=world_size`，源 rank 为 `s`，当前 rank 为 `r`。输入 `x_s` 为 `[BS,H]` 或 `[B,S,H]`。将前导维按行优先顺序合并为 `T=BS` 或 `T=B*S`：

$$
D_s[t,h] = \operatorname{FP32}(x_s[t,h])\,scales_s[t,\lfloor h/128\rfloor]
$$

$$
Z[t,h] = \sum_{s=0}^{p-1} D_s[t,h]
$$

每个 rank 返回 $Y_r=\operatorname{Cast}_{out\_dtype}(Z)$，并恢复 `x` 的原始形状。二维输入输出 `[BS,H]`，三维输入输出 `[B,S,H]`。

每个 rank 可以使用不同的输入及 scale。先将各 rank 的 INT8 值乘以自己的系数，再求 FP32 和；不先对 INT8 求和，也不将各 rank 的 scale 合并。只在归约完成后转换输出 dtype，不提前将 scale 或单 rank 反量化结果转为 FP16/BF16。归约固定为 SUM，分组大小固定为 128。

## 3. 接口规范

### 算子原型

```python
cann_bench.quant_all_reduce(
    Tensor x, Tensor scales, str hcomm_info, int world_size,
    str out_dtype="float16"
) -> Tensor
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x` | INT8 Tensor，`[BS,H]` 或 `[B,S,H]` | 当前 rank 的低比特输入，有效值为闭区间 `[-128,127]`。 |
| `scales` | FP32 Tensor，`[BS,H/128]` 或 `[B,S,H/128]` | 每 token、每 128 个连续通道共享一个系数。 |
| `hcomm_info` | str | 已初始化的专用 HCCL 通信域名称；候选 host 层须据此创建自己的 channel context，不能改用默认域。 |
| `world_size` | int | 实际通信域大小，支持 2、4、8。 |
| `out_dtype` | str | `float16`、`bfloat16` 或 `float32`；默认 `float16`。 |

### 输出

仅返回一个 Tensor `y`，dtype 由 `out_dtype` 指定。输出形状与 `x` 一致，每个 rank 返回完整归约结果。

### 规则与约束

- 仅支持 Ascend 950DT（A5、arch35），使用 UB-Memory 通信。
- `x` 与 `scales` 必须是连续 ND 张量，位于同一个设备；不支持空张量。
- `1024<=H<=8192`，且 `H%128==0`。二维输入的 `BS` 或三维输入的 `B*S` 必须能被 `world_size` 整除。
- 各 rank 的 shape、dtype、输出类型和通信属性必须一致，并按相同顺序调用。
- 一个 scale 仅作用于同一 token 的连续 128 个通道，不允许跨 token 广播。INT8 输入必须配 FP32 scale。
- 同一通信域只允许连续执行 `aclnnQuantAllReduce` 与 `aclnnQuantReduceScatter`，不允许混入其它通信算子。
- 令 `X=align32(x.nbytes)`、`S=align32(scales.nbytes)`。当前原生 QuantAllReduce 与 QuantReduceScatter 的 Python buffer 实现均按 `B=2*((X+S)*world_size+1 MiB)` 字节分配 channel buffer，本接口采用该保守容量。`HCCL_BUFFSIZE` 的单位是 MiB，应在初始化通信域前至少设为 `ceil(B/1 MiB)`；这也满足共享 tiling 检查的较小下限 `2*(X+S+1 MiB)`。
- 本接口不包含 FP8、MX scale、偏置或其它归约类型。

### 候选通信资源初始化

`hcomm_info` 是通信域名称。候选 host 层必须按传入的专用名称建立 channel context 和通信 buffer。可参考源码 `CommContextManager(hcomm_info, world_size, backend={"Ascend950":"channel"}, opName=..., customCclBufferSize=B)` 及其 `create_context()` 流程，也可在候选 host 中实现等价的通信资源初始化，不要求安装 ops-transformer Python 包。各 rank 的 `opName` 与容量保持一致，将该域的 context 和实际 `ccl_buffer_size` 传给生成的自定义 AscendC kernel。建议按通信域缓存通信资源；context、管理器和 buffer 必须保活至相应设备工作完成，释放或重建前先同步。

当前原生低层签名 `npu_quant_all_reduce(context, x, scales, hccl_buffer_size, world_size, ...)` 仅用于核对通信资源和输出类型契约，其 `output_dtype` 枚举为 `float16=5`、`bfloat16=15`、`float32=6`。候选的实际计算由生成的 AscendC kernel 完成，host 层按本接口的 `out_dtype` 分配输出。

不能简单转发到当前 `ops-transformer` 的高层 `quant_all_reduce(x, scales, hcom, ...)`：它内部固定获取 `_get_default_group()`，随后 buffer 又从该默认 group 获取名称，传入的专用 `hcom` 实际未决定 context 所属域，会破坏通信域隔离。候选 host 层应直接按上述专用名称创建并缓存通信资源。

规格依据：`ops-transformer/mc2/quant_all_reduce/README.md`；输入与 scale 校验位于共享 `quant_reduce_scatter_util_tiling.cpp`，FP32 反量化及求和位于共享 `vec_comp.h`。

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


"""Ascend 950DT INT8 per-token/per-group QuantAllReduce references."""

import torch


def _dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(name).lower()]


def _validate(ctx, inputs, attrs):
    x, scales = inputs["x"], inputs["scales"]
    p = int(ctx["world_size"])
    if p not in (2, 4, 8) or int(attrs.get("world_size", p)) != p:
        raise ValueError("world_size must match the 2-, 4-, or 8-rank group")
    if x.ndim not in (2, 3) or any(dim <= 0 for dim in x.shape):
        raise ValueError("x must be a nonempty 2D or 3D tensor")
    h = x.shape[-1]
    if not 1024 <= h <= 8192 or h % 128 or (x.numel() // h) % p:
        raise ValueError(
            "H must be 128-aligned in [1024,8192]; BS must be divisible by world_size"
        )
    if tuple(scales.shape) != tuple(x.shape[:-1]) + (h // 128,):
        raise ValueError(
            "scales must contain one FP32 value per token and 128 channels"
        )
    if x.dtype != torch.int8 or scales.dtype != torch.float32:
        raise ValueError("this contract requires INT8 x and FP32 scales")
    if not x.is_contiguous() or not scales.is_contiguous() or x.device != scales.device:
        raise ValueError("x and scales must be contiguous on the same device")
    if (
        attrs.get("x_dtype", "int8") != "int8"
        or attrs.get("scales_dtype", "float32") != "float32"
    ):
        raise ValueError("only INT8 x and FP32 scales are supported by this contract")
    if attrs.get("group_size", 128) != 128 or attrs.get("reduce_op", "sum") != "sum":
        raise ValueError("group_size=128 and reduce_op=sum are required")
    _dtype(attrs.get("out_dtype", "float16"))
    return p


def mc2_make_rank_inputs(ctx, case_payload):
    shapes, attrs = case_payload["input_shapes"], case_payload["attrs"]
    ranges = case_payload.get("value_ranges") or [[-128, 127], [0.001, 0.25]]
    seed = int(attrs.get("seed", 1)) + int(ctx["rank"]) * 1000003
    x_generator = torch.Generator(device="cpu").manual_seed(seed)
    scale_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    x = torch.randint(
        int(ranges[0][0]),
        int(ranges[0][1]) + 1,
        shapes[0],
        dtype=torch.int8,
        generator=x_generator,
    )
    scales = torch.empty(shapes[1], dtype=torch.float32).uniform_(
        float(ranges[1][0]), float(ranges[1][1]), generator=scale_generator
    )
    inputs = {"x": x.to(ctx["device"]), "scales": scales.to(ctx["device"])}
    _validate(ctx, inputs, attrs)
    return inputs


def mc2_distributed_golden(ctx, inputs, attrs):
    """Dequantize in FP32, perform all-reduce, then cast only the final result."""
    _validate(ctx, inputs, attrs)
    local = inputs["x"].float() * inputs["scales"].repeat_interleave(128, dim=-1)
    ctx["dist"].all_reduce(local, op=ctx["dist"].ReduceOp.SUM)
    return local.to(_dtype(attrs.get("out_dtype", "float16")))


def mc2_cpu_golden(ctx, inputs, attrs):
    """Independent FP32 group-axis dequantization and sum; retain input shape and precision."""
    _validate(ctx, inputs, attrs)
    x, scales = inputs["x"], inputs["scales"]
    groups = x.float().reshape(*scales.shape, 128)
    total = (groups * scales.unsqueeze(-1)).reshape(x.shape)
    ctx["dist"].all_reduce(total, op=ctx["dist"].ReduceOp.SUM)
    return total


def mc2_cpu_references(ctx, inputs, attrs):
    """Model the one final output cast after FP32 dequantization and reduction."""
    mathematical = mc2_cpu_golden(ctx, inputs, attrs)
    return mathematical, mathematical.to(
        _dtype(attrs.get("out_dtype", "float16"))
    ).float()


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["x"],
        inputs["scales"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        out_dtype=attrs.get("out_dtype", "float16"),
    )
```

## 6. 算子调用示例

在 Ascend 950DT 上，假设专用通信域已初始化，`device` 为当前 rank 的设备，`hcomm_info` 为该域名称，`world_size=2`。每个 rank 均执行：

```python
import torch
import cann_bench

x = torch.randint(-128, 128, (2, 16, 1024), dtype=torch.int8, device=device)
scales = torch.rand((2, 16, 8), dtype=torch.float32, device=device) * 0.25
y = cann_bench.quant_all_reduce(
    x, scales, hcomm_info, world_size, out_dtype="bfloat16",
)
# y: [2, 16, 1024], bfloat16
```
