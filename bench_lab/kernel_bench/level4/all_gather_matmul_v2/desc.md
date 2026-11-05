# AllGatherMatmulV2 算子 API 描述

## 1. 算子简介

AllGatherMatmulV2 将 AllGather 集合通信、矩阵乘及反量化融合执行，适用于低精度张量并行中的输入聚合与线性变换。先沿行维聚合各 rank 的输入，再使用本地权重和反量化参数计算输出，可同时返回聚合后的原始输入。

计算语义对应 `aclnnAllGatherMatmulV2`。以下 Python 接口描述 INT8 per-token/per-channel 和 FP8 per-tensor 两种量化形式；不包含原生接口的 per-block、MX、输出量化及 amax 输出。

## 2. 算子定义

设通信域包含 `p = world_size` 个 rank，编号为 `0` 到 `p-1`。本地输入 `X_r` 为 `[M, K]`，有效权重 `W_r` 为 `[K, N]`。`is_trans_b=True` 时 `W_r = x2_r.T`，否则 `W_r = x2_r`。

$$
G = \operatorname{Concat}_{\mathrm{dim}=0}(X_0, X_1, \ldots, X_{p-1})
$$

### INT8 per-token/per-channel

`x1_scale` 为每个本地输入行的尺度，`x2_scale` 为每个输出列的尺度。输入尺度须与输入按相同 rank 顺序聚合：

$$
S = \operatorname{Concat}_{\mathrm{dim}=0}(s_0, s_1, \ldots, s_{p-1})
$$

$$
Y_r = \operatorname{Cast}_{D}\left((G W_r) \odot S[:,\mathrm{None}] \odot t_r[\mathrm{None},:]\right)
$$

`G W_r` 表示 INT8 矩阵乘的整数累加结果；`S` 为 `[p*M]`，`t_r` 为 `[N]`。该形式的 `bias` 为 `None`。

### FP8 per-tensor

`x1_scale`、`x2_scale` 均为 `[1]` 的标量尺度。聚合的是 FP8 输入，不是尺度；当前 rank 的两个尺度应用于整个聚合矩阵的乘法结果：

$$
Y_r = \operatorname{Cast}_{D}\left((G W_r + b_r)\,s_r t_r\right)
$$

`bias` 为可选 FP32 列偏置，在反量化缩放之前加入累加结果；省略时取零。FP8 矩阵元素按其实际数值参与乘法，而不是把存储字节解释为 INT8。`D` 由 `out_dtype` 指定。

两种形式的 `Y_r` 均为 `[p*M, N]`。聚合结果 `G` 为 `[p*M, K]`，保持 `x1` 的原始低精度数据类型，不做反量化。

## 3. 接口规范

### 算子原型

```python
cann_bench.all_gather_matmul_v2(
    Tensor x1, Tensor x2, str hcomm_info, int world_size,
    Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None,
    bool gather_output=True, bool is_trans_b=False, str out_dtype="float16",
    str x1_dtype="int8", str x2_dtype="int8",
    str x1_scale_dtype="float32", str x2_scale_dtype="float32",
    int block_size=128, int[] group_sizes=[]
) -> Tensor | (Tensor y, Tensor gather_out)
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x1` | Tensor，`[M, K]` | 本地低精度输入，类型组合见下表。 |
| `x2` | Tensor，`[K, N]` 或 `[N, K]` | 本地权重，后一形状用于转置模式。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 通信域的 rank 数，须与实际大小一致。 |
| `bias` | 可选 Tensor，`[N]` | FP8 per-tensor 的 FP32 累加偏置；INT8 形式为 `None`。 |
| `x1_scale` | FP32 Tensor | INT8 为 `[M]`，FP8 per-tensor 为 `[1]`；量化时必须提供。 |
| `x2_scale` | FP32 Tensor | INT8 为 `[N]`，FP8 per-tensor 为 `[1]`；量化时必须提供。 |
| `gather_output` | bool | 是否返回原始聚合输入，默认 `True`。 |
| `is_trans_b` | bool | 是否在矩阵乘中转置 `x2`，默认 `False`。 |
| `out_dtype` | str | 输出类型，默认 `float16`；允许的组合见下表。 |
| `x1_dtype` | str | `x1` 的逻辑类型，须与张量实际类型匹配。 |
| `x2_dtype` | str | `x2` 的逻辑类型，须与张量实际类型匹配。 |
| `x1_scale_dtype` | str | `x1_scale` 的类型，本文两种形式均为 `float32`。 |
| `x2_scale_dtype` | str | `x2_scale` 的类型，本文两种形式均为 `float32`。 |
| `block_size` | int | 接口保留的量化块属性，声明默认值为 128；A5 per-tensor 形式须显式传 0，不表示通信切分大小。 |
| `group_sizes` | int[] | 量化分组属性，本文两种形式使用空列表 `[]`，不启用分块量化。 |

### 数据类型与输出

| 量化形式 | `x1` / `x2` | `out_dtype` | 对应产品 |
| --- | --- | --- | --- |
| INT8 per-token/per-channel | 均为 `int8` | `float16`、`bfloat16` | Atlas A2/A3 |
| FP8 per-tensor | 各自可为 `float8_e4m3fn` 或 `float8_e5m2`，允许混合 | `float16`、`bfloat16`、`float32` | Ascend 950PR/950DT |

`gather_output=True` 时返回 `(y, gather_out)`，形状分别为 `[p*M, N]`、`[p*M, K]`；为 `False` 时仅返回 `y`。`gather_out.dtype` 与 `x1.dtype` 一致，不受 `out_dtype` 影响。

### 规则与约束

- `x1`、`x2` 均为二维 ND 张量；`x1` 不转置，`x2` 支持转置。有效权重的收缩维必须等于 `K`，且 `256 <= K < 65535`。
- 各 rank 的 `x1` 形状与数据类型一致，集合通信调用顺序一致。`M` 是每个 rank 的行数，不要求被 `p` 整除。
- INT8 的行尺度随输入聚合；FP8 per-tensor 的标量尺度不聚合。不能把二者混用，也不能把低精度输入直接作为反量化后的数值。
- 原生 ACLNN 使用的 INT8 行尺度、列尺度形状分别为 `[M, 1]`、`[1, N]`；此 Python 接口用一维向量表达同一广播语义。
- Ascend 950PR/950DT 的本文接口形式要求 `x1_dtype`、`x2_dtype` 显式指定为 FP8 类型，`block_size=0`；不能沿用签名中的 INT8 默认值。
- Ascend 950PR/950DT 支持 2、4、8、16、32、64 卡以及 CCU/AICPU 通信。CCU 模式下，AllGather 总数据量 `p*M*K*sizeof(x1.dtype)` 不得超过 `63*256 MB`，实际可用上限还受形状影响。
- Atlas A2/A3 的 INT8 形式支持 2、4、8 卡，不支持空张量，通信缓冲区至少 200 MB。产品间的类型和通信约束不可互换。

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


from typing import Any, Dict, Optional, Tuple, Union

import torch


def all_gather_matmul_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0).float()
    out = torch.matmul(gathered, x2_eff.float())
    is_fp8 = x1.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8:
        if (
            x1_scale is None
            or x2_scale is None
            or x1_scale.numel() != 1
            or x2_scale.numel() != 1
        ):
            raise ValueError("FP8 AllGatherMatmulV2 requires per-tensor scalar scales")
        # Match the A5 distributed contract: accumulator bias is scaled too.
        if bias is not None:
            out = out + bias.float()
        scale = x1_scale.float() * x2_scale.float()
        scale = (scale.contiguous().view(torch.int32) & -8192).view(torch.float32)
        out = out * scale.reshape(1)
    else:
        if x1_scale is not None:
            scale = (
                torch.cat([x1_scale] * int(world_size), dim=0).float().reshape(-1, 1)
            )
            out = out * scale
        if x2_scale is not None:
            out = out * x2_scale.float().reshape(1, -1)
        if bias is not None:
            out = out + bias.float()
    out = out.to(_torch_dtype(out_dtype))
    if gather_output:
        return out, gathered.to(x1.dtype)
    return out


def mc2_make_rank_inputs(
    ctx: Dict[str, Any], case_payload: Dict[str, Any]
) -> Dict[str, Any]:
    device = ctx["device"]
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    attrs = case_payload["attrs"]
    seed = int(attrs.get("seed", 1))
    weight_same = int(attrs.get("weight_same", 1))
    rank = int(ctx["rank"])

    a5 = attrs.get("mc2_accuracy_mode") == "three_way"
    # Distinct shards make a missing all-gather observable; preserve legacy seeds.
    x1_seed = seed + rank * 1000003 if a5 else seed
    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0], x1_seed, device)
    x2_seed = seed + rank * 2 if weight_same else seed
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1], x2_seed, device)

    bias = None
    if (
        attrs.get("mc2_accuracy_mode") != "three_way"
        and bool(attrs.get("is_bias", False))
        and len(shapes) > 2
        and shapes[2] is not None
    ):
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2], rank * 3, device)

    x1_scale_index = 2 if a5 else (3 if bool(attrs.get("is_bias", False)) else 2)
    x2_scale_index = x1_scale_index + 1
    if a5:
        bias = (
            _make_tensor(shapes[4], dtypes[4], ranges[4], rank * 3, device)
            if attrs.get("is_bias", False)
            else None
        )
    x1_scale = _make_tensor(
        shapes[x1_scale_index],
        dtypes[x1_scale_index],
        ranges[x1_scale_index],
        rank * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        rank * 11,
        device,
    )
    return {
        "x1": x1,
        "x2": x2,
        "bias": bias,
        "x1_scale": x1_scale,
        "x2_scale": x2_scale,
    }


def mc2_call_candidate(
    candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    return candidate(
        inputs["x1"],
        inputs["x2"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
        gather_output=bool(attrs.get("gather_output", True)),
        is_trans_b=bool(attrs.get("is_trans_b", False)),
        out_dtype=str(attrs.get("out_dtype", "float16")),
        x1_dtype=str(attrs.get("x1_dtype", "int8")),
        x2_dtype=str(attrs.get("x2_dtype", "int8")),
        x1_scale_dtype=str(attrs.get("x1_scale_dtype", "float32")),
        x2_scale_dtype=str(attrs.get("x2_scale_dtype", "float32")),
        block_size=int(attrs.get("block_size", 128)),
        group_sizes=list(attrs.get("group_sizes", [])),
    )


def mc2_distributed_golden(
    ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    """Distributed golden following mc2_test's hccl_mm reference path."""
    torch_npu = ctx["torch_npu"]
    dist = ctx["dist"]
    world_size = int(ctx["world_size"])
    gather_shape = [inputs["x1"].shape[0] * world_size] + list(inputs["x1"].shape[1:])
    gathered = torch.empty(
        gather_shape, dtype=inputs["x1"].dtype, device=inputs["x1"].device
    )
    dist._all_gather_base(gathered, inputs["x1"])
    gathered_x1_scale = inputs.get("x1_scale")
    if gathered_x1_scale is not None and attrs.get("mc2_accuracy_mode") != "three_way":
        scale_shape = [gathered_x1_scale.shape[0] * world_size] + list(
            gathered_x1_scale.shape[1:]
        )
        gathered_scale = torch.empty(
            scale_shape, dtype=gathered_x1_scale.dtype, device=gathered_x1_scale.device
        )
        dist._all_gather_base(gathered_scale, gathered_x1_scale)
        gathered_x1_scale = gathered_scale
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    if str(attrs.get("x1_dtype", "int8")).lower() in (
        "fp16",
        "float16",
        "bf16",
        "bfloat16",
    ):
        output = torch.matmul(gathered, x2_eff)
        if inputs.get("bias") is not None:
            output = output + inputs["bias"]
    else:
        scale = inputs.get("x2_scale")
        if attrs.get("mc2_accuracy_mode") == "three_way":
            # FP8 per-tensor MC2 uses (G @ W + bias) * trunc19(s1*s2).
            # Two FP32 scales select the dynamic quant-matmul path, whose
            # FP32 bias is added AFTER dequantization. Encode the product to
            # select static T-T quantization and keep accumulator-bias semantics.
            scale = torch_npu.npu_trans_quant_param(
                (inputs["x1_scale"].float() * inputs["x2_scale"].float())
                .reshape(1)
                .contiguous()
            )
            gathered_x1_scale = None
        output = torch_npu.npu_quant_matmul(
            gathered,
            x2_eff,
            scale,
            offset=None,
            pertoken_scale=gathered_x1_scale,
            bias=inputs.get("bias"),
            output_dtype=out_dtype,
            group_sizes=list(attrs.get("group_sizes", [])),
        )
    if bool(attrs.get("gather_output", True)):
        return output, gathered
    return output


def mc2_cpu_golden(ctx, inputs, attrs):
    """AllGatherMatmulV2.get_cpu per-tensor FP8 reference."""
    x1 = inputs["x1"].float()
    gathered = torch.empty(
        (x1.shape[0] * ctx["world_size"], x1.shape[1]), dtype=torch.float32
    )
    ctx["dist"]._all_gather_base(gathered, x1)
    x2 = inputs["x2"].t() if attrs.get("is_trans_b", False) else inputs["x2"]
    out = torch.matmul(gathered, x2.float())
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
    # mc2_test scale_generate: truncate the product's low 13 mantissa bits.
    scale = inputs["x1_scale"].float() * inputs["x2_scale"].float()
    scale = (scale.contiguous().view(torch.int32) & -8192).view(torch.float32)
    out = out * scale.reshape(1)
    return (out, gathered) if attrs.get("gather_output", True) else out


def mc2_cpu_references(ctx, inputs, attrs):
    """Keep FP32 G and independently model the quantized cascade's final cast.

    Static per-tensor quant-matmul applies accumulator bias and the truncated
    scale before casting. All-gather itself only transports input values.
    """
    mathematical = mc2_cpu_golden(ctx, inputs, attrs)
    if attrs.get("mc2_accuracy_mode") != "three_way":
        return mathematical, None
    dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    if attrs.get("gather_output", True):
        output, gathered = mathematical
        return mathematical, (output.to(dtype).float(), gathered)
    return mathematical, mathematical.to(dtype).float()


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype = _torch_dtype(dtype_name)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(float(lo), float(hi))
            .to(dtype)
        )
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _torch_dtype(dtype_name: str):
    dtype = str(dtype_name).lower()
    if dtype in ("fp8_e4m3fn", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if dtype in ("fp8_e5m2", "float8_e5m2"):
        return torch.float8_e5m2
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype]
```

## 6. 算子调用示例

以下示例在 Ascend 950PR/950DT 上调用。假设通信域已初始化，`device` 为当前 rank 的 NPU 设备，`hcomm_info` 为该域名称，`world_size=2`；所有 rank 使用相同形状和调用顺序。

```python
import torch
import cann_bench

x1 = torch.randn(16, 256, device=device).to(torch.float8_e4m3fn)
x2 = torch.randn(256, 128, device=device).to(torch.float8_e4m3fn)
x1_scale = torch.ones(1, dtype=torch.float32, device=device)
x2_scale = torch.ones(1, dtype=torch.float32, device=device)
y, gather_out = cann_bench.all_gather_matmul_v2(
    x1, x2, hcomm_info, world_size,
    bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
    gather_output=True, is_trans_b=False, out_dtype="float16",
    x1_dtype="float8_e4m3fn", x2_dtype="float8_e4m3fn",
    x1_scale_dtype="float32", x2_scale_dtype="float32",
    block_size=0, group_sizes=[],
)
# y: [world_size * 16, 128]; gather_out: [world_size * 16, 256]
```
