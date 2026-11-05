# MatmulReduceScatterV2 算子 API 描述

## 1. 算子简介

MatmulReduceScatterV2 将矩阵乘、可选反量化与 ReduceScatter 集合通信融合执行。各 rank 先计算本地结果，在通信域内逐元素求和，再沿行维等分，向每个 rank 返回相应切片，适用于张量并行中的线性变换与输出分片。

计算语义对应 `aclnnMatmulReduceScatterV2`。以下 Python 接口描述 FP16/BF16 非量化和 INT8 per-token/per-channel 形式；不包含原生接口的 FP8、per-block、MX、输出量化及 amax 输出。

## 2. 算子定义

设通信域包含 `p = world_size` 个 rank，`r` 为当前 rank。各 rank 的输入 `X_r` 为 `[M, K]`，有效权重 `W_r` 为 `[K, N]`。`is_trans_b=True` 时 `W_r = x2_r.T`，否则 `W_r = x2_r`。

### 非量化

$$
L_r = X_r W_r + b_r
$$

### INT8 per-token/per-channel

$$
L_r = (\operatorname{INT32}(X_r)\operatorname{INT32}(W_r))
\odot s_r[:,\mathrm{None}] \odot t_r[\mathrm{None},:] + b_r
$$

`s_r` 为 `[M]` 的行尺度，`t_r` 为 `[N]` 的列尺度。偏置为浮点列偏置，按原生接口定义在反量化之后加入；与 INT32 累加偏置不同。省略偏置时取零。

### 归约与切分

$$
R = \sum_{s=0}^{p-1}\operatorname{Cast}_{D}(L_s), \qquad
Y_r = R[rM/p:(r+1)M/p,:]
$$

本地结果为 `[M, N]`，通信使用输出数据类型 `D`。ReduceScatter 先按元素求和，再按 rank 顺序沿第 0 维等分，当前 rank 的输出为 `[M/p, N]`。偏置在集合通信前加入，也会参与归约。

## 3. 接口规范

### 算子原型

```python
cann_bench.matmul_reduce_scatter_v2(
    Tensor x1, Tensor x2, str hcomm_info, int world_size,
    str reduce_op="sum", Tensor? bias=None, Tensor? x1_scale=None,
    Tensor? x2_scale=None, bool is_trans_b=False, str out_dtype="float16",
    str x1_dtype="int8", str x2_dtype="int8",
    str x1_scale_dtype="float32", str x2_scale_dtype="float32",
    int block_size=128, int[] group_sizes=[]
) -> Tensor y
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x1` | Tensor，`[M, K]` | 本地矩阵乘左输入。 |
| `x2` | Tensor，`[K, N]` 或 `[N, K]` | 本地权重，后一形状用于转置模式。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 通信域的 rank 数，须与实际大小一致。 |
| `reduce_op` | str | 归约操作，仅支持 `"sum"`。 |
| `bias` | 可选 Tensor，`[N]` | 非量化时与输入类型一致；INT8 形式可为 FP16/BF16/FP32，默认 `None`。 |
| `x1_scale` | 可选 FP32 Tensor，`[M]` | INT8 的 per-token 尺度；非量化时为 `None`。 |
| `x2_scale` | 可选 FP32 Tensor，`[N]` | INT8 的 per-channel 尺度；非量化时为 `None`。 |
| `is_trans_b` | bool | 是否在矩阵乘中转置 `x2`，默认 `False`。 |
| `out_dtype` | str | 输出类型，`float16` 或 `bfloat16`；非量化时须与 `x1` 一致。 |
| `x1_dtype` | str | `x1` 的逻辑类型，须与张量实际类型匹配。 |
| `x2_dtype` | str | `x2` 的逻辑类型，须与张量实际类型匹配。 |
| `x1_scale_dtype` | str | 非空行尺度的类型为 `float32`，非量化时忽略。 |
| `x2_scale_dtype` | str | 非空列尺度的类型为 `float32`，非量化时忽略。 |
| `block_size` | int | 接口保留的量化块属性，声明默认值为 128；A5 非量化形式须显式传 0，不表示通信切分大小。 |
| `group_sizes` | int[] | 量化分组属性，本文形式使用空列表 `[]`，不启用分块量化。 |

### 数据类型与输出

| 模式 | `x1` / `x2` | 反量化尺度 | `y` |
| --- | --- | --- | --- |
| 非量化 FP16 | 均为 `float16` | 均为 `None` | `float16` |
| 非量化 BF16 | 均为 `bfloat16` | 均为 `None` | `bfloat16` |
| INT8 per-token/per-channel | 均为 `int8` | FP32 行尺度和列尺度 | `float16` 或 `bfloat16` |

返回单个张量 `y`，形状为 `[M/p, N]`，不返回全量归约结果。

### 规则与约束

- `x1`、`x2` 均为二维 ND 张量；`x1` 不转置，`x2` 支持转置。有效权重的收缩维必须等于 `K`，且 `256 <= K < 65535`。
- `M` 必须能被 `p` 整除。各 rank 的本地结果形状、数据类型和集合通信调用顺序必须一致。
- INT8 per-token/per-channel 形式须同时提供两组尺度；尺度仅用于本地矩阵乘的反量化，不需要 AllGather。
- 原生 ACLNN 的 INT8 行尺度、列尺度可用 `[M, 1]`、`[1, N]` 表示；此 Python 接口用一维向量表达相同广播语义。
- Ascend 950PR/950DT 的本文接口形式为 FP16/BF16 非量化，须显式指定匹配的 `x1_dtype`、`x2_dtype`、`out_dtype` 和 `block_size=0`，两组尺度为 `None`。INT8 形式对应 Atlas A2/A3，不能使用默认 INT8 类型表示 A5 非量化输入。
- Ascend 950PR/950DT 支持 2、4、8、16、32、64 卡以及 CCU/AICPU 通信。ReduceScatter 总数据量 `M*N*sizeof(y.dtype)` 不得超过 `16*256 MB`，实际可用上限还受形状影响。
- Atlas A2/A3 的 INT8 形式支持 2、4、8 卡，不支持空张量，使用 AIV 通信且通信缓冲区至少 200 MB。

## 4. 精度要求

所有 rank 的输出均须满足形状、dtype 和可选输出契约。启用三方比较时，以相同输入的 CPU FP32 结果为参考：候选的大值域最大误差不超过设备级联的 10 倍，平均误差、误差总和及全量 RMSE 不超过级联的 2 倍。

另允许有限值逐元素满足 `abs(candidate-golden) <= atol + rtol*abs(golden)`，避免零级联误差退化为逐位相等要求。有效比较类型 FP16/BF16/FP32 的 `rtol` 分别为 `2^-10`、`2^-7`、`2^-23`，`atol` 分别为 `1e-5`、`1e-5`、`1e-9`。显式 FP32 比较和 FP8 辅助输出使用 FP32 容差，不能按输入 FP8 类型放宽。

FP16/BF16 的小值域分界为 `1e-3`，FP32 为 `1e-6`。小值元素不少于 100000 个时，候选的小值超差数不得超过级联的 2 倍；计数边界及绝对误差阈值保持不变。结构、NaN/正负无穷掩码及有限误差统计检查始终有效，容差分支不能绕过这些检查。

级联参考先接受按真实输出类型的独立固定精度校验；参考失效时评分不可用，不归为候选编译错误。未启用三方模式的历史用例保持原相对误差流程。完整判定与比较类型规则见[三方精度指南](../../../../docs/guide/mc2_accuracy.md)。

两卡非量化级联的自检参考由 CPU 独立模拟 Matmul 输出、可选 bias 加法及两卡求和各阶段的 FP16/BF16 舍入，避免中间舍入在跨卡数值抵消时被误判。该参考仅用于级联健康检查；三方误差仍相对于全程 FP32 的数学 golden 计算，倍率、候选容差及评分公式不变。其他卡数不假定相同的归约顺序，继续使用原自检方式。

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


from typing import Any, Dict, Optional

import torch


def matmul_reduce_scatter_v2(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
    out_dtype: str = "float16",
) -> torch.Tensor:
    """Single-process reference for metadata and smoke checks."""
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    if x1_scale is not None or x2_scale is not None:
        out = _apply_dequant_scale(out, x1_scale, x2_scale)
    out = out * int(world_size)
    chunk = out.shape[0] // int(world_size)
    return out[:chunk].to(_torch_dtype(out_dtype))


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
    if not _is_quant_case(attrs):
        # A5 TP slices must differ across ranks; x1/x2 share the rank seed.
        seed += rank

    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0], seed, device)
    x2_seed = seed if weight_same else seed + rank * 2
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1], x2_seed, device)

    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2], rank * 3, device)

    if not _is_quant_case(attrs):
        return {"x1": x1, "x2": x2, "bias": bias, "x1_scale": None, "x2_scale": None}

    x1_scale_index = 3 if bool(attrs.get("is_bias", False)) else 2
    x2_scale_index = x1_scale_index + 1
    x1_scale = _make_tensor(
        shapes[x1_scale_index],
        dtypes[x1_scale_index],
        ranges[x1_scale_index],
        seed * 11,
        device,
    )
    x2_scale = _make_tensor(
        shapes[x2_scale_index],
        dtypes[x2_scale_index],
        ranges[x2_scale_index],
        seed * 13,
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
        reduce_op=str(attrs.get("reduce_op", "sum")),
        bias=inputs.get("bias"),
        x1_scale=inputs.get("x1_scale"),
        x2_scale=inputs.get("x2_scale"),
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
    """Distributed CPU-semantic golden following mc2_test's get_cpu path.

    Compute matmul and dequant in float32 for precision, then cast to the
    operator's output dtype before all_reduce so the communication precision
    matches the fused MC2 kernel (which communicates in fp16/bf16).
    """
    dist = ctx["dist"]
    device = ctx["device"]
    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    # Compute matmul in float32 on CPU to avoid int8→float32 Cast kernel issues
    if attrs.get("mc2_accuracy_mode") == "three_way":
        if _is_quant_case(attrs):
            raise ValueError(
                "This three-way task currently covers non-quantized A5 cases only"
            )
        # Device cascade must actually execute matmul on NPU, not CPU.
        local = torch.matmul(inputs["x1"], x2_eff)
        if inputs.get("bias") is not None:
            local = local + inputs["bias"]
        chunk = local.shape[0] // int(ctx["world_size"])
        result = torch.empty((chunk, local.shape[1]), dtype=local.dtype, device=device)
        dist._reduce_scatter_base(result, local.contiguous(), op=dist.ReduceOp.SUM)
        return result

    # on machines where the CANN CastAiCore JSON config is not installed.
    x1_cpu = inputs["x1"].cpu().float()
    x2_cpu = x2_eff.cpu().float()
    bias_cpu = (
        inputs.get("bias").cpu().float() if inputs.get("bias") is not None else None
    )
    mm_out = torch.matmul(x1_cpu, x2_cpu)
    if bias_cpu is not None:
        mm_out = mm_out + bias_cpu
    if _is_quant_case(attrs):
        x1_scale_cpu = (
            inputs.get("x1_scale").cpu() if inputs.get("x1_scale") is not None else None
        )
        x2_scale_cpu = (
            inputs.get("x2_scale").cpu() if inputs.get("x2_scale") is not None else None
        )
        mm_out = _apply_dequant_scale(mm_out, x1_scale_cpu, x2_scale_cpu)
    # Cast to output dtype before all_reduce to match candidate's communication precision
    out_dtype = _torch_dtype(str(attrs.get("out_dtype", "float16")))
    mm_out = mm_out.to(out_dtype).to(device)
    dist.all_reduce(mm_out, op=dist.ReduceOp.SUM)
    chunk = mm_out.shape[0] // int(ctx["world_size"])
    return mm_out.narrow(0, int(ctx["rank"]) * chunk, chunk)


def mc2_cpu_golden(ctx, inputs, attrs):
    return _cpu_references(ctx, inputs, attrs, staged=False)[0]


def mc2_cpu_references(ctx, inputs, attrs):
    """Return mathematical G and an independent, stage-rounded health reference.

    Only two ranks have a single, unambiguous low-precision reduction step.
    Other world sizes retain the generic health check; do not assume a reduction
    tree/order for them. Neither result depends on NPU or candidate outputs.
    """
    return _cpu_references(ctx, inputs, attrs, staged=int(ctx["world_size"]) == 2)


def _cpu_references(ctx, inputs, attrs, staged):
    if _is_quant_case(attrs):
        raise ValueError(
            "CPU three-way reference currently covers A5 non-quantized cases"
        )
    if attrs.get("reduce_op", "sum") != "sum":
        raise ValueError("CPU three-way reference only supports sum")
    x2 = inputs["x2"].t() if attrs.get("is_trans_b", False) else inputs["x2"]
    out = torch.matmul(inputs["x1"].float(), x2.float())
    # Share the expensive FP32 matmul, not its rounded values, with mathematical G.
    dtype = inputs["x1"].dtype
    rounded = out.to(dtype).float() if staged else None
    if inputs.get("bias") is not None:
        out = out + inputs["bias"].float()
        if staged:
            rounded = (rounded + inputs["bias"].float()).to(dtype).float()
    ctx["dist"].all_reduce(out, op=ctx["dist"].ReduceOp.SUM)
    if staged:
        # Communicate FP32 over Gloo, then emulate the one FP16/BF16 sum rounding.
        ctx["dist"].all_reduce(rounded, op=ctx["dist"].ReduceOp.SUM)
        rounded = rounded.to(dtype).float()
    chunk = out.shape[0] // int(ctx["world_size"])
    start = int(ctx["rank"]) * chunk
    return (
        out.narrow(0, start, chunk),
        rounded.narrow(0, start, chunk) if staged else None,
    )


def _is_quant_case(attrs: Dict[str, Any]) -> bool:
    x1_dtype = str(attrs.get("x1_dtype", "int8")).lower()
    x2_dtype = str(attrs.get("x2_dtype", "int8")).lower()
    non_quant = {"fp16", "float16", "bf16", "bfloat16"}
    return x1_dtype not in non_quant or x2_dtype not in non_quant


def _apply_dequant_scale(
    out: torch.Tensor,
    x1_scale: Optional[torch.Tensor],
    x2_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply MC2 INT8 per-token/per-channel dequant scale.

    mc2_test truncates the float32 scale product with ``0xffffe000`` before it
    multiplies the matmul result.  Keep that behavior here, while leaving final
    dtype rounding to cann-bench's normal compare path.
    """
    if x1_scale is None and x2_scale is None:
        return out

    rows, cols = int(out.shape[-2]), int(out.shape[-1])
    device = out.device
    x1 = _scale_tensor_or_ones(x1_scale, rows, device)
    x2 = _scale_tensor_or_ones(x2_scale, cols, device)

    if x1.numel() == rows and x2.numel() == cols:
        result = out.clone()
        row_scale = x1.float().reshape(rows, 1)
        col_scale = x2.float().reshape(1, cols)
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            scale = _truncate_hw_scale(row_scale[start:end] * col_scale)
            result[start:end] = result[start:end] * scale
        return result

    scale = _broadcast_dequant_scale(x1, x2, rows, cols)
    return out * _truncate_hw_scale(scale)


def _scale_tensor_or_ones(
    scale: Optional[torch.Tensor], size: int, device: torch.device
) -> torch.Tensor:
    if scale is None:
        return torch.ones(size, dtype=torch.float32, device=device)
    return scale.float().to(device)


def _broadcast_dequant_scale(
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    x1 = x1_scale.float()
    x2 = x2_scale.float()
    if x1.numel() == 1 and x2.numel() == 1:
        return x1.reshape(1) * x2.reshape(1)
    if x1.numel() == rows and x2.numel() == 1:
        return x1.reshape(rows, 1) * x2.reshape(1)
    if x1.numel() == 1 and x2.numel() == cols:
        return x1.reshape(1) * x2.reshape(1, cols)
    if list(x1.shape) == [rows, 1] and x2.numel() == cols:
        return x1 * x2.reshape(1, cols)
    if x1.numel() == rows and list(x2.shape) == [1, cols]:
        return x1.reshape(rows, 1) * x2
    if list(x1.shape) == [rows, cols] and x2.numel() == 1:
        return x1 * x2.reshape(1)
    raise ValueError(
        "Unsupported dequant scale shapes: "
        f"x1_scale={tuple(x1_scale.shape)}, x2_scale={tuple(x2_scale.shape)}, output=({rows}, {cols})"
    )


def _truncate_hw_scale(scale: torch.Tensor) -> torch.Tensor:
    bits = scale.float().view(torch.int32)
    bits &= torch.tensor(
        -8192, dtype=torch.int32, device=scale.device
    )  # 0xFFFFE000 as signed int32
    return bits.view(torch.float32)


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

x1 = torch.randn(16, 256, dtype=torch.float16, device=device)
x2 = torch.randn(256, 128, dtype=torch.float16, device=device)
y = cann_bench.matmul_reduce_scatter_v2(
    x1, x2, hcomm_info, world_size,
    reduce_op="sum", bias=None, x1_scale=None, x2_scale=None,
    is_trans_b=False, out_dtype="float16",
    x1_dtype="float16", x2_dtype="float16",
    x1_scale_dtype="float32", x2_scale_dtype="float32",
    block_size=0, group_sizes=[],
)
# y: [16 // world_size, 128]
```
