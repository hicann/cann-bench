# MatmulAllReduce 算子 API 描述

## 1. 算子简介

MatmulAllReduce 将本地矩阵乘与 AllReduce 集合通信融合执行，用于张量并行中计算并归约各 rank 的部分结果。支持 FP16/BF16 非量化矩阵乘，以及 INT8 per-token/per-channel 量化矩阵乘；输出在通信域内求和后分发到所有 rank。

非量化语义对应 `aclnnMatmulAllReduce`，INT8 语义对应 `aclnnQuantMatmulAllReduceV4`。以下描述 Python 接口，不包含低比特通信量化。

## 2. 算子定义

设通信域包含 `p = world_size` 个 rank，编号为 `0` 到 `p-1`。`X_r` 为当前 rank 的左输入，最后一维为 `K`；有效权重 `W_r` 为 `[K, N]`。`is_trans_b=True` 时 `W_r = x2_r.T`，否则 `W_r = x2_r`。

### 非量化

$$
L_r = X_r W_r + b_r, \qquad Y_r = \sum_{s=0}^{p-1} L_s
$$

偏置在各 rank 的本地矩阵乘后相加，再参与 AllReduce；省略偏置时取零。各 rank 都提供同一非零偏置时，归约中会累加 `p` 份偏置，不能等价为只在归约后加一次。

### INT8 per-token/per-channel

$$
Q_r = \operatorname{INT32}(X_r)\operatorname{INT32}(W_r) + b_r
$$

$$
L_r = Q_r \odot s_r[:,\mathrm{None}] \odot t_r[\mathrm{None},:] + x3_r,
\qquad Y_r^{\mathrm{math}} = \sum_{s=0}^{p-1}L_s
$$

这里将左输入除最后一维外的维度展平为行维：`s_r` 为 per-token 尺度，`t_r` 为 per-channel 尺度。INT32 偏置在反量化之前加入，可选残差 `x3` 在反量化之后、集合通信之前加入；省略的偏置或残差按零处理。`D` 由 `out_dtype` 指定，通信不额外量化为 INT8。

上述数学式对应 FP32 参考 G。实际低精度级联先将反量化矩阵乘结果转换到 D，
再做可选 x3 加法并转换到 D，最后以 D 进行 AllReduce SUM。两卡独立 CPU 阶段参考 H
模拟这些舍入边界（反量化按 channel 再 token 缩放），不能用仅在最终输出转换一次来替代。

二维输入 `[M, K]` 的输出为 `[M, N]`；三维输入 `[B, S, K]` 的输出为 `[B, S, N]`。AllReduce 不改变输出形状，各 rank 获得相同的求和结果。

## 3. 接口规范

### 算子原型

```python
cann_bench.matmul_all_reduce(
    Tensor x1, Tensor x2, str hcomm_info, int world_size,
    str reduce_op="sum", Tensor? bias=None, bool is_trans_b=False,
    Tensor? dequant_scale=None, Tensor? pertoken_scale=None,
    Tensor? x3=None, str out_dtype="float16"
) -> Tensor y
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x1` | Tensor，`[M, K]` 或 `[B, S, K]` | 本地矩阵乘左输入。 |
| `x2` | Tensor，`[K, N]` 或 `[N, K]` | 本地权重，后一形状用于转置模式。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 通信域的 rank 数，必须与实际通信域大小一致。 |
| `reduce_op` | str | 归约操作，仅支持 `"sum"`。 |
| `bias` | 可选 Tensor，`[N]` | 非量化时与 `x1` 类型一致；INT8 量化时为 INT32。默认 `None`。 |
| `is_trans_b` | bool | 是否在矩阵乘中转置 `x2`，默认 `False`。 |
| `dequant_scale` | 可选 Tensor，`[N]` | INT8 的 per-channel 尺度，为 FP32/BF16；非量化时为 `None`。 |
| `pertoken_scale` | 可选 FP32 Tensor，`[M]` 或 `[B*S]` | INT8 的 per-token 尺度；非量化时为 `None`。 |
| `x3` | 可选 Tensor，与 `y` 同形状 | INT8 反量化后的本地残差，与输出数据类型一致；非量化接口不使用。 |
| `out_dtype` | str | INT8 输出类型，`float16` 或 `bfloat16`，默认 `float16`；非量化输出与 `x1` 类型一致。 |

### 数据类型与输出

| 模式 | `x1` / `x2` | `bias` | `y` |
| --- | --- | --- | --- |
| 非量化 FP16 | 均为 `float16` | `float16` 或 `None` | `float16` |
| 非量化 BF16 | 均为 `bfloat16` | `bfloat16` 或 `None` | `bfloat16` |
| INT8 per-token/per-channel | 均为 `int8` | `int32` 或 `None` | `float16` 或 `bfloat16` |

输出 `y` 的最后一维为 `N`，其余维度与 `x1` 一致。

### 规则与约束

- 输入采用 ND 格式。`x1` 为二维或三维，不转置；`x2` 为二维，支持转置，有效收缩维必须等于 `K`。
- 各 rank 的本地矩阵乘结果必须具有相同形状与数据类型；所有 rank 必须以相同顺序参加 AllReduce。本地输入、权重、偏置及残差的数值不要求相同。
- INT8 per-token/per-channel 形式须同时提供 `pertoken_scale` 和 `dequant_scale`；前者按行广播，后者按列广播。`x3` 与输出严格同形状，不作广播。
- 非量化形式不使用量化尺度或残差参数。`M`、`B*S`、`K`、`N` 不得超过 `INT32_MAX`。
- INT8 形式不支持空张量，`x1` 和 `x2` 的物理最后一维大小不得超过 65535。
- 上述原生接口在 Ascend 950PR/950DT 支持 1、2、4、8、16、32、64 卡，在 Atlas A2 支持 1、2、4、8 卡，要求 HCCS all mesh 组网。不同产品的其他数据类型或量化模式不由此 Python 接口定义。

## 4. 精度要求

所有 rank 的输出均须满足形状、dtype 和可选输出契约。启用三方比较时，以相同输入的 CPU FP32 结果为参考：候选的大值域最大误差不超过设备级联的 10 倍，平均误差、误差总和及全量 RMSE 不超过级联的 2 倍。

另允许有限值逐元素满足 `abs(candidate-golden) <= atol + rtol*abs(golden)`，避免零级联误差退化为逐位相等要求。有效比较类型 FP16/BF16/FP32 的 `rtol` 分别为 `2^-10`、`2^-7`、`2^-23`，`atol` 分别为 `1e-5`、`1e-5`、`1e-9`。显式 FP32 比较和 FP8 辅助输出使用 FP32 容差，不能按输入 FP8 类型放宽。

FP16/BF16 的小值域分界为 `1e-3`，FP32 为 `1e-6`。小值元素不少于 100000 个时，候选的小值超差数不得超过级联的 2 倍；计数边界及绝对误差阈值保持不变。结构、NaN/正负无穷掩码及有限误差统计检查始终有效，容差分支不能绕过这些检查。

级联参考先接受按真实输出类型的独立固定精度校验；参考失效时评分不可用，不归为候选编译错误。未启用三方模式的历史用例保持原相对误差流程。完整判定与比较类型规则见[三方精度指南](../../../../docs/guide/mc2_accuracy.md)。

两卡非量化与 INT8→FP16/BF16 路径使用独立 CPU 阶段参考 H 检查级联参考健康；
H 不读取候选或 NPU 输出，不替换数学 G，不改变候选误差倍率、阈值或计分规则。
其他卡数或未建模 dtype 保留原有参考检查。参考门禁未通过时尚未执行候选，不能记为候选精度失败。

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


from typing import Optional

import torch


def _dtype_from_name(dtype_name: str):
    dtype = str(dtype_name).lower()
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp32", "float32"):
        return torch.float32
    if dtype in ("int32",):
        return torch.int32
    if dtype == "int8":
        return torch.int8
    if dtype in ("int64",):
        return torch.int64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    if shape is None:
        return None
    torch.manual_seed(int(seed))
    dtype = _dtype_from_name(dtype_name)
    if value_range is None:
        value_range = [0, 1]
    lo, hi = value_range
    if dtype.is_floating_point:
        tensor = (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(float(lo), float(hi))
            .to(dtype)
        )
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def mc2_make_rank_inputs(ctx, case_payload):
    """Preserve legacy seeds; distinguish A5 rank inputs for communication checks."""
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload.get("value_ranges") or []
    attrs = case_payload["attrs"]
    rank = int(ctx["rank"])
    device = ctx["device"]
    seed = int(attrs.get("seed", 1))
    if attrs.get("quantized", False):
        seed += rank

    x1_seed = seed
    if (
        not attrs.get("quantized", False)
        and attrs.get("mc2_accuracy_mode") == "three_way"
    ):
        x1_seed += rank * 1000003
    x1 = _make_tensor(
        shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
    )
    x2 = _make_tensor(
        shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, seed, device
    )
    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            seed * 7,
            device,
        )
    inputs = {"x1": x1, "x2": x2, "bias": bias}
    if attrs.get("quantized", False):
        for index, name in ((3, "dequant_scale"), (4, "pertoken_scale"), (5, "x3")):
            inputs[name] = _make_tensor(
                shapes[index], dtypes[index], ranges[index], seed * (index + 3), device
            )
    return inputs


def matmul_all_reduce(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    reduce_op: str = "sum",
    is_trans_b: bool = False,
) -> torch.Tensor:
    """Single-process CPU reference for the MC2 matmul all-reduce task."""
    del hcomm_info
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    x2_eff = x2.t() if is_trans_b else x2
    out = torch.matmul(x1.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    # Single-process fallback assumes all ranks have identical seeded inputs.
    return out * int(world_size)


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed HCCL baseline for MatmulAllReduce.

    Use torch.addmm for fused matmul+bias to match the MC2 kernel's
    cube-unit behavior (bias added in fp32 accumulator before truncation).
    """
    x2 = inputs["x2"]
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    bias = inputs.get("bias")
    if attrs.get("quantized", False):
        shape = inputs["x1"].shape
        out = (
            ctx["torch_npu"]
            .npu_quant_matmul(
                inputs["x1"].reshape(-1, shape[-1]),
                x2_eff,
                inputs["dequant_scale"].reshape(-1),
                pertoken_scale=inputs["pertoken_scale"].reshape(-1),
                bias=bias,
                output_dtype=_dtype_from_name(attrs["out_dtype"]),
            )
            .reshape(*shape[:-1], x2_eff.shape[-1])
        )
        if inputs.get("x3") is not None:
            out = out + inputs["x3"]
        _all_reduce_sum(ctx, out, attrs)
        return out
    if bias is not None:
        out = torch.addmm(bias, inputs["x1"], x2_eff)
    else:
        out = torch.matmul(inputs["x1"], x2_eff)
    _all_reduce_sum(ctx, out, attrs)
    return out


def _all_reduce_sum(ctx, tensor: torch.Tensor, attrs):
    reduce_op = str(attrs.get("reduce_op", "sum")).lower()
    if reduce_op != "sum":
        raise ValueError(f"Unsupported reduce_op: {reduce_op}")
    ctx["dist"].all_reduce(tensor, op=ctx["dist"].ReduceOp.SUM)


def _quant_matmul_int32(inputs, attrs):
    """Accumulate integer bias before dequantization, shared by G and H."""
    x2 = inputs["x2"].t() if attrs.get("is_trans_b", False) else inputs["x2"]
    out = torch.matmul(inputs["x1"].to(torch.int32), x2.to(torch.int32))
    if inputs.get("bias") is not None:
        out = out + inputs["bias"]
    return out


def _quant_math_local(accumulator, inputs):
    """Preserve the mathematical golden's FP32 token-then-channel order."""
    out = accumulator
    shape = out.shape
    out = (
        out.float().reshape(-1, shape[-1])
        * inputs["pertoken_scale"].float().reshape(-1, 1)
    ).reshape(shape)
    out = out * inputs["dequant_scale"].float().reshape(-1)
    if inputs.get("x3") is not None:
        out = out + inputs["x3"].float()
    return out


def mc2_cpu_golden(ctx, inputs, attrs):
    """QuantMatmulAllReduceV4.get_cpu INT8 path: INT32 GEMM then FP32 scales."""
    if not attrs.get("quantized", False):
        fp32 = {
            key: value.float() if value is not None else None
            for key, value in inputs.items()
        }
        return mc2_distributed_golden(ctx, fp32, attrs)
    out = _quant_math_local(_quant_matmul_int32(inputs, attrs), inputs)
    _all_reduce_sum(ctx, out, attrs)
    return out


def mc2_cpu_references(ctx, inputs, attrs):
    """Return FP32 math and an independent two-rank, stage-rounded cascade.

    Native addmm fuses bias before rounding. INT8 quant matmul instead rounds
    after dequantization and again after optional x3 addition. HCCL SUM rounds
    the two low-precision local outputs once more. H is computed only from
    CPU inputs for reference health; candidate scoring still uses unchanged G.
    More than two ranks require a separately verified reduction-tree model.
    """
    if int(ctx["world_size"]) != 2:
        return mc2_cpu_golden(ctx, inputs, attrs), None

    if attrs.get("quantized", False):
        dtype = _dtype_from_name(attrs["out_dtype"])
        if dtype not in (torch.float16, torch.bfloat16) or any(
            inputs[name].dtype != torch.int8 for name in ("x1", "x2")
        ):
            return mc2_cpu_golden(ctx, inputs, attrs), None
        accumulator = _quant_matmul_int32(inputs, attrs)
        local = _quant_math_local(accumulator, inputs)
        # Quant matmul scales channels before tokens and materializes D output.
        shape = accumulator.shape
        rounded = accumulator.float() * inputs["dequant_scale"].float().reshape(-1)
        rounded = (
            rounded.reshape(-1, shape[-1])
            * inputs["pertoken_scale"].float().reshape(-1, 1)
        ).reshape(shape)
        rounded = rounded.to(dtype).float()
        if inputs.get("x3") is not None:
            rounded = (rounded + inputs["x3"].float()).to(dtype).float()
        _all_reduce_sum(ctx, local, attrs)
        _all_reduce_sum(ctx, rounded, attrs)
        return local, rounded.to(dtype).float()

    x2_eff = inputs["x2"].t() if bool(attrs.get("is_trans_b", False)) else inputs["x2"]
    local = torch.matmul(inputs["x1"].float(), x2_eff.float())
    bias = inputs.get("bias")
    if bias is not None:
        local = local + bias.float()
    dtype = inputs["x1"].dtype
    rounded = local.to(dtype).float()
    _all_reduce_sum(ctx, local, attrs)
    _all_reduce_sum(ctx, rounded, attrs)
    return local, rounded.to(dtype).float()


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    """Call task candidate with the MatmulAllReduce argument contract."""
    if attrs.get("quantized", False):
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
            dequant_scale=inputs["dequant_scale"],
            pertoken_scale=inputs["pertoken_scale"],
            x3=inputs.get("x3"),
            out_dtype=attrs["out_dtype"],
        )
    try:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
        )
    except TypeError:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            attrs.get("reduce_op", "sum"),
            inputs.get("bias"),
            bool(attrs.get("is_trans_b", False)),
        )
```

## 6. 算子调用示例

以下示例在 Ascend 950PR/950DT 上调用。假设通信域已初始化，`device` 为当前 rank 的 NPU 设备，`hcomm_info` 为该域名称，`world_size=2`；所有 rank 使用相同形状和调用顺序。

```python
import torch
import cann_bench

x1 = torch.randint(-8, 8, (16, 256), dtype=torch.int8).to(device)
x2 = torch.randint(-8, 8, (256, 128), dtype=torch.int8).to(device)
bias = torch.zeros(128, dtype=torch.int32, device=device)
pertoken_scale = torch.full((16,), 0.01, dtype=torch.float32, device=device)
dequant_scale = torch.full((128,), 0.01, dtype=torch.float32, device=device)
x3 = torch.zeros(16, 128, dtype=torch.float16, device=device)
y = cann_bench.matmul_all_reduce(
    x1, x2, hcomm_info, world_size,
    reduce_op="sum", bias=bias, is_trans_b=False,
    dequant_scale=dequant_scale, pertoken_scale=pertoken_scale,
    x3=x3, out_dtype="float16",
)
# y: [16, 128]，所有 rank 获得相同的归约结果。
```
