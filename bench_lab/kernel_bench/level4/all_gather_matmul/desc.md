# AllGatherMatmul 算子 API 描述

## 1. 算子简介

AllGatherMatmul 将 AllGather 集合通信与矩阵乘融合执行，适用于张量并行中先聚合各 rank 的输入、再使用本地权重进行线性变换的场景。支持 FP16/BF16 输入、可选偏置、权重转置及聚合结果输出。

计算语义对应 `aclnnAllGatherMatmul`，以下描述 Python 接口。

## 2. 算子定义

设通信域包含 `p = world_size` 个 rank，编号为 `0` 到 `p-1`。每个 rank 的输入 `X_r` 为 `[M, K]`，有效权重 `W_r` 为 `[K, N]`。`is_trans_b=True` 时，`W_r = x2_r.T`，否则 `W_r = x2_r`。

$$
G = \operatorname{Concat}_{\mathrm{dim}=0}(X_0, X_1, \ldots, X_{p-1})
$$

$$
Y_r = G W_r + b_r
$$

`G` 按通信域 rank 顺序沿行维拼接，形状为 `[p*M, K]`；`Y_r` 为 `[p*M, N]`。省略 `bias` 时不执行加偏置。偏置按列广播，在本地矩阵乘后相加，不参与 AllGather。

各 rank 得到相同的 `G`；权重和偏置可以不同，因此不要求各 rank 的 `Y_r` 相同。

## 3. 接口规范

### 算子原型

```python
cann_bench.all_gather_matmul(
    Tensor x1, Tensor x2, str hcomm_info, int world_size,
    Tensor? bias=None, bool gather_output=True, bool is_trans_b=False
) -> Tensor | (Tensor y, Tensor gather_out)
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `x1` | Tensor，`[M, K]` | 当前 rank 的输入，FP16 或 BF16。 |
| `x2` | Tensor，`[K, N]` 或 `[N, K]` | 本地权重，类型与 `x1` 相同；后一形状用于转置模式。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 通信域的 rank 数，必须与实际通信域大小一致。 |
| `bias` | 可选 Tensor，`[N]` | 列偏置，类型与 `x1` 相同，默认 `None`。 |
| `gather_output` | bool | 是否同时返回 AllGather 的聚合结果，默认 `True`。 |
| `is_trans_b` | bool | 是否在矩阵乘中转置 `x2`，默认 `False`；不转置 `x1`。 |

### 输出

| 输出 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| `y` | `[p*M, N]` | 与 `x1` 相同 | 使用当前 rank 权重计算的矩阵乘结果。 |
| `gather_out` | `[p*M, K]` | 与 `x1` 相同 | 按 rank 顺序聚合的原始输入，不包含矩阵乘或偏置。 |

`gather_output=True` 时返回 `(y, gather_out)`；为 `False` 时仅返回 `y`。

### 规则与约束

- `x1`、`x2` 均为二维 ND 张量；`x1` 不支持转置，`x2` 支持两轴转置。
- 各 rank 的 `x1` 形状与数据类型必须相同；所有 rank 必须以相同顺序参与集合通信。
- 有效权重的收缩维必须等于 `K`，且 `256 <= K < 65535`。`M` 是每个 rank 的行数，不要求被 `p` 整除。
- Ascend 950PR/950DT 支持非零偏置；Atlas A2/A3 的该原生接口暂不支持非零偏置。
- Ascend 950PR/950DT 支持 `p` 为 2、4、8、16、32、64，仅支持 HCCS all mesh 组网。AllGather 总数据量 `p*M*K*sizeof(x1.dtype)` 不得超过 `63*256 MB`，实际可用上限还受形状影响。
- Atlas A2 支持 2、4、8 卡，Atlas A3 支持 2、4、8、16、32 卡；不同产品的通信拓扑约束不能互相替代。

## 4. 精度要求

所有 rank 的输出均须满足形状、dtype 和可选输出契约。启用三方比较时，以相同输入的 CPU FP32 结果为参考：候选的大值域最大误差不超过设备级联的 10 倍，平均误差、误差总和及全量 RMSE 不超过级联的 2 倍。

另允许有限值逐元素满足 `abs(candidate-golden) <= atol + rtol*abs(golden)`，避免零级联误差退化为逐位相等要求。有效比较类型 FP16/BF16/FP32 的 `rtol` 分别为 `2^-10`、`2^-7`、`2^-23`，`atol` 分别为 `1e-5`、`1e-5`、`1e-9`。显式 FP32 比较和 FP8 辅助输出使用 FP32 容差，不能按输入 FP8 类型放宽。

FP16/BF16 的小值域分界为 `1e-3`，FP32 为 `1e-6`。小值元素不少于 100000 个时，候选的小值超差数不得超过级联的 2 倍；计数边界及绝对误差阈值保持不变。结构、NaN/正负无穷掩码及有限误差统计检查始终有效，容差分支不能绕过这些检查。

两卡 FP16/BF16 级联参考先接受独立 CPU 分阶段舍入模型校验：矩阵乘输出先转回输入类型，加 bias 后再舍入。此模型只用于参考健康检查，候选误差仍相对于未舍入的 CPU FP32 数学参考计算，不放宽全局阈值。参考失效时评分不可用，不归为候选编译错误。未启用三方模式的历史用例保持原相对误差流程。完整判定与比较类型规则见[三方精度指南](../../../../docs/guide/mc2_accuracy.md)。

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


import torch
from typing import Optional, Union, Tuple


def all_gather_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    hcomm_info: str = "",
    world_size: int = 1,
    gather_output: bool = True,
    is_trans_b: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Single-process CPU reference for the MC2 all-gather matmul task.

    The real benchmark path for cases with ``mc2_distributed: true`` uses
    ``MC2DistributedEvaluator`` and HCCL. This function keeps the task usable
    for metadata inspection and golden-only smoke checks.
    """
    del hcomm_info
    x2_eff = x2.t() if is_trans_b else x2
    gathered = torch.cat([x1] * int(world_size), dim=0)
    out = torch.matmul(gathered.float(), x2_eff.float())
    if bias is not None:
        out = out + bias.float()
    out = out.to(x1.dtype)
    if gather_output:
        return out, gathered
    return out


def mc2_distributed_golden(ctx, inputs, attrs):
    """Distributed golden hook used by MC2DistributedEvaluator."""
    dist = ctx["dist"]
    x1 = inputs["x1"]
    x2 = inputs["x2"]
    bias = inputs.get("bias")
    world_size = int(ctx["world_size"])
    gather_shape = [x1.shape[0] * world_size] + list(x1.shape[1:])
    gathered = torch.empty(gather_shape, dtype=x1.dtype, device=x1.device)
    dist._all_gather_base(gathered, x1)
    x2_eff = x2.t() if bool(attrs.get("is_trans_b", False)) else x2
    out = torch.matmul(gathered, x2_eff)
    if bias is not None:
        out = out + bias
    if bool(attrs.get("gather_output", False)):
        return out, gathered
    return out


def mc2_cpu_golden(ctx, inputs, attrs):
    """CPU/Gloo FP32 reference, without intermediate output rounding."""
    fp32 = {
        key: value.float() if value is not None else None
        for key, value in inputs.items()
    }
    return mc2_distributed_golden(ctx, fp32, attrs)


def mc2_cpu_references(ctx, inputs, attrs):
    """Return FP32 math and an independent model of the native rounding stages.

    AllGather copies the input exactly. The native cascade rounds the matmul
    output to the input dtype, then rounds again after optional bias addition.
    Only reference health uses this second result; candidate error metrics
    retain the unrounded FP32 mathematical reference.
    """
    dtype = inputs["x1"].dtype
    if int(ctx["world_size"]) != 2 or dtype not in (torch.float16, torch.bfloat16):
        return mc2_cpu_golden(ctx, inputs, attrs), None

    x1 = inputs["x1"].float()
    gathered = torch.empty(
        (x1.shape[0] * int(ctx["world_size"]), x1.shape[1]),
        dtype=torch.float32,
        device=x1.device,
    )
    ctx["dist"]._all_gather_base(gathered, x1)
    weight = inputs["x2"].float()
    if bool(attrs.get("is_trans_b", False)):
        weight = weight.t()
    mathematical = torch.matmul(gathered, weight)
    staged = mathematical.to(dtype).float()
    bias = inputs.get("bias")
    if bias is not None:
        mathematical = mathematical + bias.float()
        staged = (staged + bias.float()).to(dtype).float()
    if bool(attrs.get("gather_output", False)):
        return (mathematical, gathered), (staged, gathered)
    return mathematical, staged
```

## 6. 算子调用示例

以下示例在 Ascend 950PR/950DT 上调用。假设通信域已初始化，`device` 为当前 rank 的 NPU 设备，`hcomm_info` 为该域名称，`world_size=2`；所有 rank 使用相同形状和调用顺序。

```python
import torch
import cann_bench

x1 = torch.randn(16, 256, dtype=torch.float16, device=device)
x2 = torch.randn(256, 128, dtype=torch.float16, device=device)
y, gather_out = cann_bench.all_gather_matmul(
    x1, x2, hcomm_info, world_size,
    bias=None, gather_output=True, is_trans_b=False,
)
# y: [world_size * 16, 128]; gather_out: [world_size * 16, 256]
```
