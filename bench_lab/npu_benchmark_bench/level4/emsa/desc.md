# EMSA 算子 API 描述

## 1. 算子简介

EMSA（Efficient Multi-Head Self-Attention）算子。支持两条可选的高效路径：
1. **空间缩减**（`ratio > 1`）：对 query 的二维空间表示做分组卷积 + LayerNorm，得到 key/value 输入，降低注意力计算量；
2. **注意力图变换**（`apply_transform = true`）：在 softmax 前对注意力图做 1x1 卷积，并在 softmax 后做 instance norm。

**主要应用场景**：
- 视觉 Transformer / 检测网络中的高效多头自注意力
- 需要空间下采样或多尺度注意力的视觉任务

**算子特征**：
- 难度等级：L4（Attention）
- 输入 `queries`：`[B, Lq, d_model]`，`keys` / `values`：`[B, Lk, d_model]`
- 全部线性/卷积权重与偏置显式作为输入传入
- `ratio <= 1` 时 `sr_*` 为空 tensor；`apply_transform = false` 时 `transform_*` 为空 tensor
- 输出：`[B, Lq, n_heads*d_v]`

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
q &= W_q\, queries^\top + b_q \\
kv_{in} &= \begin{cases}
\mathrm{LayerNorm}\big(\mathrm{Conv2d}_{groups=d\_model}(queries, W_{sr}, b_{sr}; stride=ratio)\big), & ratio > 1 \\
keys,\ values, & ratio \le 1
\end{cases} \\
k &= W_k\, kv_{in}^\top + b_k,\qquad v = W_v\, kv_{in}^\top + b_v \\
A &= \mathrm{softmax}\big(q k^\top / \sqrt{d_k}\big) \\
y &= W_{out}\,(A v)^\top + b_{out}
\end{aligned}
$$

当 `apply_transform` 为真且 `n_heads > 1` 时，注意力图先经过 1x1 卷积 `A' = Conv2d(A, W_t, b_t)`，再执行 `softmax` 与 `InstanceNorm`，最后与 `v` 相乘。全部计算以 fp32 进行，输出回铸为 `queries` 的 dtype。


## 3. 接口规范

### 算子原型

```python
cann_bench.emsa(Tensor queries, Tensor keys, Tensor values, Tensor q_weight, Tensor q_bias, Tensor k_weight, Tensor k_bias, Tensor v_weight, Tensor v_bias, Tensor out_weight, Tensor out_bias, Tensor sr_weight, Tensor sr_bias, Tensor sr_norm_weight, Tensor sr_norm_bias, Tensor transform_weight, Tensor transform_bias, int n_heads, int height, int width, int ratio, int d_k, int d_v, bool apply_transform) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| queries | Tensor | 必选 | query 张量 `[B, Lq, d_model]` |
| keys | Tensor | 必选 | key 张量 `[B, Lk, d_model]`（`ratio > 1` 时不参与计算） |
| values | Tensor | 必选 | value 张量 `[B, Lk, d_model]`（`ratio > 1` 时不参与计算） |
| q_weight | Tensor | 必选 | query 投影权重 `[n_heads*d_k, d_model]` |
| q_bias | Tensor | 必选 | query 投影偏置 `[n_heads*d_k]` |
| k_weight | Tensor | 必选 | key 投影权重 `[n_heads*d_k, d_model]` |
| k_bias | Tensor | 必选 | key 投影偏置 `[n_heads*d_k]` |
| v_weight | Tensor | 必选 | value 投影权重 `[n_heads*d_v, d_model]` |
| v_bias | Tensor | 必选 | value 投影偏置 `[n_heads*d_v]` |
| out_weight | Tensor | 必选 | 输出投影权重 `[d_model, n_heads*d_v]` |
| out_bias | Tensor | 必选 | 输出投影偏置 `[d_model]` |
| sr_weight | Tensor | 必选 | 空间缩减分组卷积权重 `[d_model, 1, ratio+1, ratio+1]`；`ratio <= 1` 时为 `[0]` |
| sr_bias | Tensor | 必选 | 空间缩减卷积偏置 `[d_model]`；`ratio <= 1` 时为 `[0]` |
| sr_norm_weight | Tensor | 必选 | 空间缩减 LayerNorm 权重 `[d_model]`；`ratio <= 1` 时为 `[0]` |
| sr_norm_bias | Tensor | 必选 | 空间缩减 LayerNorm 偏置 `[d_model]`；`ratio <= 1` 时为 `[0]` |
| transform_weight | Tensor | 必选 | 注意力图 1x1 卷积权重 `[n_heads, n_heads, 1, 1]`；`apply_transform=false` 时为 `[0]` |
| transform_bias | Tensor | 必选 | 注意力图 1x1 卷积偏置 `[n_heads]`；`apply_transform=false` 时为 `[0]` |
| n_heads | int | 必选 | 注意力头数 |
| height | int | 必选 | query 空间高度，`height * width == Lq` |
| width | int | 必选 | query 空间宽度，`height * width == Lq` |
| ratio | int | 必选 | 空间缩减倍率 |
| d_k | int | 必选 | 每个头的 query/key 维度 |
| d_v | int | 必选 | 每个头的 value 维度 |
| apply_transform | bool | 必选 | 是否启用注意力图 1x1 卷积 + instance norm（仅 `n_heads > 1` 生效） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, Lq, n_heads*d_v]` | 与 `queries` 相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `height * width == queries.shape[1] == Lq`
- `keys.shape[0] == values.shape[0] == queries.shape[0] == B`
- `keys.shape[-1] == values.shape[-1] == d_model == queries.shape[-1]`（`ratio <= 1` 时）
- 线性权重形状由 `n_heads`、`d_k`、`d_v`、`d_model` 推导，见输入参数表
- `ratio > 1` 时 `sr_*` 必须为合法张量；`ratio <= 1` 时传 `[0]` 空 tensor
- `apply_transform = true` 时 `transform_*` 必须为合法张量；否则传 `[0]` 空 tensor
- `ratio > 1` 时 key/value 输入由 query 经分组卷积与 LayerNorm 重新生成，`keys` / `values` 入参不参与计算
- golden 以 fp32 计算，输出 dtype 与 `queries` 一致

## 4. 标准 Golden 代码

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

import math

import torch
import torch.nn.functional as F

"""
EMSA 算子 Torch Golden 参考实现

高效多头自注意力（Efficient Multi-Head Self-Attention）。
ratio > 1 时先对 query 做分组空间缩减（Conv2d + LayerNorm）得到 key/value 输入；
apply_transform 为真时在 softmax 前对注意力图做 1x1 卷积变换并在之后做 instance norm。
公式:
    q = W_q @ queries^T + b_q
    ratio > 1 时: kv_in = LayerNorm(Conv2d(queries, sr_weight, stride=ratio, groups=d_model))
    A = softmax(q @ k^T / sqrt(d_k))
    y = W_out @ (A @ v)^T + b_out
"""


def emsa(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    sr_weight: torch.Tensor,
    sr_bias: torch.Tensor,
    sr_norm_weight: torch.Tensor,
    sr_norm_bias: torch.Tensor,
    transform_weight: torch.Tensor,
    transform_bias: torch.Tensor,
    n_heads: int,
    height: int,
    width: int,
    ratio: int,
    d_k: int,
    d_v: int,
    apply_transform: bool,
) -> torch.Tensor:
    """
    Efficient Multi-Head Self-Attention

    公式:
        q = W_q @ queries^T + b_q
        ratio > 1 时: kv_in = LayerNorm(Conv2d(queries, sr_weight, stride=ratio, groups=d_model))
        A = softmax(q @ k^T / sqrt(d_k))
        y = W_out @ (A @ v)^T + b_out

    Args:
        queries: query 张量 [B, Lq, d_model]
        keys: key 张量 [B, Lk, d_model]
        values: value 张量 [B, Lk, d_model]
        q_weight/q_bias: query 线性投影 [H*d_k, d_model] / [H*d_k]
        k_weight/k_bias: key 线性投影 [H*d_k, d_model] / [H*d_k]
        v_weight/v_bias: value 线性投影 [H*d_v, d_model] / [H*d_v]
        out_weight/out_bias: 输出线性投影 [d_model, H*d_v] / [d_model]
        sr_weight/sr_bias: 空间缩减分组卷积 [d_model, 1, ratio+1, ratio+1] / [d_model]
        sr_norm_weight/sr_norm_bias: 空间缩减后 LayerNorm [d_model]
        transform_weight/transform_bias: 注意力图 1x1 卷积 [H, H, 1, 1] / [H]
        n_heads: 注意力头数
        height: query 空间高度，须满足 height * width == Lq
        width: query 空间宽度
        ratio: 空间缩减倍率，<= 1 时不做缩减
        d_k: 每个头的 key/query 维度
        d_v: 每个头的 value 维度
        apply_transform: 是否对注意力图做 1x1 卷积变换 + instance norm

    Returns:
        输出张量 [B, Lq, H*d_v]，dtype 与 queries 相同
    """
    n_heads = int(n_heads)
    height = int(height)
    width = int(width)
    ratio = int(ratio)
    d_k = int(d_k)
    d_v = int(d_v)

    batch, query_length, d_model = queries.shape
    key_length = keys.shape[1]
    if query_length != height * width:
        raise ValueError("height * width must match the query length")

    orig_dtype = queries.dtype
    queries_f = queries.float()
    keys_f = keys.float()
    values_f = values.float()
    q_weight_f = q_weight.float()
    q_bias_f = q_bias.float()
    k_weight_f = k_weight.float()
    k_bias_f = k_bias.float()
    v_weight_f = v_weight.float()
    v_bias_f = v_bias.float()
    out_weight_f = out_weight.float()
    out_bias_f = out_bias.float()

    q = F.linear(queries_f, q_weight_f, q_bias_f)
    q = q.view(batch, query_length, n_heads, d_k).permute(0, 2, 1, 3)

    if ratio > 1:
        reduced = queries_f.permute(0, 2, 1).reshape(batch, d_model, height, width)
        reduced = F.conv2d(
            reduced,
            sr_weight.float(),
            sr_bias.float(),
            stride=ratio,
            padding=ratio // 2,
            groups=d_model,
        )
        reduced = reduced.flatten(2).transpose(1, 2)
        reduced = F.layer_norm(
            reduced,
            (d_model,),
            sr_norm_weight.float(),
            sr_norm_bias.float(),
        )
        key_input = reduced
        value_input = reduced
        key_length = reduced.shape[1]
    else:
        key_input = keys_f
        value_input = values_f

    k = F.linear(key_input, k_weight_f, k_bias_f)
    k = k.view(batch, key_length, n_heads, d_k).permute(0, 2, 3, 1)
    v = F.linear(value_input, v_weight_f, v_bias_f)
    v = v.view(batch, key_length, n_heads, d_v).permute(0, 2, 1, 3)

    attention = torch.matmul(q, k) / math.sqrt(d_k)
    if apply_transform and n_heads > 1:
        attention = F.conv2d(attention, transform_weight.float(), transform_bias.float())
        attention = F.softmax(attention, dim=-1)
        attention = F.instance_norm(attention)
        output = torch.matmul(attention, v)
    else:
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)

    output = output.permute(0, 2, 1, 3).contiguous()
    output = output.view(batch, query_length, n_heads * d_v)
    return F.linear(output, out_weight_f, out_bias_f).to(orig_dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, Lq, Lk, d_model = 2, 1056, 1056, 1200
H, dk, dv, ratio = 15, 80, 16, 3
height, width = 32, 33
queries = torch.randn(B, Lq, d_model, dtype=torch.float32, device="npu")
keys = torch.randn(B, Lk, d_model, dtype=torch.float32, device="npu")
values = torch.randn(B, Lk, d_model, dtype=torch.float32, device="npu")
q_weight = torch.randn(H * dk, d_model, dtype=torch.float32, device="npu") * 0.05
q_bias = torch.zeros(H * dk, dtype=torch.float32, device="npu")
k_weight = torch.randn(H * dk, d_model, dtype=torch.float32, device="npu") * 0.05
k_bias = torch.zeros(H * dk, dtype=torch.float32, device="npu")
v_weight = torch.randn(H * dv, d_model, dtype=torch.float32, device="npu") * 0.05
v_bias = torch.zeros(H * dv, dtype=torch.float32, device="npu")
out_weight = torch.randn(d_model, H * dv, dtype=torch.float32, device="npu") * 0.05
out_bias = torch.zeros(d_model, dtype=torch.float32, device="npu")
sr_weight = torch.randn(d_model, 1, ratio + 1, ratio + 1, dtype=torch.float32, device="npu") * 0.05
sr_bias = torch.zeros(d_model, dtype=torch.float32, device="npu")
sr_norm_weight = torch.ones(d_model, dtype=torch.float32, device="npu")
sr_norm_bias = torch.zeros(d_model, dtype=torch.float32, device="npu")
empty = torch.empty(0, dtype=torch.float32, device="npu")
y = cann_bench.emsa(
    queries, keys, values,
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias,
    sr_weight, sr_bias, sr_norm_weight, sr_norm_bias,
    empty, empty,
    H, height, width, ratio, dk, dv, False,
)
# y.shape == (2, 1056, 240)
```
