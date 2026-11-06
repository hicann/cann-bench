# MultiHeadAttention 算子 API 描述

## 1. 算子简介

标准多头注意力前向算子。query、key、value 分别经无偏置线性投影后按头拆分，做缩放点积注意力，再经输出投影得到结果；query 与 key/value 的序列长度允许不同。

**主要应用场景**：
- Transformer 编解码器中的交叉/自注意力
- 序列到序列建模中的多头注意力计算

**算子特征**：
- 难度等级：L4（Attention）
- 张量输入：`query` / `key` / `value` + 四个 `[D, D]` 无偏置投影权重；属性给出头数
- 输出：单输出 `y`，shape 为 `[B, Lq, D]`

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
q &= Q\,\mathrm{query},\quad k = K\,\mathrm{key},\quad v = V\,\mathrm{value} \\
P &= \mathrm{softmax}\!\left(\frac{q\,k^\top}{\sqrt{d_{head}}}\right) \\
y &= O\,(P\,v)
\end{aligned}
$$

其中 `Q, K, V, O ∈ R^{D×D}` 为无偏置投影权重，`d_head = D / n_heads`。打分与 softmax 在 fp32 下计算，注意力输出回铸输入 dtype 后做输出投影。

## 3. 接口规范

### 算子原型

```python
cann_bench.multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor q_proj_weight, Tensor k_proj_weight, Tensor v_proj_weight, Tensor out_proj_weight, int n_heads) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | query 张量，`[B, Lq, D]` |
| key | Tensor | 必选 | key 张量，`[B, Lk, D]` |
| value | Tensor | 必选 | value 张量，`[B, Lk, D]` |
| q_proj_weight | Tensor | 必选 | query 投影权重，`[D, D]` |
| k_proj_weight | Tensor | 必选 | key 投影权重，`[D, D]` |
| v_proj_weight | Tensor | 必选 | value 投影权重，`[D, D]` |
| out_proj_weight | Tensor | 必选 | 输出投影权重，`[D, D]` |
| n_heads | int | 必选 | 注意力头数，须整除 `D` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, Lq, D]` | 与 `query` 相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `query.shape[2] == key.shape[2] == value.shape[2] == D`
- `key.shape[:2] == value.shape[:2]`，`key.shape[1] == Lk`
- `D % n_heads == 0`，`head_dim = D // n_heads`
- 所有张量输入 dtype 一致
- golden 中打分与 softmax 使用 fp32，注意力输出回铸输入 dtype

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

"""
MultiHeadAttention 算子 Torch Golden 参考实现

标准多头注意力前向：query/key/value 分别经无偏置线性投影，按头拆分后做缩放点积
注意力，经输出投影得到结果。query 与 key/value 的序列长度允许不同。

公式:
    q = Q query,  k = K key,  v = V value
    P = softmax(q k^T / sqrt(head_dim))
    y = O (P v)
"""


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    多头注意力前向

    公式:
        q = Q query,  k = K key,  v = V value
        P = softmax(q k^T / sqrt(head_dim))
        y = O (P v)

    Args:
        query: query 张量 [B, Lq, D]
        key: key 张量 [B, Lk, D]
        value: value 张量 [B, Lk, D]
        q_proj_weight: query 投影权重 [D, D]
        k_proj_weight: key 投影权重 [D, D]
        v_proj_weight: value 投影权重 [D, D]
        out_proj_weight: 输出投影权重 [D, D]
        n_heads: 注意力头数

    Returns:
        注意力输出 [B, Lq, D]
    """
    n_heads = int(n_heads)
    batch, query_length, d_model = query.shape
    key_length = key.shape[1]
    head_dim = d_model // n_heads
    dtype = query.dtype
    linear = torch.nn.functional.linear

    q = linear(query, q_proj_weight).reshape(
        batch, query_length, n_heads, head_dim).transpose(1, 2)
    k = linear(key, k_proj_weight).reshape(
        batch, key_length, n_heads, head_dim).transpose(1, 2)
    v = linear(value, v_proj_weight).reshape(
        batch, key_length, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().reshape(batch, query_length, d_model)
    return linear(output, out_proj_weight)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, Lq, Lk, D, H = 1, 59, 89, 360, 6
query = torch.randn(B, Lq, D, dtype=torch.bfloat16)
key = torch.randn(B, Lk, D, dtype=torch.bfloat16)
value = torch.randn(B, Lk, D, dtype=torch.bfloat16)
w = 1.0 / (D ** 0.5)
q_proj_weight = torch.randn(D, D) * w
k_proj_weight = torch.randn(D, D) * w
v_proj_weight = torch.randn(D, D) * w
out_proj_weight = torch.randn(D, D) * w
y = cann_bench.multi_head_attention(
    query, key, value, q_proj_weight, k_proj_weight, v_proj_weight,
    out_proj_weight, H,
)
# y.shape == (1, 59, 360)
```

### 补充说明

源实现通过 `_cache` + `torch.manual_seed(42)` 在内部生成 4 个无偏置线性投影权重，本任务已全部提升为显式 proto 输入，shape 由 `D` 推导。
