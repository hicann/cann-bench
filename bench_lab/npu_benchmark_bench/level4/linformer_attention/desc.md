# LinformerAttention 算子 API 描述

## 1. 算子简介

Linformer 低秩线性注意力前向算子。query/key/value 先经 d_model 维线性投影，key/value 再沿序列维做低秩投影（`S -> projection_dim`），随后在低秩 key/value 上做标准缩放点积注意力，最后经输出投影得到结果。全部分析权重由输入显式给出。

**主要应用场景**：
- 长序列线性复杂度注意力建模
- 需要将序列维压缩为低秩表示的高效 Transformer

**算子特征**：
- 难度等级：L4（LinearAttention）
- 张量输入：`x` + 四个 `[D, D]` 投影权重 + 两个 `[projection_dim, S]` 序列低秩投影权重；属性给出头数与低秩维度
- 输出：单输出 `y`，shape 与 `x` 一致 `[B, S, D]`

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
q &= Q x,\quad k = K x,\quad v = V x \\
k' &= E k,\quad v' = F v \\
P &= \mathrm{softmax}\!\left(\frac{q\,k'^\top}{\sqrt{d_{head}}}\right) \\
y &= O\,(P\,v')
\end{aligned}
$$

其中 `Q, K, V, O ∈ R^{D×D}` 为 d_model 维投影权重，`E, F ∈ R^{projection_dim×S}` 为沿序列维的低秩投影权重（对最后一维 `S` 做线性变换）。打分在 fp32 下计算，softmax 后与 value 相乘并回铸输入 dtype，最后做输出投影。

## 3. 接口规范

### 算子原型

```python
cann_bench.linformer_attention(Tensor x, Tensor q_proj_weight, Tensor k_proj_weight, Tensor v_proj_weight, Tensor out_proj_weight, Tensor key_reduce_weight, Tensor value_reduce_weight, int n_heads, int projection_dim) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列特征，`[B, S, D]` |
| q_proj_weight | Tensor | 必选 | query 投影权重，`[D, D]` |
| k_proj_weight | Tensor | 必选 | key 投影权重，`[D, D]` |
| v_proj_weight | Tensor | 必选 | value 投影权重，`[D, D]` |
| out_proj_weight | Tensor | 必选 | 输出投影权重，`[D, D]` |
| key_reduce_weight | Tensor | 必选 | key 序列维低秩投影权重，`[projection_dim, S]` |
| value_reduce_weight | Tensor | 必选 | value 序列维低秩投影权重，`[projection_dim, S]` |
| n_heads | int | 必选 | 注意力头数，须整除 `D` |
| projection_dim | int | 必选 | 序列维低秩投影维度，取值 `[1, S]` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, S, D]` | 与 `x` 相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `D % n_heads == 0`，`head_dim = D // n_heads`
- `1 <= projection_dim <= S`
- 所有张量输入 dtype 一致
- golden 中打分与 softmax 使用 fp32，输出前回铸输入 dtype

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
LinformerAttention 算子 Torch Golden 参考实现

Linformer 低秩线性注意力：query/key/value 先做 d_model 维线性投影，key/value 再沿
序列维做低秩投影（S -> projection_dim），随后在投影后的 key/value 上做缩放点积注意力。
投影层与低秩投影权重全部由输入给出。

公式:
    q = Q x,  k = K x,  v = V x
    k' = E k,  v' = F v                      # E, F: S -> projection_dim
    P = softmax(q k'^T / sqrt(head_dim))
    y = O (P v')
"""


def linformer_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    key_reduce_weight: torch.Tensor,
    value_reduce_weight: torch.Tensor,
    n_heads: int,
    projection_dim: int,
) -> torch.Tensor:
    """
    Linformer 低秩线性注意力前向

    公式:
        q = Q x,  k = K x,  v = V x
        k' = E k,  v' = F v
        P = softmax(q k'^T / sqrt(head_dim))
        y = O (P v')

    Args:
        x: 输入序列特征 [B, S, D]
        q_proj_weight: query 投影权重 [D, D]
        k_proj_weight: key 投影权重 [D, D]
        v_proj_weight: value 投影权重 [D, D]
        out_proj_weight: 输出投影权重 [D, D]
        key_reduce_weight: key 序列维低秩投影权重 [projection_dim, S]
        value_reduce_weight: value 序列维低秩投影权重 [projection_dim, S]
        n_heads: 注意力头数
        projection_dim: 序列维低秩投影维度

    Returns:
        注意力输出 [B, S, D]
    """
    n_heads = int(n_heads)
    projection_dim = int(projection_dim)
    batch, sequence, d_model = x.shape
    head_dim = d_model // n_heads
    dtype = x.dtype
    linear = torch.nn.functional.linear

    q = linear(x, q_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
    k = linear(x, k_proj_weight).reshape(batch, sequence, n_heads, head_dim).permute(0, 2, 3, 1)
    v = linear(x, v_proj_weight).reshape(batch, sequence, n_heads, head_dim).permute(0, 2, 3, 1)
    k = linear(k, key_reduce_weight).transpose(-2, -1)
    v = linear(v, value_reduce_weight).transpose(-2, -1)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    weights = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().reshape(batch, sequence, d_model)
    return linear(output, out_proj_weight)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, D, H, P = 2, 149, 832, 13, 53
x = torch.randn(B, S, D, dtype=torch.bfloat16)
w = 1.0 / (D ** 0.5)
q_proj_weight = torch.randn(D, D) * w
k_proj_weight = torch.randn(D, D) * w
v_proj_weight = torch.randn(D, D) * w
out_proj_weight = torch.randn(D, D) * w
key_reduce_weight = torch.randn(P, S) * w
value_reduce_weight = torch.randn(P, S) * w
y = cann_bench.linformer_attention(
    x, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight,
    key_reduce_weight, value_reduce_weight, H, P,
)
# y.shape == (2, 149, 832)
```

### 补充说明

源实现通过 `_cache` + `torch.manual_seed(42)` 在内部生成 6 个无偏置线性投影权重，本任务已全部提升为显式 proto 输入，shape 由 `D` / `S` / `projection_dim` 推导。本 golden 与源 `Model.forward` 逐算子对齐。
