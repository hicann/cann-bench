# ViTAttention 算子 API 描述

## 1. 算子简介

Vision Transformer 自注意力前向算子。输入 token 序列同源生成 query/key/value，分别经无偏置线性投影后按头拆分，做缩放点积注意力，再经输出投影得到结果。

**主要应用场景**：
- Vision Transformer 编码器中的自注意力
- 图像/视频 patch token 序列的全局建模

**算子特征**：
- 难度等级：L4（VisionAttention）
- 张量输入：`x` + 四个 `[D, D]` 无偏置投影权重；属性给出头数
- 输出：单输出 `y`，shape 与 `x` 一致 `[B, S, D]`

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
q &= Q x,\quad k = K x,\quad v = V x \\
P &= \mathrm{softmax}\!\left(\frac{q\,k^\top}{\sqrt{d_{head}}}\right) \\
y &= O\,(P\,v)
\end{aligned}
$$

其中 `Q, K, V, O ∈ R^{D×D}` 为无偏置投影权重，`d_head = D / n_heads`。打分与 softmax 在 fp32 下计算，注意力输出回铸输入 dtype 后做输出投影。

## 3. 接口规范

### 算子原型

```python
cann_bench.vit_attention(Tensor x, Tensor q_proj_weight, Tensor k_proj_weight, Tensor v_proj_weight, Tensor out_proj_weight, int n_heads) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入 token 序列，`[B, S, D]` |
| q_proj_weight | Tensor | 必选 | query 投影权重，`[D, D]` |
| k_proj_weight | Tensor | 必选 | key 投影权重，`[D, D]` |
| v_proj_weight | Tensor | 必选 | value 投影权重，`[D, D]` |
| out_proj_weight | Tensor | 必选 | 输出投影权重，`[D, D]` |
| n_heads | int | 必选 | 注意力头数，须整除 `D` |

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
ViTAttention 算子 Torch Golden 参考实现

Vision Transformer 自注意力前向：输入 token 序列分别经 query/key/value 无偏置线性
投影，按头拆分后做缩放点积注意力，再经输出投影得到结果。

公式:
    q = Q x,  k = K x,  v = V x
    P = softmax(q k^T / sqrt(head_dim))
    y = O (P v)
"""


def vit_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    ViT 自注意力前向

    公式:
        q = Q x,  k = K x,  v = V x
        P = softmax(q k^T / sqrt(head_dim))
        y = O (P v)

    Args:
        x: 输入 token 序列 [B, S, D]
        q_proj_weight: query 投影权重 [D, D]
        k_proj_weight: key 投影权重 [D, D]
        v_proj_weight: value 投影权重 [D, D]
        out_proj_weight: 输出投影权重 [D, D]
        n_heads: 注意力头数

    Returns:
        注意力输出 [B, S, D]
    """
    n_heads = int(n_heads)
    batch, sequence, d_model = x.shape
    head_dim = d_model // n_heads
    dtype = x.dtype
    linear = torch.nn.functional.linear

    q = linear(x, q_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
    k = linear(x, k_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)
    v = linear(x, v_proj_weight).reshape(batch, sequence, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().reshape(batch, sequence, d_model)
    return linear(output, out_proj_weight)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, D, H = 1, 50, 896, 14
x = torch.randn(B, S, D, dtype=torch.bfloat16)
w = 1.0 / (D ** 0.5)
q_proj_weight = torch.randn(D, D) * w
k_proj_weight = torch.randn(D, D) * w
v_proj_weight = torch.randn(D, D) * w
out_proj_weight = torch.randn(D, D) * w
y = cann_bench.vit_attention(
    x, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight, H,
)
# y.shape == (1, 50, 896)
```

### 补充说明

源实现通过 `_cache` + `torch.manual_seed(42)` 在内部生成 4 个无偏置线性投影权重，本任务已全部提升为显式 proto 输入，shape 由 `D` 推导。
