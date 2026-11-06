# ScaledDotProductAttentionBackward 算子 API 描述

## 1. 算子简介

缩放点积注意力（SDPA）的反向计算。利用前向保存的 `logsumexp` 重建 softmax 概率矩阵 `P`，无需重新执行前向 softmax，即可同时计算 query、key、value 三个梯度；支持可选因果掩码。

**主要应用场景**：
- Transformer 训练中的注意力反向传播
- FlashAttention 反向算法的参考实现

**算子特征**：
- 难度等级：L4（Attention）
- 输入：`grad_output/query/key/value/attn_output` 为 `[B, H, S, D]`，`logsumexp` 为 `[B, H, S]`
- 输出：`grad_query`/`grad_key`/`grad_value` 三个梯度
- golden 全程 fp32 计算，输出回铸输入 dtype
- 无内部随机权重，所有输入均为显式 tensor

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
P &= \exp\left(scale \cdot Q K^\top - logsumexp\right) \quad (\text{causal 时上三角为 } 0) \\
dV &= P^\top dO \\
dP &= dO\, V^\top \\
D &= \mathrm{rowsum}(dO \odot O) \\
dS &= P \odot (dP - D) \\
dQ &= dS\, K \cdot scale,\qquad dK = dS^\top Q \cdot scale
\end{aligned}
$$

其中 `scale = 1/sqrt(D)`；因果掩码在 softmax 前把 `j > i` 的位置置为 `-inf`，从而重建出的 `P` 上三角为 0。`get_input` 会按前向公式重算 `attn_output` 与 `logsumexp`，保证反向输入自洽。

## 3. 接口规范

### 算子原型

```python
cann_bench.scaled_dot_product_attention_backward(Tensor grad_output, Tensor query, Tensor key, Tensor value, Tensor attn_output, Tensor logsumexp, bool is_causal=false) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| grad_output | Tensor | 必选 | 输出梯度 `dO` `[B, H, S_q, D]` |
| query | Tensor | 必选 | 前向 query `[B, H, S_q, D]` |
| key | Tensor | 必选 | 前向 key `[B, H, S_k, D]` |
| value | Tensor | 必选 | 前向 value `[B, H, S_k, D]` |
| attn_output | Tensor | 必选 | 前向注意力输出 `O` `[B, H, S_q, D]` |
| logsumexp | Tensor | 必选 | 前向 softmax 的 logsumexp `[B, H, S_q]`，float32 |
| is_causal | bool | false | 是否使用因果掩码 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| grad_query | `[B, H, S_q, D]` | 与 `query` 相同 | query 梯度 |
| grad_key | `[B, H, S_k, D]` | 与 `key` 相同 | key 梯度 |
| grad_value | `[B, H, S_k, D]` | 与 `value` 相同 | value 梯度 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `query`、`grad_output`、`attn_output` 形状为 `[B, H, S_q, D]`；`key`、`value` 形状为 `[B, H, S_k, D]`
- `logsumexp` 形状为 `[B, H, S_q]`，dtype 固定为 float32
- `is_causal=True` 时通常要求 `S_q == S_k`，位置 `i` 只能看见 `j <= i`
- 全程 fp32 计算，梯度输出分别回铸为 `query`/`key`/`value` 的 dtype

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
ScaledDotProductAttentionBackward 算子 Torch Golden 参考实现

融合缩放点积注意力反向：由前向保存的 logsumexp 重建 softmax 概率，计算
query/key/value 梯度，支持因果掩码。全程 fp32 计算，输出回铸输入 dtype。

公式:
    P = exp(scale * Q K^T - logsumexp)   # causal 时上三角置 0
    dV = P^T dO
    dP = dO V^T
    D  = rowsum(dO ⊙ O)
    dS = P ⊙ (dP - D)
    dQ = dS K scale
    dK = dS^T Q scale
"""


def _causal_mask(seq_q, seq_k):
    return torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=seq_k - seq_q + 1)


def scaled_dot_product_attention_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
) -> tuple:
    """
    缩放点积注意力反向

    公式:
        P = exp(scale * Q K^T - logsumexp)   # causal 时上三角置 0
        dV = P^T dO
        dP = dO V^T
        D  = rowsum(dO ⊙ O)
        dS = P ⊙ (dP - D)
        dQ = dS K scale
        dK = dS^T Q scale

    Args:
        grad_output: 输出梯度 dO [B, H, S_q, D]
        query: 前向 query [B, H, S_q, D]
        key: 前向 key [B, H, S_k, D]
        value: 前向 value [B, H, S_k, D]
        attn_output: 前向注意力输出 O [B, H, S_q, D]
        logsumexp: 前向 softmax 的 logsumexp [B, H, S_q]
        is_causal: 是否使用因果掩码

    Returns:
        (grad_query, grad_key, grad_value)，shape 分别与 query/key/value 一致
    """
    batch, n_heads, seq_q, dim = query.shape
    seq_k = key.shape[2]
    scale = 1.0 / math.sqrt(dim)

    mask = _causal_mask(seq_q, seq_k) if is_causal else None

    grad_q = torch.empty(batch, n_heads, seq_q, dim, dtype=torch.float32)
    grad_k = torch.empty(batch, n_heads, seq_k, dim, dtype=torch.float32)
    grad_v = torch.empty(batch, n_heads, seq_k, dim, dtype=torch.float32)

    for b in range(batch):
        for h in range(n_heads):
            q = query[b, h].float()
            k = key[b, h].float()
            v = value[b, h].float()
            do = grad_output[b, h].float()
            o = attn_output[b, h].float()
            lse = logsumexp[b, h].float()

            scores = torch.matmul(q, k.transpose(0, 1)) * scale
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            probs = torch.exp(scores - lse.unsqueeze(-1))

            d_value = torch.matmul(probs.transpose(0, 1), do)
            d_probs = torch.matmul(do, v.transpose(0, 1))
            d_term = (do * o).sum(dim=-1, keepdim=True)
            d_scores = probs * (d_probs - d_term)

            grad_q[b, h] = torch.matmul(d_scores, k) * scale
            grad_k[b, h] = torch.matmul(d_scores.transpose(0, 1), q) * scale
            grad_v[b, h] = d_value

    return grad_q.to(query.dtype), grad_k.to(key.dtype), grad_v.to(value.dtype)


def get_input(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
    **kwargs,
) -> list:
    """按前向公式重算 attn_output 与 logsumexp，保证反向输入自洽。"""
    q = query.float()
    k = key.float()
    v = value.float()
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        scores = scores.masked_fill(_causal_mask(seq_q, seq_k), float('-inf'))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse.unsqueeze(-1))
    attn_output = torch.matmul(probs, v).to(attn_output.dtype)
    logsumexp = lse.to(logsumexp.dtype)
    return [grad_output, query, key, value, attn_output, logsumexp]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, S, D = 1, 8, 256, 64
grad_output = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
query = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
key = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
value = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
attn_output = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
logsumexp = torch.randn(B, H, S, dtype=torch.float32, device="npu")
dq, dk, dv = cann_bench.scaled_dot_product_attention_backward(grad_output, query, key, value, attn_output, logsumexp, False)
# dq.shape == dk.shape == dv.shape == (1, 8, 256, 64)
```
