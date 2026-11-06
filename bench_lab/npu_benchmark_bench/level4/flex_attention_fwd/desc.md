# FlexAttentionFwd 算子 API 描述

## 1. 算子简介

FlexAttentionFwd 实现 FlexAttention 前向中由 benchmark 覆盖的数学子集：`score_mod = None`、可选的因果掩码（`mask_mod`）以及可选的 GQA（分组查询注意力）。其计算为标准的缩放点积注意力；因果掩码按右下角对齐，因此允许 query 与 key 序列长度不同。

**主要应用场景**：
- 大模型训练/推理中的可配置注意力（因果、全注意力、GQA）
- 变长 query/key 的注意力计算

**算子特征**：
- 难度等级：L4（Attention）
- 输入：`query` `[B, H, S_q, D]`、`key` / `value` `[B, H_kv, S_k, D]`
- 输出：`[B, H, S_q, D]`
- 支持 float16 / float32 / bfloat16，内部 fp32 计算后回铸输入 dtype

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
S &= \frac{Q K^{\top}}{\sqrt{D}} \\
S_{ij} &= -\infty \quad \text{若 is\_causal 且 } j > i + (S_k - S_q) \\
P &= \exp\!\left(S - \mathrm{logsumexp}(S, \dim=-1)\right) \\
O &= P V
\end{aligned}
$$

`enable_gqa = true` 时，先把 KV 头按 `rep = H // H_kv` 复制（`repeat_interleave`）到 query 头数，再做上述计算。所有计算在 fp32 下进行，输出回铸为 `query` 的 dtype。

## 3. 接口规范

### 算子原型

```python
cann_bench.flex_attention_fwd(query, key, value, is_causal=false, enable_gqa=false) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | query 张量 `[B, H, S_q, D]` |
| key | Tensor | 必选 | key 张量 `[B, H_kv, S_k, D]` |
| value | Tensor | 必选 | value 张量 `[B, H_kv, S_k, D]` |
| is_causal | bool | false | 是否使用右下角对齐的因果掩码 |
| enable_gqa | bool | false | 是否启用 GQA |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, H, S_q, D]` | 与 `query` 相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `query`、`key`、`value` 均为 4 维，且 `D` 一致；`key` 与 `value` 的 `B`、`H_kv`、`S_k` 一致。
- `is_causal = true` 时使用右下角对齐掩码：位置 `(i, j)` 被屏蔽当且仅当 `j > i + (S_k - S_q)`；推荐 `S_q <= S_k`，以保证每个 query 至少有一个可见 key。
- `enable_gqa = true` 时要求 `H % H_kv == 0`，KV 头按 `H // H_kv` 复制到 query 头数；`enable_gqa = false` 时要求 `H == H_kv`。
- golden 使用 fp32 计算并按 query 分块，避免一次物化完整的 `[B, H, S_q, S_k]` 打分矩阵。

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
FlexAttentionFwd 算子 Torch Golden 参考实现

FlexAttention 前向的数学子集（score_mod 为空、可选因果掩码与 GQA）：
    S = Q @ K^T * (1 / sqrt(D))
    if is_causal: S[i, j] = -inf，当 j > i + (S_k - S_q)
    P = exp(S - logsumexp(S, dim=-1))
    O = P @ V
enable_gqa 时先把 KV 头按 H // H_kv 复制到 Q 头数。
"""


def flex_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    FlexAttention 前向

    公式:
        S = Q @ K^T * (1 / sqrt(D))
        if is_causal: S[i, j] = -inf，当 j > i + (S_k - S_q)
        P = exp(S - logsumexp(S, dim=-1))
        O = P @ V

    Args:
        query: query 张量 [B, H, S_q, D]
        key: key 张量 [B, H_kv, S_k, D]
        value: value 张量 [B, H_kv, S_k, D]
        is_causal: 是否使用右下角对齐的因果掩码
        enable_gqa: 是否启用 GQA

    Returns:
        注意力输出 [B, H, S_q, D]
    """
    batch, heads, s_q, dim = query.shape
    h_kv = key.shape[1]
    s_k = key.shape[2]
    scale = 1.0 / math.sqrt(dim)

    query_f = query.float()
    key_f = key.float()
    value_f = value.float()

    if enable_gqa and heads != h_kv:
        rep = heads // h_kv
        key_f = key_f.repeat_interleave(rep, dim=1)
        value_f = value_f.repeat_interleave(rep, dim=1)

    causal_mask = None
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, dtype=torch.bool), diagonal=s_k - s_q + 1
        )

    # 按 query 分块，避免一次物化 [B, H, S_q, S_k]
    out = torch.empty(batch, heads, s_q, value_f.shape[-1], dtype=torch.float32)
    budget = 2 ** 24
    blk = max(1, min(s_q, budget // max(1, batch * heads * s_k)))
    for start in range(0, s_q, blk):
        end = min(start + blk, s_q)
        scores = torch.matmul(query_f[:, :, start:end], key_f.transpose(-2, -1)) * scale
        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask[start:end].view(1, 1, end - start, s_k), float("-inf")
            )
        logsumexp = torch.logsumexp(scores, dim=-1)
        attn_probs = torch.exp(scores - logsumexp.unsqueeze(-1))
        out[:, :, start:end] = torch.matmul(attn_probs, value_f)

    return out.to(dtype=query.dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, S_q, S_k, D = 2, 4, 8192, 8192, 64
query = torch.randn(B, H, S_q, D, dtype=torch.bfloat16, device="npu")
key = torch.randn(B, 1, S_k, D, dtype=torch.bfloat16, device="npu")
value = torch.randn(B, 1, S_k, D, dtype=torch.bfloat16, device="npu")
y = cann_bench.flex_attention_fwd(query, key, value, True, True)
# y.shape == (2, 4, 8192, 64)
```
