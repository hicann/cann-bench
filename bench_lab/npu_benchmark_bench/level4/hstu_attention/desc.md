# HstuAttention 算子 API 描述

## 1. 算子简介

HSTU（Hierarchical Sequential Transduction Unit）注意力是生成式推荐场景中的变长因果注意力算子。与传统 softmax 注意力不同，它使用逐点 SiLU 激活后直接对 value 加权求和，且不做归一化。

**主要应用场景**：
- 生成式推荐模型（如 HSTU、Meta 推荐系统）中的序列建模
- 变长序列的因果注意力计算

**算子特征**：
- 难度等级：L4（Attention）
- 输入：`q` / `k` / `v` 为按序列拼接（varlen）的 `[sum_L, H, D]` 张量；`seq_offsets` 描述各序列边界
- 输出：`[sum_L, H, Dv]`
- 因果掩码仅保留 `j <= i`（含对角线）

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
s_{ij} &= \alpha \cdot (q_i \cdot k_j) \\
w_{ij} &= \frac{\mathrm{SiLU}(s_{ij})}{max\_seq\_len}, \quad j \le i \\
y_i &= \sum_{j \le i} w_{ij}\, v_j
\end{aligned}
$$

其中 `SiLU(x) = x * sigmoid(x)`。`seq_offsets` 为长度 `B+1` 的前缀和边界，第 `b` 条序列占据行区间 `[seq_offsets[b], seq_offsets[b+1])`；打分在使用前统一升 fp32 计算，输出回铸为 `v` 的 dtype。


## 3. 接口规范

### 算子原型

```python
cann_bench.hstu_attention(Tensor q, Tensor k, Tensor v, int[] seq_offsets, int max_seq_len, float alpha) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | query 张量 `[sum_L, H, D]` |
| k | Tensor | 必选 | key 张量 `[sum_L, H, D]` |
| v | Tensor | 必选 | value 张量 `[sum_L, H, Dv]` |
| seq_offsets | List[int] | 必选 | 长度 `B+1` 的序列边界（cumsum），`seq_offsets[0] == 0`，末值等于 `sum_L` |
| max_seq_len | int | 必选 | 序列最大长度，同时作为打分缩放分母 |
| alpha | float | 必选 | 打分缩放系数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[sum_L, H, Dv]` | 与 `v` 相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `q.shape == k.shape`，`v.shape[:2] == q.shape[:2]`，`q.shape[-1] == k.shape[-1]`
- `seq_offsets` 长度为 `B+1`，严格非递减，`seq_offsets[0] == 0`，`seq_offsets[-1] == q.shape[0]`
- 每条序列长度 `seq_offsets[b+1] - seq_offsets[b]` 不超过 `max_seq_len`
- 因果掩码：位置 `i` 只能看见 `j <= i`
- golden 使用 fp32 累加，输出 dtype 与 `v` 一致

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

import torch

"""
HstuAttention 算子 Torch Golden 参考实现

HSTU（生成式推荐）的变长因果注意力：以 SiLU 取代 softmax，且不做归一化。
q / k / v 为按序列拼接（varlen）的 [sum_L, H, D] 张量，seq_offsets 给出各序列边界。
公式:
    scores[i, j] = alpha * (q[i] . k[j])
    w[i, j]      = SiLU(scores[i, j]) / max_seq_len, 仅保留 j <= i（因果）
    y[i]         = sum_j w[i, j] * v[j]
"""


def hstu_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets,
    max_seq_len: int,
    alpha: float,
) -> torch.Tensor:
    """
    HSTU 变长因果注意力

    公式:
        scores[i, j] = alpha * (q[i] . k[j])
        w[i, j]      = SiLU(scores[i, j]) / max_seq_len, 仅保留 j <= i（因果）
        y[i]         = sum_j w[i, j] * v[j]

    Args:
        q: query 张量 [sum_L, H, D]
        k: key 张量 [sum_L, H, D]
        v: value 张量 [sum_L, H, Dv]
        seq_offsets: 长度 B+1 的序列边界（cumsum），seq_offsets[0] == 0，末值 == sum_L
        max_seq_len: 序列最大长度，同时作为打分的缩放分母
        alpha: 打分缩放系数

    Returns:
        输出张量 [sum_L, H, Dv]
    """
    if isinstance(seq_offsets, torch.Tensor):
        seq_offsets = seq_offsets.tolist()
    offsets = [int(x) for x in seq_offsets]
    total, heads, dim = q.shape
    dim_v = v.shape[-1]
    batch = len(offsets) - 1
    rows = int(max_seq_len)

    pos = torch.arange(rows, dtype=torch.long)
    lengths = torch.tensor([offsets[i + 1] - offsets[i] for i in range(batch)], dtype=torch.long)
    valid = pos.unsqueeze(0) < lengths.unsqueeze(1)
    starts = torch.tensor(offsets[:-1], dtype=torch.long)
    gidx = (starts.unsqueeze(1) + pos.unsqueeze(0)).reshape(-1).clamp(max=total - 1)

    q_pad = q.index_select(0, gidx).float().reshape(batch, rows, heads, dim)
    k_pad = k.index_select(0, gidx).float().reshape(batch, rows, heads, dim)
    v_pad = v.index_select(0, gidx).float().reshape(batch, rows, heads, dim_v)

    # 按 query 分块，避免一次物化 [B, H, L, L]
    out = torch.zeros(batch, rows, heads, dim_v, dtype=torch.float32)
    budget = 2 ** 24
    blk = max(1, min(rows, budget // max(1, batch * heads * rows)))
    for start in range(0, rows, blk):
        end = min(start + blk, rows)
        scores = torch.einsum("bqhd,bkhd->bhqk", q_pad[:, start:end], k_pad[:, :end]) * alpha
        weights = torch.nn.functional.silu(scores) / rows
        keep = (pos[start:end].unsqueeze(1) >= pos[:end].unsqueeze(0)).reshape(1, 1, end - start, end)
        weights = weights * keep
        weights = weights * valid[:, :end].reshape(batch, 1, 1, end)
        out[:, start:end] = torch.einsum("bhqk,bkhd->bqhd", weights, v_pad[:, :end])
    out = out.reshape(batch * rows, heads, dim_v)

    flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
    buf = torch.zeros(total + 1, heads, dim_v, dtype=torch.float32)
    buf.index_copy_(0, flat_idx, out)
    return buf[:total].to(v.dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

sum_L, H, D, Dv = 455, 4, 64, 64
q = torch.randn(sum_L, H, D, dtype=torch.bfloat16, device="npu")
k = torch.randn(sum_L, H, D, dtype=torch.bfloat16, device="npu")
v = torch.rand(sum_L, H, Dv, dtype=torch.bfloat16, device="npu")
seq_offsets = [0, 109, 215, 349, 455]
y = cann_bench.hstu_attention(q, k, v, seq_offsets, 134, 0.125)
# y.shape == (455, 4, 64)
```
