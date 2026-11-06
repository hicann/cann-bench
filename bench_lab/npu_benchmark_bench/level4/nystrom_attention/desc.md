# NystromAttention 算子 API 描述

## 1. 算子简介

Nyström 近似注意力算子。当序列较长且特征维较高时，完整注意力矩阵 `[S, S]` 代价过高；Nyström 方法选取 `m` 个地标（landmarks），用低秩矩阵近似完整 softmax 注意力矩阵。

**主要应用场景**：
- 长序列 Transformer 的高效注意力
- 对注意力矩阵做低秩近似的推理/训练加速

**算子特征**：
- 难度等级：L4（Attention）
- 输入 `q` / `k` / `v`：`[B, H, S, D]`
- 输出：`[B, S, H*D]`
- `landmarks = min(S, num_landmarks)`；当 `landmarks >= S` 时退化为完整 softmax 注意力

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
m &= \min(S, num\_landmarks),\quad scale = 1/\sqrt{D} \\
q_l &= \mathrm{AvgPool}_m(q),\quad k_l = \mathrm{AvgPool}_m(k) \\
K_1 &= \mathrm{softmax}(q k_l^\top \cdot scale) \\
K_2 &= \mathrm{softmax}(q_l k_l^\top \cdot scale) \\
K_3 &= \mathrm{softmax}(q_l k^\top \cdot scale) \\
y &= \begin{cases}
\mathrm{softmax}(q k^\top \cdot scale)\, v, & m \ge S \\
K_1 K_2^{+} K_3 v, & m < S
\end{cases}
\end{aligned}
$$

其中 `K_2^{+}` 为 `K_2` 的伪逆，由 6 次 Newton–Schulz 迭代近似（初始化时按行/列绝对和做归一化）。


## 3. 接口规范

### 算子原型

```python
cann_bench.nystrom_attention(Tensor q, Tensor k, Tensor v, int num_landmarks) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | query 张量 `[B, H, S, D]` |
| k | Tensor | 必选 | key 张量 `[B, H, S, D]` |
| v | Tensor | 必选 | value 张量 `[B, H, S, D]` |
| num_landmarks | int | 必选 | landmark 数量，实际使用 `min(S, num_landmarks)` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, S, H*D]` | 与输入相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `q`、`k`、`v` 形状相同，均为 `[B, H, S, D]`
- `num_landmarks >= 1`
- 当 `min(S, num_landmarks) >= S` 时走完整 softmax 注意力分支
- 输出在 `H` 维与 `D` 维上合并为 `[B, S, H*D]`
- golden 使用 fp32 计算，输出 dtype 与输入一致

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
NystromAttention 算子 Torch Golden 参考实现

Nyström 近似注意力：用 landmarks 个地标近似完整注意力矩阵。
    landmarks = min(sequence, num_landmarks)
    K1 = softmax(q @ k_landmarks^T * scale)
    K2 = softmax(q_landmarks @ k_landmarks^T * scale)
    K3 = softmax(q_landmarks @ k^T * scale)
    y  = K1 @ pinv(K2) @ K3 @ v
其中 pinv(K2) 由 6 次 Newton-Schulz 迭代近似；当 landmarks >= sequence 时退化为完整 softmax 注意力。
公式:
    y = softmax(q @ k^T * scale) @ v                              (landmarks >= sequence)
    y = K1 @ K2^+ @ K3 @ v                                        (landmarks <  sequence)
"""


def nystrom_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_landmarks: int,
) -> torch.Tensor:
    """
    Nyström 近似注意力

    公式:
        y = softmax(q @ k^T * scale) @ v                              (landmarks >= sequence)
        y = K1 @ K2^+ @ K3 @ v                                        (landmarks <  sequence)

    Args:
        q: query 张量 [B, H, S, D]
        k: key 张量 [B, H, S, D]
        v: value 张量 [B, H, S, D]
        num_landmarks: landmark 数量，实际使用 min(S, num_landmarks)

    Returns:
        输出张量 [B, S, H * D]
    """
    batch, n_heads, sequence, head_dim = q.shape
    q = q.float()
    k = k.float()
    v_float = v.float()
    landmarks = min(sequence, int(num_landmarks))
    scale = 1.0 / math.sqrt(head_dim)
    if landmarks >= sequence:
        output = torch.matmul(
            torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1),
            v_float,
        )
    else:
        q_landmarks = F.adaptive_avg_pool1d(
            q.permute(0, 1, 3, 2).reshape(batch * n_heads, head_dim, sequence),
            landmarks,
        ).reshape(batch, n_heads, head_dim, landmarks).transpose(-2, -1)
        k_landmarks = F.adaptive_avg_pool1d(
            k.permute(0, 1, 3, 2).reshape(batch * n_heads, head_dim, sequence),
            landmarks,
        ).reshape(batch, n_heads, head_dim, landmarks).transpose(-2, -1)
        kernel_1 = torch.softmax(
            torch.matmul(q, k_landmarks.transpose(-2, -1)) * scale, dim=-1
        )
        kernel_2 = torch.softmax(
            torch.matmul(q_landmarks, k_landmarks.transpose(-2, -1)) * scale, dim=-1
        )
        kernel_3 = torch.softmax(
            torch.matmul(q_landmarks, k.transpose(-2, -1)) * scale, dim=-1
        )

        absolute = kernel_2.abs()
        max_row = absolute.sum(dim=-1).amax(dim=-1)
        max_column = absolute.sum(dim=-2).amax(dim=-1)
        denominator = (max_row * max_column).clamp_min(1e-6)
        inverse = kernel_2.transpose(-2, -1)
        inverse = inverse / denominator.unsqueeze(-1).unsqueeze(-1)
        identity = torch.eye(landmarks, dtype=q.dtype).view(1, 1, landmarks, landmarks)
        for _ in range(6):
            product = torch.matmul(kernel_2, inverse)
            inverse = 0.25 * torch.matmul(
                inverse,
                13.0 * identity
                - torch.matmul(
                    product,
                    15.0 * identity - torch.matmul(product, 7.0 * identity - product),
                ),
            )
        output = torch.matmul(
            kernel_1, torch.matmul(inverse, torch.matmul(kernel_3, v_float))
        )

    output = output.to(dtype=v.dtype).transpose(1, 2).contiguous()
    return output.view(batch, sequence, n_heads * head_dim)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, S, D = 2, 8, 383, 64
q = torch.rand(B, H, S, D, dtype=torch.float32, device="npu")
k = torch.rand(B, H, S, D, dtype=torch.float32, device="npu")
v = torch.rand(B, H, S, D, dtype=torch.float32, device="npu")
y = cann_bench.nystrom_attention(q, k, v, 11)
# y.shape == (2, 383, 512)
```
