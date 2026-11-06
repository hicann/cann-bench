# CrissCrossAttention 算子 API 描述

## 1. 算子简介

CrissCrossAttention 实现 Criss-Cross Attention 模块的前向。它对特征图上的每个空间位置，分别沿其所在行（横向）和所在列（纵向）收集 $W$ 个与 $H$ 个上下文 key，拼接后用 softmax 得到十字交叉注意力权重，再对 value 的同行/同列元素加权求和。两路输出相加后按可学习标量 `gamma` 缩放，并做残差连接。该模块可用两条十字路径高效捕获全图上下文。

**主要应用场景**：
- 语义分割中的长程上下文建模（CCNet 系列）
- 需要低复杂度全图上下文聚合的视觉任务

**算子特征**：
- 难度等级：L4（VisionAttention）
- 8 个必选张量输入：输入特征图 + query/key/value 三组 1x1 卷积的权重与偏置 + 缩放标量 `gamma`
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，三组 1x1 投影为
$\mathbf{Q}=\mathrm{Conv}_{1\times1}^{q}(\mathbf{x})\in\mathbb{R}^{B\times C/8\times H\times W}$、
$\mathbf{K}=\mathrm{Conv}_{1\times1}^{k}(\mathbf{x})\in\mathbb{R}^{B\times C/8\times H\times W}$、
$\mathbf{V}=\mathrm{Conv}_{1\times1}^{v}(\mathbf{x})\in\mathbb{R}^{B\times C\times H\times W}$。对空间位置 $(y,x)$：

$$
\begin{aligned}
e_H[b, y, x, j] &= \sum_c Q[b, c, y, x]\,K[b, c, j, x], \quad e_H[b, y, x, y] = -\infty \\
e_W[b, y, x, j] &= \sum_c Q[b, c, y, x]\,K[b, c, y, j] \\
\mathbf{a} &= \mathrm{softmax}\!\left([e_H; e_W], \dim=-1\right) \\
o_H[b, c, y, x] &= \sum_j V[b, c, j, x]\,a_H[b, y, x, j] \\
o_W[b, c, y, x] &= \sum_j V[b, c, y, j]\,a_W[b, y, x, j] \\
\mathbf{y} &= \gamma \cdot (o_H + o_W) + \mathbf{x}
\end{aligned}
$$

其中 $a_H$、$a_W$ 分别为拼接注意力在横向、纵向部分的切片；`gamma` 为形状 `[1]` 的显式张量输入。

## 3. 接口规范

### 算子原型

```python
cann_bench.criss_cross_attention(
    x,
    conv_q_weight, conv_q_bias,
    conv_k_weight, conv_k_bias,
    conv_v_weight, conv_v_bias,
    gamma,
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图 `[B, C, H, W]` |
| conv_q_weight | Tensor | 必选 | query 1x1 投影卷积权重 `[C//8, C, 1, 1]` |
| conv_q_bias | Tensor | 必选 | query 1x1 投影卷积偏置 `[C//8]` |
| conv_k_weight | Tensor | 必选 | key 1x1 投影卷积权重 `[C//8, C, 1, 1]` |
| conv_k_bias | Tensor | 必选 | key 1x1 投影卷积偏置 `[C//8]` |
| conv_v_weight | Tensor | 必选 | value 1x1 投影卷积权重 `[C, C, 1, 1]` |
| conv_v_bias | Tensor | 必选 | value 1x1 投影卷积偏置 `[C]` |
| gamma | Tensor | 必选 | 注意力支路的可学习缩放标量 `[1]` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, C, H, W]`（与 x 相同） | 与输入相同 | Criss-Cross 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `x` 必须为 4 维 `[B, C, H, W]`；所有张量输入 dtype 必须一致。
- `C` 必须能被 8 整除，且 `C // 8 >= 1`；query/key 投影的通道数为 `C // 8`，value 投影为 `C`。
- 横向 energy 的对角线（同一空间位置自注意力）在 softmax 前置为 $-\infty$。
- `gamma` 为标量张量（shape `[1]`），output = `gamma * (o_H + o_W) + x`。
- golden 内部统一升 fp32 计算后回铸为输入 dtype；权重与 `gamma` 全部来自显式入参。

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
import torch.nn.functional as F

"""
CrissCrossAttention 算子 Torch Golden 参考实现

在特征图的每一行（横向）与每一列（纵向）上分别计算十字交叉注意力：
对每个空间位置，收集同一行的 W 个 key 与同一列的 H 个 key，拼接后 softmax，
再分别对 value 的同行/同列元素加权求和。两路结果相加后按标量 gamma 缩放并残差。
公式:
    Q = Conv1x1_q(x), K = Conv1x1_k(x), V = Conv1x1_v(x)
    e_H[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, j, x]，j == y 时置 -inf
    e_W[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, y, j]
    a = softmax(cat([e_H, e_W], dim=-1))
    o_H[b, c, y, x] = sum_j V[b, c, j, x] * a_H[b, y, x, j]
    o_W[b, c, y, x] = sum_j V[b, c, y, j] * a_W[b, y, x, j]
    y = gamma * (o_H + o_W) + x
"""


def criss_cross_attention(
    x: torch.Tensor,
    conv_q_weight: torch.Tensor,
    conv_q_bias: torch.Tensor,
    conv_k_weight: torch.Tensor,
    conv_k_bias: torch.Tensor,
    conv_v_weight: torch.Tensor,
    conv_v_bias: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Criss-Cross 注意力前向

    公式:
        Q = Conv1x1_q(x), K = Conv1x1_k(x), V = Conv1x1_v(x)
        e_H[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, j, x]，j == y 时置 -inf
        e_W[b, y, x, j] = sum_c Q[b, c, y, x] * K[b, c, y, j]
        a = softmax(cat([e_H, e_W], dim=-1))
        o_H[b, c, y, x] = sum_j V[b, c, j, x] * a_H[b, y, x, j]
        o_W[b, c, y, x] = sum_j V[b, c, y, j] * a_W[b, y, x, j]
        y = gamma * (o_H + o_W) + x

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_q_weight: query 1x1 投影卷积权重 [C//8, C, 1, 1]
        conv_q_bias: query 1x1 投影卷积偏置 [C//8]
        conv_k_weight: key 1x1 投影卷积权重 [C//8, C, 1, 1]
        conv_k_bias: key 1x1 投影卷积偏置 [C//8]
        conv_v_weight: value 1x1 投影卷积权重 [C, C, 1, 1]
        conv_v_bias: value 1x1 投影卷积偏置 [C]
        gamma: 注意力支路的缩放标量 [1]

    Returns:
        Criss-Cross 注意力输出 [B, C, H, W]
    """
    x_f = x.float()
    b, c, h, w = x_f.shape

    proj_q = F.conv2d(x_f, conv_q_weight.float(), conv_q_bias.float())
    proj_k = F.conv2d(x_f, conv_k_weight.float(), conv_k_bias.float())
    proj_v = F.conv2d(x_f, conv_v_weight.float(), conv_v_bias.float())

    # 横向能量：固定列，对所有行两两打分；纵向能量：固定行，对所有列两两打分。
    # 直接用 einsum 在 [B, C, H, W] 上完成，避免逐列/逐行的 permute+bmm 展开。
    energy_h = torch.einsum("bcyx,bcjx->byxj", proj_q, proj_k)
    energy_w = torch.einsum("bcyx,bcyj->byxj", proj_q, proj_k)

    # 横向自注意力位置（行下标相等）置 -inf，与源实现一致
    diag_mask = torch.eye(h, dtype=torch.bool, device=x_f.device).view(1, h, 1, h)
    energy_h = energy_h.masked_fill(diag_mask, float("-inf"))

    attn = F.softmax(torch.cat([energy_h, energy_w], dim=-1), dim=-1)
    attn_h = attn[..., :h]
    attn_w = attn[..., h:]

    out_h = torch.einsum("bcjx,byxj->bcyx", proj_v, attn_h)
    out_w = torch.einsum("bcyj,byxj->bcyx", proj_v, attn_w)
    out = gamma.float() * (out_h + out_w) + x_f
    return out.to(dtype=x.dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 1, 128, 5, 7
x = torch.randn(B, C, H, W, dtype=torch.float32, device="npu")
q_w = torch.randn(C // 8, C, 1, 1, dtype=torch.float32, device="npu") * 0.1
q_b = torch.randn(C // 8, dtype=torch.float32, device="npu") * 0.1
k_w = torch.randn(C // 8, C, 1, 1, dtype=torch.float32, device="npu") * 0.1
k_b = torch.randn(C // 8, dtype=torch.float32, device="npu") * 0.1
v_w = torch.randn(C, C, 1, 1, dtype=torch.float32, device="npu") * 0.1
v_b = torch.randn(C, dtype=torch.float32, device="npu") * 0.1
gamma = torch.tensor([0.3], dtype=torch.float32, device="npu")

y = cann_bench.criss_cross_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, gamma)
# y.shape == (1, 128, 5, 7)
```
