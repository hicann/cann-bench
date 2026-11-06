# SelfAttention 算子 API 描述

## 1. 算子简介

SelfAttention 是多头自注意力的完整前向实现（Q = K = V = x）：输入序列先经 Q/K/V 三组
线性投影，按头拆分后计算按 $\sqrt{d}$ 缩放的 softmax 注意力，合并多头后再经输出投影。
源实现内部用固定随机种子初始化的四个 `nn.Linear` 权重，本任务已全部提升为显式张量输入。

**主要应用场景**：
- Transformer 编码器/解码器的自注意力层
- 教学、精度对齐与算子融合测试中的标准 MHA 参考

**算子特征**：
- 难度等级：L4（Attention）
- 5 个必选张量输入（输入 + 四组投影权重）+ 1 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times L\times d}$，头数 $h$，$d_h = d/h$：

$$
\mathbf{q} = \mathbf{x}\mathbf{W}_q^{\top},\quad
\mathbf{k} = \mathbf{x}\mathbf{W}_k^{\top},\quad
\mathbf{v} = \mathbf{x}\mathbf{W}_v^{\top}
$$

$$
\mathbf{S} = \left(\frac{\mathbf{q}}{\sqrt{d_h}}\right)\mathbf{k}^{\top},\qquad
\mathbf{O} = \mathrm{softmax}(\mathbf{S})\,\mathbf{v}
$$

$$
\mathbf{y} = \mathrm{merge\_heads}(\mathbf{O})\,\mathbf{W}_o^{\top}
$$

其中 $\mathbf{W}_q,\mathbf{W}_k,\mathbf{W}_v,\mathbf{W}_o\in\mathbb{R}^{d\times d}$ 为显式权重输入，
softmax 沿最后一维进行，`merge_heads` 将 `[B, h, L, d_h]` 还原为 `[B, L, d]`。
全程以 fp32 计算，输出回铸为 `x` 的 dtype。

### 实现建议

1. 线性投影即 `x @ W.T`（等价于 `F.linear(x, W)`，无 bias）。
2. 按 query 行分块计算打分矩阵，避免一次物化 `[B, h, L, L]`。
3. softmax 在 fp32 中计算，注意力输出再与 $\mathbf{W}_o$ 相乘。
4. 源实现把缩放系数融合进 q（`q / sqrt(d_h)`），与对 scores 缩放数学等价。

## 3. 接口规范

### 算子原型

```python
cann_bench.self_attention(
    x, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight, n_heads
) -> y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列特征 `[B, L, d_model]` |
| q_proj_weight | Tensor | 必选 | query 投影权重 `[d_model, d_model]` |
| k_proj_weight | Tensor | 必选 | key 投影权重 `[d_model, d_model]` |
| v_proj_weight | Tensor | 必选 | value 投影权重 `[d_model, d_model]` |
| out_proj_weight | Tensor | 必选 | 输出投影权重 `[d_model, d_model]` |
| n_heads | int | 必选 | 注意力头数，须整除 `d_model` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, L, d_model]` | 与 `x` 相同 | 自注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `x` 必须为 3 维 `[B, L, d_model]`；`d_model % n_heads == 0`。
- 四个权重均为方阵 `[d_model, d_model]`，dtype 与 `x` 一致。
- 无权重的 bias 项（源 `nn.Linear(..., bias=False)`）。
- golden 以 fp32 执行投影、打分与 softmax，输出回铸为输入 dtype。

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
SelfAttention 算子 Torch Golden 参考实现

多头自注意力前向（Q=K=V=x）：Q/K/V 输入投影 + 按 sqrt(head_dim) 缩放的
softmax 注意力 + 输出投影。原实现内部随机初始化并缓存的四个 nn.Linear
权重已全部提升为显式张量入参。

公式:
    q = x @ Wq^T,  k = x @ Wk^T,  v = x @ Wv^T
    q, k, v = split_heads(q, k, v, n_heads)
    scores  = (q / sqrt(D)) @ k^T
    y       = softmax(scores, dim=-1) @ v
    y       = merge_heads(y) @ Wo^T
"""


def self_attention(
    x: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    SelfAttention 前向

    公式:
        q = x @ Wq^T,  k = x @ Wk^T,  v = x @ Wv^T
        q, k, v = split_heads(q, k, v, n_heads)
        scores  = (q / sqrt(D)) @ k^T
        y       = softmax(scores, dim=-1) @ v
        y       = merge_heads(y) @ Wo^T

    Args:
        x: 输入序列特征 [B, L, d_model]
        q_proj_weight: query 投影权重 [d_model, d_model]
        k_proj_weight: key 投影权重 [d_model, d_model]
        v_proj_weight: value 投影权重 [d_model, d_model]
        out_proj_weight: 输出投影权重 [d_model, d_model]
        n_heads: 注意力头数

    Returns:
        自注意力输出 [B, L, d_model]
    """
    heads = int(n_heads)
    batch, length, d_model = x.shape
    if d_model % heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = d_model // heads
    inv_scale = 1.0 / (head_dim ** 0.5)

    xf = x.float()
    q = torch.matmul(xf, q_proj_weight.float().transpose(-1, -2))
    k = torch.matmul(xf, k_proj_weight.float().transpose(-1, -2))
    v = torch.matmul(xf, v_proj_weight.float().transpose(-1, -2))

    q = q.view(batch, length, heads, head_dim).transpose(1, 2)
    k = k.view(batch, length, heads, head_dim).transpose(1, 2)
    v = v.view(batch, length, heads, head_dim).transpose(1, 2)

    out = torch.zeros(batch, heads, length, head_dim, dtype=torch.float32)
    # 按 query 行分块，避免一次物化 [B, N, L, L]
    budget = 2 ** 24
    blk = max(1, min(length, budget // max(1, batch * heads * length)))
    for start in range(0, length, blk):
        end = min(start + blk, length)
        scores = torch.matmul(q[:, :, start:end] * inv_scale, k.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, start:end] = torch.matmul(probs, v)

    out = out.transpose(1, 2).contiguous().view(batch, length, d_model)
    y = torch.matmul(out, out_proj_weight.float().transpose(-1, -2))
    return y.to(x.dtype)

```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, L, d_model, n_heads = 1, 1024, 512, 8
x = torch.randn(B, L, d_model, dtype=torch.float16, device="npu")
wq = torch.randn(d_model, d_model, dtype=torch.float16, device="npu") * 0.02
wk = torch.randn(d_model, d_model, dtype=torch.float16, device="npu") * 0.02
wv = torch.randn(d_model, d_model, dtype=torch.float16, device="npu") * 0.02
wo = torch.randn(d_model, d_model, dtype=torch.float16, device="npu") * 0.02

y = cann_bench.self_attention(x, wq, wk, wv, wo, n_heads)
# y: [1, 1024, 512], float16
```
