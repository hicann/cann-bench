# GCModule 算子 API 描述

## 1. 算子简介

GCModule 是 GCNet 的 Global Context 模块：用 1×1 卷积从输入特征图生成每个空间位置的全局注意力
logits，对全局上下文做 bottleneck（`Conv1x1 -> LayerNorm -> ReLU -> Conv1x1`）变换后与输入残差相加。
源实现虽然在 `__init__` 中定义了 softmax，但 `context_modeling` 从未调用它；本任务严格遵循源行为，
**不对 logits 做 softmax 归一化**。

**主要应用场景**：
- 图像分类、检测、分割主干网络中的全局上下文建模
- 需要以极低参数量增强长程依赖的视觉特征重标定

**算子特征**：
- 难度等级：L4（FusedComposite）
- 9 个必选张量输入（输入特征图 + 注意力卷积与 bottleneck 变换的全部权重/偏置）+ 2 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，bottleneck 宽度 $r=C/\text{reduction}$，
$\mathbf{x}_{\text{flat}}\in\mathbb{R}^{B\times C\times HW}$ 为空间展平。

$$
\mathbf{a} = \mathrm{Conv1x1}_{C\to1}(\mathbf{x}) \in \mathbb{R}^{B\times 1\times HW}
$$

$$
\mathbf{g} = \mathbf{x}_{\text{flat}}\,\mathbf{a}^{\top} \in \mathbb{R}^{B\times C\times 1}
$$

$$
\mathbf{y} = \mathbf{x} + \mathrm{Conv1x1}_{r\to C}\!\left(\mathrm{ReLU}\!\left(\mathrm{LN}\!\left(\mathrm{Conv1x1}_{C\to r}(\mathbf{g})\right)\right)\right)
$$

### 实现建议

1. `conv1_w/conv1_b` 做 $1\times1$ 卷积得到 `[B, 1, H, W]`，直接 reshape 为 `[B, 1, H*W]`，不做 softmax。
2. 与 `x` 的 `[B, C, H*W]` 做 batch matmul，得到 `[B, C, 1]`，再 unsqueeze 为 `[B, C, 1, 1]`。
3. bottleneck 两层 $1\times1$ 卷积之间用 `LayerNorm([C//reduction, 1, 1])` 和 ReLU。
4. 最后与 `x` 逐元素相加。

## 3. 接口规范

### 算子原型

```python
cann_bench.gc_module(
    x, conv1_w, conv1_b, t_w1, t_b1, ln_w, ln_b, t_w2, t_b2,
    channel, reduction
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `[B, C, H, W]` | 输入特征图 |
| `conv1_w` | `[1, C, 1, 1]` | 注意力 1×1 卷积权重 |
| `conv1_b` | `[1]` | 注意力 1×1 卷积偏置 |
| `t_w1` | `[C//reduction, C, 1, 1]` | bottleneck 第一层 1×1 卷积权重 |
| `t_b1` | `[C//reduction]` | bottleneck 第一层 1×1 卷积偏置 |
| `ln_w` | `[C//reduction, 1, 1]` | LayerNorm 仿射权重 |
| `ln_b` | `[C//reduction, 1, 1]` | LayerNorm 仿射偏置 |
| `t_w2` | `[C, C//reduction, 1, 1]` | bottleneck 第二层 1×1 卷积权重 |
| `t_b2` | `[C]` | bottleneck 第二层 1×1 卷积偏置 |

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| `y` | `[B, C, H, W]` | 输出特征图 |

### 数据类型

输入与输出均支持 `float16`、`float32`、`bfloat16`，且所有张量使用同一 dtype。

### 规则与约束

- `channel` 必须等于 `x.shape[1]`。
- `reduction` 必须整除 `channel`，且 `channel // reduction >= 1`。
- 源实现未对注意力 logits 做 softmax，实现时不得额外归一化。

## 4. 精度要求

golden 全程使用输入 dtype 计算，结果与 golden 的误差需满足 bench 的统一精度门限；
推荐以约 $10^{-3}$（fp16/bf16）相对误差为参考。

## 5. 标准 Golden 代码

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
GCModule 算子 Torch Golden 参考实现

GCNet Global Context 模块：
    attn    = conv1x1(x)                    # (B, 1, H, W) raw logits
    context = x_flat @ attn^T               # (B, C, 1, 1)
    y       = x + conv1x1(ReLU(LN(conv1x1(context))))

注意：源实现 __init__ 中定义了 softmax，但 context_modeling 从未调用，
本 golden 严格遵循源行为，不对 attn 做 softmax 归一化。
"""


def gc_module(
    x: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    t_w1: torch.Tensor,
    t_b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    t_w2: torch.Tensor,
    t_b2: torch.Tensor,
    channel: int,
    reduction: int,
) -> torch.Tensor:
    """
    GCNet Global Context 模块

    Args:
        x: 输入特征图 [B, C, H, W]
        conv1_w: 注意力 1x1 卷积权重 [1, C, 1, 1]
        conv1_b: 注意力 1x1 卷积偏置 [1]
        t_w1: Transform 第一层 1x1 卷积权重 [C//reduction, C, 1, 1]
        t_b1: Transform 第一层 1x1 卷积偏置 [C//reduction]
        ln_w: LayerNorm 仿射权重 [C//reduction, 1, 1]
        ln_b: LayerNorm 仿射偏置 [C//reduction, 1, 1]
        t_w2: Transform 第二层 1x1 卷积权重 [C, C//reduction, 1, 1]
        t_b2: Transform 第二层 1x1 卷积偏置 [C]
        channel: 输入通道数（须等于 C）
        reduction: bottleneck 缩减比

    Returns:
        输出特征图 [B, C, H, W]
    """
    batch, channels, height, width = x.shape

    attn = F.conv2d(x, conv1_w, conv1_b)
    b_att, _, h_att, w_att = attn.shape
    attn = attn.view(b_att, 1, h_att * w_att)

    x_flat = x.view(batch, channels, -1)
    context = torch.bmm(x_flat, attn.transpose(1, 2)).unsqueeze(-1)

    y = F.conv2d(context, t_w1, t_b1)
    y = F.layer_norm(y, y.shape[1:], ln_w, ln_b)
    y = F.relu(y, inplace=True)
    y = F.conv2d(y, t_w2, t_b2)

    return x + y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 512, 31, 11
reduction = 2
hidden = C // reduction

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
conv1_w = torch.randn(1, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv1_b = torch.zeros(1, dtype=torch.float16, device="npu")
t_w1 = torch.randn(hidden, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
t_b1 = torch.zeros(hidden, dtype=torch.float16, device="npu")
ln_w = torch.ones(hidden, 1, 1, dtype=torch.float16, device="npu")
ln_b = torch.zeros(hidden, 1, 1, dtype=torch.float16, device="npu")
t_w2 = torch.randn(C, hidden, 1, 1, dtype=torch.float16, device="npu") * 0.05
t_b2 = torch.zeros(C, dtype=torch.float16, device="npu")

y = cann_bench.gc_module(
    x, conv1_w, conv1_b, t_w1, t_b1, ln_w, ln_b, t_w2, t_b2,
    C, reduction,
)
# y: [2, 512, 31, 11], float16
```
