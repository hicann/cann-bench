# SKAttention 算子 API 描述

## 1. 算子简介

SKAttention（Selective Kernel Attention）通过多个不同核大小的卷积分支并行提取多尺度特征，
再把各分支输出求和并做全局平均池化，用两层全连接生成每个分支的通道注意力权重，沿**分支维度**
做 softmax 后对多分支特征加权融合。它让网络能够根据输入自适应地选择合适大小的感受野。

**主要应用场景**：
- 视觉主干网络（ResNet、MobileNet 等）中的选择性感受野增强
- 目标检测、语义分割等需要多尺度特征融合的任务

**算子特征**：
- 难度等级：L4（VisionAttention / FusedComposite）
- 9 个必选张量输入：输入特征图 + 6 个 `List[Tensor]`（每分支 conv 权重/偏置、BN affine、升维 fc 权重/偏置）+ 2 个单张量（降维 fc 权重/偏置）
- 5 个属性：`channel`、`kernels`（list）、`reduction`、`group`、`L`；`len(kernels)` 即分支数 K ∈ {2, 3, 4}，决定各 TensorList 的长度
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，卷积分支核大小列表
$\{k_i\}_{i=0}^{K-1}$，卷积分组数 $g$，瓶颈维度 $d=\max(L,\ C/\text{reduction})$。

分支特征（含 BatchNorm 训练态批统计）：

$$
\mathbf{U}_i = \mathrm{ReLU}\!\left(\mathrm{BN}_i\!\left(\mathrm{Conv}_{k_i,\,\text{groups}=g}(\mathbf{x})\right)\right),
\qquad i=0,\dots,K-1
$$

各分支求和并做全局平均池化：

$$
\mathbf{U} = \sum_{i=0}^{K-1}\mathbf{U}_i,\qquad
\mathbf{s} = \frac{1}{HW}\sum_{h,w}\mathbf{U}_{:,:,h,w}\in\mathbb{R}^{B\times C}
$$

降维全连接与逐分支升维全连接：

$$
\mathbf{z} = W_{fc}\,\mathbf{s} + b_{fc}\in\mathbb{R}^{B\times d},\qquad
\mathbf{w}_i = W^{i}_{fcs}\,\mathbf{z} + b^{i}_{fcs}\in\mathbb{R}^{B\times C}
$$

分支维 softmax 与加权融合：

$$
\mathbf{a} = \mathrm{softmax}_{i}\!\left(\mathbf{w}_i\right),\qquad
\mathbf{y} = \sum_{i=0}^{K-1}\mathbf{a}_i\odot\mathbf{U}_i
$$

### 实现建议

1. 第 i 个分支：`F.conv2d(x, conv_w[i], conv_b[i], padding=kernels[i]//2, groups=group)` → BatchNorm → ReLU。
2. **BatchNorm 必须使用训练态批统计**（源 Module 默认 `training=True`）：
   `F.batch_norm(y, None, None, bn_w[i], bn_b[i], training=True, momentum=0.1, eps=1e-5)`。
3. `U = sum(conv_outs)`；`S = U.mean(-1).mean(-1)`；`Z = F.linear(S, fc_w, fc_b)`。
4. 每个分支 `F.linear(Z, fcs_w[i], fcs_b[i]).view(B, C, 1, 1)`，`stack(dim=0)` 后在 `dim=0` 做 softmax，再与分支特征 `stack(dim=0)` 加权求和。

## 3. 接口规范

### 算子原型

```python
cann_bench.sk_attention(
    x, conv_w, conv_b, bn_w, bn_b, fc_w, fc_b, fcs_w, fcs_b,
    channel, kernels, reduction, group, L
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `x` | Tensor | `[B, C, H, W]` | 输入特征图 |
| `conv_w` | List[Tensor] | K 个 `[C, C//group, k_i, k_i]` | 各分支卷积权重 |
| `conv_b` | List[Tensor] | K 个 `[C]` | 各分支卷积偏置 |
| `bn_w` | List[Tensor] | K 个 `[C]` | 各分支 BatchNorm 仿射权重 |
| `bn_b` | List[Tensor] | K 个 `[C]` | 各分支 BatchNorm 仿射偏置 |
| `fc_w` | Tensor | `[d, C]` | 降维全连接权重 |
| `fc_b` | Tensor | `[d]` | 降维全连接偏置 |
| `fcs_w` | List[Tensor] | K 个 `[C, d]` | 各分支升维全连接权重 |
| `fcs_b` | List[Tensor] | K 个 `[C]` | 各分支升维全连接偏置 |

其中 `d = max(L, channel // reduction)`，K = len(kernels)。

### 属性

| 参数 | 类型 | 说明 |
|---|---|---|
| `channel` | int | 输入通道数（等于 x 的 C） |
| `kernels` | list[int] | K 个卷积分支的核大小列表，决定各 TensorList 的长度 |
| `reduction` | int | 全连接瓶颈的通道缩减比 |
| `group` | int | 卷积分组数，须整除 channel |
| `L` | int | 瓶颈维度下界，d = max(L, channel // reduction) |

### 输出

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `y` | `[B, C, H, W]`（与 x 相同） | 与输入相同 | 选择性感受野融合后的特征图 |

### 数据类型

`float16` / `float32` / `bfloat16`，所有张量输入与输出使用同一 dtype。

### 规则与约束

- `channel` 等于 `x.shape[1]`，且 `channel % group == 0`。
- 卷积 padding 为 `kernels[i] // 2`（same padding），本任务所有核均为奇数。
- BatchNorm 使用训练态批统计（不使用 running_mean/running_var），`momentum=0.1`、`eps=1e-5`。
- 所有 TensorList 的长度等于 `len(kernels)`。

## 4. 精度要求

- golden 全程使用输入 dtype 计算，输出 dtype 与输入一致；不含随机数，仅依赖显式权重输入。
- 计算链包含卷积、BatchNorm、全局池化、softmax 与加权求和；评测按 bench 统一的
  fp16/bf16/fp32 精度门限比较。

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
from typing import List

"""
SKAttention 算子 Torch Golden 参考实现

Selective Kernel Attention (SK Attention)：
    U_i = ReLU(BN_i(Conv2d(x, k_i, padding=k_i//2, groups=group)))   # K 个分支
    U   = sum_i U_i
    S   = mean(U, dim=(-1, -2))                                       # [B, C]
    Z   = fc(S)                                                       # [B, d]
    w_i = softmax_i(fc_i(Z))                                          # 沿分支维 softmax
    y   = sum_i (w_i * U_i)

说明：
- 源实现把 K 组 conv/BN 与 fc/fcs 缓存在 _cache 中；本 golden 用 List[Tensor]
  逐分支传入（TensorList 长度 = len(kernels)），数学上完全等价。
- BatchNorm 使用训练态批统计（源 Module 默认 training=True），
  momentum=0.1、eps=1e-5，与 nn.BatchNorm2d 默认值一致。
- 全程使用输入 dtype 计算，输出 dtype 与输入一致。
"""


def sk_attention(
    x: torch.Tensor,
    conv_w: List[torch.Tensor],
    conv_b: List[torch.Tensor],
    bn_w: List[torch.Tensor],
    bn_b: List[torch.Tensor],
    fc_w: torch.Tensor,
    fc_b: torch.Tensor,
    fcs_w: List[torch.Tensor],
    fcs_b: List[torch.Tensor],
    channel: int,
    kernels: list,
    reduction: int,
    group: int,
    L: int,
) -> torch.Tensor:
    """
    Selective Kernel Attention (SK Attention)

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_w: K 个卷积分支权重，第 i 个 [C, C//group, k_i, k_i]
        conv_b: K 个卷积分支偏置，每个 [C]
        bn_w/bn_b: K 个 BatchNorm 仿射权重/偏置，每个 [C]
        fc_w/fc_b: 降维全连接权重 [d, C] / 偏置 [d]，d = max(L, C//reduction)
        fcs_w/fcs_b: K 个升维全连接权重 [C, d] / 偏置 [C]
        channel: 输入通道数
        kernels: K 个卷积分支的核大小列表
        reduction: 全连接瓶颈通道缩减比
        group: 卷积分组数
        L: 瓶颈维度下界

    Returns:
        输出特征图 [B, C, H, W]，dtype 与 x 相同
    """
    b, c, h, w = x.shape
    if channel is None:
        channel = c
    d = max(L, channel // reduction)
    num_k = len(kernels)

    # ---------- Split ----------
    conv_outs = []
    for i in range(num_k):
        y = F.conv2d(x, conv_w[i], conv_b[i], padding=kernels[i] // 2, groups=group)
        y = F.batch_norm(
            y, None, None, bn_w[i], bn_b[i],
            training=True, momentum=0.1, eps=1e-5,
        )
        y = F.relu(y)
        conv_outs.append(y)
    feats = torch.stack(conv_outs, 0)  # (K, b, c, h, w)

    # ---------- Fuse ----------
    U = sum(conv_outs)  # (b, c, h, w)

    # ---------- Reduction ----------
    S = U.mean(-1).mean(-1)  # (b, c)
    Z = F.linear(S, fc_w, fc_b)  # (b, d)

    # ---------- Attention weights ----------
    weights = []
    for i in range(num_k):
        w = F.linear(Z, fcs_w[i], fcs_b[i]).view(b, c, 1, 1)  # (b, c, 1, 1)
        weights.append(w)
    attention_weights = torch.stack(weights, 0)  # (K, b, c, 1, 1)
    attention_weights = F.softmax(attention_weights, dim=0)  # 沿分支维 softmax

    # ---------- Fuse ----------
    V = (attention_weights * feats).sum(0)  # (b, c, h, w)
    return V
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 4, 128, 11, 7
channel, kernels, reduction, group, L = 128, [3, 5, 7], 16, 1, 32
d = max(L, channel // reduction)
K = len(kernels)

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
conv_w = [
    torch.randn(C, C // group, k, k, dtype=torch.float16, device="npu") * 0.05 for k in kernels
]
conv_b = [torch.zeros(C, dtype=torch.float16, device="npu") for _ in range(K)]
bn_w = [torch.ones(C, dtype=torch.float16, device="npu") for _ in range(K)]
bn_b = [torch.zeros(C, dtype=torch.float16, device="npu") for _ in range(K)]
fc_w = torch.randn(d, channel, dtype=torch.float16, device="npu") * 0.05
fc_b = torch.zeros(d, dtype=torch.float16, device="npu")
fcs_w = [torch.randn(channel, d, dtype=torch.float16, device="npu") * 0.05 for _ in range(K)]
fcs_b = [torch.zeros(channel, dtype=torch.float16, device="npu") for _ in range(K)]

y = cann_bench.sk_attention(
    x, conv_w, conv_b, bn_w, bn_b, fc_w, fc_b, fcs_w, fcs_b,
    channel, kernels, reduction, group, L,
)
# y: [4, 128, 11, 7], float16
```
