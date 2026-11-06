# PSA 算子 API 描述

## 1. 算子简介

PSA（Pyramid Squeeze Attention）是一种多尺度通道注意力模块。它把输入通道均分为 S 组，每组用
逐组递增核大小的卷积提取不同感受野的多尺度特征（SPC，Split and Concolve）；随后每组经
全局平均池化与 SE 风格的两层 1x1 卷积生成通道权重（SPA，Squeeze and Attention），在**组维度**
做 softmax 归一化后对各组特征加权求和。相比逐通道 softmax，组维 softmax 让不同尺度的分支
相互竞争，实现自适应的多尺度选择。

**主要应用场景**：
- 视觉主干网络与检测/分割头中的多尺度通道注意力增强
- 需要以低参数量融合多感受野信息的特征重标定

**算子特征**：
- 难度等级：L4（VisionAttention / FusedComposite）
- 5 个必选张量输入：输入特征图 + 4 个 `List[Tensor]`（每组一卷积权重/偏置、每组 SE 两层 1x1 卷积权重）
- 3 个属性：`channel`、`reduction`、`S`；`S` 决定所有 TensorList 的长度（S ∈ {2, 4, 8}）
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，分组数 $S$，
组内通道数 $c_i=C/S$。把输入按通道切分为 $S$ 组：
$\mathbf{X}=[\mathbf{X}_0,\dots,\mathbf{X}_{S-1}]$，$\mathbf{X}_g\in\mathbb{R}^{B\times c_i\times H\times W}$。

第 $g$ 组（$g=0,\dots,S-1$）使用核大小 $k_g=2(g+1)+1$、padding $g+1$ 的卷积：

$$
\mathbf{U}_g = \mathrm{Conv}_{k_g}\!\left(\mathbf{X}_g\right)
$$

SE 权重（全局平均池化 + 两层无 bias 的 1x1 卷积）：

$$
\mathbf{s}_g = \sigma\!\left(W^{se2}_g\,\mathrm{ReLU}\!\left(W^{se1}_g\,\mathrm{GAP}(\mathbf{U}_g)\right)\right)
\in\mathbb{R}^{B\times c_i\times 1\times 1}
$$

组维 softmax 与加权求和：

$$
\mathbf{a} = \mathrm{softmax}_{g}\!\left(\mathbf{s}_g\right),\qquad
\mathbf{y} = \sum_{g=0}^{S-1}\mathbf{U}_g\odot\mathbf{a}_g
$$

### 实现建议

1. `x.view(B, S, c_i, H, W)` 后对第 g 组调用 `F.conv2d(·, conv_w[g], conv_b[g], padding=g+1)`；各卷积核大小由权重 shape 决定。
2. 注意 `view` 后先 `clone`，避免原地写回污染调用方输入（源实现已修复该问题）。
3. SE 部分：`adaptive_avg_pool2d(group, 1)` → `conv2d(se_w1[g])`（无 bias）→ `ReLU` → `conv2d(se_w2[g])`（无 bias）→ `Sigmoid`。
4. 把 S 个 `[B, c_i, 1, 1]` 权重 `stack(dim=1)` 后 `expand_as` 到 `[B, S, c_i, H, W]`，在 `dim=1` 做 softmax，再与 SPC 输出逐元素相乘并 `view(B, -1, H, W)`。

## 3. 接口规范

### 算子原型

```python
cann_bench.psa(x, conv_w, conv_b, se_w1, se_w2, channel, reduction, S) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `x` | Tensor | `[B, C, H, W]` | 输入特征图，C = S * ci |
| `conv_w` | List[Tensor] | S 个 `[ci, ci, 2(g+1)+1, 2(g+1)+1]` | 第 g 组的分组卷积权重 |
| `conv_b` | List[Tensor] | S 个 `[ci]` | 第 g 组的分组卷积偏置 |
| `se_w1` | List[Tensor] | S 个 `[ci//reduction, ci, 1, 1]` | SE 第一层 1x1 卷积权重（无 bias） |
| `se_w2` | List[Tensor] | S 个 `[ci, ci//reduction, 1, 1]` | SE 第二层 1x1 卷积权重（无 bias） |

### 属性

| 参数 | 类型 | 说明 |
|---|---|---|
| `channel` | int | 输入通道数参数，组内通道数 `ci = channel // S` |
| `reduction` | int | SE 通道缩减比，须整除 ci |
| `S` | int | 分组数，同时决定所有 TensorList 的长度 |

### 输出

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `y` | `[B, C, H, W]`（与 x 相同） | 与输入相同 | 多尺度注意力加权后的特征图 |

### 数据类型

`float16` / `float32` / `bfloat16`，所有张量输入与输出使用同一 dtype。

### 规则与约束

- `channel` 与 `x.shape[1]` 满足 `(channel // S) * S == x.shape[1]`（本任务用例中 channel 等于 x 的 C）。
- `ci = channel // S` 必须能被 `reduction` 整除且 `ci // reduction >= 1`。
- 第 g 组卷积核大小固定为 `2*(g+1)+1`、padding 为 `g+1`（same padding）。
- 所有 TensorList 的长度等于 `S`。

## 4. 精度要求

- golden 全程使用输入 dtype 计算，输出 dtype 与输入一致；不含随机数，仅依赖显式权重输入。
- 计算链为 分组卷积 → 全局池化 → 1x1 卷积 → softmax → 加权求和；评测按 bench 统一的
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
PSA 算子 Torch Golden 参考实现

Pyramid Squeeze Attention (PSA)：
    - SPC：把通道均分为 S 组，第 g 组用核大小 2(g+1)+1、padding g+1 的卷积
      （每组内 ci = channel // S 个通道独立卷积，相当于 depthwise 分组卷积）
    - SPA：每组经全局平均池化 + 两层 1x1 卷积（ReLU -> Sigmoid）得到通道权重，
      在组维度 softmax 后对各组特征加权求和

说明：
- 源实现把 S 组 conv / SE 权重缓存在 _cache 中；本 golden 用 List[Tensor]
  逐组传入（每个 TensorList 长度 = S），数学上完全等价。
- 全程使用输入 dtype 计算，输出 dtype 与输入一致。
"""


def psa(
    x: torch.Tensor,
    conv_w: List[torch.Tensor],
    conv_b: List[torch.Tensor],
    se_w1: List[torch.Tensor],
    se_w2: List[torch.Tensor],
    channel: int,
    reduction: int,
    S: int,
) -> torch.Tensor:
    """
    Pyramid Squeeze Attention (PSA)

    Args:
        x: 输入特征图 [B, C, H, W]，C = S * ci
        conv_w: S 组分组卷积权重，第 g 个为 [ci, ci, 2(g+1)+1, 2(g+1)+1]
        conv_b: S 组分组卷积偏置，每个 [ci]
        se_w1: S 组 SE 第一层 1x1 卷积权重（无 bias），每个 [ci//reduction, ci, 1, 1]
        se_w2: S 组 SE 第二层 1x1 卷积权重（无 bias），每个 [ci, ci//reduction, 1, 1]
        channel: 输入通道数参数（ci = channel // S）
        reduction: SE 通道缩减比
        S: 分组数，同时是各 TensorList 的长度

    Returns:
        输出特征图 [B, C, H, W]，dtype 与 x 相同
    """
    b, c, h, w = x.shape
    # 第 g 组的卷积核大小与 padding 由组号决定：k = 2*(g+1)+1, padding = g+1
    SPC_out = x.view(b, S, -1, h, w).clone()  # (b, S, ci, h, w)
    for idx in range(S):
        SPC_out[:, idx, :, :, :] = F.conv2d(
            SPC_out[:, idx, :, :, :], conv_w[idx], conv_b[idx], padding=idx + 1
        )

    # 每组 SE：全局平均池化 -> 1x1 conv -> ReLU -> 1x1 conv -> Sigmoid
    se_outs = []
    for idx in range(S):
        group = SPC_out[:, idx, :, :, :]          # (b, ci, h, w)
        pooled = F.adaptive_avg_pool2d(group, 1)  # (b, ci, 1, 1)
        t = F.conv2d(pooled, se_w1[idx])
        t = F.relu(t, inplace=True)
        t = F.conv2d(t, se_w2[idx])
        se_outs.append(torch.sigmoid(t))
    SE_out = torch.stack(se_outs, dim=1)          # (b, S, ci, 1, 1)
    SE_out = SE_out.expand_as(SPC_out)            # (b, S, ci, h, w)

    # 组维度 softmax + 加权求和
    softmax_out = F.softmax(SE_out, dim=1)
    PSA_out = SPC_out * softmax_out
    return PSA_out.view(b, -1, h, w)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 64, 15, 15
channel, reduction, S = 64, 4, 4
ci = channel // S

x = torch.randn(B, C, H, W, dtype=torch.bfloat16, device="npu")
conv_w = [
    torch.randn(ci, ci, 2 * (g + 1) + 1, 2 * (g + 1) + 1, dtype=torch.bfloat16, device="npu") * 0.05
    for g in range(S)
]
conv_b = [torch.zeros(ci, dtype=torch.bfloat16, device="npu") for _ in range(S)]
se_w1 = [
    torch.randn(ci // reduction, ci, 1, 1, dtype=torch.bfloat16, device="npu") * 0.05
    for _ in range(S)
]
se_w2 = [
    torch.randn(ci, ci // reduction, 1, 1, dtype=torch.bfloat16, device="npu") * 0.05
    for _ in range(S)
]

y = cann_bench.psa(x, conv_w, conv_b, se_w1, se_w2, channel, reduction, S)
# y: [2, 64, 15, 15], bfloat16
```
