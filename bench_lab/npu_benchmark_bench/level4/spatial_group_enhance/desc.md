# SpatialGroupEnhance 算子 API 描述

## 1. 算子简介

Spatial Group Enhance (SGE) 是一个空间分组增强模块：把通道分成若干组，组内用全局
平均池化得到组特征，与每个空间位置特征逐元素相乘后求和，得到逐位置的重要度；再对每个
样本的 g 个组分别做标准化，经逐组仿射与 sigmoid 得到空间门控，乘回输入。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉主干网络中的空间注意力增强
- 需要以极低开销增强空间显著性的场合

**算子特征**：
- 难度等级：L4（VisionAttention）
- 3 个张量输入（输入特征图 + 2 个逐组仿射 Parameter）+ 1 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16，全程保持输入 dtype

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，分组数 $G$。分组展平：

$$
\mathbf{x}_r = \mathbf{x}.\mathrm{view}(BG,\ C/G,\ H,\ W)
$$

组内全局平均池化并与逐位置特征相乘求和：

$$
\mathbf{a} = \mathrm{GAP}(\mathbf{x}_r),\qquad
\mathbf{s} = \sum_{c=1}^{C/G} \mathbf{x}_r[:,c,:,:]\odot \mathbf{a}[:,c,:,:] \in \mathbb{R}^{BG\times H\times W}
$$

逐样本（g 个组一起）标准化与逐组仿射：

$$
\mathbf{t} = \frac{\mathbf{s} - \mathrm{mean}(\mathbf{s})}{\mathrm{std}(\mathbf{s}) + 10^{-5}},\qquad
\mathbf{g} = \mathbf{t}.\mathrm{view}(B,G,H,W)\odot \mathbf{w} + \mathbf{b}
$$

$$
\mathbf{y} = \mathbf{x}_r \odot \sigma\!\left(\mathbf{g}.\mathrm{view}(BG,1,H,W)\right)
$$

其中 `std` 使用 PyTorch 默认的无偏估计（unbiased=True），与源实现一致。

### 实现建议

1. 按 `groups` 把 `x` 视作 `[B*G, C//G, H, W]`，做一次 1x1 全局平均池化。
2. `xn = x_reshaped * avg_pool` 后沿通道维求和得到 `[B*G, 1, H, W]`，reshape 为
   `[B*G, H*W]`。
3. 在 `dim=1` 上去均值、除以 `std + 1e-5`，再 reshape 为 `[B, G, H, W]` 与
   `weight`/`bias`（形状 `[1, G, 1, 1]`）逐组仿射。
4. sigmoid 门控后乘回分组特征，最后 reshape 回 `[B, C, H, W]`；全程保持输入 dtype。

## 3. 接口规范

### 算子原型

```python
cann_bench.spatial_group_enhance(Tensor x, Tensor weight, Tensor bias, int groups) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必选 | 描述 |
|------|------|------|------|
| x | Tensor | 是 | 输入特征图 [B, C, H, W] |
| weight | Tensor | 是 | 逐组缩放参数 [1, groups, 1, 1] |
| bias | Tensor | 是 | 逐组偏移参数 [1, groups, 1, 1] |

### 属性说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| groups | int | - | 通道分组数，须整除 C |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, C, H, W] | 与输入相同 | 输出特征图 [B, C, H, W] |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有张量输入的 dtype 必须一致。
- x: [B, C, H, W]；weight/bias: [1, groups, 1, 1]；y: [B, C, H, W]；要求 groups 整除 C

## 4. 精度要求

- 输出与输入 `x` 同 shape、同 dtype；golden 全程保持输入 dtype。
- 标准化使用 `torch.std` 默认的无偏估计（unbiased=True），并加 1e-5 防止除零，与源实现一致。
- 评测按常规 fp16/bf16/fp32 阈值比较；golden 仅依赖显式权重输入，不含随机数。
- 与源 `Model` 逐 case 对比（weight/bias 直接取自 `_cache`），20 个 case 的 maxdiff 均为 0。

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
Spatial Group Enhance (SGE) 算子 Torch Golden 参考实现

按通道分组，组内用全局平均池化特征与逐空间位置特征做相似度，再对每个样本的
g 个组各自做标准化，经逐组仿射后 sigmoid 得到空间门控，乘回输入。

源实现全程保持输入 dtype（内部不升精度）；标准化使用有偏估计之外的
torch.std 默认行为（unbiased=True），与源实现一致。
"""


def spatial_group_enhance(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """
    Spatial Group Enhance 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        weight: 逐组缩放参数 [1, groups, 1, 1]
        bias: 逐组偏移参数 [1, groups, 1, 1]
        groups: 通道分组数，须整除 C

    Returns:
        增强后的特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape

    x_reshaped = x.view(b * groups, -1, h, w)           # (b*g, c//g, h, w)
    avg_pool = F.adaptive_avg_pool2d(x_reshaped, 1)     # (b*g, c//g, 1, 1)
    xn = x_reshaped * avg_pool                          # (b*g, c//g, h, w)
    xn = xn.sum(dim=1, keepdim=True)                    # (b*g, 1, h, w)
    t = xn.view(b * groups, -1)                         # (b*g, h*w)
    t = t - t.mean(dim=1, keepdim=True)
    std = t.std(dim=1, keepdim=True) + 1e-5
    t = t / std
    t = t.view(b, groups, h, w)                         # (b, g, h, w)
    t = t * weight + bias
    t = t.view(b * groups, 1, h, w)                     # (b*g, 1, h, w)
    out = x_reshaped * torch.sigmoid(t)
    return out.view(b, c, h, w)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W, G = 2, 256, 15, 15, 4

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
weight = torch.zeros(1, G, 1, 1, dtype=torch.float16, device="npu")
bias = torch.zeros(1, G, 1, 1, dtype=torch.float16, device="npu")

y = cann_bench.spatial_group_enhance(x, weight, bias, G)
# y: [2, 256, 15, 15], float16
```
