# ShuffleAttention 算子 API 描述

## 1. 算子简介

Shuffle Attention 是一个轻量视觉注意力模块，将通道划分为 G 组，每组通道再一分为二：
前半子特征做**通道注意力**（全局平均池化 + 逐组 1x1 仿射 + sigmoid），后半子特征做
**空间注意力**（GroupNorm + 逐组 1x1 仿射 + sigmoid）。两个分支拼接后按 2 组做
channel shuffle，恢复跨组信息交互。

**主要应用场景**：
- 图像分类、目标检测等视觉主干网络中的高效注意力增强
- 需要在通道与空间两个维度同时建模、且计算开销较低的场合

**算子特征**：
- 难度等级：L4（VisionAttention）
- 7 个张量输入（输入特征图 + GroupNorm affine + 4 个逐组仿射 Parameter）+ 3 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16，全程保持输入 dtype

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，分组数 $G$，记
$P=C/(2G)$。先把通道按 $G$ 组切分并展平为 $[BG, 2P, H, W]$，再沿通道一分为二：

$$
\mathbf{x}_0,\mathbf{x}_1 = \mathrm{chunk}(\mathbf{x}, 2, \dim=1),\qquad \mathbf{x}_i\in\mathbb{R}^{BG\times P\times H\times W}
$$

通道注意力分支：

$$
\mathbf{a} = \sigma\!\left(\mathbf{w}_c \odot \mathrm{GAP}(\mathbf{x}_0) + \mathbf{b}_c\right),\qquad
\mathbf{y}_0 = \mathbf{x}_0 \odot \mathbf{a}
$$

空间注意力分支（GroupNorm 组数为 $P$，即每组 1 个通道）：

$$
\mathbf{s} = \sigma\!\left(\mathbf{w}_s \odot \mathrm{GroupNorm}(\mathbf{x}_1) + \mathbf{b}_s\right),\qquad
\mathbf{y}_1 = \mathbf{x}_1 \odot \mathbf{s}
$$

拼接并按 2 组做 channel shuffle：

$$
\mathbf{y} = \mathrm{Shuffle}_{2}\!\left(\mathrm{cat}([\mathbf{y}_0,\mathbf{y}_1], \dim=1)\right)
$$

channel shuffle 等价于 `reshape(B, 2, C/2, H, W) -> permute(0,2,1,3,4) -> reshape(B, C, H, W)`。

### 实现建议

1. 输入 `x` 先 `view(B*G, -1, H, W)`，再 `chunk(2, dim=1)` 得到两个子特征。
2. 通道分支：`adaptive_avg_pool2d` 到 1x1，与 `channel_weight`/`channel_bias`（形状均为
   `[1, P, 1, 1]`）做广播乘加，sigmoid 后乘回 `x_0`。
3. 空间分支：`F.group_norm(x_1, P, gn_weight, gn_bias, eps=1e-5)`，再与
   `spatial_weight`/`spatial_bias` 广播乘加，sigmoid 后乘回 `x_1`。
4. 拼接后做 `reshape -> permute -> reshape` 的 channel shuffle；`reduction` 属性不参与计算。

## 3. 接口规范

### 算子原型

```python
cann_bench.shuffle_attention(Tensor x, Tensor gn_weight, Tensor gn_bias, Tensor channel_weight, Tensor channel_bias, Tensor spatial_weight, Tensor spatial_bias, int channel, int reduction, int G) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必选 | 描述 |
|------|------|------|------|
| x | Tensor | 是 | 输入特征图 [B, C, H, W] |
| gn_weight | Tensor | 是 | GroupNorm 仿射权重 [channel // (2G)] |
| gn_bias | Tensor | 是 | GroupNorm 仿射偏置 [channel // (2G)] |
| channel_weight | Tensor | 是 | 通道注意力缩放参数 [1, channel // (2G), 1, 1] |
| channel_bias | Tensor | 是 | 通道注意力偏移参数 [1, channel // (2G), 1, 1] |
| spatial_weight | Tensor | 是 | 空间注意力缩放参数 [1, channel // (2G), 1, 1] |
| spatial_bias | Tensor | 是 | 空间注意力偏移参数 [1, channel // (2G), 1, 1] |

### 属性说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| channel | int | - | 通道数属性，决定 GroupNorm 组数 channel // (2 * G)，须使该值等于 x 的 C // (2 * G) |
| reduction | int | 16 | 接口兼容属性，源实现未使用 |
| G | int | 8 | 分组数，须整除 C 且 C // G 为偶数 |

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
- x: [B, C, H, W]；P = channel // (2 * G)；gn_weight/gn_bias: [P]；channel_weight/channel_bias/spatial_weight/spatial_bias: [1, P, 1, 1]；y: [B, C, H, W]；要求 G 整除 C 且 C // G 为偶数

## 4. 精度要求

- 输出与输入 `x` 同 shape、同 dtype；golden 全程保持输入 dtype。
- GroupNorm 以 eps=1e-5、组数 `channel // (2G)`（每组 1 通道）计算，与源 `nn.GroupNorm` 一致。
- 评测按常规 fp16/bf16/fp32 阈值比较；golden 仅依赖显式权重输入，不含随机数。
- 与源 `Model` 逐 case 对比（GroupNorm affine 与 4 个 Parameter 直接取自 `_cache`），
  20 个 case 的 maxdiff 均为 0。

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
Shuffle Attention 算子 Torch Golden 参考实现

将通道按 G 组切分，每组再一分为二：前半做通道注意力（全局平均池化 + 1x1 个
Parameter 仿射 + sigmoid），后半做空间注意力（GroupNorm + 1x1 个 Parameter
仿射 + sigmoid），拼接后按 2 组做 channel shuffle。

源实现全程保持输入 dtype（内部不升精度）。GroupNorm 的组数取 channel // (2 * G)，
与每个子特征的通道数一致。
"""


def shuffle_attention(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    channel_weight: torch.Tensor,
    channel_bias: torch.Tensor,
    spatial_weight: torch.Tensor,
    spatial_bias: torch.Tensor,
    channel: int,
    reduction: int = 16,
    G: int = 8,
) -> torch.Tensor:
    """
    Shuffle Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        gn_weight: GroupNorm 仿射权重 [C // (2 * G)]
        gn_bias: GroupNorm 仿射偏置 [C // (2 * G)]
        channel_weight: 通道注意力缩放参数 [1, C // (2 * G), 1, 1]
        channel_bias: 通道注意力偏移参数 [1, C // (2 * G), 1, 1]
        spatial_weight: 空间注意力缩放参数 [1, C // (2 * G), 1, 1]
        spatial_bias: 空间注意力偏移参数 [1, C // (2 * G), 1, 1]
        channel: 通道数属性（接口兼容，实际通道数取自 x.shape[1]）
        reduction: 接口兼容参数，源实现未使用
        G: 分组数，须整除 C 且 C // G 为偶数

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    gn_groups = channel // (2 * G)

    # 分组：每组的通道再一分为二
    x = x.view(b * G, -1, h, w)
    x_0, x_1 = x.chunk(2, dim=1)

    # ---------- 通道注意力分支 ----------
    avg_pool = F.adaptive_avg_pool2d(x_0, 1)
    x_channel = channel_weight * avg_pool + channel_bias
    x_channel = x_0 * torch.sigmoid(x_channel)

    # ---------- 空间注意力分支 ----------
    x_spatial = F.group_norm(x_1, gn_groups, gn_weight, gn_bias, eps=1e-5)
    x_spatial = spatial_weight * x_spatial + spatial_bias
    x_spatial = x_1 * torch.sigmoid(x_spatial)

    # 拼接并按 2 组做 channel shuffle
    out = torch.cat([x_channel, x_spatial], dim=1)
    out = out.contiguous().view(b, -1, h, w)
    out = out.reshape(b, 2, -1, h, w)
    out = out.permute(0, 2, 1, 3, 4)
    return out.reshape(b, -1, h, w)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 256, 9, 13
G = 8
P = C // (2 * G)

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
gn_weight = torch.ones(P, dtype=torch.float16, device="npu")
gn_bias = torch.zeros(P, dtype=torch.float16, device="npu")
channel_weight = torch.zeros(1, P, 1, 1, dtype=torch.float16, device="npu")
channel_bias = torch.ones(1, P, 1, 1, dtype=torch.float16, device="npu")
spatial_weight = torch.zeros(1, P, 1, 1, dtype=torch.float16, device="npu")
spatial_bias = torch.ones(1, P, 1, 1, dtype=torch.float16, device="npu")

y = cann_bench.shuffle_attention(
    x, gn_weight, gn_bias, channel_weight, channel_bias,
    spatial_weight, spatial_bias, C, 16, G,
)
# y: [2, 256, 9, 13], float16
```
