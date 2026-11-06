# DAModule 算子 API 描述

## 1. 算子简介

DAModule（Dual Attention Module）是视觉领域的双重注意力模块：位置注意力分支把特征图展平成
`B × N × C`（`N = H*W`）后在空间位置之间做 self-attention，通道注意力分支把特征图展平成
`B × C × N` 后在通道之间做 self-attention，两路结果转回输入 dtype 后逐元素相加。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉主干网络中的特征增强
- 需要同时建模空间长程依赖与通道相关性的特征重标定

**算子特征**：
- 难度等级：L4（VisionAttention）
- 5 个必选张量输入（输入特征图 + 两个分支的卷积权重/偏置）+ 1 个可选属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，空间位置数 $N=H\cdot W$，
两个分支的卷积为 $\mathrm{Conv}_{pos}$、$\mathrm{Conv}_{ch}$（核大小 $K$，padding $K/2$）。

位置注意力：

$$
\mathbf{p} = \mathrm{flatten}_{N}\!\left(\mathrm{Conv}_{pos}(\mathbf{x})\right)^{\top}\in\mathbb{R}^{B\times N\times C},
\qquad
\mathbf{A}_{pos} = \mathrm{softmax}\!\left(\frac{\mathbf{p}_f\,\mathbf{p}_f^{\top}}{\sqrt{C}}\right)
$$

$$
\mathbf{o}_{pos} = \mathrm{reshape}_{H\times W}\!\left(\mathbf{A}_{pos}\,\mathbf{p}_f\right)^{\top}\in\mathbb{R}^{B\times C\times H\times W}
$$

通道注意力：

$$
\mathbf{q} = \mathrm{flatten}_{N}\!\left(\mathrm{Conv}_{ch}(\mathbf{x})\right)\in\mathbb{R}^{B\times C\times N},
\qquad
\mathbf{A}_{ch} = \mathrm{softmax}\!\left(\frac{\mathbf{q}_f\,\mathbf{q}_f^{\top}}{\sqrt{H\cdot W}}\right)
$$

$$
\mathbf{o}_{ch} = \mathrm{reshape}_{H\times W}\!\left(\mathbf{A}_{ch}\,\mathbf{q}_f\right)\in\mathbb{R}^{B\times C\times H\times W}
$$

最终输出：

$$
\mathbf{y} = \mathbf{o}_{pos} + \mathbf{o}_{ch}
$$

其中 $\mathbf{p}_f=\mathbf{p}.\mathrm{float}()$、$\mathbf{q}_f=\mathbf{q}.\mathrm{float}()$。

### 实现建议

1. 两个卷积均使用 `padding = kernel_size // 2`，$K$ 为奇数时输出空间尺寸不变。
2. 注意力分数必须在 fp32 下计算（源实现先 `.float()` 再 `bmm`），计算完成后
   `.to(x.dtype)` 转回输入 dtype。
3. 位置分支的注意力矩阵形状为 `[B, N, N]`，通道分支为 `[B, C, C]`；当 `N` 或 `C`
   较大时注意显存占用。
4. 两路输出都转回输入 dtype 后再相加，不要提前把结果留在 fp32。

## 3. 接口规范

### 算子原型

```python
cann_bench.da_module(
    x, conv_pos_w, conv_pos_b, conv_ch_w, conv_ch_b,
    kernel_size
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `[B, C, H, W]` | 输入特征图 |
| `conv_pos_w` | `[C, C, K, K]` | 位置注意力分支卷积权重 |
| `conv_pos_b` | `[C]` | 位置注意力分支卷积偏置 |
| `conv_ch_w` | `[C, C, K, K]` | 通道注意力分支卷积权重 |
| `conv_ch_b` | `[C]` | 通道注意力分支卷积偏置 |

### 属性

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `kernel_size` | int | 3 | 两个卷积的核大小，padding = kernel_size // 2 |

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| `y` | `[B, C, H, W]` | 输出特征图 |

### 数据类型

输入与输出均支持 `float16`、`float32`、`bfloat16`，且所有张量使用同一 dtype；
中间注意力分数使用 fp32。

### 规则与约束

- `conv_pos_w/conv_ch_w` 的第一维与第二维都必须等于 `x.shape[1]`（即 C）。
- `conv_pos_b/conv_ch_b` 的长度必须等于 C。
- 卷积输出空间尺寸须与输入一致，即 `kernel_size` 应为奇数。
- 禁止对注意力 logits 做额外的归一化或掩码，softmax 的维度为最后一维。

## 4. 精度要求

注意力分数在 fp32 下计算，输出转回输入 dtype；golden 全程复刻源实现的 dtype 流转。
结果与 golden 的误差需满足 bench 的统一精度门限，推荐以约 $10^{-3}$（fp16/bf16）相对误差为参考。

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
DAModule 算子 Torch Golden 参考实现

Dual Attention Module：位置注意力分支与通道注意力分支并行计算后逐元素相加。

    y_pos    = conv_pos(x).view(B, C, -1).permute(0, 2, 1)           # (B, N, C)
    attn_pos = softmax(bmm(y_pos.float(), y_pos.float()^T) * C^-0.5)
    out_pos  = bmm(attn_pos, y_pos.float()) -> (B, C, H, W) -> x.dtype

    y_ch     = conv_ch(x).view(B, C, -1)                             # (B, C, N)
    attn_ch  = softmax(bmm(y_ch.float(), y_ch.float()^T) * (H*W)^-0.5)
    out_ch   = bmm(attn_ch, y_ch.float()) -> (B, C, H, W) -> x.dtype

    y = out_pos + out_ch

dtype 策略与源实现一致：两个分支的注意力分数都先上转 fp32 计算，
结果转回输入 dtype 后再相加；所有输入输出使用同一 dtype。
"""


def da_module(
    x: torch.Tensor,
    conv_pos_w: torch.Tensor,
    conv_pos_b: torch.Tensor,
    conv_ch_w: torch.Tensor,
    conv_ch_b: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Dual Attention Module 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_pos_w: 位置注意力分支卷积权重 [C, C, K, K]
        conv_pos_b: 位置注意力分支卷积偏置 [C]
        conv_ch_w: 通道注意力分支卷积权重 [C, C, K, K]
        conv_ch_b: 通道注意力分支卷积偏置 [C]
        kernel_size: 卷积核大小，padding 取 kernel_size // 2

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    padding = kernel_size // 2

    # Position attention
    y_pos = F.conv2d(x, conv_pos_w, conv_pos_b, padding=padding).view(b, c, -1).permute(0, 2, 1)
    y_pos_f = y_pos.float()
    attn_pos = F.softmax(torch.bmm(y_pos_f, y_pos_f.transpose(1, 2)) * (c ** -0.5), dim=-1)
    out_pos = torch.bmm(attn_pos, y_pos_f).permute(0, 2, 1).view(b, c, h, w).to(x.dtype)

    # Channel attention
    y_ch = F.conv2d(x, conv_ch_w, conv_ch_b, padding=padding).view(b, c, -1)
    y_ch_f = y_ch.float()
    attn_ch = F.softmax(torch.bmm(y_ch_f, y_ch_f.transpose(1, 2)) * ((h * w) ** -0.5), dim=-1)
    out_ch = torch.bmm(attn_ch, y_ch_f).view(b, c, h, w).to(x.dtype)

    return out_pos + out_ch
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 1, 64, 23, 23
K = 3

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
conv_pos_w = torch.randn(C, C, K, K, dtype=torch.float16, device="npu") * 0.05
conv_pos_b = torch.zeros(C, dtype=torch.float16, device="npu")
conv_ch_w = torch.randn(C, C, K, K, dtype=torch.float16, device="npu") * 0.05
conv_ch_b = torch.zeros(C, dtype=torch.float16, device="npu")

y = cann_bench.da_module(x, conv_pos_w, conv_pos_b, conv_ch_w, conv_ch_b, K)
# y: [1, 64, 23, 23], float16
```
