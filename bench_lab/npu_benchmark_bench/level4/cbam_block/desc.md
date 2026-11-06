# CBAMBlock 算子 API 描述

## 1. 算子简介

CBAM（Convolutional Block Attention Module）依次串联通道注意力与空间注意力：
通道支路分别对全局平均池化与全局最大池化的通道描述经**共享 MLP**（两层无 bias 的 1x1 卷积），
两路结果相加后 sigmoid 得到通道权重；空间支路对通道加权后的特征沿通道维取均值与最大值并拼接，
经一个 2->1 的 2D 卷积 + sigmoid 得到空间权重；最后与原始输入做残差相加。

**主要应用场景**：
- 视觉主干网络（ResNet/ConvNeXt 等）中的即插即用注意力增强
- 检测、分割、细粒度分类等任务的特征重标定

**算子特征**：
- 难度等级：L4（FusedComposite）
- 5 个必选张量输入（输入特征图 + 通道/空间注意力全部权重与偏置）+ 3 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，$\mathrm{Avg}_c$、$\mathrm{Max}_c$ 分别为沿空间维的全局平均池化与全局最大池化，
$f$ 为共享的两层 MLP（1x1 卷积 + ReLU，无 bias）。

通道注意力：

$$
\mathbf{a}^{c} = \sigma\!\Big(f(\mathrm{Avg}_c(\mathbf{x})) + f(\mathrm{Max}_c(\mathbf{x}))\Big),
\qquad
\mathbf{u} = \mathbf{x}\odot\mathbf{a}^{c}
$$

空间注意力：

$$
\mathbf{a}^{s} = \sigma\!\left(\mathrm{Conv2d}_{2\to1}\!\left(\mathrm{concat}\big(\mathrm{Max}_c^{ch}(\mathbf{u}),\ \mathrm{Mean}_c^{ch}(\mathbf{u})\big)\right)\right),
\qquad
\mathbf{y} = \mathbf{u}\odot\mathbf{a}^{s} + \mathbf{x}
$$

其中 $\mathrm{Mean}_c^{ch}$ / $\mathrm{Max}_c^{ch}$ 为沿通道维求均值/最大值（keepdim），
$\mathrm{Conv2d}_{2\to1}$ 的核大小为 `kernel_size`、padding 为 `kernel_size // 2`、含 bias。

### 实现建议

1. 通道支路两次共享同一组权重（`ca_fc1_weight`、`ca_fc2_weight`），注意 avg/max 两路必须复用同一 MLP。
2. 空间支路的通道统计沿 `dim=1` 计算，拼接顺序为 **max 在前、mean 在后**（与源实现一致）。
3. 输出需加回原始输入 `x` 形成残差。

## 3. 接口规范

### 算子原型

```python
cann_bench.cbam_block(
    x, ca_fc1_weight, ca_fc2_weight, sa_conv_weight, sa_conv_bias,
    channel, reduction=16, kernel_size=7
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape `[B, C, H, W]` |
| ca_fc1_weight | Tensor | 必选 | 通道注意力第一层 1x1 卷积权重，shape `[hidden, C, 1, 1]`（无 bias） |
| ca_fc2_weight | Tensor | 必选 | 通道注意力第二层 1x1 卷积权重，shape `[C, hidden, 1, 1]`（无 bias） |
| sa_conv_weight | Tensor | 必选 | 空间注意力 2->1 卷积权重，shape `[1, 2, kernel_size, kernel_size]` |
| sa_conv_bias | Tensor | 必选 | 空间注意力 2->1 卷积偏置，shape `[1]` |
| channel | int | 必选 | 输入通道数，保留用于接口兼容，实际通道数取自 `x.shape[1]` |
| reduction | int | 16 | 通道注意力 MLP 的瓶颈降维比 |
| kernel_size | int | 7 | 空间注意力卷积核大小 |

其中 `hidden = max(1, C // reduction)`。

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, C, H, W]`（与 x 相同） | 与输入相同 | 注意力细化后的特征图 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- `x` 必须为 4 维 `[B, C, H, W]`；所有张量输入的 dtype 必须一致。
- `hidden = max(1, C // reduction)`：当 `C < reduction` 时瓶颈宽度取 1（不得为 0）。
- `channel` 仅保留接口兼容，实际通道数取自 `x.shape[1]`；`ca_fc1_weight` / `ca_fc2_weight` 无 bias。
- 空间注意力卷积 padding 固定为 `kernel_size // 2`，`kernel_size` 建议为奇数以保证输出与输入同尺寸。

## 4. 精度要求

- 输出为残差形式 `u * sigmoid(...) + x`，与输入同量级；评测按常规 fp16/bf16/fp32 阈值比较。
- golden 全程保持输入 dtype 语义，不含随机数，仅依赖显式权重输入。

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
CBAMBlock 算子 Torch Golden 参考实现

Convolutional Block Attention Module (CBAM) 前向：通道注意力（avg-pool 与
max-pool 两路共享 MLP）与空间注意力（通道 mean/max 拼接后 2->1 卷积）顺序
加权，最后加残差。

公式:
    ca = sigmoid(MLP(avgpool(x)) + MLP(maxpool(x)))
    out = x * ca
    sa = sigmoid(Conv2d(cat([max_c(out), mean_c(out)])))
    y = out * sa + x
"""


def cbam_block(
    x: torch.Tensor,
    ca_fc1_weight: torch.Tensor,
    ca_fc2_weight: torch.Tensor,
    sa_conv_weight: torch.Tensor,
    sa_conv_bias: torch.Tensor,
    channel: int,
    reduction: int = 16,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    CBAM 卷积块注意力模块前向

    公式:
        ca = sigmoid(MLP(avgpool(x)) + MLP(maxpool(x)))
        out = x * ca
        sa = sigmoid(Conv2d(cat([max_c(out), mean_c(out)])))
        y = out * sa + x

    Args:
        x: 输入特征图 [B, C, H, W]
        ca_fc1_weight: 通道注意力第一层 1x1 卷积权重 [C // reduction, C, 1, 1]（无 bias）
        ca_fc2_weight: 通道注意力第二层 1x1 卷积权重 [C, C // reduction, 1, 1]（无 bias）
        sa_conv_weight: 空间注意力 2->1 卷积权重 [1, 2, kernel_size, kernel_size]
        sa_conv_bias: 空间注意力 2->1 卷积偏置 [1]
        channel: 输入通道数，保留用于接口兼容，实际通道数取自 x.shape[1]
        reduction: 通道注意力 MLP 的瓶颈降维比
        kernel_size: 空间注意力卷积核大小

    Returns:
        注意力细化后的特征图 [B, C, H, W]
    """
    residual = x

    # ---------- 通道注意力 ----------
    avg_p = F.adaptive_avg_pool2d(x, 1)
    max_p = F.adaptive_max_pool2d(x, 1)

    def shared_mlp(v):
        v = F.conv2d(v, ca_fc1_weight)
        v = F.relu(v)
        return F.conv2d(v, ca_fc2_weight)

    channel_att = torch.sigmoid(shared_mlp(avg_p) + shared_mlp(max_p))
    out = x * channel_att

    # ---------- 空间注意力 ----------
    avg_out = torch.mean(out, dim=1, keepdim=True)
    max_out = torch.max(out, dim=1, keepdim=True)[0]
    spatial_in = torch.cat([max_out, avg_out], dim=1)
    spatial_att = torch.sigmoid(
        F.conv2d(spatial_in, sa_conv_weight, sa_conv_bias, padding=kernel_size // 2)
    )
    out = out * spatial_att

    # ---------- 残差 ----------
    return out + residual

```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 32, 11, 7
reduction, kernel_size = 4, 5
hidden = max(1, C // reduction)

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
ca_fc1_weight = torch.randn(hidden, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
ca_fc2_weight = torch.randn(C, hidden, 1, 1, dtype=torch.float16, device="npu") * 0.05
sa_conv_weight = torch.randn(1, 2, kernel_size, kernel_size, dtype=torch.float16, device="npu") * 0.05
sa_conv_bias = torch.randn(1, dtype=torch.float16, device="npu") * 0.05

y = cann_bench.cbam_block(x, ca_fc1_weight, ca_fc2_weight, sa_conv_weight, sa_conv_bias,
                          C, reduction, kernel_size)
# y: [2, 32, 11, 7], float16
```
