# TripletAttention 算子 API 描述

## 1. 算子简介

Triplet Attention 是一个三分支视觉注意力模块：三个分支分别在 (C,H)、(C,W)、(H,W)
维度上做 Z-Pool（通道维 mean/max 拼接），再经 k x k 卷积、BatchNorm（训练态 batch
统计）、ReLU 与 sigmoid 得到门控，乘回分支输入；三个分支输出求平均。

**主要应用场景**：
- 图像分类、目标检测等视觉主干网络中的跨维度注意力增强
- 需要在通道、高度、宽度两两交互上建模的场合

**算子特征**：
- 难度等级：L4（VisionAttention）
- 13 个张量输入（输入特征图 + 3 组 conv/BN 共 12 个张量）+ 1 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16，全程保持输入 dtype
- BatchNorm 固定为训练态（batch 统计，momentum=0.1，eps=1e-5）

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，卷积核大小 $k$（奇数），
$P=(k-1)/2$。对分支输入张量 $\mathbf{t}$ 定义：

$$
\mathrm{ZPool}(\mathbf{t}) = \mathrm{cat}\!\left[\mathrm{mean}_{c}(\mathbf{t}),\ \mathrm{max}_{c}(\mathbf{t})\right] \in \mathbb{R}^{B\times 2\times H'\times W'}
$$

$$
\mathrm{gate}(\mathbf{t}) = \mathbf{t}\odot \sigma\!\left(\mathrm{ReLU}\!\left(\mathrm{BN}_{\text{train}}\!\left(\mathrm{Conv}_{k}(\mathrm{ZPool}(\mathbf{t}), \text{padding}=P)\right)\right)\right)
$$

三个分支：

$$
\mathbf{y}_{ch} = \mathrm{gate}(\mathbf{x}^{(0,3,1,2)})^{(0,2,3,1)},\qquad
\mathbf{y}_{cw} = \mathrm{gate}(\mathbf{x}^{(0,2,1,3)})^{(0,2,1,3)},\qquad
\mathbf{y}_{hw} = \mathrm{gate}(\mathbf{x})
$$

$$
\mathbf{y} = \frac{\mathbf{y}_{ch} + \mathbf{y}_{cw} + \mathbf{y}_{hw}}{3}
$$

BatchNorm 使用训练态 batch 统计（`training=True, momentum=0.1, eps=1e-5`），
不依赖 running mean/var。

### 实现建议

1. 三个分支共享同一个 `gate` 逻辑，仅 permute 方式不同：
   (C,H) 分支 `permute(0,3,1,2)` / 逆 `permute(0,2,3,1)`；
   (C,W) 分支 `permute(0,2,1,3)`（自逆）；(H,W) 分支不 permute。
2. Z-Pool 沿 dim=1 取 mean 与 max，拼接成 2 通道后做 `F.conv2d(..., padding=(k-1)//2)`。
3. `F.batch_norm(y, None, None, bn_w, bn_b, training=True, momentum=0.1, eps=1e-5)`
   （或传入一次性的 zeros/ones running stats）复刻源实现的训练态 BN。
4. 经 ReLU、sigmoid 得到门控乘回分支输入，三分支求和后除以 3；全程保持输入 dtype。

## 3. 接口规范

### 算子原型

```python
cann_bench.triplet_attention(Tensor x, Tensor conv_ch_weight, Tensor conv_ch_bias, Tensor bn_ch_weight, Tensor bn_ch_bias, Tensor conv_cw_weight, Tensor conv_cw_bias, Tensor bn_cw_weight, Tensor bn_cw_bias, Tensor conv_hw_weight, Tensor conv_hw_bias, Tensor bn_hw_weight, Tensor bn_hw_bias, int kernel_size) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必选 | 描述 |
|------|------|------|------|
| x | Tensor | 是 | 输入特征图 [B, C, H, W] |
| conv_ch_weight | Tensor | 是 | (C,H) 分支卷积权重 [1, 2, kernel_size, kernel_size] |
| conv_ch_bias | Tensor | 是 | (C,H) 分支卷积偏置 [1] |
| bn_ch_weight | Tensor | 是 | (C,H) 分支 BatchNorm 仿射权重 [1] |
| bn_ch_bias | Tensor | 是 | (C,H) 分支 BatchNorm 仿射偏置 [1] |
| conv_cw_weight | Tensor | 是 | (C,W) 分支卷积权重 [1, 2, kernel_size, kernel_size] |
| conv_cw_bias | Tensor | 是 | (C,W) 分支卷积偏置 [1] |
| bn_cw_weight | Tensor | 是 | (C,W) 分支 BatchNorm 仿射权重 [1] |
| bn_cw_bias | Tensor | 是 | (C,W) 分支 BatchNorm 仿射偏置 [1] |
| conv_hw_weight | Tensor | 是 | (H,W) 分支卷积权重 [1, 2, kernel_size, kernel_size] |
| conv_hw_bias | Tensor | 是 | (H,W) 分支卷积偏置 [1] |
| bn_hw_weight | Tensor | 是 | (H,W) 分支 BatchNorm 仿射权重 [1] |
| bn_hw_bias | Tensor | 是 | (H,W) 分支 BatchNorm 仿射偏置 [1] |

### 属性说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| kernel_size | int | - | 三个分支卷积核大小（奇数），padding = (kernel_size - 1) // 2 |

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
- x: [B, C, H, W]；每分支 conv_weight: [1, 2, kernel_size, kernel_size]；conv_bias/bn_weight/bn_bias: [1]；y: [B, C, H, W]

## 4. 精度要求

- 输出与输入 `x` 同 shape、同 dtype；golden 全程保持输入 dtype。
- BatchNorm 必须使用训练态 batch 统计（不能用 running stats 的推理态），momentum=0.1、eps=1e-5。
- 评测按常规 fp16/bf16/fp32 阈值比较；golden 仅依赖显式权重输入，不含随机数。
- 与源 `Model.forward` 逐 case 对比（同一组 12 个张量按嵌套结构传入源实现），
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
Triplet Attention 算子 Torch Golden 参考实现

三个分支分别在 (C,H)、(C,W)、(H,W) 维度上做 Z-Pool（通道维 mean/max 拼接）与
k x k 卷积，BatchNorm 使用训练态（batch 统计，momentum=0.1，eps=1e-5），
经 ReLU 与 sigmoid 得到门控乘回分支输入，最后三分支求平均。

源实现全程保持输入 dtype（内部不升精度）。源 forward 的 gate_weights 为嵌套
结构，本 golden 将其拆成 12 个独立张量输入（每分支 conv 权重/偏置 + BN
仿射权重/偏置）。
"""


def triplet_attention(
    x: torch.Tensor,
    conv_ch_weight: torch.Tensor,
    conv_ch_bias: torch.Tensor,
    bn_ch_weight: torch.Tensor,
    bn_ch_bias: torch.Tensor,
    conv_cw_weight: torch.Tensor,
    conv_cw_bias: torch.Tensor,
    bn_cw_weight: torch.Tensor,
    bn_cw_bias: torch.Tensor,
    conv_hw_weight: torch.Tensor,
    conv_hw_bias: torch.Tensor,
    bn_hw_weight: torch.Tensor,
    bn_hw_bias: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Triplet Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_ch_weight: (C,H) 分支卷积权重 [1, 2, kernel_size, kernel_size]
        conv_ch_bias: (C,H) 分支卷积偏置 [1]
        bn_ch_weight: (C,H) 分支 BatchNorm 仿射权重 [1]
        bn_ch_bias: (C,H) 分支 BatchNorm 仿射偏置 [1]
        conv_cw_weight: (C,W) 分支卷积权重 [1, 2, kernel_size, kernel_size]
        conv_cw_bias: (C,W) 分支卷积偏置 [1]
        bn_cw_weight: (C,W) 分支 BatchNorm 仿射权重 [1]
        bn_cw_bias: (C,W) 分支 BatchNorm 仿射偏置 [1]
        conv_hw_weight: (H,W) 分支卷积权重 [1, 2, kernel_size, kernel_size]
        conv_hw_bias: (H,W) 分支卷积偏置 [1]
        bn_hw_weight: (H,W) 分支 BatchNorm 仿射权重 [1]
        bn_hw_bias: (H,W) 分支 BatchNorm 仿射偏置 [1]
        kernel_size: 卷积核大小（奇数），padding = (kernel_size - 1) // 2

    Returns:
        输出特征图 [B, C, H, W]
    """
    padding = (kernel_size - 1) // 2

    def gate_fn(t, conv_w, conv_b, bn_w, bn_b, permute=None, permute_back=None):
        if permute is not None:
            t = t.permute(permute)
        z = torch.cat([t.mean(dim=1, keepdim=True),
                       t.max(dim=1, keepdim=True)[0]], dim=1)
        y = F.conv2d(z, conv_w, conv_b, padding=padding)
        y = F.batch_norm(
            y, torch.zeros_like(bn_w), torch.ones_like(bn_w),
            bn_w, bn_b, training=True, momentum=0.1, eps=1e-5
        )
        y = F.relu(y)
        attn = torch.sigmoid(y)
        out = t * attn
        if permute_back is not None:
            out = out.permute(permute_back)
        return out

    x_ch = gate_fn(
        x, conv_ch_weight, conv_ch_bias, bn_ch_weight, bn_ch_bias,
        permute=(0, 3, 1, 2), permute_back=(0, 2, 3, 1)
    )
    x_cw = gate_fn(
        x, conv_cw_weight, conv_cw_bias, bn_cw_weight, bn_cw_bias,
        permute=(0, 2, 1, 3), permute_back=(0, 2, 1, 3)
    )
    x_hw = gate_fn(x, conv_hw_weight, conv_hw_bias, bn_hw_weight, bn_hw_bias)

    return (x_ch + x_cw + x_hw) / 3.0
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W, k = 2, 128, 9, 13, 7

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")

def make_branch():
    conv_w = torch.randn(1, 2, k, k, dtype=torch.float16, device="npu") * 0.05
    conv_b = torch.zeros(1, dtype=torch.float16, device="npu")
    bn_w = torch.ones(1, dtype=torch.float16, device="npu")
    bn_b = torch.zeros(1, dtype=torch.float16, device="npu")
    return conv_w, conv_b, bn_w, bn_b

ch = make_branch()
cw = make_branch()
hw = make_branch()

y = cann_bench.triplet_attention(
    x,
    ch[0], ch[1], ch[2], ch[3],
    cw[0], cw[1], cw[2], cw[3],
    hw[0], hw[1], hw[2], hw[3],
    k,
)
# y: [2, 128, 9, 13], float16
```
