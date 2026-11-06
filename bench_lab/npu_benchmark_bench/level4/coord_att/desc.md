# CoordAtt 算子 API 描述

## 1. 算子简介

CoordAtt（Coordinate Attention）把通道注意力分解为两个一维的坐标方向注意力：
分别沿高度方向（每列）与宽度方向（每行）做全局池化，得到两组带位置信息的通道描述；
将它们在空间维拼接后经 1x1 卷积 + BatchNorm（batch 统计）+ h_swish 融合，
再 split 回 H/W 两路，各经 1x1 卷积 + sigmoid 生成坐标注意力权重，
最终以 `x * a_w * a_h` 的形式广播回输入。

**主要应用场景**：
- 移动端视觉主干网络（MobileNet/EdgeNeXt 等）中的轻量注意力
- 需要保留长程空间位置信息的检测、分割任务

**算子特征**：
- 难度等级：L4（FusedComposite）
- 9 个必选张量输入（输入特征图 + 三组 1x1 卷积的权重/偏置 + BN 参数）+ 3 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，中间通道数 $mip=\max(8, C/\text{reduction})$。

$$
\mathbf{x}^{h} = \mathrm{AdaptiveAvgPool}(\mathbf{x}, (\text{None}, 1)) \in \mathbb{R}^{B\times C\times H\times 1},
\qquad
\mathbf{x}^{w} = \mathrm{AdaptiveAvgPool}(\mathbf{x}, (1, \text{None})) \in \mathbb{R}^{B\times C\times 1\times W}
$$

$$
\mathbf{z} = \mathrm{hswish}\Big(\mathrm{BN}\big(\mathrm{Conv}_{1\times1}(\mathrm{concat}([\mathbf{x}^{h}, \mathbf{x}^{w}], \text{axis}=2))\big)\Big)
$$

$$
\mathbf{x}^{h}, \mathbf{x}^{w} = \mathrm{split}(\mathbf{z}, [H, W], \text{axis}=2),
\qquad
\mathbf{a}^{h} = \sigma(\mathrm{Conv}_{1\times1}(\mathbf{x}^{h})),
\qquad
\mathbf{a}^{w} = \sigma(\mathrm{Conv}_{1\times1}(\mathbf{x}^{w}))
$$

$$
\mathbf{y} = \mathbf{x}\odot\mathbf{a}^{w}\odot\mathbf{a}^{h}
$$

其中 $\mathrm{hswish}(v) = v\cdot\frac{\mathrm{ReLU6}(v+3)}{6}$，BN 使用当前 batch 的统计量
（`training=True`，`momentum=0.1`，`eps=1e-5`）。

### 实现建议

1. H 方向的池化输出形状为 `[B, C, H, 1]`；W 方向池化输出为 `[B, C, 1, W]` 后转置为 `[B, C, W, 1]`，二者沿第 2 维拼接得到 `[B, C, H+W, 1]`。
2. 融合卷积为 1x1 卷积（可用 einsum 实现），随后 BatchNorm 使用 batch 统计（不走 running stats）。
3. split 后 H 分支保持 `[B, C, H, 1]`；W 分支为 `[B, C, W, 1]`，需转置回 `[B, C, 1, W]` 再与输入广播相乘。
4. 输出为 `x * a_w * a_h`，两个注意力均为 sigmoid 输出，取值范围 $(0, 1)$。

## 3. 接口规范

### 算子原型

```python
cann_bench.coord_att(
    x, conv1_weight, conv1_bias, bn_weight, bn_bias,
    conv_h_weight, conv_h_bias, conv_w_weight, conv_w_bias,
    inp, oup, reduction=32
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape `[B, C, H, W]` |
| conv1_weight | Tensor | 必选 | 融合支路 1x1 卷积权重，shape `[mip, C, 1, 1]` |
| conv1_bias | Tensor | 必选 | 融合支路 1x1 卷积偏置，shape `[mip]` |
| bn_weight | Tensor | 必选 | BatchNorm 缩放参数，shape `[mip]` |
| bn_bias | Tensor | 必选 | BatchNorm 偏移参数，shape `[mip]` |
| conv_h_weight | Tensor | 必选 | H 方向注意力 1x1 卷积权重，shape `[C, mip, 1, 1]` |
| conv_h_bias | Tensor | 必选 | H 方向注意力 1x1 卷积偏置，shape `[C]` |
| conv_w_weight | Tensor | 必选 | W 方向注意力 1x1 卷积权重，shape `[C, mip, 1, 1]` |
| conv_w_bias | Tensor | 必选 | W 方向注意力 1x1 卷积偏置，shape `[C]` |
| inp | int | 必选 | 输入通道数，保留用于接口兼容，实际通道数由 `x.shape[1]` 推导 |
| oup | int | 必选 | 输出通道数，保留用于接口兼容，不参与计算 |
| reduction | int | 32 | 中间通道降维比，`mip = max(8, C // reduction)` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, C, H, W]`（与 x 相同） | 与输入相同 | 坐标注意力加权后的特征图 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- `x` 必须为 4 维 `[B, C, H, W]`；所有张量输入的 dtype 必须一致。
- `inp` / `oup` 仅保留接口兼容（源用例中可与 `x.shape[1]` 不同），实际通道数取自 `x.shape[1]`，中间通道 `mip = max(8, C // reduction)`。
- BatchNorm 固定使用 batch 统计量（`training=True`，`momentum=0.1`，`eps=1e-5`），不使用 running stats。
- 融合支路输出在 split 时以 `[H, W]` 切分，故 `conv1_weight` 的输出通道必须是 `mip`。

## 4. 精度要求

- 源参考实现的 1x1 卷积使用显式 einsum；本 golden 保持与输入相同的浮点精度执行（评测框架内部以 fp64 运行），不做额外的 fp32 下采，从而得到真正的数学真值。
- 输出为 `x` 与两个 $(0,1)$ 注意力权重的乘积，数值范围不超过输入的绝对值上界。

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
CoordAtt 算子 Torch Golden 参考实现

Coordinate Attention 前向：分别沿 H、W 方向做全局池化，拼接后经 1x1 卷积 +
BatchNorm（batch 统计）+ h_swish 得到中间特征，再 split 成 H/W 两路各经 1x1
卷积 + sigmoid 得到坐标注意力权重，与输入逐元素相乘。

公式:
    x_h = adaptive_avg_pool2d(x, (None, 1))
    x_w = adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
    y = h_swish(BN(Conv1x1(cat([x_h, x_w], dim=2))))
    x_h, x_w = split(y, [H, W], dim=2)
    a_h = sigmoid(Conv1x1(x_h))
    a_w = sigmoid(Conv1x1(x_w.permute(0, 1, 3, 2)))
    y = x * a_w * a_h
"""


def _conv1x1(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """1x1 卷积的显式 einsum 实现，与源参考实现一致。"""
    weight = weight.squeeze(-1).squeeze(-1)
    out = torch.einsum("oc,bchw->bohw", weight, x)
    return out + bias.view(1, -1, 1, 1)


def coord_att(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    conv_h_weight: torch.Tensor,
    conv_h_bias: torch.Tensor,
    conv_w_weight: torch.Tensor,
    conv_w_bias: torch.Tensor,
    inp: int,
    oup: int,
    reduction: int = 32,
) -> torch.Tensor:
    """
    Coordinate Attention 前向

    公式:
        x_h = adaptive_avg_pool2d(x, (None, 1))
        x_w = adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        y = h_swish(BN(Conv1x1(cat([x_h, x_w], dim=2))))
        x_h, x_w = split(y, [H, W], dim=2)
        a_h = sigmoid(Conv1x1(x_h))
        a_w = sigmoid(Conv1x1(x_w.permute(0, 1, 3, 2)))
        y = x * a_w * a_h

    Args:
        x: 输入特征图 [B, C, H, W]
        conv1_weight: 融合支路 1x1 卷积权重 [mip, C, 1, 1]
        conv1_bias: 融合支路 1x1 卷积偏置 [mip]
        bn_weight: BatchNorm 缩放参数 [mip]
        bn_bias: BatchNorm 偏移参数 [mip]
        conv_h_weight: H 方向注意力 1x1 卷积权重 [C, mip, 1, 1]
        conv_h_bias: H 方向注意力 1x1 卷积偏置 [C]
        conv_w_weight: W 方向注意力 1x1 卷积权重 [C, mip, 1, 1]
        conv_w_bias: W 方向注意力 1x1 卷积偏置 [C]
        inp: 输入通道数，保留用于接口兼容，实际通道数由 x.shape[1] 推导
        oup: 输出通道数，保留用于接口兼容，不参与计算
        reduction: 中间通道降维比，mip = max(8, C // reduction)

    Returns:
        坐标注意力加权后的特征图 [B, C, H, W]
    """
    identity = x
    b, c, h, w = x.shape

    x_h = F.adaptive_avg_pool2d(x, (None, 1))
    x_w = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)

    y = torch.cat([x_h, x_w], dim=2)
    y = _conv1x1(y, conv1_weight, conv1_bias)
    y = F.batch_norm(
        y,
        torch.zeros_like(bn_weight),
        torch.ones_like(bn_weight),
        bn_weight,
        bn_bias,
        training=True,
        momentum=0.1,
        eps=1e-5,
    )
    y = y * (F.relu6(y + 3) / 6)

    x_h, x_w = torch.split(y, [h, w], dim=2)
    x_w = x_w.permute(0, 1, 3, 2)

    a_h = torch.sigmoid(_conv1x1(x_h, conv_h_weight, conv_h_bias))
    a_w = torch.sigmoid(_conv1x1(x_w, conv_w_weight, conv_w_bias))

    return identity * a_w * a_h

```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 96, 9, 7
reduction = 16
mip = max(8, C // reduction)

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
conv1_weight = torch.randn(mip, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv1_bias = torch.randn(mip, dtype=torch.float16, device="npu") * 0.05
bn_weight = torch.rand(mip, dtype=torch.float16, device="npu") * 0.2 + 0.9
bn_bias = torch.randn(mip, dtype=torch.float16, device="npu") * 0.1
conv_h_weight = torch.randn(C, mip, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv_h_bias = torch.randn(C, dtype=torch.float16, device="npu") * 0.05
conv_w_weight = torch.randn(C, mip, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv_w_bias = torch.randn(C, dtype=torch.float16, device="npu") * 0.05

y = cann_bench.coord_att(x, conv1_weight, conv1_bias, bn_weight, bn_bias,
                         conv_h_weight, conv_h_bias, conv_w_weight, conv_w_bias,
                         C, C, reduction)
# y: [2, 96, 9, 7], float16
```
