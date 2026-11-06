# BAM 算子 API 描述

## 1. 算子简介

BAM（Bottleneck Attention Module）是一个轻量的卷积注意力模块，串接在任意卷积主干网络之后。
它并行计算**通道注意力**与**空间注意力**：通道支路对全局平均池化后的通道描述做瓶颈 MLP，
空间支路用 1x1 卷积降维后接两层扩张卷积（dilated conv）扩大感受野，再升维为单通道；
两路注意力相加后经 sigmoid 得到逐元素门控，并通过残差连接叠加回输入。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉主干网络中的注意力增强
- 需要同时建模通道重要性与空间重要性的特征重标定

**算子特征**：
- 难度等级：L4（FusedComposite）
- 13 个必选张量输入（输入特征图 + 通道/空间两路全部权重与偏置）+ 4 个属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，瓶颈宽度 $h=C/\text{reduction}$，
全局平均池化 $\mathrm{GAP}(\mathbf{x})=\frac{1}{HW}\sum_{h,w}\mathbf{x}_{:,:,h,w}$。

通道注意力：

$$
\mathbf{c}_{b,:} = W_2\,\mathrm{ReLU}\!\left(W_1\,\mathrm{GAP}(\mathbf{x})_{b,:} + b_1\right) + b_2
\in \mathbb{R}^{C}
$$

空间注意力（$\mathrm{Conv}^{(0)}_{1\times1}$ 为降维 1x1 卷积，$\mathrm{Conv}^{(1)}_{k,d}$、$\mathrm{Conv}^{(2)}_{k,d}$ 为核大小 $k$、扩张率 $d$ 的扩张卷积，$\mathrm{Conv}^{(3)}_{1\times1}$ 为升维 1x1 卷积）：

$$
\mathbf{s}' = \mathrm{Conv}^{(3)}_{1\times1}\!\left(\mathrm{ReLU}\!\left(\mathrm{Conv}^{(2)}_{k,d}\!\left(\mathrm{ReLU}\!\left(\mathrm{Conv}^{(1)}_{k,d}\!\left(\mathrm{Conv}^{(0)}_{1\times1}(\mathbf{x})\right)\right)\right)\right)\right),
\qquad
\mathbf{s} = \mathrm{AdaptiveAvgPool}_{H\times W}(\mathbf{s}')
$$

融合与残差：

$$
\mathbf{y} = \mathbf{x} + \mathbf{x}\odot\sigma\!\left(\mathbf{c} \oplus \mathbf{s}\right)
$$

其中 $\oplus$ 表示通道/空间广播相加，$\sigma$ 为 sigmoid。

### 实现建议

1. 通道支路：对 `x` 做全局平均池化得到 `[B, C]`，依次做 `Linear(C -> C/reduction) + ReLU` 与 `Linear(C/reduction -> C)`，再 reshape 为 `[B, C, 1, 1]`。
2. 空间支路：`1x1 conv (C -> C/reduction)` → 两次 `ReLU + dilated conv (k, dilation=d, padding=d)` → `1x1 conv (C/reduction -> 1)`，再用自适应平均池化恢复至 `[B, 1, H, W]`（当 padding 使卷积输出尺寸偏离 $H\times W$ 时由该池化对齐）。
3. 两路广播相加后 sigmoid，与 `x` 相乘并与 `x` 相加。
4. 收缩核逻辑（见规则与约束）需在 host 侧按 `x` 的 `H, W` 预先确定 `keff/deff`，卷积权重 shape 已由 cases 按同一规则给出。

## 3. 接口规范

### 算子原型

```python
cann_bench.bam(
    x, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
    conv1_weight, conv1_bias, conv2a_weight, conv2a_bias,
    conv2b_weight, conv2b_bias, conv3_weight, conv3_bias,
    channel, reduction=16, kernel_size=3, dilation_val=4
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape `[B, C, H, W]` |
| mlp_fc1_weight | Tensor | 必选 | 通道 MLP 第一层权重，shape `[C // reduction, C]` |
| mlp_fc1_bias | Tensor | 必选 | 通道 MLP 第一层偏置，shape `[C // reduction]` |
| mlp_fc2_weight | Tensor | 必选 | 通道 MLP 第二层权重，shape `[C, C // reduction]` |
| mlp_fc2_bias | Tensor | 必选 | 通道 MLP 第二层偏置，shape `[C]` |
| conv1_weight | Tensor | 必选 | 空间支路 1x1 降维卷积权重，shape `[C // reduction, C, 1, 1]` |
| conv1_bias | Tensor | 必选 | 空间支路 1x1 降维卷积偏置，shape `[C // reduction]` |
| conv2a_weight | Tensor | 必选 | 第一层扩张卷积权重，shape `[C // reduction, C // reduction, keff, keff]` |
| conv2a_bias | Tensor | 必选 | 第一层扩张卷积偏置，shape `[C // reduction]` |
| conv2b_weight | Tensor | 必选 | 第二层扩张卷积权重，shape `[C // reduction, C // reduction, keff, keff]` |
| conv2b_bias | Tensor | 必选 | 第二层扩张卷积偏置，shape `[C // reduction]` |
| conv3_weight | Tensor | 必选 | 空间支路 1x1 升维卷积权重，shape `[1, C // reduction, 1, 1]` |
| conv3_bias | Tensor | 必选 | 空间支路 1x1 升维卷积偏置，shape `[1]` |
| channel | int | 必选 | 输入通道数 C，须等于 `x.shape[1]`，用于推导权重形状 |
| reduction | int | 16 | 通道瓶颈降维比，`C // reduction` 为瓶颈宽度 |
| kernel_size | int | 3 | 空间支路扩张卷积核大小 |
| dilation_val | int | 4 | 空间支路扩张卷积的扩张率 |

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
- `channel` 必须等于 `x.shape[1]`；`hidden = channel // reduction` 必须为正整数。
- 扩张卷积的 padding 等于 `dilation_val`（与源实现一致）。
- **收缩核逻辑**：令 `eff = dilation_val * (kernel_size - 1) + 1`、`m = min(H, W)`。当 `eff > m` 时，源实现会收缩核与扩张率：
  - 若 `m <= 1`，取 `keff = 1, deff = 1`；
  - 否则 `deff = min(dilation_val, m)`，`keff = min(kernel_size, (m - 1) // deff + 1)`，若 `keff > 1` 且为偶数则 `keff -= 1`。
  未触发时 `keff = kernel_size, deff = dilation_val`。`conv2a_weight`/`conv2b_weight` 的 `keff` 必须按此规则推导。
- 空间支路的卷积输出尺寸可能与 `[H, W]` 不同，最终由 `adaptive_avg_pool2d` 对齐到 `[H, W]`。

## 4. 精度要求

- 输出为残差形式 `x + x * sigmoid(...)`，与输入同量级；评测按常规 fp16/bf16/fp32 阈值比较。
- golden 全程保持输入 dtype 语义（框架内部以高精度执行），不含随机数，仅依赖显式权重输入。

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
BAM 算子 Torch Golden 参考实现

Bottleneck Attention Module (BAM) 前向：通道注意力（全局平均池化 + 两层 MLP）
与空间注意力（1x1 降维卷积 + 两层扩张卷积 + 1x1 升维卷积）相加后经 sigmoid
得到注意力权重，并做残差连接。

公式:
    c_attn = MLP(avgpool(x))
    s_attn = Conv1x1(Conv_kd(ReLU(Conv_kd(ReLU(Conv1x1(x))))))
    s_attn = adaptive_avg_pool2d(s_attn, (H, W))
    y = x + x * sigmoid(c_attn + s_attn)
"""


def bam(
    x: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2a_weight: torch.Tensor,
    conv2a_bias: torch.Tensor,
    conv2b_weight: torch.Tensor,
    conv2b_bias: torch.Tensor,
    conv3_weight: torch.Tensor,
    conv3_bias: torch.Tensor,
    channel: int,
    reduction: int = 16,
    kernel_size: int = 3,
    dilation_val: int = 4,
) -> torch.Tensor:
    """
    BAM 瓶颈注意力模块前向

    公式:
        c_attn = MLP(avgpool(x))
        s_attn = Conv1x1(Conv_kd(ReLU(Conv_kd(ReLU(Conv1x1(x))))))
        s_attn = adaptive_avg_pool2d(s_attn, (H, W))
        y = x + x * sigmoid(c_attn + s_attn)

    Args:
        x: 输入特征图 [B, C, H, W]
        mlp_fc1_weight: 通道 MLP 第一层权重 [C // reduction, C]
        mlp_fc1_bias: 通道 MLP 第一层偏置 [C // reduction]
        mlp_fc2_weight: 通道 MLP 第二层权重 [C, C // reduction]
        mlp_fc2_bias: 通道 MLP 第二层偏置 [C]
        conv1_weight: 空间支路 1x1 降维卷积权重 [C // reduction, C, 1, 1]
        conv1_bias: 空间支路 1x1 降维卷积偏置 [C // reduction]
        conv2a_weight: 第一层扩张卷积权重 [C // reduction, C // reduction, keff, keff]
        conv2a_bias: 第一层扩张卷积偏置 [C // reduction]
        conv2b_weight: 第二层扩张卷积权重 [C // reduction, C // reduction, keff, keff]
        conv2b_bias: 第二层扩张卷积偏置 [C // reduction]
        conv3_weight: 空间支路 1x1 升维卷积权重 [1, C // reduction, 1, 1]
        conv3_bias: 空间支路 1x1 升维卷积偏置 [1]
        channel: 输入通道数，保留用于接口兼容，须等于 x.shape[1]
        reduction: 通道瓶颈降维比
        kernel_size: 扩张卷积核大小
        dilation_val: 扩张卷积的扩张率

    Returns:
        注意力细化后的特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape

    # 与源实现一致：扩张卷积的等效核超出最小空间维时收缩核与扩张率
    eff = dilation_val * (kernel_size - 1) + 1
    min_size = min(h, w)
    if eff > min_size:
        if min_size <= 1:
            safe_dilation = 1
            safe_kernel = 1
        else:
            safe_dilation = min(dilation_val, min_size)
            safe_kernel = min(kernel_size, (min_size - 1) // safe_dilation + 1)
            if safe_kernel > 1 and safe_kernel % 2 == 0:
                safe_kernel -= 1
        kernel_size = safe_kernel
        dilation_val = safe_dilation

    # ---------- 通道注意力 ----------
    y_ch = F.adaptive_avg_pool2d(x, 1).reshape(b, c)
    y_ch = F.linear(y_ch, mlp_fc1_weight, mlp_fc1_bias)
    y_ch = F.relu(y_ch)
    y_ch = F.linear(y_ch, mlp_fc2_weight, mlp_fc2_bias).view(b, c, 1, 1)

    # ---------- 空间注意力 ----------
    y_sp = F.conv2d(x, conv1_weight, conv1_bias)
    y_sp = F.relu(F.conv2d(y_sp, conv2a_weight, conv2a_bias, padding=dilation_val, dilation=dilation_val))
    y_sp = F.relu(F.conv2d(y_sp, conv2b_weight, conv2b_bias, padding=dilation_val, dilation=dilation_val))
    y_sp = F.conv2d(y_sp, conv3_weight, conv3_bias)
    y_sp = F.adaptive_avg_pool2d(y_sp, (h, w))

    # ---------- 融合 + 残差 ----------
    attn = torch.sigmoid(y_ch + y_sp)
    return x + x * attn

```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 128, 19, 21
reduction, kernel_size, dilation_val = 16, 3, 4
hidden = C // reduction

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
mlp_fc1_weight = torch.randn(hidden, C, dtype=torch.float16, device="npu") * 0.05
mlp_fc1_bias = torch.randn(hidden, dtype=torch.float16, device="npu") * 0.05
mlp_fc2_weight = torch.randn(C, hidden, dtype=torch.float16, device="npu") * 0.05
mlp_fc2_bias = torch.randn(C, dtype=torch.float16, device="npu") * 0.05
conv1_weight = torch.randn(hidden, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv1_bias = torch.randn(hidden, dtype=torch.float16, device="npu") * 0.05
conv2a_weight = torch.randn(hidden, hidden, kernel_size, kernel_size, dtype=torch.float16, device="npu") * 0.05
conv2a_bias = torch.randn(hidden, dtype=torch.float16, device="npu") * 0.05
conv2b_weight = torch.randn(hidden, hidden, kernel_size, kernel_size, dtype=torch.float16, device="npu") * 0.05
conv2b_bias = torch.randn(hidden, dtype=torch.float16, device="npu") * 0.05
conv3_weight = torch.randn(1, hidden, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv3_bias = torch.randn(1, dtype=torch.float16, device="npu") * 0.05

y = cann_bench.bam(
    x, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
    conv1_weight, conv1_bias, conv2a_weight, conv2a_bias,
    conv2b_weight, conv2b_bias, conv3_weight, conv3_bias,
    C, reduction, kernel_size, dilation_val,
)
# y: [2, 128, 19, 21], float16
```
