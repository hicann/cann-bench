# DoubleAttention 算子 API 描述

## 1. 算子简介

DoubleAttention（Double Attention Network）用全局描述子建模二阶特征交互：先由 1×1 卷积分别
得到汇聚特征 $A$、注意力分布 $B$ 与值特征 $V$，再通过两次 batch matmul 计算
`C × c_n` 大小的全局描述子并与值特征相乘，最后经 1×1 卷积重建回输入通道维度。

**主要应用场景**：
- 视觉分类/检测主干中的全局二阶特征交互
- 需要以较小计算代价捕获长程通道相关性的注意力模块

**算子特征**：
- 难度等级：L4（VisionAttention）
- 9 个必选张量输入（输入特征图 + 4 个 1×1 卷积的权重与偏置），无属性
- 单输出，shape 与输入 `x` 一致；`B/C/H/W` 任一为 0 时原样返回输入
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，$N=H\cdot W$，
压缩通道数 $c_n=\max(1, C/8)$，四个 1×1 卷积分别为 $\mathrm{Conv}_A$、$\mathrm{Conv}_B$、
$\mathrm{Conv}_V$（输出 $c_n$ 通道）与 $\mathrm{Conv}_Z$（输出 $C$ 通道）。

$$
\mathbf{A} = \mathrm{flat}_N(\mathrm{Conv}_A(\mathbf{x}))\in\mathbb{R}^{B\times C\times N}
$$

$$
\mathbf{B} = \mathrm{softmax}_{N}\!\left(\mathrm{flat}_N(\mathrm{Conv}_B(\mathbf{x}))\right)\in\mathbb{R}^{B\times c_n\times N},
\qquad
\mathbf{V} = \mathrm{softmax}_{N}\!\left(\mathrm{flat}_N(\mathrm{Conv}_V(\mathbf{x}))\right)\in\mathbb{R}^{B\times c_n\times N}
$$

$$
\mathbf{G} = \mathbf{A}\,\mathbf{B}^{\top}\in\mathbb{R}^{B\times C\times c_n},
\qquad
\mathbf{Z} = \mathbf{G}\,\mathbf{V}\in\mathbb{R}^{B\times C\times N}
$$

$$
\mathbf{y} = \mathrm{Conv}_Z\!\left(\mathrm{reshape}_{H\times W}(\mathbf{Z})\right)\in\mathbb{R}^{B\times C\times H\times W}
$$

其中 softmax 沿最后一维（空间维 $N$）归一化。

### 实现建议

1. 四个卷积均为核大小 1、stride 1、无 padding，`c_n = max(1, C // 8)`。
2. `B`、`V` 的 softmax 沿空间维 $N$ 做归一化。
3. 源实现全程使用输入 dtype 计算，softmax 不做 fp32 上转，golden 需保持一致。
4. 源 `forward` 声明了 `a` / `v` 两个外部特征参数但从未参与计算，本任务将其省略。
5. 源实现在 `B/C/H/W` 任一为 0 时直接返回输入，实现需保持该提前返回语义。

## 3. 接口规范

### 算子原型

```python
cann_bench.double_attention(
    x, conv_a_w, conv_a_b, conv_b_w, conv_b_b,
    conv_v_w, conv_v_b, conv_z_w, conv_z_b
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `[B, C, H, W]` | 输入特征图 |
| `conv_a_w` | `[C, C, 1, 1]` | 汇聚卷积 A 权重 |
| `conv_a_b` | `[C]` | 汇聚卷积 A 偏置 |
| `conv_b_w` | `[c_n, C, 1, 1]` | 注意力卷积 B 权重 |
| `conv_b_b` | `[c_n]` | 注意力卷积 B 偏置 |
| `conv_v_w` | `[c_n, C, 1, 1]` | 注意力卷积 V 权重 |
| `conv_v_b` | `[c_n]` | 注意力卷积 V 偏置 |
| `conv_z_w` | `[C, C, 1, 1]` | 重建卷积 Z 权重 |
| `conv_z_b` | `[C]` | 重建卷积 Z 偏置 |

其中 `c_n = max(1, C // 8)`。

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| `y` | `[B, C, H, W]` | 输出特征图 |

### 数据类型

输入与输出均支持 `float16`、`float32`、`bfloat16`，且所有张量使用同一 dtype。

### 规则与约束

- `conv_b_w/conv_v_w` 的第一维必须等于 `c_n = max(1, C // 8)`，第二维必须等于 C。
- 四个卷积的 bias 均存在（源实现使用 `nn.Conv2d` 默认 bias=True）。
- `B/C/H/W` 任一为 0 时输出应为原输入，不得报错。
- 源 forward 的 `a` / `v` 参数未使用，接口不提供这两个输入。

## 4. 精度要求

golden 全程使用输入 dtype 计算；结果与 golden 的误差需满足 bench 的统一精度门限，
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
DoubleAttention 算子 Torch Golden 参考实现

Double Attention Network：用全局描述子做二阶特征交互。

    A  = convA(x).view(B, C, N)                       # C = c_m
    B  = softmax(convB(x).view(B, c_n, N), dim=-1)    # c_n = max(1, C // 8)
    V  = softmax(convV(x).view(B, c_n, N), dim=-1)
    G  = bmm(A, B^T)                                  # (B, C, c_n)
    Z  = bmm(G, V).view(B, C, H, W)
    y  = convZ(Z)

源实现 forward 的 a / v 外部参数实际未参与计算，本 golden 省略；
源实现全程使用输入 dtype 计算（softmax 也不做 fp32 上转），本 golden 保持一致。
b / c / h / w 任一为 0 时源实现直接返回输入，本 golden 同样提前返回。
"""


def double_attention(
    x: torch.Tensor,
    conv_a_w: torch.Tensor,
    conv_a_b: torch.Tensor,
    conv_b_w: torch.Tensor,
    conv_b_b: torch.Tensor,
    conv_v_w: torch.Tensor,
    conv_v_b: torch.Tensor,
    conv_z_w: torch.Tensor,
    conv_z_b: torch.Tensor,
) -> torch.Tensor:
    """
    Double Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        conv_a_w: 汇聚卷积 A 权重 [C, C, 1, 1]
        conv_a_b: 汇聚卷积 A 偏置 [C]
        conv_b_w: 注意力卷积 B 权重 [c_n, C, 1, 1]
        conv_b_b: 注意力卷积 B 偏置 [c_n]
        conv_v_w: 注意力卷积 V 权重 [c_n, C, 1, 1]
        conv_v_b: 注意力卷积 V 偏置 [c_n]
        conv_z_w: 重建卷积 Z 权重 [C, C, 1, 1]
        conv_z_b: 重建卷积 Z 偏置 [C]

    Returns:
        输出特征图 [B, C, H, W]；若输入任一维为 0 则原样返回输入
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x

    c_m = c
    c_n = max(1, c // 8)

    a = F.conv2d(x, conv_a_w, conv_a_b).view(b, c_m, -1)
    b_mat = F.softmax(F.conv2d(x, conv_b_w, conv_b_b).view(b, c_n, -1), dim=-1)
    v = F.softmax(F.conv2d(x, conv_v_w, conv_v_b).view(b, c_n, -1), dim=-1)

    global_descriptors = torch.bmm(a, b_mat.permute(0, 2, 1))
    tmp_z = torch.bmm(global_descriptors, v)
    tmp_z = tmp_z.view(b, c_m, h, w)
    return F.conv2d(tmp_z, conv_z_w, conv_z_b)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 3, 1024, 31, 21
c_n = max(1, C // 8)

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
conv_a_w = torch.randn(C, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv_a_b = torch.zeros(C, dtype=torch.float16, device="npu")
conv_b_w = torch.randn(c_n, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv_b_b = torch.zeros(c_n, dtype=torch.float16, device="npu")
conv_v_w = torch.randn(c_n, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv_v_b = torch.zeros(c_n, dtype=torch.float16, device="npu")
conv_z_w = torch.randn(C, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv_z_b = torch.zeros(C, dtype=torch.float16, device="npu")

y = cann_bench.double_attention(
    x, conv_a_w, conv_a_b, conv_b_w, conv_b_b,
    conv_v_w, conv_v_b, conv_z_w, conv_z_b,
)
# y: [3, 1024, 31, 21], float16
```
