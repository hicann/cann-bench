# Col2Im 算子 API 描述

## 1. 算子简介

Col2Im（col2im / fold）是 im2col 的逆操作：把滑窗展开得到的列张量按滑窗位置还原并累加重叠贡献，重新组装成完整图像。本算子对标 ops-cv `col2_im_v2` 融合实现，单核完成列到图像的重排与累加。

**主要应用场景**：
- 卷积神经网络反向传播中把卷积结果的列形式还原为特征图
- 基于 im2col 的滑窗运算（如 deconvolution、可变形卷积）的列到图像重排
- 图像分块（patch）处理后重建完整图像

**算子特征**：
- 难度等级：L3（Vision）
- 单输入 `x` + 5 个 ListInt 属性，单输出 `y`
- 输出像素由所有覆盖它的滑窗块累加得到，未被覆盖的位置为 0
- 输入 shape 为 `[N, C*kH*kW, L]`，输出 shape 为 `[N, C, H_out, W_out]`

## 2. 算子定义

### 数学公式

$$
y[n, c, h, w] = \sum_{k_h, k_w} x[n,\; c\,k_H k_W + k_h k_W + k_w,\; l]
$$

其中列索引与输出坐标满足：

$$
l = h_{col} L_w + w_{col},\quad
h = h_{col} s_H - p_H + k_h d_H,\quad
w = w_{col} s_W - p_W + k_w d_W
$$

求和仅包含存在合法滑窗位置 $(h_{col}, w_{col})$ 的 $(k_h, k_w)$；没有任何滑窗块覆盖的输出位置取 0。

### 实现建议

- 可直接调用 PyTorch 原生融合算子 `torch.nn.functional.fold`（等价于 col2im），一次完成全部重排与累加。
- 列长度 $L$ 必须等于 $L_h L_w$，其中 $L_h = \lfloor (H_{out} + 2p_H - d_H(k_H-1) - 1)/s_H \rfloor + 1$，$L_w$ 同理。

## 3. 接口规范

### 算子原型

```python
cann_bench.col2im(Tensor x, int[] output_size, int[] kernel_size,
                  int[] dilation=[1, 1], int[] padding=[0, 0], int[] stride=[1, 1]) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 滑窗列张量 [N, C*kH*kW, L]，由 im2col 得到 |
| output_size | ListInt | 必选 | 输出图像尺寸 [H_out, W_out] |
| kernel_size | ListInt | 必选 | 滑窗核尺寸 [kH, kW] |
| dilation | ListInt | [1, 1] | 膨胀系数 [dilH, dilW] |
| padding | ListInt | [0, 0] | 填充大小 [padH, padW] |
| stride | ListInt | [1, 1] | 滑窗步长 [strideH, strideW] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [N, C, H_out, W_out] | 与输入相同 | 重排回图像的结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |

### 规则与约束

- `x` 第 1 维必须能被 `kH*kW` 整除，得到通道数 `C = dim1 / (kH*kW)`
- `x` 第 2 维 `L` 必须等于 $L_h L_w$，否则 `F.fold` 报错
- `output_size`、`kernel_size`、`dilation`、`padding`、`stride` 均为长度为 2 的正整数列表（`dilation` 可为 1）
- 输出 dtype 与输入 `x` 一致；重叠滑窗位置的结果为各块贡献之和，未覆盖位置为 0

## 4. 标准 Golden 代码

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
Col2Im 算子 Torch Golden 参考实现

滑窗列重排回图像（col2im / fold）：对每个输出像素，累加所有覆盖它的滑窗块贡献，
未被覆盖的位置补 0。
公式:
    y[n, c, h, w] = sum_{kh, kw} x[n, c * kH * kW + kh * kW + kw, l]
    l = h_col * Lw + w_col
    h = h_col * strideH - padH + kh * dilH
    w = w_col * strideW - padW + kw * dilW
"""


def col2im(
    x: torch.Tensor,
    output_size: list,
    kernel_size: list,
    dilation: list = None,
    padding: list = None,
    stride: list = None,
) -> torch.Tensor:
    """
    滑窗列重排回图像（col2im / fold）

    公式:
        y[n, c, h, w] = sum_{kh, kw} x[n, c * kH * kW + kh * kW + kw, l]
        l = h_col * Lw + w_col
        h = h_col * strideH - padH + kh * dilH
        w = w_col * strideW - padW + kw * dilW

    Args:
        x: 滑窗列张量 [N, C*kH*kW, L]
        output_size: 输出图像尺寸 [H_out, W_out]
        kernel_size: 滑窗核尺寸 [kH, kW]
        dilation: 膨胀系数 [dilH, dilW]
        padding: 填充大小 [padH, padW]
        stride: 滑窗步长 [strideH, strideW]

    Returns:
        输出张量 [N, C, H_out, W_out]
    """
    if dilation is None:
        dilation = [1, 1]
    if padding is None:
        padding = [0, 0]
    if stride is None:
        stride = [1, 1]
    return F.fold(
        x,
        tuple(output_size),
        tuple(kernel_size),
        dilation=tuple(dilation),
        padding=tuple(padding),
        stride=tuple(stride),
    )
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# im2col 得到的列张量：N=1, C=64, kH=kW=3, L=16
x = torch.randn(1, 64 * 3 * 3, 16, dtype=torch.float32)
y = cann_bench.col2im(x, output_size=[4, 4], kernel_size=[3, 3],
                      dilation=[1, 1], padding=[1, 1], stride=[1, 1])
# y.shape == (1, 64, 4, 4)

# stride=2 的滑窗重排
x = torch.randn(1, 128 * 3 * 3, 49, dtype=torch.float16)
y = cann_bench.col2im(x, output_size=[13, 13], kernel_size=[3, 3],
                      padding=[1, 1], stride=[2, 2])
```
