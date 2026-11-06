# GlobalFilter 算子 API 描述

## 1. 算子简介

GlobalFilter 是 GFNet（Global Filter Networks）中的全局频域滤波算子：把 `[B, N, C]` 的 token
序列 reshape 成 `[B, H, W, C]` 的二维网格，在空间两维上做实数 FFT，与可学习的复数滤波器
逐元素相乘，再做逆 FFT 变换回空间域并还原为 token 序列。该操作以 $O(N\log N)$ 的代价实现
全局 token 混叠，是视觉 Transformer 中 self-attention 的频域替代方案。

**主要应用场景**：
- 图像分类 / 检测骨干网络中的全局 token 混合
- 需要长程依赖但希望避免 $O(N^2)$ 注意力开销的视觉模型

**算子特征**：
- 难度等级：L4（FusedComposite）
- 2 个必选张量输入（token 序列 + 频域滤波器）+ 3 个属性（dim / h / w）
- 单输出，shape 与输入 `x` 一致
- `x` 支持 float16 / float32 / bfloat16；滤波器参数固定为 float32
- 内部以 fp32 复数域计算，最后转回输入 dtype

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times N\times C}$，令 $N=H\cdot W$，
滤波器参数 $\mathbf{W}\in\mathbb{R}^{H\times (W/2+1)\times C\times 2}$，
其中最后一维为实部/虚部堆叠，记其复数形式为 $\mathbf{W}_c$。

$$
\mathbf{x}_s = \mathrm{reshape}_{H\times W}\!\left(\mathbf{x}\right).\mathrm{float}()
\in\mathbb{R}^{B\times H\times W\times C}
$$

$$
\mathbf{X}_f = \mathrm{rfft2}\!\left(\mathbf{x}_s;\ \mathrm{dim}=(1,2),\ \mathrm{norm}=\text{'ortho'}\right)
\in\mathbb{C}^{B\times H\times (W/2+1)\times C}
$$

$$
\mathbf{Y}_f = \mathbf{X}_f \odot \mathbf{W}_c
$$

$$
\mathbf{y} = \mathrm{reshape}_{N}\!\left(
\mathrm{irfft2}\!\left(\mathbf{Y}_f;\ s=(H,W),\ \mathrm{dim}=(1,2),\ \mathrm{norm}=\text{'ortho'}\right)
\right).\mathrm{to}(\mathrm{dtype}(\mathbf{x}))
\in\mathbb{R}^{B\times N\times C}
$$

### 实现建议

1. 滤波器以 `[H, W//2+1, C, 2]` 的 float32 实数张量存储，用 `torch.view_as_complex`
   转成 complex64 后再与 `rfft2` 的输出做逐元素乘；等价于分别对实部/虚部做乘加。
2. `rfft2` 与 `irfft2` 都必须使用 `norm='ortho'`，且 `irfft2` 需显式给出 `s=(H, W)`。
3. 输入 `x` 先 `.float()` 上转 fp32 再进 FFT；最终输出 `.to(x.dtype)` 转回输入 dtype。
4. 滤波器参数始终为 float32，与输入 dtype 无关（源实现用 `torch.randn(..., dtype=torch.float32)`）。
5. `dim` 属性在源实现中只参与缓存键，不参与数学计算；`h * w` 必须等于 `N`。

## 3. 接口规范

### 算子原型

```python
cann_bench.global_filter(x, weight, dim, h, w) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `[B, N, C]` | 输入 token 序列，`N = H*W` |
| `weight` | `[H, W//2+1, C, 2]` | 频域滤波器，实部/虚部堆叠，float32 |

### 属性

| 参数 | 类型 | 说明 |
|---|---|---|
| `dim` | int | 通道维大小（源实现仅用于缓存键，不参与计算） |
| `h` | int | 频域网格高度，须满足 `h * w == N` |
| `w` | int | 频域网格宽度，须满足 `h * w == N` |

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| `y` | `[B, N, C]` | 输出 token 序列，dtype 与 `x` 一致 |

### 数据类型

- `x` 与 `y`：`float16` / `float32` / `bfloat16`，二者 dtype 相同。
- `weight`：固定 `float32`。

### 规则与约束

- `h * w` 必须严格等于 `x.shape[1]`，否则源实现抛出 `ValueError`。
- `weight.shape == [h, w // 2 + 1, C, 2]`，且 `weight` 为连续张量（`view_as_complex` 要求）。
- FFT / iFFT 必须使用 `norm='ortho'`，否则数值与 golden 不一致。

## 4. 精度要求

golden 在 fp32 复数域完成 FFT、复乘与逆变换，最终转回输入 dtype；结果与 golden 的误差需满足
bench 的统一精度门限，推荐以约 $10^{-3}$（fp16/bf16）相对误差为参考。

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

"""
GlobalFilter 算子 Torch Golden 参考实现

Global Filter Network：在频域对 token 做逐元素复乘滤波后逆变换回空间域。

    x_s    = x.view(B, H, W, C).float()                 # 内部上转 fp32
    X_f    = rfft2(x_s, dim=(1, 2), norm='ortho')       # (B, H, W//2+1, C) complex64
    W_c    = view_as_complex(weight)                    # weight: (H, W//2+1, C, 2) fp32
    Y_f    = X_f * W_c
    y      = irfft2(Y_f, s=(H, W), dim=(1, 2), norm='ortho').reshape(B, N, C).to(input_dtype)

源实现的滤波器参数始终以 float32 存储（与输入 dtype 无关），因此本 golden 的
weight 输入固定为 float32；其余计算在 fp32 复数域完成，最终转回输入 dtype。
dim 属性仅用于源实现的缓存键，不参与数学计算。
"""


def global_filter(
    x: torch.Tensor,
    weight: torch.Tensor,
    dim: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Global Filter 前向

    Args:
        x: 输入 token 序列 [B, N, C]，N = h * w
        weight: 频域滤波器实部/虚部堆叠 [h, w // 2 + 1, C, 2]，float32
        dim: 通道维大小（源实现仅用于缓存键，不参与计算）
        h: 空间网格高度
        w: 空间网格宽度

    Returns:
        输出 token 序列 [B, N, C]，dtype 与 x 一致
    """
    batch, n, c = x.shape
    if h * w != n:
        raise ValueError(f"GlobalFilter: h*w ({h}*{w}={h * w}) must equal N ({n})")

    dtype = x.dtype
    x = x.view(batch, h, w, c).float()
    x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
    weight_c = torch.view_as_complex(weight)
    x_fft = x_fft * weight_c
    x_ifft = torch.fft.irfft2(x_fft, s=(h, w), dim=(1, 2), norm="ortho")
    return x_ifft.reshape(batch, n, c).to(dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, N, C = 2, 77, 32
h, w = 7, 11
r_w = w // 2 + 1

x = torch.randn(B, N, C, dtype=torch.bfloat16, device="npu")
weight = torch.randn(h, r_w, C, 2, dtype=torch.float32, device="npu") * 0.05

y = cann_bench.global_filter(x, weight, C, h, w)
# y: [2, 77, 32], bfloat16
```
