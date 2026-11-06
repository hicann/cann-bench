# SpatialTransformer 算子 API 描述

## 1. 算子简介

SpatialTransformer（STN 仿射空间变换）：用 2x3 仿射矩阵 theta 把输出网格映射到输入坐标，再对输入张量做双线性采样，等价于 `F.affine_grid` + `F.grid_sample` 的组合。对应 ops-cv 的 `spatial_transformer` AscendC kernel（兼容 TensorFlow SpatialTransformer），单核一次完成仿射网格生成 + 采样。

**主要应用场景**：
- 图像对齐、配准与几何校正（旋转、缩放、平移、剪切）
- STN 网络中可学习的空间变换模块
- 视频 / 医学影像的仿射配准预处理

**算子特征**：
- 难度等级：L3（Vision）
- 双输入（特征图 x 与仿射矩阵 theta）、单输出，输入为 NCHW 4 维张量
- 支持 `theta` 按 batch 广播、`output_size` 沿用输入尺寸、`align_corners`，以及 `default_theta` / `use_default_theta` 逐元素覆盖

## 2. 算子定义

### 数学公式

$$
\text{grid} = \mathrm{affine\_grid}(\theta, [B, C, H_{out}, W_{out}], \text{align\_corners}),\qquad
y = \mathrm{grid\_sample}(x, \text{grid}, \text{align\_corners})
$$

对归一化输出坐标 $(x_t, y_t) \in [-1, 1]^2$，仿射映射为

$$
\begin{aligned}
x_s &= \theta_{00} x_t + \theta_{01} y_t + \theta_{02}, \\
y_s &= \theta_{10} x_t + \theta_{11} y_t + \theta_{12},
\end{aligned}
$$

再在输入特征图对应位置做双线性插值；越界位置按零填充（`padding_mode="zeros"`）。

若 `use_default_theta` 非空，则对 theta 展平后的第 $k$ 个元素，
当 `use_default_theta[k] > 0` 时取 `default_theta[k]`，否则取输入 theta 的对应元素。

**实现建议**：先按 batch 生成输出网格坐标，再用 2x3 矩阵做仿射变换，最后逐像素双线性采样；可按 N*C 或输出行切分到多核。

## 3. 接口规范

### 算子原型

```python
cann_bench.spatial_transformer(Tensor x, Tensor theta, int[] output_size, bool align_corners=False, float[] default_theta=[], int[] use_default_theta=[]) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图 [B, C, H, W] |
| theta | Tensor | 必选 | 仿射变换矩阵 [B, 2, 3] 或 [2, 3]（[2, 3] 时广播到 batch） |
| output_size | ListInt | 必选 | 目标输出尺寸 (out_h, out_w)，元素 <= 0 表示沿用输入 H/W |
| align_corners | bool | false | 角像素中心对齐标志 |
| default_theta | ListFloat | [] | 默认仿射参数（6 个），与 use_default_theta 配合使用 |
| use_default_theta | ListInt | [] | 6 个 0/1 标志，1 表示 theta 对应位置取 default_theta |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, C, out_h, out_w] | 与输入 x 相同 | 空间变换后的特征图 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| x: float16, theta: float32 | float16 |
| x: float32, theta: float32 | float32 |

### 规则与约束

- 输入 x 为 4 维 NCHW 张量；theta 为 [B, 2, 3] 或 [2, 3]，batch 维须与 x 的 B 一致（或通过 [2, 3] 广播）。
- `output_size` 长度为 2；任一元素 <= 0 时该维沿用输入尺寸。
- `default_theta` 非空时长度须为 6；`use_default_theta` 非空时长度须为 6，取值为 0/1。
- 采样使用双线性插值，越界坐标按 0 填充；`align_corners=False` 时归一化坐标覆盖像素中心区域。
- 输出 dtype 与输入 x 的 dtype 一致；theta 只参与网格计算，不改变输出 dtype。

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
SpatialTransformer 算子 Torch Golden 参考实现

STN 仿射空间变换：用 2x3 仿射矩阵 theta 把输出网格映射到输入坐标，再做双线性采样。
公式: grid = affine_grid(theta, [B, C, H_out, W_out], align_corners)
      y = grid_sample(x, grid, align_corners)
"""


def spatial_transformer(
    x: torch.Tensor,
    theta: torch.Tensor,
    output_size: list,
    align_corners: bool = False,
    default_theta: list = None,
    use_default_theta: list = None,
) -> torch.Tensor:
    """
    STN 仿射空间变换（affine_grid + grid_sample）

    公式: grid = affine_grid(theta, [B, C, H_out, W_out], align_corners)
          y = grid_sample(x, grid, align_corners)

    Args:
        x: 输入特征图 [B, C, H, W]
        theta: 仿射变换矩阵 [B, 2, 3] 或 [2, 3]（[2, 3] 时广播到 batch）
        output_size: 目标输出尺寸 (out_h, out_w)，元素 <= 0 表示沿用输入尺寸
        align_corners: 角像素中心对齐标志
        default_theta: 默认仿射参数（6 个）
        use_default_theta: 6 个 0/1 标志，1 表示对应位置取 default_theta

    Returns:
        输出张量 [B, C, out_h, out_w]
    """
    b, c, h, w = x.shape
    if theta.dim() == 2:                          # [2, 3] -> [B, 2, 3]
        theta = theta.unsqueeze(0).expand(b, -1, -1).contiguous()

    if use_default_theta:
        mask = torch.tensor(use_default_theta, dtype=torch.float32,
                            device=theta.device)
        default = torch.tensor(default_theta, dtype=theta.dtype,
                               device=theta.device)
        theta_flat = torch.where(mask[None, :] > 0,
                                 default[None, :].expand(b, 6),
                                 theta.reshape(b, 6))
        theta = theta_flat.reshape(b, 2, 3)

    out_h = h if output_size[0] <= 0 else int(output_size[0])
    out_w = w if output_size[1] <= 0 else int(output_size[1])

    grid = F.affine_grid(theta, (b, c, out_h, out_w),
                         align_corners=align_corners)
    # aclnnGridSampler2D 要求 grid 与 input dtype 一致
    grid = grid.to(x.dtype)
    return F.grid_sample(x, grid, align_corners=align_corners)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(8, 64, 48, 48, dtype=torch.float16, device="npu")
theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32).repeat(8, 1, 1)
y = cann_bench.spatial_transformer(x, theta, [48, 48])  # 恒等变换 [8, 64, 48, 48]

# 使用 default_theta 覆盖 theta 的部分元素
y2 = cann_bench.spatial_transformer(
    x, theta, [-1, -1], False,
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1, 0, 1, 0, 0, 1],
)
```
