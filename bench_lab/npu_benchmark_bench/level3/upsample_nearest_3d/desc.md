# UpsampleNearest3D 算子 API 描述

## 1. 算子简介

最近邻 3D 上采样：按目标尺寸 (D_out, H_out, W_out) 把每个输出体素映射回输入特征体中的最近邻坐标并取值，不做插值。对应 ops-cv 的 `upsample_nearest3d` AscendC kernel（aclnnUpsampleNearest3D）。

**主要应用场景**：
- 视频 / 体数据（CT、MRI、点云体素）的 3D 分辨率提升
- 3D 检测、分割网络解码器中的特征图上采样
- 时空特征（T、H、W）的快速无插值放大

**算子特征**：
- 难度等级：L3（Vision）
- 单输入单输出，输入为 NC(D)(H)(W) 5 维张量
- 通过 `output_size` 属性指定目标 3D 尺寸，各维可独立缩放

## 2. 算子定义

### 数学公式

$$
y[n, c, i, j, k] = input\left[n, c,\ \left\lfloor \frac{i \cdot D_{in}}{D_{out}} \right\rfloor,\ \left\lfloor \frac{j \cdot H_{in}}{H_{out}} \right\rfloor,\ \left\lfloor \frac{k \cdot W_{in}}{W_{out}} \right\rfloor\right]
$$

坐标映射与 `torch.nn.functional.interpolate(input, size=output_size, mode="nearest")`（align_corners=False）一致。

**实现建议**：每个输出体素独立计算 3 个源坐标并做一次 gather，计算量与输出元素数成正比；可按 N*C 或输出 (D,H) 切片切分到多核。

## 3. 接口规范

### 算子原型

```python
cann_bench.upsample_nearest_3d(Tensor input, int[] output_size) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入 5D 张量 [N, C, D_in, H_in, W_in] |
| output_size | ListInt | 必选 | 目标输出尺寸 (D_out, H_out, W_out)，均须为正整数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [N, C, D_out, H_out, W_out] | 与输入 input 相同 | 最近邻 3D 上采样结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |

### 规则与约束

- 输入 input 为 5 维 NCDHW 张量；N、C 可为任意正整数，D、H、W 须大于 0。
- `output_size` 三个元素均须为正整数，YAML 中以长度为 3 的整数列表给出。
- 输出各维可以大于、等于或小于输入对应维度（支持上采样与下采样），映射统一使用 `floor(i * in / out)`。
- 输出 dtype 与输入 dtype 保持一致。
- 单 tensor 元素数建议不超过 64M（本目录 cases 已按此约束控制输出规模）。

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
UpsampleNearest3D 算子 Torch Golden 参考实现

最近邻 3D 上采样，按目标尺寸逐元素取最近邻源坐标。
公式: y[n, c, i, j, k] = input[n, c, floor(i * D_in / D_out), floor(j * H_in / H_out),
                              floor(k * W_in / W_out)]
"""


def upsample_nearest_3d(input: torch.Tensor, output_size: list) -> torch.Tensor:
    """
    最近邻 3D 上采样

    公式: y[n, c, i, j, k] = input[n, c, floor(i * D_in / D_out), floor(j * H_in / H_out),
                                  floor(k * W_in / W_out)]

    Args:
        input: 输入 5D 张量 [N, C, D_in, H_in, W_in]
        output_size: 目标输出尺寸 (D_out, H_out, W_out)

    Returns:
        输出张量 [N, C, D_out, H_out, W_out]
    """
    size = (int(output_size[0]), int(output_size[1]), int(output_size[2]))
    return F.interpolate(input, size=size, mode="nearest")
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1, 8, 8, 64, 64, dtype=torch.float16, device="npu")
y = cann_bench.upsample_nearest_3d(x, [16, 128, 128])  # [1, 8, 16, 128, 128]

x32 = torch.randn(1, 4, 32, 256, 256, dtype=torch.float32, device="npu")
y32 = cann_bench.upsample_nearest_3d(x32, [64, 256, 256])  # [1, 4, 64, 256, 256]
```
