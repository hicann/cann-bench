# DeformableOffsets 算子 API 描述

## 1. 算子简介

DeformableOffsets（image warp offsets）实现偏移量驱动的图像采样：把 `offsets` 张量中的每个元素视为输入图像按行展开 `[H*W*C]` 后的线性索引，从 `images[b]` 连续取出 C 个通道值，组成 `warp_images[b, p, h, w, :]`。本算子对标 ops-cv `image_warp_offsets` 融合实现（`aclnnIMGWarpOffsets`），等价于可变形采样中“按偏移取点”的一步（整数索引、无插值）。

**主要应用场景**：
- 可变形卷积/可变形注意力中的偏移采样
- 图像配准、光流 warp 等按索引重采样任务
- 检测/分割网络中 RoI 或关键点特征的偏移寻址

**算子特征**：
- 难度等级：L3（Vision）
- 双张量输入（图像、偏移），无属性，单输出
- `offsets` 可以是 int32 或 float32，计算时截断为整数索引
- 输出 shape 为 `[B, P, new_h, new_w, C]`，dtype 与 `images` 一致

## 2. 算子定义

### 数学公式

将 `images` 按 batch 展平为 `[B, H*W*C]`，对每个偏移位置：

$$
\text{idx} = \text{offsets}[b, p, h, w]
$$

$$
y[b, p, h, w, c] = \text{images}.\text{reshape}(B,\, H W C)[b,\; \text{idx} + c],\qquad c = 0,\ldots,C-1
$$

即每个偏移指向一个线性起始位置，连续读取 C 个通道值。

### 实现建议

- 用 `torch.gather` 一次完成全部偏移寻址：把 `images` 展平成 `[B, H*W*C]`，对 `offsets` 分别加 `[0, 1, ..., C-1]` 后 gather 出各通道，再 reshape 回目标 shape，避免逐位置 Python 循环。
- `offsets` 需先转为 int64 再作为索引；float32 偏移按截断（向零取整）转整数。
- 合法索引需满足 `0 <= idx <= H*W*C - C`，保证 `idx + C - 1 < H*W*C`。

## 3. 接口规范

### 算子原型

```python
cann_bench.deformable_offsets(Tensor images, Tensor offsets) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| images | Tensor | 必选 | 输入图像 [B, H, W, C]（NHWC，C 通常为 3） |
| offsets | Tensor | 必选 | 采样偏移 [B, P, new_h, new_w]，每个元素为 images[b] 按 [H*W*C] 展平后的线性索引 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, P, new_h, new_w, C] | 与 images 一致 | 采样结果 |

### 数据类型

| 输入 (images) dtype | 输入 (offsets) dtype | 输出 dtype |
|--------------------|---------------------|-----------|
| float16 | int32 | float16 |
| float16 | float32 | float16 |
| float32 | int32 | float32 |
| float32 | float32 | float32 |

### 规则与约束

- `images` 为 NHWC 布局，通道数 `C` 由 `images.shape[3]` 决定
- `offsets` 前两维分别对应 batch（`B`）与采样点数（`P`），后两维为输出空间尺寸
- `offsets` 中每个元素必须位于 `[0, H*W*C - C]`；`get_input` 会对随机输入做 clamp 保证不越界
- float32 偏移在计算索引时按截断转成整数
- 输出 dtype 与 `images` 一致

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

"""
DeformableOffsets 算子 Torch Golden 参考实现

偏移量驱动的图像采样：把 offsets 中的元素视为输入图像按行展开 [H*W*C] 的线性索引，
从 images[b] 连续取出 C 个通道值。
公式:
    idx = offsets[b, p, h, w]
    y[b, p, h, w, c] = images.reshape(B, H * W * C)[b, idx + c], c = 0..C-1
"""


def deformable_offsets(
    images: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """
    偏移量驱动的图像采样

    公式:
        idx = offsets[b, p, h, w]
        y[b, p, h, w, c] = images.reshape(B, H * W * C)[b, idx + c], c = 0..C-1

    Args:
        images: 输入图像 [B, H, W, C]（NHWC）
        offsets: 采样偏移 [B, P, new_h, new_w]，每个元素为 images[b] 展平 [H*W*C] 的线性索引

    Returns:
        输出张量 [B, P, new_h, new_w, C]，dtype 与 images 一致
    """
    b, h, w, ch = images.shape
    _, points, new_h, new_w = offsets.shape
    device = images.device

    images_flat = images.reshape(b, h * w * ch)             # [B, H*W*C]
    idx = offsets.to(torch.int64).reshape(b, -1)            # [B, P*new_h*new_w]
    # 每个偏移指向一个线性位置，连续读 ch 个通道
    warp_parts = [
        torch.gather(images_flat, 1, idx + c) for c in range(ch)
    ]
    warp_flat = torch.stack(warp_parts, dim=-1)             # [B, M, C]
    return warp_flat.reshape(b, points, new_h, new_w, ch)


def get_input(
    images: torch.Tensor,
    offsets: torch.Tensor,
    **kwargs,
):
    """规整 offsets 到合法线性索引范围，使采样不越界。

    offsets 必须是 [0, H*W*C - C] 内的整数，才能保证 idx + c (c < C) 不越过
    images[b] 的展平长度 H*W*C。这里对随机生成的 offsets 做 clamp，同时替换
    golden 与候选的输入，保证比较公平。

    Returns:
        [images, offsets]，顺序与 deformable_offsets 签名的张量输入顺序一致。
    """
    b, h, w, ch = images.shape
    max_index = h * w * ch - ch
    offsets = offsets.clamp(min=0, max=max_index)
    return [images, offsets]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# images: NHWC 图像；offsets: 展平线性索引
images = torch.randn(1, 32, 32, 3, dtype=torch.float32)
offsets = torch.randint(0, 32 * 32 * 3 - 3, (1, 4, 8, 8), dtype=torch.int32)
y = cann_bench.deformable_offsets(images, offsets)
# y.shape == (1, 4, 8, 8, 3)

# float32 偏移（计算时截断为整数索引）
offsets_f = torch.rand(2, 4, 4, 4, dtype=torch.float32) * (8 * 12 * 3 - 3)
images_f = torch.randn(2, 8, 12, 3, dtype=torch.float16)
y = cann_bench.deformable_offsets(images_f, offsets_f)
# y.shape == (2, 4, 4, 4, 3)
```
