# HaloAttention 算子 API 描述

## 1. 算子简介

HaloAttention 是 HaloNet 风格的局部自注意力：把特征图切成 `block_size × block_size` 的
不重叠块，每个块的 query 只与它自身及周围 halo 邻域（由 `F.unfold` 提取）产生的
key/value 做多头注意力。相比全局注意力，它的计算量随分辨率线性增长，适合高分辨率视觉任务。

**主要应用场景**：
- 高分辨率图像分类 / 检测 / 分割主干中的局部自注意力
- 需要以滑动窗口方式限制注意力感受野的视觉 Transformer

**算子特征**：
- 难度等级：L4（VisionAttention）
- 5 个必选张量输入（输入特征图 + 3 个 Linear 的权重/偏置）+ 4 个属性
- 单输出，shape 与输入 `x` 一致；`B/C/H/W` 任一为 0 时原样返回输入
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，块边长 $b$，halo 半径 $r$，
head 数 $n_h$，每个 head 维度 $d_h$，内部维度 $d = n_h d_h$。

分块与邻域提取：

$$
\mathbf{Q}_{in} = \mathrm{blockify}_b(\mathbf{x})\in\mathbb{R}^{B\cdot\frac{H}{b}\cdot\frac{W}{b}\times b^2\times C}
$$

$$
\mathbf{KV}_{in} = \mathrm{unfold}\!\left(\mathbf{x};\ \text{kernel}=b+2r,\ \text{stride}=b,\ \text{pad}=r\right)
\in\mathbb{R}^{B\cdot\frac{H}{b}\cdot\frac{W}{b}\times (b+2r)^2\times C}
$$

线性投影与多头拆分：

$$
\mathbf{Q} = \mathbf{Q}_{in}\mathbf{W}_q^{\top},\qquad
\mathbf{K},\mathbf{V} = \mathrm{split}\!\left(\mathbf{KV}_{in}\mathbf{W}_{kv}^{\top}\right)
$$

其中 $\mathbf{W}_q\in\mathbb{R}^{d\times C}$、$\mathbf{W}_{kv}\in\mathbb{R}^{2d\times C}$，均无偏置。

缩放点积注意力（逐 head）：

$$
\mathbf{A} = \mathrm{softmax}\!\left(\frac{\mathbf{Q}_h\,\mathbf{K}_h^{\top}}{\sqrt{d_h}}\right),
\qquad
\mathbf{O}_h = \mathbf{A}\,\mathbf{V}_h
$$

输出投影与反分块：

$$
\mathbf{y} = \mathrm{unblockify}_b\!\left(\mathbf{O}\,\mathbf{W}_o^{\top} + \mathbf{b}_o\right)\in\mathbb{R}^{B\times C\times H\times W}
$$

其中 $\mathbf{W}_o\in\mathbb{R}^{C\times d}$、$\mathbf{b}_o\in\mathbb{R}^{C}$。

### 实现建议

1. `q` 的块内顺序为 `(block_row, block_col, row_in_block, col_in_block, channel)`；
   `F.unfold` 的 KV 顺序为 `(block, kernel_pos, channel)`，需按源实现完整复刻。
2. KV 提取使用 `kernel_size = b + 2*r`、`stride = b`、`padding = r`，源实现未对 padding
   位置做 mask。
3. 注意力缩放因子为 `dim_head ** -0.5`，softmax 沿最后一维；全程使用输入 dtype 计算，
   不做 fp32 上转。
4. `to_q` / `to_kv` 无偏置，`to_out` 带偏置。
5. 源实现假设 `H`、`W` 能被 `block_size` 整除。

## 3. 接口规范

### 算子原型

```python
cann_bench.halo_attention(
    x, q_w, kv_w, out_w, out_b,
    block_size, halo_size, dim_head, heads
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `[B, C, H, W]` | 输入特征图 |
| `q_w` | `[dim_head * heads, C]` | query 线性层权重（无偏置） |
| `kv_w` | `[2 * dim_head * heads, C]` | key/value 线性层权重（无偏置） |
| `out_w` | `[C, dim_head * heads]` | 输出线性层权重 |
| `out_b` | `[C]` | 输出线性层偏置 |

### 属性

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `block_size` | int | - | 分块边长 b，H/W 须能被其整除 |
| `halo_size` | int | - | halo 半径 r |
| `dim_head` | int | 64 | 每个 head 的维度 d_h |
| `heads` | int | 8 | head 数 n_h |

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| `y` | `[B, C, H, W]` | 输出特征图 |

### 数据类型

输入与输出均支持 `float16`、`float32`、`bfloat16`，且所有张量使用同一 dtype。

### 规则与约束

- `H` 与 `W` 必须能被 `block_size` 整除。
- `q_w.shape[0] == dim_head * heads`，`kv_w.shape[0] == 2 * dim_head * heads`，
  `out_w.shape == [C, dim_head * heads]`，`out_b.shape == [C]`。
- `B/C/H/W` 任一为 0 时输出应为原输入，不得报错。
- KV 提取的 unfold 不做 padding mask，与源实现一致。

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
HaloAttention 算子 Torch Golden 参考实现

Halo Attention：基于 block 的局部自注意力，KV 通过 unfold 提取带 halo 的邻域块。

    q_inp = blockify(x)                                      # (B*num_blocks, b*b, C)
    kv_inp = unfold(x, k=b+2*halo, stride=b, padding=halo)    # (B*num_blocks, (b+2*halo)^2, C)
    q = to_q(q_inp);  k, v = chunk(to_kv(kv_inp), 2, dim=-1)
    attn = softmax((q * dim_head^-0.5) @ k^T)
    y = to_out(merge(attn @ v))

源实现全程使用输入 dtype 计算（softmax 不做 fp32 上转），本 golden 保持一致；
b / c / h / w 任一为 0 时源实现直接返回输入，本 golden 同样提前返回。
"""


def halo_attention(
    x: torch.Tensor,
    q_w: torch.Tensor,
    kv_w: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    block_size: int,
    halo_size: int,
    dim_head: int,
    heads: int,
) -> torch.Tensor:
    """
    Halo Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        q_w: query 线性层权重 [dim_head * heads, C]（无偏置）
        kv_w: key/value 线性层权重 [2 * dim_head * heads, C]（无偏置）
        out_w: 输出线性层权重 [C, dim_head * heads]
        out_b: 输出线性层偏置 [C]
        block_size: 分块边长 b
        halo_size: halo 半径 halo
        dim_head: 每个 head 的维度
        heads: head 数

    Returns:
        输出特征图 [B, C, H, W]；若输入任一维为 0 则原样返回输入
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x

    block = block_size
    halo = halo_size

    q_inp = x.reshape(b, c, h // block, block, w // block, block).permute(0, 2, 4, 3, 5, 1).reshape(
        b * (h // block) * (w // block), block * block, c
    )

    kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
    kv_inp = kv_inp.reshape(b, c, -1, kv_inp.shape[-1]).permute(0, 3, 2, 1).reshape(
        b * kv_inp.shape[-1], -1, c
    )

    q = F.linear(q_inp, q_w)
    k, v = F.linear(kv_inp, kv_w).chunk(2, dim=-1)

    q = q.reshape(q.shape[0], q.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(
        q.shape[0] * heads, q.shape[1], -1
    )
    k = k.reshape(k.shape[0], k.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(
        k.shape[0] * heads, k.shape[1], -1
    )
    v = v.reshape(v.shape[0], v.shape[1], heads, -1).permute(0, 2, 1, 3).reshape(
        v.shape[0] * heads, v.shape[1], -1
    )

    scale = dim_head ** -0.5
    q = q * scale
    sim = torch.bmm(q, k.transpose(1, 2))
    # 源实现未对 padding 位置做 mask
    attn = F.softmax(sim, dim=-1)
    out = torch.bmm(attn, v)
    out = out.reshape(-1, heads, out.shape[1], out.shape[2]).permute(0, 2, 1, 3).reshape(
        -1, out.shape[1], heads * out.shape[2]
    )
    out = F.linear(out, out_w, out_b)
    out = out.reshape(b, h // block, w // block, block, block, c).permute(0, 5, 1, 3, 2, 4).reshape(
        b, c, h, w
    )
    return out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 1, 8, 2, 2
block_size, halo_size = 2, 1
dim_head, heads = 64, 8
inner = dim_head * heads

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
q_w = torch.randn(inner, C, dtype=torch.float16, device="npu") * 0.05
kv_w = torch.randn(2 * inner, C, dtype=torch.float16, device="npu") * 0.05
out_w = torch.randn(C, inner, dtype=torch.float16, device="npu") * 0.05
out_b = torch.zeros(C, dtype=torch.float16, device="npu")

y = cann_bench.halo_attention(
    x, q_w, kv_w, out_w, out_b,
    block_size, halo_size, dim_head, heads,
)
# y: [1, 8, 2, 2], float16
```
