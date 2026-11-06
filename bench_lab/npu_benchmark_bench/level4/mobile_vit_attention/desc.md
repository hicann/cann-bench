# MobileViTAttention 算子 API 描述

## 1. 算子简介

MobileViTAttention 是 MobileViT 网络中的全局表征模块：先用两层卷积得到局部表征，再按
`patch_size` 把特征图 unfold 成 token 序列，做若干层 Transformer（PreNorm 多头自注意力 +
PreNorm FFN/SiLU）后 fold 回特征图，最后与原始输入拼接并经卷积融合。它把 CNN 的局部归纳偏置
与 Transformer 的全局建模能力结合在同一算子内。

**主要应用场景**：
- 移动端/边缘端视觉主干网络（图像分类、检测、分割）中的轻量全局注意力
- 需要在特征图上以 patch 为单位做自注意力、同时保留卷积融合路径的视觉任务

**算子特征**：
- 难度等级：L4（VisionAttention / FusedComposite）
- 20 个必选张量输入（输入特征图 + 局部卷积、3 层 Transformer、融合卷积的全部权重/偏置）+ 3 个属性
- 其中 11 个输入是 `List[Tensor]`：3 层同构权重逐层传入（DEPTH=3 为源实现硬编码结构常量）
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，patch 边长 $p_s$，
$p=p_s^2$，$n=\frac{H}{p_s}\cdot\frac{W}{p_s}$，隐层维度 $d=\text{dim}$，卷积核 $k$。

局部表征（无中间激活）：

$$
\mathbf{y}_0 = \mathrm{Conv}_{1\times1}\!\left(\mathrm{Conv}_{k}\!\left(\mathbf{x}\right)\right)\in\mathbb{R}^{B\times d\times H\times W}
$$

unfold（把每个 patch 内像素作为 token，patch 位置作为序列）：

$$
\mathbf{t} = \mathrm{unfold}(\mathbf{y}_0)\in\mathbb{R}^{B\times p\times n\times d}
$$

对第 $l$ 层（$l=0,\dots,2$，共 3 层），记 $\mathrm{MHA}$ 为 8 头、每头 64 维的自注意力：

$$
\mathbf{t} \leftarrow \mathbf{t} + W^{o}_{l}\,\mathrm{MHA}\!\left(\mathrm{LN}^{att}_{l}(\mathbf{t});\,W^{qkv}_{l}\right)
$$

$$
\mathbf{t} \leftarrow \mathbf{t} + W^{ff2}_{l}\,\mathrm{SiLU}\!\left(W^{ff1}_{l}\,\mathrm{LN}^{ffn}_{l}(\mathbf{t}) + b^{ff1}_{l}\right) + b^{ff2}_{l}
$$

其中 $W^{qkv}_l\in\mathbb{R}^{3\cdot 512\times d}$（无 bias），$W^{o}_l\in\mathbb{R}^{d\times 512}$，
$W^{ff1}_l\in\mathbb{R}^{1024\times d}$，$W^{ff2}_l\in\mathbb{R}^{d\times 1024}$。

fold 回特征图后做融合：

$$
\mathbf{y} = \mathrm{Conv}_{k}\!\left(\mathrm{cat}\!\left[\mathbf{x},\ \mathrm{Conv}_{1\times1}(\mathbf{t})\right]\right)\in\mathbb{R}^{B\times C\times H\times W}
$$

### 实现建议

1. `conv1` 为 `C -> C`、kernel `k`、same padding（`k // 2`）；`conv2` 为 `C -> dim` 的 1x1 卷积，两者之间无激活。
2. unfold 用 `reshape(B, dim, nh, ph, nw, pw).permute(0, 3, 5, 2, 4, 1)` 实现，等价于按 patch 重排；fold 为其逆变换 `reshape(B, ph, pw, nh, nw, dim).permute(0, 5, 3, 1, 4, 2)`。
3. Attention：`qkv = Linear(dim, 3*512)` 后沿最后一维均分 3 份，各自 reshape 为 `[B, p, n, 8, 64]` 并 permute 成 `[B, p, 8, n, 64]`；scores 乘 `64**-0.5` 后 softmax。
4. FFN 中间维固定为 1024，激活为 SiLU。
5. 三个结构常量来自源实现：`DEPTH = 3`、`HEADS = 8`、`HEAD_DIM = 64`，即 `INNER_DIM = 512`、`MLP_DIM = 1024`。

## 3. 接口规范

### 算子原型

```python
cann_bench.mobile_vit_attention(
    x, conv1_w, conv1_b, conv2_w, conv2_b,
    ln_att_w, ln_att_b, qkv_w, out_w, out_b,
    ln_ffn_w, ln_ffn_b, ff1_w, ff1_b, ff2_w, ff2_b,
    conv3_w, conv3_b, conv4_w, conv4_b,
    patch_size, kernel_size, dim
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `[B, C, H, W]` | 输入特征图，H/W 可被 `patch_size` 整除 |
| `conv1_w` / `conv1_b` | `[C, C, k, k]` / `[C]` | 局部表征第一层卷积权重/偏置 |
| `conv2_w` / `conv2_b` | `[dim, C, 1, 1]` / `[dim]` | 局部表征 1x1 卷积权重/偏置 |
| `ln_att_w` / `ln_att_b` | `List`（3 个 `[dim]`） | 各层注意力 PreNorm 的 LayerNorm affine |
| `qkv_w` | `List`（3 个 `[3*512, dim]`） | 各层 qkv 融合线性权重（无 bias） |
| `out_w` / `out_b` | `List`（3 个 `[dim, 512]` / `[dim]`） | 各层注意力输出投影权重/偏置 |
| `ln_ffn_w` / `ln_ffn_b` | `List`（3 个 `[dim]`） | 各层 FFN PreNorm 的 LayerNorm affine |
| `ff1_w` / `ff1_b` | `List`（3 个 `[1024, dim]` / `[1024]`） | 各层 FFN 第一层线性权重/偏置 |
| `ff2_w` / `ff2_b` | `List`（3 个 `[dim, 1024]` / `[dim]`） | 各层 FFN 第二层线性权重/偏置 |
| `conv3_w` / `conv3_b` | `[C, dim, 1, 1]` / `[C]` | 融合前 1x1 卷积权重/偏置 |
| `conv4_w` / `conv4_b` | `[C, 2*C, k, k]` / `[C]` | 融合卷积权重/偏置 |

### 属性

| 参数 | 类型 | 说明 |
|---|---|---|
| `patch_size` | int | unfold/fold 的 patch 边长，H、W 须能被其整除 |
| `kernel_size` | int | conv1/conv4 的卷积核大小 k（same padding = k // 2） |
| `dim` | int | Transformer 隐层维度，等于 `conv2_w.shape[0]` |

### 输出

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `y` | `[B, C, H, W]`（与 x 相同） | 与输入相同 | 融合后的输出特征图 |

### 数据类型

`float16` / `float32` / `bfloat16`，所有张量输入与输出使用同一 dtype。

### 规则与约束

- `H % patch_size == 0` 且 `W % patch_size == 0`。
- 各 `List[Tensor]` 输入长度固定为 DEPTH=3，与源实现的结构常量一致。
- `x.shape[1]` 应等于 `conv1_w.shape[0]` 与 `conv4_w.shape[0] / 2`。
- 源实现 dropout 恒为 0，golden 不含随机性。

## 4. 精度要求

- golden 全程使用输入 dtype 计算，输出 dtype 与输入一致；不含随机数，仅依赖显式权重输入。
- 该算子为 ~30 个算子组成的复合块，包含 LayerNorm / softmax / SiLU 等非线性链；评测按
  bench 统一的 fp16/bf16/fp32 精度门限比较，建议实现内部以 fp32 累加以控制误差。

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
from typing import List

"""
MobileViTAttention 算子 Torch Golden 参考实现

MobileViT Attention block（原版语义）：
    y0 = Conv2d(C->C, k) -> Conv2d(C->dim, 1x1)              # 无激活的局部表征
    unfold: (b, dim, nh*ph, nw*pw) -> (b, ph*pw, nh*nw, dim)
    DEPTH=3 x [ 残差(PreNorm(LN -> 8 头 attention -> proj))
                残差(PreNorm(LN -> Linear(dim->1024) -> SiLU -> Linear(1024->dim))) ]
    fold 回 (b, dim, h, w)
    y = Conv2d(dim->C, 1x1) -> cat([x, y]) -> Conv2d(2C->C, k)

说明：
- 源码把 DEPTH=3 层的同构权重按 depth 维堆叠后传入；本 golden 用 List[Tensor]
  表达为每层一个张量（列表长度为 3），展开后逐层计算，数学上完全等价。
- 原版 dropout 全为 0.0（eval 恒等），不纳入。
- 全程使用输入 dtype 计算，输出 dtype 与输入一致（源 forward 的降精度出口
  目标就是 case 标称 dtype，与本任务 case 的输入 dtype 相同）。
"""

DEPTH = 3
HEADS = 8
HEAD_DIM = 64
MLP_DIM = 1024
INNER_DIM = HEADS * HEAD_DIM  # 512


def mobile_vit_attention(
    x: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    conv2_w: torch.Tensor,
    conv2_b: torch.Tensor,
    ln_att_w: List[torch.Tensor],
    ln_att_b: List[torch.Tensor],
    qkv_w: List[torch.Tensor],
    out_w: List[torch.Tensor],
    out_b: List[torch.Tensor],
    ln_ffn_w: List[torch.Tensor],
    ln_ffn_b: List[torch.Tensor],
    ff1_w: List[torch.Tensor],
    ff1_b: List[torch.Tensor],
    ff2_w: List[torch.Tensor],
    ff2_b: List[torch.Tensor],
    conv3_w: torch.Tensor,
    conv3_b: torch.Tensor,
    conv4_w: torch.Tensor,
    conv4_b: torch.Tensor,
    patch_size: int,
    kernel_size: int,
    dim: int,
) -> torch.Tensor:
    """
    MobileViT Attention 块

    Args:
        x: 输入特征图 [B, C, H, W]，H/W 可被 patch_size 整除
        conv1_w/conv1_b: 局部表征 kxk 卷积权重 [C, C, k, k] / 偏置 [C]
        conv2_w/conv2_b: 局部表征 1x1 卷积权重 [dim, C, 1, 1] / 偏置 [dim]
        ln_att_w/ln_att_b: 3 层注意力 PreNorm 的 LayerNorm affine，每个 [dim]
        qkv_w: 3 层 qkv 融合线性权重（无 bias），每个 [3*512, dim]
        out_w/out_b: 3 层注意力输出投影权重 [dim, 512] / 偏置 [dim]
        ln_ffn_w/ln_ffn_b: 3 层 FFN PreNorm 的 LayerNorm affine，每个 [dim]
        ff1_w/ff1_b: 3 层 FFN 第一层线性权重 [1024, dim] / 偏置 [1024]
        ff2_w/ff2_b: 3 层 FFN 第二层线性权重 [dim, 1024] / 偏置 [dim]
        conv3_w/conv3_b: Fusion 前 1x1 卷积权重 [C, dim, 1, 1] / 偏置 [C]
        conv4_w/conv4_b: Fusion 融合 kxk 卷积权重 [C, 2C, k, k] / 偏置 [C]
        patch_size: unfold/fold 的 patch 边长
        kernel_size: conv1/conv4 的卷积核大小 k
        dim: Transformer 隐层维度

    Returns:
        输出特征图 [B, C, H, W]，dtype 与 x 相同
    """
    b, c, h, w = x.shape
    dim = conv2_w.shape[0]
    k = conv1_w.shape[-1]
    ph = pw = patch_size
    nh, nw = h // ph, w // pw
    p, n = ph * pw, nh * nw

    # Local Representation（原版无中间激活）
    y = F.conv2d(x, conv1_w, conv1_b, padding=k // 2)
    y = F.conv2d(y, conv2_w, conv2_b)

    # unfold
    y = y.reshape(b, dim, nh, ph, nw, pw).permute(0, 3, 5, 2, 4, 1).reshape(b, p, n, dim)

    # Global Representation: depth=3 x (PreNorm-Attn + PreNorm-FFN), 残差
    for l in range(DEPTH):
        a = F.layer_norm(y, (dim,), ln_att_w[l], ln_att_b[l])
        qkv = F.linear(a, qkv_w[l]).chunk(3, dim=-1)
        q, kk, v = [t.reshape(b, p, n, HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4) for t in qkv]
        attn = torch.softmax(torch.matmul(q, kk.transpose(-1, -2)) * (HEAD_DIM ** -0.5), dim=-1)
        o = torch.matmul(attn, v).permute(0, 1, 3, 2, 4).reshape(b, p, n, INNER_DIM)
        y = y + F.linear(o, out_w[l], out_b[l])

        f = F.layer_norm(y, (dim,), ln_ffn_w[l], ln_ffn_b[l])
        f = F.silu(F.linear(f, ff1_w[l], ff1_b[l]))
        y = y + F.linear(f, ff2_w[l], ff2_b[l])

    # fold
    y = y.reshape(b, ph, pw, nh, nw, dim).permute(0, 5, 3, 1, 4, 2).reshape(b, dim, h, w)

    # Fusion
    y = F.conv2d(y, conv3_w, conv3_b)
    y = torch.cat([x, y], 1)
    out = F.conv2d(y, conv4_w, conv4_b, padding=k // 2)
    return out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 32, 33, 9
patch_size, kernel_size, dim = 3, 3, 384
DEPTH = 3

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
conv1_w = torch.randn(C, C, kernel_size, kernel_size, dtype=torch.float16, device="npu") * 0.05
conv1_b = torch.zeros(C, dtype=torch.float16, device="npu")
conv2_w = torch.randn(dim, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv2_b = torch.zeros(dim, dtype=torch.float16, device="npu")

ln_att_w = [torch.ones(dim, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
ln_att_b = [torch.zeros(dim, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
qkv_w = [torch.randn(3 * 512, dim, dtype=torch.float16, device="npu") * 0.05 for _ in range(DEPTH)]
out_w = [torch.randn(dim, 512, dtype=torch.float16, device="npu") * 0.05 for _ in range(DEPTH)]
out_b = [torch.zeros(dim, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
ln_ffn_w = [torch.ones(dim, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
ln_ffn_b = [torch.zeros(dim, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
ff1_w = [torch.randn(1024, dim, dtype=torch.float16, device="npu") * 0.05 for _ in range(DEPTH)]
ff1_b = [torch.zeros(1024, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
ff2_w = [torch.randn(dim, 1024, dtype=torch.float16, device="npu") * 0.05 for _ in range(DEPTH)]
ff2_b = [torch.zeros(dim, dtype=torch.float16, device="npu") for _ in range(DEPTH)]
conv3_w = torch.randn(C, dim, 1, 1, dtype=torch.float16, device="npu") * 0.05
conv3_b = torch.zeros(C, dtype=torch.float16, device="npu")
conv4_w = torch.randn(C, 2 * C, kernel_size, kernel_size, dtype=torch.float16, device="npu") * 0.05
conv4_b = torch.zeros(C, dtype=torch.float16, device="npu")

y = cann_bench.mobile_vit_attention(
    x, conv1_w, conv1_b, conv2_w, conv2_b,
    ln_att_w, ln_att_b, qkv_w, out_w, out_b,
    ln_ffn_w, ln_ffn_b, ff1_w, ff1_b, ff2_w, ff2_b,
    conv3_w, conv3_b, conv4_w, conv4_b,
    patch_size, kernel_size, dim,
)
# y: [2, 32, 33, 9], float16
```
