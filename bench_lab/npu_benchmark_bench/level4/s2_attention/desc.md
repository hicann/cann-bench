# S2Attention 算子 API 描述

## 1. 算子简介

S2（Spatial Shift + Split）Attention 先用一个线性层把通道扩展到 3 倍，再把通道切成三份：
第一份沿一个方向做 spatial shift，第二份沿另一个方向做 spatial shift，第三份保持不变。
三份分支通过 Split Attention（全局平均 -> 两层 MLP + GELU -> 分支 softmax）生成逐分支
逐通道权重，加权融合后经通道投影还原。

**主要应用场景**：
- 视觉主干网络中的轻量空间注意力（S2-MLP / S2 attention 系列）
- 以极低开销引入跨空间位置的交互

**算子特征**：
- 难度等级：L4（VisionAttention）
- 9 个张量输入（输入特征图 + 4 组 Linear 权重/偏置，全部显式化）
- 无属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，先重排为 NHWC 风格
$\mathbf{u}=\mathrm{permute}(\mathbf{x})\in\mathbb{R}^{B\times W\times H\times C}$。

$$
\mathbf{u}=\mathrm{Linear}_{C\to3C}(\mathbf{u}),\qquad
\mathbf{u}_1=\mathrm{shift}_1(\mathbf{u}_{:,:,:,:C}),\;
\mathbf{u}_2=\mathrm{shift}_2(\mathbf{u}_{:,:,:,C:2C}),\;
\mathbf{u}_3=\mathbf{u}_{:,:,:,2C:}
$$

$$
\mathbf{a}=\frac{1}{3N}\sum_{i=1}^{3}\sum_{j=1}^{N}\mathbf{u}_i^{(j)}\in\mathbb{R}^{B\times C},
\qquad
\mathbf{a}=\mathrm{softmax}\!\left(\mathrm{Linear}_{C\to3C}\!\left(\mathrm{GELU}\!\left(\mathrm{Linear}_{3C\to C}(\mathbf{a})\right)\right)\right)
$$

$$
\mathbf{y}=\mathrm{Linear}_{C\to C}\!\left(\sum_{i=1}^{3}\mathbf{a}_{:,i,:}\odot\mathbf{u}_i\right),
\qquad
\mathbf{y}=\mathrm{permute}(\mathbf{y})
$$

其中两种 spatial shift 分别把通道四等分，对各段沿 W 或 H 方向做相反方向的平移
（见 golden 中 `_spatial_shift1` / `_spatial_shift2`）。

### 实现建议

1. `x` 先 `permute(0,3,2,1).contiguous()` 变成 `[B, W, H, C]`，再经 `mlp1` 扩展到 `3C`。
2. 通道切分为 `[0:C]`、`[C:2C]`、`[2C:3C]`；两份分别执行 spatial shift（clone 后按切片
   赋值，源实现用 `c//4`、`c//2`、`3*c//4` 分段）。
3. 三份 `torch.stack([x1,x2,x3], dim=1)` 得到 `[B, 3, W, H, C]`，再 view 成 `[B, 3, N, C]`。
4. 全局平均先 `.float()` 再求均值、`.to(x.dtype)` 回原 dtype（避免 fp16 大 N 溢出）；
   经 `mlp3 -> GELU -> mlp4` 得到 `[B, 3, C]`，softmax 同样在 fp32 下完成后再 cast 回。
5. 按分支加权求和、`mlp2` 投影、再 permute 回 `[B, C, H, W]`。

## 3. 接口规范

### 算子原型

```python
cann_bench.s2_attention(
    x,
    mlp1_weight, mlp1_bias,
    mlp2_weight, mlp2_bias,
    mlp3_weight, mlp3_bias,
    mlp4_weight, mlp4_bias,
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| x | `[B, C, H, W]` | 输入特征图 |
| mlp1_weight / mlp1_bias | `[3C, C]` / `[3C]` | 通道扩展投影 |
| mlp2_weight / mlp2_bias | `[C, C]` / `[C]` | 输出投影 |
| mlp3_weight / mlp3_bias | `[C, C]` / `[C]` | Split Attention 第一层 |
| mlp4_weight / mlp4_bias | `[3C, C]` / `[3C]` | Split Attention 第二层 |

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| y | `[B, C, H, W]` | 输出特征图，dtype 与输入一致 |

### 数据类型

输入与输出支持 `float16`、`float32`、`bfloat16`；所有张量使用同一 dtype。除全局平均与
softmax 的 fp32 中转外，源实现保持输入 dtype。

### 规则与约束

- `x` 必须为 4 维 `[B, C, H, W]`；权重 shape 中的 C 必须与 `x.shape[1]` 一致。
- spatial shift 的分段使用 `c//4`、`c//2`、`3*c//4`，C 不必是 4 的倍数。
- 全局平均与分支 softmax 必须先升 fp32 再回到输入 dtype（与源实现一致）。

## 4. 精度要求

golden 保持源实现的 dtype 策略（均值/softmax 用 fp32 中转），无随机性。fp32 下期望逐位
一致；fp16/bf16 参考相对误差约 $10^{-3}$。

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
S2 Attention 算子 Torch Golden 参考实现

S2 (Spatial Shift + Split) Attention：先做通道 3 倍扩展投影，再把通道切成 3 份，
其中两份分别做两种方向的 spatial shift，第三份保持不变；三份分支经 Split Attention
（全局平均 -> MLP -> GELU -> MLP -> softmax 权重）加权融合，最后经通道投影回原维度。

    1. x -> [B, W, H, C] -> Linear(C, 3C)
    2. x1 = shift1(x[..., :C]); x2 = shift2(x[..., C:2C]); x3 = x[..., 2C:]
    3. a = softmax(Linear(3C->C) -> GELU -> Linear(C->3C))   # 每个分支每个通道的权重
    4. y = Linear(C->C)(sum_k a_k * x_k)

注意：源实现中全局平均与 softmax 会先升 fp32 再 cast 回输入 dtype，本 golden 保持一致；
其余计算保持输入 dtype（源实现不做整体 fp32 升精度）。
"""


def _spatial_shift1(x: torch.Tensor) -> torch.Tensor:
    """沿 W/H 两个方向分别平移部分通道（源实现 spatial_shift1）。"""
    b, w, h, c = x.shape
    out = x.clone()
    out[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    out[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    out[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    out[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return out


def _spatial_shift2(x: torch.Tensor) -> torch.Tensor:
    """沿 H/W 两个方向分别平移部分通道（源实现 spatial_shift2）。"""
    b, w, h, c = x.shape
    out = x.clone()
    out[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    out[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    out[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    out[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return out


def s2_attention(
    x: torch.Tensor,
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    mlp3_weight: torch.Tensor,
    mlp3_bias: torch.Tensor,
    mlp4_weight: torch.Tensor,
    mlp4_bias: torch.Tensor,
) -> torch.Tensor:
    """
    S2 Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        mlp1_weight: 扩展投影权重 [3C, C]
        mlp1_bias: 扩展投影偏置 [3C]
        mlp2_weight: 输出投影权重 [C, C]
        mlp2_bias: 输出投影偏置 [C]
        mlp3_weight: Split Attention 第一层权重 [C, C]
        mlp3_bias: Split Attention 第一层偏置 [C]
        mlp4_weight: Split Attention 第二层权重 [3C, C]
        mlp4_bias: Split Attention 第二层偏置 [3C]

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x

    x = x.permute(0, 3, 2, 1).contiguous()  # B,C,H,W -> B,W,H,C
    x = F.linear(x, mlp1_weight, mlp1_bias)  # B,W,H,3C
    x1 = _spatial_shift1(x[:, :, :, :c])
    x2 = _spatial_shift2(x[:, :, :, c:2 * c])
    x3 = x[:, :, :, 2 * c:]
    x_all = torch.stack([x1, x2, x3], 1)  # B,3,W,H,C

    # ---------- Split Attention ----------
    B, k, W, H, C_ = x_all.shape
    x_all = x_all.view(B, k, -1, C_)  # B,3,N,C
    # 分支/Token 维平均在 fp32 下进行，避免 fp16 大 N 溢出
    a = x_all.float().mean(dim=(1, 2)).to(x.dtype)  # B,C
    a = F.linear(a, mlp3_weight, mlp3_bias)
    a = F.gelu(a)
    a = F.linear(a, mlp4_weight, mlp4_bias).view(B, 3, C_)  # B,3,C
    a = a.float().softmax(dim=1).to(x.dtype)  # B,3,C
    out = (a.unsqueeze(-2) * x_all).sum(1).view(B, W, H, C_)  # B,W,H,C
    out = F.linear(out, mlp2_weight, mlp2_bias)
    out = out.permute(0, 3, 2, 1)  # B,W,H,C -> B,C,H,W
    return out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 8, 255, 14, 32
dt = torch.float32

x = torch.randn(B, C, H, W, dtype=dt, device="npu")

def lin_w(oc, ic):
    return torch.randn(oc, ic, dtype=dt, device="npu") * 0.05

y = cann_bench.s2_attention(
    x,
    lin_w(3 * C, C), torch.zeros(3 * C, dtype=dt, device="npu"),
    lin_w(C, C), torch.zeros(C, dtype=dt, device="npu"),
    lin_w(C, C), torch.zeros(C, dtype=dt, device="npu"),
    lin_w(3 * C, C), torch.zeros(3 * C, dtype=dt, device="npu"),
)
# y: [8, 255, 14, 32], float32
```
