# WeightedPermuteMLP 算子 API 描述

## 1. 算子简介

Weighted Permute MLP（ViP）。输入特征图 `[B, H, W, C]` 分别沿 channel、height、width 三个方向
做线性投影，再用学习到的 softmax 重加权系数融合三个分支，最后做输出投影。原实现内部随机初始化的
`nn.Linear` 权重已全部提升为显式 tensor 入参。

**主要应用场景**：
- 视觉 Transformer 中的 permute-MLP 结构
- 多方向特征混合与重加权

**算子特征**：
- 难度等级：L4（FusedComposite）
- 13 个 tensor 输入 + 4 个属性
- `seg_dim` 不整除时实现向下调整到能同时整除 `C` 与 `H*W` 的最大值
- 输出与输入同 shape

## 2. 算子定义

### 数学公式

记 `S = C / seg_dim`，`phi_h` / `phi_w` 为 height / width 方向的 reshape-permute 重排：

$$
c=\mathrm{Linear}_c(x),\quad h=\phi_h^{-1}\big(\mathrm{Linear}_h(\phi_h(x))\big),\quad
w=\phi_w^{-1}\big(\mathrm{Linear}_w(\phi_w(x))\big)
$$

$$
z=\mathrm{mean}_{H,W}(c+h+w),\quad
a=\mathrm{softmax}\big(\mathrm{reshape}(\mathrm{Linear}_{fc2}(\mathrm{GELU}(\mathrm{Linear}_{fc1}(z))),C,3),0\big)
$$

$$
y=\mathrm{Linear}_p\big(c\cdot a_0+w\cdot a_1+h\cdot a_2\big)
$$

### 实现建议

1. 先由 `C`、`H*W`、`seg_dim` 推导实际分组 `seg_dim` 与 `S`；
2. 三个方向的投影分别 reshape/permute 到对应二维平面后做线性层；
3. 重加权分支为 `C -> C//4 -> 3C` 的两层 MLP（中间 GELU）；
4. 三个权重沿 dim=0 softmax 后逐通道融合；
5. 推理路径 `proj_drop` 视为 identity。

## 3. 接口规范

### 算子原型

```python
weighted_permute_mlp(Tensor x, Tensor mlp_c_weight, Tensor mlp_c_bias, Tensor mlp_h_weight,
                     Tensor mlp_h_bias, Tensor mlp_w_weight, Tensor mlp_w_bias,
                     Tensor mlp_fc1_weight, Tensor mlp_fc1_bias, Tensor mlp_fc2_weight,
                     Tensor mlp_fc2_bias, Tensor proj_weight, Tensor proj_bias,
                     int dim, int seg_dim, bool qkv_bias=False, float proj_drop=0.0) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，[B, H, W, C] |
| mlp_c_weight | Tensor | 必选 | channel 方向投影权重，[C, C] |
| mlp_c_bias | Tensor | 必选 | channel 方向偏置，[C]，qkv_bias 为 True 时使用 |
| mlp_h_weight | Tensor | 必选 | height 方向投影权重，[H*S, H*S] |
| mlp_h_bias | Tensor | 必选 | height 方向偏置，[H*S]，qkv_bias 为 True 时使用 |
| mlp_w_weight | Tensor | 必选 | width 方向投影权重，[W*S, W*S] |
| mlp_w_bias | Tensor | 必选 | width 方向偏置，[W*S]，qkv_bias 为 True 时使用 |
| mlp_fc1_weight | Tensor | 必选 | 重加权 MLP 第一层权重，[C//4, C] |
| mlp_fc1_bias | Tensor | 必选 | 重加权 MLP 第一层偏置，[C//4] |
| mlp_fc2_weight | Tensor | 必选 | 重加权 MLP 第二层权重，[3C, C//4] |
| mlp_fc2_bias | Tensor | 必选 | 重加权 MLP 第二层偏置，[3C] |
| proj_weight | Tensor | 必选 | 输出投影权重，[C, C] |
| proj_bias | Tensor | 必选 | 输出投影偏置，[C] |
| dim | int | 必选 | 特征维度（实现以实际 channel 为准） |
| seg_dim | int | 必选 | 分组大小 |
| qkv_bias | bool | false | 三个方向投影是否使用偏置 |
| proj_drop | float | 0.0 | 输出 dropout 概率（推理路径为 identity） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, H, W, C] | float32 | 输出特征图 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |

### 规则与约束

- `S = C / seg_dim`；`seg_dim` 不整除 `C` 或 `H*W` 时向下调整到最大可行值（最小 1）
- `mlp_h_weight` / `mlp_w_weight` 的尺寸随实际 `seg_dim` 与 `H/W` 推导
- `qkv_bias == False` 时三个方向投影不使用偏置（对应偏置输入被忽略）
- 推理路径 `proj_drop` 视为 identity
- 输出与输入同 shape

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
WeightedPermuteMLP 算子 Torch Golden 参考实现

Weighted Permute MLP（ViP）：沿 channel / height / width 三个方向分别做线性投影，
再用学习到的 softmax 重加权系数融合，最后做输出投影。原实现内部随机初始化的
nn.Linear 权重已提升为显式 tensor 入参。

公式:
    c_embed = x @ Wc^T + bc
    h_embed = permute(mlp_h(permute(x)), ...)    # 沿 H*S 维投影后还原
    w_embed = permute(mlp_w(permute(x)), ...)    # 沿 W*S 维投影后还原
    z = (c_embed + h_embed + w_embed).mean(H, W)
    a = softmax(reshape(fc2(GELU(fc1(z))), C, 3), dim=0)
    y = (c_embed * a[0] + w_embed * a[1] + h_embed * a[2]) @ Wp^T + bp
"""


def weighted_permute_mlp(
    x: torch.Tensor,
    mlp_c_weight: torch.Tensor,
    mlp_c_bias: torch.Tensor,
    mlp_h_weight: torch.Tensor,
    mlp_h_bias: torch.Tensor,
    mlp_w_weight: torch.Tensor,
    mlp_w_bias: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    dim: int,
    seg_dim: int,
    qkv_bias: bool = False,
    proj_drop: float = 0.0,
) -> torch.Tensor:
    """
    Weighted Permute MLP

    Args:
        x: 输入 [B, H, W, C]
        mlp_c_weight: channel 方向投影权重 [C, C]
        mlp_c_bias: channel 方向投影偏置 [C]（qkv_bias 为 True 时使用）
        mlp_h_weight: height 方向投影权重 [H*S, H*S]
        mlp_h_bias: height 方向投影偏置 [H*S]（qkv_bias 为 True 时使用）
        mlp_w_weight: width 方向投影权重 [W*S, W*S]
        mlp_w_bias: width 方向投影偏置 [W*S]（qkv_bias 为 True 时使用）
        mlp_fc1_weight: 重加权 MLP 第一层权重 [C//4, C]
        mlp_fc1_bias: 重加权 MLP 第一层偏置 [C//4]
        mlp_fc2_weight: 重加权 MLP 第二层权重 [3*C, C//4]
        mlp_fc2_bias: 重加权 MLP 第二层偏置 [3*C]
        proj_weight: 输出投影权重 [C, C]
        proj_bias: 输出投影偏置 [C]
        dim: 特征维度（实现以实际输入 channel 为准）
        seg_dim: 分组大小
        qkv_bias: 三个方向投影是否使用偏置
        proj_drop: 输出 dropout 概率（推理路径恒为 identity）

    Returns:
        输出 [B, H, W, C]
    """
    b, h, w, c = x.shape
    if b == 0 or h == 0 or w == 0 or c == 0:
        return x
    dim = c
    seg_dim = min(seg_dim, c, h * w)
    seg_dim = max(1, seg_dim)
    while c % seg_dim != 0 or (h * w) % seg_dim != 0:
        seg_dim -= 1
        if seg_dim <= 1:
            seg_dim = 1
            break
    s = c // seg_dim

    bias_c = mlp_c_bias if qkv_bias else None
    bias_h = mlp_h_bias if qkv_bias else None
    bias_w = mlp_w_bias if qkv_bias else None

    # ---------- c_embed ----------
    c_embed = F.linear(x, mlp_c_weight, bias_c)                  # [B, H, W, C]

    # ---------- h_embed ----------
    h_embed = x.reshape(b, h, w, seg_dim, s).permute(0, 3, 2, 1, 4).contiguous()
    h_embed = h_embed.reshape(b, seg_dim, w, h * s)
    h_embed = F.linear(h_embed, mlp_h_weight, bias_h)
    h_embed = h_embed.reshape(b, seg_dim, w, h, s).permute(0, 3, 2, 1, 4).contiguous()
    h_embed = h_embed.reshape(b, h, w, c)

    # ---------- w_embed ----------
    w_embed = x.reshape(b, h, w, seg_dim, s).permute(0, 3, 1, 2, 4).contiguous()
    w_embed = w_embed.reshape(b, seg_dim, h, w * s)
    w_embed = F.linear(w_embed, mlp_w_weight, bias_w)
    w_embed = w_embed.reshape(b, seg_dim, h, w, s).permute(0, 2, 3, 1, 4).contiguous()
    w_embed = w_embed.reshape(b, h, w, c)

    # ---------- 重加权 ----------
    weight = (c_embed + h_embed + w_embed).permute(0, 3, 1, 2).contiguous()
    weight = weight.flatten(2).mean(dim=2)                       # [B, C]
    weight = F.linear(weight, mlp_fc1_weight, mlp_fc1_bias)
    weight = F.gelu(weight)
    weight = F.linear(weight, mlp_fc2_weight, mlp_fc2_bias)      # [B, 3C]
    weight = weight.reshape(b, c, 3).permute(2, 0, 1)            # [3, B, C]
    weight = weight.softmax(dim=0)
    weight = weight.unsqueeze(2).unsqueeze(2)                    # [3, B, C, 1, 1]

    x_out = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]
    return F.linear(x_out, proj_weight, proj_bias)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, W, C = 4, 9, 9, 256
seg_dim = 2
S = C // seg_dim
x = torch.randn(B, H, W, C)
args = [
    x,
    torch.randn(C, C) * 0.05, torch.zeros(C),
    torch.randn(H * S, H * S) * 0.05, torch.zeros(H * S),
    torch.randn(W * S, W * S) * 0.05, torch.zeros(W * S),
    torch.randn(C // 4, C) * 0.05, torch.zeros(C // 4),
    torch.randn(3 * C, C // 4) * 0.05, torch.zeros(3 * C),
    torch.randn(C, C) * 0.05, torch.zeros(C),
]
y = cann_bench.weighted_permute_mlp(*args, dim=C, seg_dim=seg_dim,
                                    qkv_bias=False, proj_drop=0.0)
```

### 测试用例覆盖

- dtype：float32（源用例唯一 dtype）
- C ∈ {64, 128, 256, 512}；seg_dim ∈ {2, 4, 8}（含不整除自动降为 1 的场景）
- H/W 对齐与非对齐（4~11），非方形（5x7、7x11 等）
- B ∈ {1, 2, 4}
- qkv_bias：False / True 两种路径
