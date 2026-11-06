# CoTAttention 算子 API 描述

## 1. 算子简介

CoTAttention 实现 Contextual Transformer（CoT）注意力模块的前向。与传统自注意力只在 value 上做全图 softmax 不同，CoT 先用分组卷积（depth-wise 形式）从输入中提取每个位置的 $K\times K$ 邻域上下文作为 key 分支；将 key 分支与输入拼接后经两层 1x1 卷积生成 $K\times K$ 个邻域注意力图，对 value 分支做逐位置加权求和，最后与 key 分支残差相加。

**主要应用场景**：
- 视觉主干网络中的上下文增强（CoTNet 系列）
- 需要局部邻域自注意力建模的图像分类/检测/分割

**算子特征**：
- 难度等级：L4（VisionAttention）
- 9 个必选张量输入：输入特征图 + key/value/注意力三组卷积的权重与偏置
- 1 个属性 `kernel_size`
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，$n=HW$，核大小 $K$（`kernel_size`），填充 $K//2$，分组数
$g = 4$（$C \bmod 4 = 0$）或 $2$（$C \bmod 2 = 0$）或 $1$，瓶颈宽度 $mid=\max(1, \lfloor C/4\rfloor)$。

$$
\begin{aligned}
\mathbf{k}_1 &= \mathrm{ReLU}\!\left(\mathrm{DepthWiseConv}_{K,g}(\mathbf{x})\right) \\
\mathbf{v} &= \mathrm{Conv}_{1\times1}(\mathbf{x}) \\
\mathbf{a} &= \mathrm{Conv}_{1\times1}\!\left(\mathrm{ReLU}\!\left(\mathrm{Conv}_{1\times1}\!\left([\mathbf{k}_1; \mathbf{x}]\right)\right)\right) \\
\mathbf{a} &= \frac{1}{K^2}\sum_{d=0}^{K^2-1}\mathrm{reshape}(\mathbf{a})_{:,:,d,:,:} \in \mathbb{R}^{B\times C\times n} \\
\mathbf{k}_2 &= \mathrm{softmax}_{n}(\mathbf{a}) \odot \mathbf{v} \\
\mathbf{y} &= \mathbf{k}_1 + \mathbf{k}_2
\end{aligned}
$$

其中三组卷积的权重与偏置全部由显式张量输入给出（源实现内部初始化的卷积参数已提升为算子输入）。

### 实现建议

1. key 分支：`F.conv2d(x, key_embed_weight, key_embed_bias, padding=K//2, groups=g)` 后接 ReLU。
2. value 分支：`F.conv2d(x, value_embed_weight, value_embed_bias)` 后 reshape 为 `[B, C, H*W]`。
3. 注意力分支：`cat([k1, x], dim=1)` → `Conv1x1 + ReLU` → `Conv1x1`，得到 `[B, K*K*C, H, W]`，reshape 为 `[B, C, K*K, H, W]` 并对 `K*K` 维求均值，得到 `[B, C, H*W]`。
4. 对最后一维做 softmax 后与 value 相乘，reshape 回 `[B, C, H, W]`，与 key 分支相加。

## 3. 接口规范

### 算子原型

```python
cann_bench.cot_attention(
    x,
    key_embed_weight, key_embed_bias,
    value_embed_weight, value_embed_bias,
    att_w1_weight, att_b1_bias,
    att_w2_weight, att_b2_bias,
    kernel_size=3,
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图 `[B, C, H, W]` |
| key_embed_weight | Tensor | 必选 | key 分支分组卷积权重 `[C, C // groups, K, K]` |
| key_embed_bias | Tensor | 必选 | key 分支分组卷积偏置 `[C]` |
| value_embed_weight | Tensor | 必选 | value 分支 1x1 卷积权重 `[C, C, 1, 1]` |
| value_embed_bias | Tensor | 必选 | value 分支 1x1 卷积偏置 `[C]` |
| att_w1_weight | Tensor | 必选 | 注意力第一层 1x1 卷积权重 `[mid, 2C, 1, 1]` |
| att_b1_bias | Tensor | 必选 | 注意力第一层 1x1 卷积偏置 `[mid]` |
| att_w2_weight | Tensor | 必选 | 注意力第二层 1x1 卷积权重 `[K*K*C, mid, 1, 1]` |
| att_b2_bias | Tensor | 必选 | 注意力第二层 1x1 卷积偏置 `[K*K*C]` |
| kernel_size | int | 3 | key 分支分组卷积核大小 `K`（奇数），`padding = K // 2` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, C, H, W]`（与 x 相同） | 与输入相同 | CoT 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `x` 必须为 4 维 `[B, C, H, W]`；所有张量输入 dtype 必须一致。
- 分组数按 `groups = 4 if C % 4 == 0 else (2 if C % 2 == 0 else 1)` 推导；`key_embed_weight` 的第二维必须等于 `C // groups`。
- 瓶颈宽度 `mid = max(1, C // 4)`，需与 `att_w1_weight` 的第 0 维及 `att_w2_weight` 的第 1 维一致。
- `att_w2_weight` 的第 0 维必须为 `K*K*C`；`padding = K // 2`。
- 卷积输出尺寸为 `H + 2*(K//2) - K + 1 = H`、`W + 2*(K//2) - K + 1 = W`（K 为奇数）。
- golden 内部统一升 fp32 计算后回铸为输入 dtype；不含随机数，全部权重来自入参。

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
CoTAttention 算子 Torch Golden 参考实现

Contextual Transformer (CoT) 注意力前向：key 分支用分组卷积提取 KxK 邻域上下文，
value 分支做 1x1 投影；两者与输入拼接后经两层 1x1 卷积生成 KxK 邻域注意力，
对 value 加权求和后与 key 分支残差相加。
公式:
    k1  = ReLU(DepthWiseConv_{K}(x))
    v   = Conv1x1(x)
    att = Conv1x1(ReLU(Conv1x1(cat([k1, x]))))
    att = mean_{K x K}(reshape(att, [B, C, K*K, H, W]))
    k2  = softmax_{H*W}(att) * v
    y   = k1 + k2
"""


def cot_attention(
    x: torch.Tensor,
    key_embed_weight: torch.Tensor,
    key_embed_bias: torch.Tensor,
    value_embed_weight: torch.Tensor,
    value_embed_bias: torch.Tensor,
    att_w1_weight: torch.Tensor,
    att_b1_bias: torch.Tensor,
    att_w2_weight: torch.Tensor,
    att_b2_bias: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    CoT 注意力模块前向

    公式:
        k1  = ReLU(DepthWiseConv_{K}(x))
        v   = Conv1x1(x)
        att = Conv1x1(ReLU(Conv1x1(cat([k1, x]))))
        att = mean_{K x K}(reshape(att, [B, C, K*K, H, W]))
        k2  = softmax_{H*W}(att) * v
        y   = k1 + k2

    Args:
        x: 输入特征图 [B, C, H, W]
        key_embed_weight: key 分支分组卷积权重 [C, C // groups, K, K]
        key_embed_bias: key 分支分组卷积偏置 [C]
        value_embed_weight: value 分支 1x1 卷积权重 [C, C, 1, 1]
        value_embed_bias: value 分支 1x1 卷积偏置 [C]
        att_w1_weight: 注意力第一层 1x1 卷积权重 [mid, 2C, 1, 1]
        att_b1_bias: 注意力第一层 1x1 卷积偏置 [mid]
        att_w2_weight: 注意力第二层 1x1 卷积权重 [K*K*C, mid, 1, 1]
        att_b2_bias: 注意力第二层 1x1 卷积偏置 [K*K*C]
        kernel_size: key 分支分组卷积核大小 K，padding = K // 2

    Returns:
        CoT 注意力输出 [B, C, H, W]
    """
    b, c, h, w = x.shape
    n = h * w
    k = int(kernel_size)
    padding = k // 2
    groups = 4 if c % 4 == 0 else (2 if c % 2 == 0 else 1)

    # 统一升 fp32 计算，避免 fp16/bf16 在长归约中溢出或精度损失
    x_f = x.float()
    k1 = F.conv2d(
        x_f, key_embed_weight.float(), key_embed_bias.float(), padding=padding, groups=groups
    )
    k1 = F.relu(k1)
    v = F.conv2d(x_f, value_embed_weight.float(), value_embed_bias.float()).view(b, c, n)

    y = torch.cat([k1, x_f], dim=1)
    att = F.conv2d(y, att_w1_weight.float(), att_b1_bias.float())
    att = F.relu(att)
    att = F.conv2d(att, att_w2_weight.float(), att_b2_bias.float())
    att = att.reshape(b, c, k * k, h, w).mean(dim=2).reshape(b, c, n)

    k2 = torch.softmax(att, dim=-1) * v
    k2 = k2.view(b, c, h, w)
    return (k1 + k2).to(dtype=x.dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W, K = 1, 384, 27, 23, 3
mid, groups = C // 4, 4

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
key_w = torch.randn(C, C // groups, K, K, dtype=torch.float16, device="npu") * 0.05
key_b = torch.randn(C, dtype=torch.float16, device="npu") * 0.05
val_w = torch.randn(C, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
val_b = torch.randn(C, dtype=torch.float16, device="npu") * 0.05
att_w1 = torch.randn(mid, 2 * C, 1, 1, dtype=torch.float16, device="npu") * 0.05
att_b1 = torch.randn(mid, dtype=torch.float16, device="npu") * 0.05
att_w2 = torch.randn(K * K * C, mid, 1, 1, dtype=torch.float16, device="npu") * 0.05
att_b2 = torch.randn(K * K * C, dtype=torch.float16, device="npu") * 0.05

y = cann_bench.cot_attention(
    x, key_w, key_b, val_w, val_b, att_w1, att_b1, att_w2, att_b2, K
)
# y.shape == (1, 384, 27, 23)
```
