# SequentialPolarizedSelfAttention 算子 API 描述

## 1. 算子简介

Sequential Polarized Self-Attention 是一个视觉注意力模块，按顺序串联**通道极化注意力**与
**空间极化注意力**：通道支路用 1x1 卷积生成 value/query，经 softmax 与矩阵乘得到通道描述，
再经过 1x1 卷积与 LayerNorm 得到通道门控；空间支路在通道加权后的特征上用 1x1 卷积与全局
平均池化生成 query/value，经 softmax 与矩阵乘得到逐位置的空间门控。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉主干网络中的注意力增强
- 需要同时建模通道重要性与空间位置重要性的特征重标定

**算子特征**：
- 难度等级：L4（VisionAttention）
- 13 个张量输入（输入特征图 + 5 个 1x1 卷积的权重/偏置 + LayerNorm affine）+ 3 个接口兼容属性
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16，全程保持输入 dtype

## 2. 算子定义

### 数学公式

设输入为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，$C_2=\max(1,\lfloor C/2\rfloor)$。

通道极化注意力：

$$
\mathbf{v} = \mathrm{Conv}_{1\times1}(\mathbf{x})_v \in \mathbb{R}^{B\times C_2\times HW},\qquad
\mathbf{q} = \mathrm{softmax}\!\left(\mathrm{Conv}_{1\times1}(\mathbf{x})_q\right) \in \mathbb{R}^{B\times HW\times 1}
$$

$$
\mathbf{z} = \mathrm{Conv}_{1\times1}\!\left(\mathbf{v}\,\mathbf{q}\right) \in \mathbb{R}^{B\times C\times 1\times 1},\qquad
\mathbf{w}_c = \sigma\!\left(\mathrm{LayerNorm}(\mathbf{z})\right)
$$

$$
\mathbf{y}_c = \mathbf{w}_c \odot \mathbf{x}
$$

空间极化注意力：

$$
\mathbf{v}_s = \mathrm{Conv}_{1\times1}(\mathbf{y}_c)_v \in \mathbb{R}^{B\times C_2\times HW},\qquad
\mathbf{q}_s = \mathrm{softmax}\!\left(\mathrm{GAP}\left(\mathrm{Conv}_{1\times1}(\mathbf{y}_c)_q\right)\right) \in \mathbb{R}^{B\times 1\times C_2}
$$

$$
\mathbf{w}_s = \sigma\!\left(\mathbf{q}_s\,\mathbf{v}_s\right) \in \mathbb{R}^{B\times 1\times H\times W},\qquad
\mathbf{y} = \mathbf{w}_s \odot \mathbf{y}_c
$$

其中 $\sigma$ 为 sigmoid，LayerNorm 的 normalized_shape 为 $C$，eps 取 PyTorch 默认 1e-5。

### 实现建议

1. 通道支路：对 `x` 依次做 5 个 1x1 卷积中的前三个（value/query/output），注意 value 输出
   reshape 为 `[B, C2, H*W]`、query 输出 reshape 为 `[B, H*W, 1]` 并在 dim=1 上做 softmax。
2. `v @ q` 得到 `[B, C2, 1]`，须 unsqueeze 到 `[B, C2, 1, 1]` 再做 output 1x1 卷积；
   LayerNorm 作用在最后一维（长度 C）上，再 permute/reshape 为 `[B, C, 1, 1]` 门控。
3. 空间支路：query 1x1 卷积后先做 `adaptive_avg_pool2d(..., (1,1))`，再 reshape 为
   `[B, C2, 1]`、permute 成 `[B, 1, C2]` 后对 dim=-1 做 softmax；`q @ v` 得到 `[B, 1, H*W]`。
4. 全程使用输入 dtype 计算，不做内部升精度；`mix_weight/use_channel/scale_factor`
   仅为接口兼容属性，不参与计算。

## 3. 接口规范

### 算子原型

```python
cann_bench.sequential_polarized_self_attention(Tensor x, Tensor ch_wv_weight, Tensor ch_wv_bias, Tensor ch_wq_weight, Tensor ch_wq_bias, Tensor ch_wz_weight, Tensor ch_wz_bias, Tensor ln_weight, Tensor ln_bias, Tensor sp_wv_weight, Tensor sp_wv_bias, Tensor sp_wq_weight, Tensor sp_wq_bias, float mix_weight, bool use_channel, float scale_factor) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必选 | 描述 |
|------|------|------|------|
| x | Tensor | 是 | 输入特征图 [B, C, H, W] |
| ch_wv_weight | Tensor | 是 | 通道注意力 value 1x1 卷积权重 [C2, C, 1, 1] |
| ch_wv_bias | Tensor | 是 | 通道注意力 value 1x1 卷积偏置 [C2] |
| ch_wq_weight | Tensor | 是 | 通道注意力 query 1x1 卷积权重 [1, C, 1, 1] |
| ch_wq_bias | Tensor | 是 | 通道注意力 query 1x1 卷积偏置 [1] |
| ch_wz_weight | Tensor | 是 | 通道注意力 output 1x1 卷积权重 [C, C2, 1, 1] |
| ch_wz_bias | Tensor | 是 | 通道注意力 output 1x1 卷积偏置 [C] |
| ln_weight | Tensor | 是 | LayerNorm 仿射权重 [C] |
| ln_bias | Tensor | 是 | LayerNorm 仿射偏置 [C] |
| sp_wv_weight | Tensor | 是 | 空间注意力 value 1x1 卷积权重 [C2, C, 1, 1] |
| sp_wv_bias | Tensor | 是 | 空间注意力 value 1x1 卷积偏置 [C2] |
| sp_wq_weight | Tensor | 是 | 空间注意力 query 1x1 卷积权重 [C2, C, 1, 1] |
| sp_wq_bias | Tensor | 是 | 空间注意力 query 1x1 卷积偏置 [C2] |

### 属性说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| mix_weight | float | - | 接口兼容属性，源实现未使用 |
| use_channel | bool | - | 接口兼容属性，源实现未使用 |
| scale_factor | float | - | 接口兼容属性，源实现未使用 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, C, H, W] | 与输入相同 | 输出特征图 [B, C, H, W] |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有张量输入的 dtype 必须一致。
- x: [B, C, H, W]；C2 = max(1, C // 2)；ch_wv_weight/sp_wv_weight/sp_wq_weight: [C2, C, 1, 1]；ch_wq_weight: [1, C, 1, 1]；ch_wz_weight: [C, C2, 1, 1]；ln_weight/ln_bias: [C]；y: [B, C, H, W]

## 4. 精度要求

- 输出与输入 `x` 同 shape、同 dtype；golden 全程保持输入 dtype（源实现不升精度）。
- 网络输出为两层 sigmoid 门控连乘，数值范围稳定在输入量级附近。
- 评测按常规 fp16/bf16/fp32 阈值比较；golden 仅依赖显式权重输入，不含随机数。
- 与源 `Model` 逐 case 对比（源内部 `_cache` 中的 1x1 卷积与 LayerNorm 权重直接取出），
  20 个 case 的 maxdiff 均为 0。

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
Sequential Polarized Self-Attention 算子 Torch Golden 参考实现

依次执行通道极化注意力与空间极化注意力，并在通道注意力的门控上使用 LayerNorm：
    ch_wv_out = conv1x1_v(x).view(B, C2, HW)
    ch_wq_out = softmax(conv1x1_q(x).view(B, HW, 1), dim=1)
    ch_z      = conv1x1_z(ch_wv_out @ ch_wq_out)
    ch_weight = sigmoid(LayerNorm(ch_z)).view(B, C, 1, 1)
    channel_out = ch_weight * x
    sp_wv_out = conv1x1_v2(channel_out).view(B, C2, HW)
    sp_wq_out = softmax(adaptive_avg_pool2d(conv1x1_q2(channel_out), 1).view(B, 1, C2), dim=-1)
    sp_weight = sigmoid(sp_wq_out @ sp_wv_out).view(B, 1, H, W)
    y = sp_weight * channel_out

源实现全程保持输入 dtype（内部不升精度）。mix_weight / use_channel / scale_factor
仅为接口兼容参数，源 forward 未使用，本 golden 同样忽略。
"""


def sequential_polarized_self_attention(
    x: torch.Tensor,
    ch_wv_weight: torch.Tensor,
    ch_wv_bias: torch.Tensor,
    ch_wq_weight: torch.Tensor,
    ch_wq_bias: torch.Tensor,
    ch_wz_weight: torch.Tensor,
    ch_wz_bias: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    sp_wv_weight: torch.Tensor,
    sp_wv_bias: torch.Tensor,
    sp_wq_weight: torch.Tensor,
    sp_wq_bias: torch.Tensor,
    mix_weight: float = None,
    use_channel: bool = None,
    scale_factor: float = None,
) -> torch.Tensor:
    """
    Sequential Polarized Self-Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        ch_wv_weight: 通道注意力 value 1x1 卷积权重 [C2, C, 1, 1]
        ch_wv_bias: 通道注意力 value 1x1 卷积偏置 [C2]
        ch_wq_weight: 通道注意力 query 1x1 卷积权重 [1, C, 1, 1]
        ch_wq_bias: 通道注意力 query 1x1 卷积偏置 [1]
        ch_wz_weight: 通道注意力 output 1x1 卷积权重 [C, C2, 1, 1]
        ch_wz_bias: 通道注意力 output 1x1 卷积偏置 [C]
        ln_weight: LayerNorm 仿射权重 [C]
        ln_bias: LayerNorm 仿射偏置 [C]
        sp_wv_weight: 空间注意力 value 1x1 卷积权重 [C2, C, 1, 1]
        sp_wv_bias: 空间注意力 value 1x1 卷积偏置 [C2]
        sp_wq_weight: 空间注意力 query 1x1 卷积权重 [C2, C, 1, 1]
        sp_wq_bias: 空间注意力 query 1x1 卷积偏置 [C2]
        mix_weight: 接口兼容参数，源实现未使用
        use_channel: 接口兼容参数，源实现未使用
        scale_factor: 接口兼容参数，源实现未使用

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    c2 = max(1, c // 2)

    # ---------- 通道极化注意力 ----------
    ch_wv_out = F.conv2d(x, ch_wv_weight, ch_wv_bias).view(b, c2, -1)
    ch_wq_out = F.conv2d(x, ch_wq_weight, ch_wq_bias).view(b, -1, 1)
    ch_wq_out = F.softmax(ch_wq_out, dim=1)
    ch_wz_out = torch.matmul(ch_wv_out, ch_wq_out).unsqueeze(-1)
    ch_wz_out = F.conv2d(ch_wz_out, ch_wz_weight, ch_wz_bias).view(b, c, 1).permute(0, 2, 1)
    ch_weight = torch.sigmoid(F.layer_norm(ch_wz_out, (c,), ln_weight, ln_bias))
    ch_weight = ch_weight.permute(0, 2, 1).view(b, c, 1, 1)
    channel_out = ch_weight * x

    # ---------- 空间极化注意力 ----------
    sp_wv_out = F.conv2d(channel_out, sp_wv_weight, sp_wv_bias).view(b, c2, -1)
    sp_wq_out = F.conv2d(channel_out, sp_wq_weight, sp_wq_bias)
    sp_wq_out = F.adaptive_avg_pool2d(sp_wq_out, (1, 1)).view(b, -1, 1)
    sp_wq_out = F.softmax(sp_wq_out.permute(0, 2, 1), dim=-1)
    sp_wz_out = torch.matmul(sp_wq_out, sp_wv_out).view(b, 1, h, w)
    sp_weight = torch.sigmoid(sp_wz_out)

    return sp_weight * channel_out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 64, 17, 19
c2 = max(1, C // 2)

x = torch.randn(B, C, H, W, dtype=torch.float16, device="npu")
ch_wv_weight = torch.randn(c2, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
ch_wv_bias = torch.zeros(c2, dtype=torch.float16, device="npu")
ch_wq_weight = torch.randn(1, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
ch_wq_bias = torch.zeros(1, dtype=torch.float16, device="npu")
ch_wz_weight = torch.randn(C, c2, 1, 1, dtype=torch.float16, device="npu") * 0.05
ch_wz_bias = torch.zeros(C, dtype=torch.float16, device="npu")
ln_weight = torch.ones(C, dtype=torch.float16, device="npu")
ln_bias = torch.zeros(C, dtype=torch.float16, device="npu")
sp_wv_weight = torch.randn(c2, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
sp_wv_bias = torch.zeros(c2, dtype=torch.float16, device="npu")
sp_wq_weight = torch.randn(c2, C, 1, 1, dtype=torch.float16, device="npu") * 0.05
sp_wq_bias = torch.zeros(c2, dtype=torch.float16, device="npu")

y = cann_bench.sequential_polarized_self_attention(
    x, ch_wv_weight, ch_wv_bias, ch_wq_weight, ch_wq_bias,
    ch_wz_weight, ch_wz_bias, ln_weight, ln_bias,
    sp_wv_weight, sp_wv_bias, sp_wq_weight, sp_wq_bias,
    0.3, False, 0.1,
)
# y: [2, 64, 17, 19], float16
```
