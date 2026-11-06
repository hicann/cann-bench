# ParallelPolarizedSelfAttention 算子 API 描述

## 1. 算子简介

Parallel Polarized Self-Attention 由通道支路与空间支路并行组成：通道支路用
`1x1 conv -> 空间 softmax -> 1x1 conv -> LayerNorm -> sigmoid` 生成逐通道门控；
空间支路用 `1x1 conv -> 全局平均池化 softmax -> 矩阵乘 -> sigmoid` 生成逐像素门控。
两路门控分别与输入相乘后相加。

**主要应用场景**：
- 视觉主干网络中的轻量注意力增强（极化自注意力）
- 需要同时强调重要通道与重要空间位置的检测/分割任务

**算子特征**：
- 难度等级：L4（VisionAttention）
- 13 个张量输入（输入特征图 + 6 个 Conv2d 与 1 个 LayerNorm 的全部权重/偏置）
- 4 个属性：`use_channel` / `use_spatial` / `mix_weight` / `scale_factor`
- 单输出，shape 与输入 `x` 一致
- 支持 float16 / float32 / bfloat16

**关于属性的重要说明**：源 `Model.forward` 的签名只有 `(x, scale_factor=None)`，且
`scale_factor` 在函数体内从未被使用；`use_channel` / `use_spatial` / `mix_weight` 只出现在
源 json 的用例属性中，源 forward 既不接收也不使用。因此这四个属性对本算子的输出**没有
任何影响**，golden 中仅作接口兼容占位。

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$，$c_2=\max(1,\lfloor C/2\rfloor)$。

通道支路：

$$
\mathbf{v}_c=\mathrm{Conv1x1}_{C\to c_2}(\mathbf{x})_{\text{flat}}\in\mathbb{R}^{B\times c_2\times HW},\qquad
\mathbf{q}_c=\mathrm{softmax}\!\left(\mathrm{Conv1x1}_{C\to1}(\mathbf{x})_{\text{flat}},\,dim=1\right)
$$

$$
\mathbf{z}_c=\mathrm{Conv1x1}_{c_2\to C}\!\left(\mathbf{v}_c\mathbf{q}_c\right)\in\mathbb{R}^{B\times C\times1},
\qquad
\mathbf{w}_c=\sigma\!\left(\mathrm{LayerNorm}_C(\mathbf{z}_c)\right)
$$

空间支路：

$$
\mathbf{v}_s=\mathrm{Conv1x1}_{C\to c_2}(\mathbf{x})_{\text{flat}},\qquad
\mathbf{q}_s=\mathrm{softmax}\!\left(\mathrm{GAP}(\mathrm{Conv1x1}_{C\to c_2}(\mathbf{x})),\,dim=-1\right)
$$

$$
\mathbf{w}_s=\sigma\!\left(\mathbf{q}_s\mathbf{v}_s\right)\in\mathbb{R}^{B\times1\times H\times W}
$$

输出：

$$
\mathbf{y}=\mathbf{w}_s\odot\mathbf{x}+\mathbf{w}_c\odot\mathbf{x}
$$

### 实现建议

1. 通道支路的 `ch_wq` 是 `C -> 1` 的 1x1 卷积，flatten 后形状 `[B, HW, 1]`，**沿空间维**
   （dim=1）做 softmax；`ch_wv @ ch_wq` 得到 `[B, c2, 1]`，经 `ch_wz` 升回 C 通道。
2. LayerNorm 作用在长度 C 的最后一维上（输入 `[B, 1, C]`），eps 使用 PyTorch 默认的
   `1e-5`，affine 权重/偏置显式输入。
3. 空间支路的 `sp_wq` 经全局平均池化后形状 `[B, c2, 1]`，转置为 `[B, 1, c2]` 后对 c2 做
   softmax；与 `sp_wv [B, c2, HW]` 矩阵乘得到 `[B, 1, HW]`，再 view 成 `[B, 1, H, W]`。
4. 最后按源实现顺序返回 `spatial_out + channel_out`。

## 3. 接口规范

### 算子原型

```python
cann_bench.parallel_polarized_self_attention(
    x,
    ch_wv_weight, ch_wv_bias, ch_wq_weight, ch_wq_bias,
    ch_wz_weight, ch_wz_bias, ln_weight, ln_bias,
    sp_wv_weight, sp_wv_bias, sp_wq_weight, sp_wq_bias,
    use_channel=False, use_spatial=False, mix_weight=0.5, scale_factor=0.1,
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| x | `[B, C, H, W]` | 输入特征图 |
| ch_wv_weight / ch_wv_bias | `[c2, C, 1, 1]` / `[c2]` | 通道支路 value 1x1 卷积 |
| ch_wq_weight / ch_wq_bias | `[1, C, 1, 1]` / `[1]` | 通道支路 query 1x1 卷积 |
| ch_wz_weight / ch_wz_bias | `[C, c2, 1, 1]` / `[C]` | 通道支路输出 1x1 卷积 |
| ln_weight / ln_bias | `[C]` / `[C]` | 通道支路 LayerNorm 仿射参数 |
| sp_wv_weight / sp_wv_bias | `[c2, C, 1, 1]` / `[c2]` | 空间支路 value 1x1 卷积 |
| sp_wq_weight / sp_wq_bias | `[c2, C, 1, 1]` / `[c2]` | 空间支路 query 1x1 卷积 |
| use_channel | bool，默认 False | 源实现未使用 |
| use_spatial | bool，默认 False | 源实现未使用 |
| mix_weight | float，默认 0.5 | 源实现未使用 |
| scale_factor | float，默认 0.1 | 源实现未使用 |

其中 `c2 = max(1, C // 2)`。

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| y | `[B, C, H, W]` | 输出特征图，dtype 与输入一致 |

### 数据类型

输入与输出支持 `float16`、`float32`、`bfloat16`；所有张量使用同一 dtype，源实现全程保持
输入 dtype。

### 规则与约束

- `c2 = max(1, C // 2)`；C 为奇数时向下取整，C=1 时 c2=1。
- 通道支路 softmax 沿空间维（`dim=1`），空间支路 softmax 沿通道维（`dim=-1`），不可混淆。
- 四个属性对输出无影响，实现时可直接忽略。

## 4. 精度要求

golden 全程保持输入 dtype，无随机性。fp32 下期望与源实现逐位一致；fp16/bf16 参考相对
误差约 $10^{-3}$。

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
Parallel Polarized Self-Attention 算子 Torch Golden 参考实现

通道支路与空间支路并行计算：

通道支路:
    ch_wv = conv1x1_{C->C/2}(x).flatten(2)          # [B, C/2, HW]
    ch_wq = softmax(conv1x1_{C->1}(x).flatten(2), dim=1)  # [B, HW, 1]
    ch_wz = conv1x1_{C/2->C}(ch_wv @ ch_wq)         # [B, C, 1, 1]
    ch_w  = sigmoid(LayerNorm_C(ch_wz))             # [B, C, 1, 1]
空间支路:
    sp_wv = conv1x1_{C->C/2}(x).flatten(2)          # [B, C/2, HW]
    sp_wq = softmax(avgpool(conv1x1_{C->C/2}(x)), dim=-1)  # [B, 1, C/2]
    sp_w  = sigmoid(sp_wq @ sp_wv)                  # [B, 1, H, W]
输出:
    y = sp_w * x + ch_w * x

源实现全程保持输入 dtype（不做 fp32 升精度）。
注意：源 Model.forward 只接收 x 与 scale_factor，且 scale_factor 在函数体内未被使用；
use_channel / use_spatial / mix_weight 仅出现在源 json 中，源 forward 未接收、未使用。
因此本 golden 中这四个属性均为接口兼容占位，不参与计算。
"""


def parallel_polarized_self_attention(
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
    use_channel: bool = False,
    use_spatial: bool = False,
    mix_weight: float = 0.5,
    scale_factor: float = 0.1,
) -> torch.Tensor:
    """
    Parallel Polarized Self-Attention 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        ch_wv_weight: 通道支路 value 1x1 卷积权重 [C/2, C, 1, 1]
        ch_wv_bias: 通道支路 value 1x1 卷积偏置 [C/2]
        ch_wq_weight: 通道支路 query 1x1 卷积权重 [1, C, 1, 1]
        ch_wq_bias: 通道支路 query 1x1 卷积偏置 [1]
        ch_wz_weight: 通道支路输出 1x1 卷积权重 [C, C/2, 1, 1]
        ch_wz_bias: 通道支路输出 1x1 卷积偏置 [C]
        ln_weight: 通道支路 LayerNorm 仿射权重 [C]
        ln_bias: 通道支路 LayerNorm 仿射偏置 [C]
        sp_wv_weight: 空间支路 value 1x1 卷积权重 [C/2, C, 1, 1]
        sp_wv_bias: 空间支路 value 1x1 卷积偏置 [C/2]
        sp_wq_weight: 空间支路 query 1x1 卷积权重 [C/2, C, 1, 1]
        sp_wq_bias: 空间支路 query 1x1 卷积偏置 [C/2]
        use_channel: 源 forward 未使用（接口兼容占位）
        use_spatial: 源 forward 未使用（接口兼容占位）
        mix_weight: 源 forward 未使用（接口兼容占位）
        scale_factor: 源 forward 未使用（接口兼容占位）

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    if b == 0 or c == 0 or h == 0 or w == 0:
        return x
    c2 = max(1, c // 2)

    # ---------- 通道支路 ----------
    ch_wv_out = F.conv2d(x, ch_wv_weight, ch_wv_bias).view(b, c2, -1)
    ch_wq_out = F.conv2d(x, ch_wq_weight, ch_wq_bias).view(b, -1, 1)
    ch_wq_out = F.softmax(ch_wq_out, dim=1)
    ch_wz_out = torch.matmul(ch_wv_out, ch_wq_out).unsqueeze(-1)
    ch_wz_out = F.conv2d(ch_wz_out, ch_wz_weight, ch_wz_bias).view(b, c, 1).permute(0, 2, 1)
    ch_weight = torch.sigmoid(
        F.layer_norm(ch_wz_out, (c,), ln_weight, ln_bias)
    ).permute(0, 2, 1).view(b, c, 1, 1)
    channel_out = ch_weight * x

    # ---------- 空间支路 ----------
    sp_wv_out = F.conv2d(x, sp_wv_weight, sp_wv_bias).view(b, c2, -1)
    sp_wq_out = F.adaptive_avg_pool2d(
        F.conv2d(x, sp_wq_weight, sp_wq_bias), (1, 1)
    ).view(b, -1, 1)  # B, c2, 1
    sp_wq_out = F.softmax(sp_wq_out.permute(0, 2, 1), dim=-1)  # B, 1, c2
    sp_wz_out = torch.matmul(sp_wq_out, sp_wv_out).view(b, 1, h, w)
    sp_weight = torch.sigmoid(sp_wz_out)
    spatial_out = sp_weight * x

    return spatial_out + channel_out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 64, 23, 41
c2 = max(1, C // 2)
dt = torch.float16

x = torch.randn(B, C, H, W, dtype=dt, device="npu")

def conv_w(oc, ic):
    return torch.randn(oc, ic, 1, 1, dtype=dt, device="npu") * 0.05

y = cann_bench.parallel_polarized_self_attention(
    x,
    conv_w(c2, C), torch.zeros(c2, dtype=dt, device="npu"),
    conv_w(1, C), torch.zeros(1, dtype=dt, device="npu"),
    conv_w(C, c2), torch.zeros(C, dtype=dt, device="npu"),
    torch.ones(C, dtype=dt, device="npu"), torch.zeros(C, dtype=dt, device="npu"),
    conv_w(c2, C), torch.zeros(c2, dtype=dt, device="npu"),
    conv_w(c2, C), torch.zeros(c2, dtype=dt, device="npu"),
)
# y: [2, 64, 23, 41], float16
```
