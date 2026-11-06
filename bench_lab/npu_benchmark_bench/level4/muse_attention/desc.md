# MUSEAttention 算子 API 描述

## 1. 算子简介

MUSE（Multi-Scale）Attention 是面向视觉主干网络的多尺度自注意力模块。它将标准多头
缩放点积自注意力（MHA）与一条多尺度卷积分支并联：卷积分支对投影后的 value 序列做
kernel = 1/3/5 的 depthwise separable 一维卷积，三路结果由可学习的 softmax 动态参数
加权求和，经自适应平均池化对齐到 query 长度后与自注意力输出相加。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉主干网络中的多尺度全局建模
- 需要同时利用 token 间长程依赖与局部多尺度感受野的特征/序列建模

**算子特征**：
- 难度等级：L4（FusedComposite）
- 22 个张量输入（3 个主输入 + 19 组内部权重/偏置，全部显式化）+ 4 个整型属性
- 单输出，shape 为 `[B, nq, d_model]`
- 支持 float16 / float32 / bfloat16，源实现全程保持输入 dtype（无 fp32 升精度）

## 2. 算子定义

### 数学公式

设 $\mathbf{Q}\in\mathbb{R}^{B\times n_q\times d_{model}}$、
$\mathbf{K},\mathbf{V}\in\mathbb{R}^{B\times n_k\times d_{model}}$，投影矩阵
$W_q,W_k\in\mathbb{R}^{h d_k\times d_{model}}$、$W_v\in\mathbb{R}^{h d_v\times d_{model}}$、
$W_o\in\mathbb{R}^{d_{model}\times h d_v}$。

$$
\mathbf{q}=W_q\mathbf{Q},\quad \mathbf{k}=W_k\mathbf{K},\quad \mathbf{v}=W_v\mathbf{V}
$$

$$
\mathbf{A}=\mathrm{softmax}\!\left(\frac{\mathbf{q}\mathbf{k}^{\top}}{\sqrt{d_k}}\right),
\qquad
\mathbf{o}=W_o\left(\mathbf{A}\mathbf{v}\right)
$$

将 $\mathbf{v}$ 重排为 $\mathbf{v}_2\in\mathbb{R}^{B\times h d_v\times n_k}$，卷积分支：

$$
\mathbf{c}_1=\mathrm{Conv1d}_{1\times1}(\mathbf{v}_2)
$$

$$
\mathbf{c}_3=\mathrm{Conv1d}_{1\times1}\!\left(\mathrm{DWConv1d}_{k=3,\,groups=h d_v,\,p=1}(\mathbf{v}_2)\right)
$$

$$
\mathbf{c}_5=\mathrm{Conv1d}_{1\times1}\!\left(\mathrm{DWConv1d}_{k=5,\,groups=h d_v,\,p=2}(\mathbf{v}_2)\right)
$$

$$
\mathbf{w}=\mathrm{softmax}(\mathbf{dy\_paras})\in\mathbb{R}^{3},\qquad
\mathbf{o}_2=w_0\mathbf{c}_1+w_1\mathbf{c}_3+w_2\mathbf{c}_5
$$

$$
\mathbf{y}=\mathbf{o}+\left(\mathrm{AdaptiveAvgPool1d}_{n_q}(\mathbf{o}_2)\right)^{\top}
\quad(\text{当 } n_k\ne n_q\text{ 时才做池化})
$$

### 实现建议

1. Q/K/V/O 四组投影均用显式权重 + `F.linear`；q/k/v 分别 reshape + permute 成
   `[B, h, nq, d_k]`、`[B, h, d_k, nk]`、`[B, h, nk, d_v]`。
2. 注意力 logits 除以 `sqrt(d_k)` 后沿最后一维 softmax（无 dropout）。
3. 卷积分支的输入是**投影后**的 value：`v.permute(0,1,3,2).reshape(B, h*d_v, nk)`。
   kernel=1 没有 depthwise 层（恒等映射），只有 pointwise 卷积；kernel=3/5 的 depthwise
   卷积 `groups=h*d_v`，权重形状分别为 `[h*d_v, 1, 3]`、`[h*d_v, 1, 5]`。
4. `dy_paras` 先做 softmax，再按 $w_0,w_1,w_2$ 的顺序加权求和（保持源实现的加法顺序）。
5. 当 `nk != nq` 时对 `[B, d_model, nk]` 做 `adaptive_avg_pool1d` 到 `nq`，最后转置并与
   注意力输出相加。

## 3. 接口规范

### 算子原型

```python
cann_bench.muse_attention(
    queries, keys, values,
    fc_q_weight, fc_q_bias, fc_k_weight, fc_k_bias,
    fc_v_weight, fc_v_bias, fc_o_weight, fc_o_bias,
    conv1_weight, conv1_bias,
    conv3_depth_weight, conv3_depth_bias,
    conv3_point_weight, conv3_point_bias,
    conv5_depth_weight, conv5_depth_bias,
    conv5_point_weight, conv5_point_bias,
    dy_paras,
    d_model, d_k, d_v, h,
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| queries | `[B, nq, d_model]` | 查询 |
| keys | `[B, nk, d_model]` | 键 |
| values | `[B, nk, d_model]` | 值 |
| fc_q_weight / fc_q_bias | `[h*d_k, d_model]` / `[h*d_k]` | Q 投影权重/偏置 |
| fc_k_weight / fc_k_bias | `[h*d_k, d_model]` / `[h*d_k]` | K 投影权重/偏置 |
| fc_v_weight / fc_v_bias | `[h*d_v, d_model]` / `[h*d_v]` | V 投影权重/偏置 |
| fc_o_weight / fc_o_bias | `[d_model, h*d_v]` / `[d_model]` | 输出投影权重/偏置 |
| conv1_weight / conv1_bias | `[d_model, h*d_v, 1]` / `[d_model]` | kernel=1 pointwise 卷积 |
| conv3_depth_weight / conv3_depth_bias | `[h*d_v, 1, 3]` / `[h*d_v]` | kernel=3 depthwise 卷积（groups=h*d_v） |
| conv3_point_weight / conv3_point_bias | `[d_model, h*d_v, 1]` / `[d_model]` | kernel=3 分支 pointwise 卷积 |
| conv5_depth_weight / conv5_depth_bias | `[h*d_v, 1, 5]` / `[h*d_v]` | kernel=5 depthwise 卷积（groups=h*d_v） |
| conv5_point_weight / conv5_point_bias | `[d_model, h*d_v, 1]` / `[d_model]` | kernel=5 分支 pointwise 卷积 |
| dy_paras | `[3]` | 多尺度动态参数，源实现初始化为全 1 |
| d_model | int | 模型维度，须等于主输入最后一维 |
| d_k | int | 每个 head 的 q/k 维度 |
| d_v | int | 每个 head 的 v 维度 |
| h | int | head 数 |

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| y | `[B, nq, d_model]` | 融合输出 |

### 数据类型

输入与输出均支持 `float16`、`float32`、`bfloat16`；所有张量使用同一 dtype，源实现全程
保持输入 dtype 计算。

### 规则与约束

- `d_model` 必须等于 `queries/keys/values.shape[-1]`。
- `h*d_v` 既是卷积分支通道数，也是 depthwise 卷积的 groups。
- `nk != nq` 时依赖 `adaptive_avg_pool1d` 对齐序列长度。
- `dy_paras` 必须经过 softmax 后使用，不得直接加权。

## 4. 精度要求

golden 全程使用输入 dtype 计算（与源实现一致），无随机性，仅依赖显式权重输入。
fp32 下期望与源实现逐位一致；fp16/bf16 参考相对误差约 $10^{-3}$。

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

import math

import torch
import torch.nn.functional as F

"""
MUSE Attention 算子 Torch Golden 参考实现

MUSE (Multi-Scale) Attention 将标准多头缩放点积自注意力与多尺度卷积分支并联：
卷积分支对投影后的 value 依次做 depthwise + pointwise 一维卷积（核大小 1/3/5），
三路输出由可学习的 softmax 动态参数加权求和，再经自适应平均池化对齐到 query 长度，
最后与自注意力输出逐元素相加。

源实现全程保持输入 dtype（不做 fp32 升精度）。
"""


def muse_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    fc_q_weight: torch.Tensor,
    fc_q_bias: torch.Tensor,
    fc_k_weight: torch.Tensor,
    fc_k_bias: torch.Tensor,
    fc_v_weight: torch.Tensor,
    fc_v_bias: torch.Tensor,
    fc_o_weight: torch.Tensor,
    fc_o_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv3_depth_weight: torch.Tensor,
    conv3_depth_bias: torch.Tensor,
    conv3_point_weight: torch.Tensor,
    conv3_point_bias: torch.Tensor,
    conv5_depth_weight: torch.Tensor,
    conv5_depth_bias: torch.Tensor,
    conv5_point_weight: torch.Tensor,
    conv5_point_bias: torch.Tensor,
    dy_paras: torch.Tensor,
    d_model: int,
    d_k: int,
    d_v: int,
    h: int,
) -> torch.Tensor:
    """
    MUSE 多尺度注意力前向

    Args:
        queries: 查询 [B, nq, d_model]
        keys: 键 [B, nk, d_model]
        values: 值 [B, nk, d_model]
        fc_q_weight: Q 投影权重 [h*d_k, d_model]
        fc_q_bias: Q 投影偏置 [h*d_k]
        fc_k_weight: K 投影权重 [h*d_k, d_model]
        fc_k_bias: K 投影偏置 [h*d_k]
        fc_v_weight: V 投影权重 [h*d_v, d_model]
        fc_v_bias: V 投影偏置 [h*d_v]
        fc_o_weight: 输出投影权重 [d_model, h*d_v]
        fc_o_bias: 输出投影偏置 [d_model]
        conv1_weight: 1 核 pointwise 卷积权重 [d_model, h*d_v, 1]
        conv1_bias: 1 核 pointwise 卷积偏置 [d_model]
        conv3_depth_weight: 3 核 depthwise 卷积权重 [h*d_v, 1, 3]（groups=h*d_v）
        conv3_depth_bias: 3 核 depthwise 卷积偏置 [h*d_v]
        conv3_point_weight: 3 核 pointwise 卷积权重 [d_model, h*d_v, 1]
        conv3_point_bias: 3 核 pointwise 卷积偏置 [d_model]
        conv5_depth_weight: 5 核 depthwise 卷积权重 [h*d_v, 1, 5]（groups=h*d_v）
        conv5_depth_bias: 5 核 depthwise 卷积偏置 [h*d_v]
        conv5_point_weight: 5 核 pointwise 卷积权重 [d_model, h*d_v, 1]
        conv5_point_bias: 5 核 pointwise 卷积偏置 [d_model]
        dy_paras: 多尺度动态参数 [3]，源实现初始化为全 1
        d_model: 模型维度
        d_k: 每个 head 的 query/key 维度
        d_v: 每个 head 的 value 维度
        h: head 数

    Returns:
        输出 [B, nq, d_model]
    """
    b_s, nq = queries.shape[:2]
    nk = keys.shape[1]
    in_ch = h * d_v

    # ---------- 自注意力分支 ----------
    q = F.linear(queries, fc_q_weight, fc_q_bias).view(b_s, nq, h, d_k).permute(0, 2, 1, 3)
    k = F.linear(keys, fc_k_weight, fc_k_bias).view(b_s, nk, h, d_k).permute(0, 2, 3, 1)
    v = F.linear(values, fc_v_weight, fc_v_bias).view(b_s, nk, h, d_v).permute(0, 2, 1, 3)

    att = torch.matmul(q, k) / math.sqrt(d_k)
    att = F.softmax(att, dim=-1)

    out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, h * d_v)
    out = F.linear(out, fc_o_weight, fc_o_bias)

    # ---------- 多尺度卷积分支 ----------
    v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)

    # kernel=1：depthwise 退化为恒等映射，只有 pointwise 卷积
    conv1_out = F.conv1d(v2, conv1_weight, conv1_bias)
    # kernel=3：depthwise(groups=in_ch) + pointwise
    mid3 = F.conv1d(v2, conv3_depth_weight, conv3_depth_bias, padding=1, groups=in_ch)
    conv3_out = F.conv1d(mid3, conv3_point_weight, conv3_point_bias)
    # kernel=5：depthwise(groups=in_ch) + pointwise
    mid5 = F.conv1d(v2, conv5_depth_weight, conv5_depth_bias, padding=2, groups=in_ch)
    conv5_out = F.conv1d(mid5, conv5_point_weight, conv5_point_bias)

    # 动态参数 softmax 加权求和
    w = F.softmax(dy_paras, dim=0)
    out2 = w[0] * conv1_out + w[1] * conv3_out + w[2] * conv5_out

    # 序列长度对齐
    if nk != nq:
        out2 = F.adaptive_avg_pool1d(out2, nq)
    out2 = out2.permute(0, 2, 1)

    return out + out2
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, nq, nk = 2, 17, 29
d_model, d_k, d_v, h = 64, 32, 32, 2
in_ch = h * d_v
dt = torch.float16

queries = torch.randn(B, nq, d_model, dtype=dt, device="npu")
keys = torch.randn(B, nk, d_model, dtype=dt, device="npu")
values = torch.randn(B, nk, d_model, dtype=dt, device="npu")

def w(*shape):
    return (torch.randn(*shape, dtype=dt, device="npu") * 0.05)

y = cann_bench.muse_attention(
    queries, keys, values,
    w(h * d_k, d_model), torch.zeros(h * d_k, dtype=dt, device="npu"),
    w(h * d_k, d_model), torch.zeros(h * d_k, dtype=dt, device="npu"),
    w(h * d_v, d_model), torch.zeros(h * d_v, dtype=dt, device="npu"),
    w(d_model, h * d_v), torch.zeros(d_model, dtype=dt, device="npu"),
    w(d_model, in_ch, 1), torch.zeros(d_model, dtype=dt, device="npu"),
    w(in_ch, 1, 3), torch.zeros(in_ch, dtype=dt, device="npu"),
    w(d_model, in_ch, 1), torch.zeros(d_model, dtype=dt, device="npu"),
    w(in_ch, 1, 5), torch.zeros(in_ch, dtype=dt, device="npu"),
    w(d_model, in_ch, 1), torch.zeros(d_model, dtype=dt, device="npu"),
    torch.ones(3, dtype=dt, device="npu"),
    d_model, d_k, d_v, h,
)
# y: [2, 17, 64], float16
```
