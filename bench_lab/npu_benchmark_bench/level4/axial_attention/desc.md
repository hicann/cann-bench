# AxialAttention 算子 API 描述

## 1. 算子简介

Axial Attention 算子。对特征图分别沿高度轴（H）和宽度轴（W）做多头自注意力，两路结果相加后回写为原特征图布局。q / kv 的线性投影参数在两轴之间共享。

**主要应用场景**：
- 图像/视频骨干网络中的轴向注意力（Axial Attention）
- 需要同时建模行列两个方向依赖的视觉 Transformer 变体

**算子特征**：
- 难度等级：L4（Attention）
- 输入特征图 `x`：`[B, C, H, W]`
- 权重：共享的 `to_q` / `to_kv` 投影与 `to_out` 输出投影
- 输出：`[B, C, H, W]`
- 源实现的 `to_q` / `to_kv` 线性层无 bias，`to_out` 带 bias

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
q_H &= X_H W_q^\top,\quad [k_H, v_H] = X_H W_{kv}^\top \\
y_H &= \mathrm{OutProj}\big(\mathrm{softmax}(q_H k_H^\top / \sqrt{d_h})\, v_H\big) \\
q_W &= X_W W_q^\top,\quad [k_W, v_W] = X_W W_{kv}^\top \\
y_W &= \mathrm{OutProj}\big(\mathrm{softmax}(q_W k_W^\top / \sqrt{d_h})\, v_W\big) \\
y &= \mathrm{reshape}(y_H + y_W) \in \mathbb{R}^{B \times C \times H \times W}
\end{aligned}
$$

其中 `X_H` 为按行展开的序列 `(B*W, H, C)`，`X_W` 为按列展开的序列 `(B*H, W, C)`，`d_h = dim_heads`（未给出时为 `dim / heads`），`OutProj` 含 bias。


## 3. 接口规范

### 算子原型

```python
cann_bench.axial_attention(Tensor x, Tensor to_q_weight, Tensor to_kv_weight, Tensor to_out_weight, Tensor to_out_bias, int dim, int heads, int dim_heads) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图 `[B, C, H, W]` |
| to_q_weight | Tensor | 必选 | query 投影权重 `[heads*d_h, C]` |
| to_kv_weight | Tensor | 必选 | key/value 投影权重 `[2*heads*d_h, C]` |
| to_out_weight | Tensor | 必选 | 输出投影权重 `[C, heads*d_h]` |
| to_out_bias | Tensor | 必选 | 输出投影偏置 `[C]` |
| dim | int | 必选 | 特征维，须等于 `C` |
| heads | int | 必选 | 注意力头数 |
| dim_heads | int | 必选 | 每个头的维度 `d_h` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[B, C, H, W]` | 与输入相同 | 输出特征图 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `dim == x.shape[1] == C`
- `to_q_weight.shape == [heads*d_h, C]`，`to_kv_weight.shape == [2*heads*d_h, C]`，`to_out_weight.shape == [C, heads*d_h]`，`to_out_bias.shape == [C]`，其中 `d_h = dim_heads`
- `C % heads == 0`（当 `dim_heads` 未显式给出时要求 `dim_heads = dim / heads` 整除）
- 输出 dtype 与 `x` 一致
- golden 内部以 fp32 计算后回铸为输入 dtype

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
AxialAttention 算子 Torch Golden 参考实现

Axial Attention：沿高度轴与宽度轴分别做多头自注意力，两路输出相加。
    y_H = out_proj(softmax(q_H k_H^T / sqrt(d_h)) v_H)
    y_W = out_proj(softmax(q_W k_W^T / sqrt(d_h)) v_W)
    y   = reshape(y_H + y_W)  -> [B, C, H, W]
其中 q/kv 由共享的 to_q / to_kv 线性层得到，源实现的 to_q / to_kv 无 bias。
"""


def axial_attention(
    x: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_kv_weight: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    dim: int,
    heads: int,
    dim_heads: int,
) -> torch.Tensor:
    """
    Axial Attention（H 轴 + W 轴自注意力相加）

    公式:
        y_H = out_proj(softmax(q_H k_H^T / sqrt(d_h)) v_H)
        y_W = out_proj(softmax(q_W k_W^T / sqrt(d_h)) v_W)
        y   = reshape(y_H + y_W)  -> [B, C, H, W]

    Args:
        x: 输入特征图 [B, C, H, W]
        to_q_weight: query 投影权重 [heads * d_h, C]
        to_kv_weight: key/value 投影权重 [2 * heads * d_h, C]
        to_out_weight: 输出投影权重 [C, heads * d_h]
        to_out_bias: 输出投影偏置 [C]
        dim: 特征维（须等于 C）
        heads: 注意力头数
        dim_heads: 每个头的维度 d_h

    Returns:
        输出特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape
    effective_dim = int(dim)
    effective_heads = int(heads)
    head_dim = int(dim_heads) if dim_heads is not None else effective_dim // effective_heads

    x_perm = x.permute(0, 2, 3, 1).contiguous().float()  # [B, H, W, C]
    w_q = to_q_weight.float()
    w_kv = to_kv_weight.float()
    w_out = to_out_weight.float()
    b_out = to_out_bias.float()

    # Height axis: 每条序列长 H，共 B*W 条
    seq_h = x_perm.permute(0, 2, 1, 3).contiguous().view(b * w, h, c)
    q_h = F.linear(seq_h, w_q)
    kv_h = F.linear(seq_h, w_kv)
    k_h, v_h = kv_h.chunk(2, dim=-1)
    q_h = q_h.view(q_h.shape[0], q_h.shape[1], effective_heads, -1).transpose(1, 2)
    k_h = k_h.view(k_h.shape[0], k_h.shape[1], effective_heads, -1).transpose(1, 2)
    v_h = v_h.view(v_h.shape[0], v_h.shape[1], effective_heads, -1).transpose(1, 2)
    attn_h = F.softmax(torch.matmul(q_h, k_h.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1)
    out_h = torch.matmul(attn_h, v_h).transpose(1, 2).contiguous().view(seq_h.shape[0], seq_h.shape[1], -1)
    out_h = F.linear(out_h, w_out, b_out).view(b, w, h, c).permute(0, 2, 1, 3)

    # Width axis: 每条序列长 W，共 B*H 条
    seq_w = x_perm.view(b * h, w, c)
    q_w = F.linear(seq_w, w_q)
    kv_w = F.linear(seq_w, w_kv)
    k_w, v_w = kv_w.chunk(2, dim=-1)
    q_w = q_w.view(q_w.shape[0], q_w.shape[1], effective_heads, -1).transpose(1, 2)
    k_w = k_w.view(k_w.shape[0], k_w.shape[1], effective_heads, -1).transpose(1, 2)
    v_w = v_w.view(v_w.shape[0], v_w.shape[1], effective_heads, -1).transpose(1, 2)
    attn_w = F.softmax(torch.matmul(q_w, k_w.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1)
    out_w = torch.matmul(attn_w, v_w).transpose(1, 2).contiguous().view(seq_w.shape[0], seq_w.shape[1], -1)
    out_w = F.linear(out_w, w_out, b_out).view(b, h, w, c)

    out = out_h + out_w
    return out.permute(0, 3, 1, 2).contiguous().to(x.dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, C, H, W = 2, 384, 11, 15
heads, dh = 12, 32
inner = heads * dh
x = torch.randn(B, C, H, W, dtype=torch.float32, device="npu")
to_q_weight = torch.randn(inner, C, dtype=torch.float32, device="npu") * 0.05
to_kv_weight = torch.randn(2 * inner, C, dtype=torch.float32, device="npu") * 0.05
to_out_weight = torch.randn(C, inner, dtype=torch.float32, device="npu") * 0.05
to_out_bias = torch.randn(C, dtype=torch.float32, device="npu") * 0.05
y = cann_bench.axial_attention(x, to_q_weight, to_kv_weight, to_out_weight, to_out_bias, C, heads, dh)
# y.shape == (2, 384, 11, 15)
```
