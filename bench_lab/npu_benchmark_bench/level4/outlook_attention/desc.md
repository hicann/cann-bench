# OutlookAttention 算子 API 描述

## 1. 算子简介

Outlook Attention 是一种局部窗口注意力：对特征图的每个像素，用 `unfold` 展开其
$k\times k$ 邻域作为 value，再由线性层直接预测该邻域的注意力权重（softmax 归一化），
加权聚合后 `fold` 回原空间尺寸，最后做输出投影。它避免了全局 attention 的二次复杂度。

**主要应用场景**：
- 高分辨率视觉主干网络中的局部窗口注意力（如 VOLO 系列）
- 需要以线性复杂度建模像素级局部关系的图像/视频任务

**算子特征**：
- 难度等级：L4（VisionAttention）
- 7 个张量输入（输入特征图 + 3 组 Linear 权重/偏置，全部显式化）+ 4 个属性
- 输入输出均为 NHWC `[B, H, W, C]`
- 支持 float16 / float32 / bfloat16
- **精度策略**：权重按输入 dtype 给出，源实现将计算链整体升到 fp32（默认，可用环境变量
  `REF_COMPUTE_DTYPE` 覆盖），结果再 cast 回输入 dtype；本 golden 固定使用 fp32 计算

## 2. 算子定义

### 数学公式

设输入 $\mathbf{x}\in\mathbb{R}^{B\times H\times W\times C}$，有效核
$k=\min(\text{kernel\_size},H,W)$（仅当 kernel_size 超过 H 或 W 时收缩），
注意力维度 $A=(C//n_h)\cdot n_h$，head 维度 $d=A/n_h$。

$$
\mathbf{V}=\mathrm{Linear}_{C\to A}(\mathbf{x})^{\top}\in\mathbb{R}^{B\times A\times H\times W}
$$

$$
\mathbf{V}_u=\mathrm{Unfold}_{k,p,s}(\mathbf{V})\in\mathbb{R}^{B\times A k^2\times L},
\qquad
\mathbf{A}_0=\mathrm{Linear}_{C\to k^4 n_h}\!\left(\mathrm{AvgPool}_{s}(\mathbf{x})\right)
$$

$$
\mathbf{A}=\mathrm{softmax}\!\left(\frac{\mathrm{reshape}(\mathbf{A}_0)}{\sqrt{d}}\right)
\in\mathbb{R}^{B\times n_h\times L\times k^2\times k^2}
$$

$$
\mathbf{O}=\mathrm{Fold}_{k,p,s}\!\left(\left(\mathbf{A}\mathbf{V}_u\right)\right)
\in\mathbb{R}^{B\times A\times H\times W}
$$

$$
\mathbf{y}=\mathrm{Linear}_{A\to C}\!\left(\mathbf{O}^{\top}\right)\in\mathbb{R}^{B\times H\times W\times C}
$$

### 实现建议

1. 按源实现先修正参数：`kernel_size = max(1, k)`；若 `k > H` 或 `k > W` 则收缩为
   `min(k, H, W)`；`stride > H` 或 `> W` 时回退为 1；`padding >= k` 时取 `k // 2`。
2. `attn_dim = (C // num_heads) * num_heads`，为 0 时取 `num_heads`。
3. 计算链整体升 fp32：`x`、三组权重/偏置全部 cast 到 fp32，再执行 unfold/fold。
4. `avg_pool2d(x, kernel_size=stride, stride=stride)` 后若网格尺寸与 unfold 的
   `h_out/w_out` 不一致，用 `adaptive_avg_pool2d` 对齐。
5. `attn_pj` 的输出 reshape 为 `[B, L, n_h, k^2, k^2]` 后沿最后一维 softmax；与 value
   相乘得到 `[B, n_h, L, k^2, d]`，再 permute/reshape 成 `[B, A*k^2, L]` 做 fold。
6. 输出投影后 cast 回输入 dtype。

## 3. 接口规范

### 算子原型

```python
cann_bench.outlook_attention(
    x, v_pj_weight, v_pj_bias, attn_pj_weight, attn_pj_bias,
    proj_weight, proj_bias,
    num_heads=1, kernel_size=3, padding=1, stride=1,
) -> Tensor y
```

### 输入参数说明

| 参数 | 形状 | 说明 |
|---|---|---|
| x | `[B, H, W, C]` | 输入特征图（NHWC） |
| v_pj_weight / v_pj_bias | `[attn_dim, C]` / `[attn_dim]` | value 投影 |
| attn_pj_weight / attn_pj_bias | `[keff^4 * num_heads, C]` / `[keff^4 * num_heads]` | 注意力权重投影 |
| proj_weight / proj_bias | `[C, attn_dim]` / `[C]` | 输出投影 |
| num_heads | int，默认 1 | head 数；源 json 未给出，按源 forward 默认值 1 处理 |
| kernel_size | int，默认 3 | 局部窗口核大小 |
| padding | int，默认 1 | 窗口 padding |
| stride | int，默认 1 | 窗口步长 |

其中 `attn_dim = (C // num_heads) * num_heads`，`keff` 为按 H/W 收缩后的有效核大小。
注意 `attn_pj_weight` 的 shape 使用 **keff** 而非原始 `kernel_size`（源实现先收缩核再建层）。

### 输出

| 参数 | 形状 | 说明 |
|---|---|---|
| y | `[B, H, W, C]` | 输出特征图，dtype 与输入一致 |

### 数据类型

输入与输出支持 `float16`、`float32`、`bfloat16`；所有张量使用同一 dtype。计算在 fp32
下进行，输出 cast 回输入 dtype。

### 规则与约束

- 输入必须是 NHWC 4 维张量。
- `num_heads` 必须整除 `C`（源实现默认 1，等价于单 head）。
- `keff` 由 `x` 的 H/W 决定，权重 shape 必须与之一致。
- `stride` 用于 `avg_pool2d` 与 unfold/fold，需在同一套参数下配套使用。

## 4. 精度要求

golden 与源实现一致地使用 fp32 compute dtype，输出再 cast 回输入 dtype，无随机性。
fp32 下期望逐位一致；fp16/bf16 输出经过一次 cast，参考相对误差约 $10^{-3}$。

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
Outlook Attention 算子 Torch Golden 参考实现

Outlook Attention 将每个像素到其 k x k 邻域的注意力权重通过 unfold 展开后
由线性层预测，再 fold 回原空间尺寸：

    v      = v_pj(x)                     # [B, attn_dim, H, W]
    v_unf  = Unfold(k, p, s)(v)
    attn   = attn_pj(avgpool(x))         # [B, L, k^2*head, k^2*head]
    attn   = softmax(attn * head_dim^-0.5, dim=-1)
    out    = Fold((attn @ v_unf))        # [B, attn_dim, H, W]
    y      = proj(out)

精度策略（与源实现一致）：权重按输入 dtype 给出，计算链整体升到 fp32 执行，
最后把结果 cast 回输入 dtype。源实现可用环境变量 REF_COMPUTE_DTYPE 覆盖该
compute dtype（默认 float32），本 golden 固定使用 float32。
"""


def outlook_attention(
    x: torch.Tensor,
    v_pj_weight: torch.Tensor,
    v_pj_bias: torch.Tensor,
    attn_pj_weight: torch.Tensor,
    attn_pj_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    num_heads: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 1,
) -> torch.Tensor:
    """
    Outlook Attention 前向

    Args:
        x: 输入特征图 [B, H, W, C]（NHWC）
        v_pj_weight: value 投影权重 [attn_dim, C]
        v_pj_bias: value 投影偏置 [attn_dim]
        attn_pj_weight: 注意力投影权重 [kernel_size**4 * num_heads, C]
        attn_pj_bias: 注意力投影偏置 [kernel_size**4 * num_heads]
        proj_weight: 输出投影权重 [C, attn_dim]
        proj_bias: 输出投影偏置 [C]
        num_heads: 注意力 head 数；attn_dim = (C // num_heads) * num_heads
        kernel_size: 局部窗口核大小（会被 clamp 到 min(H, W)）
        padding: 窗口 padding
        stride: 窗口步长

    Returns:
        输出特征图 [B, H, W, C]，dtype 与输入 x 一致
    """
    B, H, W, C = x.shape
    if B == 0 or H == 0 or W == 0 or C == 0:
        return x

    # 与源实现一致：按实际输入尺寸修正参数
    kernel_size = max(1, kernel_size)
    stride = max(1, stride)
    padding = max(0, padding)
    if kernel_size > H or kernel_size > W:
        kernel_size = min(kernel_size, H, W)
        if kernel_size < 1:
            kernel_size = 1
    if stride > H or stride > W:
        stride = 1
    if padding >= kernel_size:
        padding = kernel_size // 2

    dim = C
    attn_dim = (dim // num_heads) * num_heads
    if attn_dim == 0:
        attn_dim = num_heads

    # ===== fp32 精度升级：权重无损升到 fp32，计算链整体使用 compute dtype =====
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    v_pj_w = v_pj_weight.to(torch.float32)
    v_pj_b = v_pj_bias.to(torch.float32)
    attn_pj_w = attn_pj_weight.to(torch.float32)
    attn_pj_b = attn_pj_bias.to(torch.float32)
    proj_w = proj_weight.to(torch.float32)
    proj_b = proj_bias.to(torch.float32)

    h_grid = (H + 2 * padding - kernel_size) // stride + 1
    w_grid = (W + 2 * padding - kernel_size) // stride + 1
    if h_grid <= 0 or w_grid <= 0:
        return x.to(orig_dtype)

    v = F.linear(x, v_pj_w, v_pj_b).permute(0, 3, 1, 2)  # B, attn_dim, H, W
    v_unfold = F.unfold(v, kernel_size=kernel_size, padding=padding, stride=stride)
    L = v_unfold.shape[-1]
    head_dim = attn_dim // num_heads
    # unfold 产生的真实空间网格
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    if h_out * w_out != L:
        # 与源实现一致的兜底：按 H/W 比例由 L 反推网格
        ratio = H / max(1, W)
        w_out = int((L / ratio) ** 0.5)
        w_out = max(1, min(L, w_out))
        h_out = L // w_out
        while h_out * w_out != L and w_out > 1:
            w_out -= 1
            h_out = L // w_out
    v = v_unfold.reshape(B, num_heads, head_dim, kernel_size * kernel_size, L).permute(0, 1, 4, 3, 2)

    attn = x.permute(0, 3, 1, 2)  # B, C, H, W
    attn = F.avg_pool2d(attn, kernel_size=stride, stride=stride, ceil_mode=False)
    if attn.shape[2] != h_out or attn.shape[3] != w_out:
        attn = F.adaptive_avg_pool2d(attn, (h_out, w_out))
    attn = attn.permute(0, 2, 3, 1)  # B, h_out, w_out, C
    attn = F.linear(attn, attn_pj_w, attn_pj_b).reshape(
        B, L, num_heads, kernel_size * kernel_size, kernel_size * kernel_size
    ).permute(0, 2, 1, 3, 4)
    scale = head_dim ** -0.5
    attn = F.softmax(attn * scale, dim=-1)

    out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, attn_dim * kernel_size * kernel_size, L)
    # F.fold 是 Unfold 的逆：折回原始空间尺寸
    out = F.fold(out, output_size=(H, W), kernel_size=kernel_size,
                 padding=padding, stride=stride)
    out = F.linear(out.permute(0, 2, 3, 1), proj_w, proj_b)  # B, H, W, C
    return out.to(orig_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, W, C = 2, 8, 15, 1280
k, p, s, nh = 3, 1, 2, 1
attn_dim = (C // nh) * nh
dt = torch.float16

x = torch.randn(B, H, W, C, dtype=dt, device="npu")
v_pj_w = torch.randn(attn_dim, C, dtype=dt, device="npu") * 0.05
v_pj_b = torch.zeros(attn_dim, dtype=dt, device="npu")
attn_pj_w = torch.randn(k ** 4 * nh, C, dtype=dt, device="npu") * 0.05
attn_pj_b = torch.zeros(k ** 4 * nh, dtype=dt, device="npu")
proj_w = torch.randn(C, attn_dim, dtype=dt, device="npu") * 0.05
proj_b = torch.zeros(C, dtype=dt, device="npu")

y = cann_bench.outlook_attention(
    x, v_pj_w, v_pj_b, attn_pj_w, attn_pj_b, proj_w, proj_b,
    nh, k, p, s,
)
# y: [2, 8, 15, 1280], float16
```
