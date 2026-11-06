# FusionAttention 算子 API 描述

## 1. 算子简介

FusionAttention 是 Transformer 注意力前向的融合实现：输入 query / key / value 按
BNSD 或 BSH 布局组织，内部完成布局归一化、GQA 头扩展、`QK^T * scale`、按行 softmax
与 `softmax @ V`，并在输出主结果的同时返回每行 softmax 的最大值与归一化分母。

**主要应用场景**：
- 大模型推理/训练中的标准多头注意力（MHA）与分组查询注意力（GQA）
- 需要同时获得注意力输出与 softmax 统计量的融合算子场景

**算子特征**：
- 难度等级：L4（Attention）
- 3 个必选张量输入 + 3 个属性
- 3 个输出：注意力主输出（与 query 同布局）与两个 float32 softmax 统计量
- 支持 float16 / float32 / bfloat16

## 2. 算子定义

### 数学公式

设输入按布局转换为 BNSD 形式，query 头数为 $N$，key/value 头数为 $N_{kv}$，
分组扩展倍数 $G = N / N_{kv}$：

$$
\mathbf{q},\mathbf{k},\mathbf{v} = \mathrm{layout\_to\_bnsd}(Q,K,V),\qquad
\mathbf{k},\mathbf{v} = \mathrm{repeat\_kv}(\mathbf{k},\mathbf{v},G)
$$

$$
\mathbf{S} = \mathbf{q}\mathbf{k}^{\top}\cdot \alpha,\qquad
\mathbf{P} = \mathrm{softmax}(\mathbf{S}),\qquad
\mathbf{O} = \mathbf{P}\mathbf{v}
$$

$$
m_i = \max_j S_{ij},\qquad \ell_i = \sum_j \exp(S_{ij} - m_i)
$$

其中 $\alpha$ 为属性 `scale`。注意力输出为 $\mathbf{O}$ 转换回 query 的原布局；
`softmax_max` / `softmax_sum` 为 $m_i$ / $\ell_i$ 在最后一维广播 8 份后的 float32 张量。
打分与 softmax 全程以 fp32 计算，主输出回铸为 query 的 dtype。

### 实现建议

1. 先按 `input_layout` 把 query/key/value 统一到 BNSD；BSH 下 head_dim 由隐藏维除以头数得到。
2. BSH 布局下 key/value 的隐藏维可能与 query 不同：若 `query_hidden != key_hidden`，
   则 $N_{kv} = N \cdot \text{key\_hidden} / \text{query\_hidden}$，否则 $N_{kv} = N$。
3. GQA 通过 `repeat_interleave` 扩展 key/value 头，或按 $(N_{kv}, G)$ reshape 后广播，避免物化扩展后的 K/V。
4. 按 query 行分块计算 $[B,N,S_q,S_{kv}]$ 打分矩阵，控制峰值显存。
5. `softmax_max` / `softmax_sum` 形状为 `[B, N, Sq, 8]`，8 个 lane 为同一值的广播副本。

## 3. 接口规范

### 算子原型

```python
cann_bench.fusion_attention(
    query, key, value, head_num, input_layout, scale=1.0
) -> (attention_out, softmax_max, softmax_sum)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | query 张量，BNSD `[B, N, Sq, D]` 或 BSH `[B, Sq, N*D]` |
| key | Tensor | 必选 | key 张量，BNSD `[B, Nkv, Skv, Dkv]` 或 BSH `[B, Skv, Nkv*Dkv]` |
| value | Tensor | 必选 | value 张量，BNSD `[B, Nkv, Skv, Dv]` 或 BSH `[B, Skv, Nkv*Dv]` |
| head_num | int | 必选 | query 头数 $N$ |
| input_layout | string | 必选 | 张量布局，支持 `"BNSD"` 与 `"BSH"` |
| scale | float | 1.0 | `QK^T` 的打分缩放系数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| attention_out | BNSD `[B, N, Sq, Dv]` / BSH `[B, Sq, N*Dv]` | 与 query 相同 | 注意力输出 |
| softmax_max | `[B, N, Sq, 8]` | float32 | 每行 softmax 最大值，8 个 lane 广播 |
| softmax_sum | `[B, N, Sq, 8]` | float32 | 每行 softmax 指数和，8 个 lane 广播 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16（attention_out）/ float32（softmax 统计量） |
| float32 | float32 |
| bfloat16 | bfloat16（attention_out）/ float32（softmax 统计量） |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `head_num` 必须整除 BNSD 下的 query 通道维或 BSH 下的 query 隐藏维。
- BNSD 下 key/value 的头数可小于 query 头数（GQA），但必须整除 `head_num`。
- 打分与 softmax 以 fp32 计算；`attention_out` 的 dtype 与 query 一致。
- `input_layout` 仅支持 `"BNSD"` 与 `"BSH"`；其它布局报错。
- `softmax_max` / `softmax_sum` 始终为 float32，且最后一维固定为 8。

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

"""
FusionAttention 算子 Torch Golden 参考实现

融合注意力前向：支持 BNSD / BSH 布局与 MHA / GQA 头扩展，按 scale 缩放 QK^T
后做 softmax，并同时返回每行 softmax 的最大值与归一化分母。

公式:
    q, k, v = layout_to_bnsd(query, key, value)
    k, v    = repeat_kv(k, v, num_q_heads // num_kv_heads)
    scores  = (q @ k^T) * scale
    p       = softmax(scores, dim=-1)
    out     = p @ v
    attention_out = bnsd_to_layout(out)
    softmax_max[i] = max_j scores[i, j]
    softmax_sum[i] = sum_j exp(scores[i, j] - softmax_max[i])
"""


def _to_bnsd(tensor: torch.Tensor, layout: str, heads: int) -> torch.Tensor:
    if layout == "BNSD":
        return tensor
    if layout == "BSH":
        b, s, h = tensor.shape
        d = h // heads
        return tensor.view(b, s, heads, d).transpose(1, 2)
    raise ValueError(f"Unsupported layout: {layout}")


def _from_bnsd(tensor: torch.Tensor, layout: str, heads: int) -> torch.Tensor:
    if layout == "BNSD":
        return tensor
    if layout == "BSH":
        b, n, s, d = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(b, s, n * d)
    raise ValueError(f"Unsupported layout: {layout}")


def fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: int,
    input_layout: str,
    scale: float = 1.0,
) -> tuple:
    """
    FusionAttention 前向

    公式:
        q, k, v = layout_to_bnsd(query, key, value)
        k, v    = repeat_kv(k, v, num_q_heads // num_kv_heads)
        scores  = (q @ k^T) * scale
        attention_out = bnsd_to_layout(softmax(scores, dim=-1) @ v)
        softmax_max[i] = max_j scores[i, j]
        softmax_sum[i] = sum_j exp(scores[i, j] - softmax_max[i])

    Args:
        query: query 张量，BNSD [B, N, Sq, D] 或 BSH [B, Sq, N*D]
        key: key 张量，BNSD [B, Nkv, Skv, Dkv] 或 BSH [B, Skv, Nkv*Dkv]
        value: value 张量，BNSD [B, Nkv, Skv, Dv] 或 BSH [B, Skv, Nkv*Dv]
        head_num: query 头数 N
        input_layout: "BNSD" 或 "BSH"
        scale: QK^T 的打分缩放系数

    Returns:
        (attention_out, softmax_max, softmax_sum)
    """
    nq = int(head_num)
    if input_layout == "BSH":
        q_hidden = query.shape[-1]
        k_hidden = key.shape[-1]
        nkv = nq * k_hidden // q_hidden if q_hidden != k_hidden else nq
    else:
        nkv = nq

    q = _to_bnsd(query, input_layout, nq).float()
    k = _to_bnsd(key, input_layout, nkv).float()
    v = _to_bnsd(value, input_layout, nkv).float()

    b, n, sq, dk = q.shape
    skv = k.shape[2]
    dv = v.shape[-1]

    group = nq // nkv
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    out = torch.zeros(b, n, sq, dv, dtype=torch.float32)
    softmax_max = torch.zeros(b, n, sq, 1, dtype=torch.float32)
    softmax_sum = torch.zeros(b, n, sq, 1, dtype=torch.float32)

    # 按 query 行分块，避免一次物化 [B, N, Sq, Skv]
    budget = 2 ** 24
    blk = max(1, min(sq, budget // max(1, b * n * skv)))
    for start in range(0, sq, blk):
        end = min(start + blk, sq)
        scores = torch.matmul(q[:, :, start:end], k.transpose(-2, -1)) * float(scale)
        row_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - row_max)
        row_sum = exp_scores.sum(dim=-1, keepdim=True)
        out[:, :, start:end] = torch.matmul(exp_scores / row_sum, v)
        softmax_max[:, :, start:end] = row_max
        softmax_sum[:, :, start:end] = row_sum

    attention_out = _from_bnsd(out, input_layout, nq).to(query.dtype)
    softmax_max = softmax_max.expand(b, n, sq, 8).contiguous().to(torch.float32)
    softmax_sum = softmax_sum.expand(b, n, sq, 8).contiguous().to(torch.float32)
    return attention_out, softmax_max, softmax_sum

```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, N, Sq, Skv, D = 2, 16, 1024, 1024, 128
query = torch.randn(B, N, Sq, D, dtype=torch.float16, device="npu")
key = torch.randn(B, N, Skv, D, dtype=torch.float16, device="npu")
value = torch.randn(B, N, Skv, D, dtype=torch.float16, device="npu")
attention_out, softmax_max, softmax_sum = cann_bench.fusion_attention(
    query, key, value, N, "BNSD", 0.0625
)
# attention_out: [2, 16, 1024, 128], float16
# softmax_max / softmax_sum: [2, 16, 1024, 8], float32
```
