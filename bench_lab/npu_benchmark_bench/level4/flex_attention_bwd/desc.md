# FlexAttentionBwd 算子 API 描述

## 1. 算子简介

FlexAttentionBwd 是 FlexAttention 的反向算子（template-style，纯反向逻辑）。前向保存的
`attn_output` 与 `logsumexp` 作为输入提供，反向据此重建 softmax 概率并计算
query / key / value 三路梯度。数学与 PyTorch 官方 Inductor backward template
（`torch/_inductor/kernel/flex/flex_attention.py`）一致，支持可选因果掩码与 GQA。

**主要应用场景**：
- 使用 `torch.nn.attention.flex_attention` 的模型训练反传
- 需要自定义掩码 / score_mod 的注意力训练（本任务取 score_mod 为空的标准反向）

**算子特征**：
- 难度等级：L4（Attention）
- 6 个张量输入（含 2 个前向 saved tensor）+ 2 个属性
- 3 个张量输出（`grad_query/grad_key/grad_value`）
- 支持因果掩码与 GQA（KV 头扩展后梯度组内求和）
- 源用例覆盖 float16 / bfloat16 / float32

## 2. 算子定义

### 数学公式

设 $\text{scale} = 1/\sqrt{D}$，$O$ 为前向输出，$\text{lse}$ 为前向 logsumexp：

$$
P = \exp\!\left(\text{scale}\, Q K^\top - \text{lse}\right), \qquad
\delta = \operatorname{rowsum}(dO \odot O)
$$

$$
dV = P^\top dO, \qquad dP = dO V^\top, \qquad dS = P \odot (dP - \delta)
$$

$$
dQ = dS\, K \cdot \text{scale}, \qquad dK = dS^\top Q \cdot \text{scale}
$$

因果掩码时把 $S$ 上三角置为 $-\infty$（`torch.triu(..., diagonal=S_k - S_q + 1)`），
此时 $P$ 对应位置为 0。GQA 时 KV 头先 `repeat_interleave` 扩展到 Q 头数，反向时
$dK/dV$ 按 $G = H / H_{kv}$ 分组求和恢复原 KV 头数。

### 实现建议

1. 全程 fp32 计算：`query/key/value/grad_output/attn_output` 先 `.float()`。
2. `attn_weight = matmul(q, k^T) * scale`；因果时上三角 `masked_fill(-inf)`。
3. `P = exp(attn_weight - logsumexp.unsqueeze(-1))`，不要重复做 softmax。
4. `delta` 由 `grad_output * attn_output` 在 D 维求和得到。
5. GQA 时先扩展 K/V，最后把 `dK/dV` view 成 `[B, H_kv, G, S_k, D]` 后 `sum(dim=2)`。
6. 输出回铸 `query/key/value` 的 dtype。

## 3. 接口规范

### 算子原型

```python
cann_bench.flex_attention_bwd(
    grad_output, query, key, value, attn_output, logsumexp,
    is_causal=False, enable_gqa=False,
) -> (grad_query, grad_key, grad_value)
```

### 输入参数说明

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `grad_output` | `[B, H, S_q, D]` | float16 / bfloat16 / float32 | 输出梯度 dO |
| `query` | `[B, H, S_q, D]` | 同 grad_output | 前向 query |
| `key` | `[B, H_kv, S_k, D]` | 同 grad_output | 前向 key |
| `value` | `[B, H_kv, S_k, D]` | 同 grad_output | 前向 value |
| `attn_output` | `[B, H, S_q, D]` | 同 grad_output | 前向注意力输出（saved） |
| `logsumexp` | `[B, H, S_q]` | float32 | 前向 logsumexp（saved） |

### 属性

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `is_causal` | bool | False | 是否使用因果掩码 |
| `enable_gqa` | bool | False | 是否启用 GQA |

### 输出

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `grad_query` | `[B, H, S_q, D]` | 同 query | query 梯度 |
| `grad_key` | `[B, H_kv, S_k, D]` | 同 key | key 梯度 |
| `grad_value` | `[B, H_kv, S_k, D]` | 同 value | value 梯度 |

### 数据类型

输入与输出支持 float16 / bfloat16 / float32；`logsumexp` 固定为 float32；
`grad_output/query/key/value/attn_output` 使用同一 dtype。

### 规则与约束

- `enable_gqa=True` 时 `H % H_kv == 0` 且 `H_kv < H`；否则 `H_kv == H`。
- 因果掩码采用左下三角对齐（`diagonal=S_k - S_q + 1`），支持 `S_q != S_k`。
- `logsumexp` 必须是自然底 logsumexp（与标准 softmax 一致）。

## 4. 精度要求

golden 全程使用 fp32 计算，输出回铸输入 dtype；结果与 golden 的误差需满足 bench
的统一精度门限，推荐以约 $10^{-3}$（fp16/bf16）相对误差为参考。

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

"""
FlexAttentionBwd 算子 Torch Golden 参考实现

FlexAttention 反向（template-style，纯反向逻辑）。由前向保存的 attn_output 与
logsumexp 重建 softmax 概率，计算 query / key / value 梯度，数学与官方 Inductor
backward template（torch/_inductor/kernel/flex/flex_attention.py）一致：

    P   = exp(scale * Q K^T - logsumexp)          # causal 时上三角为 0
    dv  = P^T dO
    dP  = dO V^T
    D   = rowsum(dO ⊙ O)
    dS  = P ⊙ (dP - D)
    dq  = dS K · scale
    dk  = dS^T Q · scale

支持可选因果掩码与 GQA（KV 头 expand 到 Q 头，梯度在组内求和恢复 KV 头数）。
全程 fp32 计算，输出回铸 query/key/value 的 dtype。
"""


def flex_attention_bwd(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> tuple:
    """
    FlexAttention 反向

    Args:
        grad_output: 输出梯度 dO [B, H, S_q, D]
        query: 前向 query [B, H, S_q, D]
        key: 前向 key [B, H_kv, S_k, D]（enable_gqa 时 H_kv < H）
        value: 前向 value [B, H_kv, S_k, D]
        attn_output: 前向注意力输出 O [B, H, S_q, D]
        logsumexp: 前向 softmax 的 logsumexp [B, H, S_q]
        is_causal: 是否使用因果掩码
        enable_gqa: 是否启用 GQA（KV 头扩展 + 梯度组内求和）

    Returns:
        (grad_query, grad_key, grad_value)，shape 分别与 query/key/value 一致
    """
    B, H, S_q, D = query.shape
    H_kv = key.shape[1]
    S_k = key.shape[2]
    scale = 1.0 / math.sqrt(D)

    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    grad_output_f = grad_output.float()

    # GQA: expand KV heads to Q heads
    if enable_gqa and H != H_kv:
        rep = H // H_kv
        key_f = key_f.repeat_interleave(rep, dim=1)
        value_f = value_f.repeat_interleave(rep, dim=1)

    attn_weight = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=query.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1)
        attn_weight.masked_fill_(causal_mask, float('-inf'))

    attn_probs = torch.exp(attn_weight - logsumexp.unsqueeze(-1).float())

    grad_value = torch.matmul(attn_probs.transpose(-2, -1), grad_output_f)
    grad_attn_probs = torch.matmul(grad_output_f, value_f.transpose(-2, -1))
    D_term = (grad_output_f * attn_output.float()).sum(dim=-1, keepdim=True)
    grad_attn_weight = attn_probs * (grad_attn_probs - D_term)

    grad_query = torch.matmul(grad_attn_weight, key_f) * scale
    grad_key = torch.matmul(grad_attn_weight.transpose(-2, -1), query_f) * scale

    # GQA: sum expanded-head grads back to KV heads
    if enable_gqa and H != H_kv:
        rep = H // H_kv
        grad_key = grad_key.view(B, H_kv, rep, S_k, D).sum(dim=2)
        grad_value = grad_value.view(B, H_kv, rep, S_k, D).sum(dim=2)

    return (grad_query.to(query.dtype),
            grad_key.to(key.dtype),
            grad_value.to(value.dtype))


def get_input(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
    enable_gqa: bool = False,
    **kwargs,
) -> list:
    """按前向公式重算 attn_output 与 logsumexp，保证反向输入自洽。

    单区间 value_range 无法表达「saved tensor 是 q/k/v 的确定性函数」这一约束，
    这里用 fp32 重放前向（含 causal 掩码与 GQA 头扩展）后返回；q/k/v/dO 原样保留。

    Returns:
        [grad_output, query, key, value, attn_output, logsumexp]
    """
    q_f = query.float()
    k_f = key.float()
    v_f = value.float()
    B, H, S_q, D = q_f.shape
    H_kv = k_f.shape[1]
    S_k = k_f.shape[2]
    scale = 1.0 / math.sqrt(D)

    if enable_gqa and H != H_kv:
        rep = H // H_kv
        k_f = k_f.repeat_interleave(rep, dim=1)
        v_f = v_f.repeat_interleave(rep, dim=1)

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=q_f.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    attn_probs = torch.exp(scores - lse.unsqueeze(-1))
    new_attn_output = torch.matmul(attn_probs, v_f).to(attn_output.dtype)
    new_logsumexp = lse.to(logsumexp.dtype)
    return [grad_output, query, key, value, new_attn_output, new_logsumexp]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, H_kv, S, D = 2, 8, 2, 1024, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
k = torch.randn(B, H_kv, S, D, dtype=torch.float16, device="npu")
v = torch.randn(B, H_kv, S, D, dtype=torch.float16, device="npu")
do = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
o = torch.randn(B, H, S, D, dtype=torch.float16, device="npu")
lse = torch.rand(B, H, S, dtype=torch.float32, device="npu")

dq, dk, dv = cann_bench.flex_attention_bwd(
    do, q, k, v, o, lse, is_causal=True, enable_gqa=True,
)
# dq: [2, 8, 1024, 64], dk/dv: [2, 2, 1024, 64], float16
```
