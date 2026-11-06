# DecoderMhaAttention 算子 API 描述

## 1. 算子简介

DecoderMhaAttention 是 SGLang 风格 decode 阶段的多头注意力。每条序列只有一个新 query token，
其历史 KV 以全局 pool + `kv_indptr` / `kv_indices` 的变长索引方式给出。kernel 采用 split-KV
两段式（分段局部 softmax + lse 归并），数学上等价于一次全局 softmax。

**主要应用场景**：
- 大语言模型自回归解码阶段的 MHA 注意力
- 带 paged / 连续 KV pool 的批量 decode

**算子特征**：
- 难度等级：L4（Attention）
- 输入：`q [B, H, D]`、全局 `k_buffer` / `v_buffer`、KV 变长索引
- 输出：`[B, H, Dv]`，dtype 与 `q` 一致
- 仅支持 MHA（`H_q == H_kv`）

## 2. 算子定义

### 数学公式

$$
\mathrm{idx}_b = \mathrm{kv\_indices}\big[\mathrm{kv\_indptr}[b] : \mathrm{kv\_indptr}[b+1]\big]
$$

$$
s[b,h,j] = \mathrm{sm\_scale}\cdot q[b,h,:]\cdot k\_buffer[\mathrm{idx}_b[j], h, :]
$$

$$
o[b,h,:] = \mathrm{softmax}_j\big(s[b,h,:]\big)\, v\_buffer[\mathrm{idx}_b, h, :]
$$

split-KV 将序列切分为若干段分别计算局部 softmax 与 lse，再在 stage2 归并；其结果与上式一次
全局 softmax 等价。全程 fp32 计算后回铸为 `q.dtype`。

## 3. 接口规范

### 算子原型

```python
cann_bench.decoder_mha_attention(Tensor q, Tensor k_buffer, Tensor v_buffer,
                                 Tensor kv_indptr, Tensor kv_indices,
                                 int max_kv_len, float sm_scale) -> Tensor o
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | 每条序列一个新 token 的 query `[B, H, D]` |
| k_buffer | Tensor | 必选 | 全局 KV pool 中的 key `[P, H, D]` |
| v_buffer | Tensor | 必选 | 全局 KV pool 中的 value `[P, H, Dv]` |
| kv_indptr | Tensor | 必选 | 每条序列在 `kv_indices` 上的起止 `[B + 1]`，int32 |
| kv_indices | Tensor | 必选 | 每条序列命中的 KV pool token 号 `[nnz]`，int32 |
| max_kv_len | int | 必选 | 批内最长 KV 长度（形状提示） |
| sm_scale | float | 必选 | 打分缩放因子，通常为 `1/sqrt(D)` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| o | `[B, H, Dv]` | 与 `q` 相同 | 注意力输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 / bfloat16（q、k_buffer、v_buffer），int32（kv_indptr、kv_indices） | float16 / bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `q.shape == [B, H, D]`，`k_buffer.shape == [P, H, D]`，`v_buffer.shape == [P, H, Dv]`
- `kv_indptr` 长度为 `B + 1`，单调非递减，`kv_indptr[0] == 0`，`kv_indptr[-1] == nnz`
- `kv_indices` 的元素必须落在 `[0, P)` 内
- 当 `nnz == B * max_kv_len` 时视为等长批，`kv_indices` 直接按 `[B, max_kv_len]` 解释
- `get_input` 会按 shape 重建合法的单调 `kv_indptr` 与 pool 内的 `kv_indices`

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
DecoderMhaAttention 算子 Torch Golden 参考实现

SGLang 风格 decode 阶段 MHA 注意力（split-KV 两段式的数学等价实现）。
对每条序列按 kv_indptr / kv_indices 取出命中的 KV pool token，做全局 softmax
注意力；split-KV 归并后的结果与一次全局 softmax 等价。

公式:
    idx[b] = kv_indices[kv_indptr[b] : kv_indptr[b+1]]
    scores[b, h, j] = sm_scale * q[b, h, :] . k_buffer[idx[b][j], h, :]
    o[b, h, :] = softmax(scores[b, h, :]) @ v_buffer[idx[b], h, :]
"""


def decoder_mha_attention(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    max_kv_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """
    Decoder MHA 注意力前向（decode，split-KV 等价全局 softmax）

    公式:
        scores = sm_scale * q k^T，softmax 后与 v 相乘

    Args:
        q: 每条序列一个新 token 的 query [B, H, D]
        k_buffer: 全局 KV pool 中的 key [P, H, D]
        v_buffer: 全局 KV pool 中的 value [P, H, Dv]
        kv_indptr: 每条序列在 kv_indices 上的起止 [B + 1]，int32
        kv_indices: 每条序列命中的 KV pool token 号 [nnz]，int32
        max_kv_len: 批内最长 KV 长度（形状提示）
        sm_scale: 打分缩放因子，通常为 1/sqrt(D)

    Returns:
        注意力输出 [B, H, Dv]，dtype 与 q 一致
    """
    device = q.device
    batch, heads, dim = q.shape
    dim_v = v_buffer.shape[-1]
    total = kv_indices.shape[0]
    rows = int(max_kv_len)

    if total == batch * rows:
        # 等长批：kv_indices 本身即 [B, max_kv_len] 连续布局，无需 padding 与掩码
        token = kv_indices
        valid = None
    else:
        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = kv_indptr[1:] - kv_indptr[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        gather = kv_indptr[:-1].reshape(batch, 1) + pos.reshape(1, rows)
        token = torch.index_select(kv_indices, 0, gather.reshape(-1).clamp(max=total - 1))

    k = torch.index_select(k_buffer, 0, token.reshape(-1)).to(torch.float32)
    v = torch.index_select(v_buffer, 0, token.reshape(-1)).to(torch.float32)
    k = k.reshape(batch, rows, heads, dim)
    v = v.reshape(batch, rows, heads, dim_v)

    scores = torch.einsum("bhd,bkhd->bhk", q.to(torch.float32), k) * sm_scale
    if valid is not None:
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhk,bkhd->bhd", probs, v)
    return out.to(q.dtype)


def get_input(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    max_kv_len: int = 1,
    sm_scale: float = 1.0,
    **kwargs,
):
    """重建满足算子里约的 KV 变长索引结构。

    kv_indptr 必须是单调前缀和，kv_indices 必须是 KV pool 中的合法（可被打散的）
    token 号，单区间 value_range 无法表达。这里按实际 shape 重新构造：
      - nnz 均分到 B 条序列（每条不超过 max_kv_len）；
      - kv_indices 用固定种子对 pool 做打乱后取前 nnz 个。
    q / k_buffer / v_buffer 原样保留。

    Returns:
        [q, k_buffer, v_buffer, kv_indptr, kv_indices]
    """
    batch = int(kv_indptr.numel()) - 1
    pool = int(k_buffer.shape[0])
    total = int(kv_indices.numel())
    rows = int(max_kv_len)
    device = kv_indptr.device

    if total == batch * rows and rows > 0:
        lens = [rows] * batch
    else:
        base, rem = divmod(total, max(batch, 1))
        lens = [min(base + (1 if b < rem else 0), rows) for b in range(batch)]
        short = total - sum(lens)
        for b in range(batch):
            take = min(short, rows - lens[b])
            if take > 0:
                lens[b] += take
                short -= take

    indptr = [0]
    for length in lens:
        indptr.append(indptr[-1] + length)

    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(pool, generator=gen).to(torch.int32)
    indices = perm[:total]
    if total > 0:
        indices = indices.clamp(max=max(pool - 1, 0))

    new_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    new_indices = indices.to(device)
    return [q, k_buffer, v_buffer, new_indptr, new_indices]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, D, P = 1, 4, 64, 11264
L = 8192
q = torch.randn(B, H, D, dtype=torch.float16, device="npu") * 0.1
k_buffer = torch.randn(P, H, D, dtype=torch.float16, device="npu") * 0.1
v_buffer = torch.rand(P, H, D, dtype=torch.float16, device="npu")
kv_indptr = torch.tensor([0, L], dtype=torch.int32, device="npu")
kv_indices = torch.randperm(P, device="npu")[:L].to(torch.int32)
o = cann_bench.decoder_mha_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, L, 0.125)
# o.shape == (1, 4, 64), float16
```
