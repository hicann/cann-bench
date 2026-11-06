# NsaTopkSparseAttentionBwd 算子 API 描述

## 1. 算子简介

NsaTopkSparseAttentionBwd 是 DeepSeek Native Sparse Attention（NSA）TopK 选块注意力的反向算子。
前向对每个 query 位置只在 `topk_idx` 选中的 KV 块且因果可见（`j <= i`）的 key 上做 log2 域
softmax；反向消费前向保存的注意力输出 `o` 与 **以 log2 为底**的 `lse`，重建稀疏概率并计算
query / key / value 三路梯度。

**主要应用场景**：
- 长上下文大模型（DeepSeek 系列）的 NSA 稀疏注意力训练反传
- 需要在超长序列上把注意力计算量按 top-k 选块压低的训练场景

**算子特征**：
- 难度等级：L4（Attention）
- 8 个必选张量输入（`q/k/v/topk_idx/cu_seqlens/o/lse/do`）+ 3 个属性
- 3 个张量输出（`dq/dk/dv`）
- 支持 GQA（`Hq % Hk == 0`）与 varlen 变长批
- 源实现用例为 float32（fp16/bf16 下 kernel 内 `ds.to(q.dtype)` 的降精度误差超出验证门限，
  故本任务 proto 只声明 float32）

## 2. 算子定义

### 数学公式

记允许集合 $S(i) = \{ j \mid j /\!\!/ \text{block\_size} \in \text{topk\_idx}[h, i] \} \cap \{ j \le i \}$，
$\mathrm{log2e} = 1.44269504$，则

$$
p[i,j] = \exp_2\!\left(\mathbf{q}_i \cdot \mathbf{k}_j \cdot \text{sm\_scale} \cdot \log_2 e - \text{lse}[i]\right), \quad j \in S(i)
$$

$$
\mathbf{dv}_j = \sum_i p[i,j]\, \mathbf{do}_i, \qquad
\mathbf{dp}[i,j] = \mathbf{do}_i \cdot \mathbf{v}_j, \qquad
\delta_i = \sum_d \mathbf{o}_{i,d}\, \mathbf{do}_{i,d}
$$

$$
\mathbf{ds}[i,j] = p[i,j]\, (\mathbf{dp}[i,j] - \delta_i)
$$

$$
\mathbf{dq}_i = \text{sm\_scale} \sum_j \mathbf{ds}[i,j]\, \mathbf{k}_j, \qquad
\mathbf{dk}_j = \text{sm\_scale} \sum_i \mathbf{ds}[i,j]\, \mathbf{q}_i
$$

其中 $j \notin S(i)$ 时 $p[i,j] = 0$。这与 kernel 的 `backward_dq` / `backward_dkdv`
是同一套公式；`o` 与 `lse` 作为前向 saved tensor 由输入统一提供，两侧不各自重算前向。

### 实现建议

1. 先把 `topk_idx`（-1 视为 padding）散点到 `[B, Hk, Lq, blocks+1]` 的布尔选块表，
   丢弃哨兵列后按 `block_size` 展开成 `[B, Hk, Lq, Lk]` 的块掩码，再与因果掩码
   `i >= j` 相与得到允许集合。
2. 等长批（`total_len == B * max_seqlen`）可直接 reshape，按 `[B, Lq, Hk, G, D]`
   组织；变长批按 `cu_seqlens` gather / scatter 回写，越界行用哨兵列丢弃。
3. `p = exp2(qk * sm_scale * log2e - lse)`，不要换成自然底 softmax。
4. `delta` 用前向输出 `o` 与 `do` 的逐点积在 `D` 维求和得到。
5. 输出 dtype 分别回铸为 `q/k/v` 的 dtype。

## 3. 接口规范

### 算子原型

```python
cann_bench.nsa_topk_sparse_attention_bwd(
    q, k, v, topk_idx, cu_seqlens, o, lse, do,
    block_size, max_seqlen, sm_scale,
) -> (dq, dk, dv)
```

### 输入参数说明

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `q` | `[total_len, Hq, D]` | float32 | varlen 拼接的 query |
| `k` | `[total_len, Hk, D]` | float32 | varlen 拼接的 key |
| `v` | `[total_len, Hk, D]` | float32 | varlen 拼接的 value |
| `topk_idx` | `[Hk, total_len, topk]` | int32 | 每个 query 选中的 KV 块号，-1 表示 padding |
| `cu_seqlens` | `[B+1]` | int32 | 序列前缀和，末元素等于 `total_len` |
| `o` | `[total_len, Hq, D]` | float32 | 前向注意力输出（saved tensor） |
| `lse` | `[Hq, total_len]` | float32 | 前向 logsumexp（以 log2 为底） |
| `do` | `[total_len, Hq, D]` | float32 | 上游输出梯度 |

### 属性

| 参数 | 类型 | 说明 |
|---|---|---|
| `block_size` | int | KV 分块大小（32/64/128/256） |
| `max_seqlen` | int | 批内最长序列长度（varlen 形状提示） |
| `sm_scale` | float | softmax 缩放因子，通常为 `1/sqrt(D)` |

### 输出

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `dq` | `[total_len, Hq, D]` | float32 | query 梯度 |
| `dk` | `[total_len, Hk, D]` | float32 | key 梯度 |
| `dv` | `[total_len, Hk, D]` | float32 | value 梯度 |

### 数据类型

本任务用例统一使用 float32；`topk_idx` / `cu_seqlens` 为 int32，输出与输入同 dtype。

### 规则与约束

- `Hq % Hk == 0`，同组 query head 共享 `topk_idx`。
- `cu_seqlens` 必须单调不减，且 `cu_seqlens[-1] == total_len`。
- `topk_idx` 的块号必须落在 `[0, ceil(max_seqlen/block_size))` 内，-1 为 padding；
  每个 query 至少选中一个因果可见块（块 0），保证 softmax 分母非空。
- `lse` 必须以 log2 为底（与 kernel 前向口径一致）。

## 4. 精度要求

golden 全程使用 fp32 计算，结果与 golden 的误差需满足 bench 的统一精度门限；
推荐以约 $10^{-5}$ 相对误差为参考。

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
NsaTopkSparseAttentionBwd 算子 Torch Golden 参考实现

DeepSeek Native Sparse Attention 的 TopK 选块注意力反向。前向对每个 query 位置
只在 topk_idx 选中的 KV 块且因果可见（j <= i）的 key 上做 log2 域 softmax；
反向消费前向保存的 o（注意力输出）与 lse（**以 log2 为底**的 logsumexp）：

    S(i)     = { j | j // block_size ∈ topk_idx[h, i] } ∩ { j <= i }
    p[i, j]  = exp2(q_i·k_j * sm_scale * log2(e) - lse[i])     (j ∈ S(i)，否则 0)
    dv[j]    = sum_i p[i, j] * do[i]
    dp[i, j] = do[i] · v[j]
    delta[i] = sum_d o[i, d] * do[i, d]
    ds[i, j] = p[i, j] * (dp[i, j] - delta[i])
    dq[i]    = sm_scale * sum_j ds[i, j] * k[j]
    dk[j]    = sm_scale * sum_i ds[i, j] * q[i]

o 与 lse 作为前向 saved tensor 由输入统一提供，golden 不重算前向。全程 fp32 计算，
输出回铸输入 dtype。等长批（total_len == B * max_seqlen）直接 reshape；变长批按
cu_seqlens 逐序列 gather / scatter 回写。
"""

_LOG2E = 1.44269504


def nsa_topk_sparse_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_size: int,
    max_seqlen: int,
    sm_scale: float,
) -> tuple:
    """
    NSA TopK 稀疏注意力反向

    Args:
        q: varlen 拼接的 query [total_len, Hq, D]
        k: varlen 拼接的 key [total_len, Hk, D]
        v: varlen 拼接的 value [total_len, Hk, D]
        topk_idx: 每个 query 选中的 KV 块号 [Hk, total_len, topk]，-1 表示 padding
        cu_seqlens: 序列前缀和 [B + 1] int32
        o: 前向注意力输出 [total_len, Hq, D]
        lse: 前向保存的 logsumexp [Hq, total_len]（以 log2 为底）
        do: 上游输出梯度 [total_len, Hq, D]
        block_size: KV 分块大小
        max_seqlen: 批内最长序列长度
        sm_scale: softmax 缩放因子

    Returns:
        (dq, dk, dv)，shape 分别与 q / k / v 一致
    """
    device = q.device
    total, hq, dim = q.shape
    hk = k.shape[1]
    group = hq // hk
    batch = cu_seqlens.shape[0] - 1
    rows = int(max_seqlen)
    topk = topk_idx.shape[-1]

    # 内部统一 fp32（源实现口径）；兼容 evaluator 的 fp64 golden precision 与
    # bf16/fp16 原生精度两条路径。输出按原始输入 dtype 回铸。
    q_dtype, k_dtype, v_dtype = q.dtype, k.dtype, v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    o = o.float()
    lse = lse.float()
    do = do.float()

    def allowed_mask(sel_idx, device):
        # 选块 + 因果 → [B, Hk, Lq, Lk] 布尔允许集合（-1 落到哨兵列后丢弃）
        blocks = (rows + block_size - 1) // block_size
        idx = torch.where(sel_idx < 0, torch.full_like(sel_idx, blocks), sel_idx)
        idx = idx.permute(1, 0, 2, 3).to(torch.int64)
        sel = torch.zeros(batch, hk, rows, blocks + 1, device=device, dtype=torch.bool)
        sel.scatter_(-1, idx, True)
        blk_of = torch.arange(blocks, device=device).reshape(blocks, 1)
        blk_of = blk_of.expand(blocks, block_size).reshape(-1)[:rows]
        block_mask = torch.index_select(sel[..., :blocks], -1, blk_of)
        span = torch.arange(rows, device=device)
        return block_mask & (span.reshape(rows, 1) >= span.reshape(1, rows)).reshape(1, 1, rows, rows)

    uniform = total == batch * rows
    if uniform:
        # 等长批：sum(len) == B * max_seqlen 且各 len <= max_seqlen ⇒ 各条等长
        gidx = None
        q_pad = q.to(torch.float32).reshape(batch, rows, hk, group, dim)
        k_pad = k.to(torch.float32).reshape(batch, rows, hk, dim)
        v_pad = v.to(torch.float32).reshape(batch, rows, hk, dim)
        o_pad = o.to(torch.float32).reshape(batch, rows, hk, group, dim)
        do_pad = do.to(torch.float32).reshape(batch, rows, hk, group, dim)
        sel_idx = topk_idx.reshape(hk, batch, rows, topk)
        lse_pad = lse.reshape(hq, batch, rows).permute(1, 0, 2).reshape(batch, hk, group, rows)
        valid = None
    else:
        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        gidx = (cu_seqlens[:-1].reshape(batch, 1) + pos.reshape(1, rows))
        gidx = gidx.reshape(-1).clamp(max=total - 1).to(torch.int64)
        q_pad = torch.index_select(q, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
        k_pad = torch.index_select(k, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
        v_pad = torch.index_select(v, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
        o_pad = torch.index_select(o, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
        do_pad = torch.index_select(do, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
        sel_idx = torch.index_select(topk_idx, 1, gidx).reshape(hk, batch, rows, topk)
        lse_pad = torch.index_select(lse, 1, gidx).reshape(hq, batch, rows)
        lse_pad = lse_pad.permute(1, 0, 2).reshape(batch, hk, group, rows)
        do_pad = do_pad * valid.reshape(batch, rows, 1, 1, 1)

    keep = allowed_mask(sel_idx, device)
    if valid is not None:
        keep = keep & valid.reshape(batch, 1, 1, rows)
    keep = keep.reshape(batch, hk, 1, rows, rows)

    # p = exp2(qk * sm_scale * log2e - lse)，与 kernel 前向/反向同一套 log2 口径
    scores = torch.einsum("bqhgd,bkhd->bhgqk", q_pad, k_pad) * (sm_scale * _LOG2E)
    probs = torch.exp2(scores - lse_pad[..., None])
    probs = probs * keep

    delta = (o_pad * do_pad).sum(-1).permute(0, 2, 3, 1)                     # [B,Hk,g,Lq]
    grad_p = torch.einsum("bqhgd,bkhd->bhgqk", do_pad, v_pad)
    grad_s = probs * (grad_p - delta[..., None])

    grad_q = torch.einsum("bhgqk,bkhd->bqhgd", grad_s, k_pad) * sm_scale
    grad_k = torch.einsum("bhgqk,bqhgd->bkhd", grad_s, q_pad) * sm_scale
    grad_v = torch.einsum("bhgqk,bqhgd->bkhd", probs, do_pad)

    grad_q = grad_q.reshape(batch * rows, hq, dim)
    grad_k = grad_k.reshape(batch * rows, hk, dim)
    grad_v = grad_v.reshape(batch * rows, hk, dim)
    if uniform:
        return grad_q.to(q_dtype), grad_k.to(k_dtype), grad_v.to(v_dtype)

    flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
    out = []
    for grad, heads, width, ref in ((grad_q, hq, dim, q_dtype), (grad_k, hk, dim, k_dtype), (grad_v, hk, dim, v_dtype)):
        buf = torch.zeros(total + 1, heads, width, device=device, dtype=torch.float32)
        buf.index_copy_(0, flat_idx, grad)
        out.append(buf[:total].to(ref))
    return out[0], out[1], out[2]


def _build_topk_idx(lengths, hk, topk, block_size, seed):
    """按上游 test 的选块规则重建合法 topk_idx（每个 query 至少选中块 0）。

    规则与源 get_input_groups 一致：每个 query 在其因果可见块里随机选 topk 个，
    升序排序后把最小的一个强制换成块 0，超出因果范围的块号置 -1 表示 padding。
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    chunks = []
    for length in lengths:
        nblk = (length + block_size - 1) // block_size
        local = torch.arange(length).reshape(1, length, 1)
        score = torch.rand(hk, length, nblk, generator=generator)
        take = min(topk, nblk)
        idx = score.topk(take, dim=-1).indices.to(torch.int32)
        idx, _ = torch.sort(idx, dim=-1)
        if take < topk:
            pad = torch.full((hk, length, topk - take), nblk, dtype=torch.int32)
            idx = torch.cat([idx, pad], dim=-1)
        idx[:, :, 0] = 0
        idx = torch.where(idx > (local // block_size).to(torch.int32),
                          torch.full_like(idx, -1), idx)
        chunks.append(idx)
    return torch.cat(chunks, dim=1).contiguous()


def _saved_forward(q, k, v, topk_idx, block_size, lengths, sm_scale):
    """前向 saved tensor（o / lse）的确定性重建，口径与源 get_input_groups 一致。

    lse 以 log2 为底：scores 先乘 sm_scale*log2(e)，logsumexp 用自然底算完后
    再乘回 log2(e)。返回 (o: q.dtype [total, Hq, D], lse: fp32 [Hq, total])。
    """
    device = q.device
    total, hq, dim = q.shape
    hk = k.shape[1]
    group = hq // hk
    rows = max(lengths)
    batch = len(lengths)
    starts = [0]
    for length in lengths[:-1]:
        starts.append(starts[-1] + length)
    pos = torch.arange(rows, device=device)
    valid = pos.reshape(1, rows) < torch.tensor(lengths, device=device).reshape(batch, 1)
    gidx = (torch.tensor(starts, device=device).reshape(batch, 1) + pos.reshape(1, rows))
    gidx = gidx.reshape(-1).clamp(max=total - 1)

    q_pad = q[gidx].to(torch.float32).reshape(batch, rows, hk, group, dim)
    k_pad = k[gidx].to(torch.float32).reshape(batch, rows, hk, dim)
    v_pad = v[gidx].to(torch.float32).reshape(batch, rows, hk, dim)
    sel_idx = topk_idx[:, gidx].reshape(hk, batch, rows, topk_idx.shape[-1])
    blocks = (rows + block_size - 1) // block_size
    idx = torch.where(sel_idx < 0, torch.full_like(sel_idx, blocks), sel_idx)
    idx = idx.permute(1, 0, 2, 3).to(torch.int64)
    sel = torch.zeros(batch, hk, rows, blocks + 1, dtype=torch.bool, device=device)
    sel.scatter_(-1, idx, True)
    blk_of = torch.arange(blocks, device=device).reshape(blocks, 1).expand(blocks, block_size).reshape(-1)[:rows]
    keep = torch.index_select(sel[..., :blocks], -1, blk_of)
    keep = keep & (pos.reshape(rows, 1) >= pos.reshape(1, rows)).reshape(1, 1, rows, rows)
    keep = keep & valid.reshape(batch, 1, 1, rows)
    keep = keep.reshape(batch, hk, 1, rows, rows)

    scores = torch.einsum("bqhgd,bkhd->bhgqk", q_pad, k_pad) * (sm_scale * _LOG2E)
    scores = scores.masked_fill(~keep, float("-inf"))
    lse_pad = torch.logsumexp(scores * math.log(2.0), dim=-1) * _LOG2E     # [B,Hk,g,Lq]
    probs = torch.exp2(scores - lse_pad[..., None])
    out = torch.einsum("bhgqk,bkhd->bqhgd", probs, v_pad).reshape(batch * rows, hq, dim)

    o_flat = torch.zeros(total + 1, hq, dim, dtype=torch.float32, device=device)
    flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
    o_flat.index_copy_(0, flat_idx, out)
    lse_flat = torch.zeros(hq, total + 1, dtype=torch.float32, device=device)
    lse_flat.index_copy_(1, flat_idx, lse_pad.reshape(batch, hq, rows).permute(1, 0, 2).reshape(hq, -1))
    return o_flat[:total].to(q.dtype), lse_flat[:, :total].contiguous()


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_size: int = 64,
    max_seqlen: int = 1,
    sm_scale: float = 1.0,
    **kwargs,
) -> list:
    """重建满足本算子契约的结构化输入（同时替换 golden 与候选输入，比较公平）。

    单区间 value_range 无法表达三条契约：
      1. cu_seqlens 必须是单调前缀和，且 cu_seqlens[-1] == total_len；
      2. topk_idx 的每个块号必须落在 [0, ceil(len/block_size)) 内，且每个 query
         至少选中一个因果可见块（保证 softmax 分母非空）；
      3. o / lse 必须是 q/k/v/topk_idx 的确定性前向结果（反向消费的 saved tensor）。
    这里按实际 shape 用等长切分重建 cu_seqlens 与 topk_idx，并按同一套
    log2 域前向重算 o / lse；q/k/v/do 原样保留。

    Returns:
        [q, k, v, topk_idx, cu_seqlens, o, lse, do]，顺序与 golden 签名一致。
    """
    device = q.device
    total = int(q.shape[0])
    hk = int(k.shape[1])
    batch = max(int(cu_seqlens.numel()) - 1, 1)
    block_size = int(block_size)
    topk = int(topk_idx.shape[-1])
    sm_scale = float(sm_scale)

    base = total // batch
    rem = total % batch
    lengths = [base + (1 if b < rem else 0) for b in range(batch)]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    new_cu = torch.tensor(cu, dtype=cu_seqlens.dtype, device=device)

    new_topk = _build_topk_idx(lengths, hk, topk, block_size, 46).to(
        dtype=topk_idx.dtype, device=device)
    new_o, new_lse = _saved_forward(q, k, v, new_topk, block_size, lengths, sm_scale)
    new_lse = new_lse.to(lse.dtype)
    return [q, k, v, new_topk, new_cu, new_o, new_lse, do]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

total, hq, hk, d, topk = 8192, 4, 1, 64, 5
block_size, max_seqlen, sm_scale = 32, 8192, 1.0 / (d ** 0.5)

q = torch.randn(total, hq, d, dtype=torch.float32, device="npu")
k = torch.randn(total, hk, d, dtype=torch.float32, device="npu")
v = torch.randn(total, hk, d, dtype=torch.float32, device="npu")
topk_idx = torch.randint(0, max_seqlen // block_size, (hk, total, topk),
                         dtype=torch.int32, device="npu")
cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device="npu")
o = torch.randn(total, hq, d, dtype=torch.float32, device="npu")
lse = torch.rand(hq, total, dtype=torch.float32, device="npu")
do = torch.randn(total, hq, d, dtype=torch.float32, device="npu")

dq, dk, dv = cann_bench.nsa_topk_sparse_attention_bwd(
    q, k, v, topk_idx, cu_seqlens, o, lse, do,
    block_size, max_seqlen, sm_scale,
)
# dq: [8192, 4, 64], dk/dv: [8192, 1, 64], float32
```
