# SlaFwd 算子 API 描述

## 1. 算子简介

SparseLinearAttention（SLA）前向。它把块稀疏注意力分支 `o_s` 与线性注意力分支 `o_l`
融合输出：前者在每个 query 块选中的 key 块上做标准 softmax 注意力，后者用特征图
`phi` 计算线性注意力并做输出投影。

**主要应用场景**：
- 长序列训练 / 推理中的稀疏 + 线性注意力融合
- thu-ml SLA 的 SparseLinearAttention 前向

**算子特征**：
- 难度等级：L4（SparseAttention）
- 4 个输入（q、k、v、W_l）+ 4 个属性
- 非因果（non-causal）注意力
- 块选择遵循官方精度路径：块均值与块分数回铸到 bf16/fp16 后 topk
- 输出 dtype 与 q 一致

## 2. 算子定义

### 数学公式

$$
\tilde k = k-\mathrm{mean}(k,\ dim=-2),\quad
q_m=\mathrm{block\_mean}(q,BLKQ),\quad k_m=\mathrm{block\_mean}(\tilde k,BLKK)
$$

$$
o_s=\mathrm{softmax}\big(q k^{T}\cdot D^{-0.5}\ \big|\ \text{选中的 key 块}\big)\,v,
\qquad
o_l=\frac{\phi(q)\,(\phi(k)^{T}v)}{10^{-5}+\big(\phi(q)\cdot \sum \phi(k)\big)}\,W_l^{T}
$$

$$
y=o_s+o_l
$$

其中 `phi` 由 `feature_map` 选择：`softmax` / `elu+1` / `relu`。

### 实现建议

1. 对 q / k 做块均值得到池化分数，按 `topk_ratio` 选 key 块；
2. 稀疏分支在选中块展开的 token 上做 masked softmax（尾块越界列屏蔽）；
3. 线性分支按官方 torch 路径逐步做 dtype 舍入镜像；
4. 两分支 fp32 相加后一次回铸到 q.dtype。

## 3. 接口规范

### 算子原型

```python
sla_fwd(Tensor q, Tensor k, Tensor v, Tensor W_l, float topk, str feature_map,
        int BLKQ, int BLKK) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | query，[B, H, L, D] |
| k | Tensor | 必选 | key，[B, H, L, D] |
| v | Tensor | 必选 | value，[B, H, L, D] |
| W_l | Tensor | 必选 | 线性分支输出投影权重，[D, D] |
| topk | float | 必选 | key 块选择比例，0 < topk <= 1 |
| feature_map | string | 必选 | "elu" / "relu" / "softmax" |
| BLKQ | int | 必选 | query 分块大小 |
| BLKK | int | 必选 | key 分块大小 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, H, L, D] | 与 q 相同 | 融合输出 |

### 数据类型

| 输入 dtype（q/k/v/W_l） | 输出 dtype |
|-------------------------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- q / k / v 形状与 dtype 完全一致（MHA）
- `0 < topk <= 1`，且 `min(NK, int(topk * NK)) >= 1`（否则官方 kernel 输出 NaN，未定义）
- `feature_map` 仅支持 softmax / elu / relu
- `BLKQ`、`BLKK` 为正整数，尾块按实际 token 数取均值
- 块选择在低精度下完成（与官方 kernel 一致），离散选择对舍入敏感

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
SlaFwd 算子 Torch Golden 参考实现

SparseLinearAttention 前向：块稀疏注意力分支 o_s 与线性注意力分支 o_l 的可训练融合。

公式:
    arg_k = k - mean(k, dim=-2)
    qm = 块均值(q, BLKQ), km = 块均值(arg_k, BLKK)
    score = qm @ km^T, 每个 query 块选 topk_ratio 比例的最高分 key 块
    o_s = softmax(sparse(q @ k^T * scale)) @ v           (块稀疏分支)
    o_l = (phi(q) @ (phi(k)^T @ v)) / (1e-5 + (phi(q) * sum(phi(k))).sum(-1)) @ W_l^T   (线性分支)
    y = o_s + o_l
"""


def _block_mean(x: torch.Tensor, blk: int) -> torch.Tensor:
    """按最后一维之外的时间维分块求均值（尾块按实际 token 数）。"""
    device = x.device
    b, h, length, d = x.shape
    lb = (length + blk - 1) // blk
    pad = lb * blk - length
    xp = F.pad(x.float(), (0, 0, 0, pad)).view(b, h, lb, blk, d)
    cnt = torch.full((lb,), blk, dtype=torch.float32, device=device)
    if pad:
        cnt[-1] = blk - pad
    return (xp.sum(3) / cnt.view(1, 1, lb, 1)).to(x.dtype)


def sla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W_l: torch.Tensor,
    topk: float,
    feature_map: str,
    BLKQ: int,
    BLKK: int,
) -> torch.Tensor:
    """
    SparseLinearAttention 前向

    Args:
        q: query [B, H, L, D]
        k: key [B, H, L, D]
        v: value [B, H, L, D]
        W_l: 线性分支输出投影权重 [D, D]
        topk: key 块选择比例，0 < topk <= 1
        feature_map: 线性分支特征图，"elu" / "relu" / "softmax"
        BLKQ: query 分块大小
        BLKK: key 分块大小

    Returns:
        融合输出 [B, H, L, D]
    """
    device = q.device
    b, h, length, d = q.shape
    assert k.shape == v.shape == q.shape, "SLA 要求 q/k/v 同形状（MHA）"
    assert q.dtype == k.dtype == v.dtype
    assert isinstance(feature_map, str) and feature_map in ("elu", "relu", "softmax")
    assert 0.0 < topk <= 1.0

    dtype = q.dtype

    # ---- 1. 块选择（get_block_map，精度路径镜像官方实现） ----
    arg_k = k - k.float().mean(dim=-2, keepdim=True).to(dtype)
    qm = _block_mean(q, BLKQ)
    km = _block_mean(arg_k, BLKK)
    pooled = (qm.float() @ km.float().transpose(-1, -2)).to(dtype)
    NK = pooled.shape[-1]
    real_topk = min(NK, int(topk * NK))
    assert real_topk >= 1, "real_topk=0 时官方 kernel 输出 NaN，未定义"
    lut = torch.topk(pooled.float(), real_topk, dim=-1, sorted=False).indices
    lut = lut.sort(dim=-1).values                            # [B, H, MQ, real_topk]

    scale = d ** -0.5
    if feature_map == "softmax":
        fmap = lambda x: F.softmax(x, dim=-1)
    elif feature_map == "elu":
        fmap = lambda x: F.elu(x) + 1
    else:
        fmap = lambda x: F.relu(x)

    MQ = pooled.shape[2]
    kf, vf, qf = k.float(), v.float(), q.float()

    # ---- 2. 稀疏分支 o_s（_attn_fwd 精度路径镜像） ----
    tok_idx = lut[..., None] * BLKK + torch.arange(BLKK, device=device)
    S = real_topk * BLKK
    tok_valid = (tok_idx < length).reshape(b, h, MQ, S)
    tok_idx = tok_idx.reshape(b, h, MQ, S)

    k2 = kf.reshape(b * h, length, d)
    v2 = vf.reshape(b * h, length, d)
    idx2 = tok_idx.reshape(b * h, MQ, S).clamp(0, length - 1)
    tv2 = tok_valid.reshape(b * h, MQ, 1, S)
    qpad = F.pad(qf.reshape(b * h, length, d),
                 (0, 0, 0, MQ * BLKQ - length)).reshape(b * h, MQ, BLKQ, d) * scale

    o_s = torch.empty(b * h, MQ, BLKQ, d, dtype=torch.float32, device=device)
    mq_chunk = 16
    for s0 in range(0, MQ, mq_chunk):
        e0 = min(MQ, s0 + mq_chunk)
        gi = idx2[:, s0:e0].unsqueeze(-1).expand(b * h, e0 - s0, S, d)
        k_g = torch.gather(k2.unsqueeze(1).expand(b * h, e0 - s0, length, d), 2, gi)
        v_g = torch.gather(v2.unsqueeze(1).expand(b * h, e0 - s0, length, d), 2, gi)
        sc = torch.einsum("bmqd,bmsd->bmqs", qpad[:, s0:e0], k_g)
        sc = sc.masked_fill(~tv2[:, s0:e0], float("-inf"))
        # kernel: p 以 fp32 求 exp2 与分母 l_i，PV 前舍回 v.dtype，落盘再舍回 v.dtype
        e = torch.exp(sc - sc.amax(dim=-1, keepdim=True))
        l_i = e.sum(-1, keepdim=True)
        acc = torch.einsum("bmqs,bmsd->bmqd", e.to(dtype).float(), v_g)
        o_s[:, s0:e0] = (acc / l_i).to(dtype).float()
    o_s = o_s.reshape(b, h, MQ * BLKQ, d)[:, :, :length]

    # ---- 3. 线性分支 o_l（core.py 官方路径，全程模型 dtype） ----
    c_q = fmap(qf).to(dtype).float()
    c_k = fmap(kf).to(dtype).float()
    kvsum = (c_k.transpose(-1, -2) @ vf).to(dtype).float()
    ksum = c_k.sum(dim=-2, keepdim=True).to(dtype).float()
    num = (c_q @ kvsum).to(dtype).float()
    den = (c_q * ksum).to(dtype).float().sum(-1, keepdim=True).to(dtype).float()
    den = (1e-5 + den).to(dtype).float()
    o_l = (num / den).to(dtype).float()
    o_l = (o_l @ W_l.float().transpose(-1, -2)).to(dtype).float()

    # ---- 4. 融合 ----
    return (o_s + o_l).to(dtype)
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, H, L, D = 1, 4, 1024, 64
q = torch.randn(B, H, L, D, dtype=torch.bfloat16)
k = torch.randn(B, H, L, D, dtype=torch.bfloat16)
v = torch.randn(B, H, L, D, dtype=torch.bfloat16)
W_l = torch.randn(D, D, dtype=torch.bfloat16) / D ** 0.5
y = cann_bench.sla_fwd(q, k, v, W_l, topk=0.125, feature_map="softmax", BLKQ=64, BLKK=64)
```

### 测试用例覆盖

- dtype：float16、bfloat16
- feature_map：softmax / elu / relu
- topk：1.0 / 0.5 / 0.25 / 0.166667 / 0.125
- BLKQ：64 / 128；L 从 1 到 1024（覆盖小于分块的非整块场景）
- D ∈ {16, 32, 64}；H ∈ {1, 2, 4, 8}
