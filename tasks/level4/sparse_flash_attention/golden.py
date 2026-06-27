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
SparseFlashAttention算子Torch Golden参考实现

稀疏注意力，支持 GQA（N1 != N2）、不同 head dim（Dk != Dv）和 BSND/BNSD 布局
query 仅与 sparseIndices 指定的 KV 子集计算注意力

语义约束:
    value 与 key 在数值上同源 (MLA latent KV cache):
      - Dk == Dv 时: value == key (完全相同)
      - Dk >  Dv 时: value == key[..., :Dv] (前缀切片，典型 MLA Dk=576 Dv=512)
    接口保留独立入参以兼容通用 attention API；调用方需保证此关系，
    本 Golden 实现不做强制检查。

公式: mask = scatter(sparseIndices) -> bool[B, N2, S1, S2]
      scores = Q @ K^T * scaleValue，mask 外位置置 -inf
      y = softmax(scores) @ V          (V 语义上等于 K[..., :Dv])
假定 sparseIndices 在同一 (b, n2, s1) 下无重复值
"""


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """
    稀疏 FlashAttention，支持 GQA、不同 head dim 和 BSND/BNSD 布局

    语义约束: value 与 key 在数值上同源 (MLA latent KV cache):
        - Dk == Dv 时: value == key
        - Dk >  Dv 时: value == key[..., :Dv]   (典型 MLA: Dk=576, Dv=512)
        本 Golden 不做强制检查，调用方需保证此关系。

    Args:
        query: 查询张量，BSND: [B, S1, N1, Dk]，BNSD: [B, N1, S1, Dk]
        key: 键张量，BSND: [B, S2, N2, Dk]，BNSD: [B, N2, S2, Dk]
        value: 值张量，BSND: [B, S2, N2, Dv]，BNSD: [B, N2, S2, Dv]；语义上等于 key[..., :Dv]
        sparseIndices: 稀疏索引（int32），BSND: [B, S1, N2, topK]，BNSD: [B, N2, S1, topK]
        scaleValue: 缩放因子
        inputLayout: 张量布局，"BSND" 或 "BNSD"
        is_causal: 是否启用因果掩码（右下角对齐），True 时在稀疏选择之上额外屏蔽
            KV 序列位置 idx > s1 + (S2 - S1) 的条目。要求 S1 <= S2。

    Returns:
        注意力输出，布局与输入一致
    """
    # 统一转为 BNSD 内部计算
    if inputLayout == "BSND":
        q = query.permute(0, 2, 1, 3)              # [B, N1, S1, Dk]
        k = key.permute(0, 2, 1, 3)                 # [B, N2, S2, Dk]
        v = value.permute(0, 2, 1, 3)               # [B, N2, S2, Dv]
        si = sparseIndices.permute(0, 2, 1, 3)      # [B, N2, S1, topK]
    else:  # BNSD
        q, k, v, si = query, key, value, sparseIndices

    B, N1, S1, Dk = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    G = N1 // N2

    # 用 scatter 把稀疏索引转成 bool mask: [B, N2, S1, S2]
    mask = torch.zeros(B, N2, S1, S2, dtype=torch.bool, device=q.device)
    mask.scatter_(-1, si.long(), True)

    # 因果掩码（右下角对齐）：j > i + (S2 - S1) 的列从可选集合中剔除
    if is_causal:
        s1_idx = torch.arange(S1, device=q.device).unsqueeze(-1)  # [S1, 1]
        s2_idx = torch.arange(S2, device=q.device).unsqueeze(0)   # [1, S2]
        causal_keep = s2_idx <= (s1_idx + (S2 - S1))               # [S1, S2]
        mask = mask & causal_keep  # 广播到 [B, N2, S1, S2]

    # GQA: 通过 reshape 把 N1 拆成 (N2, G)，避免物化 K/V 的 head 扩展
    q_g = q.reshape(B, N2, G, S1, Dk)

    # 计算完整 scores 后用 mask 屏蔽未选位置: [B, N2, G, S1, S2]
    # 在同一 (b, n2, s1) 下 sparseIndices 无重复，mask 与 gather 语义等价
    scores = torch.einsum('bngsd,bnkd->bngsk', q_g, k) * scaleValue
    scores.masked_fill_(~mask.unsqueeze(2), float('-inf'))

    # 原地 softmax，避免同时持有 scores 和 attn_weights 两份张量
    # 处理全 mask 行：当某行全被 mask 时，max 为 -inf，需要特殊处理
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)  # 检测全 -inf 行
    scores -= scores_max
    scores.exp_()
    scores_sum = scores.sum(dim=-1, keepdim=True)
    # 全 mask 行保持为 0，不进行除法
    scores = torch.where(all_masked, torch.zeros_like(scores), scores / scores_sum)

    # 加权求和: [B, N2, G, S1, Dv] -> [B, N1, S1, Dv]
    out = torch.einsum('bngsk,bnkd->bngsd', scores, v)
    out = out.reshape(B, N1, S1, -1)

    # 转回原始布局
    if inputLayout == "BSND":
        return out.permute(0, 2, 1, 3)   # [B, S1, N1, Dv]
    else:
        return out                        # [B, N1, S1, Dv]


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float = 1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
    **kwargs,
):
    """重建两个结构化输入，满足本算子的契约（同时替换 golden 与候选输入，比较公平）。

    单区间 value_range 无法表达两条契约：
      1. sparseIndices 在同一 (b, n2, s1) 下必须是 [0, S2) 内**无重复**的 topK 个索引
         （golden 用 scatter 建 mask，重复会塌缩成更少的 True；见 golden 注释 L32/L95）。
         通用生成器按 value_range 填随机 int，既越界又大量重复 -> mask 与 kernel 的
         gather 语义不一致 -> 整体发散。
      2. value 与 key 数值同源：value == key[..., :Dv]（同一份 latent KV cache 的前缀
         视图，proto L? 明示；Dk==Dv 时退化为 value==key）。通用生成器给的是无关随机张量。

    这里据 key/value/sparseIndices 的实际形状（不写死，覆盖 BSND/BNSD、GQA、Dk!=Dv）
    重生成这两者；query、key 原样保留。索引不做排序（保持与 golden 的 scatter 语义一致，
    无序无重复即可）。

    Returns:
        [query, key, value, sparseIndices]，顺序与 sparse_flash_attention 签名一致。
    """
    # S2 sits on a different axis per layout (BSND key dim1, BNSD key dim2)
    S2 = int(key.shape[1] if inputLayout == "BSND" else key.shape[2])
    Dv = int(value.shape[-1])

    # contract 2: value must be key's [:Dv] prefix view (same latent KV cache, not independent)
    new_value = key[..., :Dv].clone().to(dtype=value.dtype, device=value.device)

    # contract 1: golden's scatter mask collapses duplicate indices -> they must be distinct
    lead = list(sparseIndices.shape[:-1])
    topK = int(sparseIndices.shape[-1])
    num_groups = 1
    for d in lead:
        num_groups *= int(d)
    k = min(topK, S2)
    g = torch.Generator().manual_seed(0)  # fixed seed: must be reproducible across eval runs
    # argsort of random == permutation; first k are distinct (order irrelevant to scatter)
    perm = torch.rand(num_groups, S2, generator=g).argsort(dim=1)
    sel = perm[:, :k]
    if k < topK:  # topK > S2 (degenerate): can't draw enough distinct -> pad with last col
        sel = torch.cat([sel, sel[:, -1:].expand(num_groups, topK - k)], dim=1)
    new_si = sel.reshape(*lead, topK).to(dtype=sparseIndices.dtype, device=sparseIndices.device)

    return [query, key, new_value, new_si]
