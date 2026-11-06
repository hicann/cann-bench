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
BlockSparseAttnBwd 算子 Torch Golden 参考实现

MIT Han Lab Block-Sparse-Attention（基于 FlashAttention-2.4.2 改造）的
`BlockSparseAttnFunc.backward`。前向的 softmax_max / softmax_sum / attention_in
作为 saved tensor 预给，反向只做纯梯度计算：

    P = exp(S - softmax_max) / softmax_sum      (S = scale * Q K^T，按头掩码置 -inf)
    dV = P^T dY
    dP = dY V^T
    delta = rowsum(dY ⊙ O)
    dS = P ⊙ (dP - delta)
    dQ = dS K scale
    dK = dS^T Q scale

头掩码由 head_mask_type 决定：0=dense，1=block sparse（base_blockmask），
-1=streaming（streaming_info 的 sink/local 窗口）。GQA：KV 头按 H // HK 扩展，
梯度在组内求和恢复 [total_k, HK, D]。逐序列循环（varlen 打包输入），全程 fp32
计算后回铸 q.dtype。
"""

# kernel 固定的块粒度（block streaming / blocksparse 均为 128×128 块）
BLOCK = 128


def _ceil_div(x, y):
    return (x + y - 1) // y


def _streaming_keep_blocks(sq, sk, causal, device):
    """块级 streaming 掩码的公共部分：返回 (jj, mr, valid_row)。"""
    nrow, ncol = _ceil_div(sq, BLOCK), _ceil_div(sk, BLOCK)
    ii = torch.arange(nrow, device=device).view(1, nrow, 1)
    jj = torch.arange(ncol, device=device).view(1, 1, ncol)
    if causal:
        start = max((sq - sk) // BLOCK, 0)
        mr = _ceil_div(max(sk - sq, 0), BLOCK) + 1 + (ii - start)
        valid_row = ii >= start
    else:
        mr = torch.full_like(ii, ncol)
        valid_row = torch.ones(1, nrow, 1, dtype=torch.bool, device=device)
    return jj, mr, valid_row


def _build_blocked_all(bs_head_idx, bs_ranks, blk_mask_b,
                       st_head_idx, st_sink, st_local,
                       sq, sk, is_causal, exact_streaming, H, device):
    """返回 [H, sq, sk] 的 blocked 掩码；无需任何掩码时返回 None。"""
    has_bs = bs_head_idx is not None
    has_st = st_head_idx is not None
    if not is_causal and not has_bs and not has_st:
        return None                                    # 全 dense 且非因果：原实现同样跳过
    blocked = torch.zeros(H, sq, sk, dtype=torch.bool, device=device)

    if has_bs:
        bm = blk_mask_b[bs_ranks]                      # [nb, nrow, ncol]，gather 一次
        nb_, nr_, nc_ = bm.shape
        # 128×128 块展开：expand+reshape（与 repeat_interleave 逐元素一致）
        bm = (bm[:, :, None, :, None]
              .expand(nb_, nr_, BLOCK, nc_, BLOCK)
              .reshape(nb_, nr_ * BLOCK, nc_ * BLOCK))
        blocked[bs_head_idx] = ~bm[:, :sq, :sk]

    if has_st:
        if exact_streaming:
            row = torch.arange(sq, device=device).view(1, sq, 1)
            col = torch.arange(sk, device=device).view(1, 1, sk)
            ns = st_head_idx.numel()
            sink = st_sink.view(ns, 1, 1)
            local = st_local.view(ns, 1, 1)
            # 与官方 construct_exact_streaming_mask 逐元素一致
            blocked_st = torch.logical_or(
                col > torch.minimum(row + sk - sq, torch.full_like(col, sk)),
                torch.logical_and(col < row + sk - sq - (local - 1),
                                  col >= sink))
            blocked[st_head_idx] = blocked_st
        else:
            jj, mr, valid_row = _streaming_keep_blocks(sq, sk, is_causal, device)
            ns = st_head_idx.numel()
            sink = st_sink.view(ns, 1, 1)
            local = st_local.view(ns, 1, 1)
            # keep = valid_row & (窗口 | sink)，blocked 取其反
            ncol = _ceil_div(sk, BLOCK)
            win = (jj >= (mr - local).clamp(min=0)) & (jj < mr.clamp(max=ncol))
            keep_blk = valid_row & (win | (jj < sink))   # [ns, nrow, ncol]
            ns_, nr_, nc_ = keep_blk.shape
            keep_blk = (keep_blk[:, :, None, :, None]
                        .expand(ns_, nr_, BLOCK, nc_, BLOCK)
                        .reshape(ns_, nr_ * BLOCK, nc_ * BLOCK))
            blocked[st_head_idx] = ~keep_blk[:, :sq, :sk]

    if is_causal:
        row = torch.arange(sq, device=device).view(1, sq, 1)
        col = torch.arange(sk, device=device).view(1, 1, sk)
        blocked |= col > row + (sk - sq)               # 原实现 keep & causal_keep 的对偶
    return blocked


def block_sparse_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    base_blockmask: torch.Tensor,
    dout: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_in: torch.Tensor,
    softmax_scale: float = None,
    is_causal: bool = False,
    exact_streaming: bool = False,
) -> tuple:
    """
    BlockSparseAttnBwd 反向（varlen 打包输入）

    Args:
        q: query [total_q, H, D]
        k: key [total_k, HK, D]，H % HK == 0
        v: value [total_k, HK, D]
        cu_seqlens_q: query 序列前缀和 [B+1] int32
        cu_seqlens_k: key 序列前缀和 [B+1] int32
        head_mask_type: [H] int32，0=dense / 1=blocksparse / -1=streaming
        streaming_info: [2H] int32，(sink, local) 成对
        base_blockmask: 稀疏块掩码 [B, H//3, nrow, ncol] bool
        dout: 输出梯度 [total_q, H, D]
        softmax_max: 前向行最大值 [B, H, max_q] fp32（saved tensor）
        softmax_sum: 前向行归一化因子 [B, H, max_q] fp32（saved tensor）
        attention_in: 前向注意力输出 [total_q, H, D] fp32（saved tensor）
        softmax_scale: 缩放因子，None 表示 D ** -0.5
        is_causal: 是否因果掩码
        exact_streaming: streaming 掩码是否用 exact 语义

    Returns:
        (dq, dk, dv)，shape 分别与 q / k / v 一致（q.dtype）
    """
    total_q, H, D = q.shape
    total_k, HK, _ = k.shape
    B = cu_seqlens_q.shape[0] - 1
    dtype = q.dtype
    G = H // HK
    scale = softmax_scale if softmax_scale is not None else D ** -0.5

    # 内部统一 fp32（源实现口径）；兼容 evaluator 的 fp64 golden precision 与
    # bf16/fp16 原生精度两条路径。输出按原始 q dtype 回铸。
    q = q.float()
    k = k.float()
    v = v.float()
    dout = dout.float()
    softmax_max = softmax_max.float()
    softmax_sum = softmax_sum.float()
    attention_in = attention_in.float()

    dq = torch.zeros(total_q, H, D, dtype=torch.float32, device=q.device)
    dk = torch.zeros(total_k, HK, D, dtype=torch.float32, device=q.device)
    dv = torch.zeros(total_k, HK, D, dtype=torch.float32, device=q.device)

    # 元数据一次同步取回（int32 小向量拼接后单次 tolist）
    meta = torch.cat([cu_seqlens_q, cu_seqlens_k,
                      head_mask_type, streaming_info]).tolist()
    cuq = meta[:B + 1]
    cuk = meta[B + 1:2 * B + 2]
    hmt = meta[2 * B + 2:2 * B + 2 + H]
    si = meta[2 * B + 2 + H:]

    # 头下标（主机侧构造后一次上送）
    bs_head_l = [i for i, t in enumerate(hmt) if t == 1]
    st_head_l = [i for i, t in enumerate(hmt) if t == -1]
    bs_head_idx = torch.tensor(bs_head_l, device=q.device) if bs_head_l else None
    st_head_idx = torch.tensor(st_head_l, device=q.device) if st_head_l else None
    bs_ranks = torch.arange(len(bs_head_l), device=q.device) if bs_head_l else None
    st_sink = torch.tensor([si[2 * i] for i in st_head_l],
                           device=q.device) if st_head_l else None
    st_local = torch.tensor([si[2 * i + 1] for i in st_head_l],
                            device=q.device) if st_head_l else None

    for b in range(B):
        qs, qe = cuq[b], cuq[b + 1]
        ks, ke = cuk[b], cuk[b + 1]
        sq, sk = qe - qs, ke - ks
        q_b = q[qs:qe].float()                               # [sq, H, D]
        # GQA 头复用：expand+reshape（与 repeat_interleave(G, dim=1) 逐元素一致）
        k_b = (k[ks:ke].float()[:, :, None, :]
               .expand(sk, HK, G, D).reshape(sk, H, D))      # [sk, H, D]
        v_b = (v[ks:ke].float()[:, :, None, :]
               .expand(sk, HK, G, D).reshape(sk, H, D))      # [sk, H, D]
        dy_b = dout[qs:qe].float()                           # [sq, H, D]
        o_b = attention_in[qs:qe].float()                    # 前向输出，预给

        S = torch.einsum('t h d, s h d -> h t s', q_b * scale, k_b)
        blocked = _build_blocked_all(
            bs_head_idx, bs_ranks,
            base_blockmask[b] if bs_head_idx is not None else None,
            st_head_idx, st_sink, st_local,
            sq, sk, is_causal, exact_streaming, H, q.device)
        if blocked is not None:
            S.masked_fill_(blocked, float('-inf'))
        m = softmax_max[b, :, :sq].unsqueeze(-1)               # [H, sq, 1] 预给
        l = softmax_sum[b, :, :sq].unsqueeze(-1)               # [H, sq, 1] 预给
        P = torch.exp(S - m) / l
        P = torch.nan_to_num(P, nan=0.0)                       # 全掩码行 P=0
        delta = (dy_b * o_b).sum(dim=-1).transpose(0, 1).unsqueeze(-1)  # rowsum(o⊙dY) [H,sq,1]
        dP = torch.einsum('t h d, s h d -> h t s', dy_b, v_b)
        dS = P * (dP - delta)
        dv_full = torch.einsum('h t s, t h d -> s h d', P, dy_b)          # dV = Pᵀ·dY
        dq_b = torch.einsum('h t s, s h d -> t h d', dS, k_b) * scale     # dQ = dS·K·scale
        dk_full = torch.einsum('h t s, t h d -> s h d', dS, q_b) * scale  # dK = dSᵀ·Q·scale

        dq[qs:qe] = dq_b
        dk[ks:ke] = dk_full.view(sk, HK, G, D).sum(dim=2)
        dv[ks:ke] = dv_full.view(sk, HK, G, D).sum(dim=2)

    return dq.to(dtype), dk.to(dtype), dv.to(dtype)


def _gen_base_blockmask(Bsz, nblk, nrow, ncol, sparsity, causal, device):
    """按官方 generate_base_sparsity_mask 的逐行同构逻辑重建块掩码（固定种子）。"""
    gen = torch.Generator()
    gen.manual_seed(56)
    bm = torch.zeros(Bsz, nblk, nrow, ncol, dtype=torch.bool)
    for b in range(Bsz):
        for h in range(nblk):
            if sparsity != 0.0 and sparsity != 1.0:
                for i in range(nrow):
                    idx = nrow - i - 1
                    avail = max(0, ncol - i) if causal else ncol
                    num_one = max(1, int(sparsity * avail))
                    if avail > 0:
                        bm[b, h, idx, torch.randperm(avail, generator=gen)[:num_one]] = True
            elif sparsity == 1.0:
                bm[b, h] = True
    return bm.to(device)


def _forward_saved(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                   streaming_info, base_blockmask, softmax_scale, is_causal,
                   exact_streaming):
    """由叶子输入重建前向 saved tensor（softmax_max / softmax_sum / attention_in）。

    与源 get_input_groups 中预计算统计量的逻辑逐行同构；fp32 计算。
    """
    total_q, H, D = q.shape
    total_k, HK, _ = k.shape
    B = cu_seqlens_q.shape[0] - 1
    G = H // HK
    sc = softmax_scale if softmax_scale is not None else D ** -0.5

    cuq = cu_seqlens_q.tolist()
    cuk = cu_seqlens_k.tolist()
    hmt_l = head_mask_type.tolist()
    max_q = max(cuq[i + 1] - cuq[i] for i in range(B))
    max_k = max(cuk[i + 1] - cuk[i] for i in range(B))

    is_bs = torch.tensor([t == 1 for t in hmt_l], device=q.device)
    is_st = torch.tensor([t == -1 for t in hmt_l], device=q.device)
    bs_head_idx = is_bs.nonzero(as_tuple=True)[0]
    st_head_idx = is_st.nonzero(as_tuple=True)[0]
    bs_ranks = (is_bs.cumsum(0) - 1)[bs_head_idx] if bs_head_idx.numel() else None
    st_sink = streaming_info[2 * st_head_idx] if st_head_idx.numel() else None
    st_local = streaming_info[2 * st_head_idx + 1] if st_head_idx.numel() else None
    bs_head_idx = bs_head_idx if bs_head_idx.numel() else None
    st_head_idx = st_head_idx if st_head_idx.numel() else None

    softmax_max = torch.zeros(B, H, max_q, dtype=torch.float32, device=q.device)
    softmax_sum = torch.ones(B, H, max_q, dtype=torch.float32, device=q.device)
    attention_in = torch.zeros(total_q, H, D, dtype=torch.float32, device=q.device)

    for b in range(B):
        qs, qe = cuq[b], cuq[b + 1]
        ks, ke = cuk[b], cuk[b + 1]
        sq, sk = qe - qs, ke - ks
        q_b = q[qs:qe].float()
        k_b = k[ks:ke].float().repeat_interleave(G, dim=1)
        v_b = v[ks:ke].float().repeat_interleave(G, dim=1)
        S = torch.einsum('t h d, s h d -> h t s', q_b * sc, k_b)
        blocked = _build_blocked_all(
            bs_head_idx, bs_ranks,
            base_blockmask[b] if bs_head_idx is not None else None,
            st_head_idx, st_sink, st_local,
            sq, sk, is_causal, exact_streaming, H, q.device)
        if blocked is not None:
            S.masked_fill_(blocked, float('-inf'))
        m = S.amax(dim=-1)                          # [H, sq]
        p = torch.nan_to_num(torch.exp(S - m.unsqueeze(-1)), nan=0.0)
        l = p.sum(dim=-1)                           # [H, sq]，全掩码行为 0
        P = torch.nan_to_num(p / l.unsqueeze(-1), nan=0.0)
        row_empty = (l == 0)
        m = torch.where(row_empty, torch.zeros_like(m), m)
        l = torch.where(row_empty, torch.ones_like(l), l)
        softmax_max[b, :, :sq] = m
        softmax_sum[b, :, :sq] = l
        attention_in[qs:qe] = torch.einsum('h t s, s h d -> t h d', P, v_b)
    return softmax_max, softmax_sum, attention_in


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    base_blockmask: torch.Tensor,
    dout: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_in: torch.Tensor,
    softmax_scale: float = None,
    is_causal: bool = False,
    exact_streaming: bool = False,
    q_lens=None,
    k_lens=None,
    hk=None,
    sparsity: float = 0.5,
    sink: int = 1,
    local: int = 1,
    **kwargs,
) -> list:
    """重建满足算子契约的结构化输入（同时替换 golden 与候选输入，比较公平）。

    单区间 value_range 无法表达下列约束，统一在此重建：
      1. cu_seqlens_q/k 必须是单调前缀和且末元素等于 total_q/total_k；
      2. head_mask_type 必须是 0/1/-1，且 1 与 -1 的个数与 base_blockmask 的
         nblk 维（H//3）一致；
      3. streaming_info 必须是 (sink, local) 成对重复；
      4. base_blockmask 的 [B, H//3, nrow, ncol] 必须与 128×128 块划分匹配；
      5. softmax_max / softmax_sum / attention_in 必须是 q/k/v 的确定性前向结果。

    Returns:
        12 个张量，顺序与 golden 签名一致。
    """
    device = q.device
    total_q, H, D = q.shape
    total_k, HK, _ = k.shape
    B = int(cu_seqlens_q.numel()) - 1
    scale = softmax_scale if softmax_scale is not None else D ** -0.5

    # 1) cu_seqlens：优先采用 source case 的 q_lens/k_lens，否则等分
    def _lens(value, total, batch):
        if value is not None:
            lens = [int(x) for x in (value.tolist() if isinstance(value, torch.Tensor) else value)]
            if len(lens) == batch and sum(lens) == total:
                return lens
        base = total // batch
        rem = total % batch
        return [base + (1 if b < rem else 0) for b in range(batch)]

    q_lens_l = _lens(q_lens, total_q, B)
    k_lens_l = _lens(k_lens, total_k, B)
    cuq = [0]
    for length in q_lens_l:
        cuq.append(cuq[-1] + length)
    cuk = [0]
    for length in k_lens_l:
        cuk.append(cuk[-1] + length)
    new_cuq = torch.tensor(cuq, dtype=torch.int32, device=device)
    new_cuk = torch.tensor(cuk, dtype=torch.int32, device=device)

    # 2) head_mask_type / streaming_info
    ns, nb = H // 3, H // 3
    new_hmt = torch.tensor([0] * (H - ns - nb) + [1] * nb + [-1] * ns,
                           dtype=torch.int32, device=device)
    new_si = torch.tensor([int(sink), int(local)] * H, dtype=torch.int32, device=device)

    # 4) base_blockmask（与块划分匹配）
    nrow = (max(q_lens_l) + BLOCK - 1) // BLOCK
    ncol = (max(k_lens_l) + BLOCK - 1) // BLOCK
    new_bm = _gen_base_blockmask(B, nb, nrow, ncol, float(sparsity), bool(is_causal), device)

    # 5) 前向 saved tensor
    new_smax, new_smin, new_attn = _forward_saved(
        q, k, v, new_cuq, new_cuk, new_hmt, new_si, new_bm,
        scale, bool(is_causal), bool(exact_streaming))

    return [q, k, v, new_cuq, new_cuk, new_hmt, new_si, new_bm, dout,
            new_smax, new_smin, new_attn]
