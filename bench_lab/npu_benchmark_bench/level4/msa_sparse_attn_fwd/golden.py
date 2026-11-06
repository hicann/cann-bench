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
MsaSparseAttnFwd 算子 Torch Golden 参考实现

MiniMax-M3 block-sparse attention forward（prefill）。k2q CSR 逆推得到每个 query
选中的 KV 块，再在选中块展开的 token 上（叠加 seqused 截断与 bottom-right 因果）
做 masked softmax 注意力。

公式:
    S(g, b, i) = 由 k2q CSR 还原的块集合
    scores[t] = softmax_scale * q[b, i, h] . k[b, t, g],  t in S 且 t < seqused_k[b]
               且 t <= i + (seqused_k[b] - qlen_b)
    out[b, i, h] = softmax(scores) @ v[b, :, g]
    无可见 token 的行输出 0
"""


def msa_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    topK: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int,
    causal: int,
    softmax_scale: float,
) -> torch.Tensor:
    """
    MSA 块稀疏注意力前向

    Args:
        q: varlen query [total_q, Hq, D], bf16/fp16
        k: paged KV cache [num_pages, Hkv, blk_kv, D]
        v: paged KV cache [num_pages, Hkv, blk_kv, D]
        k2q_row_ptr: k2q CSR 行指针 [Hkv, total_rows + 1] int32
        k2q_q_indices: k2q CSR query 索引 [Hkv, total_q * topK] int32，尾部 -1 填充
        cu_seqlens_q: q 前缀和 [B+1] int32
        cu_seqlens_k: kv 前缀和 [B+1] int32
        seqused_k: 每请求有效 kv 长度 [B] int32
        page_table: 页表 [B, max_pages] int32
        topK: 每 query 选中的 KV 块数
        max_seqlen_q: 最长 query 长度（形状提示）
        max_seqlen_k: 最长 kv 长度（形状提示）
        blk_kv: 稀疏块大小（等于页大小）
        causal: 是否启用 bottom-right 因果掩码（0/1）
        softmax_scale: softmax 缩放因子

    Returns:
        注意力输出 [total_q, Hq, D] bf16
    """
    device = q.device
    total_q, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    B = cu_seqlens_q.numel() - 1
    cu_q = cu_seqlens_q.long()
    topK = int(topK)
    blk_kv = int(blk_kv)
    num_pages = k.shape[0]

    meta = torch.cat([cu_seqlens_q, cu_seqlens_k, seqused_k]).tolist()
    cuq = [int(x) for x in meta[:B + 1]]
    cuk = [int(x) for x in meta[B + 1:2 * B + 2]]
    seqused = [int(x) for x in meta[2 * B + 2:]]

    # ---- 1. CSR 全局行 -> (batch, 块号) 逆映射（块号优先、批次次之） ----
    kv_lens = [max(0, cuk[b + 1] - cuk[b]) for b in range(B)]
    rows_pb = [(L + blk_kv - 1) // blk_kv for L in kv_lens]
    total_rows = k2q_row_ptr.shape[1] - 1
    max_rows = max(rows_pb) if B > 0 else 0
    row_batch_l, row_block_l = [], []
    for j in range(max_rows):
        for b in range(B):
            if j < rows_pb[b]:
                row_batch_l.append(b)
                row_block_l.append(j)
    if len(row_batch_l) != total_rows:
        # 仅对不满足 CSR 契约的输入生效，合法输入下是 no-op
        while len(row_batch_l) < total_rows:
            row_batch_l.append(0)
            row_block_l.append(0)
        row_batch_l = row_batch_l[:total_rows]
        row_block_l = row_block_l[:total_rows]

    q2k = torch.full((Hkv, total_q, topK), -1, dtype=torch.long, device=device)
    if total_rows > 0:
        row_batch = torch.tensor(row_batch_l, dtype=torch.long, device=device)
        row_block = torch.tensor(row_block_l, dtype=torch.long, device=device)

        # ---- 2. k2q CSR -> q2k [Hkv, total_q, topK]（集合语义还原） ----
        counts = (k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]).long()
        cum = counts.cumsum(dim=1)
        nnz_total = cum[:, -1:]
        T = k2q_q_indices.shape[1]
        pos = torch.arange(T, device=device, dtype=torch.long)
        pos_h = pos.unsqueeze(0).expand(Hkv, T).contiguous()
        rows_h = torch.searchsorted(cum, pos_h, right=True)
        valid = pos_h < nnz_total
        rows_c = rows_h.clamp(min=0, max=total_rows - 1)
        qg = cu_q[row_batch[rows_c]] + k2q_q_indices.long()
        qg = qg.clamp(min=0, max=max(0, total_q - 1))
        jb = row_block[rows_c]
        key = torch.where(valid, qg.float(),
                          torch.full_like(qg, float("inf"), dtype=torch.float32))
        order = torch.argsort(key, dim=1, stable=True)
        qg_s = qg.gather(1, order)
        jb_s = jb.gather(1, order)
        valid_s = valid.gather(1, order)
        is_start = torch.ones(Hkv, T, dtype=torch.bool, device=device)
        is_start[:, 1:] = qg_s[:, 1:] != qg_s[:, :-1]
        first_pos = torch.where(is_start, pos_h, pos_h - pos_h - 1).cummax(dim=1).values
        rank = pos_h - first_pos
        keep = valid_s & (rank < topK)
        h_idx = torch.arange(Hkv, device=device, dtype=torch.long).unsqueeze(1).expand(Hkv, T)
        flat = (h_idx * (total_q * topK) + qg_s * topK + rank)
        flat = flat.clamp(min=0, max=max(0, q2k.numel() - 1))
        q2k.view(-1)[flat[keep]] = jb_s[keep]

    # ---- 3. 块稀疏注意力（fp32 内部计算，按 query 分块控制显存） ----
    out = torch.zeros(total_q, Hq, D, dtype=torch.float32, device=device)
    arange_blk = torch.arange(blk_kv, device=device)
    S_all = topK * blk_kv
    chunk = max(1, (1 << 27) // max(1, Hkv * S_all * D * 4))

    for b in range(B):
        q0 = max(0, min(total_q, int(cuq[b])))
        q1 = max(0, min(total_q, int(cuq[b + 1])))
        qlen = q1 - q0
        if qlen <= 0:
            continue
        L = int(seqused[b]) if b < len(seqused) else 0
        L = min(L, page_table.shape[1] * blk_kv)
        if L <= 0:
            continue
        nblocks = (L + blk_kv - 1) // blk_kv
        pages = page_table[b, :nblocks].long().clamp(min=0, max=num_pages - 1)
        k_bT = k[pages].reshape(nblocks * blk_kv, Hkv, D)[:L].float().permute(1, 0, 2)
        v_bT = v[pages].reshape(nblocks * blk_kv, Hkv, D)[:L].float().permute(1, 0, 2)
        q_pos = torch.arange(qlen, device=device) + (L - qlen)   # bottom-right 对齐
        sel = q2k[:, q0:q1]                                      # [Hkv, qlen, topK]
        q_b = q[q0:q1].float().view(qlen, Hkv, G, D) * softmax_scale

        for c0 in range(0, qlen, chunk):
            c1 = min(c0 + chunk, qlen)
            sel_c = sel[:, c0:c1]
            cq = c1 - c0
            tok4 = sel_c[..., None] * blk_kv + arange_blk          # [Hkv, cq, topK, blk]
            tok_idx = tok4.reshape(Hkv, cq, S_all)
            tok_valid = ((sel_c[..., None] >= 0) & (sel_c[..., None] < nblocks)
                         & (tok4 < L))
            if causal:
                tok_valid &= tok4 <= q_pos[c0:c1].view(1, cq, 1, 1)
            tok_valid = tok_valid.reshape(Hkv, cq, S_all)

            idx = tok_idx.clamp(0, L - 1)
            gidx = idx.reshape(Hkv, -1, 1).expand(-1, -1, D)
            k_g = k_bT.gather(1, gidx).reshape(Hkv, cq, S_all, D)
            v_g = v_bT.gather(1, gidx).reshape(Hkv, cq, S_all, D)
            scores = torch.einsum("qhgd,hqsd->hqgs", q_b[c0:c1], k_g)
            scores = scores.masked_fill(~tok_valid.unsqueeze(2), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)               # 无可见 token 的行输出 0
            out[q0 + c0:q0 + c1] = torch.einsum(
                "hqgs,hqsd->qhgd", attn, v_g).reshape(cq, Hq, D)

    return out.to(torch.bfloat16)


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    topK: int = 8,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 1,
    blk_kv: int = 128,
    causal: int = 1,
    softmax_scale: float = 1.0,
    **kwargs,
):
    """重建满足本算子契约的 6 个结构化输入（同时替换 golden 与候选输入，比较公平）。

    cu_seqlens_q/k、seqused_k、k2q CSR 与 page_table 之间存在严格的整数关系，
    逐元素均匀随机生成无法满足。这里按实际 shape 重新构造：
      - total_q 均分到 B 条请求；
      - k2q_row_ptr 的 total_rows 均分到 B 条请求，seqused_k = 块数 * blk_kv；
      - 每个 query 从其 bottom-right 因果可见块中取前 min(visible, topK) 个；
      - CSR 按“块号优先、批次次之”的行序、行内按 q 升序打包，尾部 -1 填充；
      - page_table 用固定种子打乱后的页号填充。
    q/k/v 原样保留。

    Returns:
        [q, k, v, k2q_row_ptr, k2q_q_indices, cu_seqlens_q, cu_seqlens_k, seqused_k, page_table]
    """
    total_q, Hq, D = q.shape
    num_pages, Hkv, blk, _ = k.shape
    B = int(cu_seqlens_q.numel()) - 1
    topK = int(topK)
    blk_kv = int(blk_kv)
    total_rows = int(k2q_row_ptr.shape[1]) - 1
    max_pages = int(page_table.shape[1])

    # 请求长度与块数均分
    q_base, q_rem = divmod(total_q, max(B, 1))
    qlens = [q_base + (1 if b < q_rem else 0) for b in range(B)]
    r_base, r_rem = divmod(total_rows, max(B, 1))
    nblk = [r_base + (1 if b < r_rem else 0) for b in range(B)]
    nblk = [min(nb, max_pages) for nb in nblk]
    if sum(nblk) != total_rows:
        nblk = [min(max_pages, total_rows // max(B, 1))] * B
        nblk[0] += total_rows - sum(nblk)

    cuq = [0]
    for length in qlens:
        cuq.append(cuq[-1] + length)
    seqused = [nb * blk_kv for nb in nblk]
    cuk = [0]
    for L in seqused:
        cuk.append(cuk[-1] + L)

    # 每个 query 选其 bottom-right 因果可见块中的前若干块（升序）
    q2k = torch.full((Hkv, total_q, topK), -1, dtype=torch.long)
    for b in range(B):
        q0 = cuq[b]
        qlen = qlens[b]
        L = seqused[b]
        nbb = nblk[b]
        for h in range(Hkv):
            for qi in range(qlen):
                visible = min((qi + L - qlen) // blk_kv + 1, nbb)
                if visible <= 0:
                    continue
                take = min(visible, topK)
                q2k[h, q0 + qi, :take] = torch.arange(take, dtype=torch.long)

    # 行线性化：块号优先、批次次之
    row_of = {}
    r = 0
    for j in range(max(nblk) if nblk else 0):
        for b in range(B):
            if j < nblk[b]:
                row_of[(b, j)] = r
                r += 1

    row_ptr = torch.zeros(Hkv, total_rows + 1, dtype=torch.int32)
    qidx = torch.full((Hkv, total_q * topK), -1, dtype=torch.int32)
    for h in range(Hkv):
        rows_flat, q_flat = [], []
        for b in range(B):
            q0, q1 = cuq[b], cuq[b + 1]
            sel = q2k[h, q0:q1]
            qlen = q1 - q0
            rm = torch.tensor([row_of[(b, j)] for j in range(nblk[b])], dtype=torch.long)
            valid = sel >= 0
            rows_flat.append(rm[sel.clamp(min=0)][valid])
            q_flat.append(torch.arange(qlen).unsqueeze(1).expand(qlen, topK)[valid])
        rows_cat = torch.cat(rows_flat) if rows_flat else torch.zeros(0, dtype=torch.long)
        q_cat = torch.cat(q_flat) if q_flat else torch.zeros(0, dtype=torch.long)
        counts = torch.zeros(total_rows, dtype=torch.long)
        counts.scatter_add_(0, rows_cat, torch.ones_like(rows_cat))
        row_ptr[h, 1:] = counts.cumsum(0).int()
        order = torch.argsort(rows_cat * (total_q + 1) + q_cat, stable=True)
        qidx[h, :rows_cat.numel()] = q_cat[order].int()

    # 页表：固定种子打乱页号，按 batch 依次分配
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(num_pages, generator=g)
    page_tab = torch.zeros(B, max_pages, dtype=torch.int32)
    cursor = 0
    for b in range(B):
        nb = nblk[b]
        page_tab[b, :nb] = perm[cursor:cursor + nb].int()
        cursor += nb

    new_cu_q = torch.tensor(cuq, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    new_cu_k = torch.tensor(cuk, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device)
    new_seq = torch.tensor(seqused, dtype=seqused_k.dtype, device=seqused_k.device)
    return [q, k, v, row_ptr, qidx, new_cu_q, new_cu_k, new_seq,
            page_tab.to(device=page_table.device)]
