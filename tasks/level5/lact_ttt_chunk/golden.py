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
from typing import Tuple

"""
LactTttChunk 算子 Torch Golden 参考实现

Large-Chunk Test-Time Training（LaCT）层的前向：快权重是逐样本、逐头的 SwiGLU MLP
f_W(x) = W2 [SiLU(W1 x) ⊙ (W3 x)]，序列按 chunk_size 切块，每块用块内所有 token 的
(k̂, v, lr) 对快权重做一步梯度更新（每个 (b, h) 独立）:
    q̂, k̂ = q, k 沿 D 做 L2 归一化（eps 1e-6）
    块 c 的损失 ℒ_c = Σ_{i∈I_c} lr_i · (−f_W(k̂_i)ᵀ v_i)，g = ∇_W ℒ_c（W1/W2/W3 三个矩阵）
    Δ = Muon(g)（use_muon=True，Newton–Schulz 5 步 zeropower）或 g（use_muon=False）
    W ← RowL2Normalize(W − Δ)（对每个输出神经元的权重向量做 L2 归一，eps 1e-6）
    update_first=False：块内先 y_t = f_{W^{(c)}}(q̂_t) 再更新得 W^{(c+1)}（因果，apply-then-update）
    update_first=True ：先更新再 apply（块内可见未来）
输出 y = f_W(q̂) 逐 token 拼接，以及最终快权重 w1_out / w2_out / w3_out。
输入约定: lr > 0（由 value_range 保证）；末块可残缺（按实际长度处理）。
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
lact_ttt_chunk_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""

# Newton–Schulz 5 步 zeropower 的多项式系数（Keller Jordan Muon）
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315
_NS_STEPS = 5


def _row_l2_normalize(w):
    """对每个输出神经元的权重向量（最后一维）做 L2 归一化，eps 加在范数上。"""
    return w / (w.norm(dim=-1, keepdim=True) + 1e-6)


def _muon_zeropower(g):
    """Muon：Newton–Schulz 5 步 zeropower（对最后两维的每个矩阵独立），返回近似正交化的 g。"""
    x = g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)  # 保证 rows <= cols，A = X Xᵀ 更小
    for _ in range(_NS_STEPS):
        a_mat = x @ x.transpose(-2, -1)
        b_mat = _NS_B * a_mat + _NS_C * (a_mat @ a_mat)
        x = _NS_A * x + b_mat @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x


def _lact_ttt_chunk_core(
    q, k, v, w1, w2, w3, lr, chunk_size, use_muon, update_first, compute_dtype
):
    """核心计算：以 compute_dtype 精度执行块循环（apply / update），返回 (y, w1, w2, w3)。"""
    Bsz, L, H, D = q.shape

    q_f = q.to(compute_dtype)
    k_f = k.to(compute_dtype)
    # 沿 D 做 L2 归一化后转为 [B, H, L, D]，便于逐 (b, h) 做 batched matmul
    q_hat = (q_f / (q_f.norm(dim=-1, keepdim=True) + 1e-6)).permute(0, 2, 1, 3)
    k_hat = (k_f / (k_f.norm(dim=-1, keepdim=True) + 1e-6)).permute(0, 2, 1, 3)
    v_bh = v.to(compute_dtype).permute(0, 2, 1, 3)  # [B, H, L, D]
    lr_bh = lr.to(compute_dtype).permute(0, 2, 1).unsqueeze(-1)  # [B, H, L, 1]
    W1 = w1.to(compute_dtype).clone()  # [B, H, Dh, D]
    W2 = w2.to(compute_dtype).clone()  # [B, H, D, Dh]
    W3 = w3.to(compute_dtype).clone()  # [B, H, Dh, D]

    y = torch.empty(Bsz, H, L, D, dtype=compute_dtype, device=q.device)

    def _apply(s, e):
        # y_t = W2 [SiLU(W1 q̂_t) ⊙ (W3 q̂_t)]，t ∈ [s, e)
        x = q_hat[:, :, s:e]  # [B, H, Lc, D]
        h1 = x @ W1.transpose(-2, -1)  # [B, H, Lc, Dh]
        h3 = x @ W3.transpose(-2, -1)
        u = torch.nn.functional.silu(h1) * h3
        y[:, :, s:e] = u @ W2.transpose(-2, -1)  # [B, H, Lc, D]

    def _update(s, e):
        nonlocal W1, W2, W3
        x = k_hat[:, :, s:e]  # [B, H, Lc, D]
        # 前向：h1 = W1 k̂, h3 = W3 k̂, u = SiLU(h1) ⊙ h3, out = W2 u
        h1 = x @ W1.transpose(-2, -1)  # [B, H, Lc, Dh]
        h3 = x @ W3.transpose(-2, -1)
        sig = torch.sigmoid(h1)
        s1 = h1 * sig  # SiLU(h1)
        u = s1 * h3
        # ℒ = Σ_i lr_i · (−outᵢᵀ vᵢ) 的闭式梯度：∂out = −lr ⊙ v
        d_out = -lr_bh[:, :, s:e] * v_bh[:, :, s:e]  # [B, H, Lc, D]
        d_w2 = d_out.transpose(-2, -1) @ u  # dW2 = Σ ∂out uᵀ   [B, H, D, Dh]
        d_u = d_out @ W2  # du = W2ᵀ ∂out     [B, H, Lc, Dh]
        d_h3 = d_u * s1  # dh3 = du ⊙ SiLU(h1)
        d_h1 = d_u * h3 * (sig * (1.0 + h1 * (1.0 - sig)))  # dh1 = du ⊙ h3 ⊙ SiLU'(h1)
        d_w1 = d_h1.transpose(-2, -1) @ x  # dW1 = Σ dh1 k̂ᵀ    [B, H, Dh, D]
        d_w3 = d_h3.transpose(-2, -1) @ x  # dW3 = Σ dh3 k̂ᵀ    [B, H, Dh, D]
        if use_muon:
            d_w1 = _muon_zeropower(d_w1)
            d_w2 = _muon_zeropower(d_w2)
            d_w3 = _muon_zeropower(d_w3)
        # W ← RowL2Normalize(W − Δ)
        W1 = _row_l2_normalize(W1 - d_w1)
        W2 = _row_l2_normalize(W2 - d_w2)
        W3 = _row_l2_normalize(W3 - d_w3)

    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)  # 末块可残缺
        if update_first:
            _update(s, e)
            _apply(s, e)
        else:
            _apply(s, e)
            _update(s, e)

    # 评测框架要求输出 contiguous，permute 后需实体化
    return y.permute(0, 2, 1, 3).contiguous(), W1, W2, W3


def lact_ttt_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    lr: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    update_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LaCT 大块 TTT golden reference（块循环 + 闭式梯度；plain golden = bench：fp32 计算）

    Args:
        q: [B, L, H, D] 查询，算子内部沿 D 做 L2 归一化
        k: [B, L, H, D] 键，算子内部沿 D 做 L2 归一化
        v: [B, L, H, D] 值（快权重更新的回归目标）
        w1: [B, H, Dh, D] 快权重 W1（SwiGLU 门分支）
        w2: [B, H, D, Dh] 快权重 W2（输出投影）
        w3: [B, H, Dh, D] 快权重 W3（SwiGLU 线性分支）
        lr: [B, L, H] 逐 token 逐头学习率，恒正（评测取值范围 [0.001, 0.05]）
        chunk_size: 块大小 C，序列按 C 切块、末块可残缺
        use_muon: True 时 Δ = Muon(g)（Newton–Schulz 5 步 zeropower），False 时 Δ = g
        update_first: False 为 apply-then-update（因果），True 为 update-then-apply（块内可见未来）

    Returns:
        y: [B, L, H, D] 输出序列，dtype 与 q 一致
        w1_out / w2_out / w3_out: 序列末尾的快权重，shape 与输入同、dtype 与 q 一致
    """
    y, w1_o, w2_o, w3_o = _lact_ttt_chunk_core(
        q,
        k,
        v,
        w1,
        w2,
        w3,
        lr,
        chunk_size,
        use_muon,
        update_first,
        torch.float64 if q.dtype == torch.float64 else torch.float32,
    )
    return y.to(q.dtype), w1_o.to(q.dtype), w2_o.to(q.dtype), w3_o.to(q.dtype)


def lact_ttt_chunk_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    lr: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    update_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _lact_ttt_chunk_core(
        q, k, v, w1, w2, w3, lr, chunk_size, use_muon, update_first, q.dtype
    )


def lact_ttt_chunk_bench(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    lr: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    update_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 working math for quantized inputs; same output dtype as the API."""
    return lact_ttt_chunk(q, k, v, w1, w2, w3, lr, chunk_size, use_muon, update_first)
