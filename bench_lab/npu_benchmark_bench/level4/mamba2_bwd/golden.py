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
Mamba2Bwd 算子 Torch Golden 参考实现

Mamba-2 SSD chunk-scan 反向（逐 token 精确伴随）。前向（记 a_t = exp(A_h·dt'_t)）：

    S_t = a_t ⊙ S_{t-1} + dt'_t·x_t ⊗ B_t
    y_t = C_t^T·S_t (+ D·x_t)；z 门控 silu(z)·y_t

反向（dS 从 dfinal_states 初始化，逆序传播）：

    dy    = dout_t·silu(z_t)
    dz_t  = dout_t·y_t·silu'(z_t)
    dC_t  = S_t^T·dy ; dS += dy ⊗ C_t
    dD   += dy ⊙ x_t ; dx_t = D ⊙ dy
    dx_t += dt'_t·(dS·B_t)
    dB_t  = dt'_t·(dS^T·x_t)
    ddt'_t += Σ dS·(x_t⊗B_t)
    ddA_t = a_t·Σ(dS ⊙ S_{t-1})
    dA_h += ddA_t·dt'_t ; ddt'_t += ddA_t·A_h
    dS   <- a_t ⊙ dS

dt 预处理链反传（与 _chunk_cumsum_bwd 一致）：

    d(raw) = ddt'·[lo≤sp≤hi]·(σ(raw) 若 raw≤20 否则 1)
    ddt = d(raw) ; ddt_bias[h] = Σ_{b,t} d(raw)

前向中间量（S_all/y_all/a/dtr/raw/sp/sil/dsil）由输入预给，golden 只做反向递推。
GQA：dB/dC 组内各头求和恢复 [B,T,G,N]。全程 fp32 计算，输出按输入 dtype 回铸。
"""


def mamba2_bwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_states: torch.Tensor,
    dout: torch.Tensor,
    dfinal_states: torch.Tensor,
    S_all: torch.Tensor,
    y_all: torch.Tensor,
    a: torch.Tensor,
    dtr: torch.Tensor,
    raw: torch.Tensor,
    sp: torch.Tensor,
    sil: torch.Tensor,
    dsil: torch.Tensor,
    chunk_size: int = 64,
    dt_softplus: int = 1,
    dt_limit_lo: float = 0.0,
    dt_limit_hi: float = 1.0,
) -> tuple:
    """
    Mamba-2 SSD chunk-scan 反向（融合显式递推版）

    Args:
        x: 输入 [B, T, H, P]
        dt: 步长原始值 [B, T, H]
        A: 衰减参数 [H]
        B: SSM B [B, T, G, N]
        C: SSM C [B, T, G, N]
        D: 跳连参数 [H] 或 [B, H, P]；numel()==0 表示无
        z: 门控 silu 输入 [B, T, H, P]；shape[0]==0 表示无
        dt_bias: 步长偏置 [H]
        initial_states: 初始状态 [B, H, P, N]；shape[0]==0 表示无
        dout: 输出梯度 [B, T, H, P]
        dfinal_states: 末状态梯度 [B, H, P, N]；shape[0]==0 表示无
        S_all: 前向每步状态 [T, B, H, P, N] fp32
        y_all: 前向每步输出（含 D 跳连）[B, T, H, P] fp32
        a: 每步衰减 exp(dt'·A) [B, T, H] fp32
        dtr: clamp 后 dt' [B, T, H] fp32
        raw: dt + dt_bias [B, T, H] fp32
        sp: softplus(raw) 或 raw [B, T, H] fp32
        sil: silu(z) [B, T, H, P] fp32；shape[0]==0 表示无
        dsil: silu'(z) [B, T, H, P] fp32；shape[0]==0 表示无
        chunk_size: chunk 大小（不影响反向递推）
        dt_softplus: 1 表示 softplus 预处理
        dt_limit_lo: dt' 下界
        dt_limit_hi: dt' 上界

    Returns:
        (dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states)
    """
    device = x.device
    Bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    R = H // G

    # 内部统一 fp32（源实现口径）；兼容 evaluator 的 fp64 golden precision 与
    # bf16 原生精度两条路径。输出按原始输入 dtype 回铸。
    x_dtype, dt_dtype = x.dtype, dt.dtype
    B_dtype, C_dtype, z_dtype = B.dtype, C.dtype, z.dtype
    x = x.float()
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()
    z = z.float()
    dt_bias = dt_bias.float()
    initial_states = initial_states.float()
    dout = dout.float()
    dfinal_states = dfinal_states.float()
    S_all = S_all.float()
    y_all = y_all.float()
    a = a.float()
    dtr = dtr.float()
    raw = raw.float()
    sp = sp.float()
    sil = sil.float()
    dsil = dsil.float()

    xf = x.float()
    Bh = B.float().repeat_interleave(R, dim=2)   # [B,T,H,N]
    Ch = C.float().repeat_interleave(R, dim=2)
    has_z = z.shape[0] > 0
    has_init = initial_states.shape[0] > 0
    has_dD = D.numel() > 0
    Af = A.float()
    Df = D.float() if has_dD else None
    Dv = (Df.view(1, H, 1) if Df.dim() == 1
          else Df.view(1, H, P)) if has_dD else None

    S0 = (initial_states.float() if has_init
          else torch.zeros(Bsz, H, P, N, device=device))

    dof = dout.float()
    dxf = torch.zeros_like(xf)
    dBh = torch.zeros_like(Bh)
    dCh = torch.zeros_like(Ch)
    ddtp = torch.zeros_like(dtr)
    dAf = torch.zeros_like(Af)
    dDf = torch.zeros_like(Df) if has_dD else None
    dzf = (torch.zeros(Bsz, T, H, P, device=device)
           if has_z else None)
    dS = (dfinal_states.float().clone() if dfinal_states.shape[0] > 0
          else torch.zeros(Bsz, H, P, N, device=device))

    for t in reversed(range(T)):
        dy = dof[:, t] * sil[:, t] if has_z else dof[:, t]
        if has_z:
            dzf[:, t] = dof[:, t] * y_all[:, t] * dsil[:, t]
        S_t = S_all[t]
        dCh[:, t] = (S_t * dy.unsqueeze(-1)).sum(-2)          # [B,H,N]
        dS = dS + dy.unsqueeze(-1) * Ch[:, t].unsqueeze(-2)   # dy ⊗ C_t
        if has_dD:
            acc = dy * xf[:, t]                               # [B,H,P]
            dDf += acc.sum((0, 2)) if Df.dim() == 1 else acc.sum(0)
            dxf[:, t] = Dv * dy
        # S_t = a⊙S_{t-1} + dt'·x⊗B
        dxf[:, t] += dtr[:, t].unsqueeze(-1) * (dS * Bh[:, t].unsqueeze(-2)).sum(-1)
        dBh[:, t] = dtr[:, t].unsqueeze(-1) * (dS * xf[:, t].unsqueeze(-1)).sum(-2)
        ddtp[:, t] += (dS * (xf[:, t].unsqueeze(-1)
                             * Bh[:, t].unsqueeze(-2))).sum((-1, -2))
        S_prev = S_all[t - 1] if t > 0 else S0
        ddA = a[:, t] * (dS * S_prev).sum((-1, -2))           # [B,H]
        dAf += (ddA * dtr[:, t]).sum(0)
        ddtp[:, t] += ddA * Af
        dS = a[:, t].unsqueeze(-1).unsqueeze(-1) * dS
    dinitial = dS

    # dt 预处理链反传
    unclamped = (sp >= dt_limit_lo) & (sp <= dt_limit_hi)
    draw = ddtp * unclamped
    if dt_softplus:
        draw = draw * torch.where(raw <= 20.0, torch.sigmoid(raw),
                                  torch.ones_like(raw))
    ddt_bias = draw.sum((0, 1))                               # [H]

    # GQA：组内各头梯度求和，恢复 [B,T,G,N]
    dB = dBh.view(Bsz, T, G, R, N).sum(3).to(B_dtype)
    dC = dCh.view(Bsz, T, G, R, N).sum(3).to(C_dtype)

    # 空 tensor 也携带 proto 声明的输出 dtype（dz 与 z 同 dtype），
    # 否则 evaluator 的输出 dtype 契约会把 float32 空 tensor 判负。
    dz_out = dzf.to(z_dtype) if has_z else torch.empty(0, dtype=z_dtype, device=device)
    dinit_out = dinitial if has_init else torch.empty(0, H, P, N, device=device)
    return (dxf.to(x_dtype), draw.to(dt_dtype), dAf, dB, dC,
            dDf if has_dD else torch.empty(0, device=device),
            dz_out, ddt_bias, dinit_out)


def _forward_intermediates(x, dt, A, B, C, D, z, dt_bias, initial_states,
                           dt_softplus, dt_limit_lo, dt_limit_hi):
    """按源前向重算 saved 中间量（全部 fp32）。

    返回 (S_all [T,B,H,P,N], y_all [B,T,H,P], a [B,T,H], dtr [B,T,H],
    raw [B,T,H], sp [B,T,H], sil [B,T,H,P] 或 [0], dsil 同上)。
    """
    device = x.device
    Bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    R = H // G

    xf = x.float()
    Bh = B.float().repeat_interleave(R, dim=2)
    Ch = C.float().repeat_interleave(R, dim=2)
    has_z = z.shape[0] > 0
    has_init = initial_states.shape[0] > 0
    has_dD = D.numel() > 0

    raw = dt.float() + dt_bias.float()
    sp = F.softplus(raw) if dt_softplus else raw
    dtr = sp.clamp(dt_limit_lo, dt_limit_hi)
    a = (dtr * A.float()).exp()
    Dv = None
    if has_dD:
        Df = D.float()
        Dv = Df.view(1, H, 1) if Df.dim() == 1 else Df.view(1, H, P)

    S = (initial_states.float().clone() if has_init
         else torch.zeros(Bsz, H, P, N, device=device))
    S_all = torch.empty(T, Bsz, H, P, N, device=device)
    y_all = torch.empty(Bsz, T, H, P, device=device)
    for t in range(T):
        S = (a[:, t].unsqueeze(-1).unsqueeze(-1) * S
             + (dtr[:, t].unsqueeze(-1).unsqueeze(-1)
                * xf[:, t].unsqueeze(-1)) * Bh[:, t].unsqueeze(-2))
        S_all[t] = S
        y_all[:, t] = (S * Ch[:, t].unsqueeze(-2)).sum(-1)
    if has_dD:
        y_all = y_all + xf * Dv.unsqueeze(0)

    if has_z:
        zf = z.float()
        sig = torch.sigmoid(zf)
        sil = zf * sig
        dsil = sig * (1.0 + zf * (1.0 - sig))
    else:
        sil = torch.empty(0, device=device)
        dsil = torch.empty(0, device=device)

    return S_all, y_all, a, dtr, raw, sp, sil, dsil


def get_input(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_states: torch.Tensor,
    dout: torch.Tensor,
    dfinal_states: torch.Tensor,
    S_all: torch.Tensor,
    y_all: torch.Tensor,
    a: torch.Tensor,
    dtr: torch.Tensor,
    raw: torch.Tensor,
    sp: torch.Tensor,
    sil: torch.Tensor,
    dsil: torch.Tensor,
    chunk_size: int = 64,
    dt_softplus: int = 1,
    dt_limit_lo: float = 0.0,
    dt_limit_hi: float = 1.0,
    **kwargs,
) -> list:
    """由叶子输入重算前向中间量（S_all/y_all/a/dtr/raw/sp/sil/dsil）。

    这些张量是 x/dt/A/B/C/D/z/dt_bias 的确定性函数，单区间 value_range 无法表达；
    这里用 fp32 完整重放源前向递推后返回，保证反向递推使用的前向量自洽。
    leaf 输入（x/dt/A/B/C/D/z/dt_bias/initial_states/dout/dfinal_states）原样保留。

    Returns:
        19 个张量，顺序与 golden 签名一致。
    """
    new_S_all, new_y_all, new_a, new_dtr, new_raw, new_sp, new_sil, new_dsil = \
        _forward_intermediates(x, dt, A, B, C, D, z, dt_bias, initial_states,
                               bool(dt_softplus), float(dt_limit_lo), float(dt_limit_hi))
    return [x, dt, A, B, C, D, z, dt_bias, initial_states, dout, dfinal_states,
            new_S_all, new_y_all, new_a, new_dtr, new_raw, new_sp, new_sil, new_dsil]
