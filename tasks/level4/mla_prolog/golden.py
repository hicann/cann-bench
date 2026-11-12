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
MlaProlog算子Torch Golden参考实现

Multi-Head Latent Attention前处理，融合 Query/Key 投影、RMSNorm 和 RoPE
"""


def rms_norm(x, gamma, epsilon):
    """
    RMSNorm: gamma * x / sqrt(mean(x^2) + epsilon).
    """
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f**2, dim=-1, keepdim=True) + epsilon)
    return (gamma.float() * x_f / rms).to(x.dtype)


def apply_rope(x, rope_cos, rope_sin):
    """
    Apply RoPE with pre-indexed sin/cos.
    """
    cos = rope_cos.float()
    sin = rope_sin.float()
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rotated * sin).to(x.dtype)


def mla_prolog(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-Head Latent Attention 前处理

    Args:
        token_x: [B, S, He] 输入 hidden states, bfloat16
        w_dq: [He, Hcq] query 下投影权重, bfloat16
        w_uq_qr: [Hcq, N*(D+Dr)] query 上投影+RoPE 权重(合并), bfloat16
        w_uk: [N, D, Hckv] key 上投影权重(吸收到 query 侧), bfloat16
        w_dkv_kr: [He, Hckv+Dr] KV 下投影+Key RoPE 权重(合并), bfloat16
        rmsnorm_gamma_cq: [Hcq] c_q 的 RMSNorm gamma, bfloat16
        rmsnorm_gamma_ckv: [Hckv] c_kv 的 RMSNorm gamma, bfloat16
        rope_sin: [B, S, Dr] RoPE 正弦(已按位置索引), bfloat16
        rope_cos: [B, S, Dr] RoPE 余弦(已按位置索引), bfloat16
        n_heads: 注意力头数 N
        rmsnorm_epsilon_cq: c_q RMSNorm epsilon
        rmsnorm_epsilon_ckv: c_kv RMSNorm epsilon

    Returns:
        query: [B, S, N, Hckv] 吸收 W_UK 后的 query, bfloat16
        query_rope: [B, S, N, Dr] query 位置编码, bfloat16
        c_kv: [B, S, Hckv] 归一化后的压缩 KV, bfloat16
        k_rope: [B, S, Dr] key 位置编码, bfloat16
    """
    original_dtype = token_x.dtype
    _low_prec = original_dtype in (torch.float16, torch.bfloat16)
    if _low_prec:
        token_x = token_x.float()
        w_dq = w_dq.float()
        w_uq_qr = w_uq_qr.float()
        w_uk = w_uk.float()
        w_dkv_kr = w_dkv_kr.float()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.float()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.float()
        rope_sin = rope_sin.float()
        rope_cos = rope_cos.float()

    B, S, He = token_x.shape
    N = n_heads
    Hckv = w_uk.shape[2]
    D = w_uk.shape[1]
    Dr = rope_sin.shape[-1]

    # === Query Path ===
    c_q_raw = torch.matmul(token_x, w_dq)
    c_q = rms_norm(c_q_raw, rmsnorm_gamma_cq, rmsnorm_epsilon_cq)
    qr = torch.matmul(c_q, w_uq_qr)
    qr = qr.reshape(B, S, N, D + Dr)
    q_c = qr[..., :D]
    q_r_raw = qr[..., D:]
    query = torch.einsum("bsnd,ndh->bsnh", q_c, w_uk)
    cos_exp = rope_cos.unsqueeze(2).expand(-1, -1, N, -1)
    sin_exp = rope_sin.unsqueeze(2).expand(-1, -1, N, -1)
    query_rope = apply_rope(q_r_raw, cos_exp, sin_exp)

    # === Key Path ===
    dkv_kr = torch.matmul(token_x, w_dkv_kr)
    ckv_raw = dkv_kr[..., :Hckv]
    kr_raw = dkv_kr[..., Hckv:]
    c_kv = rms_norm(ckv_raw, rmsnorm_gamma_ckv, rmsnorm_epsilon_ckv)
    k_rope = apply_rope(kr_raw, rope_cos, rope_sin)

    if _low_prec:
        query = query.to(original_dtype)
        query_rope = query_rope.to(original_dtype)
        c_kv = c_kv.to(original_dtype)
        k_rope = k_rope.to(original_dtype)

    # einsum / slice 可能留下 view, 而输出契约要求 contiguous (issue #146)；
    # 已连续时 .contiguous() 原样返回, 无额外拷贝
    return (
        query.contiguous(),
        query_rope.contiguous(),
        c_kv.contiguous(),
        k_rope.contiguous(),
    )


def get_input(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int = 1,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    **kwargs,
):
    """把 rope_sin / rope_cos 重建为同一角度的正弦与余弦，满足 sin^2 + cos^2 = 1。

    与 level2/apply_rotary_pos_emb 同一问题：RoPE 的 sin / cos 是同一组位置角 theta 的
    正弦与余弦（proto 也写明"已按位置索引"），对应保范数的平面旋转。通用生成器把它们
    当成两个互相独立的随机张量，(cos, sin) 落在正方形而非单位圆上，不对应任何位置角。

    后果同样有两层：利用该恒等式实现的候选（由 cos 还原 sin、融合复数乘）在真实 RoPE 上
    正确却会与 golden 分叉；且精度阈值按保范数旋转标定，独立随机下失去参照。

    **与 level2 的关键差异：本算子的 rope_cos / rope_sin 是全宽 Dr，不是半宽。**
    level2 的 cos 形状为 (S, D/2)，其 golden 内部会 `cos.repeat(1, 1, 1, 2)` 补齐到全宽，
    因此在输入张量上逐元素取角即可。而本算子的 apply_rope（见本文件 L33-42）拿到的就是
    全宽 Dr，直接逐元素相乘、**没有内部 repeat**：

        x1, x2 = xf.chunk(2, dim=-1);  rotated = cat([-x2, x1])
        y = xf * cos + rotated * sin

    展开后（h = Dr/2、j in [0, h)）：

        y[j]   = x[j]   * cos[j]   - x[j+h] * sin[j]
        y[j+h] = x[j+h] * cos[j+h] + x[j]   * sin[j+h]

    这对 (x[j], x[j+h]) 构成保范数旋转，**当且仅当** cos[j] == cos[j+h] 且
    sin[j] == sin[j+h] —— 即全宽张量的前后半必须是同一角度的复制，正是 RoPE
    `emb = cat([freqs, freqs])` 的约定（desc §7 内嵌的 apply_rope 与此一致）。

    所以这里在**半宽 (..., Dr/2) 上取角**，再复制拼接到全宽；若像 level2 那样在全宽上
    逐元素独立取角，虽然 sin^2+cos^2 = 1 处处成立，整体却不是旋转（实测配对模长比在
    0.023 ~ 1.996 之间），反而与本 PR 的目标背道而驰。

    t 取固定种子的 U(-pi, pi)。用随机角而非真实频率结构，是为了让数据不可被提交侧预测
    写死；kernel 对 sin / cos 只做逐元素乘加，频率结构不影响其算术路径。
    其余 7 个输入原样保留。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
         rope_sin, rope_cos]，顺序与 mla_prolog 签名一致。
    """
    unchanged = [
        token_x,
        w_dq,
        w_uq_qr,
        w_uk,
        w_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
    ]
    if not isinstance(rope_sin, torch.Tensor) or not isinstance(rope_cos, torch.Tensor):
        return unchanged
    if rope_sin.shape != rope_cos.shape:
        return unchanged

    # 特殊值压力用例原样放行：含 NaN/Inf 或恒为常数（c20 value_range=[0,0]）时，本就
    # 不是"某个位置角的正弦/余弦"，其用意是特殊值与零值边界覆盖，不应被重建抹掉。
    if not bool(torch.isfinite(rope_sin).all()) or not bool(
        torch.isfinite(rope_cos).all()
    ):
        return unchanged
    if bool((rope_sin == rope_sin.reshape(-1)[0]).all()) and bool(
        (rope_cos == rope_cos.reshape(-1)[0]).all()
    ):
        return unchanged

    dr = int(rope_sin.shape[-1])
    if dr < 2 or dr % 2 != 0:
        # apply_rope 的 chunk(2, dim=-1) 本就要求 Dr 为偶数；奇数时配对约定不成立，
        # 不臆测，保持原样
        return unchanged

    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    half_shape = list(rope_sin.shape[:-1]) + [dr // 2]
    theta = (
        torch.rand(half_shape, generator=g, dtype=torch.float64) * 2 - 1
    ) * torch.pi
    # 前后半复制同一角度：apply_rope 无内部 repeat，靠输入自身满足配对约定
    sin_full = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    cos_full = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    new_sin = sin_full.to(dtype=rope_sin.dtype, device=rope_sin.device)
    new_cos = cos_full.to(dtype=rope_cos.dtype, device=rope_cos.device)
    return [
        token_x,
        w_dq,
        w_uq_qr,
        w_uk,
        w_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        new_sin,
        new_cos,
    ]
