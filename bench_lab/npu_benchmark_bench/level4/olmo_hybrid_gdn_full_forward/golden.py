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
OlmoHybridGdnFullForward 算子 Torch Golden 参考实现

OLMo 混合 Gated Delta Net（GDN）全前向：QKV/门控投影、因果 depthwise 卷积、
门控 delta rule 分块递推、per-head 门控 RMSNorm 与输出投影。内部 fp32 计算，
输出回铸输入 dtype。

公式:
    qkv_gate = X W_qkvg^T,  b = X W_b^T,  a = X W_a^T
    mixed = SiLU(causal_depthwise_conv1d(qkv_gate))
    g = -exp(A_log) * softplus(a + dt_bias),  beta = sigmoid(b) * 2
    chunkwise gated delta rule over (q, k, v, g, beta) -> o
    o = RMSNorm(o) * SiLU(gate) ,  y = o W_o
"""

_LN2 = math.log(2.0)
_CHUNK = 64


def _rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    dims = tuple(range(-len(normalized_shape), 0))
    variance = x.pow(2).mean(dims, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight
    return out


def _silu(x):
    return x * torch.sigmoid(x)


def _chunk_all(qf, kf, vf, gf, bf):
    """整段序列分块递推（每头标量门控）。"""
    batch, seq_len, heads, key_dim = qf.shape
    val_dim = vf.shape[-1]
    dev = qf.device
    chunk = _CHUNK
    pad_len = ((seq_len + chunk - 1) // chunk) * chunk
    pad = pad_len - seq_len
    qp = torch.nn.functional.pad(qf.transpose(1, 2), (0, 0, 0, pad))
    kp = torch.nn.functional.pad(kf.transpose(1, 2), (0, 0, 0, pad))
    vp = torch.nn.functional.pad(vf.transpose(1, 2), (0, 0, 0, pad))
    gp = torch.nn.functional.pad(gf.transpose(1, 2), (0, pad))
    bp = torch.nn.functional.pad(bf.transpose(1, 2), (0, pad))

    state = torch.zeros(batch, heads, key_dim, val_dim, device=dev, dtype=torch.float32)
    out = torch.zeros(batch, heads, pad_len, val_dim, device=dev, dtype=torch.float32)
    eye = torch.eye(chunk, device=dev, dtype=torch.float32).expand(batch, heads, chunk, chunk).clone()
    num_chunks = pad_len // chunk
    for c in range(num_chunks):
        qc = qp[:, :, c * chunk:(c + 1) * chunk]
        kc = kp[:, :, c * chunk:(c + 1) * chunk]
        vc = vp[:, :, c * chunk:(c + 1) * chunk]
        gc = gp[:, :, c * chunk:(c + 1) * chunk]
        bc = bp[:, :, c * chunk:(c + 1) * chunk]

        gk = gc.cumsum(2) / _LN2
        gl = torch.exp2(gk[:, :, -1:])
        decay = torch.exp2(torch.clamp(gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))

        qg = qc * torch.exp2(gk).unsqueeze(-1)
        kg = kc * torch.exp2(gk[:, :, -1:] - gk).unsqueeze(-1)

        aqk = torch.tril(torch.einsum(
            'bhik,bhjk,bhijk->bhij', qc, kc, decay.unsqueeze(-1)), diagonal=0)
        lower = torch.tril(torch.einsum(
            'bhik,bhjk,bhijk->bhij', kc, kc, decay.unsqueeze(-1))
            * bc.unsqueeze(-1), diagonal=-1)
        a_plus_l = eye + lower

        w = torch.linalg.solve_triangular(
            a_plus_l, bc.unsqueeze(-1) * kc * torch.exp2(gk).unsqueeze(-1),
            upper=False, unitriangular=True)
        u = torch.linalg.solve_triangular(
            a_plus_l, bc.unsqueeze(-1) * vc, upper=False, unitriangular=True)

        v_new = u - torch.einsum('bhik,bhkv->bhiv', w, state)
        out[:, :, c * chunk:(c + 1) * chunk] = torch.einsum('bhik,bhkv->bhiv', qg, state) \
            + torch.einsum('bhij,bhjv->bhiv', aqk, v_new)
        state = gl.unsqueeze(-1) * state + torch.einsum('bhik,bhiv->bhkv', kg, v_new)

    return out[:, :, :seq_len].transpose(1, 2).contiguous(), state


def olmo_hybrid_gdn_full_forward(
    hidden_states: torch.Tensor,
    output_buffer: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    w_conv1d: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    w_o_norm: torch.Tensor,
    w_o_proj: torch.Tensor,
    num_heads_k: int,
    num_heads_v: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
) -> torch.Tensor:
    """
    OLMo 混合 GDN 全前向

    公式:
        qkv_gate = X W_qkvg^T,  b = X W_b^T,  a = X W_a^T
        mixed = SiLU(causal_depthwise_conv1d(qkv_gate))
        g = -exp(A_log) * softplus(a + dt_bias),  beta = sigmoid(b) * 2
        chunkwise gated delta rule over (q, k, v, g, beta) -> o
        o = RMSNorm(o) * SiLU(gate) ,  y = o W_o

    Args:
        hidden_states: 输入隐藏状态 [T, C]
        output_buffer: 输出缓冲区 [T, C]
        w_qkvg: QKV 与门控融合投影权重 [2*nk*hk + 2*nv*hv, C]
        w_b: 擦除门投影权重 [nv, C]
        w_a: 衰减投影权重 [nv, C]
        w_conv1d: 因果 depthwise 卷积权重 [2*nk*hk + nv*hv, conv_kernel_size]
        dt_bias: 时间步偏置 [nv]
        A_log: 衰减参数对数值 [nv]
        w_o_norm: 输出 RMSNorm 权重 [hv]
        w_o_proj: 输出投影权重 [C, nv*hv]
        num_heads_k: key/query 头数
        num_heads_v: value 头数
        head_k_dim: key/query 头维度
        head_v_dim: value 头维度
        conv_kernel_size: 因果卷积核长度

    Returns:
        全前向输出 [T, C]
    """
    num_tokens = hidden_states.shape[0]
    dtype = hidden_states.dtype

    a_log = A_log.clamp(-3.0, 3.0)
    dt_bias = dt_bias.clamp(-3.0, 3.0)

    key_dim = num_heads_k * head_k_dim
    value_dim = num_heads_v * head_v_dim
    conv_dim = key_dim * 2 + value_dim

    # QKV 与门控投影（提升后的显式权重）
    x = hidden_states.float()
    projected = torch.nn.functional.linear(x, w_qkvg.float())
    proj_b = torch.nn.functional.linear(x, w_b.float())
    proj_a = torch.nn.functional.linear(x, w_a.float())

    mixed_qkv = projected[:, :conv_dim]
    gate = projected[:, conv_dim:]

    # 因果 depthwise conv1d + SiLU
    mixed_t = mixed_qkv.unsqueeze(0).transpose(1, 2)
    conv_w = w_conv1d.float().reshape(conv_dim, 1, conv_kernel_size)
    mixed_padded = torch.nn.functional.pad(mixed_t, (conv_kernel_size - 1, 0))
    mixed_conv = torch.nn.functional.conv1d(mixed_padded, conv_w, groups=conv_dim)
    mixed_conv = mixed_conv.squeeze(0).transpose(0, 1)
    mixed_conv = _silu(mixed_conv)

    query = mixed_conv[:, :key_dim].reshape(num_tokens, num_heads_k, head_k_dim)
    key = mixed_conv[:, key_dim:2 * key_dim].reshape(num_tokens, num_heads_k, head_k_dim)
    value = mixed_conv[:, 2 * key_dim:].reshape(num_tokens, num_heads_v, head_v_dim)

    # GQA 扩展
    if num_heads_v > num_heads_k:
        ratio = num_heads_v // num_heads_k
        query = query.unsqueeze(2).expand(-1, -1, ratio, -1).reshape(num_tokens, num_heads_v, head_k_dim)
        key = key.unsqueeze(2).expand(-1, -1, ratio, -1).reshape(num_tokens, num_heads_v, head_k_dim)

    # 门控（fp32）
    g = -a_log.float().exp().unsqueeze(0) * torch.nn.functional.softplus(
        proj_a.float() + dt_bias.unsqueeze(0))
    beta = torch.sigmoid(proj_b.float()) * 2.0

    q_f = query.float().unsqueeze(0)
    k_f = key.float().unsqueeze(0)
    v_f = value.float().unsqueeze(0)
    g_f = g.float().unsqueeze(0)
    b_f = beta.float().unsqueeze(0)
    scale = 1.0 / math.sqrt(head_k_dim)

    k_f = k_f / (torch.norm(k_f, dim=-1, keepdim=True) + 1e-6)
    q_f = q_f / (torch.norm(q_f, dim=-1, keepdim=True) + 1e-6) * scale

    core_out, _ = _chunk_all(q_f, k_f, v_f, g_f, b_f)
    core_out = core_out[0].to(dtype)

    # per-head 门控 RMSNorm + SiLU
    gate = gate.reshape(num_tokens, num_heads_v, head_v_dim).clamp(-10.0, 10.0)
    core_flat = core_out.reshape(-1, head_v_dim)
    gate_flat = gate.reshape(-1, head_v_dim)
    normed = _rms_norm(core_flat.float(), (head_v_dim,), weight=w_o_norm.float())
    core_normed = normed * gate_flat.float() * torch.sigmoid(gate_flat.float())

    core_reshaped = core_normed.reshape(num_tokens, value_dim)
    output = torch.nn.functional.linear(core_reshaped, w_o_proj.float())
    return output.to(dtype)
