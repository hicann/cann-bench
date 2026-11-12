#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""KDA forward core: normalized queries/keys and fine-grained delta recurrence.

The equations follow Kimi Linear, arXiv:2510.26692v1, Eq. (1).
The benchmark contract deliberately fixes equal query/key/value head counts,
key-first states, preactivated log decay/beta, and always returns the final state.
This is an independently written reference, not a model layer or FLA wrapper.
"""

import math
from typing import Tuple

import numpy as np
import torch


def _check_inputs(q, k, v, g, beta, initial_state, use_qk_l2norm, scaleValue):
    """Check metadata only; finite/value-domain checks belong to case validation."""
    if q.ndim != 4 or min(q.shape) <= 0:
        raise ValueError("q must have nonempty shape [B,T,H,Dk]")
    batch, length, heads, key_dim = q.shape
    if k.shape != q.shape or g.shape != q.shape:
        raise ValueError("k and g must have the same shape as q")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3] or v.shape[3] <= 0:
        raise ValueError("v must have shape [B,T,H,Dv]")
    if beta.shape != (batch, length, heads):
        raise ValueError("beta must have shape [B,T,H]")
    if initial_state.shape != (batch, heads, key_dim, v.shape[3]):
        raise ValueError("initial_state must have shape [B,H,Dk,Dv]")
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if q.dtype not in allowed_dtypes or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k and v must share a supported floating-point dtype")
    if any(
        t.dtype not in (torch.float32, torch.float64) for t in (g, beta, initial_state)
    ):
        raise TypeError(
            "g, beta and initial_state must be FP32 (or FP64 for oracle checks)"
        )
    if any(t.device != q.device for t in (k, v, g, beta, initial_state)):
        raise ValueError("all tensor inputs must be on the same device")
    if not isinstance(use_qk_l2norm, bool):
        raise TypeError("use_qk_l2norm must be bool")
    if isinstance(scaleValue, bool) or not isinstance(scaleValue, (int, float)):
        raise TypeError("scaleValue must be a finite number")
    if not math.isfinite(scaleValue):
        raise ValueError("scaleValue must be finite")
    return key_dim**-0.5 if scaleValue <= 0 else float(scaleValue)


def kimi_delta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool = True,
    scaleValue: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (y [B,T,H,Dv], final_state [B,H,Dk,Dv]).

    g is per-token natural-log decay, not a cumulative log gate or raw logit.
    beta is an already activated update strength in [0,1]. Neither is activated
    again. Normalization uses sqrt(sum(x*x)+1e-6), and scale multiplies q only.
    scaleValue <= 0 selects 1/sqrt(Dk); a positive value selects an explicit scale.
    FP16/BF16/FP32 inputs accumulate in FP32. FP64 input is supported solely for
    developer precision checking and accumulates in FP64.
    """
    scale = _check_inputs(q, k, v, g, beta, initial_state, use_qk_l2norm, scaleValue)
    accumulation_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    query = q.to(accumulation_dtype)
    key = k.to(accumulation_dtype)
    value = v.to(accumulation_dtype)
    decay = g.to(accumulation_dtype).exp()
    update_strength = beta.to(accumulation_dtype)
    if use_qk_l2norm:
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * scale

    memory = initial_state.to(accumulation_dtype).clone()
    output = torch.empty(value.shape, dtype=accumulation_dtype, device=q.device)
    for token in range(q.shape[1]):
        faded_memory = memory * decay[:, token].unsqueeze(-1)
        token_key = key[:, token]
        predicted_value = torch.sum(faded_memory * token_key.unsqueeze(-1), dim=-2)
        correction = (value[:, token] - predicted_value) * update_strength[
            :, token
        ].unsqueeze(-1)
        memory = faded_memory + token_key.unsqueeze(-1) * correction.unsqueeze(-2)
        output[:, token] = torch.sum(memory * query[:, token].unsqueeze(-1), dim=-2)

    state_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    return output.to(q.dtype).contiguous(), memory.to(state_dtype).contiguous()


def kimi_delta_attention_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool = True,
    scaleValue: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent CPU/NumPy FP64 oracle, using value-first recurrent states.

    The evaluator's FP64 inputs and outputs stay FP64 throughout. This path
    uses independent contractions, a transposed state layout and a different
    array library. It is validation-only, not a performance baseline.
    Work is O(B*T*H*Dk*Dv), with no quadratic-in-sequence intermediate.
    """
    scale = _check_inputs(q, k, v, g, beta, initial_state, use_qk_l2norm, scaleValue)

    def as_array(tensor):
        return tensor.detach().to(device="cpu", dtype=torch.float64).numpy().copy()

    query, key, value = (as_array(tensor) for tensor in (q, k, v))
    log_decay, strength = as_array(g), as_array(beta)
    memory_vk = np.swapaxes(as_array(initial_state), -1, -2)
    if use_qk_l2norm:
        query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    output = np.empty(value.shape, dtype=np.float64)
    for token in range(q.shape[1]):
        kt = key[:, token]
        vt = value[:, token]
        memory_vk = memory_vk * np.exp(log_decay[:, token])[..., None, :]
        projected = np.squeeze(memory_vk @ kt[..., None], axis=-1)
        residual = (vt - projected) * strength[:, token, :, None]
        memory_vk = memory_vk + residual[..., :, None] @ kt[..., None, :]
        output[:, token] = np.squeeze(
            memory_vk @ (query[:, token] * scale)[..., None], axis=-1
        )

    y = torch.from_numpy(output.copy()).to(device=q.device, dtype=q.dtype)
    state_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    final_state = torch.from_numpy(np.swapaxes(memory_vk, -1, -2).copy()).to(
        device=q.device, dtype=state_dtype
    )
    return y, final_state


def kimi_delta_attention_bench(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool = True,
    scaleValue: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional placeholder only; not an optimized or NPU-validated baseline."""
    return kimi_delta_attention(
        q, k, v, g, beta, initial_state, use_qk_l2norm, scaleValue
    )
