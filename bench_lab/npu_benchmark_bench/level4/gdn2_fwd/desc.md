# Gdn2Fwd 算子 API 描述

## 1. 算子简介

GDN-2（Gated DeltaNet 2）的 chunkwise 前向。逐 token 递推包含逐通道衰减、擦除门读取衰减后状态、写入门 delta 规则与秩一状态更新；实现上以 64 为块长改写为块内矩阵运算加块间状态传播，输出每步读出 `o` 与末状态 `final_state`。

**主要应用场景**：
- 线性注意力 / 状态空间模型的长序列前向
- Gated DeltaNet 系列模型的训练与推理

**算子特征**：
- 难度等级：L4（LinearAttention）
- 输入：`q/k/g/b` 为 `[B, T, H, K]`，`v/w` 为 `[B, T, H, V]`，`initial_state` 为 `[N, H, K, V]`
- 输出：`o` 为 `[B, T, H, V]`，`final_state` 为 `[N, H, K, V]`（float32）
- golden 全程 fp32 计算，`o` 回铸 `v` 的 dtype
- 无内部随机权重，所有输入均为显式 tensor

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
S &\leftarrow S \odot \exp(g_t) \\
erase &= (b_t \odot k_t)^\top S \\
v\_new &= w_t \odot v_t - erase \\
S &\leftarrow S + k_t \otimes v\_new \\
o_t &= (scale \cdot q_t)^\top S
\end{aligned}
$$

其中 `S \in \mathbb{R}^{K 	imes V}`，`g_t` 为逐 K 通道的对数域衰减（恒为负），`b_t` 为擦除门、`w_t` 为写入门（取值均在 (0, 1)）。分块实现以 `C = 64` 为块长，块内用 `cumsum(g)/ln2` 构造衰减矩阵并做三角求解，块间传播状态，与逐 token 递推数学等价。

## 3. 接口规范

### 算子原型

```python
cann_bench.gdn2_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor b, Tensor w, Tensor initial_state, float scale, int use_qk_l2norm_in_kernel, int[] cu_seqlens=[]) -> (Tensor o, Tensor final_state)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | query `[B, T, H, K]` |
| k | Tensor | 必选 | key `[B, T, H, K]` |
| v | Tensor | 必选 | value `[B, T, H, V]` |
| g | Tensor | 必选 | 对数域逐通道衰减 `[B, T, H, K]`，float32，恒为负 |
| b | Tensor | 必选 | 擦除门 `[B, T, H, K]`，取值 (0, 1) |
| w | Tensor | 必选 | 写入门 `[B, T, H, V]`，取值 (0, 1) |
| initial_state | Tensor | 必选 | 初始状态 `[N, H, K, V]`，float32，`shape[0] == 0` 表示无 |
| scale | float | 必选 | 输出缩放系数，通常为 `1/sqrt(K)` |
| use_qk_l2norm_in_kernel | int | 必选 | 是否在核内对 q/k 做 L2 归一化（1 启用，0 关闭） |
| cu_seqlens | List[int] | [] | varlen 前缀和，长度 `N+1`；为空表示等长 `[B, T]` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| o | `[B, T, H, V]` | 与 `v` 相同 | 每步输出 |
| final_state | `[N, H, K, V]` | float32 | 末状态 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16（o）/ float32（final_state） |
| float32 | float32 / float32 |
| bfloat16 | bfloat16（o）/ float32（final_state） |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `q`、`k`、`g`、`b` 形状为 `[B, T, H, K]`；`v`、`w` 形状为 `[B, T, H, V]`
- `g`、`initial_state` 的 dtype 固定为 float32
- `initial_state.shape[0]` 等于 varlen 序列数 `N` 或 0（无初始状态）
- varlen 模式下要求 `B == 1`，且 `cu_seqlens[-1] == T`
- `g` 恒为负；`b`、`w` 取值在 (0, 1) 内
- `use_qk_l2norm_in_kernel=0` 时调用方需先对 q/k 做 L2 归一化（`get_input` 会自动完成）
- 计算内部统一使用 fp32，`o` 回铸为 `v` 的 dtype，`final_state` 保持 fp32

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

import math

import torch

"""
Gdn2Fwd 算子 Torch Golden 参考实现

GDN-2（Gated DeltaNet 2）chunkwise 前向。逐 token 递推（S ∈ R^{K×V}，* 为逐通道
Hadamard 积）:
    S <- S * exp(g_t)
    erase = (b_t * k_t)^T S
    v_new = w_t * v_t - erase
    S <- S + k_t ⊗ v_new
    o_t = (scale * q_t)^T S
改写为「块内矩阵 + 块间状态传播」的分块递推（C = 64），与逐步递推数学等价。
全程 fp32，输出 o 转回 v.dtype，final_state 保持 fp32。

公式:
    S <- S * exp(g_t)                       # K 轴逐通道衰减
    erase = (b_t * k_t)^T S
    v_new = w_t * v_t - erase
    S <- S + k_t ⊗ v_new
    o_t = (scale * q_t)^T S
"""

_LN2 = math.log(2.0)
_CHUNK = 64


def _chunk_segment(qf, kf, vf, gf, bf, wf, state0, scale):
    """整段序列分块递推，返回 (o([B,T,H,V] fp32), S([B,H,K,V] fp32))。"""
    batch, seq_len, heads, key_dim = qf.shape
    val_dim = vf.shape[-1]
    dev = qf.device
    chunk = _CHUNK
    pad_len = ((seq_len + chunk - 1) // chunk) * chunk
    pad = pad_len - seq_len
    qp = torch.nn.functional.pad(qf.transpose(1, 2), (0, 0, 0, pad))
    kp = torch.nn.functional.pad(kf.transpose(1, 2), (0, 0, 0, pad))
    gp = torch.nn.functional.pad(gf.transpose(1, 2), (0, 0, 0, pad))
    bp = torch.nn.functional.pad(bf.transpose(1, 2), (0, 0, 0, pad))
    vp = torch.nn.functional.pad(vf.transpose(1, 2), (0, 0, 0, pad))
    wp = torch.nn.functional.pad(wf.transpose(1, 2), (0, 0, 0, pad))

    state = state0.contiguous()
    out = torch.zeros(batch, heads, pad_len, val_dim, device=dev, dtype=torch.float32)
    eye = torch.eye(chunk, device=dev, dtype=torch.float32)
    num_chunks = pad_len // chunk
    for c in range(num_chunks):
        qc = qp[:, :, c * chunk:(c + 1) * chunk]
        kc = kp[:, :, c * chunk:(c + 1) * chunk]
        gc = gp[:, :, c * chunk:(c + 1) * chunk]
        bc = bp[:, :, c * chunk:(c + 1) * chunk]
        vc = vp[:, :, c * chunk:(c + 1) * chunk]
        wc = wp[:, :, c * chunk:(c + 1) * chunk]

        gk = gc.cumsum(2) / _LN2
        kr = bc * kc
        qs = qc * scale

        decay = torch.exp2(torch.clamp(gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))

        aqk = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', qs, kc, decay), diagonal=0)
        lower = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', kr, kc, decay), diagonal=-1)
        a_plus_l = eye + lower

        w_sol = torch.linalg.solve_triangular(
            a_plus_l, kr * torch.exp2(gk), upper=False, unitriangular=True)
        u_sol = torch.linalg.solve_triangular(
            a_plus_l, wc * vc, upper=False, unitriangular=True)

        v_new = u_sol - torch.einsum('bhik,bhkv->bhiv', w_sol, state)

        qg = qs * torch.exp2(gk)
        out[:, :, c * chunk:(c + 1) * chunk] = torch.einsum('bhik,bhkv->bhiv', qg, state) \
            + torch.einsum('bhij,bhjv->bhiv', aqk, v_new)
        kg = kc * torch.exp2(gk[:, :, -1:, :] - gk)
        state = torch.exp2(gk[:, :, -1:, :]).transpose(-1, -2) * state \
            + torch.einsum('bhik,bhiv->bhkv', kg, v_new)

    return out[:, :, :seq_len].transpose(1, 2).contiguous(), state


def gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: int,
    cu_seqlens=None,
) -> tuple:
    """
    GDN-2 chunkwise 前向

    公式:
        S <- S * exp(g_t)                       # K 轴逐通道衰减
        erase = (b_t * k_t)^T S
        v_new = w_t * v_t - erase
        S <- S + k_t ⊗ v_new
        o_t = (scale * q_t)^T S

    Args:
        q: query [B, T, H, K]
        k: key [B, T, H, K]
        v: value [B, T, H, V]
        g: 对数域逐通道衰减 [B, T, H, K]
        b: 擦除门 [B, T, H, K]
        w: 写入门 [B, T, H, V]
        initial_state: 初始状态 [N, H, K, V]，shape[0] == 0 表示无
        scale: 输出缩放系数
        use_qk_l2norm_in_kernel: 是否在核内对 q/k 做 L2 归一化
        cu_seqlens: varlen 前缀和，空列表表示等长 [B, T]

    Returns:
        (o, final_state)，o 的 shape 为 [B, T, H, V]，final_state 为 [N, H, K, V]
    """
    device = q.device
    batch, seq_len, heads, _ = q.shape
    qf, kf = q.float(), k.float()
    if int(use_qk_l2norm_in_kernel):
        qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    vf, gf, bf, wf = v.float(), g.float(), b.float(), w.float()

    has_init = initial_state is not None and initial_state.shape[0] > 0
    if cu_seqlens is not None and len(cu_seqlens) > 0:
        if isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = cu_seqlens.tolist()
        num_seq = len(cu_seqlens) - 1
        out = torch.empty(1, seq_len, heads, v.shape[-1], dtype=torch.float32, device=device)
        final = torch.empty(num_seq, heads, q.shape[-1], v.shape[-1], dtype=torch.float32, device=device)
        for n in range(num_seq):
            start, end = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
            state0 = (initial_state[n:n + 1].float() if has_init
                      else torch.zeros(1, heads, q.shape[-1], v.shape[-1], device=device))
            out_n, state_n = _chunk_segment(
                qf[:, start:end], kf[:, start:end], vf[:, start:end],
                gf[:, start:end], bf[:, start:end], wf[:, start:end], state0, scale)
            out[:, start:end] = out_n
            final[n] = state_n[0]
    else:
        state0 = initial_state.float() if has_init else torch.zeros(
            batch, heads, q.shape[-1], v.shape[-1], device=device)
        out, final = _chunk_segment(qf, kf, vf, gf, bf, wf, state0, scale)

    return out.to(v.dtype), final


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: int,
    cu_seqlens=None,
    **kwargs,
) -> list:
    """保证输入满足 GDN-2 良定义约束：g <= 0、b/w 在 (0, 1)、关闭核内归一化时 q/k 预先归一化。"""
    if not int(use_qk_l2norm_in_kernel):
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    g = torch.clamp(g.float(), max=-1e-3).to(g.dtype)
    b = torch.clamp(b.float(), min=1e-4, max=1.0 - 1e-4).to(b.dtype)
    w = torch.clamp(w.float(), min=1e-4, max=1.0 - 1e-4).to(w.dtype)
    return [q, k, v, g, b, w, initial_state]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, T, H, K, V = 1, 26, 16, 37, 46
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="npu")
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="npu")
v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="npu")
g = -torch.rand(B, T, H, K, dtype=torch.float32, device="npu") * 5.0
b = torch.rand(B, T, H, K, dtype=torch.bfloat16, device="npu")
w = torch.rand(B, T, H, V, dtype=torch.bfloat16, device="npu")
initial_state = torch.zeros(0, H, K, V, dtype=torch.float32, device="npu")
o, final_state = cann_bench.gdn2_fwd(q, k, v, g, b, w, initial_state, 0.164399, 1, [0, 6, 13, 26])
# o.shape == (1, 26, 16, 46), final_state.shape == (3, 16, 37, 46)
```
