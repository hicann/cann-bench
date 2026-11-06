# Gdn2Bwd 算子 API 描述

## 1. 算子简介

Gdn2Bwd 是 GDN-2（Gated DeltaNet 2）线性注意力的反向算子（显式对偶递推）。前向按 token
递推维护状态 $S\in\mathbb{R}^{K\times V}$，并导出每步衰减状态 `S_dec_stack` 与新写入向量
`v_new_stack`；反向从末状态梯度 `dht` 初始化 $dS$，逆序逐 token 传播伴随状态，产出
`dq / dk / dv / dg / db / dw / dh0`。

**主要应用场景**：
- Gated DeltaNet 系列线性注意力模型的训练反向
- chunkwise / 逐 token 精确伴随的算子级性能基线

**算子特征**：
- 难度等级：L4（LinearAttention）
- 11 个张量输入（含前向中间量 `S_dec_stack` / `v_new_stack`）+ 3 个属性
- 7 个张量输出；全程 fp32 累加，梯度回铸输入 dtype，`dh0` 恒 fp32
- 支持等长 `[B, T]` 与 varlen（`cu_seqlens`，此时 `B == 1`）两种布局
- q/k 支持核内 L2 归一化，并在输出时经链式法则传回梯度

## 2. 算子定义

### 数学公式

前向（逐 token，$S\in\mathbb{R}^{K\times V}$）：

$$
\begin{aligned}
S^{dec}_t &= \mathrm{Diag}(\exp(g_t))\,S_{t-1}\\
e_t &= (b_t\cdot k_t)^\top S^{dec}_t\\
v^{new}_t &= w_t\cdot v_t - e_t\\
S_t &= S^{dec}_t + k_t\otimes v^{new}_t\\
o_t &= (\mathrm{scale}\cdot q_t)^\top S_t
\end{aligned}
$$

反向（$dS$ 从 `dht` 初始化，逆序传播）：

$$
\begin{aligned}
dq_t &= \mathrm{scale}\cdot (S_t\,do_t)\\
dS &\mathrel{+}= (\mathrm{scale}\cdot q_t)\otimes do_t\\
dk_t &= dS\,v^{new}_t,\qquad dv^{new}_t = dS^\top k_t\\
dw_t &= dv^{new}_t\cdot v_t,\qquad dv_t = dv^{new}_t\cdot w_t\\
d e_t &= -dv^{new}_t,\qquad dbk_t = S^{dec}_t\,d e_t\\
dS^{dec} &= dS + (b_t\cdot k_t)\otimes d e_t\\
db_t &= dbk_t\cdot k_t,\qquad dk_t \mathrel{+}= dbk_t\cdot b_t\\
dg_t &= \sum_V \left(dS^{dec}\cdot S^{dec}_t\right)\\
dS &\leftarrow dS^{dec}\cdot \exp(g_t)
\end{aligned}
$$

`dh0` 为递推结束时的 $dS$。当 `use_qk_l2norm_in_kernel == 1` 时，先对 q/k 做 L2 归一化
$q_n = q / \sqrt{\sum q^2 + 10^{-6}}$ 再进入递推，反向输出时按归一化的链式法则变换 `dq/dk`。

## 3. 接口规范

### 算子原型

```python
cann_bench.gdn2_bwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor b, Tensor w,
                    Tensor initial_state, Tensor do, Tensor dht,
                    Tensor S_dec_stack, Tensor v_new_stack,
                    float scale, int use_qk_l2norm_in_kernel, int[] cu_seqlens=[])
    -> (Tensor dq, Tensor dk, Tensor dv, Tensor dg, Tensor db, Tensor dw, Tensor dh0)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | query `[B, T, H, K]`，bf16 |
| k | Tensor | 必选 | key `[B, T, H, K]`，bf16 |
| v | Tensor | 必选 | value `[B, T, H, V]`，bf16 |
| g | Tensor | 必选 | 对数衰减 `[B, T, H, K]`，fp32，取负值 |
| b | Tensor | 必选 | 擦除门 `[B, T, H, K]`，bf16 |
| w | Tensor | 必选 | 写入门 `[B, T, H, V]`，bf16 |
| initial_state | Tensor | 必选 | 初始状态 `[N, H, K, V]`，fp32；`shape[0] == 0` 表示无 |
| do | Tensor | 必选 | 输出梯度 `[B, T, H, V]`，bf16 |
| dht | Tensor | 必选 | 末状态梯度 `[N, H, K, V]`，fp32；`shape[0] == 0` 表示无 |
| S_dec_stack | Tensor | 必选 | 前向每步衰减状态 `[T, B, H, K, V]`，fp32 |
| v_new_stack | Tensor | 必选 | 前向每步新写入向量 `[T, B, H, V]`，fp32 |
| scale | float | 必选 | 输出缩放系数 |
| use_qk_l2norm_in_kernel | int | 必选 | 是否在核内对 q/k 做 L2 归一化（1/0） |
| cu_seqlens | List[int] | `[]` | varlen 前缀和，长度 `N+1`；为空表示等长 `[B, T]` |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| dq | `[B, T, H, K]` | 与 `q` 相同 | query 梯度 |
| dk | `[B, T, H, K]` | 与 `k` 相同 | key 梯度 |
| dv | `[B, T, H, V]` | 与 `v` 相同 | value 梯度 |
| dg | `[B, T, H, K]` | 与 `g` 相同（fp32） | 对数衰减梯度 |
| db | `[B, T, H, K]` | 与 `b` 相同 | 擦除门梯度 |
| dw | `[B, T, H, V]` | 与 `w` 相同 | 写入门梯度 |
| dh0 | `[N, H, K, V]` | float32 | 初始状态梯度 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| bfloat16（q、k、v、b、w、do、S_dec_stack/v_new_stack 为 fp32），float32（g、initial_state、dht） | dq/dk/dv/db/dw 为 bfloat16，dg/dh0 为 float32 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- `q/k/g/b` 形状一致，`v/w/do` 形状一致；`v.shape[-1] == w.shape[-1] == do.shape[-1]`
- `S_dec_stack.shape == [T, B, H, K, V]`，`v_new_stack.shape == [T, B, H, V]`
- 等长布局下 `N == B`；varlen 布局下 `B == 1` 且 `N == len(cu_seqlens) - 1`
- `initial_state` / `dht` 的首维为 `N`；首维为 0 时等价于无初始状态/无末状态梯度
- `S_dec_stack` / `v_new_stack` 为前向确定性导出的中间量；`get_input` 会由
  `q/k/v/g/b/w/initial_state` 以 fp32 完整重放前向递推得到，与反向使用值一致
- 全程 fp32 累加；输出回铸输入 dtype（`dh0` 恒 fp32）

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

"""
Gdn2Bwd 算子 Torch Golden 参考实现

GDN-2（Gated DeltaNet 2）反向（显式对偶递推）。使用前向预计算并传入的每步衰减
状态 S_dec_stack 与新写入向量 v_new_stack，从 dht 初始化 dS 逆序逐 token 传播。

前向（逐 token，S ∈ R^{K×V}）:
    S_dec = Diag(exp(g_t)) · S_{t-1}
    erase = (b_t · k_t)^T · S_dec
    v_new = w_t · v_t − erase
    S_t   = S_dec + k_t ⊗ v_new
    o_t   = (scale · q_t)^T · S_t
反向（dS 从 dht 初始化，逆序传播）:
    dqs_t = S_t @ do_t ; dq_t = scale · dqs_t
    dS    += (scale · q_t) ⊗ do_t
    dk_t   = dS @ v_new_t ; dv_new = dS^T @ k_t
    dw_t   = dv_new · v_t ; dv_t = dv_new · w_t
    derase = −dv_new ; dbk = S_dec_t @ derase
    dS_dec = dS + (b_t · k_t) ⊗ derase
    db_t   = dbk · k_t ; dk_t += dbk · b_t
    dg_t   = Σ_V(dS_dec · S_dec_t)
    dS    <- dS_dec · exp(g_t)
    dh0    = 递推结束时的 dS
q/k 的 L2 归一化在进入递推前施加，并在输出时经链式法则传回 q/k 梯度。
"""


def _segment_bwd(qn, kn, vf, gf, bf, wf, dof,
                 S_dec_stack, v_new_stack, s, e, dS0, scale):
    """对 token 区间 [s, e) 执行反向递推（dS 从 dS0 初始化）。"""
    dq = torch.zeros_like(qn)
    dk = torch.zeros_like(kn)
    dv = torch.zeros_like(vf)
    dg = torch.zeros_like(gf)
    db = torch.zeros_like(bf)
    dw = torch.zeros_like(wf)
    dS = dS0
    for t in range(e - 1, s - 1, -1):
        S_dec = S_dec_stack[t]
        v_new = v_new_stack[t]
        rel = t - s

        # S_t = S_dec + k_t ⊗ v_new
        S_t = S_dec + kn[:, rel].unsqueeze(-1) * v_new.unsqueeze(-2)
        # 输出路径
        dq[:, rel] = scale * (S_t * dof[:, rel].unsqueeze(-2)).sum(-1)
        dS = dS + (scale * qn[:, rel]).unsqueeze(-1) * dof[:, rel].unsqueeze(-2)
        # 更新路径
        dk_t = (dS * v_new.unsqueeze(-2)).sum(-1)
        dv_new = (dS * kn[:, rel].unsqueeze(-1)).sum(-2)
        dw[:, rel] = dv_new * vf[:, rel]
        dv[:, rel] = dv_new * wf[:, rel]
        # 擦除门读取路径
        derase = -dv_new
        dbk = (S_dec * derase.unsqueeze(-2)).sum(-1)
        dS_dec = dS + (bf[:, rel] * kn[:, rel]).unsqueeze(-1) * derase.unsqueeze(-2)
        db[:, rel] = dbk * kn[:, rel]
        dk_t = dk_t + dbk * bf[:, rel]
        # 衰减路径
        dg[:, rel] = (dS_dec * S_dec).sum(-1)
        dS = dS_dec * gf[:, rel].unsqueeze(-1).exp()
        dk[:, rel] = dk_t
    return dq, dk, dv, dg, db, dw, dS


def gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    S_dec_stack: torch.Tensor,
    v_new_stack: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: int = 1,
    cu_seqlens=None,
) -> tuple:
    """
    GDN-2 反向（融合显式递推版）

    Args:
        q: query [B, T, H, K]
        k: key [B, T, H, K]
        v: value [B, T, H, V]
        g: 对数衰减 [B, T, H, K]（fp32）
        b: 擦除门 [B, T, H, K]
        w: 写入门 [B, T, H, V]
        initial_state: 初始状态 [N, H, K, V]（fp32），shape[0] == 0 表示无
        do: 输出梯度 [B, T, H, V]
        dht: 末状态梯度 [N, H, K, V]（fp32），shape[0] == 0 表示无
        S_dec_stack: 每步衰减状态 [T, B, H, K, V]（fp32）
        v_new_stack: 每步新写入向量 [T, B, H, V]（fp32）
        scale: 输出缩放系数
        use_qk_l2norm_in_kernel: 是否在核内对 q/k 做 L2 归一化（0/1）
        cu_seqlens: varlen 前缀和，None/空表示等长 [B, T]

    Returns:
        (dq, dk, dv, dg, db, dw, dh0)，前六个与对应输入同 dtype，dh0 恒 fp32
    """
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = cu_seqlens.tolist()
    B, T, H, K = q.shape
    V = v.shape[-1]

    qf, kf, vf = q.float(), k.float(), v.float()
    gf, bf, wf = g.float(), b.float(), w.float()
    dof = do.float()

    if use_qk_l2norm_in_kernel:
        q_rstd = torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        k_rstd = torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        qn = qf * q_rstd
        kn = kf * k_rstd
    else:
        qn, kn = qf, kf
        q_rstd = k_rstd = None

    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    dg = torch.zeros_like(gf)
    db = torch.zeros_like(bf)
    dw = torch.zeros_like(wf)

    has_init = initial_state is not None and initial_state.shape[0] > 0
    has_dht = dht is not None and dht.shape[0] > 0

    if cu_seqlens is not None and len(cu_seqlens) > 0:
        # varlen：B 必须为 1，按 segment 独立反向
        N = len(cu_seqlens) - 1
        dh0 = torch.zeros(N, H, K, V, dtype=torch.float32, device=q.device)
        for n in range(N):
            s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
            dS0 = dht[n:n + 1].float() if has_dht else \
                torch.zeros(1, H, K, V, device=q.device)
            r = _segment_bwd(
                qn[:, s:e], kn[:, s:e], vf[:, s:e],
                gf[:, s:e], bf[:, s:e], wf[:, s:e],
                dof[:, s:e], S_dec_stack, v_new_stack,
                s, e, dS0, scale)
            dq[:, s:e], dk[:, s:e], dv[:, s:e] = r[0], r[1], r[2]
            dg[:, s:e], db[:, s:e], dw[:, s:e] = r[3], r[4], r[5]
            dh0[n] = r[6][0]
    else:
        dS0 = dht.float() if has_dht else \
            torch.zeros(B, H, K, V, device=q.device)
        dq, dk, dv, dg, db, dw, dh0 = _segment_bwd(
            qn, kn, vf, gf, bf, wf, dof,
            S_dec_stack, v_new_stack, 0, T, dS0, scale)

    if use_qk_l2norm_in_kernel:
        dq = q_rstd * (dq - qn * (dq * qn).sum(-1, keepdim=True))
        dk = k_rstd * (dk - kn * (dk * kn).sum(-1, keepdim=True))

    return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
            dg.to(g.dtype), db.to(b.dtype), dw.to(w.dtype), dh0)


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    S_dec_stack: torch.Tensor,
    v_new_stack: torch.Tensor,
    scale: float = 1.0,
    use_qk_l2norm_in_kernel: int = 1,
    cu_seqlens=None,
    **kwargs,
):
    """由叶子输入重算前向中间量 S_dec_stack / v_new_stack。

    这两个张量对应 kernel 前向导出的逐步中间状态，是 q/k/v/g/b/w/initial_state
    的确定性函数，单区间 value_range 无法表达。这里用 fp32 完整重放前向递推后
    返回，保证与 golden 反向使用的中间量一致。其余输入原样保留。

    Returns:
        [q, k, v, g, b, w, initial_state, do, dht, S_dec_stack, v_new_stack]
    """
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = cu_seqlens.tolist()
    B, T, H, K = q.shape
    V = v.shape[-1]
    device = q.device

    qf, kf, vf = q.float(), k.float(), v.float()
    gf, bf, wf = g.float(), b.float(), w.float()
    if use_qk_l2norm_in_kernel:
        q_rstd = torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        k_rstd = torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        qn = qf * q_rstd
        kn = kf * k_rstd
    else:
        qn, kn = qf, kf

    has_init = initial_state is not None and initial_state.shape[0] > 0
    S_dec_all = []
    v_new_all = []
    if cu_seqlens is not None and len(cu_seqlens) > 0:
        N = len(cu_seqlens) - 1
        for n in range(N):
            s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
            S = qf.new_zeros(1, H, K, V)
            if has_init:
                S = S + initial_state[n:n + 1].float()
            for t in range(s, e):
                S_dec = S * gf[:, t].unsqueeze(-1).exp()
                erase = ((bf[:, t].unsqueeze(-1) * kn[:, t].unsqueeze(-1)) * S_dec).sum(-2)
                v_new = wf[:, t] * vf[:, t] - erase
                S = S_dec + kn[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
                S_dec_all.append(S_dec)
                v_new_all.append(v_new)
    else:
        S = qf.new_zeros(B, H, K, V)
        if has_init:
            S = S + initial_state.float()
        for t in range(T):
            S_dec = S * gf[:, t].unsqueeze(-1).exp()
            erase = ((bf[:, t].unsqueeze(-1) * kn[:, t].unsqueeze(-1)) * S_dec).sum(-2)
            v_new = wf[:, t] * vf[:, t] - erase
            S = S_dec + kn[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
            S_dec_all.append(S_dec)
            v_new_all.append(v_new)

    new_S_dec = torch.stack(S_dec_all, dim=0).to(device)
    new_v_new = torch.stack(v_new_all, dim=0).to(device)
    return [q, k, v, g, b, w, initial_state, do, dht, new_S_dec, new_v_new]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, T, H, K, V = 1, 26, 16, 37, 46
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="npu") * 0.1
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="npu") * 0.1
v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="npu") * 0.1
g = torch.empty(B, T, H, K, dtype=torch.float32, device="npu").uniform_(-1.0, -0.02)
b = torch.rand(B, T, H, K, dtype=torch.bfloat16, device="npu")
w = torch.rand(B, T, H, V, dtype=torch.bfloat16, device="npu")
initial_state = torch.empty(0, H, K, V, dtype=torch.float32, device="npu")
do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="npu") * 0.1
dht = torch.empty(0, H, K, V, dtype=torch.float32, device="npu")
S_dec_stack = torch.randn(T, B, H, K, V, dtype=torch.float32, device="npu")
v_new_stack = torch.randn(T, B, H, V, dtype=torch.float32, device="npu")
dq, dk, dv, dg, db, dw, dh0 = cann_bench.gdn2_bwd(
    q, k, v, g, b, w, initial_state, do, dht, S_dec_stack, v_new_stack,
    0.164399, 1, [0, 6, 13, 26])
# dq: [1, 26, 16, 37] bfloat16, dh0: [3, 16, 37, 46] float32
```
