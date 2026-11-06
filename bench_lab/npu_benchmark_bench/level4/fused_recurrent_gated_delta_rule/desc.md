# FusedRecurrentGatedDeltaRule 算子 API 描述

## 1. 算子简介

FusedRecurrentGatedDeltaRule 是 GDN（Gated DeltaNet）的 fused recurrent 前向算子，
对齐 fla-org `fused_recurrent_gated_delta_rule_fwd_kernel`（别名 `fused_recurrent_gdn`）。
逐 token 递推：先按门控衰减状态，再用 delta-rule 写入 $v - k^\top S$，最后读出
$o_t = (q\cdot\text{scale})^\top S$。

**主要应用场景**：
- Gated DeltaNet / 线性注意力模型的自回归推理与短序列训练
- 需要逐 token 精确递推（不做 chunk 近似）的 gated delta rule 场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 10 个张量输入 + 10 个属性
- 2 个张量输出（`o`，以及可选的 `final_state`）
- 支持标量 / key 通道 / value 通道三种门控（`gate_mode` 位掩码）、GQA、
  q/k L2 归一化、sigmoid beta 与 state first-V 布局
- 源用例 dtype 为 bfloat16（另含 1 个 float16），状态相关张量为 float32

## 2. 算子定义

### 数学公式

逐 token（$t = 0 \dots T-1$，状态 $S \in \mathbb{R}^{B\times H_v\times K\times V}$ 为 fp32）：

$$
S \leftarrow S \odot \exp(g_t) \;\text{(mode 1)},\quad
S \leftarrow S \odot \exp(gk_t)\;\text{(mode 2)},\quad
S \leftarrow S \odot \exp(gv_t)\;\text{(mode 4)}
$$

其中 mode 1 在 `use_gate_in_kernel` 时由参数生成：

$$
g_t = -\exp(A_{log}) \cdot \mathrm{softplus}(g_t + b_{dt}) \quad (\text{use\_bias 时加 } b_{dt})
$$

$$
\beta_t = \mathrm{sigmoid}(\beta_t)\ (\text{use\_beta\_sigmoid}),\qquad
\beta_t \leftarrow 2\beta_t\ (\text{allow\_neg\_eigval})
$$

$$
\text{residual}_t = \left(v_t - k_t^\top S\right) \odot \beta_t, \qquad
S \leftarrow S + k_t \otimes \text{residual}_t
$$

$$
o_t = \left(q_t \cdot \text{scale}\right)^\top S
$$

GQA：$k/q$ 头按 $H_v/H$ `repeat_interleave` 扩展（$i_h = i_{hv} // (H_v/H)$）。
`state_v_first` 时状态按 $[\dots,V,K]$ 布局出入（内部转置为 $[K,V]$）。
`use_qk_l2norm` 时 $x \leftarrow x/\sqrt{\sum x^2 + 10^{-6}}$（eps 在根号内）。

### 实现建议

1. 逐 token 串行递推（数据依赖），先做门控衰减再计算 residual 与写入。
2. `o` 用更新后的状态读出，`scale` 预乘在 query 上。
3. `output_final_state=True` 时返回 `state`（fp32）；否则返回空 tensor `[0]`
   （源实现返回 None，本 bench 输出 ABI 以空 tensor 表示缺位）。
4. `o` 回铸 `v.dtype`；`final_state` 保持 fp32。

## 3. 接口规范

### 算子原型

```python
cann_bench.fused_recurrent_gated_delta_rule(
    q, k, v, g, gk, gv, beta, A_log, dt_bias, initial_state,
    scale, gate_mode, use_gate_in_kernel, use_bias, use_initial_state,
    output_final_state, use_qk_l2norm, use_beta_sigmoid, allow_neg_eigval,
    state_v_first,
) -> (o, final_state)
```

### 输入参数说明

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `q` | `[B,T,H,K]` | bfloat16 / float16 | query |
| `k` | `[B,T,H,K]` | bfloat16 / float16 | key |
| `v` | `[B,T,HV,V]` | bfloat16 / float16 | value（`HV % H == 0`） |
| `g` | `[B,T,HV]` | bfloat16 / float16 | 标量门控（fuse 时为 pre-activation） |
| `gk` | `[B,T,HV,K]` | bfloat16 / float16 | key 通道门控（log 域衰减） |
| `gv` | `[B,T,HV,V]` | bfloat16 / float16 | value 通道门控（log 域衰减） |
| `beta` | `[B,T,HV]` | bfloat16 / float16 | beta 门（use_beta_sigmoid 时为 raw logits） |
| `A_log` | `[HV]` | float32 | 标量门控参数 |
| `dt_bias` | `[HV]` | float32 | 标量门控偏置 |
| `initial_state` | `[B,HV,K,V]` | float32 | 初始状态（state_v_first 时 `[B,HV,V,K]`） |

> 注意：输入 q/k 需近似单位范数（源用例恒做 L2 归一化）。若 $\|k\|^2$ 明显大于 1，
> delta-rule 的 rank-1 更新因子 $(1-\beta\|k\|^2)$ 幅值超过 1，状态会指数爆炸。
> 本任务的 `get_input` 钩子已按源生成规则重建归一化 q/k 与合法门控。

### 属性

| 参数 | 类型 | 说明 |
|---|---|---|
| `scale` | float | query 缩放系数 |
| `gate_mode` | int | 位掩码：1=标量 / 2=key 通道 / 4=value 通道门控 |
| `use_gate_in_kernel` | bool | 是否由 A_log/dt_bias 生成标量门控 |
| `use_bias` | bool | 标量门控是否加 dt_bias |
| `use_initial_state` | bool | 是否使用 initial_state |
| `output_final_state` | bool | 是否输出 final_state |
| `use_qk_l2norm` | bool | 是否对 q/k 做 L2 归一化 |
| `use_beta_sigmoid` | bool | 是否对 beta 做 sigmoid |
| `allow_neg_eigval` | bool | beta sigmoid 后是否乘 2 |
| `state_v_first` | bool | 状态是否按 `[..,V,K]` 布局出入 |

### 输出

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `o` | `[B,T,HV,V]` | 同 v | 注意力输出 |
| `final_state` | `[B,HV,K,V]` | float32 | 末状态；`output_final_state=False` 时为空 tensor `[0]` |

### 数据类型

`q/k/v/g/gk/gv/beta` 为 bfloat16 或 float16；`A_log/dt_bias/initial_state` 与
`final_state` 为 float32；`o` 与 `v` 同 dtype。

### 规则与约束

- `HV % H == 0`。
- `gate_mode` 为位掩码，可在 `use_gate_in_kernel` 下由 A_log/dt_bias 生成标量门控。
- `g/gk/gv` 为 log 域衰减（≤ 0）；fuse 标量门控时为 pre-activation。
- `state_v_first=True` 时 `initial_state/final_state` 的形状为 `[B,HV,V,K]`。

## 4. 精度要求

golden 内部使用 fp32 递推，`o` 回铸 `v.dtype`；结果与 golden 的误差需满足 bench 的
统一精度门限，推荐以约 $10^{-3}$（fp16/bf16）相对误差为参考。

## 5. 标准 Golden 代码

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
import torch.nn.functional as F

"""
FusedRecurrentGatedDeltaRule 算子 Torch Golden 参考实现

对齐 fla-org `fused_recurrent_gated_delta_rule_fwd_kernel`（GDN 逐 token 递推；
别名 fused_recurrent_gdn）。逐 token（按 t 递增，S 为 [B,HV,K,V] fp32）：

    mode 1（标量门控）: S *= exp(g_t)                    # g [B,T,HV] 每 hv 标量
        若 use_gate_in_kernel（等价 USE_GATE_IN_KERNEL，要求 A_log 非 None）:
            g = -exp(A_log) * softplus(g + dt_bias)      # A_log/dt_bias [HV] 标量
            （use_bias <-> HAS_DT_BIAS）
    mode 2: S *= exp(gk_t) 沿 K 通道                     # gk [B,T,HV,K]
    mode 4: S *= exp(gv_t) 沿 V 通道                     # gv [B,T,HV,V]
    门控按 g→gk→gv 连乘，均在 residual 之前（先衰减后读出）
    β: use_beta_sigmoid 时 sigmoid(β)（allow_neg_eigval 时再 ×2）
    residual = (v − kᵀS)·β ；S += k⊗residual
    o_t = (q·scale)ᵀS  （读出用更新后状态；scale 预乘在 q 上）

自然 e 底门控；o 回铸 v.dtype，final_state fp32；state_v_first 时状态出入
[.., V, K]（内部 [K,V] 转置）；GQA：i_h = i_hv // (HV//H)。

use_qk_l2norm 时按 kernel 约定归一：x / sqrt(Σx² + 1e-6)（eps 在根号内）。
"""


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    gate_mode: int,
    use_gate_in_kernel: bool,
    use_bias: bool,
    use_initial_state: bool,
    output_final_state: bool,
    use_qk_l2norm: bool,
    use_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
) -> tuple:
    """
    GDN fused recurrent 前向

    Args:
        q: query [B, T, H, K]
        k: key [B, T, H, K]
        v: value [B, T, HV, V]（HV % H == 0）
        g: 标量门控 [B, T, HV]（fuse 时原始 pre-activation，否则 log 域衰减）
        gk: key 通道门控 [B, T, HV, K]（log 域衰减）
        gv: value 通道门控 [B, T, HV, V]（log 域衰减）
        beta: beta 门 [B, T, HV]（use_beta_sigmoid 时 raw logits，否则已 sigmoid）
        A_log: [HV] fp32
        dt_bias: [HV] fp32
        initial_state: 初始状态 [B, HV, K, V]（state_v_first 时 [B, HV, V, K]）
        scale: query 缩放系数
        gate_mode: 位掩码，1=标量门控 / 2=key 通道门控 / 4=value 通道门控
        use_gate_in_kernel: 是否在核内由 A_log/dt_bias 生成标量门控
        use_bias: 标量门控是否加 dt_bias
        use_initial_state: 是否使用初始状态
        output_final_state: 是否输出末状态
        use_qk_l2norm: 是否对 q/k 做 L2 归一化
        use_beta_sigmoid: 是否对 beta 做 sigmoid
        allow_neg_eigval: beta sigmoid 后是否再乘 2
        state_v_first: 状态是否按 [.., V, K] 布局出入

    Returns:
        (o, final_state)；o: [B,T,HV,V]（v.dtype）；final_state: [B,HV,K,V] fp32
        （state_v_first 时 [B,HV,V,K]）。output_final_state=False 时按本 bench 的
        输出 ABI 返回空 tensor [0]（源实现返回 None，空 tensor 与之语义等价）。
    """
    batch, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if value_heads % heads != 0:
        raise ValueError("value heads must be divisible by query heads")
    query = q.float()
    key = k.float()
    if use_qk_l2norm:
        query = _l2norm(query)
        key = _l2norm(key)
    query = query.repeat_interleave(value_heads // heads, dim=2)
    key = key.repeat_interleave(value_heads // heads, dim=2)

    scalar_gate = g.float()
    if gate_mode & 1 and use_gate_in_kernel:
        if use_bias:
            scalar_gate = scalar_gate + dt_bias.float()
        scalar_gate = -torch.exp(A_log.float()) * F.softplus(
            scalar_gate
        )
    key_gate = gk.float()
    value_gate = gv.float()
    beta_value = beta.float()
    if use_beta_sigmoid:
        beta_value = torch.sigmoid(beta_value)
        if allow_neg_eigval:
            beta_value = beta_value * 2.0

    if use_initial_state:
        state = initial_state.float().clone()
        if state_v_first:
            state = state.transpose(-2, -1).contiguous()
    else:
        state = torch.zeros(
            batch,
            value_heads,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
    output = torch.empty_like(v)
    for token in range(tokens):
        if gate_mode & 1:
            state = state * torch.exp(
                scalar_gate[:, token, :, None, None]
            )
        if gate_mode & 2:
            state = state * torch.exp(
                key_gate[:, token, :, :, None]
            )
        if gate_mode & 4:
            state = state * torch.exp(
                value_gate[:, token, :, None, :]
            )
        key_token = key[:, token]
        residual = v[:, token].float() - torch.einsum(
            "bhk,bhkv->bhv", key_token, state
        )
        residual = residual * beta_value[:, token, :, None]
        state = state + torch.einsum(
            "bhk,bhv->bhkv", key_token, residual
        )
        output[:, token] = (
            torch.einsum(
                "bhk,bhkv->bhv", query[:, token] * scale, state
            ).to(v.dtype)
        )
    if state_v_first:
        state = state.transpose(-2, -1).contiguous()
    final_state = (state if output_final_state
                   else torch.empty(0, dtype=torch.float32, device=q.device))
    return output, final_state


def _l2norm(xf):
    """kernel 约定：x / sqrt(sum(x^2) + 1e-6)，eps 在根号内。"""
    return xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)


def get_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float = 1.0,
    gate_mode: int = 7,
    use_gate_in_kernel: bool = False,
    use_bias: bool = False,
    use_initial_state: bool = False,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    **kwargs,
) -> list:
    """按源 get_input_groups 的分布重建所有张量输入。

    单区间 value_range 无法表达两条关键约束：
      1. q/k 必须近似单位范数（源用例恒做 L2 归一化）。若 ‖k‖² 明显大于 1，
         delta-rule 的 rank-1 更新因子 (1 − beta‖k‖²) 幅值 > 1，状态会指数爆炸；
      2. g/gk/gv 是 log 域衰减，必须 <= 0（fuse 标量门控时为 pre-activation），
         beta 在 use_beta_sigmoid=False 时必须已在 (0,1)。
    这里按源生成规则（randn 缩放 / softplus 负衰减 / sigmoid beta / 小尺寸
    A_log、dt_bias、initial_state）重建，保证递推数值稳定。

    Returns:
        10 个张量，顺序与 golden 签名一致。
    """
    seed = 50
    for dim in list(q.shape) + list(v.shape) + list(gk.shape) + list(gv.shape):
        seed = (seed * 131 + int(dim)) % 1000003
    seed = (seed * 131 + int(gate_mode) + 17 * int(bool(use_gate_in_kernel))
            + 31 * int(bool(use_qk_l2norm)) + 47 * int(bool(use_beta_sigmoid))
            + 61 * int(bool(allow_neg_eigval)) + 71 * int(bool(state_v_first))) % 1000003
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _randn(shape, dtype_, scale_=1.0):
        tensor = torch.randn(tuple(shape), generator=gen, dtype=torch.float32) * scale_
        return tensor.to(dtype_)

    def _normalized(shape, dtype_):
        tensor = torch.randn(tuple(shape), generator=gen, dtype=torch.float32)
        tensor = F.normalize(tensor, p=2.0, dim=-1, eps=1e-6)
        return tensor.to(dtype_)

    def _decay(shape, dtype_, raw_gate):
        tensor = torch.randn(tuple(shape), generator=gen, dtype=torch.float32)
        if raw_gate:
            tensor = tensor * 0.8
        else:
            tensor = -F.softplus(tensor) * 0.05
        return tensor.to(dtype_)

    def _beta(shape, dtype_, raw_beta):
        tensor = torch.randn(tuple(shape), generator=gen, dtype=torch.float32)
        if not raw_beta:
            tensor = torch.sigmoid(tensor)
        return tensor.to(dtype_)

    fused_scalar = bool(int(gate_mode) & 1) and bool(use_gate_in_kernel)
    raw_beta = bool(use_beta_sigmoid)
    return [
        _normalized(q.shape, q.dtype),
        _normalized(k.shape, k.dtype),
        _randn(v.shape, v.dtype, 0.2),
        _decay(g.shape, g.dtype, fused_scalar),
        _decay(gk.shape, gk.dtype, False),
        _decay(gv.shape, gv.dtype, False),
        _beta(beta.shape, beta.dtype, raw_beta),
        _randn(A_log.shape, torch.float32, 0.4),
        _randn(dt_bias.shape, torch.float32, 0.3),
        _randn(initial_state.shape, torch.float32, 0.04),
    ]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, T, H, HV, K, V = 1, 640, 1, 2, 32, 32
q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).to(torch.bfloat16).npu()
k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1).to(torch.bfloat16).npu()
v = (torch.randn(B, T, HV, V) * 0.2).to(torch.bfloat16).npu()
g = (-torch.nn.functional.softplus(torch.randn(B, T, HV)) * 0.05).to(torch.bfloat16).npu()
gk = (-torch.nn.functional.softplus(torch.randn(B, T, HV, K)) * 0.05).to(torch.bfloat16).npu()
gv = (-torch.nn.functional.softplus(torch.randn(B, T, HV, V)) * 0.05).to(torch.bfloat16).npu()
beta = torch.sigmoid(torch.randn(B, T, HV)).to(torch.bfloat16).npu()
A_log = (torch.randn(HV) * 0.4).npu()
dt_bias = (torch.randn(HV) * 0.3).npu()
initial_state = (torch.randn(B, HV, K, V) * 0.04).npu()

o, final_state = cann_bench.fused_recurrent_gated_delta_rule(
    q, k, v, g, gk, gv, beta, A_log, dt_bias, initial_state,
    0.1767767, 7, False, False, False, False, False, False, False, False,
)
# o: [1, 640, 2, 32] bfloat16; final_state: 空 tensor [0]
```
