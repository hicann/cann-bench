# KimiDeltaAttention 算子 API 描述

版本：2026-10-30 质量修订。level5 任务定义保持不变；本修订检查 CPU 参考、合法输入域和测试证据，不声明新的 NPU 精度、时延或硬件下界测量。

## 1. 算子简介

Kimi Delta Attention的门控状态递推核心；可选QK归一化；输出序列及最终状态。

参考计算流程：Q/K L2归一化（可选） → 逐通道Exp衰减 → 状态读取与残差 → Rank-1状态更新 → Q读取输出＋末态。

### 本次评测边界

包含QK归一化开关、逐通道log gate、Delta残差更新、跨token状态依赖和末态输出。
不含QKV线性投影、短卷积、门控参数生成、输出门控/RMSNorm、输出投影、反向或变长cu_seqlens。
本任务冻结Kimi Linear的KDA核心，不宣称与Kimi K3整层参数化完全相同。

## 2. 算子定义


令归一化后的查询和键为 $q_t,k_t$（关闭属性时直接使用输入）。
\[
\bar H_t=\operatorname{diag}(e^{g_t})H_{t-1},\quad
r_t=v_t-k_t^T\bar H_t,\quad
H_t=\bar H_t+\beta_t k_t r_t^T,\quad
y_t=\mathrm{scale}\,q_t^T H_t.
\]
归一化严格使用 $x/\sqrt{\sum x^2+10^{-6}}$。先衰减再计算残差，使用更新后的状态生成输出。
序列被分块是实现选择，不是语义属性；任意分块传递末态必须与一次处理整段等价。


## 3. 接口规范

```text
kimi_delta_attention(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor initial_state, bool use_qk_l2norm=True, float scaleValue=-1.0) -> (Tensor y, Tensor final_state)
```

### 输入

| 参数 | Shape | dtype | 含义 |
|---|---|---|---|
| `q` | `[B,T,H,Dk]` | float16 / bfloat16 / float32 | 未归一化或按属性直接使用的查询 |
| `k` | `[B,T,H,Dk]` | float16 / bfloat16 / float32 | 键；归一化属于被测计算 |
| `v` | `[B,T,H,Dv]` | float16 / bfloat16 / float32 | 值 |
| `g` | `[B,T,H,Dk]` | float32 | 逐token逐key通道自然对数衰减；不是累计gate |
| `beta` | `[B,T,H]` | float32 | 直接给定写入强度[0,1]；不再sigmoid |
| `initial_state` | `[B,H,Dk,Dv]` | float32 | 必选初态；无历史时传零张量；不可原地修改 |

### 输出

| 参数 | Shape | dtype | 含义 |
|---|---|---|---|
| `y` | `[B,T,H,Dv]` | float16 / bfloat16 / float32 | dtype与q一致 |
| `final_state` | `[B,H,Dk,Dv]` | float32 | 末态；必须参与精度比较 |

### 支持范围与约束

B∈[1,4]，T∈[1,512]，H∈[1,16]，Dk/Dv∈[1,128]；允许矩形状态和非对齐维。
q/k/v同dtype；g、beta和initial_state为FP32；所有输入有限，g∈[-5,0]、beta∈[0,1]。
开启归一化时q/k∈[-1,1]；关闭时q/k∈[-0.05,0.05]以保持有限序列的数值范围；v/state∈[-0.5,0.5]。
scaleValue为有限数，正值≤1；非正值均表示自动缩放。initial_state必选、只读；不能只校验y忽略末态。

接口只定义有限合法输入，不把预期异常混入20个计分case；本版不做NaN/Inf语义承诺。代码中的形状检查不意味着穷举检查了全部合法域。

## 4. 精度要求

Golden/bench 对 FP16/BF16/FP32 输入采用 FP32 核心计算；开发者 FP64 输入全程采用 FP64，y转回输入dtype；末态保持FP32。oracle由评测器提供FP64输入，输出保持FP64，不允许假FP64回落。
独立CPU自测使用：FP32 `rtol=1e-4, atol=1e-5`；FP16 `rtol=3e-3, atol=3e-4`；BF16 `rtol=2e-2, atol=2e-3`；末态 `rtol=2e-4, atol=2e-5`。
这些是本包验证门限，不覆盖仓库正式精度检查器。正式上线仍需用仓库checker和同精度参考验证NPU候选，不写入未经真机校准的算子级阈值。
本实现不承诺与公开低精度kernel逐位相同，也不能用该串行/分块PyTorch Golden的速度作为有竞争力的性能baseline。

## 5. 标准 Golden 代码

下方内容与同目录golden.py逐字一致，入口、oracle及bench签名保持一致。

```python

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
```

## 6. 测试用例与验收

恰好20个公开Core候选用例；Smoke为ID 1、9、15、18的子集，不能重复进入计分分母。另有80个内部用例（ID 21–100），公开与内部输入域均在本次修订中检查；边界用例单独归类，不作为代表性工作量的证据。
用途配额为4热点/4形态/3尾块/3精度/3属性/2边界/1性能。热点是模型核心维度或缩小的代表性情景，并非平台实测热点排名。

| ID | 主要目的 | 输入尺寸摘要 | dtype | 设计要点 |
|---:|---|---|---|---|
| 1 | hotspot | [1, 64, 8, 128]; state=[1, 8, 128, 128] | float16 | typical-prefill-zero-state |
| 2 | hotspot | [1, 128, 16, 128]; state=[1, 16, 128, 128] | bfloat16 | typical-prefill-bf16 |
| 3 | hotspot | [4, 1, 16, 128]; state=[4, 16, 128, 128] | bfloat16 | batched-decode-state |
| 4 | hotspot | [2, 64, 8, 128]; state=[2, 8, 128, 128] | float16 | continuation-chunk |
| 5 | shape | [1, 1, 1, 4]; state=[1, 1, 4, 3] | float32 | tiny-rectangular-state |
| 6 | shape | [4, 32, 4, 64]; state=[4, 4, 64, 96] | float16 | batch-head-independence |
| 7 | shape | [1, 512, 4, 64]; state=[1, 4, 64, 64] | bfloat16 | long-recurrence |
| 8 | shape | [1, 128, 8, 64]; state=[1, 8, 64, 128] | float16 | rectangular-value-width |
| 9 | tail | [1, 63, 3, 63]; state=[1, 3, 63, 65] | float16 | tail-T63-D63x65 |
| 10 | tail | [1, 65, 5, 64]; state=[1, 5, 64, 96] | bfloat16 | tail-T65-H5 |
| 11 | tail | [2, 129, 2, 33]; state=[2, 2, 33, 31] | float32 | tail-T129-D33x31 |
| 12 | dtype | [1, 96, 4, 64]; state=[1, 4, 64, 64] | float32 | fp32-reference-path |
| 13 | dtype | [1, 17, 2, 32]; state=[1, 2, 32, 48] | float16 | near-zero-qk-epsilon |
| 14 | dtype | [1, 127, 4, 64]; state=[1, 4, 64, 64] | bfloat16 | strong-decay-bf16 |
| 15 | attr | [1, 31, 2, 16]; state=[1, 2, 16, 24] | float32 | no-normalization-explicit-scale |
| 16 | attr | [1, 32, 4, 64]; state=[1, 4, 64, 64] | float16 | no-decay-explicit-scale |
| 17 | attr | [1, 8, 2, 16]; state=[1, 2, 16, 24] | bfloat16 | read-only-initial-state |
| 18 | boundary | [1, 7, 2, 16]; state=[1, 2, 16, 24] | float32 | zero-query-key-state-output |
| 19 | boundary | [1, 17, 3, 32]; state=[1, 3, 32, 48] | float16 | channel-decay-no-write |
| 20 | perf | [1, 256, 16, 128]; state=[1, 16, 128, 128] | bfloat16 | bounded-recurrence-throughput |

### 输入生成与公平性

KDA无需get_input；通用DataGenerator按每输入值域生成，归一化必须留在算子内。
测试固定seed保证可复现，并以不同seed复测非结构数值。CSV与YAML严格对齐，无baseline_perf_us或t_hw_us字段。

### 成本边界

20例最大输入输出合计约 8.094 MiB，全部小于1GB。本版不为达到大张量占比强行扩大注意力/递推任务，S/M/L按本算子的工作量相对分档；这是有界Core设计，不代表执行成本通过评审。
百万token、超大batch、完整模型层以及压缩器/indexer融合留待Extended方案；NPU时间和峰值内存仍需人工复核。

### 当前发布条件

本次交付算子定义、参考实现和20个公开用例。NPU精度、性能baseline和硬件下界尚未验证；本PR不包含Ascend C实现，不改变现有主线计分集合。
已有公开实现不因更换名称或shape而消失。本题目范围及用例设计不构成不可复用公开代码的承诺。

### 参考文档

- [Kimi Linear，v2，公式1](https://arxiv.org/html/2510.26692v2#S3)
- [FLA首次KDA参考，固定提交](https://github.com/fla-org/flash-linear-attention/blob/b5d48b7d2376b7c9b344d603591cb06d93c13aea/fla/ops/kda/naive.py)
- [FLA v0.4.0归一化epsilon约定](https://raw.githubusercontent.com/fla-org/flash-linear-attention/v0.4.0/fla/ops/gated_delta_rule/fused_recurrent.py)
