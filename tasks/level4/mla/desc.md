# MLA 算子 API 描述

## 1. 算子简介

多头潜在注意力 (Multi-Head Latent Attention) 算子，仅包含注意力计算部分（不含 KV 解压缩），是 DeepSeek-V2/V3 等模型的核心注意力机制。

**主要应用场景**：
- 大语言模型推理中的 MLA 注意力计算（如 DeepSeek-V2、DeepSeek-V3）
- 超长序列推理场景
- 需要在推理效率和模型质量之间取得平衡的大规模 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（q_nope, q_rope, k_nope, k_rope, v）单输出
- Q 和 K 均分为 nope 和 rope 两部分传入，内部拼接后计算注意力
- V 与 K_nope 不仅 shape 相同 (`[B, S_kv, N_kv, d_nope]`)，**数值上也完全相同**——生产环境中 V 就是从 latent KV cache 读出的 K_nope 那段内存，共享同一份数据；算子接口为兼容通用 attention API 保留为独立入参
- 输出 head dim 为 d_nope
- 支持 GQA 模式（N_q 个 query head 共享 N_kv 个 KV head），MLA 典型配置 N_kv=1
- 支持 BSND 和 BNSD 两种输入输出 layout
- 支持 `is_causal` 因果掩码：仅计算 attention 矩阵 [S, S_kv] 中从右下角向左上方延伸 45° 对角线及其下方部分（其余位置在 softmax 前置 -inf）

## 2. 算子定义

### 数学公式

$$
Q = \text{concat}(Q_{nope}, Q_{rope}) \quad \text{dim: } d_{nope} + d_{rope}
$$

$$
K = \text{concat}(K_{nope}, K_{rope}) \quad \text{dim: } d_{nope} + d_{rope}
$$

$$
y = \text{softmax}\left(Q \times K^T \times \text{scaleValue}\right) \times V
$$

其中：
- $Q_{nope}$、$K_{nope}$ 为 nope 部分，维度 $d_{nope}$
- $Q_{rope}$、$K_{rope}$ 为 rope 部分（经过 RoPE 编码），维度 $d_{rope}$
- $V$ 的 head dim 为 $d_{nope}$，因此输出的 head dim 也为 $d_{nope}$
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{d_{nope} + d_{rope}}$）
- 当 $N_q > N_{kv}$ 时进行 GQA 扩展，每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

具体子步骤：
1. **Q 拼接**：$Q = \text{concat}(Q_{nope}, Q_{rope})$
2. **K 拼接**：$K = \text{concat}(K_{nope}, K_{rope})$
3. **GQA 扩展**：将 KV head 复制 $N_q / N_{kv}$ 次匹配 query head 数
4. **缩放点积**：$\text{scores} = Q \times K^T \times \text{scaleValue}$
5. **因果掩码（可选）**：当 `is_causal=True` 时，对 `scores[..., i, j]` 满足 $j > i + (S_{kv} - S)$ 的位置置为 $-\infty$（即仅保留从右下角向左上方 45° 延伸的对角线及其下方部分），$S=S_{kv}$ 时退化为标准下三角掩码
6. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
7. **加权求和**：$y = \text{attn\_weights} \times V$

## 3. 接口规范

### 算子原型

```python
cann_bench.mla(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor v, int numKVHeads=1, float scaleValue=-1.0, str inputLayout="BSND", bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q_nope | Tensor | 必选 | query 的 nope 部分，BSND: [B, S, N_q, d_nope]，BNSD: [B, N_q, S, d_nope] |
| q_rope | Tensor | 必选 | query 的 rope 部分，BSND: [B, S, N_q, d_rope]，BNSD: [B, N_q, S, d_rope] |
| k_nope | Tensor | 必选 | key 的 nope 部分，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope] |
| k_rope | Tensor | 必选 | key 的 rope 部分，BSND: [B, S_kv, N_kv, d_rope]，BNSD: [B, N_kv, S_kv, d_rope] |
| v | Tensor | 必选 | 值张量，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]。语义上要求 `v == k_nope`（数值完全相同，共享 latent KV cache） |
| numKVHeads | int | 1 | KV 头数 |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(d_nope + d_rope) |
| inputLayout | str | "BSND" | 输入输出 layout，"BSND" 或 "BNSD" |
| is_causal | bool | False | 是否启用因果掩码。False 时全计算；True 时仅计算 [S, S_kv] attention 矩阵中从右下角向左上方 45° 延伸的对角线及其下方部分（即满足 $j \le i + (S_{kv} - S)$ 的位置），上方部分在 softmax 前置 -inf |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | BSND: [B, S, N_q, d_nope]；BNSD: [B, N_q, S, d_nope] | 与输入相同 | 注意力输出张量，head dim 为 d_nope |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor 的 dtype 必须一致
- N_q 必须能被 N_kv 整除
- q_nope 和 k_nope 的 head dim 一致（d_nope），q_rope 和 k_rope 的 head dim 一致（d_rope）
- v 的 head dim = d_nope，输出的 head dim 也为 d_nope
- **语义约束 `v == k_nope`**：v 与 k_nope 在数值上必须完全相同（共享同一份 latent KV cache）。算子接口保留独立入参以兼容标准 attention API；评测时 `get_input` 会将 v 规整为 k_nope 的副本，并将同一组合法输入同时交给 Golden 与被测算子
- 所有输入和输出遵循相同的 layout（BSND 或 BNSD）
- `is_causal=True` 时要求 $S \le S_{kv}$（否则 mask 会将部分 query 行全部屏蔽，导致 softmax 出现 NaN）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 128 | 测试覆盖域内典型值与独立边界 |
| `S`（query 序列长度） | 1 ~ 512 | 测试覆盖域内典型值与独立边界 |
| `S_kv`（KV 序列长度） | 128 ~ 2048 | 测试覆盖域内典型值与独立边界 |
| `N_q`（query head 数） | 32 ~ 128 | 测试覆盖域内典型值与独立边界 |
| `N_kv`（KV head 数 = `numKVHeads`） | 1 | MLA 典型配置，固定为 1 |
| `d_nope`（nope head dim） | 128 ~ 512，64 对齐 | 测试覆盖域内典型值与独立边界 |
| `d_rope`（rope head dim） | {64, 128}（原范围与 64 对齐约束的交集） | 测试覆盖域内典型值与独立边界 |
| `scaleValue` | float (`<=0` 自动取 $1/\sqrt{d_{nope}+d_{rope}}$) | 测试覆盖域内典型值与独立边界 |
| `inputLayout` | `"BSND"` / `"BNSD"` | 测试覆盖域内典型值与独立边界 |
| `is_causal` | `True` / `False` | 测试覆盖域内典型值与独立边界 |

约束：`N_q` 必须能被 `N_kv` 整除；`q_nope`、`k_nope`、`v` 的最后一维必须相同（均为 `d_nope`），`q_rope`、`k_rope` 的最后一维必须相同（均为 `d_rope`）；所有输入 dtype 必须一致；`is_causal=True` 时必须满足 $S \le S_{kv}$。

## 参考计算与测试预算

CPU 参考保留 fp16/bf16 已量化的输入，以 FP32 完成 matmul/softmax，输出转换回输入 dtype；评测器送入 FP64 时保持 FP64 oracle。参考按 query 行分块，每个 softmax 始终覆盖全部 KV。每块 scores 不超过 4,194,304 元素（KV 轴超过此预算时至少一行）；当前测试每例 score 总量不超过 8,388,608 元素，单个输入不超过 8,388,608 元素。大 batch、长序列及大 head 数由独立用例承重。现有 get_input 保持 v=k_nope.clone() 合同，首次与复验均应执行。计分输入使用有限值；控制参数 scaleValue 必须有限。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python

import torch


def get_input(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> list:
    """Preserve the shared-latent contract for both first and fresh-input attempts."""
    return [q_nope, q_rope, k_nope, k_rope, k_nope.clone()]


def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Attention-only MLA; bounded score storage and full-KV-axis softmax.

    Input quantization is retained. FP16/BF16 CPU arithmetic uses FP32; a
    FP64 input remains FP64 for the judge oracle. Only the final output is
    converted back to the input dtype. Query tiling does not approximate or
    truncate the attention matrix, and KV heads are never expanded to N_q.
    """
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    output_dtype = q_nope.dtype
    compute_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    tensors = (q_nope, q_rope, k_nope, k_rope, v)
    qn, qr, kn, kr, vv = [x.to(compute_dtype) for x in tensors]
    if inputLayout == "BNSD":
        qn, qr, kn, kr, vv = [x.permute(0, 2, 1, 3) for x in (qn, qr, kn, kr, vv)]
    B, S, Nq, Dv = qn.shape
    Sk, Nkv = kn.shape[1:3]
    if numKVHeads != Nkv or Nq % Nkv:
        raise ValueError("numKVHeads must match KV shape and divide query heads")
    if is_causal and S > Sk:
        raise ValueError("right-bottom causal attention requires S <= S_kv")
    G = Nq // Nkv
    Dqk = Dv + qr.shape[-1]
    scale = Dqk**-0.5 if scaleValue <= 0 else scaleValue
    q = torch.cat((qn, qr), -1)
    k = torch.cat((kn, kr), -1)
    out = torch.empty((B, S, Nq, Dv), dtype=compute_dtype, device=q.device)
    # A tile stores at most 4M logits, independent of B, Nq and query length.
    row_tile = max(1, min(G * S, 4194304 // Sk))
    keys = torch.arange(Sk, device=q.device)
    for b in range(B):
        for h in range(Nkv):
            rows = q[b, :, h * G : (h + 1) * G].transpose(0, 1).reshape(G * S, Dqk)
            key = k[b, :, h]
            value = vv[b, :, h]
            dest = torch.empty((G * S, Dv), dtype=compute_dtype, device=q.device)
            for start in range(0, G * S, row_tile):
                end = min(G * S, start + row_tile)
                scores = torch.matmul(rows[start:end], key.transpose(-2, -1)) * scale
                if is_causal:
                    query_rows = torch.arange(start, end, device=q.device) % S
                    scores = scores.masked_fill(
                        keys[None] > query_rows[:, None] + Sk - S, -torch.inf
                    )
                probs = torch.nn.functional.softmax(scores, dim=-1)
                dest[start:end] = torch.matmul(probs, value)
            out[b, :, h * G : (h + 1) * G] = dest.reshape(G, S, Dv).transpose(0, 1)
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)
    return out.to(output_dtype).contiguous()


# Native reference only: round stage outputs to the input dtype, as in the
# published Torch reference. The public Golden above remains FP32/FP64.
_NATIVE_SCORE_ELEMENTS = 4 * 1024**2


def _native_attention(q, k, v, scale, is_causal):
    """BSND grouped attention; full KV softmax, bounded query-row workspace."""
    batch, query_length, query_heads, width = q.shape
    key_length, kv_heads = k.shape[1:3]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if is_causal and query_length > key_length:
        raise ValueError("right-bottom causal attention requires S <= S_kv")
    group = query_heads // kv_heads
    value_width = v.shape[-1]
    out = torch.empty(
        (batch, query_length, query_heads, value_width), dtype=q.dtype, device=q.device
    )
    row_tile = max(1, min(group * query_length, _NATIVE_SCORE_ELEMENTS // key_length))
    keys = torch.arange(key_length, device=q.device)
    for b in range(batch):
        for h in range(kv_heads):
            rows = (
                q[b, :, h * group : (h + 1) * group].transpose(0, 1).reshape(-1, width)
            )
            destination = torch.empty(
                (group * query_length, value_width), dtype=q.dtype, device=q.device
            )
            for start in range(0, group * query_length, row_tile):
                end = min(group * query_length, start + row_tile)
                # Each operation stores an input-dtype result; do not replace
                # this path with the promoted public Golden.
                scores = torch.matmul(rows[start:end], k[b, :, h].transpose(0, 1))
                scores = scores * scale
                if is_causal:
                    query_rows = (
                        torch.arange(start, end, device=q.device) % query_length
                    )
                    scores = scores.masked_fill(
                        keys[None] > query_rows[:, None] + key_length - query_length,
                        -torch.inf,
                    )
                probabilities = torch.nn.functional.softmax(scores, dim=-1)
                destination[start:end] = torch.matmul(probabilities, v[b, :, h])
            out[b, :, h * group : (h + 1) * group] = destination.reshape(
                group, query_length, value_width
            ).transpose(0, 1)
    return out


def mla_bench(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Independent native-stage reference; preserves the shared-latent inputs."""
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    tensors = (q_nope, q_rope, k_nope, k_rope, v)
    if inputLayout == "BNSD":
        tensors = tuple(t.permute(0, 2, 1, 3) for t in tensors)
    qn, qr, kn, kr, value = tensors
    if numKVHeads != kn.shape[2]:
        raise ValueError("numKVHeads must match KV shape")
    q, k = torch.cat((qn, qr), -1), torch.cat((kn, kr), -1)
    scale = q.shape[-1] ** -0.5 if scaleValue <= 0 else scaleValue
    out = _native_attention(q, k, value, scale, is_causal)
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)
    return out.contiguous()
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv = 2, 128, 256
N_q, N_kv = 128, 1
d_nope, d_rope = 512, 64

# BSND layout
q_nope = torch.randn(B, S, N_q, d_nope, dtype=torch.float16, device="npu")
q_rope = torch.randn(B, S, N_q, d_rope, dtype=torch.float16, device="npu")
k_nope = torch.randn(B, S_kv, N_kv, d_nope, dtype=torch.float16, device="npu")
k_rope = torch.randn(B, S_kv, N_kv, d_rope, dtype=torch.float16, device="npu")
v = k_nope.clone()
y = cann_bench.mla(q_nope, q_rope, k_nope, k_rope, v,
                      numKVHeads=N_kv, scaleValue=-1.0, inputLayout="BSND", is_causal=False)
# y shape: [B, S, N_q, d_nope] = [2, 128, 128, 512]
```
