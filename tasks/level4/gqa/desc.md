# GQA 算子 API 描述

## 1. 算子简介

分组查询注意力 (Grouped Query Attention) 算子，多个 query head 共享一组 key/value head，对已分头的 Q/K/V 执行注意力计算，在保持模型质量的同时显著减少 KV cache 内存占用和推理计算量。

**主要应用场景**：
- 大语言模型推理中的高效注意力计算（如 LLaMA-2 70B、Mistral）
- 长序列推理场景中降低 KV cache 内存开销
- 需要在模型质量和推理效率之间平衡的 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value）单输出，执行分组缩放点积注意力
- 输入为已分头的张量，不包含 QKV 投影和输出投影步骤
- N_q 必须能被 N_kv 整除，每个 KV head 被 N_q/N_kv 个 query head 共享
- 支持 `is_causal` 因果掩码：仅计算 attention 矩阵 [S, S_kv] 中从右下角向左上方延伸 45° 对角线及其下方部分（其余位置在 softmax 前置 -inf）

## 2. 算子定义

### 数学公式

对于第 $i$ 个 query head，使用第 $\lfloor i \times N_{kv} / N_q \rfloor$ 个 KV head：

$$
\text{head}_i = \text{softmax}\left(Q_i \times K_{g(i)}^T \times \text{scaleValue}\right) \times V_{g(i)}
$$

其中：
- $N_q$ 为 query 头数，$N_{kv}$ 为 KV 头数，$N_q$ 必须能被 $N_{kv}$ 整除
- $g(i) = \lfloor i \times N_{kv} / N_q \rfloor$ 为第 $i$ 个 query head 对应的 KV head 索引
- $D$ 为每个头的维度
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{D}$）
- 每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

计算步骤（对每个 batch $b$ 与 query head 索引 $h_q \in [0, N_q)$，令 $g = g(h_q)$）：

1. **缩放点积**：$\text{scores}[i, j] = (Q[b, i, h_q, :] \cdot K[b, j, g, :]) \times \text{scaleValue}$，形状 $[S, S_{kv}]$
2. **因果掩码（可选）**：当 `is_causal=True` 时，对 `scores[i, j]` 满足 $j > i + (S_{kv} - S)$ 的位置置为 $-\infty$（即仅保留 $[S, S_{kv}]$ 矩阵从右下角向左上方 45° 延伸的对角线及其下方部分），$S = S_{kv}$ 时退化为标准下三角掩码
3. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
4. **加权求和**：$y[b, :, h_q, :] = \text{attn\_weights} \times V[b, :, g, :]$，形状 $[S, D]$

### 参考计算与资源预算

CPU 参考保留生成器已量化的 fp16/bf16 输入值，以 fp32 执行 matmul 和 softmax，最终转换回输入 dtype；评测器送入 fp64 时保留 fp64 oracle。参考按 batch/query 分块，每块 softmax 始终遍历完整 KV 轴，不改变数学结果；每块 scores 目标不超过约 4M 元素（极端维度下最少一行）。测试集将大 batch、长序列、大 head 数与大 D 分别承重，避免无必要地组合所有上界。用例仍覆盖 prefill、decode、MTP、MQA/MHA、分组比、矩形右下角因果掩码和显式 scale。

### 实现建议

> 本节为参考建议，**不是算子语义约束**。任何与上文数学公式等价、满足 §4 精度要求的实现均符合 benchmark 要求；本节仅就常见性能陷阱与误读 §5 Golden 代码的风险给出提示。

- §5 Golden 代码在各 query 块内把分组维 `G = N_q / N_kv` 折叠进 matmul 的 M 维（`Q` 变形为 `[B, N_kv, G*S, D]`），`K`/`V` 保持 `N_kv` 头随 batched matmul 复用、不物化到 `N_q` 头；该写法即下条建议的实现，可直接参考。
- 建议按头索引 `g(h_q)` 直接复用 N_kv 个 KV head；若在算子内部或调用前将 K/V 在头维度物化复制 G 份扩展到 N_q，会抹掉 GQA 相对 MHA 的 KV cache 内存收益（占用变为 $N_q \cdot S_{kv} \cdot D$ 而非 $N_{kv} \cdot S_{kv} \cdot D$），并引入冗余访存。
- CANN 原生 GQA 算子（如 `FusedInferAttentionScore`、`npu_fusion_attention`）通过 `num_key_value_heads` 属性告知 kernel 分组比、内部按索引复用 KV，可作为参考实现路径。

## 3. 接口规范

### 算子原型

```python
cann_bench.gqa(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量（已分头），shape 为 [B, S, N_q, D] |
| key | Tensor | 必选 | 键张量（已分头），shape 为 [B, S_kv, N_kv, D] |
| value | Tensor | 必选 | 值张量（已分头），shape 为 [B, S_kv, N_kv, D] |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(D) |
| is_causal | bool | False | 是否启用因果掩码。False 时全计算；True 时仅计算 [S, S_kv] attention 矩阵中从右下角向左上方 45° 延伸的对角线及其下方部分（即满足 $j \le i + (S_{kv} - S)$ 的位置），上方部分在 softmax 前置 -inf |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, N_q, D] | 与输入 query 相同 | 分组查询注意力输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（query, key, value）的 dtype 必须一致
- `query` 的 shape 为 [B, S, N_q, D]，`key` 和 `value` 的 shape 为 [B, S_kv, N_kv, D]
- N_q 必须能被 N_kv 整除，分组比 G = N_q / N_kv
- 当 N_kv == N_q 时退化为标准多头注意力 (MHA)
- 当 N_kv == 1 时退化为多查询注意力 (MQA)
- `scaleValue` 通常设置为 $1/\sqrt{D}$，当 <= 0 时自动使用该值
- `is_causal=True` 时要求 $S \le S_{kv}$（否则 mask 会将部分 query 行全部屏蔽，导致 softmax 出现 NaN）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 256 | 公开用例实测 2 ~ 128 |
| `S`（query 序列长度） | 1 ~ 4096 | 公开用例实测 1 ~ 512；S=1 / 2 / 4 对应 decode / MTP 场景 |
| `S_kv`（key/value 序列长度） | 1 ~ 8192 | 公开用例实测 64 ~ 512；`is_causal=True` 要求 $S \le S_{kv}$ |
| `N_q`（query 头数） | 1 ~ 256 | cases.csv 实测 32 ~ 128；必须满足 `N_q % N_kv == 0` |
| `N_kv`（KV 头数） | 1 ~ 256 | cases.csv 实测 1 / 4 / 8 / 32；`N_kv==N_q` 退化为 MHA，`N_kv==1` 退化为 MQA |
| `D`（每头维度） | 64 ~ 512，64 对齐 | cases.csv 实测 128 / 256 |
| `scaleValue` | 有限 float | cases.csv 实测 -1.0（自动 $1/\sqrt{D}$）和 0.08838（显式 $1/\sqrt{128}$）；<=0 时自动使用 $1/\sqrt{D}$ |
| `is_causal` | {False, True} | cases.csv 两者皆覆盖；True 时按右下角对齐生成 [S, S_kv] 因果掩码 |
| 输入 dtype | float16 / bfloat16 | cases.csv 两者皆覆盖；query / key / value 三者 dtype 必须一致 |

约束：
- `N_q % N_kv == 0`，分组比 `G = N_q / N_kv`
- `is_causal=True` 时要求 `S <= S_kv`，否则部分 query 行被全屏蔽产生 NaN
- query / key / value 三者 dtype 必须一致，且最后一维 `D` 三者相同

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


def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """BSND GQA; bottom-right causal alignment; output has query dtype.

    Half/bfloat16 values are already quantized by the generator. Promote them to
    FP32 for CPU matmul/softmax; preserve FP64 input for the judge's oracle path.
    Query tiling changes workspace, never the softmax/key reduction domain.
    """
    dtype = query.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    q, k, v = (t.to(work_dtype) for t in (query, key, value))
    B, S, Nq, D = q.shape
    Sk, Nk = k.shape[1:3]
    if Nq % Nk:
        raise ValueError("N_q must be divisible by N_kv")
    if is_causal and S > Sk:
        raise ValueError("causal GQA requires S <= S_kv")
    scale = 1.0 / D**0.5 if scaleValue <= 0 else scaleValue
    G = Nq // Nk
    out = torch.empty((B, S, Nq, D), dtype=work_dtype, device=q.device)
    # No expanded K/V. Bound scores to ~4M entries even for large batches.
    batch_tile = min(B, max(1, (4 * 1024**2) // max(1, Nq * Sk)))
    for b0 in range(0, B, batch_tile):
        b1 = min(B, b0 + batch_tile)
        nb = b1 - b0
        q_step = max(1, min(S, (4 * 1024**2) // (nb * Nq * Sk)))
        kb = k[b0:b1].permute(0, 2, 1, 3)
        vb = v[b0:b1].permute(0, 2, 1, 3)
        for i0 in range(0, S, q_step):
            i1 = min(S, i0 + q_step)
            si = i1 - i0
            qb = (
                q[b0:b1, i0:i1]
                .reshape(nb, si, Nk, G, D)
                .permute(0, 2, 3, 1, 4)
                .reshape(nb, Nk, G * si, D)
            )
            scores = torch.matmul(qb, kb.transpose(-1, -2))
            scores.mul_(scale)
            scores = scores.reshape(nb, Nk, G, si, Sk)
            if is_causal:
                rows = torch.arange(i0, i1, device=q.device).unsqueeze(-1)
                cols = torch.arange(Sk, device=q.device).unsqueeze(0)
                scores.masked_fill_(cols > rows + (Sk - S), float("-inf"))
            weights = torch.nn.functional.softmax(scores, dim=-1).reshape(
                nb, Nk, G * si, Sk
            )
            block = torch.matmul(weights, vb)
            out[b0:b1, i0:i1] = (
                block.reshape(nb, Nk, G, si, D)
                .permute(0, 3, 1, 2, 4)
                .reshape(nb, si, Nq, D)
            )
    return out.to(dtype)


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


def gqa_bench(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Independent input-dtype reference; not the high-precision oracle."""
    scale = query.shape[-1] ** -0.5 if scaleValue <= 0 else scaleValue
    return _native_attention(query, key, value, scale, is_causal)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv, D = 2, 128, 128, 128
N_q, N_kv = 32, 8
query = torch.randn(B, S, N_q, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N_kv, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N_kv, D, dtype=torch.float16, device="npu")
y = cann_bench.gqa(query, key, value, scaleValue=-1.0, is_causal=False)
```
