# JaggedDenseFlashAttention 算子 API 描述

## 1. 算子简介

JaggedDenseFlashAttention 是 FBGEMM 推荐系统场景中的变长注意力前向：query / value 为变长（jagged）序列沿第 0 维拼接，key 为每个 batch 一份的 dense 矩阵。算子对每个 batch 计算 `Q_b @ K_b^T + bias`，并**沿 jagged 序列维**做 softmax，再与 value 相乘。

**主要应用场景**：
- 推荐系统中变长用户行为序列的 attention
- FBGEMM / DLRM 类模型的变长 attention 前向
- 变长序列替代「padding + mask + SDPA」

**算子特征**：
- 难度等级：L4（Attention）
- 五输入单输出：`q`、`k`、`v`、`attn_bias`、`jagged_offsets`（int32）→ `out`
- softmax 沿序列维（dim=0），而不是沿 key 的 slot 维
- 内部 `tl.dot` 走 fp32（allow_tf32=False）

## 2. 算子定义

### 数学公式

$$
L_b = \min(\text{offsets}[b+1] - \text{offsets}[b],\ \text{max\_seq\_len})
$$

$$
S_b = Q_b K_b^\top + \text{attn\_bias}[b, :L_b], \qquad
P_b = \operatorname{softmax}(S_b, \text{dim}=0)
$$

$$
\text{out}[b] = P_b^\top V_b \in \mathbb{R}^{T \times D}
$$

其中 $Q_b = q[\text{start}_b : \text{start}_b + L_b]$，$V_b$ 同理。$L_b = 0$ 的 batch 输出保持 0（对应 kernel 的 early return）。

实现建议：按 offsets 取出每条序列的前 $L_b$ 行，计算 `baddbmm(attn_bias[b], Q_b, K_b^T)`，把越界行 mask 成 $-\infty$ 后沿序列维 softmax，再与 $V_b$ 相乘。等长批（`sum_L == B * max_seq_len`）可省略 mask/散射。

## 3. 接口规范

### 算子原型

```python
cann_bench.jagged_dense_flash_attention(Tensor q, Tensor k, Tensor v, Tensor attn_bias, Tensor jagged_offsets, int max_seq_len) -> Tensor out
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q | Tensor | 必选 | 变长 query，shape = `[sum_L, D]` |
| k | Tensor | 必选 | dense key，shape = `[B, D, T]` |
| v | Tensor | 必选 | 变长 value，shape = `[sum_L, D]` |
| attn_bias | Tensor | 必选 | 每个 batch 的加性 bias，shape = `[B, max_seq_len, T]` |
| jagged_offsets | Tensor | 必选 | 各 batch 起止偏移，shape = `[B+1]`，int32，单调不减 |
| max_seq_len | int | 必选 | 每条序列参与计算的最大行数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| out | `[B, T, D]` | 与输入 q 相同 | 每个 batch 的 attention 输出 |

### 数据类型

| 输入 q / k / v / attn_bias dtype | jagged_offsets dtype | 输出 out dtype |
|----------------------------------|----------------------|----------------|
| float16 | int32 | float16 |
| float32 | int32 | float32 |
| bfloat16 | int32 | bfloat16 |

### 规则与约束

- `q`、`v`、`attn_bias` 的 batch 数必须与 `attn_bias.shape[0]`、`k.shape[0]`、`len(jagged_offsets)-1` 一致。
- `jagged_offsets` 长度必须为 `B+1`，元素单调不减，`jagged_offsets[0] == 0`，`jagged_offsets[-1] == sum_L`。
- 每条序列长度不超过 `max_seq_len`；超出部分不参与计算。
- `attn_bias.shape[1] == max_seq_len`。
- `D`、`T` 可为任意正整数（不要求 64 对齐）。
- 当 `sum_L == B * max_seq_len` 时走等长分支，否则走变长拼装分支。
- 计算在 fp32 中完成，输出回铸 `q.dtype`。

## 4. 标准 Golden 代码

```python
import torch

"""
JaggedDenseFlashAttention 算子 Torch Golden 参考实现

变长 query（jagged）× dense key/value 的 attention 前向：
    logits = Q_b @ K_b^T + attn_bias[b, :L_b]
    P_b    = softmax(logits, dim=0)          # 沿 jagged 序列维归一化
    out[b] = P_b^T @ V_b                     # [T, D]
全程 fp32（allow_tf32=False），最后回铸 q.dtype。

公式:
    L_b = min(jagged_offsets[b+1] - jagged_offsets[b], max_seq_len)
    S_b = Q_b @ K_b^T + attn_bias[b, :L_b]
    out[b] = softmax(S_b, dim=0)^T @ V_b
"""


def _build_offsets(seed_offsets: torch.Tensor, total: int, batch: int,
                   max_seq_len: int, device: torch.device) -> torch.Tensor:
    """把随机占位 offsets 规整为合法、单调不减的 int32 offsets。

    cases.yaml 的 value_range 无法表达 offsets 的单调性与长度上界，因此按
    (total, batch, max_seq_len) 重建：
      - total == batch * max_seq_len：等长批，offsets = [0, msl, 2*msl, ...]
      - 否则把 total 行按占位值分配的抖动长度分给 batch 段，每段长度 <= max_seq_len
    """
    if batch <= 0:
        return torch.zeros(1, dtype=torch.int32, device=device)
    if total == batch * max_seq_len:
        return (torch.arange(batch + 1, dtype=torch.int64, device=device) * max_seq_len).to(torch.int32)

    total_eff = min(total, batch * max_seq_len)
    weights = [int(v) % 1024 + 1 for v in seed_offsets.reshape(-1).to(torch.int64).abs().tolist()]
    if len(weights) < batch:
        weights.extend([1] * (batch - len(weights)))

    lengths = [0] * batch
    remaining = total_eff
    for i in range(batch):
        segs_left = batch - i - 1
        if total_eff >= batch:
            lo = max(1, remaining - segs_left * max_seq_len)
            hi = min(max_seq_len, remaining - segs_left)
        else:
            lo = max(0, remaining - segs_left * max_seq_len)
            hi = min(max_seq_len, remaining)
        take = lo if hi <= lo else lo + weights[i] % (hi - lo + 1)
        lengths[i] = take
        remaining -= take

    offsets = [0] * (batch + 1)
    acc = 0
    for i, length in enumerate(lengths):
        acc += length
        offsets[i + 1] = acc
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def jagged_dense_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 attn_bias: torch.Tensor, jagged_offsets: torch.Tensor,
                                 max_seq_len: int) -> torch.Tensor:
    """
    变长 query × dense key/value attention 前向

    公式:
        L_b = min(jagged_offsets[b+1] - jagged_offsets[b], max_seq_len)
        S_b = Q_b @ K_b^T + attn_bias[b, :L_b]
        out[b] = softmax(S_b, dim=0)^T @ V_b

    Args:
        q: 变长 query [sum_L, D]
        k: dense key [B, D, T]
        v: 变长 value [sum_L, D]
        attn_bias: 每个 batch 的加性 bias [B, max_seq_len, T]
        jagged_offsets: 各 batch 在 sum_L 维上的起止偏移 [B+1]，int32，单调不减
        max_seq_len: 每条序列参与计算的最大行数

    Returns:
        输出张量 [B, T, D]，dtype 与 q 相同
    """
    batch, dim, slots = k.shape
    total = q.shape[0]
    rows = max_seq_len

    if total == batch * rows:
        # 等长批：q / v 本身即 [B, rows, D] 连续布局
        q_pad = q.reshape(batch, rows, dim).to(torch.float32)
        v_pad = v.reshape(batch, rows, dim).to(torch.float32)
        logits = torch.baddbmm(attn_bias[:, :rows, :].to(torch.float32), q_pad, k.to(torch.float32))
        probs = torch.softmax(logits, dim=1)
        return torch.bmm(probs.transpose(1, 2), v_pad).to(q.dtype)

    pos = torch.arange(rows, dtype=torch.int64)
    lengths = (jagged_offsets[1:] - jagged_offsets[:-1]).to(torch.int64)
    valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
    idx = (jagged_offsets[:-1].reshape(batch, 1).to(torch.int64) + pos.reshape(1, rows)).clamp(min=0, max=total - 1)

    q_pad = torch.index_select(q, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, dim)
    v_pad = torch.index_select(v, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, dim)

    logits = torch.baddbmm(attn_bias[:, :rows, :].to(torch.float32), q_pad, k.to(torch.float32))
    logits = logits.masked_fill(~valid[:, :, None], float("-inf"))
    probs = torch.softmax(logits, dim=1)
    return torch.bmm(probs.transpose(1, 2), v_pad).to(q.dtype)


def get_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: torch.Tensor,
              jagged_offsets: torch.Tensor, max_seq_len: int = 1, **kwargs) -> list:
    """把随机生成的 offsets 规整为合法单调范围（cases.yaml 无法表达该约束）。"""
    batch = jagged_offsets.reshape(-1).numel() - 1
    total = q.shape[0]
    return [q, k, v, attn_bias, _build_offsets(jagged_offsets, total, batch, max_seq_len, q.device)]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

batch, dim, slots, max_seq_len = 8, 64, 32, 512
sum_l = batch * max_seq_len
q = torch.randn(sum_l, dim, dtype=torch.bfloat16, device="npu")
k = torch.randn(batch, dim, slots, dtype=torch.bfloat16, device="npu")
v = torch.rand(sum_l, dim, dtype=torch.bfloat16, device="npu")
attn_bias = torch.randn(batch, max_seq_len, slots, dtype=torch.bfloat16, device="npu")
offsets = torch.arange(0, sum_l + 1, max_seq_len, dtype=torch.int32, device="npu")
out = cann_bench.jagged_dense_flash_attention(q, k, v, attn_bias, offsets, max_seq_len)
```
