# JaggedSoftmax 算子 API 描述

## 1. 算子简介

JaggedSoftmax 是面向推荐系统 / 变长序列注意力的变长（jagged）softmax。输入 `x` 由多条长度不等的序列沿第 0 维拼接而成，`x_offsets` 给出每条序列的起止位置；算子对每条序列的前 `max_seq_len` 行沿**序列维**做 softmax，超出行输出 0。

**主要应用场景**：
- FBGEMM 变长注意力中的打分归一化（jagged softmax）
- 推荐模型中变长特征序列的注意力权重计算
- 变长序列场景下替代「padding + mask + softmax」

**算子特征**：
- 难度等级：L4（NumericalStable）
- 双输入单输出：`x`、`x_offsets`（int32）→ `y`
- 归一化沿变长序列维（dim=0）进行，`H` 为并行的 head / 特征列
- 内部累加使用 fp32，输出回铸输入 dtype

## 2. 算子定义

### 数学公式

$$
N_b = \min(\text{offsets}[b+1] - \text{offsets}[b],\ \text{max\_seq\_len})
$$

$$
y[start_b + i, h] =
\begin{cases}
\dfrac{\exp(x[start_b + i, h] - m_b)}{\sum_{j=0}^{N_b-1}\exp(x[start_b + j, h] - m_b)}, & i < N_b \\[6pt]
0, & i \ge N_b
\end{cases}
$$

其中 $start_b = \text{offsets}[b]$，$m_b = \max_{j<N_b} x[start_b+j, h]$ 为数值稳定项。

实现建议：对每条序列取前 $N_b$ 行，沿序列维做 fp32 softmax；超出 $N_b$ 的行保持 0（对应 kernel 只写前 `max_seq_len` 行）。当 `sum_L == B * max_seq_len` 时各序列等长，可直接 reshape 为 `[B, max_seq_len, H]` 后 softmax。

## 3. 接口规范

### 算子原型

```python
cann_bench.jagged_softmax(Tensor x, Tensor x_offsets, int max_seq_len) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 变长序列拼接后的打分/特征，shape = `[sum_L, H]` |
| x_offsets | Tensor | 必选 | 各序列在 `sum_L` 维上的起止偏移，shape = `[B+1]`，int32，单调不减 |
| max_seq_len | int | 必选 | 每条序列参与归一化的最大行数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[sum_L, H]` | 与输入 x 相同 | softmax 结果；超出 `max_seq_len` 的行保持 0 |

### 数据类型

| 输入 x dtype | x_offsets dtype | 输出 y dtype |
|--------------|-----------------|--------------|
| float16 | int32 | float16 |
| float32 | int32 | float32 |
| bfloat16 | int32 | bfloat16 |

### 规则与约束

- `x` 为 2D ND 张量，`H` 可为任意正整数（不要求 64 对齐）。
- `x_offsets` 长度必须为 `B+1`，元素单调不减，`x_offsets[0] == 0`，`x_offsets[-1] == sum_L`。
- 每条序列长度 `x_offsets[b+1] - x_offsets[b]` 不超过 `max_seq_len`；超出部分不参与计算、输出为 0。
- `max_seq_len >= 1`。
- 当 `sum_L == B * max_seq_len` 时走等长分支（各序列长度均等于 `max_seq_len`），否则走变长拼装分支。
- 归一化在 fp32 中完成，输出回铸 `x.dtype`。

## 4. 标准 Golden 代码

```python
import torch

"""
JaggedSoftmax 算子 Torch Golden 参考实现

变长序列（jagged）softmax：对每条序列在拼接维上的前 max_seq_len 行沿序列维做归一化，
超出 max_seq_len 的行输出 0。累加全程 fp32，最后回铸输入 dtype。

公式:
    N_b = min(offsets[b+1] - offsets[b], max_seq_len)
    y[start_b + i, h] = exp(x[start_b + i, h] - m_b) / sum_{j=0}^{N_b-1} exp(x[start_b + j, h] - m_b),  i < N_b
    y[start_b + i, h] = 0,  i >= N_b
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


def jagged_softmax(x: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    变长序列 softmax

    公式:
        N_b = min(offsets[b+1] - offsets[b], max_seq_len)
        y[start_b + i, h] = softmax(x[start_b : start_b + N_b], dim=0)[i, h],  i < N_b
        y[start_b + i, h] = 0,  i >= N_b

    Args:
        x: 变长序列拼接后的打分/特征 [sum_L, H]
        x_offsets: 各序列起止偏移 [B+1]，int32，单调不减
        max_seq_len: 每条序列参与归一化的最大行数

    Returns:
        与 x 同形同 dtype 的 softmax 结果 [sum_L, H]
    """
    total, heads = x.shape
    batch = x_offsets.shape[0] - 1
    rows = max_seq_len

    if total == batch * rows:
        # 等长批：x 本身即 [B, rows, H] 连续布局
        padded = x.reshape(batch, rows, heads).to(torch.float32)
        return torch.softmax(padded, dim=1).reshape(total, heads).to(x.dtype)

    pos = torch.arange(rows, dtype=torch.int64)
    lengths = (x_offsets[1:] - x_offsets[:-1]).to(torch.int64)
    valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
    idx = (x_offsets[:-1].reshape(batch, 1).to(torch.int64) + pos.reshape(1, rows)).clamp(min=0, max=total - 1)

    padded = torch.index_select(x, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, heads)
    padded = padded.masked_fill(~valid[:, :, None], float("-inf"))
    probs = torch.softmax(padded, dim=1)

    flat_idx = torch.where(valid.reshape(-1), idx.reshape(-1),
                           torch.full_like(idx.reshape(-1), total))
    buf = torch.zeros(total + 1, heads, dtype=torch.float32, device=x.device)
    buf.index_copy_(0, flat_idx, probs.reshape(-1, heads))
    return buf[:total].to(x.dtype)


def get_input(x: torch.Tensor, x_offsets: torch.Tensor, max_seq_len: int = 1, **kwargs) -> list:
    """把随机生成的 offsets 规整为合法单调范围（cases.yaml 无法表达该约束）。"""
    batch = x_offsets.reshape(-1).numel() - 1
    total = x.shape[0]
    return [x, _build_offsets(x_offsets, total, batch, max_seq_len, x.device)]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

sum_l, heads, batch, max_seq_len = 4096, 64, 8, 512
x = torch.randn(sum_l, heads, dtype=torch.float16, device="npu")
x_offsets = torch.tensor([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
                         dtype=torch.int32, device="npu")
y = cann_bench.jagged_softmax(x, x_offsets, max_seq_len)
```
