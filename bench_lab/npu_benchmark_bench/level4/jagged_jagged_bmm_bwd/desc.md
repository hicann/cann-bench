# JaggedJaggedBmmBwd 算子 API 描述

## 1. 算子简介

JaggedJaggedBmmBwd 是变长序列批量矩阵乘 `Z_b = X_b^T @ Y_b` 的反向算子。`x`、`y` 为前向的变长拼接矩阵，`x_offsets` 给出每条序列的起止位置，`grad_output` 为上游梯度 `dZ`；算子返回 `x`、`y` 的梯度，超出 `max_seq_len` 的行梯度为 0。

**主要应用场景**：
- FBGEMM 变长注意力反向中 `X^T @ Y` 的梯度计算
- 推荐系统变长序列 bilinear / 协方差统计的反向传播
- 变长序列场景下批量矩阵乘反向的融合实现

**算子特征**：
- 难度等级：L4（Contraction）
- 四输入双输出：`x`、`y`、`x_offsets`（int32）、`grad_output` → `(grad_x, grad_y)`
- 每条序列只在前 `max_seq_len` 行内计算
- 内部 `tl.dot` 语义：输入 dtype + fp32 累加

## 2. 算子定义

### 数学公式

$$
L_b = \min(\text{offsets}[b+1] - \text{offsets}[b],\ \text{max\_seq\_len})
$$

$$
dX_b = Y_b\, dZ_b^\top, \qquad dY_b = X_b\, dZ_b
$$

其中 $X_b = x[\text{start}_b : \text{start}_b + L_b]$，$Y_b$ 同理，$dZ_b = \text{grad\_output}[b]$。越界行（$i \ge L_b$）的 $dX$、$dY$ 均为 0。

实现建议：按 offsets 取出每条序列的前 $L_b$ 行，升 fp32 后分别做 `bmm(Y_b, dZ_b^T)` 与 `bmm(X_b, dZ_b)`，再把结果散射回原位置（越界行写 0）。等长批（`sum_L == B * max_seq_len`）可直接 reshape 为 `[B, max_seq_len, *]`。

## 3. 接口规范

### 算子原型

```python
cann_bench.jagged_jagged_bmm_bwd(Tensor x, Tensor y, Tensor x_offsets, Tensor grad_output, int max_seq_len) -> (Tensor grad_x, Tensor grad_y)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 前向左矩阵，shape = `[sum_L, D]` |
| y | Tensor | 必选 | 前向右矩阵，shape = `[sum_L, T]` |
| x_offsets | Tensor | 必选 | 各序列起止偏移，shape = `[B+1]`，int32，单调不减 |
| grad_output | Tensor | 必选 | 上游梯度 dZ，shape = `[B, D, T]` |
| max_seq_len | int | 必选 | 每条序列参与计算的最大行数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| grad_x | `[sum_L, D]` | 与输入 x 相同 | x 的梯度；超出 `max_seq_len` 的行保持 0 |
| grad_y | `[sum_L, T]` | 与输入 y 相同 | y 的梯度；超出 `max_seq_len` 的行保持 0 |

### 数据类型

| 输入 x / y / grad_output dtype | x_offsets dtype | 输出 grad_x / grad_y dtype |
|--------------------------------|-----------------|----------------------------|
| float16 | int32 | float16 |
| float32 | int32 | float32 |
| bfloat16 | int32 | bfloat16 |

### 规则与约束

- `x`、`y` 的第 0 维必须相同（`sum_L`），`grad_output` 为 `[B, D, T]`。
- `x_offsets` 长度必须为 `B+1`，元素单调不减，`x_offsets[0] == 0`，`x_offsets[-1] == sum_L`。
- 每条序列长度不超过 `max_seq_len`；超出部分梯度为 0。
- `D`、`T` 可为任意正整数（不要求 64 对齐）。
- 当 `sum_L == B * max_seq_len` 时走等长分支，否则走变长拼装分支。
- 计算在 fp32 中完成，输出回铸输入 dtype。

## 4. 标准 Golden 代码

```python
import torch

"""
JaggedJaggedBmmBwd 算子 Torch Golden 参考实现

变长序列批量矩阵乘 Z[b] = X_b^T @ Y_b 的反向：
    dX_b = Y_b @ dZ_b^T
    dY_b = X_b @ dZ_b
每条序列只取前 L_b = min(len, max_seq_len) 行参与，超出行梯度保持 0。
累加全程 fp32，最后回铸输入 dtype。

公式:
    L_b = min(offsets[b+1] - offsets[b], max_seq_len)
    dX_b = Y_b @ dZ_b^T
    dY_b = X_b @ dZ_b
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


def jagged_jagged_bmm_bwd(x: torch.Tensor, y: torch.Tensor, x_offsets: torch.Tensor,
                          grad_output: torch.Tensor, max_seq_len: int):
    """
    变长序列批量矩阵乘反向

    公式:
        L_b = min(offsets[b+1] - offsets[b], max_seq_len)
        dX_b = Y_b @ dZ_b^T      # [L_b, T] @ [T, D] -> [L_b, D]
        dY_b = X_b @ dZ_b        # [L_b, D] @ [D, T] -> [L_b, T]

    Args:
        x: 前向左矩阵 [sum_L, D]
        y: 前向右矩阵 [sum_L, T]
        x_offsets: 各序列起止偏移 [B+1]，int32，单调不减
        grad_output: 上游梯度 dZ [B, D, T]
        max_seq_len: 每条序列参与计算的最大行数

    Returns:
        (grad_x, grad_y)：分别为 x、y 的梯度，形状 [sum_L, D]、[sum_L, T]
    """
    total, dim = x.shape
    cols = y.shape[1]
    batch = x_offsets.shape[0] - 1
    rows = max_seq_len
    grad_z = grad_output.to(torch.float32)

    if total == batch * rows:
        # 等长批：无需 padding 与掩码
        x_pad = x.reshape(batch, rows, dim).to(torch.float32)
        y_pad = y.reshape(batch, rows, cols).to(torch.float32)
        grad_x = torch.bmm(y_pad, grad_z.transpose(1, 2)).reshape(total, dim)
        grad_y = torch.bmm(x_pad, grad_z).reshape(total, cols)
        return grad_x.to(x.dtype), grad_y.to(y.dtype)

    pos = torch.arange(rows, dtype=torch.int64)
    lengths = (x_offsets[1:] - x_offsets[:-1]).to(torch.int64)
    valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
    idx = (x_offsets[:-1].reshape(batch, 1).to(torch.int64) + pos.reshape(1, rows)).clamp(min=0, max=total - 1)

    x_pad = torch.index_select(x, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, dim)
    y_pad = torch.index_select(y, 0, idx.reshape(-1)).to(torch.float32).reshape(batch, rows, cols)
    grad_x = torch.bmm(y_pad, grad_z.transpose(1, 2))
    grad_y = torch.bmm(x_pad, grad_z)

    flat_idx = torch.where(valid.reshape(-1), idx.reshape(-1),
                           torch.full_like(idx.reshape(-1), total))
    buf_x = torch.zeros(total + 1, dim, dtype=torch.float32, device=x.device)
    buf_y = torch.zeros(total + 1, cols, dtype=torch.float32, device=x.device)
    buf_x.index_copy_(0, flat_idx, grad_x.reshape(-1, dim))
    buf_y.index_copy_(0, flat_idx, grad_y.reshape(-1, cols))
    return buf_x[:total].to(x.dtype), buf_y[:total].to(y.dtype)


def get_input(x: torch.Tensor, y: torch.Tensor, x_offsets: torch.Tensor,
              grad_output: torch.Tensor, max_seq_len: int = 1, **kwargs) -> list:
    """把随机生成的 offsets 规整为合法单调范围（cases.yaml 无法表达该约束）。"""
    batch = x_offsets.reshape(-1).numel() - 1
    total = x.shape[0]
    return [x, y, _build_offsets(x_offsets, total, batch, max_seq_len, x.device), grad_output]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

sum_l, dim, cols, batch, max_seq_len = 4096, 128, 64, 8, 512
x = torch.randn(sum_l, dim, dtype=torch.bfloat16, device="npu")
y = torch.rand(sum_l, cols, dtype=torch.bfloat16, device="npu")
x_offsets = torch.tensor([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
                         dtype=torch.int32, device="npu")
grad_output = torch.randn(batch, dim, cols, dtype=torch.bfloat16, device="npu")
grad_x, grad_y = cann_bench.jagged_jagged_bmm_bwd(x, y, x_offsets, grad_output, max_seq_len)
```
