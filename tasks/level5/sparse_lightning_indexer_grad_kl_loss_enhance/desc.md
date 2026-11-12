# SparseLightningIndexerGradKLLossEnhance 算子 API 描述

## 1. 算子简介

SparseLightningIndexerGradKLLossEnhance 是 LightningIndexerEnhance 的训练反向算子，并融合 KL loss 计算。算子使用真实 attention 分布作为目标分布，使用 indexer 分支输出的 importance score 作为预测分布，计算 KL loss 以及 `query_index`、`key_index`、`weights` 的梯度。

**主要应用场景**：
- 训练 LightningIndexer，使稀疏索引分布逼近 attention 分布
- 长上下文稀疏 attention 的索引器蒸馏/校准
- 基于 top-k sparse index 的 KL loss 融合反向

**算子特征**：
- 难度等级：L5（FusedComposite）
- 默认布局为 `layout="TND"`，同时保留 `BSND` 布局语义
- 支持 float16 / bfloat16 输入
- 输出 `dquery_index`、`dkey_index`、`dweights` 和 `loss`

## 2. 算子定义

### 数学公式

目标 attention 分布由 `query` 和 `key` 计算：

$$
p = Softmax(Mask(scale\_value \cdot QK^T))
$$

indexer 分支 importance score 为：

$$
I = ReLU(Q_{index}K_{index}^T) \times W
$$

预测分布为：

$$
\hat{p} = Softmax(I)
$$

融合 KL loss：

$$
loss = \sum p \cdot (\log p - \log \hat{p})
$$

反向输出：

$$
dQ_{index},\quad dK_{index},\quad dW,\quad loss
$$

### 步骤说明

1. 按 `layout` 和 actual sequence 字段还原 query/key 以及 indexer 分支输入。
2. 根据 `topk_index` 选择参与 KL 计算的 key token；无效索引会被忽略。
3. 使用 `query` / `key` 计算目标 attention distribution `p`。
4. 使用 `query_index` / `key_index` / `weights` 计算 indexer 预测分布 `Softmax(I)`。
5. 对每个 query token 和 KV head 计算 KL loss，并反向累加 `dquery_index`、`dkey_index`、`dweights`。
6. 当 `sparse_mode=3` 时，对目标 attention 分布施加 rightDownCausal mask。

## 3. 接口规范

### 算子原型

```python
cann_bench.sparse_lightning_indexer_grad_kl_loss_enhance(
    Tensor query,
    Tensor key,
    Tensor query_index,
    Tensor key_index,
    Tensor weights,
    Tensor topk_index,
    Tensor? softmax_max=None,
    Tensor? softmax_sum=None,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    float scale_value=1.0,
    str layout="TND",
    Tensor|list|None actual_seq_lengths_query=None,
    Tensor|list|None actual_seq_lengths_key=None,
    int sparse_mode=3,
    bool deterministic=True,
    int sparse_block_size=1,
) -> (Tensor dquery_index, Tensor dkey_index, Tensor dweights, Tensor loss)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `TND: [Tq,Nq,D]`；`BSND: [B,Sq,Nq,D]` | 用于构造目标 attention 分布的 Query |
| key | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,D]`；`BSND: [B,Skv,Nkv,D]` | 用于构造目标 attention 分布的 Key |
| query_index | Tensor | 必选 | float16 / bfloat16 | `TND: [Tq,Nqi,Di]`；`BSND: [B,Sq,Nqi,Di]` | indexer 分支 Query 表征 |
| key_index | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,Di]`；`BSND: [B,Skv,Nkv,Di]` | indexer 分支 Key 表征 |
| weights | Tensor | 必选 | float16 / bfloat16 | `[Tq,Nqi]` 或 `[B,Sq,Nqi]` | indexer 分支权重 |
| topk_index | Tensor | 必选 | int32 | `[Tq,Nkv,K]` 或 `[B,Sq,Nkv,K]` | 参与 KL loss 的稀疏 key 索引 |
| softmax_max | Tensor? | 默认 `None` | float32 | 与正向统计输出一致 | 正向 softmax max 保留字段，标准 Golden 不使用 |
| softmax_sum | Tensor? | 默认 `None` | float32 | 与正向统计输出一致 | 正向 softmax sum 保留字段，标准 Golden 不使用 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 预留 | query rope 保留字段，标准 Golden 不使用 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 预留 | key rope 保留字段，标准 Golden 不使用 |
| scale_value | float | 默认 `1.0` | - | 标量 | 目标 attention QK 缩放系数 |
| layout | str | 默认 `TND` | - | - | 输入布局，支持 `TND` 和 `BSND` |
| actual_seq_lengths_query | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时可为前缀和 |
| actual_seq_lengths_key | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | key 实际长度；TND 时可为前缀和 |
| sparse_mode | int | 默认 `3` | - | 标量 | 0 为不加 causal mask，3 为 rightDownCausal |
| deterministic | bool | 默认 `True` | - | 标量 | 确定性计算开关，标准 Golden 不分支处理 |
| sparse_block_size | int | 默认 `1` | - | 标量 | 稀疏 block 大小 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| dquery_index | float16 / bfloat16 | 与 query_index 一致 | `query_index` 梯度 |
| dkey_index | float16 / bfloat16 | 与 key_index 一致 | `key_index` 梯度 |
| dweights | float16 / bfloat16 | 与 weights 一致 | `weights` 梯度 |
| loss | float32 | `[1]` | 累加 KL loss |

### 规则与约束

- `query` / `key` 用于目标 attention distribution；`query_index` / `key_index` / `weights` 用于 indexer 预测分支。
- `topk_index` 中有效索引必须位于 key 实际长度范围内，无效索引为负数。
- `Nq` 和 `Nqi` 均须能被 `Nkv` 整除；目标 attention 与 indexer 各自按连续 head 分组，两个分支的分组大小可以不同。`query_index` 的 head 维需与 `weights` 一致。
- `sparse_mode=3` 的目标分支在所选 token 上使用 rightDownCausal mask：局部 key 位置 `j >= Skv-Sq+s+1` 的目标 score 设为 `-40000`；indexer 预测分支仍对全部有效候选计算 softmax。若一行所有有效候选均被 mask，其目标分布为均匀分布。
- 对每个 query token / KV head，分别计算每个目标 query head 的候选 softmax，再在该 KV head 对应的目标 query heads 上取平均得到 `p`。indexer 的 `ReLU(Q_index K_index^T) * W` 则先在对应 indexer heads 上求和，再取候选 softmax 得到 `p_hat`。
- 参考 KL 使用 `p_eps=max(p,1e-8)`、`p_hat_eps=max(p_hat,1e-8)`，累加 `p_eps*(log(p_eps)-log(p_hat_eps))`。梯度包含此 clamp 的导数；`p_hat<1e-8` 时不能直接使用未截断 KL 的 `p_hat-p` 简化式。ReLU 在零点的导数为 0。
- 重复有效索引视为重复候选，其 softmax 分母、loss 和 key 梯度均按每次出现计算，`dkey_index` 必须累加全部贡献。负索引不参与 softmax、loss 和梯度；全无效行贡献为 0。
- `sparse_block_size` 当前支持值为 `1`（token 索引）。不对 block 索引做隐式展开。
- TND 的 actual sequence 字段支持长度列表或以总 token 数结束的非递减前缀和；必须精确覆盖对应输入 token，query/key 的段数必须一致。BSND 使用每个 batch 的实际长度，未使用的 padding 输出梯度为 0。
- `get_input` 会将原始 topk 占位输入替换成公开、确定性的分段非连续 token 选择，每行有效 key 索引保持唯一；覆盖 causal 未来候选、负哨兵和 padding。该构造只生成输入，不改变算子对一般合法 topk 的定义。
- `softmax_max`、`softmax_sum`、`query_rope`、`key_rope` 是接口保留字段，标准 Golden 不使用。

## 4. 精度要求

采用生态算子精度标准进行验证。

**误差指标**：

1. 平均相对误差（MERE）

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| Threshold | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

标准 Golden 参考实现位于同目录 `golden.py` 的 `sparse_lightning_indexer_grad_kl_loss_enhance` 函数。CPU 同精度参考先保留 fp16/bf16 输入量化值，再升为 fp32 进行矩阵、softmax、loss 和解析反向计算；不使用 CPU 原生半精度算术。TND 按实际段处理，query 分块限制工作内存。框架传入 fp64 输入时使用真正的 fp64 oracle。Golden 的参考输出保留计算精度，选手接口的前三个梯度输出仍按任务 dtype 返回，loss 为 fp32。

MERE/MARE 比较时，先将 oracle 截断到各实际输出 dtype，再将两侧升为 fp32 计算误差，保留上述阈值；不会通过放宽容差处理成本问题。

## 6. 额外信息

### 算子调用示例

```python

from typing import Optional, Sequence

import torch


def _ceil(value):
    ivalue = int(value)
    return ivalue if float(value) == float(ivalue) else ivalue + 1


class _Math:
    @staticmethod
    def ceil(value):
        return _ceil(value)


math = _Math()
F = torch.nn.functional


class _NumpyAlias:
    ndarray = torch.Tensor
    dtype = torch.dtype
    float32 = torch.float32
    float64 = torch.float64
    int32 = torch.int32
    int64 = torch.int64
    inf = float("inf")


np = _NumpyAlias()


def _default_lengths(
    total: Optional[int], batch: Optional[int], default_mode: Optional[str]
):
    if total is None:
        return []
    batch = max(int(batch or 1), 1)
    total = max(int(total), 0)
    if default_mode is None or default_mode == "split_total":
        base = total // batch
        out = [base] * batch
        for idx in range(total - base * batch):
            out[idx] += 1
        return out
    if default_mode == "per_batch_full":
        return [total] * batch
    raise ValueError(f"unsupported default_mode: {default_mode}")


def as_int_list(
    value,
    total: Optional[int] = None,
    batch: Optional[int] = None,
    is_tnd: bool = True,
    *,
    default_mode: Optional[str] = None,
    allow_scalar: bool = True,
    empty_uses_default: bool = True,
):
    if value is None:
        return (
            _default_lengths(total, batch, default_mode) if empty_uses_default else []
        )
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif allow_scalar and isinstance(value, (int, float)):
        value = [value]
    out = [int(x) for x in value]
    if not out:
        return (
            _default_lengths(total, batch, default_mode) if empty_uses_default else []
        )
    if not is_tnd or total is None:
        return out
    if all(a <= b for a, b in zip(out, out[1:])) and out[-1] == int(total):
        prev = 0
        lengths = []
        for item in out:
            lengths.append(max(int(item) - prev, 0))
            prev = int(item)
        return lengths
    if sum(out) == int(total):
        return out
    positive = [max(x, 0) for x in out]
    if sum(positive) == 0:
        positive = [1] * len(out)
    raw = [int(total) * x / sum(positive) for x in positive]
    lengths = [int(x) for x in raw]
    for idx in sorted(range(len(raw)), key=lambda i: raw[i] - lengths[i], reverse=True)[
        : int(total) - sum(lengths)
    ]:
        lengths[idx] += 1
    return lengths


def prefix_from_lengths(lengths: Sequence[int]):
    out = []
    total = 0
    for item in lengths:
        total += int(item)
        out.append(total)
    return out


def starts_from_lengths(lengths: Sequence[int]):
    starts = []
    cur = 0
    for item in lengths:
        starts.append(cur)
        cur += int(item)
    return starts


def page_to_bsnd(
    x: torch.Tensor, block_table: Optional[torch.Tensor], lengths: Sequence[int]
):
    x = x.detach().cpu()
    if block_table is None:
        return x
    block_num, block_size, n, d = x.shape
    table = block_table.detach().cpu().to(torch.int64)
    max_len = max([int(v) for v in lengths], default=0)
    out = torch.zeros((len(lengths), max_len, n, d), dtype=x.dtype)
    for b, seq_len in enumerate(lengths):
        if b >= table.shape[0]:
            continue
        for block_idx in range(min(math.ceil(seq_len / block_size), table.shape[1])):
            src = int(table[b, block_idx]) % max(block_num, 1)
            begin = block_idx * block_size
            end = min(begin + block_size, int(seq_len), max_len)
            if end > begin:
                out[b, begin:end] = x[src, : end - begin]
    return out


def page_to_bsnd_device(
    x: torch.Tensor, block_table: Optional[torch.Tensor], lengths: Sequence[int]
):
    if block_table is None:
        return x
    block_num, block_size, n, d = x.shape
    table = block_table.detach().to(torch.int64).to(x.device)
    max_len = max([int(v) for v in lengths], default=0)
    out = torch.zeros((len(lengths), max_len, n, d), dtype=x.dtype, device=x.device)
    for b, seq_len in enumerate(lengths):
        if b >= table.shape[0]:
            continue
        for block_idx in range(min(math.ceil(seq_len / block_size), table.shape[1])):
            src = int(table[b, block_idx].item()) % max(block_num, 1)
            begin = block_idx * block_size
            end = min(begin + block_size, int(seq_len), max_len)
            if end > begin:
                out[b, begin:end] = x[src, : end - begin]
    return out


def _to_bsnd_core(
    x: torch.Tensor, layout: str, lengths: Sequence[int], preserve_grad: bool
):
    x = x.cpu() if preserve_grad and x.requires_grad else x.detach().cpu()
    if layout == "TND":
        max_len = max(lengths, default=0)
        if preserve_grad and x.requires_grad:
            parts = []
            start = 0
            for seq_len in lengths:
                part = x[start : start + seq_len]
                start += seq_len
                if int(seq_len) < max_len:
                    pad_shape = (max_len - int(seq_len), *part.shape[1:])
                    part = torch.cat([part, part.new_zeros(pad_shape)], dim=0)
                parts.append(part)
            return (
                torch.stack(parts, dim=0)
                if parts
                else x.new_empty((0, max_len, x.shape[-2], x.shape[-1]))
            )
        out = torch.zeros(
            (len(lengths), max_len, x.shape[-2], x.shape[-1]), dtype=x.dtype
        )
        start = 0
        for b, seq_len in enumerate(lengths):
            out[b, :seq_len] = x[start : start + seq_len]
            start += seq_len
        return out
    return x


def to_bsnd_device(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    if layout == "TND":
        max_len = max(lengths, default=0)
        out = torch.zeros(
            (len(lengths), max_len, x.shape[-2], x.shape[-1]),
            dtype=x.dtype,
            device=x.device,
        )
        start = 0
        for b, seq_len in enumerate(lengths):
            out[b, :seq_len] = x[start : start + seq_len]
            start += seq_len
        return out
    return x


def to_bsnd_detached(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    return _to_bsnd_core(x, layout, lengths, preserve_grad=False)


def to_bsnd_grad(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    return _to_bsnd_core(x, layout, lengths, preserve_grad=True)


def to_bsnd(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    return to_bsnd_grad(x, layout, lengths)


def from_bsnd(x: torch.Tensor, layout: str, lengths: Sequence[int]):
    if layout == "TND":
        parts = [x[b, : int(seq_len)] for b, seq_len in enumerate(lengths)]
        return (
            torch.cat(parts, dim=0)
            if parts
            else x.new_empty((0, x.shape[-2], x.shape[-1]))
        )
    return x


def broadcast_kv(x: torch.Tensor, n1: int):
    if x.shape[-2] == n1:
        return x
    group = max(n1 // x.shape[-2], 1)
    return x.repeat_interleave(group, dim=-2)


def slice_optional(x: Optional[torch.Tensor], b: int, q_len: int, kv_len: int):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() <= 1:
        return None
    x = x.detach().cpu()
    if x.dim() == 2:
        return x[-q_len:, -kv_len:]
    if x.dim() >= 4:
        b_idx = min(b, x.shape[0] - 1)
        return x[b_idx : b_idx + 1, :, -q_len:, -kv_len:]
    return x


def slice_optional_device(x: Optional[torch.Tensor], b: int, q_len: int, kv_len: int):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() <= 1:
        return None
    if x.dim() == 2:
        return x[-q_len:, -kv_len:]
    if x.dim() >= 4:
        b_idx = min(b, x.shape[0] - 1)
        return x[b_idx : b_idx + 1, :, -q_len:, -kv_len:]
    return x


_slice_optional = slice_optional
_slice_optional_device = slice_optional_device


def sparse_lightning_kl_grad(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_index,
    scale_value=1.0,
    layout="TND",
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    sparse_mode=3,
):
    with torch.enable_grad():
        return _sparse_lightning_kl_grad_impl(
            query,
            key,
            query_index,
            key_index,
            weights,
            topk_index,
            scale_value=scale_value,
            layout=layout,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            sparse_mode=sparse_mode,
            _calc_dtype=torch.float64
            if query.dtype == torch.float64
            else torch.float32,
        )


def _checked_segment_lengths(value, total, batch=None, layout="TND"):
    """Read supported prefix/length forms; never silently rescale bad cases."""
    if value is None:
        return [int(total)] * int(batch or 1)
    values = (
        value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, torch.Tensor)
        else list(value)
    )
    values = [int(v) for v in values]
    if not values or any(v < 0 for v in values):
        raise ValueError("actual sequence lengths must be nonnegative and nonempty")
    if layout == "TND":
        if values[-1] == int(total) and all(a <= b for a, b in zip(values, values[1:])):
            values = [values[0]] + [b - a for a, b in zip(values, values[1:])]
        if sum(values) != int(total):
            raise ValueError("actual sequence lengths must sum to the TND token count")
    elif batch is not None and (
        len(values) != int(batch) or any(v > int(total) for v in values)
    ):
        raise ValueError("BSND lengths must contain one in-bounds length per batch")
    return values


def _sparse_lightning_kl_grad_impl(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_index,
    scale_value=1.0,
    layout="TND",
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    sparse_mode=3,
    _calc_dtype=torch.float32,
):
    """Bounded CPU reference; analytic derivative of the explicitly clamped KL.

    Work on unpadded segments and bounded query chunks. In particular, a ragged
    64-sequence TND input must not allocate B * max_q or B * max_k buffers.
    Half inputs retain their quantization and are promoted before arithmetic.
    The private dtype parameter exists for the independent fp64 audit only.
    """
    if layout not in ("TND", "BSND") or int(sparse_mode) not in (0, 3):
        raise ValueError("supported layouts are TND/BSND and sparse modes are 0/3")
    q_lens = _checked_segment_lengths(
        actual_seq_lengths_query,
        query.shape[0] if layout == "TND" else query.shape[1],
        query.shape[0] if layout == "BSND" else None,
        layout,
    )
    k_lens = _checked_segment_lengths(
        actual_seq_lengths_key,
        key.shape[0] if layout == "TND" else key.shape[1],
        key.shape[0] if layout == "BSND" else None,
        layout,
    )
    if len(q_lens) != len(k_lens):
        raise ValueError("query/key must describe the same number of sequences")
    nq, nk, ni = query.shape[-2], key.shape[-2], query_index.shape[-2]
    if nq % nk or ni % nk or weights.shape[-1] != ni:
        raise ValueError("query and indexer heads must be divisible by KV heads")
    q_group, i_group = nq // nk, ni // nk
    dqi = torch.zeros(query_index.shape, dtype=_calc_dtype)
    dki = torch.zeros(key_index.shape, dtype=_calc_dtype)
    dw = torch.zeros(weights.shape, dtype=_calc_dtype)
    loss = torch.zeros(1, dtype=_calc_dtype)
    q_start = k_start = 0
    # This bound includes target/indexer score and gathered key intermediates.
    max_work_elements = 4 * 1024 * 1024
    with torch.no_grad():
        for b, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
            if layout == "TND":
                qs, ks = (
                    slice(q_start, q_start + q_len),
                    slice(k_start, k_start + k_len),
                )
                q, k = query[qs], key[ks]
                qi, ki, w, idx = (
                    query_index[qs],
                    key_index[ks],
                    weights[qs],
                    topk_index[qs],
                )
                oq, ok, ow = dqi[qs], dki[ks], dw[qs]
            else:
                q, k = query[b, :q_len], key[b, :k_len]
                qi, ki, w, idx = (
                    query_index[b, :q_len],
                    key_index[b, :k_len],
                    weights[b, :q_len],
                    topk_index[b, :q_len],
                )
                oq, ok, ow = dqi[b, :q_len], dki[b, :k_len], dw[b, :q_len]
            q_start += q_len
            k_start += k_len
            if not q_len or not k_len or not idx.shape[-1]:
                continue
            k = k.detach().cpu().to(_calc_dtype)
            ki = ki.detach().cpu().to(_calc_dtype)
            for h in range(nk):
                head_indices = idx[:, h].detach().cpu().long()
                # Storage K can exceed every sequence length (e.g. K2048 for
                # 512-token packed segments). Entirely invalid columns have no
                # arithmetic or gradients; do not multiply their padding keys.
                used_columns = ((head_indices >= 0) & (head_indices < k_len)).any(dim=0)
                head_indices = head_indices[:, used_columns]
                slots = int(head_indices.shape[-1])
                if not slots:
                    continue
                per_row = slots * (
                    k.shape[-1] + ki.shape[-1] + 3 * q_group + 5 * i_group + 8
                )
                chunk = max(1, min(64, max_work_elements // max(per_row, 1)))
                qh = slice(h * q_group, (h + 1) * q_group)
                ih = slice(h * i_group, (h + 1) * i_group)
                for start in range(0, q_len, chunk):
                    end = min(start + chunk, q_len)
                    raw_idx = head_indices[start:end]
                    valid = (raw_idx >= 0) & (raw_idx < k_len)
                    nonempty = valid.any(dim=-1, keepdim=True)
                    safe_idx = raw_idx.clamp(0, k_len - 1)
                    selected_k = k[:, h][safe_idx]
                    q_part = q[start:end, qh].detach().cpu().to(_calc_dtype)
                    score = torch.bmm(q_part, selected_k.transpose(1, 2)) * float(
                        scale_value
                    )
                    if int(sparse_mode) == 3:
                        threshold = (
                            k_len - q_len + torch.arange(start, end).view(-1, 1) + 1
                        )
                        score.masked_fill_(
                            (safe_idx >= threshold).unsqueeze(1), -40000.0
                        )
                    score.masked_fill_(~valid.unsqueeze(1), -float("inf"))
                    # Empty sparse rows contribute exactly zero, without NaN intermediates.
                    score = torch.where(
                        nonempty.unsqueeze(1), score, torch.zeros_like(score)
                    )
                    p = torch.softmax(score, dim=-1).mean(dim=1) * valid
                    qi_part = qi[start:end, ih].detach().cpu().to(_calc_dtype)
                    ki_part = ki[:, h][safe_idx]
                    w_part = w[start:end, ih].detach().cpu().to(_calc_dtype)
                    dots = torch.bmm(qi_part, ki_part.transpose(1, 2))
                    positive = dots.clamp_min(0)
                    importance = (positive * w_part.unsqueeze(-1)).sum(dim=1)
                    importance.masked_fill_(~valid, -float("inf"))
                    importance = torch.where(
                        nonempty, importance, torch.zeros_like(importance)
                    )
                    pred = torch.softmax(importance, dim=-1) * valid
                    p_safe, pred_safe = p.clamp_min(1e-8), pred.clamp_min(1e-8)
                    loss += (
                        (p_safe * (p_safe.log() - pred_safe.log()) * valid)
                        .sum()
                        .reshape(1)
                    )
                    # clamp_min has zero derivative below eps. pred-p is not the
                    # derivative when a probability has been clamped.
                    active_target = p_safe * (pred >= 1e-8) * valid
                    ds = pred * active_target.sum(dim=-1, keepdim=True) - active_target
                    ow[start:end, ih] += (positive * ds.unsqueeze(1)).sum(dim=-1)
                    ddot = ds.unsqueeze(1) * w_part.unsqueeze(-1) * (dots > 0)
                    oq[start:end, ih] += torch.bmm(ddot, ki_part)
                    key_contrib = torch.bmm(ddot.transpose(1, 2), qi_part)
                    # index_add_ is required: advanced-index += loses repeated-key
                    # contributions both within one row and across query rows.
                    ok[:, h].index_add_(0, safe_idx[valid], key_contrib[valid])
    return dqi, dki, dw, loss


def sparse_lightning_indexer_grad_kl_loss_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    topk_index: torch.Tensor,
    softmax_max: torch.Tensor = None,
    softmax_sum: torch.Tensor = None,
    query_rope: torch.Tensor = None,
    key_rope: torch.Tensor = None,
    scale_value: float = 1.0,
    layout: str = "TND",
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    sparse_mode: int = 3,
    deterministic: bool = True,
    sparse_block_size: int = 1,
):
    del softmax_max, softmax_sum, query_rope, key_rope, deterministic, sparse_block_size
    return sparse_lightning_kl_grad(
        query,
        key,
        query_index,
        key_index,
        weights,
        topk_index,
        scale_value=scale_value,
        layout=layout,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        sparse_mode=sparse_mode,
    )


def _prefix_or_lengths(value, total):
    if value is None:
        return [int(total)]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (int, float)):
        value = [int(value)]
    out = [int(v) for v in value]
    if not out:
        return [int(total)]
    if (
        len(out) > 1
        and all(a <= b for a, b in zip(out, out[1:]))
        and out[-1] == int(total)
    ):
        prev = 0
        lens = []
        for item in out:
            lens.append(item - prev)
            prev = item
        return lens
    return out


def _build_st_sparse_indices(
    topk_index, q_lens, k_lens, sparse_mode, sparse_block_size
):
    """Deterministic, noncontiguous token coverage, including causal future keys.

    K is a storage bound, not permission to prune the causal mask from the test.
    Valid key indices remain unique within each row. Negative sentinels and
    padding exercise invalid-index handling. The pattern is public and
    identical across devices.
    """
    if topk_index is None:
        return None
    if int(sparse_block_size) != 1:
        raise ValueError(
            "the current task supports token indices (sparse_block_size=1)"
        )
    out = torch.full_like(topk_index, -1)
    topk, n2 = int(out.shape[-1]), int(out.shape[-2])
    token = 0
    for b, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        count = min(topk, int(k_len))
        if count:
            base = torch.arange(count, device=out.device, dtype=torch.int64)
            if count > 1:
                base = base * (int(k_len) - 1) // (count - 1)
            for s in range(int(q_len)):
                row = (base + s) % int(k_len)
                # Preserve unique valid keys; occasionally insert a negative sentinel.
                if count >= 4 and (s + b) % 11 == 7:
                    row[-1] = -1
                if out.dim() == 3:
                    out[token, :, :count] = row.to(out.dtype).view(1, -1).expand(n2, -1)
                else:
                    out[b, s, :, :count] = row.to(out.dtype).view(1, -1).expand(n2, -1)
                token += 1
        else:
            token += int(q_len)
    return out


def _to_bsnd_for_stats(x, layout, lens):
    if layout == "TND":
        return to_bsnd_grad(x, layout, lens).to(torch.float32)
    return x.detach().cpu().to(torch.float32)


def _zero_rope_like(x, rope_dim=64):
    if x is None:
        return None
    shape = list(x.shape)
    if not shape:
        return None
    shape[-1] = int(rope_dim)
    return torch.zeros(shape, dtype=x.dtype, device=x.device)


def _cat_rope_for_stats(x, rope):
    if x is None or rope is None:
        return x
    return torch.cat((x, rope.to(device=x.device, dtype=x.dtype)), dim=-1)


def _build_st_softmax_stats(
    query,
    key,
    topk_index,
    layout,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    scale_value,
    sparse_block_size,
    query_rope=None,
    key_rope=None,
):
    if query is None or key is None or topk_index is None:
        return None, None
    query = _cat_rope_for_stats(query, query_rope)
    key = _cat_rope_for_stats(key, key_rope)
    q_total = query.shape[0] if layout == "TND" else query.shape[1]
    k_total = key.shape[0] if layout == "TND" else key.shape[1]
    q_lens = _prefix_or_lengths(actual_seq_lengths_query, q_total)
    k_lens = _prefix_or_lengths(actual_seq_lengths_key, k_total)
    q_b = _to_bsnd_for_stats(query, layout, q_lens)
    k_b = _to_bsnd_for_stats(key, layout, k_lens)
    idx = topk_index.detach().cpu().to(torch.int64)
    if layout == "TND":
        idx = to_bsnd_grad(idx, layout, q_lens)
    if int(sparse_block_size) != 1:
        start = idx * int(sparse_block_size)
        offset = torch.arange(int(sparse_block_size), dtype=idx.dtype).view(
            1, 1, 1, 1, -1
        )
        idx = (start.unsqueeze(-1) + offset).reshape(
            *idx.shape[:-1], idx.shape[-1] * int(sparse_block_size)
        )

    bsz, max_q, n1, dim = q_b.shape
    n2 = k_b.shape[2]
    group = max(n1 // max(n2, 1), 1)
    max_values = []
    sum_values = []
    for b in range(bsz):
        q_len = int(q_lens[min(b, len(q_lens) - 1)])
        kv_len = int(k_lens[min(b, len(k_lens) - 1)])
        if q_len <= 0:
            continue
        q_cur = q_b[b, :q_len]
        k_cur = k_b[b, :kv_len]
        idx_cur = idx[b, :q_len]
        p_rows_max = []
        p_rows_sum = []
        for s in range(q_len):
            head_max = []
            head_sum = []
            for kv_head in range(n2):
                raw = idx_cur[s, kv_head].reshape(-1)
                safe = raw.clamp(min=0, max=max(kv_len - 1, 0))
                key_topk = k_cur[safe, kv_head]
                scores = torch.matmul(
                    q_cur[s, kv_head * group : (kv_head + 1) * group],
                    key_topk.transpose(-1, -2),
                )
                scores = scores * float(scale_value)
                invalid = (raw < 0) | (raw >= kv_len)
                if invalid.any():
                    scores[:, invalid] = -float("inf")
                score_max = torch.max(scores, dim=-1, keepdim=True).values
                score_sum = torch.exp(scores - score_max).sum(dim=-1, keepdim=True)
                head_max.append(score_max)
                head_sum.append(score_sum)
            p_rows_max.append(torch.cat(head_max, dim=0).squeeze(-1))
            p_rows_sum.append(torch.cat(head_sum, dim=0).squeeze(-1))
        max_values.append(torch.stack(p_rows_max, dim=0))
        sum_values.append(torch.stack(p_rows_sum, dim=0))
    if not max_values:
        empty = torch.empty((1, 0, n1), dtype=torch.float32)
        return empty, empty
    if layout == "TND":
        return torch.cat(max_values, dim=0).unsqueeze(0), torch.cat(
            sum_values, dim=0
        ).unsqueeze(0)
    return torch.stack(max_values, dim=0).unsqueeze(0).unsqueeze(0), torch.stack(
        sum_values, dim=0
    ).unsqueeze(0).unsqueeze(0)


def get_input(
    query: torch.Tensor = None,
    key: torch.Tensor = None,
    query_index: torch.Tensor = None,
    key_index: torch.Tensor = None,
    weights: torch.Tensor = None,
    topk_index: torch.Tensor = None,
    softmax_max: torch.Tensor = None,
    softmax_sum: torch.Tensor = None,
    query_rope: torch.Tensor = None,
    key_rope: torch.Tensor = None,
    **kwargs,
):
    layout = kwargs.get("layout", "TND")
    if query is not None and key is not None and topk_index is not None:
        q_total = query.shape[0] if layout == "TND" else query.shape[1]
        k_total = key.shape[0] if layout == "TND" else key.shape[1]
        q_lens = _checked_segment_lengths(
            kwargs.get("actual_seq_lengths_query"),
            q_total,
            query.shape[0] if layout == "BSND" else None,
            layout,
        )
        k_lens = _checked_segment_lengths(
            kwargs.get("actual_seq_lengths_key"),
            k_total,
            key.shape[0] if layout == "BSND" else None,
            layout,
        )
        if len(q_lens) != len(k_lens):
            raise ValueError("query/key must describe the same number of sequences")
        topk_index = _build_st_sparse_indices(
            topk_index,
            q_lens,
            k_lens,
            int(kwargs.get("sparse_mode", 3)),
            int(kwargs.get("sparse_block_size", 1)),
        )
    # These optional fields are reserved and unused by the task. Preserve None;
    # do not synthesize large rope/statistics buffers or pad ragged TND inputs.
    return [
        query,
        key,
        query_index,
        key_index,
        weights,
        topk_index,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
    ]
```
