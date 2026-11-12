# SparseFlashAttention 算子 API 描述

## 1. 算子简介

大序列长度推理场景的高效稀疏注意力计算算子。与标准 FlashAttention 对全部 KV 位置计算注意力不同，SparseFlashAttention 通过 `sparseIndices` 指定每个 query 位置需要关注的 KV 子集，仅对选中的 KV 位置执行缩放点积注意力，从而降低计算和内存开销。

支持 Grouped Query Attention（GQA）：query 头数 N1 可以大于或等于 KV 头数 N2（要求 N1 能被 N2 整除），每 N1/N2 个 query 头共享同一组 KV 头及其稀疏索引。同时支持 query/key 和 value 使用不同的 head dim（Dk 和 Dv）。

**主要应用场景**：
- 大语言模型长序列推理中的高效注意力计算
- 长文本理解与生成任务中降低注意力计算复杂度（从 O(S1×S2) 降至 O(S1×topK)）
- 需要稀疏注意力模式的 Transformer 推理加速
- GQA / MLA 架构下的稀疏注意力

**算子特征**：
- 难度等级：L4（FusedComposite）
- 四输入（query, key, value, sparseIndices）单输出
- 融合稀疏索引 gather、缩放点积注意力与 softmax 计算
- 支持 GQA（N1 ≥ N2，N1 % N2 == 0）和不同 head dim（Dk、Dv）
- 支持 BSND 和 BNSD 两种张量布局
- 支持 `is_causal` 因果掩码：在稀疏 gather 之上再叠加 [S1, S2] attention 矩阵的右下角对角线掩码，sparseIndices 选中但落在对角线之上（即 KV 序列位置满足 idx > s1 + (S2 - S1)）的条目在 softmax 前置 -inf
- **value 与 key 在数值上同源**（MLA 推理 latent KV cache 语义）：
  - 当 `Dk == Dv` 时，`value == key`（K 与 V 完全相同）
  - 当 `Dk > Dv`（典型 MLA: Dk=576=512_nope+64_rope, Dv=512），`value == key[..., :Dv]`（V 是 K 的前 Dv 维前缀切片）
  - 接口保留 value 为独立入参以兼容通用 attention API，调用方需保证此关系

## 2. 算子定义

### 稀疏机制

对于每个 KV 头组 `(b, n2, s1)`，`sparseIndices` 中对应位置给出该 query 需要关注的 topK 个 KV 序列位置索引（值域 `[0, S2)`）。该组内的 N1/N2 个 query 头共享相同的稀疏索引，attention 计算仅在这 topK 个选中的 KV 位置上进行。

`sparseIndices` 的前三维布局与 `inputLayout` 一致：
- BSND 布局：sparseIndices shape 为 `[B, S1, N2, topK]`
- BNSD 布局：sparseIndices shape 为 `[B, N2, S1, topK]`

### 数学公式

$$
K_{sel} = \text{gather}(K, \text{sparseIndices}) \quad V_{sel} = \text{gather}(V, \text{sparseIndices})
$$

$$
y = \text{softmax}\left(Q \times K_{sel}^T \times \text{scaleValue}\right) \times V_{sel}
$$

具体子步骤：
1. **稀疏 Gather**：根据 `sparseIndices` 从 key/value 中提取选中的 KV 子集
   - 以 BNSD 为例：$K_{sel}[b, n2, s1, i, :] = K[b, n2, \text{sparseIndices}[b, n2, s1, i], :]$，shape `[B, N2, S1, topK, Dk]`
   - $V_{sel}$ 同理，shape `[B, N2, S1, topK, Dv]`
2. **GQA 扩展**：将 `K_sel` 和 `V_sel` 沿 head 维度复制 G=N1/N2 次，扩展到 `[B, N1, S1, topK, ...]`
3. **缩放点积**：$\text{scores} = Q \cdot K_{sel}^T \times \text{scaleValue}$，shape `[B, N1, S1, topK]`
4. **因果掩码（可选）**：当 `is_causal=True` 时，对每个 query 位置 `s1` 和每个 topK 槽位 `i`，若 `sparseIndices[..., s1, i] > s1 + (S2 - S1)`（即所选 KV 位置落在 [S1, S2] attention 矩阵右下角对角线之上），则该位置 scores 置为 $-\infty$
5. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$，在 topK 维上归一化
6. **加权求和**：$y = \text{attn\_weights} \times V_{sel}$

### 布局说明

| inputLayout | query | key | value | sparseIndices | output |
|-------------|-------|-----|-------|---------------|--------|
| BSND | [B, S1, N1, Dk] | [B, S2, N2, Dk] | [B, S2, N2, Dv] | [B, S1, N2, topK] | [B, S1, N1, Dv] |
| BNSD | [B, N1, S1, Dk] | [B, N2, S2, Dk] | [B, N2, S2, Dv] | [B, N2, S1, topK] | [B, N1, S1, Dv] |

### 与标准 FlashAttention 的区别

| 项目 | FlashAttention | SparseFlashAttention |
|------|---------------|---------------------|
| 注意力范围 | 全部 S2 个 KV 位置 | sparseIndices 指定的 topK 个 KV 位置 |
| 计算复杂度 | O(S1 × S2 × D) | O(S1 × topK × D)，topK << S2 |
| 额外输入 | 无 | sparseIndices |
| softmax 范围 | S2 维 | topK 维（仅选中的 KV 位置） |
| GQA 支持 | 是 | 是（N1 % N2 == 0） |
| head dim | Q/K 共享 Dk，V 可用不同 Dv | 同左 |

## 3. 接口规范

### 算子原型

```python
sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparseIndices, float scaleValue, str inputLayout="BSND", bool is_causal=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| query | Tensor | 是 | float16/bfloat16 | BSND: [B, S1, N1, Dk]；BNSD: [B, N1, S1, Dk] | 查询张量 |
| key | Tensor | 是 | float16/bfloat16 | BSND: [B, S2, N2, Dk]；BNSD: [B, N2, S2, Dk] | 键张量，head dim 与 query 一致 |
| value | Tensor | 是 | float16/bfloat16 | BSND: [B, S2, N2, Dv]；BNSD: [B, N2, S2, Dv] | 值张量，head dim 可与 key 不同。语义上要求 `value == key[..., :Dv]`（同一份 latent KV cache 的前缀视图，Dk==Dv 时退化为 `value == key`） |
| sparseIndices | Tensor | 是 | int32 | BSND: [B, S1, N2, topK]；BNSD: [B, N2, S1, topK] | 按 KV 头分组的稀疏索引，前三维布局与 inputLayout 一致 |
| scaleValue | float | 是 | - | 标量 | 缩放因子，通常为 1/sqrt(Dk) |
| inputLayout | str | 否 | - | - | 张量布局格式，"BSND"（默认）或 "BNSD" |
| is_causal | bool | 否 | - | - | 是否启用因果掩码（右下角对齐），默认 False。True 时在稀疏 gather 之上额外屏蔽 `sparseIndices[..., s1, i] > s1 + (S2 - S1)` 的位置 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 query 一致 | BSND: [B, S1, N1, Dv]；BNSD: [B, N1, S1, Dv] | 稀疏注意力输出，布局与输入一致 |

### 数据类型

| query/key/value dtype | sparseIndices dtype | 输出 dtype |
|----------------------|--------------------|-----------| 
| bfloat16 | int32 | bfloat16 |
| float16 | int32 | float16 |

### 规则与约束

- query、key、value 的 dtype 必须一致
- query 和 key 的 head dim 必须一致（Dk），value 的 head dim（Dv）可以不同
- N1 必须能被 N2 整除（GQA 分组约束），N1 == N2 时退化为 MHA
- sparseIndices 的值域为 `[0, S2)`，即 KV 序列长度范围内的有效索引；每个 `(b, n2, s1)` 行内必须无重复
- topK（每个 query 关注的 KV 数量）满足 1 ≤ topK ≤ min(2048, S2)
- scaleValue 必须为有限正数，通常设置为 $1/\sqrt{Dk}$
- inputLayout 必须为 "BSND" 或 "BNSD"，所有张量（含 sparseIndices 和输出）的布局保持一致
- `is_causal=True` 时要求 $S1 \le S2$；若某个 query 行的全部 topK 槽位都被因果掩码屏蔽（极端情况），该行 softmax 输出按全 0 处理
- **语义约束 `value == key[..., :Dv]`**：value 必须等于 key 的前 Dv 维前缀（同一份 latent KV cache 的不同视图）。当 `Dk == Dv` 时退化为 `value == key`；当 `Dk == Dv + 64`（MLA 典型 Dk=576/Dv=512）时 value 即 key_nope。算子接口保留 value 为独立入参以兼容通用 attention API，调用方有责任保证此关系；本规范的 Golden 实现不强制检查

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 64 | 各 batch 独立计算 |
| `S1`（query 序列长度） | 1 ~ 2048 | 支持 decode、MTP 和 prefill 场景 |
| `S2`（KV 序列长度） | 1 ~ 16384 | 因果模式要求 S1 ≤ S2 |
| `N1`（query 头数） | 1 ~ 256 | 必须 N1 % N2 == 0 |
| `N2`（KV 头数） | 1 ~ 64 | 同组 query heads 共享 KV head；N1 == N2 时为 MHA |
| `Dk`（query/key head dim） | 64 对齐 | query 与 key 的最后一维必须一致 |
| `Dv`（value head dim） | 64 对齐，Dv ≤ Dk | value 等于 key 的前 Dv 维前缀 |
| `topK`（稀疏选中数量） | 1 ~ min(2048, S2) | 同一 query 对应的稀疏索引须互异 |
| `sparseIndices` 取值 | [0, S2) | int32，必须为 KV 序列长度范围内的有效索引 |
| `scaleValue` | 有限正浮点 | 通常设置为 $1/\sqrt{Dk}$ |
| `inputLayout` | {"BSND", "BNSD"} | cases.csv 实测两种均覆盖；所有张量（含 sparseIndices 和输出）的布局保持一致 |
| `is_causal` | {True, False} | cases.csv 实测两种均覆盖；True 时屏蔽 `sparseIndices[..., s1, i] > s1 + (S2 - S1)` 的位置 |
| dtype | float16 / bfloat16 | cases.csv 实测两种均覆盖；query/key/value dtype 必须一致；sparseIndices 固定 int32 |

约束：`N1 % N2 == 0`；`Dv ≤ Dk` 且语义上要求 `value == key[..., :Dv]`；`is_causal=True` 时要求 `S1 ≤ S2`；若某 query 行的全部 topK 槽位都被因果掩码屏蔽，该行 softmax 输出按全 0 处理。

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



Golden 数值路径：float16/bfloat16 输入的点积、softmax 和加权求和均以 float32 累积，最后转换为输入 dtype；FP64 oracle 接收转换后的 float64 输入，保持完整 float64 运算并返回 float64。BSND/BNSD 输出均连续。参考实现按 KV 头和 query tile gather，不分配完整 S1×S2 scores，也不复制 GQA 的 KV 头。get_input 对256个索引样本和各32个查询、键样本进行稳定散列，派生局部随机种子，不依赖全局RNG或用例编号；随机优先级块最多1,048,576个元素，topK>S2抛错。

## 5. 标准 Golden 代码

```python

import hashlib
import math

import torch


def _as_bnsd(query, key, value, sparse_indices, layout):
    if layout == "BSND":
        return tuple(x.permute(0, 2, 1, 3) for x in (query, key, value, sparse_indices))
    if layout == "BNSD":
        return query, key, value, sparse_indices
    raise ValueError("inputLayout must be BSND or BNSD")


def _validate(q, k, v, indices, scale, causal):
    if any(x.ndim != 4 for x in (q, k, v, indices)):
        raise ValueError("all four inputs must have rank four")
    b, n1, s1, dk = q.shape
    if min(q.shape) <= 0 or min(k.shape) <= 0 or min(v.shape) <= 0:
        raise ValueError("tensor dimensions must be positive")
    n2, s2, dv = k.shape[1], k.shape[2], v.shape[-1]
    if b > 64 or s1 > 2048 or s2 > 16384 or n1 > 256 or n2 > 64:
        raise ValueError("B/S1/S2/N1/N2 exceed the documented support range")
    if k.shape[0] != b or k.shape[-1] != dk or v.shape[:3] != k.shape[:3]:
        raise ValueError("query/key/value shapes are inconsistent")
    if n1 % n2 or dv > dk or dk % 64 or dv % 64:
        raise ValueError(
            "N1 must be divisible by N2; Dk/Dv must be 64-aligned, Dv <= Dk"
        )
    if indices.shape[:3] != (b, n2, s1) or not 1 <= indices.shape[-1] <= min(2048, s2):
        raise ValueError(
            "sparseIndices must have shape [B,N2,S1,topK], 1 <= topK <= min(2048,S2)"
        )
    if causal and s1 > s2:
        raise ValueError("causal attention requires S1 <= S2")
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scaleValue must be positive and finite")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("query/key/value dtypes must match")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("sparseIndices must be integer")
    if bool((indices < 0).any()) or bool((indices >= s2).any()):
        raise ValueError("sparseIndices entries must lie in [0,S2)")


def _attention(
    query,
    key,
    value,
    sparse_indices,
    scale,
    layout,
    causal,
    force_fp64=False,
    native_precision=False,
):
    q, k, v, indices = _as_bnsd(query, key, value, sparse_indices, layout)
    _validate(q, k, v, indices, scale, causal)
    b, n1, s1, dk = q.shape
    n2, s2, dv, topk = k.shape[1], k.shape[2], v.shape[-1], indices.shape[-1]
    group = n1 // n2
    if force_fp64 or q.dtype == torch.float64:
        work_dtype = torch.float64
    else:
        # The compatibility bench stores each arithmetic stage in the input
        # dtype, as the historical reference did. Public/FP64 paths keep their
        # existing arithmetic policy; all paths retain sparse bounded gathers.
        work_dtype = q.dtype if native_precision else torch.float32
    out = torch.empty((b, n1, s1, dv), device=q.device, dtype=work_dtype)
    # Bound selected-KV tensors plus scores to about 16 MiB. Tile size follows
    # shape/dtype only and never changes the attention definition.
    bytes_per_row = (topk * (dk + dv) + group * (dk + topk + dv)) * torch.empty(
        (), dtype=work_dtype
    ).element_size()
    rows_per_tile = max(1, min(64, (16 * 1024 * 1024) // bytes_per_row))
    for batch in range(b):
        for head in range(n2):
            # Convert each KV source head once, rather than repeating it G times.
            key_head = k[batch, head].to(work_dtype)
            value_head = v[batch, head].to(work_dtype)
            head_start = head * group
            for start in range(0, s1, rows_per_tile):
                stop = min(s1, start + rows_per_tile)
                ids = indices[batch, head, start:stop].long()
                selected_key = key_head.index_select(0, ids.reshape(-1)).reshape(
                    stop - start, topk, dk
                )
                selected_value = value_head.index_select(0, ids.reshape(-1)).reshape(
                    stop - start, topk, dv
                )
                query_tile = (
                    q[batch, head_start : head_start + group, start:stop]
                    .transpose(0, 1)
                    .to(work_dtype)
                )
                scores = torch.bmm(query_tile, selected_key.transpose(1, 2)) * scale
                if causal:
                    limit = (
                        torch.arange(start, stop, device=q.device).unsqueeze(-1)
                        + s2
                        - s1
                    )
                    scores.masked_fill_(~(ids <= limit).unsqueeze(1), float("-inf"))
                # Avoid -inf - -inf and zero division for fully masked rows.
                row_max = scores.amax(-1, keepdim=True)
                empty = torch.isneginf(row_max)
                scores.sub_(
                    torch.where(empty, torch.zeros_like(row_max), row_max)
                ).exp_()
                denominator = scores.sum(-1, keepdim=True)
                scores.div_(
                    torch.where(empty, torch.ones_like(denominator), denominator)
                )
                output_tile = torch.bmm(scores, selected_value)
                out[batch, head_start : head_start + group, start:stop] = (
                    output_tile.transpose(0, 1)
                )
    if layout == "BSND":
        out = out.permute(0, 2, 1, 3)
    return out.to(torch.float64 if force_fp64 else query.dtype).contiguous()


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Sparse GQA/MLA attention; value must equal key[..., :Dv].

    Indices must be unique within each (batch, KV head, query row). Causal
    masking is bottom-right aligned; fully masked rows return exact zeros.
    The output has the input layout, query dtype, and contiguous storage.
    """
    return _attention(
        query, key, value, sparseIndices, scaleValue, inputLayout, is_causal
    )


def sparse_flash_attention_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """CPU audit entry point: float64 arithmetic and float64 output."""
    return _attention(
        query,
        key,
        value,
        sparseIndices,
        scaleValue,
        inputLayout,
        is_causal,
        force_fp64=True,
    )


def sparse_flash_attention_bench(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Historical native-stage precision, with bounded sparse storage.

    This is a compatibility reference for the existing three-input checker,
    not a requirement that NPU implementations use an identical reduction
    tree. It does not change thresholds or the FP64 mathematical oracle.
    """
    return _attention(
        query,
        key,
        value,
        sparseIndices,
        scaleValue,
        inputLayout,
        is_causal,
        native_precision=True,
    )


def _structural_seed(query, key, sparse_indices):
    """Hash bounded samples, independent of the caller's global RNG state.

    Only 256 source indices and 32 Q/K elements each are copied to CPU. The
    coordinates address the original tensor directly, including strided input,
    so sampling never flattens/copies a complete noncontiguous payload.
    """
    digest = hashlib.sha256()
    for tensor, sample_count in ((sparse_indices, 256), (query, 32), (key, 32)):
        count = min(sample_count, tensor.numel())
        positions = torch.arange(count, device=tensor.device, dtype=torch.int64)
        if count > 1:
            positions = positions * (tensor.numel() - 1) // (count - 1)
        coordinates = []
        for size in reversed(tensor.shape):
            coordinates.insert(0, positions.remainder(size))
            positions = positions.div(size, rounding_mode="floor")
        sample = tensor[tuple(coordinates)].detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(bytes(sample.view(torch.uint8).tolist()))
    return int.from_bytes(digest.digest()[:8], "little") % (2**63)


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float = 1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
    **kwargs,
):
    """Construct V=K prefix and reproducible, unique sparse index sets.

    A stable hash of bounded input samples controls the structural input. A
    local generator avoids global RNG dependence or consuming its stream; a
    bounded random-priority block uses topk rather than sorting all groups.
    No duplicate padding is permitted when topK exceeds S2: that is invalid.
    """
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    s2 = int(key.shape[1] if inputLayout == "BSND" else key.shape[2])
    topk = int(sparseIndices.shape[-1])
    if not 1 <= topk <= min(2048, s2):
        raise ValueError("unique sparse indices require 1 <= topK <= min(2048,S2)")
    if value.shape[-1] > key.shape[-1]:
        raise ValueError("Dv must not exceed Dk")
    new_value = (
        key[..., : value.shape[-1]].clone().to(dtype=value.dtype, device=value.device)
    )
    groups = math.prod(sparseIndices.shape[:-1])
    selection = torch.empty((groups, topk), dtype=sparseIndices.dtype, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(
        _structural_seed(query, key, sparseIndices)
    )
    groups_per_block = max(1, (1024 * 1024) // s2)
    for start in range(0, groups, groups_per_block):
        stop = min(groups, start + groups_per_block)
        priorities = torch.rand((stop - start, s2), generator=generator)
        # Every returned column index is distinct, even when priorities tie.
        selection[start:stop] = priorities.topk(
            topk, dim=-1, largest=False, sorted=False
        ).indices
    new_indices = selection.reshape(sparseIndices.shape).to(sparseIndices.device)
    return [query, key, new_value, new_indices]
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, S1, S2, N1, N2, Dk, Dv, topK = 2, 1024, 8192, 32, 8, 128, 128, 512

# BSND 布局
query = torch.randn(B, S1, N1, Dk, dtype=torch.float16, device="npu")
key = torch.randn(B, S2, N2, Dk, dtype=torch.float16, device="npu")
value = key[..., :Dv].clone()  # 同源 latent KV cache 前缀
sparseIndices = torch.stack([
    torch.randperm(S2)[:topK] for _ in range(B * N2 * S1)
]).reshape(B, S1, N2, topK).to(dtype=torch.int32, device="npu")
y = sparse_flash_attention(query, key, value, sparseIndices,
                            scaleValue=1.0 / (Dk ** 0.5),
                            inputLayout="BSND",
                            is_causal=False)
# y.shape: [B, S1, N1, Dv]
```
