# LightningIndexerEnhance 算子 API 描述

## 1. 算子简介

LightningIndexerEnhance 是稀疏 attention 的索引器算子。算子基于 query/key 的 index 表征和 token 权重计算 importance score，并为每个 query token 返回最重要的 key token/block 索引及其分数。

**主要应用场景**：
- 长上下文模型中的稀疏 KV 选择
- Lightning Attention / Sparse Attention 的前置索引生成
- PageAttention KV cache 场景下的 top-k 稀疏索引计算

**算子特征**：
- 难度等级：L3（SortSelect）
- 支持 `BSND`、`TND` query layout
- 支持 `BSND`、`TND`、`PA_BSND` key layout
- 输出 `sparse_indices` 和 `sparse_values`

## 2. 算子定义

### 数学公式

对每个 query token、KV head 和候选 key token，按 GQA 分组计算：

$$
I(q, k) = \sum_{h \in group} ReLU(Q_{index,h}K_{index}^T) \times W_h
$$

然后在 key 维度上选择 top-k：

$$
sparse\_indices, sparse\_values = TopK(I, sparse\_count)
$$

### 步骤说明

1. 按 `layout_query` 和 `actual_seq_lengths_query` 将 query/weights 归一化为 batch 内序列。
2. 按 `layout_key`、`actual_seq_lengths_key` 和可选 `block_table` 解析 key；`PA_BSND` 通过 PageAttention block table 还原逻辑 KV 顺序。
3. 将 query head 按 KV head 数分组，形成 GQA 分组。
4. 计算 `ReLU(query @ key^T) * weights`，并在 GQA 分组内聚合。
5. 当 `sparse_mode=3` 时施加 rightDownCausal mask。
6. 在 key 维度选择 `sparse_count` 个最大值，返回索引和分数；无效位置用 -1/0 填充。

## 3. 接口规范

### 算子原型

```python
cann_bench.lightning_indexer_enhance(
    Tensor query,
    Tensor key,
    Tensor weights,
    Tensor|list|None actual_seq_lengths_query=None,
    Tensor|list|None actual_seq_lengths_key=None,
    Tensor? block_table=None,
    str layout_query="BSND",
    str layout_key="BSND",
    int sparse_count=2048,
    int sparse_mode=3,
    int pre_tokens=9223372036854775807,
    int next_tokens=9223372036854775807,
    bool return_value=False,
    int sparse_block_size=1,
    int sparse_block_mode=0,
) -> (Tensor sparse_indices, Tensor sparse_values)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `BSND: [B,Sq,Nq,D]`；`TND: [Tq,Nq,D]` | Query index 表征 |
| key | Tensor | 必选 | float16 / bfloat16 | `BSND: [B,Skv,Nkv,D]`；`TND: [Tkv,Nkv,D]`；`PA_BSND: [block_num,block_size,Nkv,D]` | Key index 表征 |
| weights | Tensor | 必选 | float16 / bfloat16 | `[B,Sq,Nq]` 或 `[Tq,Nq]` | Query head 权重 |
| actual_seq_lengths_query | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时可为前缀和 |
| actual_seq_lengths_key | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | key 实际长度；TND/PA_BSND 时可为前缀和 |
| block_table | Tensor? | 默认 `None` | int32 | `[B,max_block_num]` | PageAttention 物理 block 到逻辑序列的映射 |
| layout_query | str | 默认 `BSND` | - | - | query 布局，支持 `BSND` 或 `TND` |
| layout_key | str | 默认 `BSND` | - | - | key 布局，支持 `BSND`、`TND` 或 `PA_BSND` |
| sparse_count | int | 默认 `2048` | - | 标量 | 每个 query token、KV head 返回的 top-k 数 |
| sparse_mode | int | 默认 `3` | - | 标量 | 0 为不加 causal mask，3 为 rightDownCausal |
| pre_tokens | int | 默认 `9223372036854775807` | - | 标量 | 滑窗左侧可见 token 数，接口保留字段 |
| next_tokens | int | 默认 `9223372036854775807` | - | 标量 | 滑窗右侧可见 token 数，接口保留字段 |
| return_value | bool | 默认 `False` | - | 标量 | 控制 sparse value 输出语义；本接口返回 indices 和 values |
| sparse_block_size | int | 默认 `1` | - | 标量 | 稀疏 block 大小 |
| sparse_block_mode | int | 默认 `0` | - | 标量 | 稀疏 block 模式 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| sparse_indices | int32 | `[B,Sq,Nkv,sparse_count]` 或 `[Tq,Nkv,sparse_count]` | top-k key token/block 索引，无效位置为 -1 |
| sparse_values | float16 / bfloat16 | 与 `sparse_indices` 一致 | top-k importance score |

### 规则与约束

- `Nq` 必须能按 `Nkv` 分组；GQA 分组大小为 `Nq / Nkv`。
- `weights` 的前缀维和 head 维应与 `query` 一致。
- `layout_key="PA_BSND"` 时必须提供有效 `block_table`，key shape 按 PageAttention block layout 解释。
- `sparse_count` 大于有效 key 长度时，多余位置填充为 -1，分数为 0。
- `sparse_mode=3` 会屏蔽未来 key 位置，适用于 causal decode/prefill。

## 4. 精度要求

采用生态算子精度标准进行验证。`sparse_indices` 为离散输出，应逐元素一致；`sparse_values` 按浮点误差标准验证。

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

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold，且索引输出一致时判定为通过。

## 5. 标准 Golden 代码

标准 Golden 参考实现位于同目录 `golden.py` 的 `lightning_indexer_enhance` 函数，使用 Torch 模拟 BSND/TND/PA_BSND 布局转换、GQA 分组、rightDownCausal mask 和 top-k 索引选择。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

b, sq, skv, nq, nkv, d = 1, 4, 1024, 32, 1, 128
query = torch.randn(b, sq, nq, d, dtype=torch.float16, device="npu")
key = torch.randn(b, skv, nkv, d, dtype=torch.float16, device="npu")
weights = torch.randn(b, sq, nq, dtype=torch.float16, device="npu")

sparse_indices, sparse_values = cann_bench.lightning_indexer_enhance(
    query,
    key,
    weights,
    actual_seq_lengths_query=[sq],
    actual_seq_lengths_key=[skv],
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=128,
    sparse_mode=3,
    sparse_block_size=1,
)
```
