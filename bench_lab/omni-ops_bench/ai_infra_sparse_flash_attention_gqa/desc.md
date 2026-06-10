# AiInfraSparseFlashAttentionGqa 算子 API 描述

## 1. 算子简介

AiInfraSparseFlashAttentionGqa 是面向长序列推理场景的 Grouped-Query Sparse FlashAttention 算子。算子通过 `sparse_indices` 对 KV cache 做稀疏 block 选择，并支持 query head 数大于 key/value head 数的 GQA 计算。

**主要应用场景**：
- 大语言模型长上下文推理
- GQA/MQA 场景下的稀疏 KV cache attention
- 结合外部索引器生成的 sparse block 进行局部 attention

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 `BSND` / `TND` query layout
- 支持 `BSND` / `TND` / PageAttention KV layout
- 可选返回 `softmax_lse`

## 2. 算子定义

### 数学公式

$$
attention\_out = \text{softmax}(Q \cdot \tilde{K}^{T} \times scale\_value) \cdot \tilde{V}
$$

其中 `\tilde{K}`、`\tilde{V}` 表示按稀疏 block 索引选出的 KV 子序列。

### 步骤说明

1. 根据 `input_layout` 和 `actual_seq_lengths_query` 解析 query batch。
2. 根据 `input_layout_kv`、`block_table` 和 `actual_seq_lengths_kv` 解析 KV batch。
3. 对每个 batch、KV head 和 query token，使用 `sparse_indices` gather KV block。
4. 对 GQA 分组内的多个 query head 共享同一个 KV head。
5. 计算 `QK^T * scale_value`，按 `sparse_mode` 应用 mask。
6. 计算 softmax、输出 attention，并在需要时输出 `softmax_lse`。

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_sparse_flash_attention_gqa(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor sparse_indices,
    float scale_value,
    int sparse_block_size,
    *,
    Tensor? actual_seq_lengths_query=None,
    Tensor? actual_seq_lengths_kv=None,
    Tensor? block_table=None,
    int num_query_heads=1,
    int num_key_value_heads=0,
    str input_layout="BSH",
    str input_layout_kv="BSND",
    int sparse_mode=0,
    int block_size=0,
    bool return_softmax_lse=False,
) -> (Tensor attention_out, Tensor softmax_lse)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `[B,S1,N1,D]` 或 `[T1,N1,D]` | Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 | `[B,S2,N2,D]`、`[T2,N2,D]` 或 PA 布局 | Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | 与 key 对应，最后一维为 `Dv` | Value 输入 |
| sparse_indices | Tensor | 必选 | int32 | `[B,S1,N2,sparse_size]` 或 `[T1,N2,sparse_size]` | 稀疏 KV block 索引 |
| scale_value | float | 必选 | - | 标量 | QK 缩放系数 |
| sparse_block_size | int | 必选 | - | 标量 | 稀疏 block 大小 |
| actual_seq_lengths_query | Tensor? | 默认 `None` | int32 | `[B]` | query 实际长度；TND 时为前缀和 |
| actual_seq_lengths_kv | Tensor? | 默认 `None` | int32 | `[B]` | key/value 实际长度；TND/PA 时必选 |
| block_table | Tensor? | 默认 `None` | int32 | `[B,max_block_num]` | PageAttention block 映射表 |
| num_query_heads | int | 默认 `1` | - | 标量 | query head 数 |
| num_key_value_heads | int | 默认 `0` | - | 标量 | key/value head 数；0 表示与 query 相同 |
| input_layout | str | 默认 `BSH` | - | - | query layout |
| input_layout_kv | str | 默认 `BSND` | - | - | key/value layout |
| sparse_mode | int | 默认 `0` | - | 标量 | 0 为全量计算，3 为 rightDownCausal |
| block_size | int | 默认 `0` | - | 标量 | PageAttention block size |
| return_softmax_lse | bool | 默认 `False` | - | 标量 | 是否返回 softmax_lse |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| attention_out | float16 / bfloat16 | `[B,S1,N1,Dv]` 或 `[T1,N1,Dv]` | Attention 输出 |
| softmax_lse | float32 | `[B,N1,S1,1]` 或 `[T1,N1,1]` | 可选 softmax log-sum-exp 输出 |

### 规则与约束

- `N1` 必须能被 `N2` 整除，GQA 分组大小为 `N1 / N2`。
- `sparse_indices` 中有效 block id 必须在 KV 实际长度范围内。
- `sparse_indices` 每行无效值使用 -1，且应位于有效值之后。
- `sparse_block_size` 应能整除 PageAttention 的 `block_size`。
- `input_layout="TND"` 时，`actual_seq_lengths_query` 使用前缀和语义。
- `input_layout_kv="TND"` 或 PA 布局时，`actual_seq_lengths_kv` 必须传入。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `ai_infra_sparse_flash_attention_gqa` 函数，使用 Torch/NumPy 模拟 GQA 分组、稀疏 KV gather、rightDownCausal mask、softmax attention 和 `softmax_lse`。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

b, s1, s2, n1, n2, d = 1, 8, 128, 8, 1, 128
query = torch.randn(b, s1, n1, d, dtype=torch.float16, device="npu")
key = torch.randn(b, s2, n2, d, dtype=torch.float16, device="npu")
value = torch.randn(b, s2, n2, d, dtype=torch.float16, device="npu")
sparse_indices = torch.arange(s2, dtype=torch.int32, device="npu").view(1, 1, 1, s2).repeat(b, s1, n2, 1)

out, lse = cann_bench.ai_infra_sparse_flash_attention_gqa(
    query,
    key,
    value,
    sparse_indices,
    scale_value=d ** -0.5,
    sparse_block_size=1,
    actual_seq_lengths_query=torch.tensor([s1], dtype=torch.int32, device="npu"),
    actual_seq_lengths_kv=torch.tensor([s2], dtype=torch.int32, device="npu"),
    num_query_heads=n1,
    num_key_value_heads=n2,
    input_layout="BSND",
    input_layout_kv="BSND",
    sparse_mode=0,
    return_softmax_lse=True,
)
```
