# SparseFlashAttentionEnhance 算子 API 描述

## 1. 算子简介

SparseFlashAttentionEnhance 是面向长序列场景的稀疏 FlashAttention 算子。算子根据 `sparse_indices` 只选取部分 KV block 参与注意力计算，可减少完整 Attention 的计算量，并支持 MLA 场景下的 `query_rope` / `key_rope` 拼接。

**主要应用场景**：
- 长上下文大模型训练或推理中的稀疏注意力
- 基于重要性索引的 KV cache 稀疏读取
- MLA 结构中 nope/rope 分离输入的 attention 计算

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 `BSND` 和 `TND` query layout
- 支持 `BSND`、`TND` 和 `PA_BSND` KV layout
- 支持 `sparse_mode=0` 全量计算和 `sparse_mode=3` rightDownCausal mask

## 2. 算子定义

### 数学公式

$$
attention\_out = \text{softmax}(Q \cdot \tilde{K}^{T} \times scale\_value) \cdot \tilde{V}
$$

其中 `\tilde{K}` 和 `\tilde{V}` 是按照 `sparse_indices` 从 Key/Value 中按 block 选取出的稀疏序列。

### 步骤说明

1. **布局归一化**：将 `BSND` / `TND` 输入转换为按 batch 处理的内部布局。
2. **PageAttention 映射**：当 `layout_kv="PA_BSND"` 时，通过 `block_table` 还原每个 batch 的 KV 顺序。
3. **rope 拼接**：若传入 `query_rope` 和 `key_rope`，沿最后一维拼接后参与 QK 矩阵乘。
4. **稀疏 gather**：对每个 query token，按 `sparse_indices` 选取 KV block。
5. **mask**：`sparse_mode=3` 时应用 rightDownCausal mask。
6. **Attention**：执行 `softmax(QK^T * scale_value) @ V`。

## 3. 接口规范

### 算子原型

```python
cann_bench.sparse_flash_attention_enhance(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor sparse_indices,
    float scale_value,
    *,
    Tensor? block_table=None,
    Tensor? actual_seq_lengths_query=None,
    Tensor? actual_seq_lengths_kv=None,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    int sparse_block_size=1,
    str layout_query="BSND",
    str layout_kv="BSND",
    int sparse_mode=3,
    int pre_tokens=9223372036854775807,
    int next_tokens=9223372036854775807,
    int attention_mode=0,
    bool return_softmax_lse=False,
) -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `BSND: [B,S1,N1,D]`；`TND: [T1,N1,D]` | Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 | `BSND: [B,S2,N2,D]`；`TND: [T2,N2,D]`；`PA_BSND: [block_num,block_size,N2,D]` | Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | 与 key 对应，最后一维为 `Dv` | Value 输入 |
| sparse_indices | Tensor | 必选 | int32 | `[B,S1,N2,sparse_size]` 或 `[T1,N2,sparse_size]` | 稀疏 KV block 索引，无效值为 -1 |
| scale_value | float | 必选 | - | 标量 | QK 矩阵乘后的缩放系数 |
| block_table | Tensor? | 默认 `None` | int32 | `[B,max_block_num]` | PageAttention block 映射表 |
| actual_seq_lengths_query | Tensor? | 默认 `None` | int32 | `[B]` | query 实际长度；TND 时为前缀和 |
| actual_seq_lengths_kv | Tensor? | 默认 `None` | int32 | `[B]` | key/value 实际长度；TND/PA_BSND 时必选 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 query 前缀维一致 | MLA query rope 部分 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 key 前缀维一致 | MLA key rope 部分 |
| sparse_block_size | int | 默认 `1` | - | 标量 | 稀疏索引对应的 block 大小 |
| layout_query | str | 默认 `BSND` | - | - | query 布局，支持 `BSND` 或 `TND` |
| layout_kv | str | 默认 `BSND` | - | - | key/value 布局，支持 `BSND`、`TND` 或 `PA_BSND` |
| sparse_mode | int | 默认 `3` | - | 标量 | 0 为全量计算，3 为 rightDownCausal |
| pre_tokens | int | 默认 `9223372036854775807` | - | 标量 | 滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `9223372036854775807` | - | 标量 | 滑窗右侧可见 token 数 |
| attention_mode | int | 默认 `0` | - | 标量 | attention 模式控制字段 |
| return_softmax_lse | bool | 默认 `False` | - | 标量 | 是否按接口请求返回 softmax 统计信息 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| attention_out | float16 / bfloat16 | 与 query 的非 D 维一致，D 与 value 对应 | Attention 输出 |
| softmax_max | float32 | 可选统计输出 | softmax max 输出 |
| softmax_sum | float32 | 可选统计输出 | softmax sum 输出 |

### 规则与约束

- `num_query_heads` 需能整除 `num_key_value_heads`，即 GQA 分组为整数。
- `sparse_indices` 每行有效 block id 应位于前部，无效值使用 -1 填充。
- `sparse_block_size` 取值范围通常为 `[1,128]` 且为 2 的幂。
- `layout_query="TND"` 时，`actual_seq_lengths_query` 必须按前缀和语义传入。
- `layout_kv="TND"` 或 `layout_kv="PA_BSND"` 时，`actual_seq_lengths_kv` 必须传入。
- PageAttention 场景下必须传入有效的 `block_table`。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `sparse_flash_attention_enhance` 函数，使用 Torch/NumPy 进行布局转换、稀疏 gather、rightDownCausal mask 和 softmax attention 计算。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

b, s1, s2, n1, n2, d = 1, 4, 128, 8, 1, 128
query = torch.randn(b, s1, n1, d, dtype=torch.float16, device="npu")
key = torch.randn(b, s2, n2, d, dtype=torch.float16, device="npu")
value = torch.randn(b, s2, n2, d, dtype=torch.float16, device="npu")
sparse_indices = torch.arange(s2, dtype=torch.int32, device="npu").view(1, 1, 1, s2).repeat(b, s1, n2, 1)
act_q = torch.tensor([s1], dtype=torch.int32, device="npu")
act_kv = torch.tensor([s2], dtype=torch.int32, device="npu")

out, softmax_max, softmax_sum = cann_bench.sparse_flash_attention_enhance(
    query,
    key,
    value,
    sparse_indices,
    scale_value=d ** -0.5,
    actual_seq_lengths_query=act_q,
    actual_seq_lengths_kv=act_kv,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
)
```
