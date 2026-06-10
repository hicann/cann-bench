# AiInfraSparseFlashAttentionPioneer 算子 API 描述

## 1. 算子简介

AiInfraSparseFlashAttentionPioneer 是面向长序列推理场景的 Sparse FlashAttention 算子，重点支持 PageAttention KV cache、MLA rope 拼接以及可选 sink token。算子通过稀疏 block 索引降低 KV 访问与 attention 计算量。

**主要应用场景**：
- 大语言模型长上下文推理
- PageAttention KV cache 上的稀疏 attention
- MLA 场景下的 nope/rope 合并或分离传输
- 带 sink token 的 attention score 吸收

**算子特征**：
- 难度等级：L4（FusedComposite）
- KV layout 主要为 `PA_BSND`
- 支持 `BSND` / `TND` query layout
- 支持 `key_sink` / `value_sink`

## 2. 算子定义

### 数学公式

$$
attention\_out = \text{softmax}(Q \cdot [K_{sink}, \tilde{K}]^{T} \times scale\_value) \cdot [V_{sink}, \tilde{V}]
$$

若未传入 sink token，则退化为：

$$
attention\_out = \text{softmax}(Q \cdot \tilde{K}^{T} \times scale\_value) \cdot \tilde{V}
$$

### 步骤说明

1. 按 `layout_query` 解析 query；TND 场景根据 `actual_seq_lengths_query` 前缀和拆 batch。
2. 按 `block_table` 将 `PA_BSND` KV cache 映射回 batch 内连续序列。
3. 若传入 `query_rope` / `key_rope`，沿 D 维拼接后参与 QK 计算。
4. 按 `sparse_indices` 选取稀疏 KV block。
5. 可选将 `key_sink` / `value_sink` 拼接在稀疏 KV 前。
6. 执行 scaled softmax attention。

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_sparse_flash_attention_pioneer(
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
    Tensor? key_sink=None,
    Tensor? value_sink=None,
    int sparse_block_size=1,
    str layout_query="BSND",
    str layout_kv="PA_BSND",
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
| query | Tensor | 必选 | float16 / bfloat16 | `[B,S1,N1,D]` 或 `[T1,N1,D]` | Query 输入，可为 nope+rope 合并 |
| key | Tensor | 必选 | float16 / bfloat16 | `[block_num,block_size,N2,D]` | PA_BSND Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | `[block_num,block_size,N2,Dv]` | PA_BSND Value 输入 |
| sparse_indices | Tensor | 必选 | int32 | `[B,S1,N2,sparse_size]` 或 `[T1,N2,sparse_size]` | 稀疏 KV block 索引 |
| scale_value | float | 必选 | - | 标量 | QK 缩放系数 |
| block_table | Tensor? | 默认 `None` | int32 | `[B,max_block_num]` | PageAttention block 映射表 |
| actual_seq_lengths_query | Tensor? | 默认 `None` | int32 | `[B]` | query 实际长度；TND 时为前缀和 |
| actual_seq_lengths_kv | Tensor? | 默认 `None` | int32 | `[B]` | key/value 实际长度 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 query 前缀维一致 | MLA query rope 部分 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 key 前缀维一致 | MLA key rope 部分 |
| key_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_num,N2,D]` | 可学习 sink key |
| value_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_num,N2,Dv]` | 可学习 sink value |
| sparse_block_size | int | 默认 `1` | - | 标量 | 稀疏 block 大小 |
| layout_query | str | 默认 `BSND` | - | - | query 布局，支持 `BSND` 或 `TND` |
| layout_kv | str | 默认 `PA_BSND` | - | - | key/value 布局，通常为 `PA_BSND` |
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

- PageAttention 场景必须提供 `block_table`。
- `actual_seq_lengths_kv` 描述每个 batch 的有效 KV 长度。
- 合并传输时，`query` / `key` 可在 D 维包含 nope 与 rope 部分。
- 分离传输时，`query_rope` 与 `key_rope` 的 D 维需要一致。
- `key_sink` 与参与 QK 的 key 最后一维一致；`value_sink` 与 value 最后一维一致。
- `sparse_indices` 的有效 block id 必须在当前 batch 的 KV block 范围内。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `ai_infra_sparse_flash_attention_pioneer` 函数，覆盖 PA_BSND 映射、rope 拼接、sink token、稀疏 gather 与 softmax attention。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

b, s1, s2, n1, n2, d, block_size = 1, 1, 256, 8, 1, 512, 128
block_num = (s2 + block_size - 1) // block_size
query = torch.randn(b, s1, n1, d, dtype=torch.float16, device="npu")
key = torch.randn(b * block_num, block_size, n2, d, dtype=torch.float16, device="npu")
value = torch.randn(b * block_num, block_size, n2, d, dtype=torch.float16, device="npu")
sparse_indices = torch.arange(block_num, dtype=torch.int32, device="npu").view(1, 1, 1, block_num)
block_table = torch.arange(block_num, dtype=torch.int32, device="npu").view(b, block_num)

out, softmax_max, softmax_sum = cann_bench.ai_infra_sparse_flash_attention_pioneer(
    query,
    key,
    value,
    sparse_indices,
    scale_value=d ** -0.5,
    block_table=block_table,
    actual_seq_lengths_query=torch.tensor([s1], dtype=torch.int32, device="npu"),
    actual_seq_lengths_kv=torch.tensor([s2], dtype=torch.int32, device="npu"),
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=0,
)
```
