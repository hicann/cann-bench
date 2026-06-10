# AiInfraKvQuantSparseFlashAttention 算子 API 描述

## 1. 算子简介

AiInfraKvQuantSparseFlashAttention 是在 Sparse FlashAttention 基础上支持 KV per-tile 量化输入的推理算子。算子从 combine 格式的量化 Key 中解析 int8 KV 数据、rope 信息和反量化 scale，完成稀疏 attention 计算。

**主要应用场景**：
- 长上下文推理中的量化 KV cache attention
- Per-Token-Head-Tile-128 KV 量化方案
- PageAttention + Sparse Attention + MLA 的组合推理路径

**算子特征**：
- 难度等级：L4（FusedComposite）
- key/value 量化模式主要为 `key_quant_mode=2`、`value_quant_mode=2`
- 支持 `quant_scale_repo_mode=1` combine 存放模式
- 支持可选 sink token

## 2. 算子定义

### 数学公式

$$
attention\_out =
\text{softmax}(Q \cdot \text{Dequant}(\tilde{K}^{INT8}, Scale_K)^T \times scale\_value)
\cdot \text{Dequant}(\tilde{V}^{INT8}, Scale_V)
$$

其中 `\tilde{K}` 和 `\tilde{V}` 按 `sparse_indices` 从量化 KV cache 中选取。

### 步骤说明

1. 从 `query` 中解析 q_nope 与 q_rope。
2. 从 combine 格式 `key` 中解析 int8 key、key_rope 和反量化 scale。
3. 根据 `tile_size` 对反量化 scale 做 repeat，并将 int8 key 反量化为浮点 key。
4. 按 `block_table` 和 `actual_seq_lengths_kv` 解析 PageAttention KV。
5. 按 `sparse_indices` gather 稀疏 KV block。
6. 可选拼接 sink token，并执行 scaled softmax attention。

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_kv_quant_sparse_flash_attention(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor sparse_indices,
    float scale_value,
    int key_quant_mode=2,
    int value_quant_mode=2,
    *,
    Tensor? key_dequant_scale=None,
    Tensor? value_dequant_scale=None,
    Tensor? block_table=None,
    Tensor? actual_seq_lengths_query=None,
    Tensor? actual_seq_lengths_kv=None,
    Tensor? key_sink=None,
    Tensor? value_sink=None,
    int sparse_block_size=1,
    str layout_query="BSND",
    str layout_kv="BSND",
    int sparse_mode=3,
    int pre_tokens=9223372036854775807,
    int next_tokens=9223372036854775807,
    int attention_mode=0,
    int quant_scale_repo_mode=1,
    int tile_size=128,
    int rope_head_dim=64,
) -> Tensor attention_out
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `[B,S1,N1,D]` 或 `[T1,N1,D]` | Query 输入，通常为 q_nope+q_rope 拼接 |
| key | Tensor | 必选 | int8 | `[B,S2,N2,D]`、`[T2,N2,D]` 或 `[block_num,block_size,N2,D]` | combine 格式量化 Key |
| value | Tensor | 必选 | int8 | 与 key 对应，最后一维为 `Dv` | 量化 Value |
| sparse_indices | Tensor | 必选 | int32 | `[B,S1,N2,sparse_size]` 或 `[T1,N2,sparse_size]` | 稀疏 KV block 索引 |
| scale_value | float | 必选 | - | 标量 | QK 缩放系数 |
| key_quant_mode | int | 默认 `2` | - | 标量 | key 量化模式，2 表示 per-tile |
| value_quant_mode | int | 默认 `2` | - | 标量 | value 量化模式，2 表示 per-tile |
| key_dequant_scale | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | key 反量化 scale，combine 模式下可由 key 解析 |
| value_dequant_scale | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | value 反量化 scale，combine 模式下可由 value 解析 |
| block_table | Tensor? | 默认 `None` | int32 | `[B,max_block_num]` | PageAttention block 映射表 |
| actual_seq_lengths_query | Tensor? | 默认 `None` | int32 | `[B]` | query 实际长度；TND 时为前缀和 |
| actual_seq_lengths_kv | Tensor? | 默认 `None` | int32 | `[B]` | key/value 实际长度 |
| key_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_num,N2,D]` | sink key |
| value_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_num,N2,Dv]` | sink value |
| sparse_block_size | int | 默认 `1` | - | 标量 | 稀疏 block 大小 |
| layout_query | str | 默认 `BSND` | - | - | query 布局，支持 `BSND` 或 `TND` |
| layout_kv | str | 默认 `BSND` | - | - | key/value 布局，支持 `BSND`、`TND` 或 `PA_BSND` |
| sparse_mode | int | 默认 `3` | - | 标量 | 0 为全量计算，3 为 rightDownCausal |
| pre_tokens | int | 默认 `9223372036854775807` | - | 标量 | 滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `9223372036854775807` | - | 标量 | 滑窗右侧可见 token 数 |
| attention_mode | int | 默认 `0` | - | 标量 | attention 模式控制字段 |
| quant_scale_repo_mode | int | 默认 `1` | - | 标量 | 1 表示量化 scale combine 存放 |
| tile_size | int | 默认 `128` | - | 标量 | per-tile 量化 tile 大小 |
| rope_head_dim | int | 默认 `64` | - | 标量 | rope 头维度 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| attention_out | float16 / bfloat16 | 与 query 的非 D 维一致，D 与 value 对应 | Attention 输出 |

### 规则与约束

- `key_quant_mode` 和 `value_quant_mode` 支持 2，即 per-tile 量化。
- `quant_scale_repo_mode` 支持 1，即 combine 存放模式。
- combine key 中通常按 D 维顺序存放 key nope、key rope 和 float32 量化 scale。
- `tile_size` 默认 128，反量化 scale 会在最后一维按 tile repeat。
- PageAttention 场景需要传入 `block_table`。
- `layout_query="TND"` 时，`actual_seq_lengths_query` 使用前缀和语义。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `ai_infra_kv_quant_sparse_flash_attention` 函数，覆盖 combine key 解析、per-tile 反量化和稀疏 attention 计算。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

b, s1, s2, n1, n2, d, dr = 1, 1, 256, 8, 1, 512, 64
query = torch.randn(b, s1, n1, d + dr, dtype=torch.bfloat16, device="npu")
block_size = 128
block_num = (s2 + block_size - 1) // block_size

# 实际使用时 key 为 int8 combine 格式：key_nope + key_rope bytes + dequant_scale bytes
key = torch.randint(-128, 127, (block_num, block_size, n2, d + dr * 2 + 16), dtype=torch.int8, device="npu")
value = torch.randint(-128, 127, (block_num, block_size, n2, d), dtype=torch.int8, device="npu")
sparse_indices = torch.arange(block_num, dtype=torch.int32, device="npu").view(1, 1, 1, block_num)
block_table = torch.arange(block_num, dtype=torch.int32, device="npu").view(b, block_num)

out = cann_bench.ai_infra_kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value=d ** -0.5,
    key_quant_mode=2,
    value_quant_mode=2,
    block_table=block_table,
    actual_seq_lengths_query=torch.tensor([s1], dtype=torch.int32, device="npu"),
    actual_seq_lengths_kv=torch.tensor([s2], dtype=torch.int32, device="npu"),
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=0,
)
```
