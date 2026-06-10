# AiInfraFusedInferAttentionSink 算子 API 描述

## 1. 算子简介

AiInfraFusedInferAttentionSink 是面向增量和全量推理场景的 Fused Infer Attention 算子。算子支持 TND 输入、GQA、多种 mask/sparse mode、可选 MLA rope 以及 key/value 前拼接 sink token。

**主要应用场景**：
- 大语言模型推理中的 fused attention
- 增量推理与全量推理统一 attention 路径
- 带 sink token 的 attention score 吸收机制
- TND layout 下的 GQA/MLA attention

**算子特征**：
- 难度等级：L4（FusedComposite）
- 输入布局为 `input_layout="TND"`
- 支持 `query_rope` / `key_rope`
- 支持 `key_sink` / `value_sink` / `key_rope_sink`
- 可选返回 `softmax_lse`

## 2. 算子定义

### 数学公式

$$
attention\_out = \text{softmax}(Q \cdot K^T \times softmax\_scale + mask) \cdot V
$$

当传入 sink token 时：

$$
K' = [K_{sink}, K],\quad V' = [V_{sink}, V]
$$

并使用 `K'`、`V'` 完成 attention 计算。

### 步骤说明

1. 按 TND layout 读取 `query`、`key`、`value`。
2. 若传入 `query_rope` 和 `key_rope`，沿最后一维拼接 rope。
3. 若传入 sink token，则将 `key_sink` / `value_sink` 拼接到 KV 序列前。
4. 根据 `num_query_heads` 和 `num_key_value_heads` 执行 GQA head 复制。
5. 计算 `QK^T * softmax_scale`，应用 `atten_mask`。
6. 计算 softmax attention，并按需返回 `softmax_lse`。

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_fused_infer_attention_sink(
    Tensor query,
    Tensor key,
    Tensor value,
    *,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    Tensor? pse_shift=None,
    Tensor? atten_mask=None,
    Tensor? actual_seq_qlen=None,
    Tensor? actual_seq_kvlen=None,
    Tensor? block_table=None,
    Tensor? dequant_scale_query=None,
    Tensor? dequant_scale_key=None,
    Tensor? dequant_offset_key=None,
    Tensor? dequant_scale_value=None,
    Tensor? dequant_offset_value=None,
    Tensor? dequant_scale_key_rope=None,
    Tensor? quant_scale_out=None,
    Tensor? quant_offset_out=None,
    Tensor? meta_data=None,
    int num_query_heads=1,
    int num_key_value_heads=0,
    float softmax_scale=1.0,
    int pre_tokens=2147483647,
    int next_tokens=2147483647,
    str input_layout="TND",
    int sparse_mode=0,
    int block_size=0,
    int query_quant_mode=0,
    int key_quant_mode=0,
    int value_quant_mode=0,
    int inner_precise=0,
    bool return_softmax_lse=False,
    int sink_number=0,
    Tensor? key_sink=None,
    Tensor? value_sink=None,
    Tensor? key_rope_sink=None,
) -> (Tensor attention_out, Tensor softmax_lse)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 / int8 | `[Tq,Nq,D]` | Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 / int8 | `[Tkv,Nkv,D]` | Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 / int8 | `[Tkv,Nkv,Dv]` | Value 输入 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | `[Tq,Nq,Dr]` | MLA query rope |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | `[Tkv,Nkv,Dr]` | MLA key rope |
| pse_shift | Tensor? | 默认 `None` | float16 / bfloat16 / float32 | 可广播到 attention score | Position shift / bias |
| atten_mask | Tensor? | 默认 `None` | bool / uint8 | 可广播到 `[B,Nq,Sq,Skv]` | Attention mask，True 表示屏蔽 |
| actual_seq_qlen | Tensor? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时为前缀和 |
| actual_seq_kvlen | Tensor? | 默认 `None` | int32 / int64 | `[B]` | key/value 实际长度；TND 时为前缀和 |
| block_table | Tensor? | 默认 `None` | int32 | 实现相关 | PageAttention block 映射表保留字段 |
| dequant_scale_query | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | query 反量化 scale |
| dequant_scale_key | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | key 反量化 scale |
| dequant_offset_key | Tensor? | 默认 `None` | float32 / int32 | 实现相关 | key 反量化 offset |
| dequant_scale_value | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | value 反量化 scale |
| dequant_offset_value | Tensor? | 默认 `None` | float32 / int32 | 实现相关 | value 反量化 offset |
| dequant_scale_key_rope | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | key rope 反量化 scale |
| quant_scale_out | Tensor? | 默认 `None` | float32 / float16 / bfloat16 | 实现相关 | 输出量化 scale |
| quant_offset_out | Tensor? | 默认 `None` | float32 / int32 | 实现相关 | 输出量化 offset |
| meta_data | Tensor? | 默认 `None` | uint32 | 实现相关 | metadata 前置算子生成的 tiling 信息 |
| num_query_heads | int | 默认 `1` | - | 标量 | query head 数 |
| num_key_value_heads | int | 默认 `0` | - | 标量 | key/value head 数；0 表示与 query 相同 |
| softmax_scale | float | 默认 `1.0` | - | 标量 | QK 缩放系数 |
| pre_tokens | int | 默认 `2147483647` | - | 标量 | 滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `2147483647` | - | 标量 | 滑窗右侧可见 token 数 |
| input_layout | str | 默认 `TND` | - | - | 输入 layout，仅支持 `TND` |
| sparse_mode | int | 默认 `0` | - | 标量 | sparse/mask 模式 |
| block_size | int | 默认 `0` | - | 标量 | block size 保留字段 |
| query_quant_mode | int | 默认 `0` | - | 标量 | query 量化模式 |
| key_quant_mode | int | 默认 `0` | - | 标量 | key 量化模式 |
| value_quant_mode | int | 默认 `0` | - | 标量 | value 量化模式 |
| inner_precise | int | 默认 `0` | - | 标量 | 内部精度控制字段 |
| return_softmax_lse | bool | 默认 `False` | - | 标量 | 是否返回 softmax_lse |
| sink_number | int | 默认 `0` | - | 标量 | sink token 数 |
| key_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_number,Nkv,D]` | sink key |
| value_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_number,Nkv,Dv]` | sink value |
| key_rope_sink | Tensor? | 默认 `None` | float16 / bfloat16 | `[sink_number,Nkv,Dr]` | sink key rope |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| attention_out | float16 / bfloat16 / int8 | `[Tq,Nq,Dv]` | Attention 输出 |
| softmax_lse | float32 | `[Tq,Nq,1]` 或占位 tensor | 可选 softmax log-sum-exp 输出 |

### 规则与约束

- `input_layout` 支持 `TND`。
- `num_query_heads` 必须能被 `num_key_value_heads` 整除，或 `num_key_value_heads=0` 表示二者相等。
- `query_rope` 与 `key_rope` 需要成对传入，且最后一维一致。
- `key_sink` 与参与 QK 的 key 最后一维一致；`value_sink` 与 value 最后一维一致。
- `atten_mask=True` 表示该位置不参与 softmax。
- 返回 `softmax_lse` 时，其 dtype 为 float32。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `ai_infra_fused_infer_attention_sink` 函数，覆盖 TND 浮点路径，包括 rope 拼接、sink token、GQA head repeat、mask 和 softmax_lse。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

tq, tkv, nq, nkv, d = 1, 128, 8, 1, 128
query = torch.randn(tq, nq, d, dtype=torch.float16, device="npu")
key = torch.randn(tkv, nkv, d, dtype=torch.float16, device="npu")
value = torch.randn(tkv, nkv, d, dtype=torch.float16, device="npu")
mask = torch.zeros(tq, tkv, dtype=torch.bool, device="npu")

out, lse = cann_bench.ai_infra_fused_infer_attention_sink(
    query,
    key,
    value,
    atten_mask=mask,
    actual_seq_qlen=torch.tensor([tq], dtype=torch.int64, device="npu"),
    actual_seq_kvlen=torch.tensor([tkv], dtype=torch.int64, device="npu"),
    num_query_heads=nq,
    num_key_value_heads=nkv,
    softmax_scale=d ** -0.5,
    input_layout="TND",
    return_softmax_lse=True,
)
```
