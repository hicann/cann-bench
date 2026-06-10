# FlashAttentionScoreGradEnhance 算子 API 描述

## 1. 算子简介

FlashAttentionScoreGradEnhance 是 `FlashAttentionScoreEnhance` 的反向算子，根据正向输入、上游梯度 `dy` 以及正向 softmax 中间统计量，计算 query、key、value 和可选 rope 分支的梯度。

**主要应用场景**：
- 大语言模型训练中的 FlashAttention 反向
- MLA nope/rope 分离结构的梯度计算
- 使用正向 `softmax_max` / `softmax_sum` 中间结果的高效 attention 反向

**算子特征**：
- 难度等级：L4（FusedComposite）
- 默认布局为 `input_layout="TND"`，同时保留 `BSND` 布局语义
- 支持 float16 / bfloat16 输入
- 输出 `dq`、`dk`、`dv`、`dq_rope`、`dk_rope`

## 2. 算子定义

### 数学公式

正向 attention 为：

$$
O = Softmax(Mask(scale \cdot QK^T + pse))V
$$

反向给定上游梯度 `dy = dL/dO`，计算：

$$
dQ = \frac{\partial L}{\partial Q},\quad dK = \frac{\partial L}{\partial K},\quad dV = \frac{\partial L}{\partial V}
$$

当传入 rope 时，实际参与 QK 的输入为：

$$
Q'=[Q,Q_{rope}],\quad K'=[K,K_{rope}]
$$

并额外输出：

$$
dQ_{rope},\quad dK_{rope}
$$

### 步骤说明

1. 按 `input_layout` 和 actual sequence 字段还原 batch 内有效序列。
2. 若传入 `query_rope` 和 `key_rope`，拼接后参与 QK 梯度计算。
3. 复用正向一致的 `pse`、`atten_mask`、`sparse_mode`、`pre_tokens`、`next_tokens` 语义重建 attention score。
4. 根据 `dy` 对 softmax attention 反向传播，得到 `dq`、`dk`、`dv`。
5. 若存在 rope 输入，将拼接维度上的梯度拆分为 `dq_rope` 和 `dk_rope`。

## 3. 接口规范

### 算子原型

```python
cann_bench.flash_attention_score_grad_enhance(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor dy,
    int head_num,
    str input_layout="TND",
    Tensor? pse=None,
    Tensor? padding_mask=None,
    Tensor? atten_mask=None,
    Tensor? softmax_max=None,
    Tensor? softmax_sum=None,
    Tensor? softmax_in=None,
    Tensor? attention_in=None,
    Tensor? sink_tensor=None,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    float scale_value=1.0,
    float keep_prob=1.0,
    int pre_tokens=2147483647,
    int next_tokens=2147483647,
    int inner_precise=0,
    prefix=None,
    Tensor|list|None actual_seq_qlen=None,
    Tensor|list|None actual_seq_kvlen=None,
    int sparse_mode=0,
    int sink_num=0,
    int pse_type=1,
    str softmaxInLayout="",
    Tensor|list|None q_start_idx=None,
    Tensor|list|None kv_start_idx=None,
) -> (Tensor dq, Tensor dk, Tensor dv, Tensor dq_rope, Tensor dk_rope)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `TND: [Tq,N,D]`；`BSND: [B,Sq,N,D]` | 正向 Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,D]`；`BSND: [B,Skv,Nkv,D]` | 正向 Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,Dv]`；`BSND: [B,Skv,Nkv,Dv]` | 正向 Value 输入 |
| dy | Tensor | 必选 | float16 / bfloat16 | 与正向 `attention_out` 一致 | 上游梯度 |
| head_num | int | 必选 | - | 标量 | query head 数 |
| input_layout | str | 默认 `TND` | - | - | 输入布局，支持 `TND` 和 `BSND` |
| pse | Tensor? | 默认 `None` | float16 / bfloat16 / float32 | 可广播到 `[B,N,Sq,Skv]` | 正向使用的 position bias |
| padding_mask | Tensor? | 默认 `None` | - | 预留 | 接口保留字段 |
| atten_mask | Tensor? | 默认 `None` | bool / uint8 | 可广播到 `[B,N,Sq,Skv]` 或 `[Sq,Skv]` | 正向使用的 attention mask |
| softmax_max | Tensor? | 默认 `None` | float32 | 与正向统计输出一致 | 正向 `softmax_max` 中间结果，标准 Golden 按等价正向逻辑重建 |
| softmax_sum | Tensor? | 默认 `None` | float32 | 与正向统计输出一致 | 正向 `softmax_sum` 中间结果，标准 Golden 按等价正向逻辑重建 |
| softmax_in | Tensor? | 默认 `None` | - | 预留 | softmax 输入保留字段 |
| attention_in | Tensor? | 默认 `None` | float16 / bfloat16 | 与正向 `attention_out` 一致 | 正向 attention 输出保留字段 |
| sink_tensor | Tensor? | 默认 `None` | - | 预留 | sink 相关保留字段，配合 `sink_num` 保持接口兼容 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 query 前缀维一致 | 正向 query rope 输入 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 key 前缀维一致 | 正向 key rope 输入 |
| scale_value | float | 默认 `1.0` | - | 标量 | 正向 QK 缩放系数 |
| keep_prob | float | 默认 `1.0` | - | 标量 | 正向 dropout 保留概率 |
| pre_tokens | int | 默认 `2147483647` | - | 标量 | 正向滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `2147483647` | - | 标量 | 正向滑窗右侧可见 token 数 |
| inner_precise | int | 默认 `0` | - | 标量 | 内部精度控制字段 |
| prefix | Any? | 默认 `None` | - | - | 接口保留字段 |
| actual_seq_qlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时可按前缀和解释 |
| actual_seq_kvlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | key/value 实际长度；TND 时可按前缀和解释 |
| sparse_mode | int | 默认 `0` | - | 标量 | 正向 sparse/mask 模式 |
| sink_num | int | 默认 `0` | - | 标量 | sink token 数，保留 sink 语义 |
| pse_type | int | 默认 `1` | - | 标量 | pse 类型标识 |
| softmaxInLayout | str | 默认 `""` | - | - | softmax 中间输入布局控制字段 |
| q_start_idx | Tensor/list? | 默认 `None` | int32 / int64 | - | query 起始索引保留字段 |
| kv_start_idx | Tensor/list? | 默认 `None` | int32 / int64 | - | key/value 起始索引保留字段 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| dq | float16 / bfloat16 | 与 query 一致 | Query 梯度 |
| dk | float16 / bfloat16 | 与 key 一致 | Key 梯度 |
| dv | float16 / bfloat16 | 与 value 一致 | Value 梯度 |
| dq_rope | float16 / bfloat16 | 与 query_rope 一致或 empty tensor | Query rope 梯度 |
| dk_rope | float16 / bfloat16 | 与 key_rope 一致或 empty tensor | Key rope 梯度 |

### 规则与约束

- `softmax_max` 和 `softmax_sum` 应来自同一次 `FlashAttentionScoreEnhance` 正向输出。
- `softmax_in` 为接口保留参数，标准 Golden 不依赖该输入。
- 反向使用的 `pse`、`atten_mask`、`sparse_mode`、`pre_tokens`、`next_tokens` 应与正向保持一致。
- `query_rope` 与 `key_rope` 需要成对传入；否则 rope 梯度返回 empty tensor。
- 当 key/value head 数小于 `head_num` 时，按 GQA/MQA 语义 repeat KV head。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `flash_attention_score_grad_enhance` 函数，使用 Torch autograd 按正向等价逻辑重建 attention，并返回 Q/K/V 及 rope 梯度。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

t, n, d, dr = 1024, 8, 128, 64
query = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
key = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
value = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
dy = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
query_rope = torch.randn(t, n, dr, dtype=torch.bfloat16, device="npu")
key_rope = torch.randn(t, n, dr, dtype=torch.bfloat16, device="npu")

dq, dk, dv, dq_rope, dk_rope = cann_bench.flash_attention_score_grad_enhance(
    query,
    key,
    value,
    dy,
    head_num=n,
    input_layout="TND",
    query_rope=query_rope,
    key_rope=key_rope,
    scale_value=(d + dr) ** -0.5,
    actual_seq_qlen=[t],
    actual_seq_kvlen=[t],
    sparse_mode=3,
)
```
