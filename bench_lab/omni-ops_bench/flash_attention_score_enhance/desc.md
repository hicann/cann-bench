# FlashAttentionScoreEnhance 算子 API 描述

## 1. 算子简介

FlashAttentionScoreEnhance 是训练场景下的增强版 FlashAttention 正向算子。算子支持 TND layout、可选 `pse`、`atten_mask`、MLA rope 输入、滑窗/causal sparse mode，并输出反向所需的 softmax 统计量。

**主要应用场景**：
- 大语言模型训练中的 FlashAttention 正向
- MLA 结构中 nope/rope 分离输入的 attention 计算
- 带 sparse mode、sink token 语义的长序列训练 attention

**算子特征**：
- 难度等级：L4（FusedComposite）
- 默认布局为 `input_layout="TND"`，同时保留 `BSND` 布局语义
- 支持 float16 / bfloat16 输入
- 输出 `attention_out`、`softmax_max`、`softmax_sum`

## 2. 算子定义

### 数学公式

当未传入 rope 时：

$$
attention\_out = Dropout(Softmax(Mask(scale \cdot QK^T + pse)))V
$$

当传入 `query_rope` 和 `key_rope` 时：

$$
Q' = [Q, Q_{rope}],\quad K' = [K, K_{rope}]
$$

$$
attention\_out = Dropout(Softmax(Mask(scale \cdot Q'{K'}^T + pse)))V
$$

### 步骤说明

1. 按 `input_layout` 和 `actual_seq_qlen` / `actual_seq_kvlen` 解析 batch 内有效序列。
2. 若传入 `query_rope` 和 `key_rope`，沿最后一维拼接到 Q/K 后参与 QK 矩阵乘。
3. 计算 `QK^T * scale`，并叠加可选 `pse`。
4. 应用 `atten_mask`；若未传入显式 mask，则按 `sparse_mode` 或 `pre_tokens` / `next_tokens` 生成 causal/滑窗 mask。
5. 对 score 做 softmax，并按 `keep_prob` 处理 dropout 语义。
6. 计算 attention 输出，同时返回 `softmax_max` 和 `softmax_sum` 供反向使用。

## 3. 接口规范

### 算子原型

```python
cann_bench.flash_attention_score_enhance(
    Tensor query,
    Tensor key,
    Tensor value,
    int head_num,
    str input_layout="TND",
    Tensor? pse=None,
    Tensor? padding_mask=None,
    Tensor? atten_mask=None,
    Tensor? sink_tensor=None,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    float scale=1.0,
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
    str softmaxOutLayout="",
    Tensor|list|None q_start_idx=None,
    Tensor|list|None kv_start_idx=None,
) -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `TND: [Tq,N,D]`；`BSND: [B,Sq,N,D]` | Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,D]`；`BSND: [B,Skv,Nkv,D]` | Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,Dv]`；`BSND: [B,Skv,Nkv,Dv]` | Value 输入 |
| head_num | int | 必选 | - | 标量 | query head 数 |
| input_layout | str | 默认 `TND` | - | - | 输入布局，支持 `TND` 和 `BSND` |
| pse | Tensor? | 默认 `None` | float16 / bfloat16 / float32 | 可广播到 `[B,N,Sq,Skv]` | Position shift / bias |
| padding_mask | Tensor? | 默认 `None` | - | 预留 | 接口保留字段，标准 Golden 不使用 |
| atten_mask | Tensor? | 默认 `None` | bool / uint8 | 可广播到 `[B,N,Sq,Skv]` 或 `[Sq,Skv]` | Attention mask，True 表示屏蔽 |
| sink_tensor | Tensor? | 默认 `None` | - | 预留 | sink 相关保留字段，配合 `sink_num` 保持接口兼容 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 query 前缀维一致 | MLA query rope 部分 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 key 前缀维一致 | MLA key rope 部分 |
| scale | float | 默认 `1.0` | - | 标量 | QK 缩放系数，部分用例也记录为 `scale_value` |
| keep_prob | float | 默认 `1.0` | - | 标量 | Dropout 保留概率 |
| pre_tokens | int | 默认 `2147483647` | - | 标量 | 滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `2147483647` | - | 标量 | 滑窗右侧可见 token 数 |
| inner_precise | int | 默认 `0` | - | 标量 | 内部精度控制字段 |
| prefix | Any? | 默认 `None` | - | - | 接口保留字段 |
| actual_seq_qlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时可按前缀和解释 |
| actual_seq_kvlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | key/value 实际长度；TND 时可按前缀和解释 |
| sparse_mode | int | 默认 `0` | - | 标量 | 0 为普通 attention；2/3/4 为 sparse 或 causal 变体 |
| sink_num | int | 默认 `0` | - | 标量 | sink token 数，保留 sink 语义 |
| pse_type | int | 默认 `1` | - | 标量 | pse 类型标识，标准 Golden 不分支处理 |
| softmaxOutLayout | str | 默认 `""` | - | - | softmax 中间输出布局控制字段 |
| q_start_idx | Tensor/list? | 默认 `None` | int32 / int64 | - | query 起始索引保留字段 |
| kv_start_idx | Tensor/list? | 默认 `None` | int32 / int64 | - | key/value 起始索引保留字段 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| attention_out | float16 / bfloat16 | 与 query 的 batch/seq/head 维一致，最后一维为 `Dv` | Attention 输出 |
| softmax_max | float32 | `[Tq,N,8]` 或 `[B,Sq,N,8]` | 每行 softmax score 最大值，供反向使用 |
| softmax_sum | float32 | 与 `softmax_max` 一致 | 每行 `exp(score-max)` 求和，供反向使用 |

### 规则与约束

- `query_rope` 与 `key_rope` 需要成对传入，且最后一维一致。
- 当 key/value head 数小于 `head_num` 时，按 GQA/MQA 语义 repeat KV head。
- `atten_mask=True` 表示该位置不参与 softmax。
- `sparse_mode=3` 按 right-down causal 语义生成 mask；`sparse_mode=2` 按普通 causal 语义生成 mask。
- `pre_tokens` / `next_tokens` 仅在未传入 `atten_mask` 且未使用 sparse causal 分支时生效。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `flash_attention_score_enhance` 函数，覆盖 TND/BSND 布局转换、rope 拼接、mask、softmax attention 以及 `softmax_max` / `softmax_sum` 统计量。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

t, n, d, dr = 1024, 8, 128, 64
query = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
key = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
value = torch.randn(t, n, d, dtype=torch.bfloat16, device="npu")
query_rope = torch.randn(t, n, dr, dtype=torch.bfloat16, device="npu")
key_rope = torch.randn(t, n, dr, dtype=torch.bfloat16, device="npu")

out, softmax_max, softmax_sum = cann_bench.flash_attention_score_enhance(
    query,
    key,
    value,
    head_num=n,
    input_layout="TND",
    query_rope=query_rope,
    key_rope=key_rope,
    scale=(d + dr) ** -0.5,
    keep_prob=1.0,
    actual_seq_qlen=[t],
    actual_seq_kvlen=[t],
    sparse_mode=3,
)
```
