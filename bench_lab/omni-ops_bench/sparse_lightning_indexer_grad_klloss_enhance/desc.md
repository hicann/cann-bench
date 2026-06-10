# SparseLightningIndexerGradKLLossEnhance 算子 API 描述

## 1. 算子简介

SparseLightningIndexerGradKLLossEnhance 是 LightningIndexerEnhance 的训练反向算子，并融合 KL loss 计算。算子使用真实 attention 分布作为目标分布，使用 indexer 分支输出的 importance score 作为预测分布，计算 KL loss 以及 `query_index`、`key_index`、`weights` 的梯度。

**主要应用场景**：
- 训练 LightningIndexer，使稀疏索引分布逼近 attention 分布
- 长上下文稀疏 attention 的索引器蒸馏/校准
- 基于 top-k sparse index 的 KL loss 融合反向

**算子特征**：
- 难度等级：L4（FusedComposite）
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
cann_bench.sparse_lightning_indexer_grad_klloss_enhance(
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
- `Nq` 与 `Nkv` 按 GQA 分组；`query_index` 的 head 维需与 `weights` 一致。
- `sparse_mode=3` 会对目标分布施加 rightDownCausal mask。
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

标准 Golden 参考实现位于同目录 `golden.py` 的 `sparse_lightning_indexer_grad_klloss_enhance` 函数，使用 Torch autograd 模拟目标 attention 分布、indexer importance score、KL loss 和梯度输出。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

tq, tkv, nq, nkv, d, di, topk = 1024, 1024, 32, 1, 512, 128, 128
query = torch.randn(tq, nq, d, dtype=torch.bfloat16, device="npu")
key = torch.randn(tkv, nkv, d, dtype=torch.bfloat16, device="npu")
query_index = torch.randn(tq, nq, di, dtype=torch.bfloat16, device="npu")
key_index = torch.randn(tkv, nkv, di, dtype=torch.bfloat16, device="npu")
weights = torch.randn(tq, nq, dtype=torch.bfloat16, device="npu")
topk_index = torch.arange(topk, dtype=torch.int32, device="npu").view(1, 1, topk).repeat(tq, nkv, 1)

dqi, dki, dw, loss = cann_bench.sparse_lightning_indexer_grad_klloss_enhance(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_index,
    scale_value=d ** -0.5,
    layout="TND",
    actual_seq_lengths_query=[tq],
    actual_seq_lengths_key=[tkv],
    sparse_mode=3,
    deterministic=True,
)
```
