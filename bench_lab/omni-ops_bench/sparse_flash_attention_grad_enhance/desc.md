# SparseFlashAttentionGradEnhance 算子 API 描述

## 1. 算子简介

SparseFlashAttentionGradEnhance 是稀疏 FlashAttention 的反向算子。算子根据 `topk_indices` / `sparse_indices` 选择出的 KV block，只在被选中的 key/value 子集上计算 attention 反向，输出 query、key、value 以及可选 rope 分支的梯度。

**主要应用场景**：
- 长上下文稀疏 attention 训练反向
- 由 LightningIndexer/ESA 生成 top-k block 后的反向计算
- MLA rope 分离输入的稀疏 attention 梯度

**算子特征**：
- 难度等级：L4（FusedComposite）
- 支持 `BSND` 和 `TND` layout
- 支持 `sparse_mode=0` 和 `sparse_mode=3`
- 输出 `dquery`、`dkey`、`dvalue`、`dquery_rope`、`dkey_rope`

## 2. 算子定义

### 数学公式

对每个 query token，根据稀疏 block 索引 gather KV：

$$
\tilde{K}, \tilde{V} = Gather(K,V,topk\_indices,sparse\_block\_size)
$$

稀疏正向为：

$$
O = Softmax(Mask(scale\_value \cdot Q\tilde{K}^T))\tilde{V}
$$

反向根据 `dy = dL/dO` 计算：

$$
dQ,\quad dK,\quad dV
$$

当传入 rope 时，使用 `[Q,Q_{rope}]` 和 `[K,K_{rope}]` 参与 score 计算，并额外输出 rope 梯度。

### 步骤说明

1. 按 `layout` 和 actual sequence 字段还原 batch 内 query/key/value。
2. 将 `topk_indices` 中的 block id 展开为 key/value token 位置；无效 block id 为 -1。
3. 对每个 query token、KV head gather 选中的 KV 子序列。
4. 若传入 `query_rope` 和 `key_rope`，拼接后参与 score 计算。
5. 当 `sparse_mode=3` 时，对 selected positions 应用 rightDownCausal mask。
6. 对稀疏 attention 执行反向，累加得到 dense 形态的 `dkey` / `dvalue`。

## 3. 接口规范

### 算子原型

```python
cann_bench.sparse_flash_attention_grad_enhance(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor topk_indices,
    Tensor dy,
    Tensor attention_out,
    Tensor softmax_max,
    Tensor softmax_sum,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    Tensor|list|None actual_seq_qlen=None,
    Tensor|list|None actual_seq_kvlen=None,
    float scale_value=1.0,
    int sparse_block_size=1,
    str layout="TND",
    int sparse_mode=3,
    int pre_tokens=2147483647,
    int next_tokens=2147483647,
    bool deterministic=False,
) -> (Tensor dquery, Tensor dkey, Tensor dvalue, Tensor dquery_rope, Tensor dkey_rope)
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `TND: [Tq,Nq,D]`；`BSND: [B,Sq,Nq,D]` | 正向 Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,D]`；`BSND: [B,Skv,Nkv,D]` | 正向 Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,Dv]`；`BSND: [B,Skv,Nkv,Dv]` | 正向 Value 输入 |
| topk_indices | Tensor | 必选 | int32 | `[Tq,Nkv,K]` 或 `[B,Sq,Nkv,K]` | 稀疏 KV block 索引，无效值为 -1 |
| dy | Tensor | 必选 | float16 / bfloat16 | 与正向 attention 输出一致 | 上游梯度 |
| attention_out | Tensor | 必选 | float16 / bfloat16 | 与 dy 一致 | 正向 attention 输出保留字段 |
| softmax_max | Tensor | 必选 | float32 | 与正向统计输出一致 | 正向 softmax max，标准 Golden 按等价正向逻辑重建 |
| softmax_sum | Tensor | 必选 | float32 | 与正向统计输出一致 | 正向 softmax sum，标准 Golden 按等价正向逻辑重建 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 query 前缀维一致 | Query rope 输入 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 key 前缀维一致 | Key rope 输入 |
| actual_seq_qlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时可为前缀和 |
| actual_seq_kvlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | key/value 实际长度；TND 时可为前缀和 |
| scale_value | float | 默认 `1.0` | - | 标量 | QK 缩放系数 |
| sparse_block_size | int | 默认 `1` | - | 标量 | `topk_indices` 中每个 block 覆盖的 KV token 数 |
| layout | str | 默认 `TND` | - | - | 输入布局，支持 `TND` 或 `BSND` |
| sparse_mode | int | 默认 `3` | - | 标量 | 0 为不加 causal mask，3 为 rightDownCausal |
| pre_tokens | int | 默认 `2147483647` | - | 标量 | 滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `2147483647` | - | 标量 | 滑窗右侧可见 token 数 |
| deterministic | bool | 默认 `False` | - | 标量 | 确定性计算开关，标准 Golden 不分支处理 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| dquery | float16 / bfloat16 | 与 query 一致 | Query 梯度 |
| dkey | float16 / bfloat16 | 与 key 一致 | Key 梯度 |
| dvalue | float16 / bfloat16 | 与 value 一致 | Value 梯度 |
| dquery_rope | float16 / bfloat16 | 与 query_rope 一致或 empty tensor | Query rope 梯度 |
| dkey_rope | float16 / bfloat16 | 与 key_rope 一致或 empty tensor | Key rope 梯度 |

### 规则与约束

- `topk_indices` 中有效 block id 应在 `[0, ceil(kv_len / sparse_block_size))` 范围内。
- 每行无效 block id 使用 -1，且应位于有效值之后。
- `sparse_mode=3` 会对 gather 后的位置继续施加 rightDownCausal mask。
- `query_rope` 与 `key_rope` 需要成对传入；否则 rope 梯度返回 empty tensor。
- `attention_out`、`softmax_max`、`softmax_sum` 应与同一次稀疏正向输出对应。

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

标准 Golden 参考实现位于同目录 `golden.py` 的 `sparse_flash_attention_grad_enhance` 函数，使用 Torch autograd 模拟稀疏 KV block gather、rightDownCausal mask、rope 拆分和梯度累加。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

b, sq, skv, nq, nkv, d = 1, 4, 128, 8, 1, 128
query = torch.randn(b, sq, nq, d, dtype=torch.float16, device="npu")
key = torch.randn(b, skv, nkv, d, dtype=torch.float16, device="npu")
value = torch.randn(b, skv, nkv, d, dtype=torch.float16, device="npu")
topk_indices = torch.arange(16, dtype=torch.int32, device="npu").view(1, 1, 1, 16).repeat(b, sq, nkv, 1)
dy = torch.randn(b, sq, nq, d, dtype=torch.float16, device="npu")
attention_out = torch.randn_like(dy)
softmax_max = torch.zeros(b, sq, nq, dtype=torch.float32, device="npu")
softmax_sum = torch.ones(b, sq, nq, dtype=torch.float32, device="npu")

dq, dk, dv, dqr, dkr = cann_bench.sparse_flash_attention_grad_enhance(
    query,
    key,
    value,
    topk_indices,
    dy,
    attention_out,
    softmax_max,
    softmax_sum,
    actual_seq_qlen=[sq],
    actual_seq_kvlen=[skv],
    scale_value=d ** -0.5,
    sparse_block_size=1,
    layout="BSND",
    sparse_mode=3,
)
```
