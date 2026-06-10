# AiInfraEsaSelectTopk 算子 API 描述

## 1. 算子简介

AiInfraEsaSelectTopk 是 ESA（Efficient Sparse Attention）索引选择算子，用于在长序列 attention 前为每个 query token 选择需要参与计算的 key block。算子将 key 序列划分为 Initial Tokens、Middle Tokens 和 Local Tokens，并在压缩后的 key block 上计算相关性，输出 top-k block 索引。

**主要应用场景**：
- 长上下文 prefill/chunk prefill 中的稀疏 key block 选择
- decode 场景下对历史 KV cache 的快速 top-k block 检索
- ESA 稀疏 attention 前置索引生成

**算子特征**：
- 难度等级：L3（SortSelect）
- 输入布局为 `input_layout="TND"`
- 支持 float16 / bfloat16 输入
- 固定常用 block 语义：`blk_size=64`，`compress_blk_size=16`

## 2. 算子定义

### 数学公式

对每个 query token 和 KV head，在压缩 key block 上计算相关性分数：

$$
score(q, B_j) = \max_{k \in B_j} q \cdot k
$$

输出由 Initial block、Local block、Middle top-k block 以及一个边界补充位置组成：

$$
selected = InitialBlocks \cup TopK(MiddleScores) \cup LocalBlocks \cup ExtraBlock
$$

输出最后一维长度为：

$$
select\_count = init\_blk\_num + local\_blk\_num + topk + 1
$$

### 步骤说明

1. 按 `actual_seq_q_len_optional` 将 TND query 切分为 batch/chunk。
2. 按 `actual_seq_k_len_optional` 获取每个 batch 的原始 key 长度。
3. 按 `actual_cmp_seq_k_len_optional` 获取压缩后的 key 长度；若未传入，则由原始 key 长度和 `compress_blk_size` 推导。
4. 将压缩 key 进一步按 `blk_size / compress_blk_size` 个压缩 token 聚合为候选 block。
5. 对 Initial block 保留优先级，对未来 block 施加 causal 约束。
6. 对 Middle block 做 top-k，拼接 Initial、Local 和补充 block，输出 int32 block 索引；无效位置填 -1。

## 3. 接口规范

### 算子原型

```python
cann_bench.ai_infra_esa_select_topk(
    Tensor query,
    Tensor key,
    int blk_size=64,
    int init_blk_num=2,
    int local_blk_num=4,
    int topk=4,
    str input_layout="TND",
    Tensor|list|None actual_seq_q_len_optional=None,
    Tensor|list|None actual_seq_k_len_optional=None,
    Tensor|list|None actual_cmp_seq_k_len_optional=None,
    int compress_blk_size=16,
) -> Tensor topk_indices
```

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `[Tq,Nq,D]` | Query 输入，TND layout |
| key | Tensor | 必选 | float16 / bfloat16 | `[B,Scmp,Nkv,D]` | 压缩后的 Key 输入，`Scmp` 为压缩 key token 数 |
| blk_size | int | 默认 `64` | - | 标量 | 原始 key block 大小 |
| init_blk_num | int | 默认 `2` | - | 标量 | 强制保留的开头 block 数 |
| local_blk_num | int | 默认 `4` | - | 标量 | 强制保留的局部 block 数 |
| topk | int | 默认 `4` | - | 标量 | Middle block 中选择的 top-k 数 |
| input_layout | str | 默认 `TND` | - | - | 输入布局，仅支持 `TND` |
| actual_seq_q_len_optional | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 序列长度或 TND 前缀和 |
| actual_seq_k_len_optional | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | 原始 key 序列长度或前缀和 |
| actual_cmp_seq_k_len_optional | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | 压缩后 key 长度，非累加和，通常约为 `ceil(key_len / compress_blk_size)` |
| compress_blk_size | int | 默认 `16` | - | 标量 | 每个压缩 key token 对应的原始 token 数 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| topk_indices | int32 | `[Tq,Nkv,init_blk_num+local_blk_num+topk+1]` | 每个 query token、KV head 对应的候选 key block 索引；无效位置为 -1 |

### 规则与约束

- 标准 Golden 只支持 `input_layout="TND"`。
- `blk_size` 应为 `compress_blk_size` 的整数倍；例如默认配置下 `64 / 16 = 4` 个压缩 token 组成一个 ESA block。
- `actual_cmp_seq_k_len_optional` 表示压缩后的 K 长度，不是原始 token 长度，且按每个 batch 的长度传入而非前缀和。
- decode 场景每个 batch 的 query length 通常为 1；prefill/chunk prefill 支持一个 chunk 内多个 query token。
- 当 key 长度小于需要选择的 block 覆盖范围时，输出会按 causal 阈值将未来 block 置为 -1。

## 4. 精度要求

本算子输出为离散 int32 索引，验证时应逐元素比较输出索引是否一致。

**通过标准**：

| 输出 | 判定方式 |
|------|----------|
| topk_indices | 与 golden 逐元素一致，允许无效位置均为 -1 |

## 5. 标准 Golden 代码

标准 Golden 参考实现位于同目录 `golden.py` 的 `ai_infra_esa_select_topk` 函数，使用 Torch 模拟 ESA 压缩 key block 选择、causal 约束和 top-k 索引生成。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

tq, b, scmp, nq, nkv, d = 1024, 1, 512, 32, 2, 128
query = torch.randn(tq, nq, d, dtype=torch.float16, device="npu")
key = torch.randn(b, scmp, nkv, d, dtype=torch.float16, device="npu")

indices = cann_bench.ai_infra_esa_select_topk(
    query,
    key,
    blk_size=64,
    init_blk_num=2,
    local_blk_num=4,
    topk=4,
    input_layout="TND",
    actual_seq_q_len_optional=[tq],
    actual_seq_k_len_optional=[8192],
    actual_cmp_seq_k_len_optional=[512],
    compress_blk_size=16,
)
```
