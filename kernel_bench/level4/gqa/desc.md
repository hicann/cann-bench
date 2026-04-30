# GQA 算子 API 描述

## 1. 算子简介

分组查询注意力 (Grouped Query Attention) 算子，多个 query head 共享一组 key/value head，对已分头的 Q/K/V 执行注意力计算，在保持模型质量的同时显著减少 KV cache 内存占用和推理计算量。

**主要应用场景**：
- 大语言模型推理中的高效注意力计算（如 LLaMA-2 70B、Mistral）
- 长序列推理场景中降低 KV cache 内存开销
- 需要在模型质量和推理效率之间平衡的 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value）单输出，执行分组缩放点积注意力
- 输入为已分头的张量，不包含 QKV 投影和输出投影步骤
- N_q 必须能被 N_kv 整除，每个 KV head 被 N_q/N_kv 个 query head 共享

## 2. 算子定义

### 数学公式

对于第 $i$ 个 query head，使用第 $\lfloor i \times N_{kv} / N_q \rfloor$ 个 KV head：

$$
\text{head}_i = \text{softmax}\left(Q_i \times K_{g(i)}^T \times \text{scaleValue}\right) \times V_{g(i)}
$$

其中：
- $N_q$ 为 query 头数，$N_{kv}$ 为 KV 头数，$N_q$ 必须能被 $N_{kv}$ 整除
- $g(i) = \lfloor i \times N_{kv} / N_q \rfloor$ 为第 $i$ 个 query head 对应的 KV head 索引
- $D$ 为每个头的维度
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{D}$）
- 每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

具体子步骤：
1. **KV head 扩展**：将每个 KV head 重复 $N_q / N_{kv}$ 次以匹配 query head 数
2. **缩放点积**：$\text{scores} = Q_i \times K_{g(i)}^T \times \text{scaleValue}$
3. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
4. **加权求和**：$y_i = \text{attn\_weights} \times V_{g(i)}$

## 3. 接口规范

### 算子原型

```python
cann_bench.gqa(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量（已分头），shape 为 [B, S, N_q, D] |
| key | Tensor | 必选 | 键张量（已分头），shape 为 [B, S_kv, N_kv, D] |
| value | Tensor | 必选 | 值张量（已分头），shape 为 [B, S_kv, N_kv, D] |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(D) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, N_q, D] | 与输入 query 相同 | 分组查询注意力输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（query, key, value）的 dtype 必须一致
- `query` 的 shape 为 [B, S, N_q, D]，`key` 和 `value` 的 shape 为 [B, S_kv, N_kv, D]
- N_q 必须能被 N_kv 整除，分组比 G = N_q / N_kv
- 当 N_kv == N_q 时退化为标准多头注意力 (MHA)
- 当 N_kv == 1 时退化为多查询注意力 (MQA)
- `scaleValue` 通常设置为 $1/\sqrt{D}$，当 <= 0 时自动使用该值

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch

"""
GQA算子Torch Golden参考实现

分组查询注意力 (Grouped Query Attention)，多个 query head 共享一组 KV head
公式: 扩展 KV heads 匹配 Q heads，y = softmax(Q @ K^T * scaleValue) @ V
"""


def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """
    分组查询注意力 (Grouped Query Attention)

    Args:
        query: 查询张量 [B, S, N_q, D]（已分头）
        key: 键张量 [B, S_kv, N_kv, D]（已分头）
        value: 值张量 [B, S_kv, N_kv, D]（已分头）
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(D)

    Returns:
        输出张量 [B, S, N_q, D]
    """
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 扩展 KV heads 以匹配 Q heads
    G = N_q // N_kv
    key = key.unsqueeze(3).expand(B, S_kv, N_kv, G, D).reshape(B, S_kv, N_q, D)
    value = value.unsqueeze(3).expand(B, S_kv, N_kv, G, D).reshape(B, S_kv, N_q, D)

    # 转置为 [B, N_q, S, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N_q, D]
    return attn_output.transpose(1, 2)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv, D = 2, 128, 128, 128
N_q, N_kv = 32, 8
query = torch.randn(B, S, N_q, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, N_kv, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, N_kv, D, dtype=torch.float16, device="npu")
y = cann_bench.gqa(query, key, value, scaleValue=-1.0)
```
