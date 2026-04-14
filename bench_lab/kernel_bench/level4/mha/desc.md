# MHA 算子 API 描述

## 1. 算子简介

多头注意力机制 (Multi-Head Attention) 算子，将输入通过多个注意力头并行计算后拼接输出，支持可选偏置和 dropout，广泛应用于 Transformer 架构。

**主要应用场景**：
- Transformer 编码器和解码器中的自注意力与交叉注意力
- 大语言模型和视觉 Transformer 中的核心注意力模块
- 多模态模型中的跨模态注意力融合

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value, 投影权重和可选偏置）单输出，融合线性投影、多头缩放点积注意力与输出投影
- 支持可配置的头数、缩放因子和 dropout 比率

## 2. 算子定义

### 数学公式

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i \times K_i^T}{\sqrt{d_k}}\right) \times V_i
$$

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \times W_o
$$

其中：
- $Q_i = X_q W_q^{(i)}$，$K_i = X_k W_k^{(i)}$，$V_i = X_v W_v^{(i)}$ 为各头的投影结果
- $d_k = D / h$ 为每个头的维度，$h$ 为头数
- $\sqrt{d_k}$ 为缩放因子（由 `scaleValue` 参数指定，<=0 时自动使用 $1/\sqrt{d_k}$）
- softmax 沿最后一维计算

具体子步骤：
1. **线性投影**：$Q = \text{query} \times W_q + b_q$，$K = \text{key} \times W_k + b_k$，$V = \text{value} \times W_v + b_v$
2. **重塑为多头**：$Q, K, V$ 拆分为 $h$ 个头，每头维度 $d_k$
3. **缩放点积注意力**：$\text{scores} = Q_i \times K_i^T \times \text{scaleValue}$
4. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
5. **加权求和**：$\text{attn\_output} = \text{attn\_weights} \times V_i$
6. **拼接与输出投影**：$y = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \times W_o + b_o$

## 3. 接口规范

### 算子原型

```python
ascend_bench.mha(Tensor query, Tensor key, Tensor value, Tensor weight_q, Tensor weight_k, Tensor weight_v, Tensor weight_o, Tensor? bias_q, Tensor? bias_k, Tensor? bias_v, Tensor? bias_o, int numHeads, float scaleValue, float dropoutRate) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量，shape 为 [B, S, D] |
| key | Tensor | 必选 | 键张量，shape 为 [B, S_kv, D] |
| value | Tensor | 必选 | 值张量，shape 为 [B, S_kv, D] |
| weight_q | Tensor | 必选 | Query 投影权重，shape 为 [D, D] |
| weight_k | Tensor | 必选 | Key 投影权重，shape 为 [D, D] |
| weight_v | Tensor | 必选 | Value 投影权重，shape 为 [D, D] |
| weight_o | Tensor | 必选 | 输出投影权重，shape 为 [D, D] |
| bias_q | Tensor | None | Query 投影偏置，shape 为 [D]（可选） |
| bias_k | Tensor | None | Key 投影偏置，shape 为 [D]（可选） |
| bias_v | Tensor | None | Value 投影偏置，shape 为 [D]（可选） |
| bias_o | Tensor | None | 输出投影偏置，shape 为 [D]（可选） |
| numHeads | int | 必选 | 注意力头数 |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(d_k) |
| dropoutRate | float | 0.0 | dropout 比率 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, D] | 与输入 query 相同 | 多头注意力输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（query, key, value, weight_*, bias_*）的 dtype 必须一致
- `query` 的 shape 为 [B, S, D]，`key` 和 `value` 的 shape 为 [B, S_kv, D]
- D 必须能被 `numHeads` 整除，每头维度 d_k = D / numHeads
- `weight_q`, `weight_k`, `weight_v`, `weight_o` 的 shape 均为 [D, D]
- `bias_q`, `bias_k`, `bias_v`, `bias_o` 均为可选参数，shape 为 [D]
- `scaleValue` 通常设置为 $1/\sqrt{d_k}$，当 <= 0 时自动使用该值
- `dropoutRate` 仅在训练模式下生效

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
MHA算子Torch Golden参考实现

多头注意力机制 (Multi-Head Attention)，将输入通过多个注意力头并行计算后拼接输出
公式:
    head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
    MHA(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
"""
def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    weight_o: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    bias_k: torch.Tensor | None = None,
    bias_v: torch.Tensor | None = None,
    bias_o: torch.Tensor | None = None,
    numHeads: int = 1,
    scaleValue: float = -1.0,
    dropoutRate: float = 0.0
) -> torch.Tensor:
    """
    多头注意力机制 (Multi-Head Attention)

    公式:
        head_i = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
        MHA = Concat(head_1, ..., head_h) @ W_o

    Args:
        query: 查询张量 [B, S, D]
        key: 键张量 [B, S_kv, D]
        value: 值张量 [B, S_kv, D]
        weight_q: Query 投影权重 [D, D]
        weight_k: Key 投影权重 [D, D]
        weight_v: Value 投影权重 [D, D]
        weight_o: 输出投影权重 [D, D]
        bias_q: Query 投影偏置 [D] (可选)
        bias_k: Key 投影偏置 [D] (可选)
        bias_v: Value 投影偏置 [D] (可选)
        bias_o: 输出投影偏置 [D] (可选)
        numHeads: 注意力头数
        scaleValue: 缩放因子，<=0时自动使用 1/sqrt(d_k)
        dropoutRate: dropout比率

    Returns:
        输出张量 [B, S, D]
    """
    B, S, D = query.shape
    S_kv = key.shape[1]
    d_k = D // numHeads

    if scaleValue <= 0:
        scaleValue = 1.0 / (d_k ** 0.5)

    # 线性投影: [B, S, D] @ [D, D] -> [B, S, D]
    Q = torch.nn.functional.linear(query, weight_q, bias_q)
    K = torch.nn.functional.linear(key, weight_k, bias_k)
    V = torch.nn.functional.linear(value, weight_v, bias_v)

    # 重塑为多头: [B, S, D] -> [B, S, numHeads, d_k] -> [B, numHeads, S, d_k]
    Q = Q.reshape(B, S, numHeads, d_k).transpose(1, 2)
    K = K.reshape(B, S_kv, numHeads, d_k).transpose(1, 2)
    V = V.reshape(B, S_kv, numHeads, d_k).transpose(1, 2)

    # 缩放点积注意力: [B, numHeads, S, d_k] @ [B, numHeads, d_k, S_kv] -> [B, numHeads, S, S_kv]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # 加权求和: [B, numHeads, S, S_kv] @ [B, numHeads, S_kv, d_k] -> [B, numHeads, S, d_k]
    attn_output = torch.matmul(attn_weights, V)

    # 拼接多头: [B, numHeads, S, d_k] -> [B, S, numHeads, d_k] -> [B, S, D]
    attn_output = attn_output.transpose(1, 2).reshape(B, S, D)

    # 输出投影: [B, S, D] @ [D, D] -> [B, S, D]
    y = torch.nn.functional.linear(attn_output, weight_o, bias_o)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

B, S, D, numHeads = 2, 128, 512, 8
query = torch.randn(B, S, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S, D, dtype=torch.float16, device="npu")
weight_q = torch.randn(D, D, dtype=torch.float16, device="npu")
weight_k = torch.randn(D, D, dtype=torch.float16, device="npu")
weight_v = torch.randn(D, D, dtype=torch.float16, device="npu")
weight_o = torch.randn(D, D, dtype=torch.float16, device="npu")
y = ascend_bench.mha(query, key, value, weight_q, weight_k, weight_v, weight_o,
                      None, None, None, None,
                      numHeads=numHeads, scaleValue=-1.0, dropoutRate=0.0)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **GQA**：分组查询注意力，多个 query head 共享 KV head，是 MHA 的高效变体
- **MLA**：多头潜在注意力，通过低秩压缩 KV 缓存降低推理内存
- **SparseFlashAttention**：稀疏注意力计算，大序列长度推理场景的高效注意力计算
