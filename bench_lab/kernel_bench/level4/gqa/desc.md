# GQA 算子 API 描述

## 1. 算子简介

分组查询注意力 (Grouped Query Attention) 算子，多个 query head 共享一组 key/value head，在保持模型质量的同时显著减少 KV cache 内存占用和推理计算量。

**主要应用场景**：
- 大语言模型推理中的高效注意力计算（如 LLaMA-2 70B、Mistral）
- 长序列推理场景中降低 KV cache 内存开销
- 需要在模型质量和推理效率之间平衡的 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, key, value, 可选 weight_o）单输出，融合分组注意力与可选输出投影
- 支持可配置的 query 头数、KV 头数和缩放因子

## 2. 算子定义

### 数学公式

对于第 $i$ 个 query head，使用第 $\lfloor i \times N_{kv} / N_q \rfloor$ 个 KV head：

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i \times K_{g(i)}^T}{\sqrt{d_k}}\right) \times V_{g(i)}
$$

$$
\text{GQA} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \times W_o
$$

其中：
- $N_q$ 为 query 头数，$N_{kv}$ 为 KV 头数，$N_q$ 必须能被 $N_{kv}$ 整除
- $g(i) = \lfloor i \times N_{kv} / N_q \rfloor$ 为第 $i$ 个 query head 对应的 KV head 索引
- $d_k$ 为每个头的维度
- 每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

具体子步骤：
1. **KV head 扩展**：将每个 KV head 重复 $N_q / N_{kv}$ 次以匹配 query head 数
2. **缩放点积**：$\text{scores} = Q_i \times K_{g(i)}^T \times \text{scaleValue}$
3. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
4. **加权求和**：$\text{attn\_output} = \text{attn\_weights} \times V_{g(i)}$
5. **输出投影**（可选）：$y = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \times W_o$

## 3. 接口规范

### 算子原型

```python
ascend_bench.gqa(Tensor query, Tensor key, Tensor value, Tensor? weight_o, int numHeads, int numKVHeads, float scaleValue) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量，shape 为 [B, S, N_q, D]，N_q = numHeads |
| key | Tensor | 必选 | 键张量，shape 为 [B, S_kv, N_kv, D]，N_kv = numKVHeads |
| value | Tensor | 必选 | 值张量，shape 为 [B, S_kv, N_kv, D] |
| weight_o | Tensor | None | 输出投影权重，shape 为 [N_q * D, N_q * D]（可为 None 表示不做输出投影） |
| numHeads | int | 必选 | query 头数 |
| numKVHeads | int | 必选 | KV 头数，需整除 numHeads |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(d_k) |

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

- 所有输入 Tensor（query, key, value, weight_o）的 dtype 必须一致
- `query` 的 shape 为 [B, S, N_q, D]，`key` 和 `value` 的 shape 为 [B, S_kv, N_kv, D]
- `numHeads` 必须能被 `numKVHeads` 整除，分组比 = numHeads / numKVHeads
- 当 `numKVHeads == numHeads` 时退化为标准多头注意力 (MHA)
- 当 `numKVHeads == 1` 时退化为多查询注意力 (MQA)
- `weight_o` 为可选参数，为 None 时不进行输出投影
- `scaleValue` 通常设置为 $1/\sqrt{d_k}$，当 <= 0 时自动使用该值

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
GQA算子Torch Golden参考实现

分组查询注意力 (Grouped Query Attention)，多个 query head 共享一组 key/value head，减少 KV cache 内存占用
公式:
    对于第 i 个 query head，使用第 floor(i * numKVHeads / numHeads) 个 KV head
    head_i = softmax(Q_i @ K_g(i)^T / sqrt(d_k)) @ V_g(i)
    GQA = Concat(head_1, ..., head_h) @ W_o
"""
def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight_o: torch.Tensor | None = None,
    numHeads: int = 1,
    numKVHeads: int = 1,
    scaleValue: float = -1.0
) -> torch.Tensor:
    """
    分组查询注意力 (Grouped Query Attention)

    公式:
        对于第 i 个 query head，使用第 floor(i * numKVHeads / numHeads) 个 KV head
        head_i = softmax(Q_i @ K_g(i)^T / sqrt(d_k)) @ V_g(i)
        GQA = Concat(head_1, ..., head_h) @ W_o

    Args:
        query: 查询张量 [B, S, N_q, D]，N_q = numHeads
        key: 键张量 [B, S_kv, N_kv, D]，N_kv = numKVHeads
        value: 值张量 [B, S_kv, N_kv, D]
        weight_o: 输出投影权重 [N_q * D, N_q * D] (可为None表示不做输出投影)
        numHeads: query 头数
        numKVHeads: KV 头数，需整除 numHeads
        scaleValue: 缩放因子，<=0时自动使用 1/sqrt(d_k)

    Returns:
        输出张量 [B, S, N_q, D]
    """
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 扩展 KV heads 以匹配 Q heads: 每个 KV head 重复 numHeads // numKVHeads 次
    num_repeats = numHeads // numKVHeads
    # [B, S_kv, N_kv, D] -> [B, S_kv, N_kv, 1, D] -> [B, S_kv, N_kv, num_repeats, D] -> [B, S_kv, N_q, D]
    key = key.unsqueeze(3).expand(B, S_kv, N_kv, num_repeats, D).reshape(B, S_kv, N_q, D)
    value = value.unsqueeze(3).expand(B, S_kv, N_kv, num_repeats, D).reshape(B, S_kv, N_q, D)

    # 转置为 [B, N_q, S, D] 和 [B, N_q, S_kv, D]
    q = query.transpose(1, 2)   # [B, N_q, S, D]
    k = key.transpose(1, 2)     # [B, N_q, S_kv, D]
    v = value.transpose(1, 2)   # [B, N_q, S_kv, D]

    # 缩放点积注意力: [B, N_q, S, D] @ [B, N_q, D, S_kv] -> [B, N_q, S, S_kv]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # 加权求和: [B, N_q, S, S_kv] @ [B, N_q, S_kv, D] -> [B, N_q, S, D]
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N_q, D]
    y = attn_output.transpose(1, 2)

    # 输出投影 (可选)
    if weight_o is not None:
        # [B, S, N_q * D] @ [N_q * D, N_q * D] -> [B, S, N_q * D] -> [B, S, N_q, D]
        y_flat = y.reshape(B, S, N_q * D)
        y_flat = torch.nn.functional.linear(y_flat, weight_o)
        y = y_flat.reshape(B, S, N_q, D)

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

B, S, S_kv, D = 2, 128, 128, 64
numHeads, numKVHeads = 32, 8
query = torch.randn(B, S, numHeads, D, dtype=torch.float16, device="npu")
key = torch.randn(B, S_kv, numKVHeads, D, dtype=torch.float16, device="npu")
value = torch.randn(B, S_kv, numKVHeads, D, dtype=torch.float16, device="npu")
y = ascend_bench.gqa(query, key, value, None,
                      numHeads=numHeads, numKVHeads=numKVHeads, scaleValue=-1.0)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **MHA**：标准多头注意力，GQA 在 numKVHeads == numHeads 时退化为 MHA
- **MLA**：多头潜在注意力，通过低秩压缩 KV 缓存降低推理内存
- **SparseFlashAttention**：稀疏注意力计算，大序列长度推理场景的高效注意力计算
