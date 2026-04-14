# MLA 算子 API 描述

## 1. 算子简介

多头潜在注意力 (Multi-Head Latent Attention) 算子，通过低秩压缩 KV 缓存降低推理内存，使用潜在向量 (latent vector) 进行注意力计算，是 DeepSeek-V2 等模型的核心注意力机制。

**主要应用场景**：
- 大语言模型推理中的高效 KV 缓存压缩（如 DeepSeek-V2、DeepSeek-V3）
- 超长序列推理场景中大幅降低 KV cache 内存占用
- 需要在推理效率和模型质量之间取得平衡的大规模 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（query, compressed_kv, w_uk, w_uv）单输出，融合 KV 解压缩与多头注意力计算
- 支持可配置的头数和缩放因子，通过低秩压缩显著减少 KV 缓存大小

## 2. 算子定义

### 数学公式

$$
K = c_{kv} \times W_{uk}
$$

$$
V = c_{kv} \times W_{uv}
$$

$$
y = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d}}\right) \times V
$$

其中：
- $c_{kv}$ 为低秩压缩的 KV 缓存，维度 $D_c < N \times D$
- $W_{uk}$ 为 key 解压缩权重，$W_{uv}$ 为 value 解压缩权重
- $Q$ 为查询张量，$K$ 和 $V$ 从压缩表示中解压缩得到
- $\sqrt{d}$ 为缩放因子（由 `scaleValue` 参数指定，<=0 时自动使用 $1/\sqrt{D}$）
- softmax 沿最后一维计算

具体子步骤：
1. **KV 解压缩**：$K = c_{kv} \times W_{uk}$，将 $[B, S_{kv}, D_c]$ 映射到 $[B, S_{kv}, N, D]$
2. **KV 解压缩**：$V = c_{kv} \times W_{uv}$，将 $[B, S_{kv}, D_c]$ 映射到 $[B, S_{kv}, N, D]$
3. **缩放点积**：$\text{scores} = Q \times K^T \times \text{scaleValue}$
4. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
5. **加权求和**：$y = \text{attn\_weights} \times V$

## 3. 接口规范

### 算子原型

```python
ascend_bench.mla(Tensor query, Tensor compressed_kv, Tensor w_uk, Tensor w_uv, int numHeads, float scaleValue) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| query | Tensor | 必选 | 查询张量，shape 为 [B, S, N, D] |
| compressed_kv | Tensor | 必选 | 低秩压缩的 KV 缓存，shape 为 [B, S_kv, D_c]，D_c < N*D |
| w_uk | Tensor | 必选 | key 解压缩权重，shape 为 [D_c, N, D] |
| w_uv | Tensor | 必选 | value 解压缩权重，shape 为 [D_c, N, D] |
| numHeads | int | 必选 | 注意力头数 |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(D) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, N, D] | 与输入 query 相同 | 多头潜在注意力输出张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor（query, compressed_kv, w_uk, w_uv）的 dtype 必须一致
- `query` 的 shape 为 [B, S, N, D]，N 为注意力头数，D 为每头维度
- `compressed_kv` 的 shape 为 [B, S_kv, D_c]，其中 D_c 为压缩维度，须满足 D_c < N * D
- `w_uk` 和 `w_uv` 的 shape 均为 [D_c, N, D]，用于将压缩表示解压缩到多头维度
- `scaleValue` 通常设置为 $1/\sqrt{D}$，当 <= 0 时自动使用该值
- 压缩比 D_c / (N * D) 越小，KV cache 内存节省越多，但可能影响注意力精度

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
MLA算子Torch Golden参考实现

多头潜在注意力 (Multi-Head Latent Attention)，通过低秩压缩 KV 缓存降低推理内存，使用潜在向量 (latent vector) 进行注意力计算
公式:
    c_kv = x @ W_dkv  (低秩压缩)
    K = c_kv @ W_uk   (解压缩得到 key)
    V = c_kv @ W_uv   (解压缩得到 value)
    y = softmax(Q @ K^T / sqrt(d)) @ V
"""
def mla(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    numHeads: int = 1,
    scaleValue: float = -1.0
) -> torch.Tensor:
    """
    多头潜在注意力 (Multi-Head Latent Attention)

    公式:
        K = compressed_kv @ W_uk  (解压缩得到 key)
        V = compressed_kv @ W_uv  (解压缩得到 value)
        y = softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        query: 查询张量 [B, S, N, D]
        compressed_kv: 低秩压缩的 KV 缓存 [B, S_kv, D_c]，D_c < N*D
        w_uk: key 解压缩权重 [D_c, N, D]
        w_uv: value 解压缩权重 [D_c, N, D]
        numHeads: 注意力头数
        scaleValue: 缩放因子，<=0时自动使用 1/sqrt(D)

    Returns:
        输出张量 [B, S, N, D]
    """
    B, S_kv, D_c = compressed_kv.shape
    B, S, N, D = query.shape

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 解压缩 KV: [B, S_kv, D_c] @ [D_c, N*D] -> [B, S_kv, N, D]
    key = torch.matmul(compressed_kv, w_uk.reshape(D_c, N * D)).reshape(B, S_kv, N, D)
    value = torch.matmul(compressed_kv, w_uv.reshape(D_c, N * D)).reshape(B, S_kv, N, D)

    # 转置为注意力计算格式
    q = query.transpose(1, 2)   # [B, N, S, D]
    k = key.transpose(1, 2)     # [B, N, S_kv, D]
    v = value.transpose(1, 2)   # [B, N, S_kv, D]

    # 缩放点积注意力: [B, N, S, D] @ [B, N, D, S_kv] -> [B, N, S, S_kv]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # 加权求和: [B, N, S, S_kv] @ [B, N, S_kv, D] -> [B, N, S, D]
    out = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]
    y = out.transpose(1, 2)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

B, S, S_kv, N, D, D_c = 2, 128, 256, 16, 64, 256
query = torch.randn(B, S, N, D, dtype=torch.float16, device="npu")
compressed_kv = torch.randn(B, S_kv, D_c, dtype=torch.float16, device="npu")
w_uk = torch.randn(D_c, N, D, dtype=torch.float16, device="npu")
w_uv = torch.randn(D_c, N, D, dtype=torch.float16, device="npu")
y = ascend_bench.mla(query, compressed_kv, w_uk, w_uv,
                      numHeads=N, scaleValue=-1.0)
```

### 性能基线参考

当前暂无测试用例和性能基线数据。

### 相关算子

- **MHA**：标准多头注意力，MLA 可视为其 KV 缓存压缩优化版本
- **GQA**：分组查询注意力，通过共享 KV head 减少内存，与 MLA 思路互补
- **MlaProlog**：Multi-Head Latent Attention 前处理，作为 MLA 计算的前置步骤
