# MLA 算子 API 描述

## 1. 算子简介

多头潜在注意力 (Multi-Head Latent Attention) 算子，仅包含注意力计算部分（不含 KV 解压缩），是 DeepSeek-V2/V3 等模型的核心注意力机制。

**主要应用场景**：
- 大语言模型推理中的 MLA 注意力计算（如 DeepSeek-V2、DeepSeek-V3）
- 超长序列推理场景
- 需要在推理效率和模型质量之间取得平衡的大规模 Transformer 架构

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（q_nope, q_rope, k_nope, k_rope, v）单输出
- Q 和 K 均分为 nope 和 rope 两部分传入，内部拼接后计算注意力
- V 与 K_nope 共享 d_nope 维度，输出 head dim 为 d_nope
- 支持 GQA 模式（N_q 个 query head 共享 N_kv 个 KV head），MLA 典型配置 N_kv=1
- 支持 BSND 和 BNSD 两种输入输出 layout

## 2. 算子定义

### 数学公式

$$
Q = \text{concat}(Q_{nope}, Q_{rope}) \quad \text{dim: } d_{nope} + d_{rope}
$$

$$
K = \text{concat}(K_{nope}, K_{rope}) \quad \text{dim: } d_{nope} + d_{rope}
$$

$$
y = \text{softmax}\left(Q \times K^T \times \text{scaleValue}\right) \times V
$$

其中：
- $Q_{nope}$、$K_{nope}$ 为 nope 部分，维度 $d_{nope}$
- $Q_{rope}$、$K_{rope}$ 为 rope 部分（经过 RoPE 编码），维度 $d_{rope}$
- $V$ 的 head dim 为 $d_{nope}$，因此输出的 head dim 也为 $d_{nope}$
- $\text{scaleValue}$ 为缩放因子（<=0 时自动使用 $1/\sqrt{d_{nope} + d_{rope}}$）
- 当 $N_q > N_{kv}$ 时进行 GQA 扩展，每个 KV head 被 $N_q / N_{kv}$ 个 query head 共享

具体子步骤：
1. **Q 拼接**：$Q = \text{concat}(Q_{nope}, Q_{rope})$
2. **K 拼接**：$K = \text{concat}(K_{nope}, K_{rope})$
3. **GQA 扩展**：将 KV head 复制 $N_q / N_{kv}$ 次匹配 query head 数
4. **缩放点积**：$\text{scores} = Q \times K^T \times \text{scaleValue}$
5. **Softmax 归一化**：$\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$
6. **加权求和**：$y = \text{attn\_weights} \times V$

## 3. 接口规范

### 算子原型

```python
cann_bench.mla(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor v, int numKVHeads=1, float scaleValue=-1.0, str inputLayout="BSND") -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| q_nope | Tensor | 必选 | query 的 nope 部分，BSND: [B, S, N_q, d_nope]，BNSD: [B, N_q, S, d_nope] |
| q_rope | Tensor | 必选 | query 的 rope 部分，BSND: [B, S, N_q, d_rope]，BNSD: [B, N_q, S, d_rope] |
| k_nope | Tensor | 必选 | key 的 nope 部分，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope] |
| k_rope | Tensor | 必选 | key 的 rope 部分，BSND: [B, S_kv, N_kv, d_rope]，BNSD: [B, N_kv, S_kv, d_rope] |
| v | Tensor | 必选 | 值张量，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope] |
| numKVHeads | int | 1 | KV 头数 |
| scaleValue | float | -1.0 | 缩放因子，<=0 时自动使用 1/sqrt(d_nope + d_rope) |
| inputLayout | str | "BSND" | 输入输出 layout，"BSND" 或 "BNSD" |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | BSND: [B, S, N_q, d_nope]；BNSD: [B, N_q, S, d_nope] | 与输入相同 | 注意力输出张量，head dim 为 d_nope |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor 的 dtype 必须一致
- N_q 必须能被 N_kv 整除
- q_nope 和 k_nope 的 head dim 一致（d_nope），q_rope 和 k_rope 的 head dim 一致（d_rope）
- v 的 head dim = d_nope，输出的 head dim 也为 d_nope
- 所有输入和输出遵循相同的 layout（BSND 或 BNSD）

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
MLA算子Torch Golden参考实现

多头潜在注意力 (Multi-Head Latent Attention)，仅包含注意力计算部分
Q 和 K 均分为 nope 和 rope 两部分传入，内部拼接后计算注意力
V 与 K_nope 共享 d_nope 维度
支持 BSND 和 BNSD 两种输入 layout

公式:
    Q = concat(Q_nope, Q_rope)   dim: d_nope + d_rope
    K = concat(K_nope, K_rope)   dim: d_nope + d_rope
    y = softmax(Q @ K^T * scaleValue) @ V
"""


def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
) -> torch.Tensor:
    """
    多头潜在注意力 (Multi-Head Latent Attention)

    Args:
        q_nope: query 的 nope 部分，BSND: [B, S, N_q, d_nope]，BNSD: [B, N_q, S, d_nope]
        q_rope: query 的 rope 部分，BSND: [B, S, N_q, d_rope]，BNSD: [B, N_q, S, d_rope]
        k_nope: key 的 nope 部分，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]
        k_rope: key 的 rope 部分，BSND: [B, S_kv, N_kv, d_rope]，BNSD: [B, N_kv, S_kv, d_rope]
        v: 值张量，BSND: [B, S_kv, N_kv, d_nope]，BNSD: [B, N_kv, S_kv, d_nope]
        numKVHeads: KV 头数
        scaleValue: 缩放因子，<=0 时自动使用 1/sqrt(d_nope + d_rope)
        inputLayout: 输入 layout，"BSND" 或 "BNSD"

    Returns:
        输出张量，与输入 layout 一致，head dim 为 d_nope
    """
    # 统一转为 BSND 内部计算
    if inputLayout == "BNSD":
        q_nope = q_nope.permute(0, 2, 1, 3)
        q_rope = q_rope.permute(0, 2, 1, 3)
        k_nope = k_nope.permute(0, 2, 1, 3)
        k_rope = k_rope.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

    B, S, N_q, d_nope = q_nope.shape
    d_rope = q_rope.shape[-1]
    D_qk = d_nope + d_rope
    S_kv = k_nope.shape[1]
    N_kv = numKVHeads

    if scaleValue <= 0:
        scaleValue = 1.0 / (D_qk ** 0.5)

    # 拼接 Q = [Q_nope, Q_rope]: [B, S, N_q, d_nope + d_rope]
    q = torch.cat([q_nope, q_rope], dim=-1)

    # 拼接 K = [K_nope, K_rope]: [B, S_kv, N_kv, d_nope + d_rope]
    k = torch.cat([k_nope, k_rope], dim=-1)

    # GQA 扩展: 每个 KV head 复制 N_q // N_kv 次
    G = N_q // N_kv
    if G > 1:
        k = k.unsqueeze(3).expand(B, S_kv, N_kv, G, D_qk).reshape(B, S_kv, N_q, D_qk)
        v = v.unsqueeze(3).expand(B, S_kv, N_kv, G, d_nope).reshape(B, S_kv, N_q, d_nope)

    # 转置为 [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)  # [B, N_q, S, d_nope]

    # 转回 BSND: [B, S, N_q, d_nope]
    out = out.transpose(1, 2)

    # 按输入 layout 输出
    if inputLayout == "BNSD":
        out = out.permute(0, 2, 1, 3)

    return out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

B, S, S_kv = 2, 128, 256
N_q, N_kv = 128, 1
d_nope, d_rope = 512, 64

# BSND layout
q_nope = torch.randn(B, S, N_q, d_nope, dtype=torch.float16, device="npu")
q_rope = torch.randn(B, S, N_q, d_rope, dtype=torch.float16, device="npu")
k_nope = torch.randn(B, S_kv, N_kv, d_nope, dtype=torch.float16, device="npu")
k_rope = torch.randn(B, S_kv, N_kv, d_rope, dtype=torch.float16, device="npu")
v = torch.randn(B, S_kv, N_kv, d_nope, dtype=torch.float16, device="npu")
y = cann_bench.mla(q_nope, q_rope, k_nope, k_rope, v,
                      numKVHeads=N_kv, scaleValue=-1.0, inputLayout="BSND")
# y shape: [B, S, N_q, d_nope] = [2, 128, 128, 512]
```
