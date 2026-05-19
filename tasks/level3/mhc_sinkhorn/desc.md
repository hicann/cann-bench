# MhcSinkhorn 算子 API 描述

## 1. 算子简介

**mHC** (**Manifold-Constrained Hyper-Connections**) Sinkhorn 算子，对
`hc_mult × hc_mult` 的 stream 混合矩阵 `comb` 做 Sinkhorn-Knopp 双随机化投影，
使其落在 Birkhoff Polytope 流形（即"行和=1 且 列和=1"的双随机矩阵集合）上。
来源：DeepSeek-V4 用 mHC 机制替换 transformer 的标准 residual connection——
把残差通路扩展成 `hc_mult` 条并行 stream，stream 间通过受约束的 `comb` 矩阵交换
信息；矩阵被约束在 doubly-stochastic 流形上后 signal amplification 从 V3 的 4× 降到
~1.6×，使 1.6T 参数训练保持稳定（详见 [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)）。

**主要应用场景**：
- DeepSeek-V4 内 `DeepseekV4HyperConnection` 的 `comb` 矩阵 forward 投影
- 任何需要把方阵投影到双随机矩阵（doubly-stochastic / Birkhoff Polytope）的场景
- 软分配 / 软匹配问题的 Sinkhorn-Knopp 迭代求解

**算子特征**：
- 难度等级：L3（VVFusion）
- 单输入单输出，**方阵 inner 维**：`comb: [B, hc_mult, hc_mult]`，inner 两维是 `hc_mult × hc_mult` 的 hyper-connection mixing 方阵；**DSv4 实际取 `hc_mult=4`**（残差通路数），矩阵很小 (4 × 4)
- DSv4 production 默认 `iter_step=20`
- 算法在 **linear 域**进行——第 1 轮先做 row-softmax 把任意 logits 化为概率分布，
  后续 `iter_step - 1` 轮做交替 row / column 归一化以收敛到双随机矩阵

## 2. 算子定义

### 数学公式

设输入 $C \in \mathbb{R}^{B \times \text{hc\_mult} \times \text{hc\_mult}}$，每个 batch 独立处理。
记一轮 row-normalize 为 $R(\cdot)$、column-normalize 为 $C(\cdot)$：

$$
R(M)_{i,j} = \frac{M_{i,j}}{\sum_k M_{i,k} + \epsilon}, \quad
C(M)_{i,j} = \frac{M_{i,j}}{\sum_k M_{k,j} + \epsilon}
$$

算子做 `iter_step` 轮迭代，**第 1 轮的 row-normalize 用 softmax 实现**（把任意
real-valued logits 先转成概率分布，避免 0/负数破坏后续除法）：

$$
\begin{aligned}
\text{Iter 1:}\quad & C \leftarrow \mathrm{softmax}(C, \text{dim}{=}{-}1) + \epsilon \\
                    & C \leftarrow C / (\text{col\_sum}(C) + \epsilon) \\
\text{Iter 2..iter\_step:}\quad & C \leftarrow R(C) \\
                                & C \leftarrow C(C)
\end{aligned}
$$

### 处理流程

1. **First iteration — softmax + 列归一化**（把 raw 输入化为概率域）：
   - `row_max  = reduce_max(comb, dim=-1)` — 数值稳定 shift
   - `comb     = exp(comb - row_max)`
   - `row_sum  = reduce_sum(comb, dim=-1)`
   - `comb     = comb / row_sum + eps` ——完成 row-softmax 并加 `eps` 防 0
   - `col_sum  = reduce_sum(comb, dim=-2)`
   - `comb     = comb / (col_sum + eps)`
2. **Remaining `iter_step - 1` iterations — 纯 linear 双归一化**：
   - `row_sum  = reduce_sum(comb, dim=-1); comb = comb / (row_sum + eps)`
   - `col_sum  = reduce_sum(comb, dim=-2); comb = comb / (col_sum + eps)`
3. 返回 `comb`（已归一化的方阵，shape 与输入完全一致）。

> **与 log-domain Sinkhorn 的区别**：log-domain 方案的每轮迭代都是
> `x - logsumexp(x)`，全程在 log 空间运算、最后再 `exp` 一次；本算子直接
> 在 linear 域做 softmax + 除法迭代，无需最后再 exp。两种数学上等价
> （在无 `eps` 极限下），但本算子的实现走 linear 域，与 DSv4 kernel
> 一致——精度比对必须用 linear-domain golden。

## 3. 接口规范

### 算子原型

```python
cann_bench.mhc_sinkhorn(Tensor comb, int iter_step=20, float eps=1e-6) -> Tensor comb_out
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| comb | Tensor | 必选 | 输入 hyper-connection 混合矩阵，shape `[B, hc_mult, hc_mult]`，**inner 两维必须相等**；fp32 only |
| iter_step | int | **20** | Sinkhorn 迭代总轮数（第 1 轮含 softmax；其余 `iter_step - 1` 轮为纯 row/column normalize）；`iter_step ≥ 1` |
| eps | float | 1e-6 | 数值稳定项，所有除法分母上加该常量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| comb_out | `[B, hc_mult, hc_mult]`，与输入完全一致 | 与输入 comb 相同 (fp32) | Sinkhorn 迭代后的双随机方阵 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |

**只支持 fp32**：mHC 的核心约束（投影到 Birkhoff Polytope）对数值精度敏感，迭代 20+ 轮的累积舍入误差在 fp16 / bf16 下会破坏双随机性质（row_sum 漂移 > 1e-3）；DSv4 实际 kernel 也以 fp32 跑该算子，本算子规范上对齐。

### 规则与约束

- **输入 `comb` 必须为 3D 张量**，且 inner 两维相等（`shape[-1] == shape[-2]`），表示一组 `hc_mult × hc_mult` hyper-connection 混合方阵
- **dtype 限定 fp32**：iteration 20+ 轮的累积舍入误差对低精度敏感；fp16/bf16 下双随机性质会被破坏，故 spec 上拒绝
- `iter_step` 是 Sinkhorn 总迭代轮数；`iter_step = 1` 时只做"softmax + column normalize"一轮；DSv4 production 默认 `20`
- `eps` 必须为非负小常量；典型 1e-8 ~ 1e-6（fp32 下不会破坏数值稳定性）
- 算法在 **linear 域**进行，第 1 轮 row-softmax 自带数值稳定 shift（`exp(x - row_max)`），后续轮次直接做除法
- 输出与输入 shape / dtype 完全一致；不返回中间状态

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch / token×layer 维） | **1 ~ 16384** | cases.csv 实测 1 ~ 16384；DSv4 production 中通常等于 tokens 数 |
| `hc_mult`（hyper-connection 数量，方阵边长） | **2 ~ 16** | **DSv4 实际取 `hc_mult=4`**；矩阵本身很小 (`hc_mult × hc_mult`) |
| `iter_step` | **1 ~ 40** | 正整数；DSv4 默认 20，cases.csv 覆盖 1 / 5 / 20 / 40 |
| `eps` | 1e-8 ~ 1e-3 | 非负小常量；典型 1e-6 (fp32 default) |
| dtype | **fp32 only** | 见上节 "数据类型" 与 "规则与约束" 说明 |

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
MhcSinkhorn 算子 Torch Golden 参考实现 — linear-domain Sinkhorn-Knopp

来源对齐 DeepSeek-V4 mHC (Manifold-Constrained Hyper-Connections) 模块中
hyper-connection stream 混合矩阵的双随机化投影 kernel：第 1 轮做 row-softmax
（数值稳定地把 raw 输入转成概率分布）再做 column normalize；剩余 iter_step - 1
轮做交替 row / column 归一化，直到 row_sum 与 col_sum 都趋近 1（投影到 Birkhoff
Polytope 流形）。fp32 only。
"""
def mhc_sinkhorn(
    comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """
    mHC Sinkhorn — linear-domain doubly-stochastic projection on hc_mult × hc_mult matrices

    Args:
        comb: 输入 hyper-connection 混合矩阵，shape [B, hc_mult, hc_mult] fp32，inner 两维必须相等
        iter_step: Sinkhorn 总迭代轮数 (≥ 1)；DSv4 production 默认 20
        eps: 数值稳定项，所有除法分母上加该常量

    Returns:
        comb_out: 双随机化后的方阵，shape 与输入完全一致 (fp32)
    """
    assert comb.dtype == torch.float32, "mhc_sinkhorn 仅支持 fp32 输入"
    # First iter: row-softmax + eps，然后 column normalize
    # 等价于把 raw logits 转换成 row-probability 分布的起点
    row_max = comb.amax(dim=-1, keepdim=True)               # numerically-stable shift
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps                              # row-softmax + eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    # 后续 iter_step - 1 轮：linear-domain row + column normalize
    for _ in range(iter_step - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)

    return comb
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# DSv4 典型场景：B=1024 tokens，每个 token 一个 hc_mult=4 的 4×4 方阵 (fp32)
comb = torch.randn(1024, 4, 4, dtype=torch.float32, device="npu")
comb_out = cann_bench.mhc_sinkhorn(comb, iter_step=20)

# 单轮（仅 softmax + 1 次 column normalize）
comb_out = cann_bench.mhc_sinkhorn(comb, iter_step=1)

# 大 batch + 上界 iter_step (40 轮)
comb_large = torch.randn(16384, 4, 4, dtype=torch.float32, device="npu")
comb_out = cann_bench.mhc_sinkhorn(comb_large, iter_step=40)

# fp32 高精度场景下用更小的 eps
comb_out = cann_bench.mhc_sinkhorn(comb, iter_step=20, eps=1e-8)
```

### 参考文档

- DeepSeek-V4 mHC paper: [arXiv:2512.24880 — Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- DeepSeek-V4 HuggingFace 实现中 `DeepseekV4HyperConnection` 模块的 `(pre, post, comb)` triplet 在 forward 中用到本算子做 `comb` 的 doubly-stochastic 投影
