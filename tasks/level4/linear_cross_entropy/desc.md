# LinearCrossEntropy 算子 API 描述

## 1. 算子简介

融合 LM-head 投影与交叉熵损失的复合算子。LLM 训练中损失计算的标准路径是先做 `logits = hidden @ weight^T`（[T, H] × [H, V]），再对 [T, V] 的 logits 做 softmax 交叉熵。现代模型词表 V 已达 12.8 万 ~ 15.2 万（LLaMA3 128256、Qwen2 152064、DeepSeek-V3 129280），T×V 的 fp32 logits 单矩阵即数 GB，是训练显存的最大单项开销之一。

LinearCrossEntropy 将两步融合为单一算子：**kernel 侧沿词表维分块计算，用在线 softmax（online logsumexp）维护每个 token 的运行最大值与指数和，同时提取目标 logit，全程不落盘 T×V logits**。这既是本算子的融合意义，也是主要难度来源——需要将 CUBE 矩阵乘分块与 VEC 在线归约紧密流水。

**主要应用场景**：
- LLM 预训练 / 微调中的语言建模损失（大词表场景显存瓶颈）
- 知识蒸馏、RLHF 等需要频繁评估 token 级 NLL 的训练流程
- Fused Linear Cross Entropy（Liger-Kernel、torch-titan 等训练框架的标配融合）

**算子特征**：
- 难度等级：L4（FusedComposite）
- 三输入（hidden、weight、labels）单输出（标量 loss）
- 融合 CUBE 矩阵乘、在线 softmax 归约、目标 logit gather 与 reduction 聚合
- 输出为全局标量，kernel 需做跨核归约；mean 归一还需统计有效 token 数
- 中间 logits 规模 T×V 远大于输入输出总量，不落盘 logits 是性能与显存的关键

## 2. 算子定义

### 数学公式

**投影**（fp32 累加）：

$$
\text{logits} = \text{hidden} \cdot \text{weight}^T \in \mathbb{R}^{T \times V}
$$

**逐 token 交叉熵**（数值稳定的 log-sum-exp 形式）：

$$
\ell_t = -\text{logits}[t, y_t] + \log \sum_{v=1}^{V} \exp(\text{logits}[t, v]), \qquad y_t = \text{labels}[t]
$$

labels[t] == ignore_index 的 token 不参与计算（$\ell_t$ 记 0 且不计入有效 token 数）。

**聚合**：

$$
\text{loss} = \begin{cases}
\dfrac{\sum_t \ell_t \cdot [y_t \ne \text{ignore\_index}]}{\sum_t [y_t \ne \text{ignore\_index}]} & \text{reduction} = \text{"mean"} \\[2ex]
\sum_t \ell_t \cdot [y_t \ne \text{ignore\_index}] & \text{reduction} = \text{"sum"}
\end{cases}
$$

### 计算子步骤

1. **分块投影**：沿词表维取块 `weight[v0:v1]`，计算块内 logits `hidden @ weight[v0:v1]^T`（CUBE，fp32 累加）
2. **在线 softmax 归约**：对每个 token 维护运行最大值 m 与指数和 s，块间按 `s = s_old * exp(m_old - m_new) + s_blk * exp(m_blk - m_new)` 合并（VEC）
3. **目标 logit 提取**：labels[t] 落在当前块内时记录 `logits[t, labels[t]]`（gather）
4. **逐 token 损失**：`ℓ_t = -target_logit_t + m_t + log(s_t)`，ignore_index 的 token 置 0
5. **全局聚合**：按 reduction 求 sum 或按有效 token 数求 mean，跨核归约输出标量

### 与非融合实现的对比

| 项目 | 非融合（matmul + cross_entropy） | LinearCrossEntropy |
|------|----------------------------------|--------------------|
| 中间显存 | T×V fp32 logits（LLaMA3、T=8192 时约 4.2 GB） | O(T) 运行统计量 |
| 访存 | logits 写出 + 读回各一次 | logits 只存在于片上 |
| softmax | 两遍（max、sum）或 log_softmax 全量 | 单遍在线归约 |
| 输出 | 同 | 同（数值等价，fp32） |

## 3. 接口规范

### 算子原型

```python
linear_cross_entropy(Tensor hidden, Tensor weight, Tensor labels, str reduction="mean", int ignore_index=-100) -> Tensor loss
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| hidden | Tensor | 是 | bfloat16/float16/float32 | [T, H] | 最后一层 hidden states（LM-head 输入） |
| weight | Tensor | 是 | 与 hidden 一致 | [V, H] | LM-head 投影权重（词表方向在前），logits = hidden @ weight^T |
| labels | Tensor | 是 | int32 | [T] | 目标 token id，取值 [0, V-1] 或等于 ignore_index |
| reduction | str | 否 | - | - | 聚合方式，"mean"（默认，按有效 token 数归一）或 "sum" |
| ignore_index | int | 否 | - | - | 忽略的标签值，默认 -100 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| loss | float32 | [1] | 聚合后的交叉熵损失（无论输入 dtype，固定 fp32 输出） |

### 数据类型

| hidden/weight dtype | labels dtype | 输出 dtype | 内部计算 |
|---------------------|-------------|-----------|---------|
| bfloat16 | int32 | float32 | fp32 |
| float16 | int32 | float32 | fp32 |
| float32 | int32 | float32 | fp32 |

### 规则与约束

- hidden 与 weight 的 dtype 必须一致；labels 固定 int32
- hidden 的第二维与 weight 的第二维必须相等（均为 H）
- labels 的取值必须落在 [0, V-1] 或等于 ignore_index（评测框架按 value_range 独立随机生成 labels，值域约束由 value_range 保证）
- reduction 仅支持 "mean" / "sum"（不支持 "none"，输出固定为标量 [1]）
- "mean" 按**有效 token 数**（labels != ignore_index 的数量）归一；调用方需保证至少存在一个有效 token，否则结果为 NaN
- 矩阵乘与 log-sum-exp 归约全程 fp32；低精度输入升精度参与计算
- kernel 实现**不得物化完整 T×V logits**（性能约束）；数值上须与 Golden 的"全量 logits + cross_entropy"实现等价

### 支持范围

| 维度 / 参数 | 支持值 | 备注 |
|---|---|---|
| `T`（token 数） | 1 ~ 32768 | 热点区间 1024 ~ 8192；大词表（V ≥ 128K）与大 T 不同时取极值，控制单 case 中间量在数 GB 内 |
| `H`（hidden 维） | 63 ~ 8192（含非对齐值） | 热点枚举 {3584, 4096, 7168, 8192}：Qwen2-7B / LLaMA3-8B、LLaMA2-7B / DeepSeek-V3 / Qwen2-72B |
| `V`（词表大小） | 100 ~ 200000（含质数 / 非对齐值） | 热点枚举 {32000, 128256, 129280, 152064}：LLaMA2 / LLaMA3 / DeepSeek-V3 / Qwen2 |
| `reduction` | {"mean", "sum"} | cases.csv 两种均覆盖 |
| `ignore_index` | 任意 int | cases.csv 实测 -100（默认）、-1、0（配合窄 labels 范围触发批量忽略）与 999999（不命中任何 label） |
| `hidden` 取值 | [-1, 1] 典型 | 鲁棒性 case 使用低幅值变体（低至 [-0.001, 0.001]） |
| `weight` 取值 | [-0.02, 0.02] 典型 | 模拟 LM-head 初始化尺度，logits 幅值 O(1)；鲁棒性 case 使用低幅值变体（低至 [-0.001, 0.001]） |
| `labels` 取值 | [0, V-1] | int32 有效索引 |
| dtype | bfloat16 / float16 / float32 | cases.csv 三种均覆盖 |

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
import torch.nn.functional as F

"""
LinearCrossEntropy算子Torch Golden参考实现

融合 LM-head 线性投影与交叉熵损失：logits = hidden @ weight^T，
loss = CrossEntropy(logits, labels)，支持 reduction (mean/sum) 与 ignore_index。
kernel 侧应分块计算、在线 softmax，不物化完整 [T, V] logits。
plain golden 以 fp32 计算 logits 与损失（bench 语义）；
linear_cross_entropy_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""


def _linear_cross_entropy_core(
    hidden, weight, labels, reduction, ignore_index, compute_dtype
):
    """Token-chunked reference: retain full vocabulary and global valid-token reduction."""
    if hidden.shape[0] == 0 or reduction not in ("mean", "sum"):
        # Retain legacy edge behavior without extending the declared task domain.
        logits = torch.matmul(hidden.to(compute_dtype), weight.to(compute_dtype).t())
        return F.cross_entropy(
            logits, labels.long(), reduction=reduction, ignore_index=ignore_index
        ).reshape(1)
    weight_c = weight.to(compute_dtype)
    labels_c = labels.long()
    losses = []
    # This controls logits materialization only, not the already-promoted weight.
    for start in range(0, hidden.shape[0], 128):
        logits = torch.matmul(
            hidden[start : start + 128].to(compute_dtype), weight_c.t()
        )
        losses.append(
            F.cross_entropy(
                logits,
                labels_c[start : start + 128],
                reduction="none",
                ignore_index=ignore_index,
            )
        )
    total = torch.cat(losses).sum()
    if reduction == "mean":
        total = total / (labels_c != ignore_index).sum()
    return total.reshape(1)


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    融合 LM-head 投影 + 交叉熵损失（plain golden = bench：fp32 计算）

    Args:
        hidden: [T, H] 最后一层 hidden states, bfloat16/float16/float32
        weight: [V, H] LM-head 投影权重, dtype 与 hidden 一致
        labels: [T] 目标 token id (int32), 取值 [0, V-1] 或等于 ignore_index
        reduction: 聚合方式, "mean"（按有效 token 数归一）或 "sum"
        ignore_index: 忽略的标签值, 默认 -100

    Returns:
        loss: [1] 聚合后的交叉熵损失, float32
    """
    return _linear_cross_entropy_core(
        hidden, weight, labels, reduction, ignore_index, torch.float32
    )


def linear_cross_entropy_oracle(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _linear_cross_entropy_core(
        hidden, weight, labels, reduction, ignore_index, hidden.dtype
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch

T, H, V = 2048, 4096, 128256  # LLaMA3-8B 配置

hidden = torch.randn(T, H, dtype=torch.bfloat16, device="npu")
weight = torch.empty(V, H, dtype=torch.bfloat16, device="npu").uniform_(-0.02, 0.02)
labels = torch.randint(0, V, (T,), dtype=torch.int32, device="npu")

loss = linear_cross_entropy(hidden, weight, labels, reduction="mean", ignore_index=-100)
# loss.shape: [1], dtype: float32
```
