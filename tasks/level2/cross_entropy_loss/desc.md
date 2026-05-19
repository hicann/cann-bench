# CrossEntropyLoss 算子 API 描述

## 1. 算子简介

计算交叉熵损失，用于分类任务。

**主要应用场景**：
- 多分类任务的损失函数（图像分类、文本分类等）
- 语言模型的 next-token 预测训练
- 支持硬标签（类别索引）和软标签（概率分布）两种模式

**算子特征**：
- 难度等级：L2（NumericalStable）
- 双输入（logits 和 target）单输出（loss），涉及 softmax、对数、归约等多步计算
- 输入 x 为 (N, C) 或更高维的 logits 张量，target 为 (N,) 的类别索引或 (N, C) 的软标签

## 2. 算子定义

### 数学公式

**基本公式**：

$$
L = -\log\left(\frac{\exp(x_{target})}{\sum_{j}\exp(x_j)}\right)
$$

等价于：

$$
L = -x_{target} + \log\left(\sum_{j}\exp(x_j)\right)
$$

**带权重的公式**：

$$
L = -weight_{target} \cdot \log\left(\frac{\exp(x_{target})}{\sum_{j}\exp(x_j)}\right)
$$

其中：
- `reduction='none'` 时返回每个样本的损失，shape 为 (N,)
- `reduction='mean'` 时返回 batch 平均损失（标量）
- `reduction='sum'` 时返回 batch 总损失（标量）
- `ignore_index` 指定的标签不参与损失计算

## 3. 接口规范

### 算子原型

```python
cann_bench.cross_entropy_loss(Tensor input, Tensor target, str reduction="mean", int ignore_index=-100) -> Tensor loss
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入 logits 张量（未经 softmax） |
| target | Tensor | 必选 | 目标标签索引（hard labels）或概率分布（soft labels） |
| reduction | string | "mean" | 损失聚合方式 ('none' \| 'mean' \| 'sum') |
| ignore_index | int | -100 | 忽略的标签索引（不影响损失计算） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| loss | reduction='none' 时为 (N,)，否则为标量 | 与 input 相同 | 损失值 |

### 数据类型

| x dtype | target dtype | 输出 dtype |
|---------|-------------|-----------|
| float32 | int32 / int64 | float32 |
| float16 | int32 / int64 | float16 |
| bfloat16 | int32 / int64 | bfloat16 |
| float32 | float32 | float32 |
| float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 为 (N, C) 或 (N, C, d1, d2, ...)，其中 N 为 batch size，C 为类别数
- 硬标签模式：target 的 shape 为 (N,) 或 (N, d1, d2, ...)，值为 [0, C) 范围内的类别索引
- 软标签模式：target 的 shape 为 (N, C)，值为概率分布
- `ignore_index` 仅在硬标签模式下生效
- 输入 x 应为原始 logits（未经 softmax），内部自动应用 log_softmax
- 需注意数值稳定性：内部实现应使用 log-sum-exp 技巧避免溢出

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch size，x 第 0 维） | 1 ~ 2097152 | cases.csv 实测 2 ~ 1,000,003 |
| `C`（类别数，x 第 1 维） | 2 ~ 16384 | cases.csv 实测 2 ~ 16,384 |
| 额外空间维度 `d_i`（x 第 ≥2 维） | 1 ~ 1024 | cases.csv 实测 3 ~ 1024 |
| `rank(x)`（x 维度数） | 2 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| `rank(target)` | rank(x)-1 或 rank(x) | 硬标签缺 C 维；软标签同 x；cases.csv 全为硬标签 |
| `reduction` | "none" / "mean" / "sum" | cases.csv 三种均覆盖 |
| `ignore_index` | int64 任意值 | cases.csv 实测 -100 / -1 / 0 / 10 / 50 / 100 |

约束：硬标签模式下 target 各元素取值范围为 [0, C) 或等于 `ignore_index`；软标签模式下 target 形状须与 x 完全一致。

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
CrossEntropyLoss 算子 Torch Golden 参考实现

计算交叉熵损失，用于分类任务

公式:
    L = -log(exp(x[target]) / sum(exp(x)))
    或带 weight: L = -weight[target] * log(exp(x[target]) / sum(exp(x)))

参考 PyTorch API: torch.nn.CrossEntropyLoss
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

Parameters:
    - input: (N, C) 或 (N, C, H, W) 等 - logits 张量（未经 softmax）
    - target: (N,) 硬标签 或 (N, C) 软标签（概率分布）
    - weight: (C,) 各类别的权重（可选）
    - ignore_index: int, 默认 -100 - 忽略的标签索引
    - reduction: 'none' | 'mean' | 'sum', 默认 'mean' - 损失聚合方式
"""


def cross_entropy_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算交叉熵损失

    Args:
        x: 输入 logits 张量，shape (N, C) 或 (N, C, d1, d2, ...)
           N = batch size, C = 类别数（channel_first 约定）
           注意：输入应为 logits（未经 softmax），内部会自动应用 log_softmax
        target: 目标标签
               - 硬标签：shape (N,) 或 (N, d1, d2, ...)，值为类别索引
               - 软标签：shape (N, C)，值为概率分布
        reduction: 损失聚合方式
                  'none': 返回每个样本的损失，shape (N,)
                  'mean': 返回 batch 平均损失
                  'sum': 返回 batch 总损失
        ignore_index: 忽略的标签索引
                      当 target 为硬标签且值为 ignore_index 时，该样本不计入损失

    Returns:
        损失值：如果 reduction='none'，返回 shape (N,) 的张量
               否则返回标量张量

    Examples:
        >>> N, C = 16, 10  # 16个样本，10个类别
        >>> x = torch.randn(N, C)
        >>> target = torch.randint(0, C, (N,))
        >>> loss = cross_entropy_loss(x, target)
    """
    # 直接调用 PyTorch 标准 CrossEntropyLoss 实现
    # torch.nn.functional.cross_entropy 内部会自动应用 log_softmax
    loss = torch.nn.functional.cross_entropy(
        input=x,
        target=target,
        reduction=reduction,
        ignore_index=ignore_index
    )

    return loss
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 2048, dtype=torch.float32, device="npu")
target = torch.randint(0, 2048, (1024,), dtype=torch.int64, device="npu")

loss = cann_bench.cross_entropy_loss(x, target, reduction="mean", ignore_index=-100)
loss = cann_bench.cross_entropy_loss(x, target, reduction="sum", ignore_index=-100)
loss = cann_bench.cross_entropy_loss(x, target, reduction="none", ignore_index=-100)
```
