# Softmax 算子 API 描述

## 1. 算子简介

沿指定维度计算 Softmax 归一化。

**主要应用场景**：
- 分类模型的输出层，将 logits 转换为概率分布
- 注意力机制中计算注意力权重
- 强化学习中的策略概率输出

**算子特征**：
- 难度等级：L2（Normalization）
- 单输入单输出，涉及指数运算、求和、除法等多步计算
- 输出元素值在 [0, 1] 范围内，沿指定维度求和为 1

## 2. 算子定义

### 数学公式

**基本公式**：

$$
y_i = \frac{\exp(x_i)}{\sum_{j}\exp(x_j)}
$$

数值稳定版本（内部实现）：

$$
y_i = \frac{\exp(x_i - \max(x))}{\sum_{j}\exp(x_j - \max(x))}
$$

其中：
- `x_i` 为输入张量沿指定 dim 维度上的第 i 个元素
- 输出满足 `0 <= y_i <= 1` 且 `sum(y) = 1`（沿 dim 维度）
- 数值稳定版本减去最大值以避免指数溢出

## 3. 接口规范

### 算子原型

```python
cann_bench.softmax(Tensor x, int dim) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| dim | int | -1 | 计算 Softmax 的维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | Softmax 归一化后的张量 |

### 数据类型

| x dtype | 输出 dtype |
|---------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 可以为任意维度的张量
- dim 指定计算 Softmax 的维度，支持负数索引
- 输出 shape 和 dtype 与输入完全一致
- 需注意数值稳定性：内部实现应使用减最大值技巧避免指数溢出
- 特殊值行为（须与 `torch.nn.functional.softmax` 一致，均源自 `x - max(x)` 重整）：
  - 某切片含任意 `+inf`（无论是否同时含其它有限/`-inf` 元素）：整切片输出 `NaN`（`inf - inf = NaN` 沿切片传播）
  - 某切片全部为 `-inf`：整切片输出 `NaN`（`max = -inf`，`-inf - (-inf) = NaN`）
  - 某切片含 `-inf` 与有限元素：`-inf` 位置输出 `0`，有限元素按正常 softmax 归一化
  - 输入含 `NaN`：整切片输出 `NaN`

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 1 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| `x` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 2 ~ 1000003 |
| `dim` | `[-rank, rank-1]` | cases.csv 实测 -1 / 0 / 1 / 2；支持负数索引 |

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
Softmax 算子 Torch Golden 参考实现

沿指定维度计算 Softmax 归一化

公式:
    y_i = exp(x_i) / sum(exp(x_j))

参考 PyTorch API: torch.nn.functional.softmax
    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

Parameters:
    - x: 任意维度输入张量
    - dim: int, 默认 -1 - 计算 Softmax 的维度
"""


def softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    沿指定维度计算 Softmax 归一化

    Args:
        x: 输入张量，任意 shape
        dim: 计算 Softmax 的维度，默认为 -1（最后一维）

    Returns:
        Softmax 归一化后的张量，shape 与输入相同
        输出元素值在 [0, 1] 范围内，且沿 dim 维度求和为 1

    Examples:
        >>> x = torch.randn(1024, 2048)
        >>> y = softmax(x, dim=-1)
    """
    y = torch.nn.functional.softmax(x, dim=dim)

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 2048, dtype=torch.float32, device="npu")

y = cann_bench.softmax(x, dim=-1)
y = cann_bench.softmax(x, dim=0)
y = cann_bench.softmax(x, dim=1)
```
