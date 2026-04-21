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
- 当输入包含 inf 时，对应输出为 0 或 1；当输入包含 nan 时，输出为 nan

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

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，当前所有用例的 baseline_perf_us 均为 None，性能基线数据待补充。

### 相关算子

- **CrossEntropyLoss**：交叉熵损失函数，内部包含 log_softmax 计算
- **RMSNorm**：RMS 归一化算子，同属 Normalization 类别
- **GroupNorm**：分组归一化算子，同属 Normalization 类别
