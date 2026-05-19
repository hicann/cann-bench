# RmsNorm 算子 API 描述

## 1. 算子简介

计算 RMS (均方根) 归一化。

**主要应用场景**：
- 大语言模型中的归一化层（LLaMA、Gemma 等使用 RMSNorm 替代 LayerNorm）
- Transformer 架构中的预归一化（Pre-Norm）
- 相比 LayerNorm 省去均值计算，推理效率更高

**算子特征**：
- 难度等级：L2（Normalization）
- 双输入（x 和 gamma）单输出，涉及平方、均值、开方、除法、乘法等多步计算
- 沿最后一维进行归一化，gamma 为可学习的缩放参数

## 2. 算子定义

### 数学公式

**基本公式**：

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

展开为：

$$
y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2 + \epsilon}} \cdot \gamma_i
$$

其中：
- `D` 为最后一维的大小（归一化维度）
- `epsilon` 为数值稳定性参数，防止除零
- `gamma` 为逐元素的缩放参数，shape 为 (D,)
- 与 LayerNorm 不同，RMSNorm 不计算均值，也没有偏置（beta）参数

## 3. 接口规范

### 算子原型

```python
cann_bench.rms_norm(Tensor x, Tensor gamma, float epsilon) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| gamma | Tensor | 必选 | 缩放参数，shape 为输入最后一维大小 |
| epsilon | float | 1e-6 | 数值稳定性参数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | RMS 归一化后的张量 |

### 数据类型

| x dtype | gamma dtype | 输出 dtype |
|---------|------------|-----------|
| float16 | float16 | float16 |
| float32 | float32 | float32 |
| bfloat16 | bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 为 (..., D)，gamma 的 shape 为 (D,)，其中 D 为最后一维大小
- gamma 的 dtype 需与 x 一致
- epsilon 为正数，通常取 1e-6 或 1e-5
- 需注意数值稳定性：当输入值极小时，mean(x^2) 可能下溢；当输入值极大时，x^2 可能溢出

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 2 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| `D`（最后一维/归一化维度） | 1 ~ 16384 | cases.csv 实测 2 ~ 8192；`gamma` 的 shape 必须为 `(D,)` |
| 前导维度乘积 `S = N0*N1*...` | 1 ~ 2097152 | cases.csv 实测 231 ~ 1000003 |
| `gamma` 维度数 | 1 | 固定为 1 维 |
| `epsilon` | 1e-12 ~ 1 | cases.csv 实测 1e-12 ~ 1e-3；须为正数，常用 1e-6 / 1e-5 |

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
RmsNorm 算子 Torch Golden 参考实现

计算 RMS (均方根) 归一化

公式:
    y = x / sqrt(mean(x^2) + eps) * gamma

参考论文: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467

Parameters:
    - x: (..., D) 输入张量，最后一维为归一化维度
    - gamma: (D,) 缩放参数
    - epsilon: float, 默认 1e-6 - 数值稳定性参数
"""


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算 RMS (均方根) 归一化

    Args:
        x: 输入张量，shape (..., D)
           最后一维 D 为归一化维度
        gamma: 缩放参数，shape (D,)
               与输入最后一维大小相同
        epsilon: 数值稳定性参数，防止除零
                 默认值 1e-6

    Returns:
        RMS 归一化后的张量，shape 与输入相同

    Examples:
        >>> x = torch.randn(32, 128, 4096)
        >>> gamma = torch.ones(4096)
        >>> y = rms_norm(x, gamma, epsilon=1e-6)
    """
    # fp16 / bf16 输入升 fp32 计算，避免 |x|>256 时 x**2 溢出
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
        gamma = gamma.to(torch.float32)
    # 计算均方根
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    # 归一化并乘以缩放参数
    y = x / rms * gamma

    return y.to(out_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(32, 128, 4096, dtype=torch.float32, device="npu")
gamma = torch.ones(4096, dtype=torch.float32, device="npu")

y = cann_bench.rms_norm(x, gamma, epsilon=1e-6)
y = cann_bench.rms_norm(x, gamma, epsilon=1e-5)
```
