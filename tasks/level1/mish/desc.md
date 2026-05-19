# Mish 算子 API 描述

## 1. 算子简介

Mish 是一种自正则化的非单调神经网络激活函数，具有平滑、非单调的特性，在部分场景下性能优于 ReLU 和 Swish。

**主要应用场景**：
- YOLOv4/v5 等目标检测模型的激活层
- 深层卷积网络中替代 ReLU 的激活函数
- 需要平滑梯度的深度学习模型

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致
- 支持 0~8 维输入

## 2. 算子定义

### 数学公式

$$
y = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x → +∞ | y → x（趋近恒等） |
| x → -∞ | y → 0 |

## 3. 接口规范

### 算子原型

```python
cann_bench.mish(Tensor x) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 0~8 维 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | Mish 激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- 无额外属性参数

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 (ndim) | 1 ~ 8 | cases.csv 实测 1 ~ 5 |
| `x` 单维大小 | 1 ~ 16384 | cases.csv 实测 2 ~ 8193 |
| `x` 总元素数 | 1 ~ 64M | cases.csv 实测 ~1M ~ 64M  |

约束：输出 shape 与输入 shape 完全一致，无 broadcasting；无额外属性。

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

def mish(
    x: torch.Tensor
) -> torch.Tensor:
    """
    自正则化的非单调神经网络激活函数

    公式: y = x * tanh(softplus(x))

    Args:
        x: 输入张量

    Returns:
        输出张量，Mish激活结果
    """

    softplus = torch.nn.functional.softplus(x)
    y = x * torch.tanh(softplus)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = cann_bench.mish(x)
```
