# Gelu 算子 API 描述

## 1. 算子简介

Gelu（高斯误差线性单元）是一种广泛应用于 Transformer 架构的激活函数，支持精确计算和 tanh 近似两种模式。

**主要应用场景**：
- BERT、GPT 等 Transformer 模型的前馈网络激活层
- Vision Transformer (ViT) 中的 MLP 模块
- 各类预训练语言模型的中间激活

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致
- 支持 0~8 维输入

## 2. 算子定义

### 数学公式

**精确模式**（approximate="none"）：

$$
y = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

**tanh 近似模式**（approximate="tanh"）：

$$
y = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x → +∞ | y → x |
| x → -∞ | y → 0 |

## 3. 接口规范

### 算子原型

```python
cann_bench.gelu(Tensor x, str approximate="none") -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 0~8 维 |
| approximate | str | "none" | GELU 近似计算算法，可选值：'none'（精确计算）或 'tanh'（tanh 近似） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | GELU 激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- `approximate` 参数仅支持 "none" 和 "tanh" 两种取值
- 输入支持 0~8 维

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 0 ~ 8 | cases.csv 实测 1D ~ 5D；接口规范支持 0 ~ 8 维 |
| `x` 各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 8192（含 1D 张量长度 1000003） |
| `x` 元素总数 | 1 ~ 64M | cases.csv 实测 ~1M ~ 64M |
| `approximate` | {"none", "tanh"} | cases.csv 两种取值均覆盖；仅支持这两种字符串 |

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

def gelu(
    x: torch.Tensor,
    approximate: str = "none"
) -> torch.Tensor:
    """
    高斯误差线性单元激活函数

    公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: 输入张量
        approximate: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)

    Returns:
        输出张量，GELU 激活结果
    """

    y = torch.nn.functional.gelu(x, approximate=approximate)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = cann_bench.gelu(x)                          # 精确模式
y = cann_bench.gelu(x, approximate="tanh")       # tanh 近似模式
```
