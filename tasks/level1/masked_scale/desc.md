# MaskedScale 算子 API 描述

## 1. 算子简介

MaskedScale 算子对输入张量进行掩码缩放操作，支持输入张量 x 和掩码张量 mask 的不同数据类型组合。

**主要应用场景**：
- Transformer 中的注意力掩码缩放
- Dropout 的掩码乘法实现
- 条件计算中的选择性缩放

**算子特征**：
- 难度等级：L1（MaskPredicate）
- 双输入单输出，逐元素运算，输入 x、mask 和输出 y 的 shape 需一致

## 2. 算子定义

### 数学公式

$$
y = x \cdot mask \cdot scale
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.masked_scale(Tensor x, Tensor mask, float scale) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| mask | Tensor | 必选 | 掩码张量，shape 须与 x 一致 |
| scale | float | 1.0 | 缩放因子 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | 掩码缩放结果 |

### 数据类型

x 与 mask 支持不同 dtype 的组合：

| x dtype | mask dtype | y dtype |
|---------|-----------|---------|
| float16 | int8 / uint8 / float16 / bfloat16 / float32 | float16 |
| bfloat16 | int8 / uint8 / float16 / bfloat16 / float32 | bfloat16 |
| float32 | int8 / uint8 / float16 / bfloat16 / float32 | float32 |

### 规则与约束

- 输入 x、mask 的 shape 必须完全一致，输出 y 的 shape 也与之相同
- 输出 dtype 与输入 x 的 dtype 一致
- mask 通常取值 0 或 1，但不限于此（也支持连续值掩码）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` / `mask` 维度数 | 1 ~ 8 | cases.csv 实测 1D ~ 5D；mask 与 x 同 shape |
| `x` / `mask` 各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 1000007 |
| `x` / `mask` 元素总数 | 1 ~ 64M | cases.csv 实测 ~1M ~ ~64M |
| `scale` | -1024.0 ~ 1024.0 | cases.csv 实测 -1.0 ~ 10.0（含 inf / nan 特殊值） |

约束：`mask` 与 `x` shape 必须完全一致；输出 dtype 与 `x` dtype 相同。

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

def masked_scale(
    x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    对输入张量进行掩码缩放，支持x和mask的不同数据类型组合

    公式: y = x * mask * scale

    Args:
        x: 输入张量
        mask: 掩码张量
        scale: 缩放因子

    Returns:
        掩码缩放结果
    """

    y = x * mask * scale
    return y.to(x.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
mask = torch.randint(0, 2, (1024, 1024), dtype=torch.int8, device="npu")
y = cann_bench.masked_scale(x, mask, scale=2.0)
```
