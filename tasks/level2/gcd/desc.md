# Gcd 算子 API 描述

## 1. 算子简介

计算两个整数的最大公约数。

**主要应用场景**：
- 整数约分与最简分数计算
- 数组维度对齐与分块策略中的公因子计算
- 密码学中的模运算与欧几里得算法相关场景

**算子特征**：
- 难度等级：L2（Broadcast）
- 双输入单输出，逐元素运算，输入支持广播

## 2. 算子定义

### 数学公式

$$
y = \gcd(x_1, x_2)
$$

其中 $\gcd(a, b)$ 表示 $a$ 与 $b$ 的最大公约数，即同时整除 $a$ 和 $b$ 的最大正整数。当 $a = b = 0$ 时，$\gcd(0, 0) = 0$。

## 3. 接口规范

### 算子原型

```python
cann_bench.gcd(Tensor x1, Tensor x2) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 第1个输入张量 |
| x2 | Tensor | 必选 | 第2个输入张量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 广播后的 shape | 与输入一致 | 输出张量，最大公约数 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int16 | int16 |
| int32 | int32 |
| int64 | int64 |

### 规则与约束

- 两个输入张量的 shape 需满足广播规则，输出 shape 为广播后的 shape
- 两个输入张量的 dtype 必须一致
- 仅支持整数类型（int16、int32、int64）
- Golden 实现使用 `torch.gcd` 直接计算，输出 dtype 与输入保持一致

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank`（输入张量维度数） | 1 ~ 8 | cases.csv 实测 1D ~ 5D；x1 与 x2 rank 可不同，按广播规则对齐 |
| `shape[i]`（每一维大小） | 1 ~ 2097152 | cases.csv 实测 1 ~ 1048583；x1 与 x2 对应维度需满足广播（相等或一方为 1） |
| 广播后总元素数 | 1 ~ 2^27 | cases.csv 实测最大约 67M（8192×8192，case 4） |
| 输入 dtype | int16 / int32 / int64 | x1/x2/y 必须同 dtype；不支持浮点 |
| `x1`/`x2` 元素值（int16） | -32768 ~ 32767 | cases.csv 实测覆盖 int16 完整范围（case 4） |
| `x1`/`x2` 元素值（int32） | -2^31 ~ 2^31-1 | cases.csv 实测覆盖 int32 完整范围（case 5） |
| `x1`/`x2` 元素值（int64） | -2^63 ~ 2^63-1 | cases.csv 实测 ≤ ±100000（case 3/9/20） |

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
Gcd算子Torch Golden参考实现

计算两个整数的最大公约数
公式: y = gcd(x1, x2)
"""


def gcd(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    计算两个整数的最大公约数

    公式: y = gcd(x1, x2)

    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量

    Returns:
        输出张量，最大公约数（dtype 与输入一致）
    """
    # NPU 的 int16 GCD 并行实现存在内存/线程同步问题
    # 使用 int32 计算避免 NPU 和 CPU 结果不一致
    if x1.dtype == torch.int16:
        x1_32 = x1.int()
        x2_32 = x2.int()
        result = torch.gcd(x1_32, x2_32)
        return result.to(torch.int16)

    # NPU broadcast GCD 存在 bug，先显式广播再计算
    x1_bc, x2_bc = torch.broadcast_tensors(x1, x2)
    # CPU torch.gcd overflows on int32 min (-2**31); compute in int64
    # and cast back so results match the mathematically correct value.
    if x1_bc.dtype == torch.int32:
        return torch.gcd(x1_bc.long(), x2_bc.long()).to(torch.int32)
    y = torch.gcd(x1_bc, x2_bc)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x1 = torch.randint(-1000, 1000, (1024, 1024), dtype=torch.int32, device="npu")
x2 = torch.randint(-1000, 1000, (1024, 1024), dtype=torch.int32, device="npu")
y = cann_bench.gcd(x1, x2)

# 广播场景
x1 = torch.randint(-100, 100, (2048, 512), dtype=torch.int16, device="npu")
x2 = torch.randint(-10, 10, (1, 512), dtype=torch.int16, device="npu")
y = cann_bench.gcd(x1, x2)
```
