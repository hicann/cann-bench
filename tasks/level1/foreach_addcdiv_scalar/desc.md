# ForeachAddcdivScalar 算子 API 描述

## 1. 算子简介

ForeachAddcdivScalar 算子对多个张量列表进行逐元素的加、除、乘复合操作，是优化器（如 Adam）中常用的基础运算。

**主要应用场景**：
- Adam / AdamW 优化器的参数更新步骤
- 需要对多组参数同时执行 addcdiv 运算的场景
- 分布式训练中的批量参数更新

**算子特征**：
- 难度等级：L1（FusedComposite）
- 三组 TensorList 输入，逐元素复合运算，输出 TensorList 与输入 shape 一致

## 2. 算子定义

### 数学公式

对列表中第 $i$ 个张量：

$$
y_i = x1_i + \frac{x2_i}{x3_i} \cdot scalar
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[] y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor[] | 必选 | 第 1 个输入张量列表（TensorList），被加数 |
| x2 | Tensor[] | 必选 | 第 2 个输入张量列表（TensorList），被除数的分子 |
| x3 | Tensor[] | 必选 | 第 3 个输入张量列表（TensorList），被除数的分母 |
| scalar | float | 必选 | 缩放因子 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 TensorList 各元素 shape 相同 | 与输入 dtype 相同 | 逐元素复合运算结果列表 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x1、x2、x3 三个 TensorList 长度必须相同
- 对应位置的张量 shape 必须一致
- 列表中各张量的 dtype 须一致
- x3 中的元素不应为零（除以零会产生 inf/nan）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| TensorList 长度（`L`） | 1 ~ 64 | cases.csv 实测 1 ~ 4；x1/x2/x3 三个列表长度必须相同 |
| 每个张量维度数 | 1 ~ 8 | cases.csv 实测 1D ~ 5D |
| 每个张量各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 8193（含 1D 张量长度 1000003） |
| 每个张量元素总数 | 1 ~ 64M | cases.csv 实测 ~1M ~ 64M |
| `scalar` | -1024.0 ~ 1024.0 | cases.csv 实测 -1.0 ~ 2.0（含 inf / nan 特殊值） |

约束：x1[i]、x2[i]、x3[i] 三者 shape 与 dtype 必须一致；x3 中元素应非零。

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
from typing import List

def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对多个张量进行逐元素加、乘、除操作

    公式：y_i = x1_i + (x2_i / x3_i) * scalar

    Args:
        x1: 第 1 个输入张量列表 (TensorList)
        x2: 第 2 个输入张量列表 (TensorList)
        x3: 第 3 个输入张量列表 (TensorList)
        scalar: 缩放因子

    Returns:
        输出张量列表
    """

    y = [x1_i + (x2_i / x3_i) * scalar for x1_i, x2_i, x3_i in zip(x1, x2, x3)]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x1 = [torch.randn(1024, 1024, dtype=torch.float32, device="npu")]
x2 = [torch.randn(1024, 1024, dtype=torch.float32, device="npu")]
x3 = [torch.rand(1024, 1024, dtype=torch.float32, device="npu") + 0.1]  # 避免除零
y = cann_bench.foreach_addcdiv_scalar(x1, x2, x3, scalar=1.0)
```
