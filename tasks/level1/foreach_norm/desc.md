# ForeachNorm 算子 API 描述

## 1. 算子简介

ForeachNorm 算子对输入张量列表（TensorList）的每个张量分别进行范数运算，支持多种范数阶数。

**主要应用场景**：
- 梯度裁剪中的梯度范数计算
- 优化器中的参数范数监控
- 模型正则化中的权重范数约束

**算子特征**：
- 难度等级：L1（Reduction）
- 输入为张量列表，对每个张量独立计算范数，输出为标量张量列表
- 支持 ND 格式输入

## 2. 算子定义

### 数学公式

**通用 p 范数**：

$$
y = \left(\sum_i |x_i|^p\right)^{1/p}
$$

### 常见范数

| 范数阶数 (scalar) | 公式 | 含义 |
|-------------------|------|------|
| 0 | $\sum_i \mathbb{1}(x_i \neq 0)$ | L0 范数（非零元素个数） |
| 1 | $\sum_i \|x_i\|$ | L1 范数（绝对值之和） |
| 2 | $\sqrt{\sum_i x_i^2}$ | L2 范数（欧氏距离） |
| inf | $\max_i \|x_i\|$ | 无穷范数（最大绝对值） |

## 3. 接口规范

### 算子原型

```python
cann_bench.foreach_norm(Tensor[] x, float scalar) -> Tensor[] y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor[] | 必选 | 输入张量列表（TensorList） |
| scalar | float | 必选 | 范数阶数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 每个元素为标量张量 | 与输入 dtype 相同 | 每个输入张量的范数结果列表 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入为 TensorList，列表中每个张量独立计算范数
- 列表中各张量的 dtype 须一致
- `scalar` 支持正数、负数、0、inf 等值
- 负阶范数要求输入元素不为零

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| TensorList 长度（`L`） | 1 ~ 64 | cases.csv 实测 1 ~ 4 |
| 每个张量维度数 | 1 ~ 8 | cases.csv 实测 1D ~ 5D |
| 每个张量各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 1000003 |
| 每个张量元素总数 | 1 ~ 48M | cases.csv 每个张量实测 ~1M ~ ~47M |
| `scalar`（范数阶数） | -1024.0 ~ 1024.0 | cases.csv 实测 -1.0 ~ 5.0（含 inf）；负阶或 0 阶时输入元素须非零 |

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

def foreach_norm(
    x: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对输入张量列表的每个张量进行范数运算

    公式：y = (sum |x_i|^p)^(1/p)

    Args:
        x: 输入张量列表 (TensorList)
        scalar: 范数阶数

    Returns:
        输出张量列表，每个张量的范数结果
    """

    y = [torch.norm(tensor, p=scalar) for tensor in x]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

t1 = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
t2 = torch.randn(2048, 512, dtype=torch.float32, device="npu")
y = cann_bench.foreach_norm([t1, t2], scalar=2.0)  # L2 范数
```
