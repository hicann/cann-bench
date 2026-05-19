# Cummin 算子 API 描述

## 1. 算子简介

计算输入张量中的累积最小值。

**主要应用场景**：
- 时间序列分析中的滑动最小值追踪
- 单调约束优化问题中的前缀最小值计算
- 动态规划中的状态转移辅助操作

**算子特征**：
- 难度等级：L2（Reduction）
- P1 op：直接对齐 `torch.cummin` 官方接口，单输入双输出 `(values, indices)`
- 沿指定轴进行累积归约操作；values 与 indices 的 shape 都与 input 相同

## 2. 算子定义

### 数学公式

$$
y[i] = \min(x[0], x[1], \ldots, x[i]) \quad \text{沿指定轴}
$$

即对于输出的第 $i$ 个位置，其值为输入在指定轴上从位置 0 到位置 $i$ 的所有元素中的最小值。

## 3. 接口规范

### 算子原型

```python
cann_bench.cummin(Tensor input, int dim) -> (Tensor values, Tensor indices)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入张量 |
| dim | int64 | 必选 | 计算累积最小值的轴 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| values | 与 input 相同 | 与 input 相同 | 累积最小值张量 |
| indices | 与 input 相同 | int64 | 累积最小值对应的索引张量 |

### 数据类型

| 输入 dtype | values dtype | indices dtype |
|-----------|--------------|---------------|
| float16  | float16  | int64 |
| float32  | float32  | int64 |
| int32    | int32    | int64 |
| bfloat16 | bfloat16 | int64 |

### 规则与约束

- `values` 与 `indices` 的 shape 都与 input shape 完全一致
- `dim` 支持负数索引（如 -1 表示最后一维）
- 累积操作沿指定轴按顺序从前到后进行
- `values` 的 dtype 与 input 一致；`indices` 固定为 int64
- 当沿轴存在相等的最小值时，`indices` 保留最先出现的索引位置（与 `torch.cummin` 一致）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(x)`（输入维度数） | 1 ~ 8 | cases.csv 实测 1 ~ 5 维 |
| 每个维度大小 `dim_i` | 1 ~ 1048576 | cases.csv 实测最小 2、最大 1,000,003 |
| 张量总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 256M |
| `dim` | -rank(x) ~ rank(x)-1 | 支持负数索引；cases.csv 实测 -1 / 0 / 1 / 2 |

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


def cummin(input: torch.Tensor, dim: int):
    """Cummin 算子 Torch Golden 参考实现 (P1 op, 对齐 torch.cummin).

    公式: values[i] = min(input[0], input[1], ..., input[i]) 沿指定轴
          indices[i] = argmin(input[0:i+1]) 沿指定轴

    Args:
        input: 输入张量
        dim: 计算累积最小值的轴

    Returns:
        torch.return_types.cummin (named tuple): (values, indices)
    """
    return torch.cummin(input, dim=dim)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
values, indices = cann_bench.cummin(x, dim=-1)   # 沿最后一维计算累积最小值

x = torch.randn(2, 8, 256, 256, dtype=torch.float16, device="npu")
values, indices = cann_bench.cummin(x, dim=2)    # 沿第 2 维计算累积最小值
```
