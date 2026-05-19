# Transpose 算子 API 描述

## 1. 算子简介

对 tensor 的任意维度进行调换。

**主要应用场景**：
- 深度学习中数据格式转换（如 NCHW 与 NHWC 之间的转换）
- 注意力机制中对 Q、K、V 矩阵进行维度交换
- 矩阵运算前的维度调整（如矩阵转置）

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 单输入单输出，支持不超过 8 维的输入，通过 perm 参数指定维度置换顺序

## 2. 算子定义

### 数学公式

$$
y[i_0, ..., i_{n-1}] = x[i_{\text{perm}[0]}, ..., i_{\text{perm}[n-1]}]
$$

其中 perm 为维度置换顺序数组，指定输出张量各维度对应输入张量的哪个维度。

## 3. 接口规范

### 算子原型

```python
cann_bench.transpose(Tensor x, int[] perm) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，维度不超过 8 维 |
| perm | int[] | 必选 | 维度置换顺序 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 输入 shape 按 perm 重排后的 shape | 与输入 x 相同 | 输出张量，转置后的结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |
| int8 | int8 |
| int16 | int16 |
| int32 | int32 |
| int64 | int64 |

### 规则与约束

- 输入维度不超过 8 维
- perm 数组长度必须等于输入维度数，且为 [0, ndim) 的一个排列
- 输出 shape 为输入 shape 按 perm 重排的结果，即 output_shape[i] = input_shape[perm[i]]
- 输出 dtype 与输入 dtype 一致

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x.ndim`（输入维度数） | 2 ~ 8 | cases.csv 实测 2 ~ 5 |
| `x.shape[i]`（每个维度大小） | 1 ~ 16384 | cases.csv 实测 2 ~ 8193（含 1009 / 1021 / 4001 / 1013 等质数非对齐） |
| `x.numel()`（元素总数） | 1 ~ 2^27（约 128M） | cases.csv 实测最大 [64, 32, 512, 128] = 128M (case 1) |
| `perm`（维度置换顺序） | 长度 = `x.ndim` 的 [0, ndim) 整数排列 | cases.csv 实测覆盖 2D 转置 `[1, 0]`、4D `[0, 2, 1, 3]` / `[0, 2, 3, 1]` / `[0, 3, 1, 2]` / `[0, 1, 3, 2]`、3D 循环置换 `[2, 0, 1]`、3D/5D 全反转 `[2, 1, 0]` / `[4, 3, 2, 1, 0]` |

约束：`perm` 必须是 `[0, x.ndim)` 的一个排列（即长度等于 `x.ndim`，且每个值在 `[0, x.ndim)` 区间内且互不重复）；输出 shape 满足 `y.shape[i] = x.shape[perm[i]]`。

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
Transpose算子Torch Golden参考实现

对tensor的任意维度进行调换
公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
"""
def transpose(
    x: torch.Tensor, perm: list
) -> torch.Tensor:
    """
    对tensor的任意维度进行调换
    
    公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
    
    Args:
        x: 输入张量
        perm: 维度置换顺序
    
    Returns:
        输出张量，转置后的结果
    """

    y = torch.permute(x, perm)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 2D 矩阵转置
x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.transpose(x, [1, 0])

# 4D NCHW 转 NHWC
x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y = cann_bench.transpose(x, [0, 2, 3, 1])
```
