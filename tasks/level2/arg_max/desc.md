# ArgMax 算子 API 描述

## 1. 算子简介

返回张量在指定维度上的最大值的索引。

**主要应用场景**：
- 分类任务中获取预测类别（取 logits 最大值对应的类别索引）
- Top-1 准确率计算
- 贪心解码（Greedy Decoding）中选择概率最大的 token

**算子特征**：
- 难度等级：L2（SortSelect）
- P1 op：直接对齐 `torch.argmax` 官方接口
- 单输入单输出，沿指定维度进行归约操作；输入支持 0-8 维

## 2. 算子定义

### 数学公式

$$
\text{indices} = \arg\max_{axis=dim}(input)
$$

即返回输入张量 `input` 在指定维度 `dim` 上最大值所在的索引位置。

## 3. 接口规范

### 算子原型

```python
cann_bench.arg_max(Tensor input, int dim, bool keepdim=False) -> Tensor indices
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入张量 |
| dim | int64 | 必选 | 计算 argmax 的维度 |
| keepdim | bool | False | 是否保留约简维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| indices | `keepdim=True` 时与 input 同 rank,dim 位 size=1；`keepdim=False` 时去掉 dim 位 | int64 | 最大值的索引张量 |

### 数据类型

| 输入 dtype | indices dtype |
|-----------|---------------|
| float16  | int64 |
| float32  | int64 |
| bfloat16 | int64 |
| int32    | int64 |
| int64    | int64 |

### 规则与约束

- 输入支持 0-8 维张量
- `dim` 支持负数索引（如 -1 表示最后一维）
- `indices` dtype 固定为 int64
- 当 `dim` 维上存在多个相同的最大值时，返回第一个（最小索引）出现的位置（与 `torch.argmax` 一致）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(input)`（输入维度数） | 0 ~ 8 | cases.csv 实测 1 ~ 5 维 |
| 每个维度大小 `dim_i` | 1 ~ 2097152 | cases.csv 实测最小 2、最大 1,048,583 |
| 张量总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 67M（8192×8192） |
| `dim` | -rank(input) ~ rank(input)-1 | 支持负数索引；cases.csv 实测 -1 / 0 / 1 / 2 |

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

> **注**:argmax 输出是 int64 indices,不参与浮点 MERE/MARE 校验,采用 bitwise-equal 或 mismatch-rate 判定（见 test_baseline_prec.py 中整数输出分支）。


## 5. 标准 Golden 代码

```python
import torch


def arg_max(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """ArgMax 算子 Torch Golden 参考实现 (P1 op, 对齐 torch.argmax).

    公式: indices = argmax(input, dim=dim)

    Args:
        input: 输入张量
        dim: 计算 argmax 的维度
        keepdim: 是否保留约简维度,默认 False

    Returns:
        indices (int64): 最大值索引张量
    """
    return torch.argmax(input, dim=dim, keepdim=keepdim)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
indices = cann_bench.arg_max(x, dim=-1)            # 沿最后一维取 argmax，输出 shape [1024]

x = torch.randn(2, 8, 256, 256, dtype=torch.float16, device="npu")
indices = cann_bench.arg_max(x, dim=2, keepdim=True)  # 输出 shape [2, 8, 1, 256]
```
