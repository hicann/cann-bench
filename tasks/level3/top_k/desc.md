# TopK 算子 API 描述

## 1. 算子简介

返回 k 个最大或最小的元素及其索引。

**主要应用场景**：
- 推荐系统中选取得分最高的 k 个候选项
- 分类任务中获取 Top-K 预测类别及其置信度
- 搜索与排序场景中的部分排序加速
- MoE（Mixture of Experts）路由中选取 Top-K 专家

**算子特征**：
- 难度等级：L3（SortSelect）
- 单输入双输出（值和索引），支持 1-8 维输入，支持沿指定维度选取最大或最小的 k 个元素

## 2. 算子定义

### 数学公式

$$
y, idx = \text{topk}(x, k, dim)
$$

沿指定维度 dim 对输入张量 x 进行部分排序，返回前 k 个最大值（当 largest=true）或前 k 个最小值（当 largest=false）及其对应的索引。

## 3. 接口规范

### 算子原型

```python
cann_bench.top_k(Tensor x, int k, int dim, bool largest) -> (Tensor y, Tensor idx)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 1-8 维 |
| k | int | 必选 | 返回的 topk 数量（取值范围：1 <= k <= dim_size） |
| dim | int | 必选 | 排序维度（取值范围：-ndim ~ ndim-1） |
| largest | bool | true | 是否返回最大值（false 时返回最小值） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入相同，但 dim 维大小变为 k | 与输入 x 相同 | 输出张量，topk 值 |
| idx | 与 y 相同 | int64 | 输出索引张量（始终为 int64） |

### 数据类型

| 输入 dtype | 输出 dtype（y） | 输出 dtype（idx） |
|-----------|---------------|-----------------|
| int8 | int8 | int64 |
| uint8 | uint8 | int64 |
| int32 | int32 | int64 |
| int64 | int64 | int64 |
| float16 | float16 | int64 |
| float32 | float32 | int64 |
| bfloat16 | bfloat16 | int64 |

### 规则与约束

- 输入支持 1-8 维张量
- k 的取值范围为 1 <= k <= 指定维度的大小
- dim 支持负数索引，取值范围为 -ndim ~ ndim-1
- 当 largest=true 时返回最大的 k 个元素，largest=false 时返回最小的 k 个元素
- 输出 shape 与输入相同，仅 dim 维度大小变为 k

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（输入维度数） | 1 ~ 8 | cases.csv 实测 1D / 2D / 3D / 4D / 5D |
| `x.shape[i]`（各维大小） | 1 ~ 2^23 | cases.csv 实测 2 ~ 1048576（1D 最大 1048576；2D 最大单维 8193（case 18: [255, 8193]）；高维如 5D 最大 1013） |
| `numel(x)`（总元素数） | 1 ~ 2^26 | cases.csv 实测 ~917K ~ 64M |
| `k` | 1 ~ 2048 | cases.csv 实测 7 ~ 2000；约束 1 ≤ k ≤ x.shape[dim] |
| `dim` | -ndim ~ ndim-1 | cases.csv 实测 0 / 1 / 2 / -1；支持负数索引 |
| `largest` | {true, false} | cases.csv 实测 true / false 均覆盖 |
| `x.dtype` | int8 / uint8 / int32 / int64 / float16 / float32 / bfloat16 | cases.csv 实测 int32 / int64 / float16 / float32 / bfloat16（int8 / uint8 已声明支持但未在 cases 覆盖） |

约束：`k` 必须满足 `1 ≤ k ≤ x.shape[dim]`；`dim` 必须在 `[-ndim, ndim-1]` 范围内；输出 `y` shape 与 `x` 相同但 `y.shape[dim] = k`，`idx` 与 `y` shape 一致且 dtype 固定为 int64。

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
TopK算子Torch Golden参考实现

返回k个最大或最小的元素及其索引
公式: y, idx = topk(x, k, dim)
"""
def top_k(
    x: torch.Tensor, k: int, dim: int, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回k个最大或最小的元素及其索引
    
    公式: y, idx = topk(x, k, dim)
    
    Args:
        x: 输入张量
        k: 返回的topk数量 (取值范围: 1 <= k <= dim_size)
        dim: 排序维度 (取值范围: -ndim ~ ndim-1)
        largest: 是否返回最大值 (false时返回最小值)
    
    Returns:
        y, idx
    """

    values, indices = torch.topk(x, k=k, dim=dim, largest=largest)
    return values, indices
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y, idx = cann_bench.top_k(x, 10, -1, True)  # 每行取最大的10个元素

x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y, idx = cann_bench.top_k(x, 10, -1, False)  # 每行取最小的10个元素
```
