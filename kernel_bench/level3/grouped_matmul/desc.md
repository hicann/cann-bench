# GroupedMatmul 算子 API 描述

## 1. 算子简介

分组矩阵乘法算子，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 g 为分组个数，$m_i/k_i/n_i$ 为对应 shape。

**主要应用场景**：
- MoE（Mixture of Experts）模型中多专家的并行矩阵运算
- 多头注意力机制中的分组线性变换
- 批量处理不同大小矩阵的乘法运算

**算子特征**：
- 难度等级：L3（Contraction）
- TensorList 输入输出，每组数据维度可不同
- 支持多 tensor 和单 tensor 输出模式（split_item 控制）

## 2. 算子定义

### 数学公式

$$
y_i = x_i \times weight_i + bias_i, \quad i = 1, ..., g
$$

其中：
- $x_i$ shape 为 $[m_i, k_i]$
- $weight_i$ shape 为 $[k_i, n_i]$（transpose_weight=false）或 $[n_i, k_i]$（transpose_weight=true）
- $bias_i$ shape 为 $[n_i]$
- $y_i$ shape 为 $[m_i, n_i]$

### 支持场景

根据 x、weight、y 的 Tensor 数量支持如下 4 种场景：

| 支持场景 | 描述 | split_item |
|---------|------|------------|
| 多多多 | x、weight、y 都为 TensorList（每组数据独立） | 0/1 |
| 单多单 | x、y 为单 tensor（所有分组在 M 轴合并），weight 为 TensorList | 2/3 |
| 单多多 | x 为单 tensor，weight、y 为 TensorList | 0/1 |
| 多多单 | x、weight 为 TensorList，y 为单 tensor（结果连续存放） | 2/3 |

**说明**：单 tensor 指一个 tensor list 中所有分组的 tensor 在 M 轴上合并为 1 个；否则为多 tensor。

## 3. 接口规范

### 算子原型

```python
cann_bench.grouped_matmul(
    TensorList x, 
    TensorList weight, 
    TensorList? bias, 
    int split_item, 
    bool transpose_weight
) -> TensorList y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | TensorList | 必选 | 输入矩阵 TensorList，每个 tensor shape 为 $[m_i, k_i]$ |
| weight | TensorList | 必选 | 权重矩阵 TensorList，每个 tensor shape 为 $[k_i, n_i]$（transpose_weight=false）或 $[n_i, k_i]$（transpose_weight=true） |
| bias | TensorList | None | 偏置 TensorList（可选），每个 tensor shape 为 $[n_i]$ |
| split_item | int | 0 | 输出切分模式，0/1=输出多 tensor，2/3=输出单 tensor（结果连续存放） |
| transpose_weight | bool | false | 是否转置权重 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | TensorList（split_item=0/1）或单 tensor（split_item=2/3） | 与 x 相同 | 每组输出 shape 为 $[m_i, n_i]$ |

### 数据类型

| 输入 (x) dtype | 输入 (weight) dtype | 输入 (bias) dtype | 输出 (y) dtype |
|---------------|-------------------|-----------------|---------------|
| float16 | float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 / float32 | bfloat16 |
| float32 | float32 | float32 | float32 |

### 规则与约束

- **TensorList 长度**：x、weight、bias（如有）的 TensorList 长度必须一致，最大支持 128 个
- **split_item 使用**：
  - split_item=0/1：输出为多 tensor（TensorList），每组独立
  - split_item=2/3：输出为单 tensor，所有组的结果沿 M 轴合并
- **多多单场景约束**：当 split_item=2/3 且 y 为单 tensor 时，weight 中每个 tensor 的 N 轴必须相等
- **transpose_weight**：
  - false：weight shape 为 $[k_i, n_i]$，直接参与 matmul
  - true：weight shape 为 $[n_i, k_i]$，需转置后参与 matmul
- **维度限制**：每维大小在 32 字节对齐后应小于 int32 最大值

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
from typing import List, Optional

"""
GroupedMatmul 算子 Torch Golden 参考实现

分组矩阵乘法算子，每组矩阵乘的维度大小可以不同
公式: y_i = x_i @ weight_i + bias_i (for each group i)
"""
def grouped_matmul(
    x: List[torch.Tensor],
    weight: List[torch.Tensor],
    bias: Optional[List[torch.Tensor]] = None,
    split_item: int = 0,
    transpose_weight: bool = False
) -> List[torch.Tensor]:
    """
    分组矩阵乘法算子

    对每个分组执行独立的矩阵乘法：y_i = x_i @ weight_i + bias_i

    Args:
        x: 输入矩阵TensorList，每个tensor shape为[m_i, k_i]
        weight: 权重矩阵TensorList，每个tensor shape为[k_i, n_i]（transpose_weight=false）
               或[n_i, k_i]（transpose_weight=true）
        bias: 偏置TensorList（可选），每个tensor shape为[n_i]
        split_item: 输出切分模式
                   - 0/1: 输出多tensor（每组独立），返回TensorList
                   - 2/3: 输出单tensor（结果连续存放），返回合并后的单tensor
        transpose_weight: 是否转置权重
                         - false: weight shape为[k_i, n_i]，matmul为x[m,k] @ weight[k,n]
                         - true: weight shape为[n_i, k_i]，matmul为x[m,k] @ weight[n,k]^T

    Returns:
        输出TensorList（split_item=0/1）或单tensor（split_item=2/3）
        每组输出shape为[m_i, n_i]
    """
    num_groups = len(weight)
    results = []

    for i in range(num_groups):
        x_i = x[i].float()  # [m_i, k_i]
        weight_i = weight[i].float()

        if transpose_weight:
            # weight shape: [n_i, k_i]
            # 需要转置: [n_i, k_i]^T = [k_i, n_i]
            # matmul: [m_i, k_i] @ [k_i, n_i] = [m_i, n_i]
            y_i = torch.matmul(x_i, weight_i.transpose(-2, -1))
        else:
            # weight shape: [k_i, n_i]
            # matmul: [m_i, k_i] @ [k_i, n_i] = [m_i, n_i]
            y_i = torch.matmul(x_i, weight_i)

        # 加偏置（可选）
        if bias is not None and bias[i] is not None:
            bias_i = bias[i].float()  # [n_i]
            y_i = y_i + bias_i.unsqueeze(0)  # broadcast to [m_i, n_i]

        # 转换回输入dtype
        y_i = y_i.to(x[i].dtype)
        results.append(y_i)

    # 根据 split_item 决定输出格式
    if split_item in [0, 1]:
        # 输出多tensor（TensorList）
        return results
    else:
        # split_item in [2, 3]: 输出单tensor（连续存放）
        # 将所有结果沿 M 轴合并
        return [torch.cat(results, dim=0)]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 多多多场景：x、weight、y 都为 TensorList
x = [torch.randn(1024, 512, dtype=torch.float16, device="npu"),
     torch.randn(2048, 512, dtype=torch.float16, device="npu")]
# transpose_weight=False：weight shape 为 [k_i, n_i]，直接参与 matmul
weight = [torch.randn(512, 1024, dtype=torch.float16, device="npu"),
          torch.randn(512, 512, dtype=torch.float16, device="npu")]
bias = [torch.randn(1024, dtype=torch.float16, device="npu"),
        torch.randn(512, dtype=torch.float16, device="npu")]

y = cann_bench.grouped_matmul(x, weight, bias, split_item=0, transpose_weight=False)
# y 是 TensorList：[tensor[1024, 1024], tensor[2048, 512]]

# transpose_weight=True 示例：weight shape 为 [n_i, k_i]，需转置后参与 matmul
weight_t = [torch.randn(1024, 512, dtype=torch.float16, device="npu"),
            torch.randn(512, 512, dtype=torch.float16, device="npu")]
y = cann_bench.grouped_matmul(x, weight_t, bias, split_item=0, transpose_weight=True)

# 多多单场景：输出合并为单 tensor
y_merged = cann_bench.grouped_matmul(x, weight, bias, split_item=2, transpose_weight=False)
# y_merged 是单 tensor：[tensor[3072, N]]（N 必须各组相等）
```
