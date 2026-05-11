# WeightQuantBatchMatmul 算子 API 描述

## 1. 算子简介

权重量化批量矩阵乘法算子，支持权重的反量化计算。

**主要应用场景**：
- 大语言模型推理中的权重量化加速
- 低精度（INT8/INT4）量化模型的矩阵乘法计算
- 模型压缩与部署场景中的量化矩阵运算

**算子特征**：
- 难度等级：L3（Contraction）
- 多输入（x、weight、antiquantScale、可选 antiquantOffset、可选 bias）单输出
- weight 为量化权重（INT8/INT4），通过反量化参数转换为浮点后参与矩阵乘法

## 2. 算子定义

### 数学公式

$$
y = x \times \text{ANTIQUANT}(weight) + bias
$$

其中反量化公式：

$$
\text{ANTIQUANT}(weight) = (weight + \text{antiquantOffset}) \times \text{antiquantScale}
$$

**计算步骤**：
1. 对量化权重矩阵进行反量化（ANTIQUANT）操作
2. 执行矩阵乘法 $x \times \text{ANTIQUANT}(weight)$
3. 加上偏置 bias（可选）

## 3. 接口规范

### 算子原型

```python
cann_bench.weight_quant_batch_matmul(Tensor x, Tensor weight, Tensor antiquantScale, Tensor? antiquantOffset=None, Tensor? bias=None) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 左输入矩阵，shape 为 [M, K]，dtype 为 float16/bfloat16 |
| weight | Tensor | 必选 | 右输入矩阵（量化权重），shape 为 [K, N]，dtype 为 int8/int4 |
| antiquantScale | Tensor | 必选 | 反量化scale参数，shape 为 [N] 或 [1, N]，dtype 与 x 相同 |
| antiquantOffset | Tensor | None | 反量化offset参数（可选），shape 与 antiquantScale 相同 |
| bias | Tensor | None | 偏置张量（可选），shape 为 [N] 或 [1, N] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [M, N] | 与 x 相同 | 输出张量 |

### 数据类型

| x dtype | weight dtype | antiquantScale dtype | 输出 dtype |
|---------|-------------|---------------------|-----------|
| float16 | int8/int4 | float16 | float16 |
| bfloat16 | int8/int4 | bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 为 [M, K]，weight 的 shape 为 [K, N]，矩阵乘法要求 K 维度相等
- antiquantScale 的 shape 为 [N] 或 [1, N]，对应 weight 的 N 维度
- antiquantOffset（可选）shape 与 antiquantScale 相同
- bias（可选）shape 为 [N] 或 [1, N]
- antiquantScale/antiquantOffset/bias 的 dtype 与 x 相同

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
WeightQuantBatchMatmul 算子 Torch Golden 参考实现

权重量化批量矩阵乘法算子
公式: y = x @ ANTIQUANT(weight) + bias
      ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
"""
def weight_quant_batch_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquantScale: torch.Tensor,
    antiquantOffset: torch.Tensor = None,
    bias: torch.Tensor = None
) -> torch.Tensor:
    """
    权重量化批量矩阵乘法算子

    公式: y = x @ ANTIQUANT(weight) + bias
          ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale

    Args:
        x: 左输入矩阵，shape 为 [M, K]，dtype 为 float16/bfloat16
        weight: 右输入矩阵（量化权重），shape 为 [K, N]，dtype 为 int8/int4
        antiquantScale: 反量化scale参数，shape 为 [N] 或 [1, N]
        antiquantOffset: 反量化offset参数（可选），shape 与 antiquantScale 相同
        bias: 偏置张量（可选），shape 为 [N] 或 [1, N]

    Returns:
        输出张量，shape 为 [M, N]，dtype 与 x 相同
    """

    # 反量化 weight: (weight + antiquantOffset) * antiquantScale
    weight_float = weight.float()  # [K, N]
    scale_float = antiquantScale.float()  # [N] 或 [1, N]

    if antiquantOffset is not None:
        offset_float = antiquantOffset.float()
        weight_dequant = (weight_float + offset_float) * scale_float
    else:
        weight_dequant = weight_float * scale_float

    # 矩阵乘法: [M, K] @ [K, N] = [M, N]
    x_float = x.float()
    y_float = torch.matmul(x_float, weight_dequant)

    # 加偏置（可选）
    if bias is not None:
        bias_float = bias.float()
        y_float = y_float + bias_float

    y = y_float.to(x.dtype)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# x: [M, K] = [16, 32]
x = torch.randn(16, 32, dtype=torch.float16, device="npu")
# weight: [K, N] = [32, 64]
weight = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device="npu")
# antiquantScale: [N] = [64]
antiquantScale = torch.randn(64, dtype=torch.float16, device="npu")
# antiquantOffset: [N] = [64] (可选)
antiquantOffset = torch.randn(64, dtype=torch.float16, device="npu")
# bias: [N] = [64] (可选)
bias = torch.randn(64, dtype=torch.float16, device="npu")

y = cann_bench.weight_quant_batch_matmul(x, weight, antiquantScale, antiquantOffset, bias)
# y shape: [M, N] = [16, 64]
```
