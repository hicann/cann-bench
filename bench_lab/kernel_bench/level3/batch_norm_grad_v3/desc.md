# BatchNormGradV3 算子 API 描述

## 1. 算子简介

BatchNormGradV3 是 BatchNorm 前向算子的反向传播算子，用于计算输入张量、缩放参数和偏置参数的梯度。

**主要应用场景**：
- 深度学习模型训练中的批归一化层反向传播
- 卷积神经网络、Transformer 等模型中的归一化层梯度计算

**算子特征**：
- 难度等级：L3（Composite）
- 多输入多输出：7 个输入张量，3 个输出张量
- 支持 2-8D 输入，channel 轴固定为第 1 维
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

### 数学公式

当 `is_training=true` 时：

```
denominator = sqrt(saveVar + epsilon)
gradInput = weight / (n * denominator) * (n * gradOut - sum(gradOut) - (x - saveMean) / denominator * sum(gradOut * (x - saveMean) / denominator))
gradWeight = sum(gradOut * (x - saveMean)) / denominator
gradBias = sum(gradOut)
```

当 `is_training=false` 时：

```
denominator = sqrt(runningVar + epsilon)
gradInput = gradOut * weight / denominator
gradWeight = sum(gradOut * (x - runningMean)) / denominator
gradBias = sum(gradOut)
```

其中 `n` 为除 channel 轴外所有维度的乘积（即每个 channel 上的样本数）。

### 变量说明

| 变量 | 说明 |
|------|------|
| gradOut | 正向输出的梯度 |
| x | 正向输入 |
| weight | 缩放权重 |
| runningMean / runningVar | 训练期间累积的均值 / 方差（推理场景） |
| saveMean / saveInvstd | 前向保存的均值 / 标准差倒数（训练场景） |
| epsilon | 防止除零的极小值 |
| gradInput | 输入 x 的梯度 |
| gradWeight | 权重 weight 的梯度 |
| gradBias | 偏置的梯度 |

## 3. 接口规范

### 算子原型

```python
cann_bench.batch_norm_grad_v3(
    Tensor grad_out,
    Tensor x,
    Tensor weight,
    Tensor running_mean,
    Tensor running_var,
    Tensor save_mean,
    Tensor save_invstd,
    bool is_training=True,
    float epsilon=1e-5,
) -> (Tensor dx, Tensor dweight, Tensor dbias)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| grad_out | Tensor | 必选 | 正向输出梯度，形状与 x 相同 |
| x | Tensor | 必选 | 正向输入 |
| weight | Tensor | 必选 | 缩放权重，1D，长度等于 channel 数 |
| running_mean | Tensor | 必选 | 训练累积均值，1D，长度等于 channel 数 |
| running_var | Tensor | 必选 | 训练累积方差，1D，长度等于 channel 数 |
| save_mean | Tensor | 必选 | 前向保存均值，1D，长度等于 channel 数 |
| save_invstd | Tensor | 必选 | 前向保存标准差倒数，1D，长度等于 channel 数 |
| is_training | bool | true | 是否训练场景 |
| epsilon | float | 1e-5 | 防止除零的极小值 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| dx | 与 x 相同 | 与 x 相同 | 输入 x 的梯度 |
| dweight | [C] | fp32（x 为 fp16/bf16 时）/ 与 x 相同（x 为 fp32 时） | 权重 weight 的梯度 |
| dbias | [C] | fp32（x 为 fp16/bf16 时）/ 与 x 相同（x 为 fp32 时） | 偏置的梯度 |

### 数据类型

| 输入 dtype | dx dtype | dweight / dbias dtype |
|-----------|----------|----------------------|
| float16 | float16 | float32 |
| float32 | float32 | float32 |
| bfloat16 | bfloat16 | float32 |

> 注：当输入为 float16 或 bfloat16 时，为避免低精度累加溢出，dweight 与 dbias 在 golden 中按 **float32** 计算并返回；dx 仍保持与输入一致的 dtype。NPU 行为与此对齐。

### 规则与约束

- grad_out、x、dx 的 shape、dtype、数据格式必须一致
- weight、running_mean、running_var、save_mean、save_invstd、dweight、dbias 的 shape 长度必须为 1，且等于 channel 轴大小
- 支持 2-8D 输入，channel 轴为第 1 维
- is_training=true 时使用 save_mean/save_invstd；is_training=false 时使用 running_mean/running_var
- 输入为 float16/bfloat16 时，dweight 与 dbias 按 float32 返回（dx 仍与输入同 dtype）

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 256 | cases.csv 实测 1 ~ 127 |
| `C`（channel） | 1 ~ 8192 | cases.csv 实测 3 ~ 1024 |
| `H / W / D`（空间维度） | 1 ~ 8192 | cases.csv 实测 7 ~ 2048 |
| `is_training` | {false, true} | cases.csv 实测两种取值都覆盖 |
| `epsilon` | [1e-7, 1e-3] | cases.csv 实测 1e-7 ~ 1e-3 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：

   ```
   MERE = avg(abs(actual - golden) / (abs(golden) + 1e-7))
   ```

2. 最大相对误差（MARE）：

   ```
   MARE = max(abs(actual - golden) / (abs(golden) + 1e-7))
   ```

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple


def batch_norm_grad_v3(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    save_mean: torch.Tensor,
    save_invstd: torch.Tensor,
    is_training: bool = True,
    epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BatchNormGradV3 算子 Torch Golden 参考实现。

    参考 PyTorch API: torch.ops.aten.native_batch_norm_backward
    """
    # 低精度累加容易溢出，提升到 fp32 计算
    orig_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        grad_out = grad_out.float()
        x = x.float()
        weight = weight.float()
        running_mean = running_mean.float()
        running_var = running_var.float()
        save_mean = save_mean.float()
        save_invstd = save_invstd.float()

    dx, dweight, dbias = torch.ops.aten.native_batch_norm_backward(
        grad_out,
        x,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        is_training,
        epsilon,
        output_mask=[True, True, True],
    )

    # NPU 对 fp16/bf16 输入返回 fp32 的 dweight/dbias，golden 与之对齐
    if orig_dtype in (torch.float16, torch.bfloat16):
        return dx.to(orig_dtype), dweight, dbias
    return dx, dweight, dbias
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

grad_out = torch.randn(2, 8, 32, 32, dtype=torch.float16, device="npu")
x = torch.randn(2, 8, 32, 32, dtype=torch.float16, device="npu")
weight = torch.randn(8, dtype=torch.float16, device="npu")
running_mean = torch.randn(8, dtype=torch.float16, device="npu")
running_var = torch.rand(8, dtype=torch.float16, device="npu")
save_mean = torch.randn(8, dtype=torch.float16, device="npu")
save_invstd = torch.rand(8, dtype=torch.float16, device="npu")

dx, dweight, dbias = cann_bench.batch_norm_grad_v3(
    grad_out, x, weight, running_mean, running_var, save_mean, save_invstd
)
```
