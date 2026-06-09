# MoeGatingTopKBackward 算子 API 描述

## 1. 算子简介

完成MoE（Mixture of Experts）门控Top-K选择的反向梯度计算。该算子是MoeGatingTopK的反向算子，根据前向算子输出的归一化得分（xNorm）、上游梯度（gradY）和专家索引（expertIdx），计算输入得分矩阵的梯度（gradX）。支持sigmoid模式（normType=1）。

**主要应用场景**：
- MoE 模型门控路由的反向梯度计算
- 大规模稀疏专家模型的训练
- Top-K 专家选择策略的梯度回传

**算子特征**：
- 难度等级：L2（MoE）
- 3 输入，1 输出，4 个属性参数
- 支持 ND 格式输入
- 可选属性：renorm, norm_type, routed_scaling_factor, eps

## 2. 算子定义

### 数学公式

**Step 1: 缩放上游梯度**

$$
gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
$$

**Step 2: Gather 前向归一化分数**

$$
wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
$$

**Step 3: 计算归一化分母**

$$
D_i = \sum_{p=1}^{K} wPrime_{ip} + eps
$$

**Step 4: 归一化**

$$
w_{ip} = \frac{wPrime_{ip}}{D_i}
$$

**Step 5: Renorm 反向**

$$
\beta_i = \sum_{p=1}^{K} w_{ip} \cdot gradYScaled_{ip}
$$

$$
gradWPrime_{ip} = \frac{gradYScaled_{ip} - \beta_i}{D_i}
$$

**Step 6: Scatter 回完整维度**

$$
gradNormX_{in} = \sum_{p:\ expertIdx_{ip}=n} gradWPrime_{ip}
$$

**Step 7: Sigmoid 反向**

$$
gradX_{in} = xNorm_{in} \cdot (1 - xNorm_{in}) \cdot gradNormX_{in}
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.moe_gating_top_k_backward(Tensor x_norm, Tensor grad_y, Tensor expert_idx, int64 renorm=0, int64 norm_type=0, float routed_scaling_factor=1.0, float eps=1e-20) -> Tensor grad_x
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| x_norm | 输入 | 计算的输入 | 要求是一个2D的Tensor，维度为[M,N]，专家数（最后一维）要求不大于2048，不支持空Tensor | FLOAT32 | ND | 2 | 支持 |
| grad_y | 输入 | 前向算子输出yOut的梯度 | 要求是一个2D的Tensor，维度为[M,K]，0<K<=N，不支持空Tensor | FLOAT16/BFLOAT16/FLOAT32 | ND | 2 | 不支持 |
| expert_idx | 输入 | 前向算子的输出expertIdxOut，对应top-k专家的索引 | shape要求与grad_y一致，不支持空Tensor | INT32 | ND | 2 | 不支持 |
| renorm | 输入 | renorm标记 | 当前仅支持0，表示先进行norm再进行topk计算 | INT64 | - | - | - |
| norm_type | 输入 | norm函数类型 | 1表示使用Sigmoid函数，0表示Softmax函数，当前仅支持1 | INT64 | - | - | - |
| routed_scaling_factor | 输入 | 计算yOut使用的缩放系数 | 默认值为1.0 | DOUBLE | - | - | - |
| eps | 输入 | 前向计算使用的eps系数 | 默认值为1e-20 | DOUBLE | - | - | - |


### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| grad_x | 输出 | 前向算子输入参数x的梯度 | 数据类型与grad_y需要保持一致，shape与x_norm需要一致 | FLOAT16/BFLOAT16/FLOAT32 | ND | 2 | 不支持 |


### 数据类型

| x_norm dtype | grad_y dtype | expert_idx dtype | grad_x dtype |
|-------------|-------------|-----------------|-------------|
| float32 | float32 | int32 | float32 |
| float32 | bfloat16 | int32 | bfloat16 |
| float32 | float16 | int32 | float16 |

### 规则与约束

- x_norm 仅支持 float32 类型，grad_x 输出类型与 grad_y 一致
- 所有 Tensor 输入均为 2D，格式为 ND
- x_norm 的最后一维（专家数 N）不大于 2048
- expert_idx 的 shape 必须与 grad_y 一致，均为 [M, K]
- 当前仅支持 renorm=0, norm_type=1（sigmoid 模式）
- expert_idx 中的索引值必须在 [0, N-1] 范围内

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| M（token数，第0维） | 1 ~ 512000 | x_norm/grad_y/expert_idx 的第0维 |
| N（专家数，x_norm第1维） | 2 ~ 2048 | kernel 限制 N ≤ 2048 |
| K（top-k数，grad_y第1维） | 1 ~ N | 0 < K ≤ N |
| x_norm dtype | float32 | 仅支持 float32 |
| grad_y dtype | float16, bfloat16, float32 | 决定输出 dtype |
| expert_idx dtype | int32 | 仅支持 int32 |
| renorm | 0 | 当前仅支持 0 |
| norm_type | 1 | 当前仅支持 1（sigmoid） |
| routed_scaling_factor | 任意正浮点数 | 默认 1.0 |
| eps | 任意正浮点数 | 默认 1e-20 |

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
from typing import Optional
import sys
def moe_gating_top_k_backward(x_norm: torch.Tensor, grad_y: torch.Tensor, expert_idx: torch.Tensor,
                               renorm: int = 0, norm_type: int = 1,
                               routed_scaling_factor: float = 1.0, eps: float = 1e-20
):
    """
    MoE Gating Top-K 反向传播（普通函数实现），当前仅sigmoid模式

    参数:
        x_norm: 前向归一化得分 [M, N], float32 (sigmoid输出)
        grad_y: 上游梯度 [M, K], 数据类型支持float16、bfloat16、float32
        expert_idx: 前向选中的专家索引 [M, K], int32
        renorm: 未使用，保留参数
        norm_type: 归一化方式 (1-sigmoid, 否则-softmax)，当前仅支持1
        routed_scaling_factor: 最终权重的缩放因子, float32
        eps: 防止除零的小常数, float32

    返回:
        grad_x: [M, N] 输入得分矩阵的梯度, 数据类型与grad_y一致
    """
    # 转换为float32进行计算
    grad_y_fp32 = grad_y.float() if grad_y.dtype != torch.float32 else grad_y.clone()
    x_norm_fp32 = x_norm.float() if x_norm.dtype != torch.float32 else x_norm.clone()
    grad_y_scaled = grad_y_fp32 * routed_scaling_factor  # [M, K]

    # 步骤2: Gather前向归一化分数 (w')
    w_prime = x_norm_fp32.gather(1, expert_idx.long())  # [M, K]
    if norm_type == 1:
        D = w_prime.sum(dim=-1, keepdim=True) + eps  # [M, 1]

        inv_D = 1.0 / D  # [M, 1]

        w = w_prime * inv_D  # [M, K]
        beta = (w * grad_y_scaled).sum(dim=-1, keepdim=True)  # [M, 1]
        grad_w_prime = (grad_y_scaled - beta) * inv_D  # [M, K]
    else:
        # Softmax模式（预留）
        grad_w_prime = grad_y_scaled

    # 步骤3: Scatter回完整维度 [M, N]
    grad_norm_x = torch.zeros_like(x_norm_fp32)
    grad_norm_x.scatter_(1, expert_idx.long(), grad_w_prime)


    # 步骤4: Sigmoid反向: grad_x = x_norm * (1 - x_norm) * grad_norm_x
    if norm_type == 1:
        grad_x = x_norm_fp32 * (1.0 - x_norm_fp32) * grad_norm_x
    else:
        # Softmax反向（预留）
        grad_x = grad_norm_x

    # 转回原始数据类型
    return grad_x.to(grad_y.dtype)
    
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x_norm = torch.randn(2048, 192, dtype=torch.bfloat16, device="npu")
grad_y = torch.randn(192, dtype=torch.bfloat16, device="npu")
grad_x = cann_bench.moe_gating_top_k_backward(x_norm, grad_y, expert_idx, renorm=0, norm_type=1, routed_scaling_factor=1.0, eps=1e-20)
```
