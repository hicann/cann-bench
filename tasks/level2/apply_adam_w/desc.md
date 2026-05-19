# ApplyAdamW 算子 API 描述

## 1. 算子简介

AdamW 优化器实现，解耦权重衰减。

**主要应用场景**：
- 深度学习模型参数优化，尤其是 Transformer 类模型的训练
- 解耦权重衰减正则化，避免 L2 正则化与自适应学习率的耦合问题
- 大规模预训练模型（如 GPT、BERT）的优化器实现

**算子特征**：
- 难度等级：L2（FusedComposite）
- 四输入（var、grad、m、v）单输出（y），逐元素运算，输出 shape 与输入一致
- 支持 1-8 维张量

## 2. 算子定义

### 数学公式

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot grad
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot grad^2
$$

$$
\hat{m} = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v} = \frac{v_t}{1 - \beta_2^t}
$$

$$
var_t = var_{t-1} - lr \cdot \left( \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + weight\_decay \cdot var_{t-1} \right)
$$

其中：
- $m_t$ 为一阶矩估计（动量）
- $v_t$ 为二阶矩估计
- $\hat{m}$、$\hat{v}$ 为偏差修正后的矩估计
- 权重衰减（weight_decay）以解耦方式直接作用于参数

## 3. 接口规范

### 算子原型

```python
cann_bench.apply_adam_w(Tensor var, Tensor grad, Tensor m, Tensor v, float lr, float beta1, float beta2, float weight_decay, float epsilon=1e-8, int step=1, bool maximize=false) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| var | Tensor | 必选 | 变量张量（需要优化的参数） |
| grad | Tensor | 必选 | 梯度张量 |
| m | Tensor | 必选 | 一阶矩张量（动量） |
| v | Tensor | 必选 | 二阶矩张量 |
| lr | float | 必选 | 学习率 |
| beta1 | float | 必选 | 一阶矩估计的指数衰减率 (默认 0.9) |
| beta2 | float | 必选 | 二阶矩估计的指数衰减率 (默认 0.999) |
| weight_decay | float | 必选 | 权重衰减系数（解耦） |
| epsilon | float | 1e-8 | 数值稳定常数 |
| step | int | 1 | 当前优化步数 t，用于偏置校正分母 (1 - beta^t)；默认 1 等价于单步更新 |
| maximize | bool | false | 是否最大化目标函数（默认最小化） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 var 相同 | 与输入 var 相同 | 更新后的变量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- var、grad、m、v 四个张量的 shape 和 dtype 必须完全一致
- 支持 1-8 维张量
- 输出 shape 与输入 var 的 shape 完全一致，输出 dtype 与输入一致
- `beta1`、`beta2` 取值范围通常为 [0, 1)
- `epsilon` 用于防止除零，通常取极小正数
- 当 `maximize=true` 时，更新方向取反（用于最大化目标函数）
- `step` 默认 1（cases.yaml 不传时走 default），公式分母 `1 - beta^step` 在 step=1 时退化为 `1 - beta`，与历史 baseline 兼容；多步训练场景由调用方按需传入

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `var/grad/m/v` 维度数 (ndim) | 1 ~ 8 | cases.csv 实测 1 ~ 5；四个张量 shape/dtype 必须一致 |
| `var/grad/m/v` 单维大小 | 1 ~ 1048576 | cases.csv 实测 2 ~ 1000003 |
| `var/grad/m/v` 总元素数 | 1 ~ 256M | cases.csv 实测 ~1M ~ ~47M  |
| `lr` | 0.0 ~ 1.0 | cases.csv 实测 1e-4 ~ 0.1 |
| `beta1` | 0.0 ~ 1.0 | cases.csv 实测 0.0 ~ 0.99；取值范围 `[0, 1)` |
| `beta2` | 0.0 ~ 1.0 | cases.csv 实测 0.5 ~ 0.999；取值范围 `[0, 1)` |
| `weight_decay` | 0.0 ~ 1.0 | cases.csv 实测 0.0 ~ 0.5 |
| `epsilon` | 0.0 ~ 1.0 | cases.csv 实测 0.0 ~ 1e-4；通常取极小正数（默认 1e-8） |
| `maximize` | `false` / `true` | cases.csv 已覆盖两种取值；`true` 时更新方向取反 |

约束：四个张量 `var/grad/m/v` 的 shape 和 dtype 必须完全一致；输出 `y` 形状/dtype 与 `var` 一致。

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
ApplyAdamW 算子 Torch Golden 参考实现

AdamW 优化器实现，解耦权重衰减
公式:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    var_t = var_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * var_{t-1})
"""
def apply_adam_w(
    var: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    epsilon: float = 1e-8,
    step: int = 1,
    maximize: bool = False,
) -> torch.Tensor:
    """
    AdamW 优化器实现，解耦权重衰减

    Args:
        var/grad/m/v: 四个 tensor 输入，shape/dtype 一致
        lr/beta1/beta2/weight_decay/epsilon: 见上文
        step: 当前优化步数 t，默认 1
        maximize: 是否最大化目标函数

    Returns:
        更新后的变量
    """
    input_dtype = var.dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    var = var.to(compute_dtype)
    grad = grad.to(compute_dtype)
    m = m.to(compute_dtype)
    v = v.to(compute_dtype)

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    # 偏置校正：分母按 step 取指数
    m_hat = m_new / (1 - beta1 ** step)
    v_hat = v_new / (1 - beta2 ** step)
    update = m_hat / (v_hat.sqrt() + epsilon)
    if weight_decay != 0:
        update = update + var * weight_decay
    result = var + lr * update if maximize else var - lr * update

    if input_dtype in (torch.float16, torch.bfloat16):
        return result.to(input_dtype)
    return result
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

var = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
grad = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
m = torch.zeros(1024, 1024, dtype=torch.float32, device="npu")
v = torch.zeros(1024, 1024, dtype=torch.float32, device="npu")

y = cann_bench.apply_adam_w(var, grad, m, v, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)
y = cann_bench.apply_adam_w(var, grad, m, v, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0, epsilon=1e-8, maximize=True)
```
