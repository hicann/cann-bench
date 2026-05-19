# Maximum 算子 API 描述

## 1. 算子简介

返回两个输入张量中的最大值，支持广播。

**主要应用场景**：
- ReLU 激活函数实现（与零取最大值）
- 梯度裁剪中的阈值限制
- 多路径特征融合中的逐元素最大值选择
- 数值计算中的下界约束

**算子特征**：
- 难度等级：L2（Broadcast）
- 双输入单输出，逐元素运算，输入支持广播

## 2. 算子定义

### 数学公式

$$
y = \max(x_1, x_2)
$$

逐元素比较 $x_1$ 和 $x_2$，返回每个位置上的较大值。

## 3. 接口规范

### 算子原型

```python
cann_bench.maximum(Tensor x1, Tensor x2) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 第1个输入张量 |
| x2 | Tensor | 必选 | 第2个输入张量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 广播后的 shape | 与输入一致 | 输出张量，两个输入中的最大值 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| bfloat16 | bfloat16 |
| float16 | float16 |
| float32 | float32 |
| int8 | int8 |
| int32 | int32 |
| int64 | int64 |

### 规则与约束

- 两个输入 shape 按 [PyTorch 标准广播](https://pytorch.org/docs/stable/notes/broadcasting.html) 对齐：右对齐，对位维度需相等或一边为 1（不存在视为 1），输出 shape 取每对维度 max。例如 `[2,3,17,1024,101]` × `[1,1,1,1,101]` → `[2,3,17,1024,101]`
- 两个输入张量的 dtype 必须一致
- 支持浮点类型（bfloat16、float16、float32）和整数类型（int8、int32、int64）
- 当输入包含 NaN 时，行为与 PyTorch `torch.maximum` 一致（NaN 会传播）
- 当输入包含 inf/-inf 时，按正常数值比较规则处理

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x1` rank | 1 ~ 8 | cases.csv 实测 1D ~ 5D |
| `x2` rank | 1 ~ 8 | cases.csv 实测 1D ~ 5D；按 PyTorch 标准广播右对齐，rank 可与 `x1` 不同 |
| `x1` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 2 ~ 1000007 |
| `x2` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 1 ~ 1000007；为 1 表示该维参与广播 |
| 广播后总元素数 | ≤ 2^28（约 256M） | cases.csv 实测最大 8192×16384 = 128M（case 6） |

约束：
- `x1` 与 `x2` 的 dtype 必须严格相等
- `x1` 与 `x2` 的 shape 按 [PyTorch 标准广播](https://pytorch.org/docs/stable/notes/broadcasting.html) 右对齐：对位维度需相等或一边为 1（不存在视为 1）；输出 shape 取每对维度 max
- 输出 tensor 的 shape 与 dtype 由广播规则与输入共同决定

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
Maximum算子Torch Golden参考实现

返回两个输入张量中的最大值，支持广播
公式: y = max(x1, x2)
"""
def maximum(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    返回两个输入张量中的最大值，支持广播
    
    公式: y = max(x1, x2)
    
    Args:
        x1: 第1个输入张量
        x2: 第2个输入张量
    
    Returns:
        输出张量，两个输入中的最大值
    """

    y = torch.maximum(x1, x2)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x1 = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
x2 = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.maximum(x1, x2)

# 广播场景：标量广播
x1 = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
x2 = torch.tensor([0.0], dtype=torch.float32, device="npu")
y = cann_bench.maximum(x1, x2)  # 类似 ReLU

# 整数类型
x1 = torch.randint(-128, 127, (512, 512, 4), dtype=torch.int8, device="npu")
x2 = torch.randint(-10, 10, (1, 512, 1), dtype=torch.int8, device="npu")
y = cann_bench.maximum(x1, x2)
```
