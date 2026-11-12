# Conv2D 算子 API 描述

## 1. 算子简介

计算2D卷积。

**主要应用场景**：
- 图像分类、目标检测、语义分割等视觉任务的核心运算
- CNN 网络中特征提取的基础模块
- 信号处理中的2D滤波操作

**算子特征**：
- 难度等级：L3（Contraction）
- 三输入（特征图、卷积核、偏置）单输出，支持膨胀卷积
- 输入 x 为 [N, C_in, H, W]，卷积核 filter 为 [C_out, C_in, K_h, K_w]

## 2. 算子定义

### 数学公式

$$
y = \text{CONV}(x, \text{filter}) + \text{bias}
$$

即对输入特征图 $x$ 使用卷积核 $\text{filter}$ 进行2D卷积运算，并加上偏置 $\text{bias}$。卷积运算支持通过 strides 控制步长、pads 控制填充、dilations 控制膨胀率。

## 3. 接口规范

### 算子原型

```python
cann_bench.conv_2d(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [N, C_in, H, W] |
| filter | Tensor | 必选 | 卷积核，shape 为 [C_out, C_in, K_h, K_w] |
| bias | Tensor | 必选 | 偏置 |
| strides | int[] | 必选 | 步长 |
| pads | int[] | 必选 | 填充 |
| dilations | int[] | [1, 1] | 膨胀率 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由输入尺寸、卷积核、步长、填充和膨胀率决定 | 与输入 x 相同 | 输出特征图 |

### 数据类型

| 输入 (x, filter, bias) dtype | 输出 dtype |
|-----------------------------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 格式为 [N, C_in, H, W]
- filter 的 shape 格式为 [C_out, C_in, K_h, K_w]
- x、filter、bias 的 dtype 须一致
- strides 指定卷积的步长
- pads 指定四方向填充 [pad_top, pad_bottom, pad_left, pad_right]
- dilations 指定膨胀率，默认 [1, 1]
- 不支持分组卷积；如需分组/depthwise，使用 `depthwise_conv_2d`
- 数值路径自由：实现可采用任意内部计算路径与精度（包括 HF32 以及多趟拆分/补偿算法等），只要输出满足本文档「精度要求」即为合法实现；精度门是唯一判据，不限定与 golden 相同的计算路径。

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 256 | cases.csv 实测 2 ~ 8 |
| `C_in`（输入通道） | 1 ~ 2048 | cases.csv 实测 7 ~ 2048 |
| `C_out`（输出通道） | 1 ~ 2048 | cases.csv 实测 7 ~ 2048 |
| `H`, `W`（空间） | 8 ~ 256 | cases.csv 实测 13 ~ 128 |
| `K_h`, `K_w`（卷积核） | 1 ~ 16 | cases.csv 实测 1 / 3 / 5 |
| `strides[i]` | 1 ~ 4 | cases.csv 实测 (1,1) 和 (2,2) |
| `pads[i]` | 0 ~ 8 | cases.csv 实测 0 ~ 2 |
| `dilations[i]` | 1 ~ 16 | cases.csv 实测 1 / 2 / 3 |

约束：输出 spatial 维度 `(H_out, W_out)` 必须满足 `H_out = (H + pad_top + pad_bottom - dilation_h*(K_h-1) - 1) / stride_h + 1 ≥ 1`，`W_out` 同理。

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
Conv2D算子Torch Golden参考实现

计算2D卷积
公式: y = CONV(x, filter) + bias
"""


def conv_2d(
    x: torch.Tensor,
    filter: torch.Tensor,
    bias: torch.Tensor,
    strides: list,
    pads: list,
    dilations: list = [1, 1],
) -> torch.Tensor:
    """
    计算2D卷积

    公式: y = CONV(x, filter) + bias

    Args:
        x: 输入特征图
        filter: 卷积核
        bias: 偏置
        strides: 步长
        pads: 填充
        dilations: 膨胀率

    Returns:
        输出特征图
    """

    # pads 格式: [pad_top, pad_bottom, pad_left, pad_right]
    # PyTorch conv2d padding 格式: (left, right, top, bottom) 或 (pad_h, pad_w) 对称模式
    # 检查是否对称填充
    if pads[0] == pads[1] and pads[2] == pads[3]:
        # 对称模式: (pad_height, pad_width)
        padding = (pads[0], pads[2])
    else:
        # 非对称模式: (left, right, top, bottom)
        x = torch.nn.functional.pad(x, (pads[2], pads[3], pads[0], pads[1]))
        padding = 0
    stride = (strides[0], strides[1])
    dilation = (dilations[0], dilations[1])

    y = torch.nn.functional.conv2d(
        x, filter, bias, stride=stride, padding=padding, dilation=dilation
    )
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1, 64, 56, 56, dtype=torch.float16, device="npu")
weight = torch.randn(128, 64, 3, 3, dtype=torch.float16, device="npu")
bias = torch.randn(128, dtype=torch.float16, device="npu")

y = cann_bench.conv_2d(x, weight, bias, strides=[1, 1], pads=[1, 1, 1, 1], dilations=[1, 1])
```
