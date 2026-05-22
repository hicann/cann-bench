# AdaptiveAvgPool3D 算子 API 描述

## 1. 算子简介

完成输入张量的3D自适应平均池化计算。

**主要应用场景**：
- 3D 视频特征的空间和时间维度自适应降采样
- 点云和体素数据的空间压缩
- 全局平均池化（output_size=1）用于分类网络的特征聚合
- 不同分辨率输入统一到固定尺寸输出

**算子特征**：
- 难度等级：L3（Reduction）
- 单输入单输出，输入为 [N, C, D, H, W] 5维张量，输出空间维度由 output_size 决定

## 2. 算子定义

### 数学公式

$$
y = \text{adaptive\_avg\_pool3d}(x, \text{output\_size})
$$

自适应平均池化根据目标输出尺寸自动计算每个输出位置对应的池化窗口大小和步长，对窗口内元素取平均值。对于每个输出位置 $(d, h, w)$，其对应的输入区域由 output_size 和输入尺寸共同决定。

## 3. 接口规范

### 算子原型

```python
cann_bench.adaptive_avg_pool_3d(Tensor x, list[int] output_size) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，shape 为 [N, C, D, H, W] 的5维张量 |
| output_size | list[int] | 必选 | 输出尺寸，格式为 [output_d, output_h, output_w] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [N, C, output_size_d, output_size_h, output_size_w] | 与输入 x 相同 | 输出张量，池化结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入必须为5维张量，shape 格式为 [N, C, D, H, W]
- output_size 指定输出的空间维度大小
- 输出 dtype 与输入 dtype 一致
- 输出的 N 和 C 维度与输入保持一致，仅空间维度 (D, H, W) 发生变化
- 不支持 PyTorch `adaptive_avg_pool3d` 的两个扩展形态：
  - 4D 无 batch 输入 `[C, D, H, W]`（本算子固定 5D）
  - `output_size` 中包含 `None` 占位（表示保留该维度）；本算子要求三个维度均为正整数

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 16 | cases.csv 实测 1 ~ 8 |
| `C`（通道） | 1 ~ 512 | cases.csv 实测 7 ~ 257 |
| `D`（输入深度） | 1 ~ 256 | cases.csv 实测 13 ~ 128 |
| `H`（输入高） | 1 ~ 256 | cases.csv 实测 15 ~ 128 |
| `W`（输入宽） | 1 ~ 256 | cases.csv 实测 15 ~ 128 |
| `output_size[0]`（output_d） | 1 ~ 128 | cases.csv 实测 1 ~ 64 |
| `output_size[1]`（output_h） | 1 ~ 128 | cases.csv 实测 1 ~ 64 |
| `output_size[2]`（output_w） | 1 ~ 128 | cases.csv 实测 1 ~ 64 |

约束：`output_size` 须为长度 3 的 list，元素均 ≥ 1；支持 downsample（output_size < 输入空间维）与 upsample（output_size > 输入空间维）；output_size=[1,1,1] 等价于全局平均池化。

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
AdaptiveAvgPool3D算子Torch Golden参考实现

完成输入张量的3D自适应平均池化计算
公式: y = adaptive_avg_pool3d(x, output_size)
"""
def adaptive_avg_pool_3d(
    x: torch.Tensor, output_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    完成输入张量的3D自适应平均池化计算

    公式: y = adaptive_avg_pool3d(x, output_size)

    Args:
        x: 输入张量，shape 为 [N, C, D, H, W]
        output_size: 输出尺寸，格式为 (output_d, output_h, output_w)

    Returns:
        输出张量，池化结果
    """

    y = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 32, 16, 64, 64, dtype=torch.float16, device="npu")
y = cann_bench.adaptive_avg_pool_3d(x, [8, 8, 8])  # 自适应池化到 8x8x8

x = torch.randn(2, 64, 32, 128, 128, dtype=torch.float32, device="npu")
y = cann_bench.adaptive_avg_pool_3d(x, [1, 1, 1])  # 全局平均池化
```
