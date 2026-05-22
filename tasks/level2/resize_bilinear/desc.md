# ResizeBilinear 算子 API 描述

## 1. 算子简介

使用双线性插值调整图像大小。

**主要应用场景**：
- 图像预处理中的缩放与分辨率调整
- 特征金字塔网络（FPN）中的上采样操作
- 语义分割中的特征图恢复到原始分辨率
- 目标检测中不同尺度特征的对齐

**算子特征**：
- 难度等级：L2（FusedComposite）
- 单输入单输出，输入为 4D 张量 (N, C, H, W)，输出空间维度由 output_size 或 scale_factor 指定

## 2. 算子定义

### 数学公式

$$
y = \text{resize\_bilinear}(x, \text{size})
$$

对输入张量的空间维度 (H, W) 进行双线性插值缩放。对于输出位置 $(i, j)$，根据其在输入空间中的映射坐标 $(h, w)$，利用周围 4 个最近邻像素的值进行加权平均：

$$
y(i, j) = (1-\alpha)(1-\beta) \cdot x(\lfloor h \rfloor, \lfloor w \rfloor) + \alpha(1-\beta) \cdot x(\lceil h \rceil, \lfloor w \rfloor) + (1-\alpha)\beta \cdot x(\lfloor h \rfloor, \lceil w \rceil) + \alpha\beta \cdot x(\lceil h \rceil, \lceil w \rceil)
$$

其中 $\alpha = h - \lfloor h \rfloor$，$\beta = w - \lfloor w \rfloor$。

- **align_corners=true**：输入输出的角点像素对齐，坐标映射为 $h = i \times \frac{H_{in}-1}{H_{out}-1}$
- **align_corners=false**：坐标映射为 $h = (i + 0.5) \times \frac{H_{in}}{H_{out}} - 0.5$

## 3. 接口规范

### 算子原型

```python
cann_bench.resize_bilinear(Tensor x, int[] output_size, bool align_corners=false, float[] scale_factor=null) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，形状为 (N, C, H, W) |
| output_size | int[] | 必选 | 输出尺寸 [output_height, output_width] |
| align_corners | bool | false | 是否对齐角点 |
| scale_factor | float[] | null | 缩放因子 [scale_height, scale_width]，与 output_size 互斥 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (N, C, H_out, W_out) | 与输入 x 相同 | 输出张量，调整大小后的结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入 x 必须为 4D 张量，形状为 (N, C, H, W)
- output_size 和 scale_factor 互斥，两者不能同时指定
- output_size 为 [output_height, output_width]，指定输出空间维度大小
- scale_factor 为 [scale_height, scale_width]，指定缩放比例
- 输出 dtype 与输入 dtype 一致
- 支持上采样（输出大于输入）和下采样（输出小于输入）

### 支持范围

输入 tensor 各维度与参数的支持范围（仅 4D 输入 `(N, C, H, W)`）：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 256 | cases.csv 实测 1 ~ 127 |
| `C`（通道） | 1 ~ 1024 | cases.csv 实测 1 ~ 363 |
| `H`（输入高） | 1 ~ 8192 | cases.csv 实测 13 ~ 4097 |
| `W`（输入宽） | 1 ~ 8192 | cases.csv 实测 67 ~ 4001 |
| `output_size[0]`（输出高） | 1 ~ 4096 | cases.csv 实测 64 ~ 2048 |
| `output_size[1]`（输出宽） | 1 ~ 4096 | cases.csv 实测 64 ~ 1024 |
| `align_corners` | {false, true} | cases.csv 实测两种取值都覆盖 |
| `scale_factor` | null 或 长度=2 的正浮点数组 | cases.csv 实测均为 null（与 `output_size` 互斥，二者只能取一） |

约束：
- 输入张量 rank 严格为 4，shape 为 `(N, C, H, W)`
- `output_size` 与 `scale_factor` 互斥，必须恰好提供其一；两者长度均固定为 2（高、宽）
- 输出 shape 的非空间维（N, C）与输入一致，空间维由 `output_size` 或 `[H * scale_factor[0], W * scale_factor[1]]` 计算得到
- 不在本算子范围：1D/3D/5D 输入（linear/trilinear 模式）—— 由 PyTorch `interpolate` 的其它分支负责，不在本基准评测范围内

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
ResizeBilinear 算子 Torch Golden 参考实现

使用双线性插值调整 4D 图像 (N, C, H, W) 的空间维度大小
公式: y = resize_bilinear(x, size)

参考 PyTorch API: torch.nn.functional.interpolate (mode='bilinear')
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
"""


def resize_bilinear(
    x: torch.Tensor,
    output_size: Optional[List[int]] = None,
    align_corners: bool = False,
    scale_factor: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    使用双线性插值调整图像大小（仅 4D 输入）。

    Args:
        x: 输入张量，形状为 (N, C, H, W)
        output_size: 输出尺寸 [output_height, output_width]，与 scale_factor 互斥
        align_corners: 是否对齐角点
        scale_factor: 缩放因子 [scale_height, scale_width]，与 output_size 互斥

    Returns:
        输出张量 (N, C, H_out, W_out)，dtype 与 x 一致
    """
    if x.dim() != 4:
        raise ValueError(f"ResizeBilinear requires 4D input (N, C, H, W), got {x.dim()}D")
    return torch.nn.functional.interpolate(
        x,
        size=output_size,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=align_corners,
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 8, 512, 512, dtype=torch.float16, device="npu")
y = cann_bench.resize_bilinear(x, output_size=[256, 256], align_corners=False)  # 下采样

x = torch.randn(4, 4, 64, 64, dtype=torch.float32, device="npu")
y = cann_bench.resize_bilinear(x, output_size=[128, 128], align_corners=True)  # 上采样 + 角点对齐

x = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16, device="npu")
y = cann_bench.resize_bilinear(x, output_size=[256, 256])  # bfloat16 上采样
```
