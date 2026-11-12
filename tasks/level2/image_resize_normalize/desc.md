# ImageResizeNormalize 算子 API 描述

## 1. 算子简介

`ImageResizeNormalize` 将 NCHW 图像张量双线性 resize 到固定尺寸后按 channel 做 normalize，用于补充 research benchmark 中的模型前后处理与表格特征路径覆盖。

**算子特征**：

- 难度等级：L2（FusedComposite）
- x: [B,C,H,W], mean/std: [C], y: [B,C,out_h,out_w]
- 输出 shape 固定，便于精度比较

## 2. 算子定义

```text
y = (resize_bilinear(x, [out_h, out_w]) - mean) / std
```

## 3. 接口规范

```python
cann_bench.image_resize_normalize(Tensor x, Tensor mean, Tensor std, int out_h=224, int out_w=224, bool align_corners=False) -> Tensor y
```

### 输入参数

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `x` | `[B,C,H,W]` | float16 / bfloat16 / float32 | 输入图像张量 |
| `mean` | `[C]` | float16 / bfloat16 / float32 | channel mean |
| `std` | `[C]` | float16 / bfloat16 / float32 | channel std |
| `out_h` | 标量 | int | 输出高度 |
| `out_w` | 标量 | int | 输出宽度 |
| `align_corners` | 标量 | bool | 双线性插值是否对齐角点 |

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[B,C,out_h,out_w]` | 与主浮点输入相同 | resize + normalize 输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|---|---|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- `x` 必须为 `[B,C,H,W]`，`mean` 和 `std` 必须为 `[C]`。
- `x`、`mean`、`std` 的元素必须为有限值；`std` 必须严格大于 0。
- `out_h` 和 `out_w` 必须为正整数。
- 输出 dtype 与 `x` 一致。

### 支持范围

输入 tensor 各维度与参数的支持范围（计分 case 不超出本表）：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B` | 1 ~ 64 | |
| `C` | 1 ~ 16 | `mean`/`std` 为 `[C]` |
| `H`, `W` | 16 ~ 8192 | |
| `out_h`, `out_w` | 16 ~ 4096 | |
| 输入像素总量 `B·C·H·W` | ≤ 268435456 (2^28) | |
| 输出像素总量 `B·C·out_h·out_w` | ≤ 67108864 (2^26) | 大 case 的规模承重项之一 |
| `align_corners` | true / false | 两种模式均在计分范围内 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：MERE / MARE；bool 输出按精确匹配处理。

## 5. 标准 Golden 代码

```python

import torch


def image_resize_normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_h: int = 224,
    out_w: int = 224,
    align_corners: bool = False,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")
    channels = x.shape[1]
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError("mean/std must be [C]")
    if (
        not torch.isfinite(x).all()
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
    ):
        raise ValueError("x/mean/std must contain finite values")
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")
    out_dtype = x.dtype
    # The FP64 oracle must preserve feature, mean and scale precision.
    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    resized = torch.nn.functional.interpolate(
        x.to(calc_dtype),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    denom = std.to(calc_dtype).reshape(1, channels, 1, 1)
    y = (resized - mean.to(calc_dtype).reshape(1, channels, 1, 1)) / denom
    return y.to(out_dtype)
```
