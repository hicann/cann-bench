# StridedSlice 算子 API 描述

## 1. 算子简介

使用步长对输入张量进行多维切片，提取子张量。支持 begin_mask、end_mask 控制边界、shrink_axis_mask 收缩维度、new_axis_mask 插入新维度、ellipsis_mask 省略号等功能。

**主要应用场景**：
- 深度学习模型中对特征图进行区域裁剪和下采样
- 序列模型中按步长提取时间步或特征片段
- 数据预处理中对多维张量进行灵活的切片操作
- 模型推理中通过掩码机制实现复杂的维度操控

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 单输入单输出，支持 0-8 维输入，支持负数步长和多种掩码控制

## 2. 算子定义

### 数学公式

$$
y[i,j,k,...] = x[\text{begin}[0]:\text{end}[0]:\text{strides}[0],\ \text{begin}[1]:\text{end}[1]:\text{strides}[1],\ \text{begin}[2]:\text{end}[2]:\text{strides}[2],\ ...]
$$

各掩码参数的作用：
- **begin_mask**：二进制掩码，位 1 表示该维度从 0 开始，忽略 begin 值
- **end_mask**：二进制掩码，位 1 表示该维度切到末尾，忽略 end 值
- **ellipsis_mask**：二进制掩码，位 1 表示该维度使用省略号标记
- **shrink_axis_mask**：二进制掩码，位 1 表示该维度被收缩掉（维度大小为 1）
- **new_axis_mask**：二进制掩码，位 1 表示该位置插入大小为 1 的新维度

## 3. 接口规范

### 算子原型

```python
cann_bench.strided_slice(Tensor x, int[] begin, int[] end, int[] strides, int begin_mask, int end_mask, int ellipsis_mask, int shrink_axis_mask, int new_axis_mask) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| begin | int[] | 必选 | 切片起始位置数组，长度等于输入维度数 |
| end | int[] | 必选 | 切片结束位置数组，长度等于输入维度数 |
| strides | int[] | 必选 | 切片步长数组，长度等于输入维度数，支持负数步长 |
| begin_mask | int64_t | — | 二进制掩码，位 1 表示该维度从 0 开始，位 0 使用 begin 值 |
| end_mask | int64_t | — | 二进制掩码，位 1 表示该维度切到末尾，位 0 使用 end 值 |
| ellipsis_mask | int64_t | — | 二进制掩码，位 1 表示该维度使用省略号标记 |
| shrink_axis_mask | int64_t | — | 二进制掩码，位 1 表示该维度被收缩掉（维度大小为 1） |
| new_axis_mask | int64_t | — | 二进制掩码，位 1 表示该位置插入大小为 1 的新维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由 begin、end、strides 及各掩码决定 | 与输入 x 相同 | 输出张量，切片结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| int8 | int8 |
| uint8 | uint8 |
| int32 | int32 |
| int64 | int64 |
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入支持 0-8 维张量
- begin、end、strides 数组长度必须等于输入维度数
- strides 中每个元素不能为 0，支持负数步长（表示逆序切片）
- begin 和 end 支持负数索引（表示从末尾倒数）
- 各掩码参数以二进制位的形式对应各维度，低位对应低维度
- 输出 dtype 与输入 dtype 一致

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数（rank） | 0 ~ 8 | cases.csv 实测 1D / 2D / 3D / 4D |
| `x` 每维大小 | 1 ~ 1048576 | cases.csv 实测最小 64，最大 1048576（1D），2D 最大 8192×8192 |
| `x` 总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 128M（case 20: [64,128,128,128]） |
| `x` dtype | int8 / uint8 / int32 / int64 / float16 / float32 / bfloat16 | cases.csv 实测 int32 / int64 / float16 / float32 / bfloat16（int8 / uint8 未覆盖） |
| `begin[i]` | -2^31 ~ 2^31-1 | cases.csv 实测 0 ~ 1024（负数索引语义支持但未在 cases 中出现） |
| `end[i]` | -2^31 ~ 2^31-1 | cases.csv 实测 -1 / 64 ~ 524288（-1 表示倒数第一） |
| `strides[i]` | -2^15 ~ 2^15（非 0） | cases.csv 实测 1 ~ 4（负数步长语义支持但未在 cases 中出现） |
| `len(begin)` / `len(end)` / `len(strides)` | 等于 `x` rank（或考虑 ellipsis_mask / new_axis_mask 调整后的有效长度） | cases.csv 实测长度 1 / 2 / 3 / 4 |
| `begin_mask` | 0 ~ 2^8-1 | cases.csv 实测 0 / 1 / 3 |
| `end_mask` | 0 ~ 2^8-1 | cases.csv 实测 0 / 1 / 2 |
| `ellipsis_mask` | 0 ~ 2^8-1（至多 1 位置 1） | cases.csv 实测 0 / 1 |
| `shrink_axis_mask` | 0 ~ 2^8-1 | cases.csv 实测 0 / 1 / 2 |
| `new_axis_mask` | 0 ~ 2^8-1 | cases.csv 实测 0 / 1 / 2 |

约束：`strides[i]` 不能为 0；`ellipsis_mask` 至多有一位为 1；`shrink_axis_mask` 位与 `new_axis_mask` 位不可在同一参数位置同时为 1；切片结果各维度大小 `ceil((end - begin) / strides) ≥ 0`。

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

def strided_slice(
    x: torch.Tensor, begin: list, end: list, strides: list,
    begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
    shrink_axis_mask: int = 0, new_axis_mask: int = 0
) -> torch.Tensor:
    """
    使用步长对输入张量进行多维切片，对标 TensorFlow strided_slice。

    Args:
        x: 输入张量
        begin: 切片起始位置数组
        end: 切片结束位置数组
        strides: 切片步长数组，支持负数步长
        begin_mask: 二进制掩码，位1表示该维度从0开始
        end_mask: 二进制掩码，位1表示该维度切到末尾
        ellipsis_mask: 二进制掩码，位1表示省略号标记
        shrink_axis_mask: 二进制掩码，位1表示收缩该维度（取单元素）
        new_axis_mask: 二进制掩码，位1表示插入新维度

    Returns:
        输出张量，切片结果
    """
    ndim = x.dim()
    shape = x.shape

    # 处理 ellipsis_mask
    ellipsis_pos = None
    for i in range(32):
        if ellipsis_mask & (1 << i):
            ellipsis_pos = i
            break

    # 计算 new_axis 数量
    num_new_axis = 0
    for i in range(len(begin) if begin else 0):
        if new_axis_mask & (1 << i):
            num_new_axis += 1

    indices = []
    input_dim_idx = 0
    param_idx = 0

    if ellipsis_pos is not None:
        num_params = len(begin) if begin else 0
        num_ellipsis_dims = ndim - (num_params - num_new_axis - 1)
        if num_ellipsis_dims < 0:
            num_ellipsis_dims = 0

    while input_dim_idx < ndim or param_idx < (len(begin) if begin else 0):
        if param_idx < len(begin) and (new_axis_mask & (1 << param_idx)):
            indices.append(None)
            param_idx += 1
            continue

        if ellipsis_pos is not None and param_idx == ellipsis_pos:
            for _ in range(num_ellipsis_dims):
                indices.append(slice(None, None, None))
                input_dim_idx += 1
            param_idx += 1
            continue

        if input_dim_idx < ndim and param_idx < len(begin):
            dim_size = shape[input_dim_idx]
            b = begin[param_idx] if param_idx < len(begin) else 0
            e = end[param_idx] if param_idx < len(end) else dim_size
            s = strides[param_idx] if param_idx < len(strides) else 1

            if b < 0:
                b = b + dim_size
            if e < 0:
                e = e + dim_size

            if begin_mask & (1 << param_idx):
                b = 0 if s > 0 else dim_size - 1

            if end_mask & (1 << param_idx):
                e = dim_size if s > 0 else -1

            if shrink_axis_mask & (1 << param_idx):
                indices.append(b)
            else:
                indices.append(slice(b, e, s))

            input_dim_idx += 1
            param_idx += 1
        elif input_dim_idx < ndim:
            indices.append(slice(None, None, None))
            input_dim_idx += 1
        else:
            if param_idx < len(begin) and (new_axis_mask & (1 << param_idx)):
                indices.append(None)
            param_idx += 1

    return x[tuple(indices)]
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.strided_slice(x, [0, 0], [512, 512], [2, 2], 0, 0, 0, 0, 0)

x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y = cann_bench.strided_slice(x, [0, 0, 0, 0], [-1, -1, 128, 128], [1, 1, 2, 2], 0, 0, 0, 0, 0)
```
