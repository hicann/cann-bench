# ScatterBlockUpdate 算子 API 描述

## 1. 算子简介

ScatterBlockUpdate算子

**主要应用场景**：
- Embedding 查表与索引
- 序列特征抽取
- Top-K 结果取值

**算子特征**：
- 难度等级：L3（IndexGather）
- 3 输入，1 输出，0 个属性参数
- 支持 ND 格式输入

## 2. 算子定义

### 数学公式

设：
- `input` 的 shape 为 $(b_n, b_s, D)$
- `indices` 的 shape 为 $(T, 2)$，其中 $T$ 为待更新的索引条数
- `update` 的 shape 为 $(T, D)$

算子在 `input` 上**原地**按索引逐行覆盖（`output` 即更新后的 `input`，二者为同一块内存）：

$$
output = input
$$

$$
output\bigl[indices[k, 0],\; indices[k, 1],\; :\bigr] = update[k, :], \quad k = 0, 1, \dots, T-1
$$

其中 $indices[k, 0]$ 与 $indices[k, 1]$ 分别表示第 $k$ 条更新在 `input` 第 0 维与第 1 维上的位置。`output` 与 `input` 共享同一块内存（原地更新）；未被索引触碰的行不产生任何读写流量义务。索引不能出现重复元素。

## 3. 接口规范

### 算子原型

```python
cann_bench.scatter_block_update(Tensor input, Tensor indices, Tensor update) -> Tensor input
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入张量 `input` |
| indices | Tensor | 必选 | 输入张量 `indices` |
| update | Tensor | 必选 | 输入张量 `update` |


### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| input | 与输入相关 | bfloat16/float16/float32/int8 | 输出张量 `input` |


### 数据类型

| 参数 | 数据类型 | 数据格式 | 维度（shape） | 非连续 Tensor | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `input` | FLOAT16、BFLOAT16、FLOAT32、INT8 | ND | ($b_n$, $b_s$, $D$) | 支持 | 待更新张量，原地更新；必选参数，不能为空 Tensor |
| `indices` | INT32、INT64 | ND | ($T$, 2) | 支持 | 索引；必选参数，不能为空 Tensor，不能出现重复元素 |
| `update` | 与 `input` 一致 | 与 `input` 一致 | ($T$, $D$) | 支持 | 更新值；必选参数，不能为空 Tensor |
| `input`（输出） | 与输入 `input` 一致 | ND | ($b_n$, $b_s$, $D$) | - | 原地更新后的输出张量 |

### 规则与约束

1. **输入约束**：
   - 输入 Tensor `input`、`indices`、`update` 不能为空，且必须为 Device 侧 Tensor；
   - 所有输入/输出 Tensor 的数据格式仅支持 `ACL_FORMAT_ND`；
2. **内存约束**：
   - Workspace 内存需在 Device 侧申请，且大小需严格匹配第一段接口返回值；
   - 非连续 Tensor 无需提前转为连续，算子内部自动处理。

#### 规格约束

| 规格项 | 规格 | 规格说明 |
| :--- | :--- | :--- |
| `b_n` | 1 ~ 16384 | `b_n` 支持 1 ~ 16384 范围以内。 |
| `b_s` | 1 ~ 1024 | `b_s` 支持 1 ~ 1024 范围以内。 |
| `D` | 1 ~ 256 | `D` 支持 1 ~ 256 范围以内。 |
| `T` | 1 ~ 262144 | `T` 支持 1 ~ 262144 范围以内。 |

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `ndim`（输入维度数） | 2 ~ 3 | cases 实测范围 |
| `dim_0`（第0维大小） | 1 ~ 1048577 | cases 实测范围 |
| `dim_1`（第1维大小） | 1 ~ 2048 | cases 实测范围 |
| `dim_2`（第2维大小） | 1 ~ 2048 | cases 实测范围 |
| `dtype` | bfloat16, float16, float32, int32, int64, int8 | cases 实测覆盖 |

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

import os
import torch
import random
import numpy as np
from typing import Optional


def make_noncontig(tensor: torch.Tensor, pad: int, dim: int, device: str):
    """在指定维度后 padding，使张量在该维度上非连续。"""
    shape = list(tensor.shape)
    shape[dim] += pad
    padded = torch.zeros(shape, dtype=tensor.dtype, device=device)
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(0, tensor.shape[dim])
    padded[tuple(slices)] = tensor
    return padded[tuple(slices)]


def get_input(input, indices, update, **kwargs):
    """
    对 indices 做规则化构造，把 T 个 update 按行优先映射到 input 的 (bn, bs) 位置。
    若 case 名称包含 noncontiguous，则将 input 在第 1 维构造为非连续张量。
    """
    bn, bs, D = input.shape
    T, _ = indices.shape

    # 构造规则索引：第0列按 bs 分块，第1列在块内循环
    indices_flat = torch.arange(T, dtype=torch.int32, device=input.device)
    col1 = indices_flat // bs
    col2 = indices_flat % bs
    indices = torch.stack([col1, col2], dim=1)

    # 处理非连续输入场景
    case_name = kwargs.get("case_name", "")
    case_idx = kwargs.get("case_idx", 0)
    if "noncontiguous" in case_name:
        random.seed(case_idx)
        pad = random.randint(1, 128)
        input = make_noncontig(input, pad=pad, dim=1, device=str(input.device))

    return (input, indices, update)


def scatter_block_update(
    input: torch.Tensor, indices: torch.Tensor, update: torch.Tensor
):
    """
    ScatterBlockUpdate算子Torch Golden参考实现

    公式: scatter_block_update(...)
    """
    # Make a copy of input to avoid modifying it
    output = input.clone()
    if indices.shape[0] == 0:
        return output

    # The task requires distinct destination blocks. Assign those blocks in one
    # operation instead of dispatching a Python slice assignment for every row.
    # Keep the row-wise behavior for malformed update shapes: missing rows
    # raise, extra rows are ignored, and a flat update is not broadcast.
    if (
        indices.dtype in (torch.int32, torch.int64)
        and update.ndim == 2
        and update.shape[0] == indices.shape[0]
    ):
        rows, cols = indices.to(torch.int64).unbind(dim=1)
        # Normalize negative indices for duplicate detection only. Use the
        # original indices for assignment so out-of-bounds indices still fail.
        canonical_rows = torch.where(rows < 0, rows + input.shape[0], rows)
        canonical_cols = torch.where(cols < 0, cols + input.shape[1], cols)
        destinations = canonical_rows * input.shape[1] + canonical_cols
        if torch.unique(destinations).numel() == indices.shape[0]:
            output[rows, cols, :] = update.to(dtype=output.dtype)
            return output

    # Preserve the historical last-write-wins behavior for duplicate indices,
    # even though such inputs are outside the task's distinct-index contract.
    for k in range(indices.shape[0]):
        idx0 = indices[k, 0].item()
        idx1 = indices[k, 1].item()
        output[idx0, idx1, :] = update[k, :]
    return output
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

input = torch.randn(2048, 128, 1, dtype=torch.bfloat16, device="npu")
indices = torch.randn(2048, 2, dtype=torch.int32, device="npu")
update = torch.randn(2048, 1, dtype=torch.bfloat16, device="npu")
input = cann_bench.scatter_block_update(input, indices, update)
```
