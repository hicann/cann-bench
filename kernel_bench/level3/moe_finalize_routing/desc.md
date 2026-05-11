# MoeFinalizeRouting 算子 API 描述

## 1. 算子简介

在 MoE 计算的最后，合并 MoE FFN (Feedforward Neural Network) 的输出结果，支持共享专家残差连接和专家级偏置。

**主要应用场景**：
- Mixture of Experts (MoE) 模型中专家输出的最终聚合
- MoE Transformer 层中多专家结果的加权融合
- 稀疏激活模型中 token 级别的专家输出合并

**算子特征**：
- 难度等级：L3（FusedComposite）
- 多输入单输出，支持 drop less 和 drop pad 两种模式
- 支持共享专家残差连接（skip1/skip2）
- 支持专家级偏置（bias）

## 2. 算子定义

### 数学公式

$$
\text{expertid} = \text{expert\_for\_source\_row}[i,k]
$$

$$
\text{out}(i,j) = \text{skip1}_{i,j} + \text{skip2}_{i,j} + \sum_{k=0}^{K}(\text{scales}_{i,k} \times (\text{expanded\_permuted\_rows}_{\text{expanded\_src\_to\_dst\_row}_{i+k \times \text{num\_rows},j}} + \text{bias}_{\text{expertid},j}))
$$

### 处理流程

1. 初始化输出：`out = skip1 + skip2`（若 skip1/skip2 存在）
2. 对于每个 token i 和每个选中的专家 k：
   - 根据 `expanded_src_to_dst_row` 获取专家输出的行索引
   - 若索引为 -1（drop pad 模式），则该位置贡献为 0
   - 否则，计算 `scales[i,k] * (expanded_permuted_rows[index] + bias[expert_id])`
3. 将所有专家的贡献累加到输出

### drop_pad_mode 说明

| drop_pad_mode | 模式 | expanded_src_to_dst_row 排列 | 索引范围 |
|---------------|------|------------------------------|---------|
| 0 | drop less | 按列排列 | [0, NUM_ROWS * K - 1] |
| 1 | drop pad | 按列排列 | [-1, E * C - 1] |
| 2 | drop less | 按行排列 | [0, NUM_ROWS * K - 1] |
| 3 | drop pad | 按行排列 | [-1, E * C - 1] |

## 3. 接口规范

### 算子原型

```python
cann_bench.moe_finalize_routing(
    Tensor expanded_permuted_rows,
    Tensor? expanded_src_to_dst_row = None,
    Tensor? skip1 = None,
    Tensor? skip2 = None,
    Tensor? bias = None,
    Tensor? scales = None,
    Tensor? expert_for_source_row = None,
    int drop_pad_mode = 0
) -> Tensor out
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 | Shape |
|------|------|--------|------|-------|
| expanded_permuted_rows | Tensor | 必选 | MoE FFN 输出，经过专家处理的结果 | (NUM_ROWS * K, H) 或 (E, C, H) |
| expanded_src_to_dst_row | Tensor | None | 行索引映射，保存每个专家处理结果的索引 | (NUM_ROWS * K) |
| skip1 | Tensor | None | 共享专家1，残差连接 | (NUM_ROWS, H) |
| skip2 | Tensor | None | 共享专家2，残差连接 | (NUM_ROWS, H) |
| bias | Tensor | None | 专家偏置 | (E, H) |
| scales | Tensor | None | 路由权重，专家缩放因子 | (NUM_ROWS, K) |
| expert_for_source_row | Tensor | None | 专家索引，每行处理的专家号 | (NUM_ROWS, K) |
| drop_pad_mode | int | 0 | 模式选择，取值范围 [0, 3] | - |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| out | (NUM_ROWS, H) | 与 expanded_permuted_rows 相同 | MoE FFN 合并的输出结果 |

### 数据类型

| expanded_permuted_rows | skip1/skip2/bias/scales | expanded_src_to_dst_row | expert_for_source_row |
|------------------------|-------------------------|-------------------------|----------------------|
| float32 | float32 | int32 | int32 |
| float16 | float16 | int32 | int32 |
| bfloat16 | bfloat16 或 float32 (混合精度) | int32 | int32 |

### 规则与约束

1. `skip1` 为 None 时，`skip2` 必须也为 None
2. `scales` 不存在时，K 默认为 1
3. `bias` 存在时，`expert_for_source_row` 必须同时存在
4. `skip1`、`skip2` 的 dtype 需与 `expanded_permuted_rows` 一致
5. `expanded_src_to_dst_row` 的 dtype 为 int32
6. `expert_for_source_row` 的 dtype 为 int32，取值范围 [0, E-1]

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
import numpy as np
from copy import deepcopy

def moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    expanded_src_to_dst_row: torch.Tensor = None,
    skip1: torch.Tensor = None,
    skip2: torch.Tensor = None,
    bias: torch.Tensor = None,
    scales: torch.Tensor = None,
    expert_for_source_row: torch.Tensor = None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    """
    MoE Finalize Routing 算子 Torch Golden 参考实现

    在 MoE 计算的最后，合并 MoE FFN 的输出结果

    Args:
        expanded_permuted_rows: MoE FFN 输出，shape 为 (NUM_ROWS * K, H) 或 (E, C, H)
        expanded_src_to_dst_row: 行索引映射，shape 为 (NUM_ROWS * K)
        skip1: 共享专家1，shape 为 (NUM_ROWS, H)
        skip2: 共享专家2，shape 为 (NUM_ROWS, H)
        bias: 专家偏置，shape 为 (E, H)
        scales: 路由权重，shape 为 (NUM_ROWS, K)
        expert_for_source_row: 专家索引，shape 为 (NUM_ROWS, K)
        drop_pad_mode: 模式选择，取值范围 [0, 3]

    Returns:
        输出张量，shape 为 (NUM_ROWS, H)
    """
    # 确定 K 和 num_rows
    NK = expanded_src_to_dst_row.shape[0]
    K = 1
    if scales is not None:
        K = scales.shape[1]
    num_rows = NK // K
    H = expanded_permuted_rows.shape[-1]

    # 将 expanded_permuted_rows reshape 为 2D
    expanded_permuted_rows = expanded_permuted_rows.reshape(-1, H)

    # 初始化输出：skip1 + skip2
    if (skip1 is not None) and (skip2 is not None):
        out = skip1.clone() + skip2
    elif (skip2 is not None) and (skip1 is None):
        out = skip2.clone()
    elif (skip2 is None) and (skip1 is not None):
        out = skip1.clone()
    else:
        out = torch.zeros(num_rows, H, dtype=expanded_permuted_rows.dtype, device=expanded_permuted_rows.device)

    # 核心计算循环
    for i in range(num_rows):
        for k in range(K):
            # 根据 drop_pad_mode 获取索引位置
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                # 按列排列
                index_pos = k * num_rows + i
            else:
                # 按行排列 (drop_pad_mode == 2 or 3)
                index_pos = i * K + k

            value = expanded_src_to_dst_row[index_pos].item()

            # drop pad 模式：索引为 -1 时贡献为 0
            if value == -1:
                dst_row = torch.zeros(H, dtype=expanded_permuted_rows.dtype, device=expanded_permuted_rows.device)
            else:
                dst_row = expanded_permuted_rows[value, :]

            # 获取缩放因子
            scale_val = 1.0
            if scales is not None:
                scale_val = scales[i, k]

            # 获取专家 ID 和 bias
            if bias is not None and expert_for_source_row is not None:
                expert_id = expert_for_source_row[i, k].item()
                out[i, :] += scale_val * (dst_row + bias[expert_id, :])
            else:
                out[i, :] += scale_val * dst_row

    return out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# drop less 模式示例 (drop_pad_mode=0)
num_rows = 1024
topk = 8
hidden_dim = 512
expert_num = 16

expanded_permuted_rows = torch.randn(num_rows * topk, hidden_dim, dtype=torch.float16, device="npu")
skip1 = torch.randn(num_rows, hidden_dim, dtype=torch.float16, device="npu")
skip2 = torch.randn(num_rows, hidden_dim, dtype=torch.float16, device="npu")
bias = torch.randn(expert_num, hidden_dim, dtype=torch.float16, device="npu")
scales = torch.randn(num_rows, topk, dtype=torch.float16, device="npu")
expanded_src_to_dst_row = torch.randint(0, num_rows * topk, (num_rows * topk,), dtype=torch.int32, device="npu")
expert_for_source_row = torch.randint(0, expert_num, (num_rows, topk), dtype=torch.int32, device="npu")

out = cann_bench.moe_finalize_routing(
    expanded_permuted_rows,
    expanded_src_to_dst_row=expanded_src_to_dst_row,
    skip1=skip1, skip2=skip2,
    bias=bias, scales=scales,
    expert_for_source_row=expert_for_source_row,
    drop_pad_mode=0,
)

# drop pad 模式示例 (drop_pad_mode=1)
expert_capacity = 20
expanded_permuted_rows_3d = torch.randn(expert_num, expert_capacity, hidden_dim, dtype=torch.float16, device="npu")
expanded_src_to_dst_row_pad = torch.randint(-1, expert_num * expert_capacity - 1, (num_rows,), dtype=torch.int32, device="npu")

out = cann_bench.moe_finalize_routing(
    expanded_permuted_rows_3d,
    expanded_src_to_dst_row=expanded_src_to_dst_row_pad,
    skip1=skip1, skip2=skip2,
    bias=bias, scales=None,
    expert_for_source_row=expert_for_source_row,
    drop_pad_mode=1,
)
```
