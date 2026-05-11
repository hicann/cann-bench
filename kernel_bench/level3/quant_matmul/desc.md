# QuantMatmul 算子 API 描述

## 1. 算子简介

量化矩阵乘法算子。完成量化的矩阵乘计算，支持 int8/int4 输入，输出 float16/bfloat16/int8/int32。

**主要应用场景**：
- 大语言模型 W8A8/W4A4 推理中的 Linear 层
- KV cache 量化流水线中的量化 GEMM
- 静态 per-channel 或动态 per-token 量化方案

**算子特征**：
- 难度等级：L3（Contraction）
- 输入 2–6 维，int8 或 int4 数据；输出 float16/bfloat16/int8/int32
- 支持 per-tensor / per-channel / per-group / per-token 多级量化

## 2. 算子定义

### 数学公式

- **无 bias**：
$$
out = x1 \mathbin{@} x2 * scale + offset
$$

- **bias 为 int32**：
$$
out = (x1 \mathbin{@} x2 + bias) * scale + offset
$$

- **bias 为 bfloat16/float32（无 offset）**：
$$
out = x1 \mathbin{@} x2 * scale + bias
$$

- **带 pertoken_scale**：
$$
out = (x1 \mathbin{@} x2 * scale + offset) * pertoken\_scale
$$

### 步骤说明

1. **矩阵乘**：`mm[...,m,n] = x1[...,m,k] @ x2[...,k,n]`，int8 在硬件上累加到 int32。
2. **int32 bias（pre-scale）**：若 bias 为 int32，先与累加结果相加。
3. **反量化 scale**：按 scale 形状广播，支持 per-tensor `[1]` 或 per-channel `[n]`。
4. **offset**：反量化后的偏移调整。
5. **pertoken_scale**：可选，沿 m 维广播的 per-token 缩放。
6. **浮点 bias（post-scale）**：若 bias 为 bf16/fp32 且无 offset，在反量化后相加。
7. **cast**：按 output_dtype 输出。

## 3. 接口规范

### 算子原型

```python
cann_bench.quant_matmul(
    Tensor x1,
    Tensor x2,
    Tensor scale,
    *,
    Tensor? offset=None,
    Tensor? pertoken_scale=None,
    Tensor? bias=None,
    str? output_dtype=None,
    int[]? group_sizes=None,
) -> Tensor out
```

### 输入参数

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| x1 | Tensor (必选) | `[..., m, k]`，2–6 维 | int8 / int32 | 左矩阵。int32 表示 int4 类型，每个 int32 存放 8 个 int4 |
| x2 | Tensor (必选) | `[..., k, n]`，2–6 维；最后一维 ≤ 65535 | int8 / int32 | 右矩阵，与 x1 dtype 一致 |
| scale | Tensor (必选) | `[t]` (t=1 或 n)，或 2D `[ceil(k/group_k), n]` | float32 / int64 / bfloat16 | 量化缩放因子 |
| offset | Tensor (可选) | `[t]` (t=1 或 n)，或 2D（与 scale 相同） | float32 / float16 | 反量化偏移。scale 为 2D 时必选 |
| pertoken_scale | Tensor (可选) | `[m]` | float32 | per-token 缩放因子 |
| bias | Tensor (可选) | `[n]` / `[1, n]` / `[batch, 1, n]` | int32 / bfloat16 / float16 / float32 | 偏置项 |
| output_dtype | str (可选) | None | - | 输出 dtype："int8" / "float16" / "bfloat16" / "int32"，None 等价于 "int8" |
| group_sizes | int[] (可选) | - | - | 分组量化粒度 [group_m, group_n, group_k] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| out | `[..., m, n]` | 由 output_dtype 决定 | 计算结果 |

### 数据类型组合

**Atlas 推理系列加速卡**：
| x1 | x2 | scale | offset | bias | pertoken_scale | output_dtype |
|----|----|-------|--------|------|---------------|--------------|
| int8 | int8 | int64/float32 | None | int32/None | None | float16 |
| int8 | int8 | int64/float32 | float32/None | int32/None | None | int8 |

**Atlas A2/A3 系列**：
| x1 | x2 | scale | offset | bias | pertoken_scale | output_dtype |
|----|----|-------|--------|------|---------------|--------------|
| int8 | int8 | int64/float32 | None | int32/None | None | float16 |
| int8 | int8 | int64/float32 | float32/None | int32/None | None | int8 |
| int8 | int8 | float32/bfloat16 | None | int32/bfloat16/float32/None | float32/None | bfloat16 |
| int8 | int8 | float32 | None | int32/bfloat16/float32/None | float32 | float16 |
| int32 | int32 | int64/float32 | None | int32/None | None | float16 |
| int32 | int32 | float32 | float16 | None | float32 | bfloat16/float16 |
| int8 | int8 | float32/bfloat16 | None | int32/None | None | int32 |

### 规则与约束

- `x1.shape[-1] == x2.shape[-2] == k`；`x2.shape[-1] ≤ 65535`
- scale 为 2D 时，offset 必选且 shape 与 scale 相同
- bias：输出 2/4/5/6 维时必须 1D；输出 3 维时可为 1D 或 3D
- int4 场景：x1/x2 为 int32，每个 int32 存放 8 个 int4，shape 最后一维缩小 8 倍
- group_sizes 取值范围 [0, 65535]

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
from typing import Optional, List

def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
    group_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    量化矩阵乘法，对标 torch_npu.npu_quant_matmul
    
    Args:
        x1: [..., m, k] int8/int32 左矩阵
        x2: [..., k, n] int8/int32 右矩阵
        scale: [t] 或 2D，反量化 scale
        offset: [t] 或 2D，反量化偏移
        pertoken_scale: [m] per-token scale
        bias: [n] 或 [batch, 1, n] 偏置
        output_dtype: 输出类型，默认 int8
        group_sizes: 分组量化粒度
    
    Returns:
        out: [..., m, n]
    """
    # 矩阵乘（int8 用 float32 等效）
    mm = torch.matmul(x1.float(), x2.float())
    
    # int32 bias 在反量化前累加
    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.float()
    
    # 反量化 scale
    y = mm * scale.float()
    
    # offset
    if offset is not None:
        y = y + offset.float()
    
    # pertoken_scale
    if pertoken_scale is not None:
        y = y * pertoken_scale.float().unsqueeze(-1)
    
    # 浮点 bias（无 offset 时）
    if bias is not None and bias.dtype != torch.int32 and offset is None:
        y = y + bias.float()
    
    # 输出 dtype
    if output_dtype is None or output_dtype == "int8":
        out_dtype = torch.int8
    elif output_dtype == "float16":
        out_dtype = torch.float16
    elif output_dtype == "bfloat16":
        out_dtype = torch.bfloat16
    elif output_dtype == "int32":
        out_dtype = torch.int32
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")
    
    return y.to(out_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# int8 输入，float16 输出，per-channel scale
x1 = torch.randint(-128, 127, (1024, 3584), dtype=torch.int8, device="npu")
x2 = torch.randint(-128, 127, (3584, 3584), dtype=torch.int8, device="npu")
scale = torch.rand(3584, dtype=torch.float32, device="npu") * 0.01
out = cann_bench.quant_matmul(x1, x2, scale, output_dtype="float16")

# 带 offset + int32 bias + pertoken_scale，bfloat16 输出
x1 = torch.randint(-128, 127, (1024, 4096), dtype=torch.int8, device="npu")
x2 = torch.randint(-128, 127, (4096, 14336), dtype=torch.int8, device="npu")
scale = torch.rand(14336, dtype=torch.float32, device="npu") * 0.01
offset = torch.rand(14336, dtype=torch.float32, device="npu")
bias = torch.randint(-100, 100, (14336,), dtype=torch.int32, device="npu")
pertoken = torch.rand(1024, dtype=torch.float32, device="npu")
out = cann_bench.quant_matmul(x1, x2, scale, offset=offset, pertoken_scale=pertoken, 
                               bias=bias, output_dtype="bfloat16")
```

### CANN 底层实现

- **aclnnQuantMatmulV4**: 基础量化矩阵乘
- **aclnnQuantMatmulV5**: A8W4 / A4W4 分组量化
- **aclnnQuantMatmulWeightNz**: weight NZ 格式优化
