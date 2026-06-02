# QuantMatmul 算子 API 描述

## 1. 算子简介

量化矩阵乘法算子。对 int8 量化的矩阵乘进行反量化，输出 float16/bfloat16。

**主要应用场景**：
- 大语言模型 W8A8 推理中的 Linear 层
- KV cache 量化流水线中的量化 GEMM
- 静态 per-channel 或动态 per-token 量化方案

**算子特征**：
- 难度等级：L3（Contraction）
- 输入：int8（2–6 维）
- 输出：float16 或 bfloat16
- 支持 per-tensor / per-channel 量化，可选 per-token 缩放

## 2. 算子定义

### 数学公式

- **无 bias**：
$$
out = x1 \mathbin{@} x2 * scale
$$

- **bias 为 int32（pre-scale）**：
$$
out = (x1 \mathbin{@} x2 + bias) * scale
$$

- **bias 为 bfloat16/float16/float32（post-scale）**：
$$
out = x1 \mathbin{@} x2 * scale + bias
$$

- **带 pertoken_scale**：
$$
out = (x1 \mathbin{@} x2 * scale) * pertoken\_scale
$$

- **带 offset（非对称量化）**：
$$
out = x1 \mathbin{@} x2 * scale + offset
$$

### 步骤说明

1. **矩阵乘**：`mm[...,m,n] = x1[...,m,k] @ x2[...,k,n]`，int8 在硬件上累加到 int32。
2. **int32 bias（pre-scale）**：若 bias 为 int32，先与累加结果相加。
3. **反量化 scale**：按 scale 形状广播，支持 per-tensor `[1]` 或 per-channel `[n]`。
4. **offset（非对称量化）**：可选，反量化后加上 zero-point 偏移；对称量化时为 None。
5. **pertoken_scale**：可选，沿 m 维广播的 per-token 缩放。
6. **浮点 bias（post-scale）**：若 bias 为 bf16/fp16/fp32，在反量化后相加。
7. **cast**：按 output_dtype 输出 float16 或 bfloat16。

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
) -> Tensor out
```

### 输入参数

| 参数 | 类型 | Shape | dtype | 描述 |
|------|------|-------|-------|------|
| x1 | Tensor (必选) | `[..., m, k]`，2–6 维 | int8 | 左矩阵 |
| x2 | Tensor (必选) | `[..., k, n]`，2–6 维；最后一维 ≤ 65535 | int8 | 右矩阵 |
| scale | Tensor (必选) | `[t]` (t=1 或 n) | float32 / bfloat16 | 量化缩放因子 |
| offset | Tensor (可选) | `[t]` (t=1 或 n) | float32 | 非对称量化偏移 (zero-point)；对称量化为 None |
| pertoken_scale | Tensor (可选) | `[m]` | float32 | per-token 缩放因子 |
| bias | Tensor (可选) | `[n]` / `[1, n]` / `[batch, 1, n]` | int32 / bfloat16 / float16 / float32 | 偏置项 |
| output_dtype | str (可选) | - | - | 输出 dtype："float16"（默认）或 "bfloat16" |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| out | `[..., m, n]` | float16 / bfloat16 | 计算结果 |

### 数据类型组合

| x1 | x2 | scale | offset | pertoken_scale | bias | output_dtype |
|----|----|-------|--------|----------------|------|--------------|
| int8 | int8 | float32 | float32/None | float32/None | int32/None | float16 |
| int8 | int8 | float32 | float32/None | float32/None | int32/bfloat16/float16/float32/None | bfloat16 |
| int8 | int8 | bfloat16 | float32/None | None | int32/None | bfloat16 |

### 规则与约束

- `x1.shape[-1] == x2.shape[-2] == k`；`x2.shape[-1] ≤ 65535`
- bias：输出 2/4/5/6 维时必须 1D；输出 3 维时可为 1D 或 3D
- scale 为 bfloat16 时，output_dtype 必须为 bfloat16

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `batch`（前导批维度） | 1 ~ 256 | 2-6 维输入；2D 时无 batch；cases.csv 实测 2D 与 3D B=8 |
| `m`（x1 倒数第二维） | 1 ~ 4096 | cases.csv 实测 1 ~ 4096 |
| `k`（x1 最后一维 / x2 倒数第二维） | 16 ~ 16384 | `x1.shape[-1] == x2.shape[-2]`；cases.csv 实测 256 ~ 7168 |
| `n`（x2 最后一维） | 16 ~ 65535 | 硬件限制 ≤ 65535；cases.csv 实测 512 ~ 14336 |
| `scale.shape[0]` | 1 或 n | per-tensor=1 / per-channel=n；cases.csv 实测 1 / n |
| `offset.shape[0]` | 1 或 n | 非对称量化时与 scale 同形；cases.csv 均为对称量化（offset=None） |
| `pertoken_scale.shape[0]` | = m | cases.csv 实测 1024 |
| `bias.shape[-1]` | = n | 1D `[n]` 或 3D `[batch, 1, n]` |
| `output_dtype` | `float16` / `bfloat16` | cases.csv 全部覆盖 |

约束：x1/x2 必须同为 int8；scale 为 bfloat16 时 output_dtype 必须为 bfloat16。

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

| 数据类型 | FLOAT16 | BFLOAT16 |
|----------|---------|----------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch
from typing import Optional

def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
) -> torch.Tensor:
    """
    量化矩阵乘法

    Args:
        x1: [..., m, k] int8 左矩阵
        x2: [..., k, n] int8 右矩阵
        scale: [t] 反量化 scale (float32 / bfloat16)
        offset: [t] 非对称量化偏移 (float32, t=1 或 n)，反量化时 out += offset；
                对称量化时为 None
        pertoken_scale: [m] per-token scale (float32)
        bias: [n] 或 [batch, 1, n] 偏置，int32 走 pre-scale，浮点走 post-scale
        output_dtype: 输出类型 "float16"（默认）或 "bfloat16"

    Returns:
        out: [..., m, n] float16 或 bfloat16
    """
    # 矩阵乘：int8/int32 输入直接走 fp64 matmul。
    # PR-001: int8 × int8 单乘积 ≤ 127² = 16129；K 个累加最大 |mm| = 16129·K，
    # 即便 K=65535 也只到 ~1.06e9，远小于 fp64 整数精确上界 2^53 (≈9e15)，
    # 故 fp64 matmul 对该量程是 *精确* 的（与 int64 累加逐位相等）。
    # 不用 int64 matmul：CPU 上 int64 GEMM 无 BLAS，朴素实现比 fp64 慢 ~1000x
    # （大 shape 单次可达 ~60s），会令评测在 golden 阶段超时。fp64 走 BLAS，
    # 精度不变而速度恢复。fp32 不可用（24-bit 尾数，K>1024 会溢出）。
    if x1.dtype in (torch.int8, torch.int32) and x2.dtype in (torch.int8, torch.int32):
        mm = torch.matmul(x1.double(), x2.double())
    else:
        # bf16/fp16 输入路径维持原 fp32 等效计算
        mm = torch.matmul(x1.float(), x2.float()).double()

    # int32 bias 在反量化前累加 (pre-scale)
    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.double()

    # 反量化 scale
    y = mm * scale.double()

    # 非对称量化偏移 (zero-point 校正)：out = mm*scale + offset
    # offset 与 scale 配对（NPU 侧由 npu_trans_quant_param(scale, offset) 打包）；
    # 对称量化时 offset=None，此步为 no-op。
    if offset is not None:
        y = y + offset.double()

    # pertoken_scale 沿 m 维广播
    if pertoken_scale is not None:
        y = y * pertoken_scale.double().unsqueeze(-1)

    # 浮点 bias 在反量化后相加 (post-scale)
    if bias is not None and bias.dtype != torch.int32:
        y = y + bias.double()

    # 输出 dtype，默认 float16
    if output_dtype is None or output_dtype == "float16":
        return y.to(torch.float16)
    elif output_dtype == "bfloat16":
        return y.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")
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

# 带 int32 bias + pertoken_scale，bfloat16 输出
x1 = torch.randint(-128, 127, (1024, 4096), dtype=torch.int8, device="npu")
x2 = torch.randint(-128, 127, (4096, 14336), dtype=torch.int8, device="npu")
scale = torch.rand(14336, dtype=torch.float32, device="npu") * 0.01
bias = torch.randint(-100, 100, (14336,), dtype=torch.int32, device="npu")
pertoken = torch.rand(1024, dtype=torch.float32, device="npu")
out = cann_bench.quant_matmul(x1, x2, scale, pertoken_scale=pertoken,
                               bias=bias, output_dtype="bfloat16")
```

### CANN 底层实现

- **aclnnQuantMatmulV4**: 基础量化矩阵乘
