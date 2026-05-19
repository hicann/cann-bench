# AddRmsNormDynamicQuant 算子 API 描述

## 1. 算子简介

Add、RMSNorm 和动态量化的融合。

**主要应用场景**：
- 大语言模型推理中残差连接 + 归一化 + 量化的融合加速
- Transformer 模型中 RMSNorm 前的残差加法与后处理量化一体化
- INT8 低精度推理的动态量化预处理

**算子特征**：
- 难度等级：L3（FusedComposite）
- 多输入多输出，融合 Add、RMSNorm 和动态量化三个操作
- 输入 x1、x2 为 ND 格式张量，gamma 为缩放参数

## 2. 算子定义

### 数学公式

$$
y, xOut, scaleOut = \text{quantize}(\text{rmsnorm}(x_1 + x_2) \times \gamma)
$$

具体步骤：

1. **Add 操作**：$xOut = x_1 + x_2$
2. **RMSNorm**：$y_{norm} = \frac{xOut}{\sqrt{\text{mean}(xOut^2) + \epsilon}} \times \gamma$
3. **Per-token 对称动态量化**（沿 last-dim）：
   - $scaleOut = \frac{\max_{\text{last-dim}}(|y_{norm}|)}{127}$ — 反量化系数，shape = `x1.shape[:-1]`
   - $y = \text{round}(y_{norm} / scaleOut)$ — int8，shape = `x1.shape`
   - 下游算子用 $scaleOut$ 还原浮点：$x_{fp} = y \times scaleOut$

## 3. 接口规范

### 算子原型

```python
cann_bench.add_rms_norm_dynamic_quant(Tensor x1, Tensor x2, Tensor gamma, float epsilon) -> (Tensor y, Tensor xOut, Tensor scaleOut)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor | 必选 | 第 1 个输入张量 |
| x2 | Tensor | 必选 | 第 2 个输入张量 |
| gamma | Tensor | 必选 | 缩放参数 |
| epsilon | float | 1e-6 | epsilon 值 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x1 相同 | int8 | 量化后的输出张量 |
| xOut | 与输入 x1 相同 | float16 / bfloat16 | Add 结果，x1 + x2 |
| scaleOut | `x1.shape[:-1]` | float32 | 每 token 反量化系数（max/127）；下游 W8A8 / KV cache 反量化必需 |

### 数据类型

| 输入 (x1, x2, gamma) dtype | 输出 y dtype | 输出 xOut dtype | 输出 scaleOut dtype |
|---------------------------|-------------|----------------|-------------------|
| float16 | int8 | float16 | float32 |
| bfloat16 | int8 | bfloat16 | float32 |

### 规则与约束

- x1 和 x2 的 shape 和 dtype 必须一致
- gamma 的 dtype 须与 x1、x2 一致
- x1 为 ND 格式
- epsilon 用于 RMSNorm 的数值稳定性，默认 1e-6
- scaleOut shape = `x1.shape[:-1]`，dtype = float32；语义为**反量化系数**（max/127），与 `dynamic_quant` 一致
- 量化为对称量化（zero_point 恒为 0），scale 基于每 token last-dim 绝对值最大值
- golden 加 `clamp(min=1e-12)` 防止全零输入触发 0/0 NaN

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `M`（x1/x2 第 0 维，batch × seq） | 1 ~ 1048576 | cases.csv 实测 256 ~ 524288 |
| `N`（x1/x2 最后一维，hidden size） | 1 ~ 16384 | cases.csv 实测 128 ~ 16384；x2 / gamma 最后一维须等于 N |
| `epsilon` | 1e-12 ~ 1e-2 | cases.csv 实测 1e-6 ~ 1e-3；RMSNorm 数值稳定项，默认 1e-6 |

约束：x1、x2 形状与 dtype 须完全一致；gamma 形状为 `[N]` 且 dtype 与 x1 一致；归一化沿最后一维进行。

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
AddRmsNormDynamicQuant 算子 Torch Golden 参考实现

Add、RMSNorm 和 per-token 对称动态量化的融合，对齐 torch_npu.npu_add_rms_norm_dynamic_quant：
- scale 为 per-token 反量化系数 (max/127)，shape = x1.shape[:-1]，dtype = fp32
- 下游算子拿 scale 还原浮点：x_fp = y_int8 * scale
"""
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """公式:
        xOut     = x1 + x2
        y_norm   = xOut / sqrt(mean(xOut^2) + eps) * gamma
        scaleOut = row_max(abs(y_norm)) / 127      # 反量化系数, shape = x1.shape[:-1]
        yOut     = round(y_norm / scaleOut)         # int8, shape = x1.shape
    """

    out_dtype = x1.dtype
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    gamma = gamma.to(torch.float32)

    xOut = x1 + x2
    variance = xOut.pow(2).mean(-1, keepdim=True)
    rms = torch.sqrt(variance + epsilon)
    y_norm = xOut / rms * gamma

    abs_max = y_norm.abs().amax(dim=-1, keepdim=True)
    scale_out = (abs_max.clamp(min=1e-12) / 127.0).to(torch.float32)
    y = torch.clamp((y_norm / scale_out).round(), -128, 127).to(torch.int8)
    scale = scale_out.squeeze(-1).to(torch.float32)

    return y, xOut.to(out_dtype), scale
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x1 = torch.randn(2, 4096, dtype=torch.float16, device="npu")
x2 = torch.randn(2, 4096, dtype=torch.float16, device="npu")
gamma = torch.ones(4096, dtype=torch.float16, device="npu")

y, xOut, scaleOut = cann_bench.add_rms_norm_dynamic_quant(x1, x2, gamma, 1e-6)  # INT8 量化
```
