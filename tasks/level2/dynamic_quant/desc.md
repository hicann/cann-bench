# DynamicQuant 算子 API 描述

## 1. 算子简介

对输入张量进行 per-token 对称动态量化。

**主要应用场景**：
- 大语言模型推理加速中的动态量化（W8A8 / W4A8 等方案）
- KV Cache 量化压缩以节省显存
- 模型部署阶段的在线量化处理

**算子特征**：
- 难度等级：L2（FusedComposite）
- 单输入单输出，涉及求最大值、缩放、四舍五入等多步融合计算
- 输入支持 2-8 维张量

## 2. 算子定义

### 数学公式

$$
scaleOut = \frac{\max_{\text{last-dim}}(|x|)}{127}
$$

$$
yOut = \text{round}\left(\frac{x}{scaleOut}\right)
$$

其中：
- $\max_{\text{last-dim}}(|x|)$ 表示沿 last-dim（每个 token）取绝对值最大值
- 量化目标固定为 int8（对应 $dtypeMax = 127$）
- $\text{round}$ 为四舍五入到最近整数（half-to-even）

> **NPU API 约束**：CANN `torch_npu.npu_dynamic_quant` 只支持 per-token 量化（沿 last-dim），不暴露 axis 参数；不支持 float32 输入；不支持 1D 张量。本算子规格与 NPU API 真实能力对齐。

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_quant(Tensor x) -> (Tensor y, Tensor scale)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，shape ≥ 2 维，dtype ∈ {float16, bfloat16} |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | int8 | 量化后的张量 |
| scale | `x.shape[:-1]` | float32 | 每 token 的反量化系数；下游 W8A8 / KV cache 反量化算子必需 |

### 数据类型

| 输入 dtype | y dtype | scale dtype |
|-----------|---------|-------------|
| float16 | int8 | float32 |
| bfloat16 | int8 | float32 |

### 规则与约束

- 输入 x 必须为 2 ~ 8 维张量（NPU API 硬性要求 ≥ 2 维）
- 输入 dtype 仅支持 float16 / bfloat16（NPU API 不支持 float32）
- 量化为对称量化（zero_point 恒为 0），scale 基于每 token 绝对值最大值计算
- 输出 shape 与输入 shape 完全一致
- 全零 token 防除零：golden 对 abs_max 应用 `clamp(min=1e-12)` 保证 scale > 0，避免 0/0 NaN（NPU 实现行为类似；用户仍应避免大量全零输入以免数值无效）
- golden 对 `round(x / scale)` 应用 `clamp(-128, 127)` 防止 int8 模 256 截断（罕见数值边界场景）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(x)`（输入维度数） | 2 ~ 8 | NPU API 硬性要求 ≥ 2；cases.csv 实测 2 ~ 5 维 |
| 各维度大小 `dim_i` | 1 ~ 16384 | cases.csv 实测最小 2、最大 16,384 |
| last-dim 大小（量化粒度，每 token 长度） | 1 ~ 16384 | cases.csv 实测 67 ~ 16,384；沿此维取每 token 的 max-abs |
| token 总数（前导维度乘积） | 1 ~ 2^20 | cases.csv 实测最大约 163K（11×13×17×67） |
| 张量总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 128M |
| 输入 dtype | float16 / bfloat16 | NPU API 不支持 float32 |

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

**int8 输出特殊阈值**：

量化算子输出为整数类型，round 操作存在舍入误差，允许 ±1 的绝对误差：

| 输出类型 | 阈值 | 说明 |
|----------|------|------|
| int8 | 1.0 | 允许 \|actual - golden\| ≤ 1 |

**通过条件**：`|actual - golden| ≤ threshold`


## 5. 标准 Golden 代码

```python
import torch
from typing import Tuple

"""
DynamicQuant 算子 Torch Golden 参考实现

per-token 对称动态量化 (沿 last-dim)，对齐 NPU torch_npu.npu_dynamic_quant 默认行为：
返回 (y, scale) 双输出。scale 是下游 W8A8 / KV cache 反量化算子必需的输入，
所以是该算子的本质输出之一，不能省。
"""
def dynamic_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token 对称动态量化 (axis=-1, dtype_max=127 → int8)。

    Args:
        x: 输入张量 (fp16/bf16)，shape ≥ 2 维

    Returns:
        y:     量化后的张量 (int8, shape 与 x 一致)
        scale: per-token 反量化系数 (float32, shape = x.shape[:-1])
    """
    x_compute = x.to(torch.float32)
    abs_max = torch.max(torch.abs(x_compute), dim=-1, keepdim=True)[0]
    # clamp(min=1e-12) 防全零 token 触发 0/0 NaN
    scale_out = abs_max.clamp(min=1e-12) / 127.0
    # clamp(-128, 127) 防 int8 模 256 截断
    y = torch.clamp(torch.round(x_compute / scale_out), -128, 127).to(torch.int8)
    scale = scale_out.squeeze(-1).to(torch.float32)
    return y, scale
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 2D per-token quant
x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y, scale = cann_bench.dynamic_quant(x)
# y.shape = (1024, 1024), y.dtype = int8
# scale.shape = (1024,),    scale.dtype = float32

# 4D per-token quant（每行最后维量化）
x = torch.randn(2, 8, 256, 256, dtype=torch.bfloat16, device="npu")
y, scale = cann_bench.dynamic_quant(x)
# y.shape     = (2, 8, 256, 256), y.dtype     = int8
# scale.shape = (2, 8, 256),       scale.dtype = float32

# 下游反量化（典型用法）
dequant_x = y.to(torch.float32) * scale.unsqueeze(-1)
```
