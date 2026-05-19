# SwiGlu 算子 API 描述

## 1. 算子简介

标准 SwiGLU 激活函数(Shazeer 2020 "GLU Variants Improve Transformer")。输入在指定 dim 上拆分为 x0 / x1 两等份,x0 经 SiLU 激活(`Swish_1(x) = x · sigmoid(x)`)后与 x1 做门控乘法。

**主要应用场景**：
- LLM FFN 中的 SwiGLU 门控(Llama / PaLM / Gemma 等)
- 替代传统 GLU 与 ReLU FFN,提供更平滑的梯度与更强表达力

**算子特征**：
- 难度等级：L1（Elementwise）
- P2 op：torch 无同名接口;reference 实现取自 `torch_npu.npu_swiglu` / ACLNN `aclnnSwiGlu`,两者都固定 Swish 的 β = 1(即 SiLU)
- 单输入单输出,沿指定 dim 拆分 + 元素级运算

## 2. 算子定义

### 数学公式

$$
\text{output} = \text{SiLU}(x_0) \odot x_1 = \left( x_0 \cdot \sigma(x_0) \right) \odot x_1
\quad \text{其中} \quad (x_0, x_1) = \text{chunk}(input, 2, \text{dim})
$$

`σ(·)` 为 sigmoid;`⊙` 为逐元素乘法。

> **注**:历史上 Swish 有可调参数 β(`Swish_β(x) = x · sigmoid(β·x)`),但 SwiGLU 在文献与主流实现(Llama / PaLM / torch_npu / aclnnSwiGlu)中**统一固定 β = 1**(等价 SiLU)。本 spec 不暴露 β,与 reference 实现严格对齐。

## 3. 接口规范

### 算子原型

```python
cann_bench.swi_glu(Tensor input, int dim=-1) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入张量,dim 维 size 必须是偶数 |
| dim | int64 | -1 | 拆分维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| output | 与 input 同 rank,dim 维 size 折半 | 与 input 相同 | SwiGLU 激活输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16  | float16  |
| float32  | float32  |
| bfloat16 | bfloat16 |

### 规则与约束

- input 在 dim 维上 size 必须是偶数(否则无法等分 x0 / x1)
- `dim` 支持负数索引(如 -1 表示最后一维)
- output dtype 与 input 一致
- FP16 / BF16 输入内部计算时升精度到 FP32,再 cast 回原 dtype(与 ACLNN 一致)

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(input)` | 2 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| 每个维度大小 `dim_i` | ≥ 2 (split 维需偶) | cases.csv 实测最大 1,000,003 |
| 张量总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 134M (8192×16384) |
| `dim` | -rank(input) ~ rank(input)-1 | cases.csv 实测 dim=-1 / 0 / 1 / 2 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**:

1. 平均相对误差(MERE):采样点中相对误差平均值
2. 最大相对误差(MARE):采样点中相对误差最大值

**通过标准**:

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当 MERE < Threshold,MARE < 10 × Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch
import torch.nn.functional as F


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """标准 SwiGLU 激活的 Torch Golden 参考实现 (P2 op).

    公式: output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1
    其中 x0, x1 = input.chunk(2, dim=dim).
    """
    out_dtype = input.dtype
    x = input.to(torch.float)
    x0, x1 = x.chunk(2, dim=dim)
    output = F.silu(x0) * x1
    return output.to(out_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 2048, dtype=torch.float32, device="npu")
y = cann_bench.swi_glu(x, dim=-1)         # 输出 shape [1024, 1024]

x = torch.randn(8192, 16384, dtype=torch.float16, device="npu")
y = cann_bench.swi_glu(x, dim=0)          # 输出 shape [4096, 16384]
```
