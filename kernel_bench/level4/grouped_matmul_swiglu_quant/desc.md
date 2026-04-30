# GroupedMatmulSwigluQuant 算子 API 描述

## 1. 算子简介

分组矩阵乘法（GroupedMatmul / GMM）、反量化、SwiGLU 激活与再量化的融合算子，将四个步骤合并为一次 Kernel 调用以减少中间数据搬运。语义对齐 `torch_npu.npu_grouped_matmul_swiglu_quant_v2` 的默认配置。

**主要应用场景**：
- 大语言模型中 MoE（Mixture of Experts）结构的 FFN 层前半段（Gate+Up 投影 + SwiGLU + 量化）
- 量化推理流水线中 GMM → 激活 → 再量化的整体融合
- 对 token 分组路由后，每组 token 使用独立专家权重的高性能推理

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入、双输出；涉及 GMM、per-token / per-channel 反量化、SwiGLU、再量化
- 激活与权重均为 int8（x 已完成 per-token 量化），输出为 int8 + float32 scale

## 2. 算子定义

### 数学公式

$$
\begin{aligned}
&\text{对每个专家 } g \in [0, E),\ \text{根据 } group\_list\ (\text{cumsum}) \text{ 取属于该组的 token 行 } rows_g: \\
&mm_g = x[rows_g] \cdot weight[g] \\
&deq_g = mm_g \odot x\_scale[rows_g] \odot weight\_scale[g] \\
&\text{合并所有组得到 } deq \in \mathbb{R}^{M \times N} \\
&left, right = \text{split}(deq,\ \text{last\_dim}/2) \\
&act = \text{SiLU}(left) \odot right \\
&y,\ y\_scale = \text{PerTokenQuant}(act)
\end{aligned}
$$

### 步骤拆解

1. **分组矩阵乘法（GMM）**：`group_list` 采用 cumsum 语义，将 `x` 的 `M` 行划分到 `E` 个专家；第 `g` 组的 token 与 `weight[g]` 做矩阵乘。
2. **反量化（Dequant）**：左 per-token + 右 per-channel，`deq[i, j] = mm[i, j] * x_scale[i] * weight_scale[g(i), j]`。
3. **SwiGLU**：沿最后一维对半拆分为 `left, right`，计算 `SiLU(left) * right`，输出宽度减半为 `N/2`。
4. **再量化（Quant）**：per-token，`y_scale[i] = max_j|act[i, j]| / 127`，`y[i, j] = clamp(round(act[i, j] / y_scale[i]), -128, 127)`。

## 3. 接口规范

### 算子原型

```python
cann_bench.grouped_matmul_swiglu_quant(
    Tensor x,
    Tensor weight,
    Tensor weight_scale,
    Tensor x_scale,
    Tensor group_list,
) -> (Tensor y, Tensor y_scale)
```

### 输入参数

| 参数 | 类型 | Shape | 描述 |
|------|------|-------|------|
| x | Tensor (int8) | `[M, K]` | 激活矩阵（GMM 左矩阵，已完成 per-token 量化），允许非连续 |
| weight | Tensor (int8) | `[E, K, N]` | 专家权重（GMM 右矩阵）|
| weight_scale | Tensor (float32) | `[E, N]` | 权重 per-channel 反量化因子 |
| x_scale | Tensor (float32) | `[M]` | 激活 per-token 反量化因子 |
| group_list | Tensor (int32) | `[E]` | 每个专家的 token 累计和（cumsum） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | `[M, N/2]` | int8 | SwiGLU 后的 per-token int8 量化结果 |
| y_scale | `[M]` | float32 | 输出 per-token 反量化因子 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| x: int8；weight: int8；*_scale: float32；group_list: int32 | y: int8；y_scale: float32 |

### 规则与约束

- `x` 的 `K` 维必须与 `weight` 的 `K` 维一致。
- `weight` 的最后一维 `N` 必须为偶数，以便 SwiGLU 对半拆分。
- `group_list` 为 cumsum 序列，长度为 `E`，最终累计值不得超过 `M`。
- 输出 `y` 会被截断到 `[-128, 127]`。

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
from typing import Tuple


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    E, _, N = weight.shape
    N_out = N // 2
    ends = group_list.to(torch.int64).tolist()
    starts = [0] + ends[:-1]

    dequant = torch.empty((M, N), dtype=torch.float32, device=x.device)
    x_scale_f = x_scale.float()
    for g in range(E):
        s, e = starts[g], ends[g]
        if s == e:
            continue
        mm = torch.matmul(x[s:e].float(), weight[g].float())
        xs = x_scale_f[s:e].unsqueeze(1)
        ws = weight_scale[g].float().unsqueeze(0)
        dequant[s:e] = mm * xs * ws

    left, right = dequant[..., :N_out], dequant[..., N_out:]
    act = torch.nn.functional.silu(left) * right

    eps = torch.finfo(torch.float32).tiny
    y_scale = act.abs().amax(dim=-1).clamp_min(eps) / 127.0
    y = torch.clamp(torch.round(act / y_scale.unsqueeze(1)), -128, 127).to(torch.int8)
    return y, y_scale.to(torch.float32)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

M, K, N, E = 64, 256, 512, 4

x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="npu")
weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8, device="npu")
weight_scale = torch.rand(E, N, dtype=torch.float32, device="npu") * 0.01
x_scale = torch.rand(M, dtype=torch.float32, device="npu") * 0.01
# cumsum 语义：四组累计 16/32/48/64
group_list = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device="npu")

y, y_scale = cann_bench.grouped_matmul_swiglu_quant(
    x, weight, weight_scale, x_scale, group_list,
)
```

### 参考文档

- `torch_npu.npu_grouped_matmul_swiglu_quant_v2`：<https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul_swiglu_quant_v2.md>
