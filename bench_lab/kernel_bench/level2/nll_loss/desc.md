# NLLLoss 算子 API 描述

## 1. 算子简介

计算负对数似然损失（Negative Log Likelihood Loss），用于分类任务。与 CrossEntropyLoss 不同，NLLLoss 的输入是已经经过 `log_softmax` 的 log-probabilities。

**主要应用场景**：
- 多分类任务的损失函数（与 CrossEntropyLoss 配套使用）
- 序列建模中的 token-level 损失
- 图像分割等像素级分类任务

**算子特征**：
- 难度等级：L2（Loss）
- 三输入（input, target, weight）、双输出（out, total_weight）
- 支持 `reduction` 和 `ignore_index` 属性
- 输入为 log-probabilities，target 为硬标签索引
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

### 数学公式

设输入 `x` 为 log-probabilities 张量，`y` 为 target 类别索引，`weight` 为各类别权重，`N` 为 `target` 元素总数。

#### 1. 有效类别权重

对每个类别 `c`，先判断它是否被 `ignore_index` 忽略，再乘上 `weight[c]`：

```
valid[c] = 1  如果 c != ignore_index
valid[c] = 0  如果 c == ignore_index
w[c] = weight[c] * valid[c]
```

#### 2. 逐元素损失（reduction='none'）

对 `target` 中第 `n` 个位置，取出它的类别索引 `y[n]` 和对应权重 `w[y[n]]`：

```
loss[n] = - w[y[n]] * x[n, y[n]]
```

- `x[n, y[n]]` 表示第 `n` 个位置在类别 `y[n]` 上的 log-probability。
- 若 `y[n] == ignore_index`，则 `w[y[n]] = 0`，因此 `loss[n] = 0`。

此时输出 `out` 就是所有 `loss[n]` 按 `target` 形状排列的张量：

```
out = [loss[0], loss[1], ..., loss[N-1]]
```

`out` 的 shape 与 `target` 相同。

#### 3. 带 reduction 的损失聚合（reduction='mean' 或 'sum'）

先计算有效权重之和：

```
total_weight = sum( w[y[n]] for n in range(N) )
```

再按 reduction 模式输出标量：

| reduction | 输出 out |
|-----------|---------|
| 'none' | shape 同 `target`，每个元素为 `loss[n]` |
| 'mean' | `sum(loss[n]) / total_weight` |
| 'sum'  | `sum(loss[n])` |

`total_weight` 仅在 `reduction != 'none'` 时有效。

### 输入输出关系

- `input` shape: `(C,)` 或 `(N, C)` 或 `(N, C, d1, d2, ...)`
- `target` shape: `()` 或 `(N,)` 或 `(N, d1, d2, ...)`
- `weight` shape: `(C,)`
- `out` shape:
  - `reduction='none'` 时 shape 同 `target`
  - 否则为标量 `(1,)` 或 `()`
- `total_weight` shape:
  - `reduction='none'` 时无意义（可不对比）
  - 否则为标量

## 3. 接口规范

### 算子原型

```python
cann_bench.nll_loss(Tensor input, Tensor target, Tensor weight, str reduction="mean", int ignore_index=-100) -> (Tensor out, Tensor total_weight)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入 log-probabilities 张量 |
| target | Tensor | 必选 | 目标类别索引（hard labels） |
| weight | Tensor | 必选 | 每个类别的缩放权重 |
| reduction | string | "mean" | 损失聚合方式 ('none' \| 'mean' \| 'sum') |
| ignore_index | int | -100 | 忽略的目标标签索引 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| out | reduction='none' 时同 target，否则为标量 | 与 input 相同 | 损失值 |
| total_weight | 标量 | 与 input 相同 | 有效权重之和 |

### 数据类型

| input dtype | target dtype | weight dtype | 输出 dtype |
|---------|-------------|-------------|-----------|
| float32 | int64 / int32 | float32 | float32 |
| float16 | int64 / int32 | float16 | float16 |
| bfloat16 | int64 / int32 | bfloat16 | bfloat16 |

### 规则与约束

- `input` 应为 log-probabilities（经 log_softmax 后的值），通常为负数或 0
- `target` 元素取值范围为 `[0, C)` 或等于 `ignore_index`
- `weight` shape 必须为 `(C,)`，且 dtype 与 `input` 一致
- `reduction` 仅支持 `"none"`、`"mean"`、`"sum"`
- `ignore_index` 指定的 target 值不参与损失计算

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch size，input 第 0 维） | 1 ~ 2097152 | cases.csv 实测 1 ~ 1,000,003 |
| `C`（类别数，input 第 1 维） | 2 ~ 16384 | cases.csv 实测 2 ~ 16384 |
| 额外空间维度 `d_i`（input 第 ≥2 维） | 1 ~ 1024 | cases.csv 实测 3 ~ 1024 |
| `rank(input)`（input 维度数） | 2 ~ 5 | cases.csv 实测 2 ~ 5 维 |
| `rank(target)` | rank(input)-1 | 缺 C 维 |
| `reduction` | "none" / "mean" / "sum" | cases.csv 三种均覆盖 |
| `ignore_index` | int64 任意值 | cases.csv 实测 -100 / -1 / 0 / 10 / 50 / 100 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

```
MERE = avg(abs(actual - golden) / (abs(golden) + 1e-7))
```

2. 最大相对误差（MARE）：采样点中相对误差最大值

```
MARE = max(abs(actual - golden) / (abs(golden) + 1e-7))
```

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python
import torch
import torch.nn.functional as F

"""
NLLLoss 算子 Torch Golden 参考实现

计算负对数似然损失

公式:
    l_n = -w_{y_n} * x_{n, y_n}
    当 reduction='mean': out = sum(l_n) / sum(w_{y_n})
    当 reduction='sum':  out = sum(l_n)
    当 reduction='none': out = {l_1, ..., l_N}

参考 PyTorch API: torch.nn.functional.nll_loss
    https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html

Parameters:
    - input: (N, C) 或 (N, C, d1, ...) - log-probabilities 张量
    - target: (N,) 或 (N, d1, ...) - 类别索引
    - weight: (C,) - 各类别权重
    - ignore_index: int, 默认 -100 - 忽略的类别索引
    - reduction: 'none' | 'mean' | 'sum', 默认 'mean' - 损失聚合方式
"""


def nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str = 'mean',
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    计算负对数似然损失

    Args:
        input: 输入 log-probabilities 张量，shape (N, C) 或 (N, C, d1, d2, ...)
               C 为类别数（channel_first 约定）
        target: 目标类别索引，shape (N,) 或 (N, d1, d2, ...)
        weight: 类别权重，shape (C,)
        reduction: 损失聚合方式
                   'none': 返回每个样本的损失
                   'mean': 返回加权平均损失
                   'sum': 返回加权总损失
        ignore_index: 忽略的类别索引

    Returns:
        损失值：如果 reduction='none'，返回 shape 同 target 的张量
               否则返回标量张量
    """
    return F.nll_loss(
        input=input,
        target=target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
    )
```

## 6. NPU 调用方式

基于 `pta.py` 中的调用模板，NPU 侧使用 ATen 接口：

```python
import torch
import torch_npu

input_npu = input_tensor.npu()
target_npu = target_tensor.to(torch.int64).npu()
weight_npu = weight_tensor.npu()

reduction_map = {"none": 0, "mean": 1, "sum": 2}
reduction_int = reduction_map[reduction]

out_npu, total_weight_npu = torch.ops.aten.nll_loss_forward(
    input_npu, target_npu, weight_npu, reduction_int, ignore_index
)

out = out_npu.cpu()
total_weight = total_weight_npu.cpu()
```

注意：
- `target` 需要为 `int64` 类型
- `reduction` 需要转换为整数（0/1/2）
- `torch.ops.aten.nll_loss_forward` 返回 `(out, total_weight)` 元组

## 7. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 2048, dtype=torch.float32, device="npu").log_softmax(dim=1)
target = torch.randint(0, 2048, (1024,), dtype=torch.int64, device="npu")
weight = torch.ones(2048, dtype=torch.float32, device="npu")

loss = cann_bench.nll_loss(x, target, weight, reduction="mean", ignore_index=-100)
loss = cann_bench.nll_loss(x, target, weight, reduction="sum", ignore_index=-100)
loss = cann_bench.nll_loss(x, target, weight, reduction="none", ignore_index=-100)
```

### 与 CrossEntropyLoss 的关系

```
CrossEntropyLoss(input, target) = NLLLoss(log_softmax(input, dim=1), target)
```

因此 NLLLoss 的输入应当是 log-probabilities，而不是原始 logits。
