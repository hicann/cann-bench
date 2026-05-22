# Scatter 算子 API 描述

## 1. 算子简介

将 updates 按索引 indices 更新到 data 中。

**主要应用场景**：
- 嵌入表更新与稀疏梯度回传
- One-hot 编码生成
- 图神经网络中的消息聚合（scatter_add）
- 稀疏张量的构造与更新

**算子特征**：
- 难度等级：L2（ScatterUpdate）
- 三输入单输出，按指定维度将 updates 的值写入到 data 对应索引位置

## 2. 算子定义

### 数学公式

对于 3D 张量，当 dim=0 时：

$$
y[\text{index}[i][j][k]][j][k] = \text{src}[i][j][k]
$$

更一般地，对于任意维度 dim：

$$
y[\text{index}_0][\text{index}_1] \cdots [\text{index}_{\text{dim}}] \cdots [\text{index}_{n-1}] = \text{updates}[i_0][i_1] \cdots [i_{n-1}]
$$

其中 $\text{index}_d = \text{indices}[i_0][i_1] \cdots [i_{n-1}]$ 当 $d = \text{dim}$，否则 $\text{index}_d = i_d$。

当指定 reduce 参数时：
- **add**：$y[\ldots] = y[\ldots] + \text{updates}[\ldots]$
- **multiply**：$y[\ldots] = y[\ldots] \times \text{updates}[\ldots]$
- **amax**：$y[\ldots] = \max(y[\ldots], \text{updates}[\ldots])$
- **amin**：$y[\ldots] = \min(y[\ldots], \text{updates}[\ldots])$

## 3. 接口规范

### 算子原型

```python
cann_bench.scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| data | Tensor | 必选 | 输入数据张量 |
| dim | int | 必选 | 沿哪个维度 scatter |
| indices | Tensor | 必选 | 索引张量，值必须在 [0, data.size(dim)) 范围内 |
| updates | Tensor | 必选 | 更新值张量，与 data 维度数相同 |
| reduce | str | None | 聚合方式，可选值：None(update), 'add', 'multiply', 'amin', 'amax' |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与 data 相同 | 与 data 相同 | 输出张量，scatter 结果，与 data 形状相同 |

### 数据类型

| data dtype | indices dtype | updates dtype | 输出 dtype |
|-----------|--------------|--------------|-----------|
| float16 | int32 / int64 | float16 | float16 |
| float32 | int32 / int64 | float32 | float32 |
| bfloat16 | int32 / int64 | bfloat16 | bfloat16 |
| int32 | int32 / int64 | int32 | int32 |
| int64 | int32 / int64 | int64 | int64 |

### 规则与约束

- data、indices、updates 的维度数必须相同
- indices 的每个维度大小不能超过对应 data 或 updates 的维度大小
- indices 中的值必须在 [0, data.size(dim)) 范围内
- updates 和 data 的 dtype 必须一致
- indices 的 dtype 必须为 int32 或 int64
- reduce 为 None 时执行直接覆盖更新，为 'add' 时执行累加，为 'multiply' 时执行累乘，为 'amax'/'amin' 时取最大/最小值
- 输出 shape 与 data shape 完全一致
- 当 indices 中存在重复索引：
  - `reduce=None`（覆盖模式）：写入顺序与具体实现相关，PyTorch/NPU 均不保证确定性，最终留下来的值可能是任一个对应的 `updates` 元素
  - `reduce='add'/'multiply'/'amax'/'amin'`：聚合结果与顺序无关，且语义等价于 `scatter_reduce_(include_self=True)`

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `data` 维度数 | 1 ~ 8 | cases.csv 实测 1 ~ 5；`data`、`indices`、`updates` 维度数必须相同 |
| `data` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 2 ~ 1048583（一维大张量场景） |
| `indices` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 2 ~ 8193；每维 ≤ 对应 `data` 维度大小 |
| `updates` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 2 ~ 8193；shape 须与 `indices` 一致 |
| `indices` 值 | `[0, data.size(dim))` | cases.csv 实测覆盖完整索引范围 |
| `dim` | 0 ~ 7 | cases.csv 实测 0 / 1；支持负数索引，等价范围为 `[-rank, rank-1]` |
| `reduce` | `None` / `add` / `multiply` / `amin` / `amax` | cases.csv 实测全部 5 种取值 |

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
Scatter算子Torch Golden参考实现

将updates按索引indices更新到data中
公式: y[index[i][j][k]] = src[i][j][k] (if dim == 0)
"""
def scatter(
    data: torch.Tensor, dim: int, indices: torch.Tensor, updates: torch.Tensor, reduce: str = None
) -> torch.Tensor:
    """
    将updates按索引indices更新到data中

    公式: y[index[i][j][k]] = src[i][j][k] (if dim == 0)

    Args:
        data: 输入数据张量
        dim: 沿哪个维度scatter
        indices: 索引张量
        updates: 更新值张量
        reduce: 聚合方式 (None/update, add, multiply, amin, amax)

    Returns:
        输出张量，scatter结果
    """

    y = data.clone()
    idx = indices.long()
    if reduce is None or reduce == 'update':
        y.scatter_(dim, idx, updates)
    elif reduce == 'add':
        y.scatter_add_(dim, idx, updates)
    elif reduce == 'multiply':
        y.scatter_reduce_(dim, idx, updates, reduce="prod", include_self=True)
    elif reduce == 'amin':
        y.scatter_reduce_(dim, idx, updates, reduce="amin", include_self=True)
    elif reduce == 'amax':
        y.scatter_reduce_(dim, idx, updates, reduce="amax", include_self=True)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

data = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
indices = torch.randint(0, 1024, (1024, 512), dtype=torch.int32, device="npu")
updates = torch.randn(1024, 512, dtype=torch.float16, device="npu")
y = cann_bench.scatter(data, 1, indices, updates)  # dim=1, 直接更新

# reduce=add 模式
data = torch.randn(2048, 512, dtype=torch.float32, device="npu")
indices = torch.randint(0, 2048, (1024, 512), dtype=torch.int32, device="npu")
updates = torch.randn(1024, 512, dtype=torch.float32, device="npu")
y = cann_bench.scatter(data, 0, indices, updates, reduce="add")
```
