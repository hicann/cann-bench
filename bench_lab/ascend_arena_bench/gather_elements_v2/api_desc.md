# GatherElementsV2

## 功能描述

### 算子功能
`GatherElementsV2` 沿指定维度 `dim`，根据 `index` 中给出的下标从输入张量 `x` 中逐元素收集数据，输出张量 `y` 的 shape 与 `index` 相同。

### 计算公式
给定输入张量 `x`、索引张量 `index` 和维度 `d`，输出满足：

$$
y_{i_0,\cdots,i_{n-1}} = x_{i_0,\cdots,i_{d-1},index_{i_0,\cdots,i_{n-1}},i_{d+1},\cdots,i_{n-1}}
$$

其中 `index` 只在 `dim` 指定的轴上提供被收集的位置。

## 接口定义

### Python 接口
```python
class ModelNew(nn.Module):
    def __init__(self, dim: int = 0): ...
    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor: ...
```

参数说明：
- `x`：被收集的输入张量。
- `index`：索引张量，输出 shape 与其一致。
- `dim`：收集维度，默认值为 `0`，允许负轴。

返回说明：
- 返回与 `index` shape 相同、dtype 与 `x` 相同的输出张量。

### 输入变量
- `x`：ND 张量，标准化 run 以 `float16`、`float32`、`int32` 为主要覆盖 dtype；源码证据还表明 910B/910_93 支持 `bfloat16`。
- `index`：ND 张量，case 中使用随机生成的整数索引；标准化 run 固定生成非越界索引。

### 属性
- `dim`：`int64`，显式固定在 case 的 `attrs.dim.value` 中；取值范围为 `[-x.dim(), x.dim()-1]`。

## 使用案例

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
index = torch.tensor([[0, 2],
                      [1, 0]])

model = ModelNew(dim=1)
y = model(x, index)
```

## 约束与限制

- `x`、`index` 的维度数必须相同，且 rank 不超过 8。
- `y` 的 shape 必须与 `index` 相同。
- 除 `dim` 轴外，`index` 各维 size 不能大于 `x` 对应维度。
- `index` 中的索引值必须落在 `[-x.shape[dim], x.shape[dim] - 1]` 范围内。
- 标准化 binding 绑定公开两段式 API `aclnnGather`，而不是 level0 内部符号 `l0op::GatherElementsV2`。
