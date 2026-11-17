# GroupQuant

## 功能描述

### 算子功能
对二维输入 `x` 按 `group_index` 指定的行分段使用逐列 `scale` 做分组量化，可选叠加标量 `offset`，输出量化后的整数张量。

### 计算公式
对第 `e` 个 group 覆盖的行片段，逐元素执行：

$$
y = round(x \times scale_e + offset)
$$

其中 `offset` 缺省时按 `0` 处理。

## 接口定义

### Python 接口
```python
ModelNew(dst_type: int = 2)
forward(
    x: torch.Tensor,
    scale: torch.Tensor,
    group_index: torch.Tensor,
    offset: torch.Tensor | None = None,
) -> torch.Tensor
```

`dst_type` 在当前标准化 run 中冻结为 `2`，即 `INT8` 输出路径。返回张量与 `x` 同 shape。

### 输入变量
- `x`：二维输入张量，dtype 支持 `float16`、`float32`、`bfloat16`，格式为 `ND`。
- `scale`：二维张量，shape 为 `[E, H]`，其中 `H == x.shape[1]`。
- `group_index`：一维张量，shape 为 `[E]`，dtype 支持 `int32`、`int64`；要求非递减，末元素等于 `x.shape[0]`。
- `offset`：可选标量张量，shape 仅支持 `[]` 或 `[1]`，dtype 必须与 `scale` 一致。

### 属性
- `dst_type`：量化输出类型。原始算子文档列出 `2(INT8)` 与 `29(INT4)`；当前标准化 binding 和 case 仅覆盖 `2(INT8)`。

## 使用案例

```python
import torch
from binding.module import ModelNew

model = ModelNew(dst_type=2)
x = torch.randn(4, 8, dtype=torch.float16)
scale = torch.rand(2, 8, dtype=torch.float16)
group_index = torch.tensor([2, 4], dtype=torch.int32)
offset = torch.tensor([0.5], dtype=torch.float16)
y = model(x, scale, group_index, offset)
```

## 约束与限制

- `x` 与 `scale` 必须是二维，`group_index` 必须是一维。
- `scale.shape[0] == group_index.shape[0]`，`scale.shape[1] == x.shape[1]`。
- `group_index` 应为非递减分组结束位置序列，末元素等于 `x.shape[0]`。
- `offset` 缺省时按 0 处理；给定时 dtype 需与 `scale` 一致。
- 当前 run 未覆盖 `INT4`/packed 输出路径，benchmark 与 correctness 仅验证 `INT8`。
