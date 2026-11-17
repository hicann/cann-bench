# ApplyRotaryPosEmb

## 功能描述

### 算子功能
对 `query` 和 `key` 两路张量执行 rotary position embedding 变换，并返回与输入形状一致的两路结果。标准化绑定采用完整语义接口，允许显式传入 `layout` 与 `rotary_mode`；其中 `aclnnApplyRotaryPosEmb` 可视为 `rotary_mode="half"` 的兼容包装。

### 计算公式
当 `rotary_mode="half"` 时，将最后一维按前后两半旋转：

$$
rotate(x) = \mathrm{concat}(-x[..., d/2:], x[..., :d/2], \mathrm{dim}=-1)
$$

$$
q\_embed = query * cos + rotate(query) * sin
$$

$$
k\_embed = key * cos + rotate(key) * sin
$$

当 `rotary_mode="interleave"` 或 `rotary_mode="quarter"` 时，按原始 README / `op_graph` 定义使用不同的分块旋转规则；在 `ascend910b` 目标芯片上，benchmark case 仅冻结到 `half` 模式。

## 接口定义

### Python 接口
```python
model = ModelNew(layout=1, rotary_mode="half")
query_out, key_out = model(query, key, cos, sin)
```

`layout` 为布局枚举，`rotary_mode` 为旋转模式字符串。`forward` 接收四个张量输入并返回二元组 `(query_out, key_out)`。

### 输入变量
- `query`: 输入张量，`float16`、`float32` 或 `bfloat16`，在 `layout=1` 时为 4 维，在 `layout=4` 时为 3 维。
- `key`: 输入张量，dtype 与 `query` 相同，除 head 数之外与 `query` 保持同构。
- `cos`: 位置编码余弦张量，dtype 与 `query` 相同，head 维必须为 `1`。
- `sin`: 位置编码正弦张量，dtype 与 `cos` 相同，shape 与 `cos` 一致。

### 属性
- `layout`: `int64`，支持 `1-BSND`、`2-SBND`、`3-BNSD`、`4-TND`；`ascend910b` 实测冻结为 `1` 或 `4`。
- `rotary_mode`: `string`，支持 `"half"`、`"interleave"`、`"quarter"`；`ascend910b` case 固定使用 `"half"`。

## 使用案例

```python
import torch

query = torch.randn(1, 8, 2, 64, dtype=torch.float16)
key = torch.randn(1, 8, 3, 64, dtype=torch.float16)
cos = torch.randn(1, 8, 1, 64, dtype=torch.float16)
sin = torch.randn(1, 8, 1, 64, dtype=torch.float16)

model = ModelNew(layout=1, rotary_mode="half")
query_out, key_out = model(query, key, cos, sin)
```

## 约束与限制

- `query`、`key`、`cos`、`sin` 的 dtype 必须一致。
- `cos` 与 `sin` 的 shape 必须一致，且 head 维必须为 `1`。
- `layout=1` 时推荐使用 `BSND`；`layout=4` 时输入为 `TND` 3 维张量。
- `ascend910b` 上 `query/key/cos/sin` 的最后一维需为 `64` 或 `128`，且 `rotary_mode` 仅冻结为 `"half"`。
- `rotary_mode="half"` / `"interleave"` 要求最后一维可被 `2` 整除；`"quarter"` 要求最后一维可被 `4` 整除。
