# AdaptiveAvgPool3d

## 功能描述

### 算子功能
对 4D/5D 输入张量执行 3D 自适应平均池化。用户只需给出输出空间尺寸 `output_size=[Dout, Hout, Wout]`，算子会按输入空间尺寸自动推导每个输出位置对应的平均池化窗口。

### 计算公式
对输出位置 `(od, oh, ow)`，其值等于对应输入窗口元素的平均值：

```math
y[..., od, oh, ow] =
\frac{1}{|\Omega(od, oh, ow)|}
\sum_{(d,h,w)\in\Omega(od, oh, ow)} x[..., d, h, w]
```

其中 `\Omega(od, oh, ow)` 由输入空间尺寸与目标 `output_size` 自适应划分得到。

## 接口定义

### Python 接口
```python
AdaptiveAvgPool3d(output_size: Sequence[int]) -> nn.Module
```

- `output_size`：长度固定为 3 的整数序列，表示输出的 `(Dout, Hout, Wout)`。
- 返回值：输出张量，dtype 与输入一致，空间维替换为 `output_size`。

### 输入变量
- `x`：输入张量，支持 `float16`、`float32`、`bfloat16`。输入维度为 4D 或 5D。
  - 5D 语义：`(N, C, Din, Hin, Win)`
  - 4D 语义：`(C, Din, Hin, Win)`

### 属性
- `output_size`：`int64[3]`，显式固定值属性，指定输出在 `D/H/W` 维度上的目标大小。

## 使用案例

```python
import torch
from binding.module import ModelNew

x = torch.randn(1, 2, 4, 5, 6, device="npu", dtype=torch.float16)
model = ModelNew([2, 3, 4]).to("npu")
y = model(x)
print(y.shape)  # torch.Size([1, 2, 2, 3, 4])
```

## 约束与限制

- `output_size` 长度必须等于 3。
- 输入张量维度必须为 4 或 5。
- 输出张量的 batch/channel 维与输入保持一致，只替换空间维。
- 文档证据表明公开接口支持 `NCDHW` 与 `ND`/默认格式路径；本标准化 run 的 benchmark case 统一采用 5D 输入以避免 4D 格式歧义。
