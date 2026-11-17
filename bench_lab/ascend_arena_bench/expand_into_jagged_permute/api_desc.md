# ExpandIntoJaggedPermute

## 功能描述

### 算子功能
将表级别的置换索引扩展成批次级别的线性索引输出。对每个 `permute[i]`，算子读取 `input_offsets[permute[i]]` 作为起点，再按照 `output_offsets[i:i+2]` 给出的区间长度生成连续索引并写入输出。

### 计算公式
设 `len_i = output_offsets[i + 1] - output_offsets[i]`，则：

$$
output\_permute[output\_offsets[i]:output\_offsets[i+1]]=
\mathrm{arange}(input\_offsets[permute[i]], input\_offsets[permute[i]] + len_i)
$$

## 接口定义

### Python 接口
```python
import kernel_gen_ops

output = kernel_gen_ops.expand_into_jagged_permute(
    permute: torch.Tensor,
    input_offsets: torch.Tensor,
    output_offsets: torch.Tensor,
    output_size: int,
) -> torch.Tensor
```

`permute`、`input_offsets`、`output_offsets` 都是 1D `torch.int32` Tensor，`output_size` 为输出长度，返回值是 shape 为 `[output_size]` 的 `torch.int32` Tensor。

### 输入变量
- `permute`：1D `INT32` Tensor，长度记为 `input_len`，其值应构成 `[0, input_len - 1]` 范围内的置换。
- `input_offsets`：1D `INT32` Tensor，shape 为 `[input_len + 1]`，严格单调递增，最后一个元素等于 `output_size`。
- `output_offsets`：1D `INT32` Tensor，shape 为 `[input_len + 1]`，严格单调递增，最后一个元素等于 `output_size`。

### 属性
- `output_size`：显式整型属性，表示输出 `output_permute` 的长度，标准化用例中以 `value` 固定写入。

## 使用案例

```python
import torch
import kernel_gen_ops

permute = torch.tensor([1, 0, 2], dtype=torch.int32).npu()
input_offsets = torch.tensor([0, 3, 5, 8], dtype=torch.int32).npu()
output_offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()

result = kernel_gen_ops.expand_into_jagged_permute(
    permute, input_offsets, output_offsets, output_size=6
)
```

## 约束与限制

- 输入与输出格式固定为 `ND`。
- 3 个输入 Tensor 和输出 Tensor 的数据类型需要一致，当前冻结为 `INT32`。
- `permute`、`input_offsets`、`output_offsets` 都要求是一维 Tensor。
- `input_offsets.shape[0] == output_offsets.shape[0] == permute.shape[0] + 1`。
- `input_offsets` 与 `output_offsets` 都应严格单调递增，且最后一个元素等于 `output_size`。
