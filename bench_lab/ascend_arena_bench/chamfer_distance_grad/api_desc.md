# ChamferDistanceGrad

## 功能描述

### 算子功能
`ChamferDistanceGrad` 计算 Chamfer Distance 正向结果对应的反向梯度。输入为两组二维点集、正向最近邻索引和 `dist1/dist2` 的上游梯度，输出为点集 `xyz1` 与 `xyz2` 的梯度。

### 计算公式
对每个 batch 内的点，算子按最近邻索引把 `grad_dist1` 和 `grad_dist2` 的贡献分别回传到 `xyz1`、`xyz2`。原始 README 与 kernel 测试中的 CPU 参考都可归纳为以下更新规则：

```math
\mathrm{grad\_xyz1}[b,n] \mathrel{+}= 2 \cdot \mathrm{grad\_dist1}[b,n] \cdot (xyz1[b,n] - xyz2[b, idx1[b,n]])
```

```math
\mathrm{grad\_xyz2}[b, idx1[b,n]] \mathrel{-}= 2 \cdot \mathrm{grad\_dist1}[b,n] \cdot (xyz1[b,n] - xyz2[b, idx1[b,n]])
```

`grad_dist2` 与 `idx2` 对 `xyz2 -> xyz1` 的反向传播对称执行。

## 接口定义

### Python 接口

```python
grad_xyz1, grad_xyz2 = kernel_gen_ops.chamfer_distance_grad(
    xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
)
```

- `xyz1`、`xyz2`：点集坐标张量。
- `idx1`、`idx2`：正向最小距离对应的索引张量。
- `grad_dist1`、`grad_dist2`：正向距离输出的上游梯度。
- 返回值：二元组 `(grad_xyz1, grad_xyz2)`。

### 输入变量

- `xyz1`：`float16` 或 `float32`，推荐 shape 为 `(B, N, 2)`，格式 `ND`。
- `xyz2`：`float16` 或 `float32`，文档给出 `(B, M, 2)`，但当前 README 约束、proto、UT 与标准化 case 统一按 `(B, N, 2)` 使用。
- `idx1`：`int32`，shape 为 `(B, N)`，表示 `xyz1` 对 `xyz2` 的最近邻索引。
- `idx2`：`int32`，shape 为 `(B, N)`，表示 `xyz2` 对 `xyz1` 的最近邻索引。
- `grad_dist1`：与 `xyz1` 同 dtype，shape 为 `(B, N)`。
- `grad_dist2`：与 `xyz2` 同 dtype，shape 为 `(B, N)`。

### 属性

- 无显式 attrs。

## 使用案例

```python
import torch
import kernel_gen_ops

xyz1 = torch.randn(2, 8, 2, device="npu", dtype=torch.float16)
xyz2 = torch.randn(2, 8, 2, device="npu", dtype=torch.float16)
idx1 = torch.randint(0, 8, (2, 8), device="npu", dtype=torch.int32)
idx2 = torch.randint(0, 8, (2, 8), device="npu", dtype=torch.int32)
grad_dist1 = torch.randn(2, 8, device="npu", dtype=torch.float16)
grad_dist2 = torch.randn(2, 8, device="npu", dtype=torch.float16)

grad_xyz1, grad_xyz2 = kernel_gen_ops.chamfer_distance_grad(
    xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
)
```

## 约束与限制

- 输入格式固定为 `ND`。
- `idx1`、`idx2` 必须为 `int32`。
- `xyz1`、`xyz2`、`grad_dist1`、`grad_dist2` 仅支持 `float16` 和 `float32`。
- 输出 dtype 分别跟随 `xyz1`、`xyz2`。
- 现有文档对 `xyz2` 第二维是否允许与 `xyz1` 不同存在歧义；在该歧义消除前，标准化 run 只保证等长点集 `(B, N, 2)` 的契约。
