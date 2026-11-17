# GroupNormSiluQuant

## 功能描述

### 算子功能
`group_norm_silu_quant` 对输入 `self` 执行按组归一化，随后施加 SiLU 激活，并按 `quantScale` 做量化，返回量化结果 `out` 以及归一化阶段的统计量 `meanOut`、`rstdOut`。

### 计算公式
设输入在每个 group 内的均值为 `E[x]`，方差为 `Var[x]`，则：

$$
\text{groupNormOut} = \frac{x - E[x]}{\sqrt{Var[x] + eps}} \cdot \gamma + \beta
$$

$$
\text{siluOut} = \frac{\text{groupNormOut}}{1 + e^{-\text{groupNormOut}}}
$$

$$
out = \operatorname{round}\left(\frac{\text{siluOut}}{\text{quantScale}}\right)
$$

同时输出：

$$
meanOut = E[x], \quad rstdOut = \frac{1}{\sqrt{Var[x] + eps}}
$$

## 接口定义

### Python 接口

```python
model = ModelNew(group: int, eps: float, activate_silu: bool = True)
out, mean_out, rstd_out = model(
    self: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    quant_scale: torch.Tensor,
)
```

`ModelNew` 冻结到底层 `kernel_gen_ops.group_norm_silu_quant`，其 runtime 调用符号为 `aclnnGroupNormSiluQuant`。初始化参数只承载 attrs；前向阶段只传张量输入。

### 输入变量
- `self`: 主输入张量，dtype 支持 `float16`、`bfloat16`，格式为 `ND`，维度范围为 2 到 8，且第 0 维为 batch `N`、第 1 维为 channel `C`。
- `gamma`: 一维缩放张量，dtype 与 `self` 一致，shape 必须为 `[C]`。原始 ACL 接口允许为空，此标准化 binding 固定要求显式传入张量。
- `beta`: 一维偏移张量，dtype 与 `self` 一致，shape 必须为 `[C]`。原始 ACL 接口允许为空，此标准化 binding 固定要求显式传入张量。
- `quant_scale`: 一维量化尺度张量，dtype 为 `float32`，shape 必须为 `[1]` 或 `[C]`。

### 属性
- `group`: `int64`，表示将输入第 1 维 channel 划分为多少组，必须大于 0 且能整除 `C`。
- `eps`: `double`，归一化稳定项，必须大于 0。
- `activate_silu`: `bool`，当前仅支持 `true`。

## 使用案例

```python
import torch
from binding.module import ModelNew

model = ModelNew(group=2, eps=1e-5, activate_silu=True).to("npu")
self = torch.randn(1, 4, 2, 2, device="npu", dtype=torch.float16)
gamma = torch.ones(4, device="npu", dtype=torch.float16)
beta = torch.zeros(4, device="npu", dtype=torch.float16)
quant_scale = torch.ones(1, device="npu", dtype=torch.float32)

out, mean_out, rstd_out = model(self, gamma, beta, quant_scale)
```

## 约束与限制

- 输出 `out` 的 dtype 固定为 `int8`，shape 与 `self` 一致。
- 输出 `mean_out`、`rstd_out` 的 dtype 与 `self` 一致，shape 固定为 `[N, group]`。
- `quant_scale` 只支持 `float32`，不支持与 `self` 同 dtype 的量化尺度。
- 当前实现对 profile 可观测目标应为 `GroupNormSiluQuant`。
