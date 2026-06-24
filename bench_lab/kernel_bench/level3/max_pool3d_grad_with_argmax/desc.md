# MaxPool3dGradWithArgmax 算子 API 描述

## 1. 算子简介

MaxPool3dGradWithArgmax 是 3D 最大池化前向算子的反向传播算子，用于将正向输出端的梯度回填到每个池化窗口最大值所在的输入坐标处；相同坐标处的梯度会累加。

**主要应用场景**：
- 三维卷积神经网络（3D CNN）中的池化层反向传播
- 视频理解、医学影像等 3D 数据模型的训练

**算子特征**：
- 难度等级：L3（Composite）
- 多输入单输出：3 个输入张量，1 个输出张量
- 支持 5D NCDHW 输入
- 依赖正向 maxpool 的 indices 输入
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

### 数学描述

对于每个输出位置 `(n, c, d_o, h_o, w_o)`，在输入上以 `(d_o, h_o, w_o)` 对应的池化窗口取最大值，正向同时记录窗口内最大值的局部索引 `idx`。反向传播时，将 `grad[n, c, d_o, h_o, w_o]` 加到输入梯度 `y` 的对应坐标上：

```
window = x[n, c, d_o*sD:d_o*sD+kD, h_o*sH:h_o*sH+kH, w_o*sW:w_o*sW+kW]   (考虑 padding/dilation)
y[n, c, flat_idx_to_dhw(idx)] += grad[n, c, d_o, h_o, w_o]
```

其中 `idx` 为窗口内最大值的展平索引（DHW_ONLY 编码，范围 `[0, kD*kH*kW-1]`）。

### 变量说明

| 变量 | 说明 |
|------|------|
| x | 正向输入（5D NCDHW） |
| grad | 正向输出的梯度（5D NCDHW，shape 与正向输出一致） |
| argmax | 正向输入中最大元素的索引（5D NCDHW，shape 与 grad 一致） |
| ksize | 池化窗口大小 `[kD, kH, kW]` |
| strides | 池化步长 `[sD, sH, sW]` |
| pads | 在 D、H、W 方向上的补零层数 `[pD, pH, pW]` |
| dilation | 窗口内元素步幅 `[dD, dH, dW]` |
| ceil_mode | 计算输出形状时是否向上取整 |
| y | 输入 x 的梯度（shape 与 x 一致） |

## 3. 接口规范

### 算子原型

```python
cann_bench.max_pool3d_grad_with_argmax(
    Tensor x,
    Tensor grad,
    Tensor argmax,
    int[] ksize,
    int[] strides,
    int[] pads,
    int[] dilation=[1, 1, 1],
    bool ceil_mode=False,
) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 正向输入，5D NCDHW |
| grad | Tensor | 必选 | 正向输出的梯度，shape 与正向输出一致 |
| argmax | Tensor | 必选 | 正向输入中最大元素的索引，shape 与 grad 一致 |
| ksize | int[] | 必选 | 池化窗口大小 `[kD, kH, kW]` |
| strides | int[] | 必选 | 池化步长 `[sD, sH, sW]` |
| pads | int[] | 必选 | 在 D、H、W 方向上的补零层数 `[pD, pH, pW]` |
| dilation | int[] | [1,1,1] | 窗口内元素步幅 |
| ceil_mode | bool | false | 是否向上取整计算输出形状 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与 x 相同 | 与 x 相同 | 输入 x 的梯度 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x、grad、y 的 shape、dtype、数据格式必须一致
- grad 与 argmax 的 shape 必须一致
- argmax 的数据类型支持 int32、int64
- ksize 长度等于 1 或 3；strides 长度等于 0、1 或 3；pads 长度等于 1 或 3；dilation 长度等于 1 或 3
- padding 值需满足 `0 <= pad <= ksize / 2`

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `N`（batch） | 1 ~ 8 | cases.csv 实测 1 ~ 4 |
| `C`（channel） | 1 ~ 128 | cases.csv 实测 3 ~ 64 |
| `D / H / W`（空间维度） | 4 ~ 32 | cases.csv 实测 4 ~ 32 |
| `ksize` | [1,1,1] ~ [3,3,3] | cases.csv 实测 1~3 |
| `strides` | [1,1,1] ~ [3,3,3] | cases.csv 实测 1~3 |
| `pads` | [0,0,0] ~ [1,1,1] | cases.csv 实测 0~1 |
| `dilation` | [1,1,1] | 当前仅支持 1，cases.csv 中全部取 1 |
| `ceil_mode` | {false, true} | cases.csv 实测两种取值 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：

   ```
   MERE = avg(abs(actual - golden) / (abs(golden) + 1e-7))
   ```

2. 最大相对误差（MARE）：

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


def get_input(x, grad, argmax, ksize, strides, pads, dilation=None, ceil_mode=False, **kwargs):
    """根据正向 maxpool 生成合法的 grad 与 argmax。"""
    if dilation is None:
        dilation = [1, 1, 1]
    with torch.no_grad():
        _, indices = F.max_pool3d(
            x.float(), ksize, stride=strides, padding=pads,
            dilation=dilation, ceil_mode=ceil_mode, return_indices=True
        )
    grad = torch.randn_like(indices, dtype=x.dtype)
    return x, grad, indices


def max_pool3d_grad_with_argmax(
    x: torch.Tensor,
    grad: torch.Tensor,
    argmax: torch.Tensor,
    ksize: List[int],
    strides: List[int],
    pads: List[int],
    dilation: List[int] = None,
    ceil_mode: bool = False,
) -> torch.Tensor:
    if dilation is None:
        dilation = [1, 1, 1]
    return torch.ops.aten.max_pool3d_with_indices_backward(
        grad, x, kernel_size=ksize, stride=strides, padding=pads,
        dilation=dilation, ceil_mode=ceil_mode, indices=argmax
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 8, 8, 8, 8, dtype=torch.float16, device="npu")
grad = torch.randn(2, 8, 4, 4, 4, dtype=torch.float16, device="npu")
argmax = torch.randint(0, 8, (2, 8, 4, 4, 4), dtype=torch.int32, device="npu")

y = cann_bench.max_pool3d_grad_with_argmax(
    x, grad, argmax, ksize=[2, 2, 2], strides=[2, 2, 2], pads=[0, 0, 0]
)
```

### 注意事项

- cases.yaml 中 `grad` 与 `argmax` 的 shape 需与给定参数下正向 maxpool 的输出 shape 一致。
- 由于框架会随机生成 `argmax`，golden.py 中通过 `get_input()` 钩子重新运行正向 maxpool 以生成合法的 `argmax` 索引，确保 golden 与 NPU 使用同一套有效索引。
- `get_input()` 使用 `F.max_pool3d(..., return_indices=True)` 生成索引，其 dtype 为 `int64`；因此 cases.csv 中 `argmax` 的 dtype 标注为 `int64`，与实际使用的索引类型一致。
