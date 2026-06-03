# Exp PyPTO Selected-Case API 描述

## 1. 任务范围

本任务是 PyPTO 专用的 Exp selected-case benchmark，路径为 `bench_lab/pypto_cann_bench/exp`。它只要求覆盖当前目录 `cases.yaml` / `cases.csv` 中列出的测试集。

覆盖范围：

- `case_id`: 2, 8, 15
- 输入 dtype: float32
- 输入 rank: 2D
- 输出 shape: 与输入 `x` 相同
- 输出 dtype: float32

## 2. 算子定义

接口：

```python
cann_bench.exp(Tensor x, float base, float scale, float shift) -> Tensor y
```

数学语义：

```text
base <= 0: y = exp(scale * x + shift)
base > 0:  y = exp((scale * x + shift) * ln(base))
```

参数说明：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | float32 2D 输入张量 |
| base | float | -1.0 | 指数底数；`base <= 0` 表示自然底数 `e` |
| scale | float | 1.0 | 输入缩放因子 |
| shift | float | 0.0 | 输入偏移量 |

## 3. Selected Cases

| case_id | shape | dtype | attrs | value_range |
|---------|-------|-------|-------|-------------|
| 2 | [2048, 2048] | float32 | `{base: -1.0, scale: 1.5, shift: 0.0}` | [-2, 2] |
| 8 | [1537, 769] | float32 | `{base: 10.0, scale: 1.0, shift: 0.0}` | [-5, 10] |
| 15 | [512, 2049] | float32 | `{base: -1.0, scale: 1.0, shift: 0.5}` | [-0.5, 0.5] |

## 4. 精度要求

采用当前 cann-bench / kernel_eval 对 float32 selected cases 的默认精度判定。实现应按 `golden.py` 的计算语义返回逐元素指数结果。

## 5. Golden 代码

```python
import torch

def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    temp = scale * x + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base, dtype=x.dtype, device=x.device))
    return torch.exp(temp)
```
