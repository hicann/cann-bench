# Sigmoid PyPTO Selected-Case API 描述

## 1. 任务范围

本任务是 PyPTO 专用的 Sigmoid selected-case benchmark，路径为 `bench_lab/pypto_cann_bench/sigmoid`。它只要求覆盖当前目录 `cases.yaml` / `cases.csv` 中列出的测试集。

覆盖范围：

- `case_id`: 8, 15
- 输入 dtype: float32
- 输入 rank: 2D
- 输出 shape: 与输入 `x` 相同
- 输出 dtype: float32

## 2. 算子定义

接口：

```python
cann_bench.sigmoid(Tensor x) -> Tensor y
```

数学语义：

```text
y = 1 / (1 + e^(-x))
```

参数说明：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | float32 2D 输入张量 |

## 3. Selected Cases

| case_id | shape | dtype | attrs | value_range |
|---------|-------|-------|-------|-------------|
| 8 | [1537, 769] | float32 | `{}` | [-5, 10] |
| 15 | [512, 2049] | float32 | `{}` | [-0.5, 0.5] |

## 4. 精度要求

采用当前 cann-bench / kernel_eval 对 float32 selected cases 的默认精度判定。实现应按 `golden.py` 的计算语义返回逐元素 Sigmoid 结果。

## 5. Golden 代码

```python
import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)
```
