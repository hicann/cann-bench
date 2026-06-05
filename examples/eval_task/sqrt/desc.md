# Sqrt 算子 API 描述（评测流水线 fixture）

> **注意**：本文件是评测流水线的 fixture，非生产评测目标。仅用于验证 benchmark pipeline 的端到端流程是否正常工作。

## 1. 算子简介

Sqrt 是逐元素平方根运算，对输入张量的每个元素计算平方根值。

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致

## 2. 算子定义

### 数学公式

$$
y = \sqrt{x}
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x < 0 | y = NaN |

### 接口规范

```python
cann_bench.sqrt(Tensor x) -> Tensor y
```

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输入值应为非负数以保证数值稳定性
- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- 无额外属性参数