# Add 算子 API 描述（评测流水线 fixture）

> **注意**：本文件是评测流水线的 fixture，非生产评测目标。仅用于验证 benchmark pipeline 的端到端流程是否正常工作。

## 1. 算子简介

Add 是逐元素加法运算，对两个相同 shape 的输入张量逐位相加。

**算子特征**：
- 难度等级：L1（Elementwise）
- 双输入单输出，逐元素运算，输出 shape 与输入完全一致

## 2. 算子定义

### 数学公式

$$
z = x + y
$$

### 接口规范

```python
cann_bench.add(Tensor x, Tensor y) -> Tensor z
```

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |
| int32 | int32 |

### 规则与约束

- 两输入 shape 必须完全一致，无 broadcasting
- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- 无额外属性参数