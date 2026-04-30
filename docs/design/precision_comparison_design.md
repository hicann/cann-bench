# 精度对比方案设计文档

## 1. 概述

### 1.1 背景

在算子代码生成评测中，需要对比 AI 生成的算子输出与 Golden 参考输出的精度差异。由于浮点运算的精度限制，简单的绝对差值或相对差值判断不足以覆盖所有场景，需要设计一套科学的精度对比方案。

### 1.2 设计目标

1. **标准化**：采用业界认可的生态算子精度标准（MERE/MARE）
2. **全面性**：覆盖正常值域、小值域、相消位置等特殊场景
3. **科学性**：基于 IEEE 754 精度位数理论，区分算子 bug 和 dtype 精度限制
4. **可配置**：支持算子自定义精度阈值

### 1.3 适用范围

- 计算 类算子的精度验证
- 支持 float16、float32、bfloat16 等浮点类型
- 支持单输出和多输出算子

---

## 2. 核心算法

### 2.1 误差指标

采用生态算子开源精度标准的 MERE/MARE 指标：

**平均相对误差（MERE）**：
$$
\text{MERE} = \text{avg}\left(\frac{|\text{actual} - \text{golden}|}{|\text{golden}| + \epsilon}\right)
$$

其中 $\epsilon = 10^{-7}$，防止 golden 为 0 时除零。

**最大相对误差（MARE）**：
$$
\text{MARE} = \max\left(\frac{|\text{actual} - \text{golden}|}{|\text{golden}| + \epsilon}\right)
$$

### 2.2 精度阈值表

| 数据类型 | 精度阈值 (Threshold) | MARE 阈值 (10×Threshold) |
|----------|---------------------|-------------------------|
| float16  | $2^{-10}$ ≈ 0.00098  | $2^{-6}$ ≈ 0.0156       |
| bfloat16 | $2^{-7}$ ≈ 0.00781   | $2^{-3}$ ≈ 0.125        |
| float32  | $2^{-13}$ ≈ 0.00012  | $2^{-9}$ ≈ 0.002        |

### 2.3 基础通过标准

**正常值域通过条件**：
$$
\text{MERE} < \text{threshold} \quad \land \quad \text{MARE} < 10 \times \text{threshold}
$$

---

## 3. 小值域处理

### 3.1 问题背景

当 golden 值非常小（接近 0）时，相对误差计算不稳定：
- golden = $10^{-6}$，output = $10^{-5}$
- 相对误差 = $|10^{-5} - 10^{-6}| / 10^{-6} = 9$
- 显然超过阈值，但这可能是 dtype 精度限制而非算子错误

### 3.2 小值域阈值表

| 数据类型 | 小值域阈值 (Small Value Threshold) | 小值域误差阈值 (Small Value Error) |
|----------|-----------------------------------|----------------------------------|
| float16  | $2^{-11}$ ≈ $4.88 \times 10^{-4}$ | $2^{-16}$ ≈ $1.53 \times 10^{-5}$ |
| bfloat16 | $2^{-8}$ ≈ $3.91 \times 10^{-3}$  | $2^{-16}$ ≈ $1.53 \times 10^{-5}$ |
| float32  | $2^{-14}$ ≈ $6.10 \times 10^{-5}$ | $2^{-30}$ ≈ $9.31 \times 10^{-10}$ |

### 3.3 错误计数定义

**ErrorCount**：统计小值域位置中绝对误差超过阈值的数量：
$$
\text{ErrorCount} = \sum \mathbb{I}\left(|\text{golden}| < \text{threshold} \land |\text{actual} - \text{golden}| > \text{error}\right)
$$

其中：
- $\mathbb{I}(\cdot)$ 是指示函数（条件成立为 1，否则为 0）
- threshold 为小值域阈值
- error 为小值域误差阈值

### 3.4 小值域通过标准

采用 **CPU 同精度对照** 方式，将 NPU 表现与 CPU 同精度表现对比：

$$
\frac{\text{ErrorCount}_{\text{npu}}}{\max(\text{ErrorCount}_{\text{cpu}}, 1)} \leq 2
$$

**判断逻辑**：
- NPU 和 CPU 都有相近的"错误"数量 → dtype 精度限制 → 通过
- NPU 有大量"错误"，CPU 几乎没有 → 算子精度问题 → 不通过

---

## 4. 相消位置处理

### 4.1 理论依据

基于 **IEEE 754 浮点标准** 和 **Kahan 灾难性相消理论**：

#### IEEE 754 精度位数

不同 dtype 的尾数位数决定了有效数字范围：

| dtype | 尾数位数 | 相对精度 | 有效数字 |
|-------|----------|----------|----------|
| FP32  | 23 位    | $2^{-23} \approx 10^{-7}$ | ~7 位 |
| FP16  | 10 位    | $2^{-10} \approx 10^{-3}$ | ~3 位 |
| BF16  | 7 位     | $2^{-7} \approx 10^{-2}$  | ~2 位 |

#### Kahan 灾难性相消

当两个接近的大数相减时，结果的有效位数急剧丢失。

**示例**：FP32 中两个 $10^4$ 量级的数相减得到 $10^{-3}$，但精度只够表示 7 位有效数字，结果相对于原操作数丢失精度，可能输出为 0。

### 4.2 相消检测条件

科学的相消检测应基于精度位数理论：

```python
# 检测条件
cancel_mask = (
    |output| < cancel_zero_threshold    # output 因相消接近零
    AND
    |golden| < cancel_boundary          # golden 在精度边界附近
    AND
    |golden| >= small_value_threshold   # 排除小值域（golden 不是极小值）
    AND
    is_valid                            # 是有效值（非 NaN/Inf）
)
```

### 4.3 相消阈值表

阈值基于 dtype 精度位数设计：

| dtype | cancel_boundary | cancel_zero_threshold | 说明 |
|-------|-----------------|----------------------|------|
| FP32  | $2^{-8}$ ≈ 0.004 | $2^{-8}$ ≈ 0.004 | 覆盖 $10^{-3}$ 范围相消 |
| FP16  | $2^{-5}$ ≈ 0.031 | $2^{-5}$ ≈ 0.031 | 覆盖 $10^{-2}$ 范围相消 |
| BF16  | $2^{-3}$ ≈ 0.125 | $2^{-3}$ ≈ 0.125 | 覆盖 $10^{-1}$ 范围相消 |

**阈值选择原则**：
- 当 golden < cancel_boundary 且 output ≈ 0 时，判定为潜在相消位置
- 阈值设置为比 dtype 精度稍大的值，确保覆盖常见相消场景
- 对于 FP32，当操作数规模 ~$10^4$ 时，结果 ~$10^{-3}$ 可能相消丢失

### 4.4 相消位置通过标准

同样采用 **CPU 同精度对照** 方式：

$$
\frac{\text{CancelErrorCount}_{\text{npu}}}{\max(\text{CancelErrorCount}_{\text{cpu}}, 1)} \leq 2
$$

其中：
- $\text{CancelErrorCount}$：相消位置中相对误差超过 mare_threshold 的计数

**判断逻辑**：
- NPU 和 CPU 都因 FP32 精度位数限制产生相同的"相消误差" → dtype 精度限制 → 通过
- NPU 产生误差，CPU 正确计算 → 算子 bug → 不通过

---

## 5. 完整通过标准

### 5.1 三维度判断

精度对比的最终通过条件为三个维度全部通过：

```
passed = normal_passed && small_value_passed && cancel_passed
```

| 维度 | 通过条件 |
|------|----------|
| 正常值域 | MERE < threshold 且 MARE < 10×threshold |
| 小值域 | ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2 |
| 相消位置 | CancelErrorCount_npu / max(CancelErrorCount_cpu, 1) ≤ 2 |

### 5.2 处理流程

```
┌─────────────────────────────────────────────────────────────┐
│                      精度对比流程                            │
├─────────────────────────────────────────────────────────────┤
│  1. 计算 FP64 Golden（高精度参考）                           │
│  2. 计算 CPU 同精度输出（用于对照）                          │
│  3. 获取 NPU/算子输出                                        │
│  4. 计算相对误差（MERE/MARE）                                │
│  5. 判断三个维度：                                           │
│     ├─ 正常值域：标准 MERE/MARE 判断                         │
│     ├─ 小值域：ErrorCount 比值判断                           │
│     │   条件：|golden| < small_value_threshold              │
│     └─ 相消位置：CancelErrorCount 比值判断                   │
│         条件：|output|≈0 AND |golden|在精度边界              │
│  6. 综合判断通过/失败                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 实现细节

### 6.1 文件结构

```
src/kernel_eval/
├── utils/
│   └── precision.py          # 精度对比核心实现
└── eval/
    ├── accuracy_eval.py      # 精度评测器
    └── evaluator.py          # 评测流程（计算CPU输出）
```

### 6.2 核心函数

**compare_tensors**（precision.py）：
- 输入：NPU output、FP64 golden、dtype、threshold、CPU output
- 输出：CompareResult（含三个维度的判断结果）

**evaluate**（accuracy_eval.py）：
- 调用 compare_tensors 进行精度对比
- 返回 AccuracyResult

**evaluate_case**（evaluator.py）：
- 计算 FP64 Golden
- 计算 CPU 同精度输出
- 执行 NPU 算子
- 调用精度对比

### 6.3 CPU 输出计算

```python
# 使用原始 dtype 的输入调用 golden_fn
cpu_inputs = [item.cpu() for item in input_tensors]
cpu_params = param_builder.build_call_params(golden_fn, case, cpu_inputs)
with torch.no_grad():
    cpu_out = golden_fn(**cpu_params)

# 传入精度对比函数
accuracy_result = evaluate(
    ai_output=npu_output,
    golden_output=fp64_golden,
    cpu_output=cpu_out,  # 关键：传入CPU输出
    ...
)
```

### 6.4 多输出处理

对于多输出算子（如 ApplyRotaryPosEmb 输出 query_out 和 key_out）：
- CPU 输出保留完整的 tuple/list 格式
- precision.py 的 `_normalize_outputs` 函数将多输出标准化为张量列表
- 逐个张量对比，汇总结果

---

## 7. 特殊场景处理

### 7.1 NaN 处理

- NPU 和 Golden 的 NaN 位置必须完全一致
- 否则直接判定失败

### 7.2 Inf 处理

**饱和边界场景**：
- FP32 → FP16 截断可能产生 Inf（超过 65504）
- 单边 Inf 时，替换为 dtype 最大有限值后继续比较
- 双方都是 Inf 且符号相同 → 视为匹配

### 7.3 整数类型

整数类型要求精确匹配（threshold = 0）。

---

## 8. 配置说明

### 8.1 全局阈值配置

在 `precision.py` 中定义：

```python
# 精度阈值
PRECISION_THRESHOLDS = {
    'float16': 2**-10,
    'bfloat16': 2**-7,
    'float32': 2**-13,
}

# 小值域阈值
SMALL_VALUE_THRESHOLDS = {
    'float16': 2**-11,
    'bfloat16': 2**-8,
    'float32': 2**-14,
}

# 小值域误差阈值
SMALL_VALUE_ERROR_THRESHOLDS = {
    'float16': 2**-16,
    'bfloat16': 2**-16,
    'float32': 2**-30,
}

# 相消边界阈值（基于 IEEE 754 精度位数）
CANCEL_BOUNDARY_THRESHOLDS = {
    'float32': 2**-8,   # ≈ 0.004
    'float16': 2**-5,   # ≈ 0.031
    'bfloat16': 2**-3,  # ≈ 0.125
}
```

### 8.2 算子自定义阈值

在 `proto.yaml` 中配置：

```yaml
precision_thresholds:
  float32: 0.005   # 自定义阈值
  float16: 0.01
  bfloat16: 0.01
```

### 8.3 阈值选择原则

- 默认阈值基于 dtype 精度位数
- 涉及三角函数的算子（如 RoPE）可适当放宽
- 优化器算子（如 Adam）涉及累积误差可适当放宽
- 不建议超过默认阈值的 10 倍

---

## 9. 测试验证

### 9.1 验证结果

| 算子 | CPU 模式 | NPU 模式 |
|------|----------|----------|
| ApplyRotaryPosEmb | 100% (20/20) | 95% (19/20) |
| ApplyAdamW | 100% (20/20) | 100% (20/20) |

### 9.2 边界场景验证

| 场景 | 预期结果 | 说明 |
|------|----------|------|
| 相消：NPU=0, CPU=0, golden=1e-3 | 通过 | IEEE 754 相消，NPU和CPU一致 |
| 相消：NPU=0, CPU=1e-3, golden=1e-3 | 不通过 | NPU有bug，CPU正确 |
| 小值域：NPU=CPU 相同误差 | 通过 | 小值域比值≈1 |
| 小值域：NPU有误差 CPU无 | 不通过 | NPU精度问题 |
| 大数值：golden=0.1, NPU=0 | 不通过 | 不满足相消条件 |

---

## 10. 参考文献

1. [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
2. [IEEE 754 浮点运算标准](https://en.wikipedia.org/wiki/IEEE_754)
3. [Kahan 灾难性相消理论](https://en.wikipedia.org/wiki/Catastrophic_cancellation)
4. numpy.allclose 组合容差机制

---

## 附录 A：CompareResult 数据结构

```python
@dataclass
class CompareResult:
    passed: bool                    # 最终通过判断
    dtype: str                      # 数据类型
    threshold: float                # 精度阈值
    mere: float                     # 平均相对误差
    mare: float                     # 最大相对误差
    max_diff: float                 # 最大绝对差异
    mean_diff: float                # 平均绝对差异
    mismatch_count: int             # 不匹配位置数
    total_count: int                # 总元素数
    mismatch_ratio: float           # 不匹配比例
    small_value_error_count: int    # 小值域 NPU 错误计数
    small_value_cpu_error_count: int# 小值域 CPU 错误计数
    small_value_total_count: int    # 小值域位置总数
    cancel_error_count: int         # 相消位置 NPU 错误计数
    cancel_cpu_error_count: int     # 相消位置 CPU 错误计数
    cancel_total_count: int         # 相消位置总数
    error_msg: Optional[str]        # 错误信息
```

## 附录 B：IEEE 754 精度位数与阈值关系

阈值基于 dtype 的精度位数设计：

| dtype | 尾数位数 | 相对精度 | cancel_boundary | 设计依据 |
|-------|----------|----------|-----------------|----------|
| FP32  | 23       | $10^{-7}$ | $2^{-8} \approx 0.004$ | 7位有效数字，$10^4$操作数相消得$10^{-3}$ |
| FP16  | 10       | $10^{-3}$ | $2^{-5} \approx 0.031$ | 3位有效数字，$10^2$操作数相消得$10^{-2}$ |
| BF16  | 7        | $10^{-2}$ | $2^{-3} \approx 0.125$ | 2位有效数字，$10^1$操作数相消得$10^{-1}$ |

**相消示例**：
- FP32: $10^4 - 10^4 = 10^{-3}$，但精度只够 7 位，结果可能丢失
- 设置 cancel_boundary = $2^{-8} \approx 0.004$，覆盖 $10^{-3}$ 范围