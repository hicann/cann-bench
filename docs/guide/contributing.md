# kernel_bench 算子贡献指南

**创建算子目录**  
  在 `bench_lab/` 目录下创建以任务名（例如 `driver_bench`）+算子名称命名的文件夹（例如 `bench_lab/driver_bench/my_new_op/`）。

## 贡献规范

- **算子原型**：需符合Ascend C算子开发规范，支持动态shape、数据类型兼容性等特性。
- **参考实现**：需通过PyTorch官方接口实现，确保在CPU/NPU环境可正确运行。
- **测试用例**：需覆典型场景。
- **文档描述**：需清晰说明算子功能、适用场景及与同类算子的差异。

| 文件 | 用途 |
|------|------|
| `proto.yaml` | 算子原型定义（schema、输入/输出、属性） |
| `golden.py` | PyTorch 参考实现（用于精度对比） |
| `desc.md` | 算子 API 文档（面向使用者的完整说明） |
| `cases.yaml` | 测试用例定义（机器可读） |
| `cases.csv` | 测试用例定义（与 cases.yaml 内容一致，便于人工审查） |

---

## 1. proto.yaml

算子原型定义文件，采用 YAML 格式。所有字段名、缩进、枚举值须严格遵循本规范。

### 1.1 整体结构

```yaml
operator:
  name: Conv2D                          # 算子名：PascalCase，与目录名语义对应
  category: Contraction                 # 算子类别（见 1.2）
  difficulty: L3                        # 难度等级：L1 ~ L4
  formula: y = CONV(x, filter) + bias   # 数学公式（单行）
  # formula: |                           # 多行公式用 YAML 多行字符串
  #   m_t = beta1 * m_{t-1} + (1 - beta1) * grad
  #   v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
  description: 计算2D卷积               # 一句话中文描述
  shape_support: 'x: [N, C_in, H, W]'   # 输入 shape 约束说明
  attrs:                                # 属性列表（见 1.3）
    - name: strides
      type: ListInt
      description: 步长
      required: true
  inputs:                               # 输入张量列表（见 1.4）
    - name: x
      description: 输入特征图
      dtype:
        - float16
        - float32
  outputs:                              # 输出张量列表（见 1.5）
    - name: y
      description: 输出特征图
      dtype:
        - float16
        - float32
  schema: conv2_d(Tensor x, ...) -> Tensor y   # API 签名（见 1.6）
```

### 1.2 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 算子名称，使用 PascalCase（如 `Conv2D`、`ApplyAdamW`） |
| `category` | string | 是 | 算子类别 |
| `difficulty` | string | 是 | 难度等级，取值 `L1` / `L2` / `L3` / `L4`，须与所在 `levelN/` 目录一致 |
| `formula` | string | 是 | 数学公式，单行直接写字符串；多行使用 YAML `|` 块标量 |
| `description` | string | 是 | 一句话中文描述 |
| `shape_support` | string | 是 | 输入/输出 shape 的格式约束说明 |
| `attrs` | list | 否 | 非标量属性列表（kernelSize、stride、epsilon 等） |
| `inputs` | list | 是 | 输入张量列表 |
| `outputs` | list | 是 | 输出张量列表 |
| `schema` | string | 是 | API 函数签名 |
| `note` | string | 否 | 补充说明 |

### 1.3 category 常见枚举

| 取值 | 含义 | 代表算子 |
|------|------|----------|
| `Elementwise` | 逐元素运算 | Exp、Sigmoid |
| `FusedComposite` | 融合算子 | ApplyAdamW、SwiGLU |
| `Reduction` | 归约运算 | ArgMax、AdaptiveAvgPool3D |
| `Contraction` | 收缩/卷积/矩阵运算 | Conv2D、GroupedMatmul |
| `Transform` | 张量变换 | Transpose、StridedSlice |
| `Sort` | 排序/去重 | TopK、Unique |
| `Index` | 索引/散列 | Gather、Scatter |
| `Vision` | 视觉/检测 | ROIAlign、NMSWithMask |

### 1.4 attrs 属性定义

每个属性项包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 属性名，使用 camelCase（如 `kernelSize`、`weight_decay`） |
| `type` | string | 是 | 数据类型，取值见下方 type 枚举 |
| `description` | string | 是 | 中文描述 |
| `default` | any | 否 | 默认值。`bool` 类型须用字符串 `"true"` / `"false"` |
| `required` | bool | 是 | 是否必填，使用字符串 `"true"` / `"false"` 或 YAML bool |

#### type 枚举

| 取值 | 含义 | 示例 |
|------|------|------|
| `float` | 浮点标量 | `lr: 0.001` |
| `int` | 整数标量 | `groups: 1` |
| `bool` | 布尔标量 | `maximize: false` |
| `ListInt` | 整数列表 | `stride: [1, 1]` |
| `ListFloat` | 浮点列表 | `scale: [1.0, 2.0]` |

> **注意**：列表类型统一使用 `ListInt` / `ListFloat`，不使用 `IntArray` / `Int`（大小写不统一）。
> 标量整数用 `Int`，标量浮点用 `float`，布尔用 `bool`。

### 1.5 inputs / outputs 张量定义

每个张量项包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 张量名，如 `x`、`weight`、`y` |
| `description` | string | 是 | 中文描述 |
| `dtype` | list[string] | 是 | 支持的数据类型列表，取值见下方 dtype 枚举 |

#### dtype 枚举

| 取值 | 含义 |
|------|------|
| `float16` | 半精度浮点 |
| `float32` | 单精度浮点 |
| `bfloat16` | Brain 浮点 |
| `int32` | 32位有符号整型 |
| `int64` | 64位有符号整型 |
| `int16` | 16位有符号整型 |
| `int8` | 8位有符号整型 |
| `uint32` | 32位无符号整型 |
| `uint64` | 64位无符号整型 |
| `uint16` | 16位无符号整型 |
| `uint8` | 8位无符号整型 |
| `bool` | 布尔型 |

### 1.6 schema 签名

格式：

```
op_name(Tensor input1, Tensor input2, attr1 type1, attr2 type2, ...) -> Tensor output
```

规则：
- 函数名使用 snake_case，与 `name` 的 PascalCase 对应
- Tensor 参数只写 `Tensor <name>`，不写 dtype（dtype 在 `inputs` 中定义）
- 属性参数写 `<name>` 后跟类型简写：`int[]`（列表）、`float`、`int`、`bool`
- 有默认值的属性写 `name=defaultValue`
- 返回值统一写 `-> Tensor <name>`

示例：

```
conv2_d(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations=[1,1], int groups=1) -> Tensor y
```

---

## 2. golden.py

PyTorch 参考实现，用于精度对比。

### 2.1 文件结构

```python
import torch

"""
<OpName>算子Torch Golden参考实现

<中文描述>
公式: <数学公式>
"""
def <op_name>(
    input1: torch.Tensor,
    input2: torch.Tensor,
    attr1: list,
    attr2: int,
    ...
) -> torch.Tensor:
    """
    <中文描述>

    公式: <数学公式>

    Args:
        input1: <描述>
        attr1: <描述>

    Returns:
        <描述>
    """
    # 参数转换（list → tuple 等）
    # 调用 torch API
    return result
```

### 2.2 编写要求

| 要求 | 说明 |
|------|------|
| **优先使用 PyTorch 内置 API** | 如 `torch.nn.functional.conv2d`、`torch.nn.functional.interpolate`、`torch.unique` 等 |
| **参数名与 proto.yaml 一致** | golden 函数签名中的参数名须与 `schema` 中的参数名逐一对应 |
| **无额外依赖** | 只允许 import `torch`，不允许 import 第三方库 |
| **参数转换明确** | `list` → `tuple` 等转换须在函数体内显式写出 |
| **无冗余代码** | 不要写死代码、未使用的 import、多余类型转换 |
| **dtype 保持** | 输出 dtype 须与输入 dtype 一致（除非算子本身改变 dtype） |
| **公式与 proto 一致** | docstring 中的公式须与 `proto.yaml` 的 `formula` 字段一致 |

### 2.3 常见模式

```python
# 列表 → tuple（PyTorch API 需要 tuple 时）
stride_val = (stride[0], stride[1])

# 调用 PyTorch 函数式 API
y = torch.nn.functional.conv2d(x, weight, bias,
                                stride=stride_val,
                                padding=padding_val,
                                dilation=dilation_val,
                                groups=groups)

# 直接调用 torch 函数
y = torch.topk(x, k, dim=-1, largest=True, sorted=True)
```

---

## 3. desc.md

算子 API 文档，面向使用者。采用 Markdown 格式。

### 3.1 章节结构

```markdown
# <OpName> 算子 API 描述

## 1. 算子简介

<中文描述>

**主要应用场景**：
- <场景1>
- <场景2>

**算子特征**：
- 难度等级：L<N>（<category>）
- <输入输出描述>
- <shape 描述>

## 2. 算子定义

### 数学公式

$$
<LaTeX 公式>
$$

<公式说明文字>

## 3. 接口规范

### 算子原型

```python
cann_bench.<op_name>(<参数列表>) -> Tensor <output>
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| <name> | Tensor | 必选 | <描述> |
| <attr> | <type> | 必选 / <默认值> | <描述> |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | <shape描述> | 与输入相同 | <描述> |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

> 须与 `proto.yaml` 中 inputs/outputs 的 dtype 列表完全一致

### 规则与约束

- <约束1>
- <约束2>

## 4. 标准 Golden 代码

```python
<完整 golden.py 内容>
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

<示例代码>
```
y = cann_bench.<op_name>(x, weight, bias, stride, padding, dilation, groups)
```

### 3.2 与 proto.yaml 的一致性要求

| desc.md 内容 | 对应 proto.yaml 字段 |
|-------------|---------------------|
| 算子名称 | `operator.name` |
| 难度等级 | `operator.difficulty` |
| 公式 | `operator.formula` |
| 输入参数表 | `operator.inputs` + `operator.attrs` |
| 输出表 | `operator.outputs` |
| 数据类型表 | `operator.inputs[].dtype` + `operator.outputs[].dtype` |
| schema | `operator.schema` |
| Golden 代码 | `golden.py` 全文 |
---

## 4. cases.yaml

测试用例定义文件，机器可读格式。

### 4.1 整体结构

```yaml
cases:
- operator: <OpName>
  case_id: 1
  input_shape: [[<shape1>], [<shape2>], ...]
  dtype: [dtype1, dtype2, ...]
  attrs: {<attr_name>: <value>, ...}
  value_range: [[min1, max1], [min2, max2], ...]
  baseline_perf_us: None
  t_hw_us: None
  note: <简短描述>
- operator: <OpName>
  case_id: 2
  ...
```

### 4.2 字段定义

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `operator` | string | 是 | 算子名称，与 `proto.yaml` 中 `name` 一致 |
| `case_id` | int | 是 | 用例编号，从 1 开始递增 |
| `input_shape` | list[list[int] \| null] | 是 | 每个输入张量的 shape，顺序与 `proto.yaml` 中 `inputs` 顺序一致；可选参数用 `null` 表示不使用 |
| `dtype` | list[string] \| [string] | 是 | 每个输入张量的数据类型；支持单值简写 `[dtype]` 表示所有 tensor 使用相同类型 |
| `attrs` | dict | 是 | 属性键值对，键名与 `proto.yaml` 中 `attrs[].name` 一致 |
| `value_range` | list[list] | 是 | 每个输入张量的随机数取值范围 `[min, max]`，数量与 `input_shape` 一致。特殊值：`[-inf, inf]`、`[nan, nan]`、`[0, 0]`；可选参数对应位置用 `[0, 0]` 或 `null` |
| `baseline_perf_us` | float / None | 是 | 性能基线（微秒），PyTorch 参考实现在目标 NPU 上的实测时间，未测量时填 `None` |
| `t_hw_us` | float / None | 是 | 硬件下界 T_HW（微秒），用于 SOL-anchored 性能评分，未给出时填 `None` |
| `note` | string | 是 | 用例简短描述，建议格式：`<dtype>-<数据规模>-<对齐/非对齐>-<特征>` |

### 4.3 input_shape 可选参数处理

当算子有可选输入参数时，用 `null` 表示该参数不使用：

```yaml
# 可选参数示例：QuantMatmul 的 offset、pertoken_scale、bias 可选
input_shape:
- [1024, 3584]      # x（必选）
- [3584, 3584]      # weight（必选）
- [3584]            # scale（必选）
- null              # offset（不使用）
- null              # pertoken_scale（不使用）
- null              # bias（不使用）
dtype: [int8, int8, float32, null, null, null]
value_range:
- [-128, 127]
- [-128, 127]
- [0.001, 0.01]
- [0, 0]            # 不使用时填零值范围
- [0, 0]
- [0, 0]
attrs: {offset: null, pertoken_scale: null, bias: null, output_dtype: float16}
```

### 4.4 dtype 单值简写

当所有输入张量使用相同数据类型时，可使用单值简写：

```yaml
# 单值简写示例：所有 tensor 都用 float32
input_shape:
- [[1024, 1024], [1024, 1024]]   # TensorList x1（2个tensor）
- [[1024, 1024], [1024, 1024]]   # TensorList x2（2个tensor）
- [[1024, 1024], [1024, 1024]]   # TensorList x3（2个tensor）
dtype: [float32]                  # 单值简写，自动展开为6个float32
```

校验逻辑会自动将单值 dtype 展开为与 input_shape 相同长度的列表。

### 4.5 用例设计原则

结合评测目标，设计用例覆盖场景，例如当前kernel_bench用例覆盖泛化场景（建议 20+ 个用例）：
**note:当前场景仅供参考**
| 场景类别 | 说明 |
|---------|------|
| **基准场景** | 对齐 shape（64 的倍数）、典型 kernel/stride/padding |
| **参数变化** | 不同 kernelSize、stride、padding、dilations 组合 |
| **非对齐 shape** | 质数、奇数等非 64 对齐的 shape |
| **多形态 shape** | 小 shape、大 shape、极端 shape |
| **特殊值** | inf、nan、零值、边界值（如 float16 的 65504） |
| **dtype 覆盖** | float16、float32、bfloat16 至少各有用例 |

### 4.6 value_range 特殊值

| 取值 | 含义 |
|------|------|
| `[-1, 1]` | 均匀分布在 [-1, 1] 区间的随机值 |
| `[-inf, inf]` | 包含正负无穷 |
| `[nan, nan]` | NaN 值 |
| `[0, 0]` | 全零 |
| `[-65504, 65504]` | float16 可表示的最大范围 |

---

## 5. cases.csv

测试用例定义文件，与 cases.yaml 内容完全一致，CSV 格式便于人工审查和 diff。

### 5.1 列定义

| 列名 | 对应 cases.yaml 字段 | CSV 格式说明 |
|------|---------------------|-------------|
| `operator` | `operator` | 字符串 |
| `case_id` | `case_id` | 整数 |
| `input_shape` | `input_shape` | JSON 数组字符串，如 `"[[2, 64, 32, 32], [64, 1, 3, 3], [64]]"` |
| `dtype` | `dtype` | JSON 数组字符串，如 `"['float16', 'float16', 'float16']"` |
| `attrs` | `attrs` | Python dict 字符串，如 `"{'kernelSize': [3, 3], 'stride': [1, 1]}"` |
| `value_range` | `value_range` | JSON 数组字符串，如 `"[[-1, 1], [-1, 1], [-0.1, 0.1]]"` |
| `baseline_perf_us` | `baseline_perf_us` | 浮点数或空 |
| `t_hw_us` | `t_hw_us` | 浮点数或空（硬件下界 T_HW，单位 µs） |
| `note` | `note` | 字符串 |

### 5.2 一致性要求

- `cases.csv` 与 `cases.yaml` 的**用例数量、字段值必须完全一致**
- 修改用例时须**同时更新两个文件**
- `case_id` 顺序一致，不能跳跃或重复

---

## 6. 四文件一致性检查清单

提交前请确认：

- [ ] `proto.yaml` 的 `formula` 与 `golden.py` docstring 中的公式一致
- [ ] `proto.yaml` 的 `schema` 参数名与 `golden.py` 函数参数名一致
- [ ] `proto.yaml` 的 `inputs`/`outputs` dtype 列表与 `desc.md` 数据类型表一致
- [ ] `proto.yaml` 的 `difficulty` 与所在 `levelN/` 目录层级一致
- [ ] `proto.yaml` 的 `attrs` 键名与 `cases.yaml`/`cases.csv` 的 `attrs` 键名一致
- [ ] `cases.yaml` 与 `cases.csv` 内容完全一致
- [ ] `desc.md` 中 Golden 代码块与 `golden.py` 内容一致
- [ ] `desc.md` 中算子示例代码的参数名与 `proto.yaml` schema 一致
- [ ] `golden.py` 只 import `torch`，无额外依赖
- [ ] `proto.yaml` 中 type 使用统一命名：`ListInt`/`ListFloat`/`Int`/`float`/`bool`
