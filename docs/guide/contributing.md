# tasks 算子贡献指南

**创建算子目录**

  评测正式集放在 `tasks/levelN/<op_name>/`（`N` 取 1/2/3/4，对应难度等级）。被 `python -m kernel_eval.cli` 实际加载的就是这个目录。

  `bench_lab/` 是**实验/孵化区**：准备测试要放入 cann-bench 的算子可先放此处暂存，本地测试无误后通过 PR 转入对应 `tasks/levelN/` 主测试集合。

  示例：新增 L2 算子 `MyOp` → 创建 `tasks/level2/my_op/`，按下文 1~6 节的规范放好 `proto.yaml`、`golden.py`、`desc.md`、`cases.yaml`、`cases.csv`，并通过 §6 的一致性检查清单。

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
  schema: conv_2d(Tensor x, ...) -> Tensor y   # API 签名（见 1.6）
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

> 注：`required` 目前为声明性字段，CANN loader 暂未据此对入参做强制校验（仅供文档/审查参考）。

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

> **输出专用字段**（仅 `outputs` 项可用）：
> - `compare`（bool，默认 `true`）：该输出是否参与精度比对。对非确定性输出（如 TopK / ArgSort 的索引）设 `false` 可跳过逐值比对。
> - `index_gather`（dict）：为索引类输出声明 tie-order 无关校验，字段为 `{input: <源张量名>, dim_attr: <维度属性名>, value_output: <对应值输出名>}`。框架据此校验 `源张量.gather(dim, 本索引输出) == 值输出`，从而在不依赖并列元素顺序的前提下验证索引正确性（示例见 `tasks/level3/top_k/proto.yaml`）。

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
| `float8_e4m3fn` | FP8 E4M3（低精度，**仅 Ascend 950**，见下方 SoC 红线） |
| `float8_e5m2` | FP8 E5M2（低精度，**仅 Ascend 950**） |
| `hifloat32` | HiFLOAT32（精度阈值见 benchmark_spec §4.4） |

> **SoC dtype 红线**：Ascend 910B / 910C（含 A3）**不支持** fp8（e4m3 / e5m2）、hifloat8、mxfp8 / mxfp4、e8m0（mx scale）、fp4 等低精度类型——这些类型**仅 Ascend 950** 支持。为 910B / 910C 设计用例时请勿使用上述 dtype，否则会在评审阶段被判为无效。特例：量化算子若以整数 `dst_type` 属性表达输出、且输入仅为 fp16 / bf16，则不触发此红线。

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
conv_2d(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations=[1,1], int groups=1) -> Tensor y
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
| **纯 CPU / 不引用 torch_npu** | golden 默认以 fp64-on-CPU 执行（框架内部将输入上升为 float64 并置于 CPU 计算）；**不得 import 或调用 `torch_npu`**，否则该参考执行路径会崩溃 |
| **输出顺序与 proto 一致** | 返回值（单输出或 tuple）顺序须与 `proto.yaml` 的 `outputs` 声明顺序逐一对应——框架按**位置**将 golden 输出与候选输出配对比对 |
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

### 2.4 oracle / bench 拆分（可选，用于消除小值域/相消退化）

归约类算子（matmul、attention 等）的 golden 若对**浮点或量化操作数**硬编码 `.float()`/`.double()`，会把精度评测用的 fp64 参考也一并下采成 fp32，导致小值域/相消判定中同精度参考与真值 `|bench − oracle| ≡ 0`、CPU 侧 ErrorCount 恒为 0（判定细节见 [benchmark_spec.md](../spec/benchmark_spec.md) §4.4）。此时可为该 golden 额外提供两个带后缀的同签名函数，evaluator 自动优先使用（缺省则回退到 plain golden，未拆分的算子完全不受影响）：

| 函数 | 角色 | 精度 | 用途 |
|------|------|------|------|
| `<op_name>`（plain） | **bench (b)**，默认 | 同精度：操作数按 case dtype、累加按硬件约定（如 fp32 累加器） | 小值域/相消的 CPU 同精度对照 + golden-npu-mock ST 候选 |
| `<op_name>_oracle` | **oracle (g)** | fp64 数学真值：dtype-agnostic，不硬编码 `.float()`/`.double()` | MERE/MARE 参考 + `golden_truncated` 基准 |
| `<op_name>_bench` | **bench (b)** 覆盖 | 同精度，但精度约定与 plain 不同 | 仅当 plain golden 不是忠实同精度 bench 时才写 |

**核心约定：**

1. **plain golden 默认兼任 bench。** evaluator 的 bench 路径是 `get_bench_function() or <plain golden>`，缺 `_bench` 时回退 plain golden。因此 plain golden 应写成**忠实的同精度参考**（操作数按 case dtype 舍入、fp32 累加器），它同时是 golden-npu-mock ST 的候选。

2. **`_oracle` 用于消除退化。** 把 plain golden 里的硬 `.float()`/`.double()` 去掉、全程跟随输入精度（在 `golden_precision=fp64_cpu` 下即在 fp64 计算），使 oracle 成为真正的 fp64 真值，`|b − oracle|` 不再恒为 0。

3. **`_bench` 仅在其精度与 plain golden 不同时才写。** 判据是 golden 里的 `.float()` 落在什么操作数上：
   - **浮点操作数**（fp16/bf16）的 `.float()` = **无损升精**（值不变）→ 操作数精度仍等于硬件语义 → **plain 就是 bench**，只补 `_oracle`（如 grouped_matmul：A16W16 权重非量化）。
   - **整型操作数**（int8/int4 量化权重）的 `.float()` = **反量化到 fp32**，而硬件反量化到 bf16/fp16 → 精度不符 → plain 应直接写成硬件精度的反量化（= bench），`_oracle` 保留 fp64 反量化真值（如 weight_quant_batch_matmul：A16W8）。

**示例：**

- `tasks/level3/grouped_matmul/`：plain（fp16/bf16 操作数升 fp32 累加，即 A16W16 bench）+ `_oracle`（fp64），无 `_bench`。
- `tasks/level3/weight_quant_batch_matmul/`：plain（int8 反量化到输出精度 T + fp32 累加，即 A16W8 bench）+ `_oracle`（fp64 反量化真值），无 `_bench`。

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
| `note` | string | 是 | 用例简短描述，建议格式：`<dtype>-<数据规模>-<对齐/非对齐>-<特征>` |

> **注意（baseline 已外置）**：性能基线 `baseline_perf_us` 与硬件下界 `t_hw_us` **不再写入 cases.yaml / cases.csv**，已统一外置到 `<bench>/metadata/<hardware>.json`（由 `BaselineStore` 按硬件加载）。在 cases 文件中写这两个字段会被一致性测试 `tests/ut/test_cases_yaml_csv_consistency.py`（`test_yaml_no_baseline_fields` / `test_csv_no_baseline_columns`）判为失败。

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

结合评测目标，设计用例覆盖场景，例如当前tasks 用例覆盖泛化场景（建议 20+ 个用例）：
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
| `note` | `note` | 字符串 |

> `cases.csv` 同样**不含** `baseline_perf_us` / `t_hw_us` 列（baseline 已外置到 `<bench>/metadata/<hardware>.json`，见 §4.2 注意）。

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
