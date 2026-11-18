# cases.yaml 规范文档

## 1. input_shape 顺序规范

**核心规则**：`input_shape` 必须与 `proto.yaml` 的 `inputs` **顺序严格一致**。

省略的 optional 参数必须用 `null` 占位，不能跳过。

### 示例

假设 `proto.yaml` 定义：
```yaml
inputs:
- name: x                 # 必需
- name: weight_ih         # 必需
- name: weight_hh         # 必需
- name: bias_ih           # optional
- name: bias_hh           # optional
- name: h0                # optional
```

**正确写法**（省略 h0）：
```yaml
input_shape:
- [10, 4, 64]      # x
- [[96, 64]]       # weight_ih
- [[96, 32]]       # weight_hh
- [[96]]           # bias_ih
- [[96]]           # bias_hh
- null             # h0 省略，用 null 占位
```

**错误写法**（跳过 h0）：
```yaml
input_shape:
- [10, 4, 64]      # x
- [[96, 64]]       # weight_ih
- [[96, 32]]       # weight_hh
- [[96]]           # bias_ih
- [[96]]           # bias_hh
# 错误！缺少 h0 的 null 占位，后端无法判断是省略 h0 还是省略 bias_hh
```

## 2. TensorList 格式规范

TensorList（如 weights）用**嵌套列表**表示：

```yaml
# 单个 TensorList（包含多个 tensor）
- [[96, 64], [96, 32]]  # TensorList of 2 tensors
```

## 3. dtype 对应规范

逐输入的 `dtype` 列表应与 `input_shape` 顺序一致；统一 dtype 也可写为字符串（如 `dtype: float32`），加载器会规范化为逐输入列表。发布文件优先使用明确的逐输入写法：

```yaml
dtype:
- float32           # x 的 dtype
- [float32]         # weight_ih (TensorList，每个 tensor 的 dtype)
- [float32]         # weight_hh
- [float32]         # bias_ih
- [float32]         # bias_hh
- null              # h0 省略时保留 dtype 占位，避免后续输入错位
```

## 4. value_range 对应规范

`value_range` 支持共享区间简写和逐输入列表。`value_range: [-1, 1]` 表示所有输入使用同一区间，并非只给第一个输入；需要不同区间时按 `input_shape` 顺序列出：

```yaml
value_range:
- [-0.1, 0.1]       # x
- [[-0.1, 0.1]]     # weight_ih
- [[-0.1, 0.1]]     # weight_hh
- [[-0.01, 0.01]]   # bias_ih
- [[-0.01, 0.01]]   # bias_hh
- null              # h0 省略
```

## 5. 后端简化处理逻辑

以下仅示意已经规范化的逐输入映射。实际实现还处理共享 dtype、共享 value_range、TensorList 和 null，不能直接把原始 YAML 按索引取值。当前规范化逻辑见 [data_generator.py](../../src/kernel_eval/data/data_generator.py) 的 `generate_input_tensors_from_case` / `_normalize_value_ranges`。

```python
# 简化后的参数映射逻辑
params = {}
tensor_idx = 0
for i, input_info in enumerate(proto_inputs):
    shape = input_shapes[i]
    if shape is None:
        # optional 参数省略
        params[input_info.name] = None
    else:
        # 按 shape 生成 tensor
        tensor = generate_tensor(shape, dtypes[i], value_ranges[i])
        params[input_info.name] = tensor
```

## 6. 检查清单

编写 cases.yaml 时请检查：

| 检查项 | 说明 |
|---|---|
| ✅ 顺序一致 | `input_shape` 长度 == `proto.yaml inputs` 长度 |
| ✅ null 占位 | 每个省略的 optional 参数都有 `null` |
| ✅ dtype 对应 | 逐输入列表与 `input_shape` 对齐，或显式使用统一 dtype 字符串 |
| ✅ value_range 对应 | 逐输入列表对齐，或使用所有输入共享的 `[min, max]` 简写 |
| ✅ TensorList 格式 | 嵌套列表表示多个 tensor |

## 7. 迁移指南

现有 cases.yaml 迁移步骤：

1. 查看 `proto.yaml` 的 `inputs` 定义和顺序
2. 对照现有 `input_shape`，找出缺失的 optional 参数
3. 在正确位置插入 `null`
4. 同步更新 `dtype` 和 `value_range`
