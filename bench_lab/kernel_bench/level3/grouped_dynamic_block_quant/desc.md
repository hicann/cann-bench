# GroupedDynamicBlockQuant

## 算子简介

`GroupedDynamicBlockQuant` 是面向大模型权重/激活量化场景的**分组动态块量化**算子。它根据 `group_list` 在输入的 M 轴（倒数第二维）上将数据划分为若干 group，每个 group 内部再按 `row_block_size × col_block_size` 划分为若干 block，逐 block 计算最大绝对值并推导 scale，最后将输入按 scale 缩放到 FP8/HiFP8 范围并 cast 到目标低比特类型。

- 难度等级：L3
- 所属类别：Quantization
- 典型应用：FP8 W8A8 / W8A16 量化、KV Cache 量化
- 支持硬件：Ascend 950PR / 950DT

## 算子定义

给定输入 `x`，group 划分由 `group_list` 决定：

```
group i 的行的范围：[group_list[i-1], group_list[i])，其中 group_list[-1] 定义为 0
```

对每个 block：

```
input_max = max(|x[block_rows, block_cols]|)
scale = min(input_max / FP_MAX, 1 / min_scale)
y = cast(input / scale)
```

其中 `FP_MAX` 取决于 `dst_type`：

- `dst_type = 35 (FLOAT8_E5M2)`：FP_MAX = (2 - 2^-2) * 2^15
- `dst_type = 36 (FLOAT8_E4M3FN)`：FP_MAX = (2 - 2^-2) * 2^8
- `dst_type = 34 (HIFLOAT8)`：FP_MAX = 2^15

`cast` 表示将 FP32 值量化到目标低比特类型，遵循 IEEE-like 规则并做饱和处理。

## 接口规范

### 函数原型

```python
grouped_dynamic_block_quant(
    x: torch.Tensor,
    group_list: torch.Tensor,
    min_scale: float = 0.0,
    round_mode: str = "rint",
    dst_type: int = 35,
    row_block_size: int = 1,
    col_block_size: int = 128,
    group_list_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | float16 / bfloat16 | 待量化输入，shape 为 `[M, N]` 或 `[B, M, N]` |
| group_list | int32 | 1D 非递减数组，最后一个元素等于 `x.shape[-2]` |
| min_scale | float | 最小 scale 值，大于等于 0；为 0 时不做上限裁剪 |
| round_mode | str | FP8 仅支持 `"rint"`；HIFLOAT8 支持 `"round"` / `"hybrid"` |
| dst_type | int | 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN |
| row_block_size | int | M 轴 block 大小：1 / 128 / 256 / 512 |
| col_block_size | int | N 轴 block 大小：64 / 128 / 192 / 256 |
| group_list_type | int | 当前仅支持 0（cumsum） |

### 输出参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| y | 见 dst_type | 量化输出，shape 与 `x` 相同 |
| scale | float32 | 每个 block 的 scale，shape 为 `[M//row_block_size + groupNum, ceil(N/col_block_size)]`（2D）或 `[B, M//row_block_size + groupNum, ceil(N/col_block_size)]`（3D） |

## 支持范围

- 维度：仅支持 2D、3D 输入
- x 数据类型：FLOAT16、BFLOAT16
- group_list 数据类型：INT32
- y 数据类型：HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2
- scale 数据类型：FLOAT32
- group_list 要求：非递减、最后一个元素等于 M、元素均大于等于 0
- row_block_size / col_block_size 仅支持文档所列枚举值

## Golden 实现

`golden.py` 提供 CPU 参考实现，主要步骤：

1. 将输入转换为 float32，并替换 inf/nan 为 0（实际实现中保留 inf/nan 原始位置，通过 scale 计算后恢复）。
2. 按 `group_list` 分组，每组按 `row_block_size / col_block_size` 补零对齐。
3. 逐 block 计算 `max(abs(x))`，推导 `scale = min(max / FP_MAX, 1 / min_scale)`。
4. 将 scale 扩展回原始 x 的 shape，计算 `x / scale`。
5. 对结果做饱和裁剪到目标类型的最大有限值，并返回 float32 张量（框架在精度对比时会自动截断到目标类型）。

## 评测说明

- 共 20 个 case，覆盖 2D/3D、对齐/非对齐、单 group/多 group、不同 block size、min_scale 边界等场景。
- 当前 baseline cases 聚焦 `FLOAT8_E5M2` 与 `FLOAT8_E4M3FN`；`HIFLOAT8` 已在 proto.yaml 中声明支持，但受限于当前 cann-bench 精度阈值/类型对比基础设施，baseline cases 未覆盖。
- 性能基线通过 `test_baseline_perf.py` 在 NPU 上采集。
