# Unique 算子 API 描述

## 1. 算子简介

去除张量中的重复元素。

**主要应用场景**：
- 数据去重与统计唯一值数量
- 构建词表或标签映射时提取不重复元素
- 稀疏表示和索引压缩中提取唯一键值及逆索引

**算子特征**：
- 难度等级：L3（SortSelect）
- 单输入双输出（唯一值张量和可选的逆索引），支持 ND 格式输入

## 2. 算子定义

### 数学公式

$$
y, inverse = \text{unique}(x, \text{return\_inverse})
$$

对输入张量 x 进行去重操作，返回唯一值张量 y。当 return_inverse=True 时，同时返回逆索引 inverse，满足 $x = y[inverse]$。

## 3. 接口规范

### 算子原型

```python
cann_bench.unique(Tensor x, bool return_inverse) -> (Tensor y, Tensor inverse)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 ND 格式 |
| return_inverse | bool | false | 是否返回逆索引，用于重建原始张量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 由唯一值数量决定 | 与输入 x 相同 | 输出张量，唯一值 |
| inverse | 与输入 x 展平后相同 | int64 | 逆索引，满足 x = y[inverse]（当 return_inverse=True 时） |

### 数据类型

| 输入 dtype | 输出 dtype（y） | 输出 dtype（inverse） |
|-----------|---------------|---------------------|
| bfloat16 | bfloat16 | int64 |
| float16 | float16 | int64 |
| float32 | float32 | int64 |
| int8 | int8 | int64 |
| int32 | int32 | int64 |
| int64 | int64 | int64 |
| uint8 | uint8 | int64 |

### 规则与约束

- 输入支持 ND 格式张量，去重前会将输入展平为一维
- 输出唯一值张量 `y` 的元素按数值升序排列（与 `torch.unique(..., sorted=True)` 默认行为一致），其长度取决于输入中不重复元素的数量
- 当 `return_inverse=false` 时，`inverse` 输出为 None
- 输出 `y` 的 dtype 与输入 `x` 相同；`inverse` 的 dtype 固定为 int64
- 浮点去重语义**要求 NPU 实现遵循 `torch.unique`**：按 IEEE 754 数值相等去重（`+0.0` 与 `-0.0` 必须合并为同一唯一值，PyTorch CPU 实现保留首次出现者，通常落到 `+0.0` 这一 sign）；`+inf` 与 `-inf` 视为不同唯一值（bit pattern 与 IEEE 值都不同）；本基准的输入生成器不引入 NaN，无需考虑 NaN 去重语义。若 NPU 将 `±0.0` 视为不同唯一值，`y` 长度会与 Golden 不匹配，bit-exact 检查直接因 shape mismatch 失败

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank`（输入维度数） | 1 ~ 8 | 去重前展平为一维；cases.csv 实测 1 ~ 5 |
| `numel`（元素总数） | 1 ~ 2^30 | cases.csv 实测 ~917K ~ 256M |
| `return_inverse` | {true, false} | cases.csv 双向覆盖 |

约束（§3 规则与约束之外的结构性不变量，避免与上文重复）：
- shape 关系：输出 `y` 始终为 1D（无论 `x.rank`），长度 `len(y) <= numel(x)`；`return_inverse=True` 时 `inverse.shape == (numel(x),)`，与 `x.flatten()` 等长。
- 重建不变量：`return_inverse=True` 时严格有 `x.flatten() == y[inverse]`（逐元素相等）。
- 索引值域：`inverse` 元素 ∈ [0, len(y) - 1]，每个 `[0, len(y))` 中的 ID 至少出现一次（满射）。

## 4. 精度要求

采用**逐位精确（bit-exact）**标准进行验证：对每个输出张量，要求 NPU 输出与 Golden 参考输出 **逐元素按位完全相等**。

### 判定方式

| 输出 | dtype | 判定条件 |
|---|---|---|
| `y` (整数) | int8 / int32 / int64 / uint8 | `torch.equal(actual, golden) == True`（整数 dtype 字节天然唯一） |
| `y` (浮点) | bfloat16 / float16 / float32 | `torch.equal(actual.view(int_dtype), golden.view(int_dtype)) == True`，其中 `int_dtype` 与浮点宽度对应（fp16/bf16 → int16，fp32 → int32），严格区分 `±0.0`、`±inf` 与 NaN payload |
| `inverse` | int64 | `torch.equal(actual, golden) == True` |

实现机制：`proto.yaml` 中将全部 dtype 阈值设为 `0`，触发 `src/kernel_eval/utils/precision.py` 的 bit-exact 分支（整数路径 `torch.equal`、浮点路径 `.view(int_dtype)` 后 `torch.equal`）。

### 选用理由

- `y` 的所有元素都直接来自 `x`（没有任何算术运算），bit-exact 比较良定义；
- `inverse` 为离散整数索引，天然适合按位比较；
- 输入空间离散（浮点 dtype 的 bit pattern 集合有限，整数 dtype 完全离散），不存在浮点累积误差来源。

### Bit-exact 前置条件

- **形状一致**：NPU 与 Golden 必须给出相同长度的 `y`（即对相同输入产生相同的去重计数），否则比较直接判失败；
- **排序约定**：双方均按数值升序输出 `y`；NPU 实现需遵循 `torch.unique` 的排序语义；
- **特殊值处理**：cases 仅覆盖 ±inf（不含 NaN），±inf 在升序约定下分别位于 `y` 的首/末位置。

### 字节级严格性

`torch.unique` 按 IEEE 754 数值相等做去重，但保留哪个 sign（`+0.0` 还是 `-0.0`）由实现决定。
本算子要求 NPU 与 Golden 在 sign 选择上一致——为防止双方选择分歧而被"数值相等"路径放过，
bit-exact 判定通过 `src/kernel_eval/utils/precision.py` 中浮点 `threshold == 0` 的字节级
分支实现：对 fp16 / bf16 / fp32 输出执行 `out.view(int_dtype) == golden.view(int_dtype)`
的逐位比较（harness 已在 Golden fp64 → 目标 dtype 的 round-trip 中处理 dtype 转换）。
因此以下场景均会被显式判失败：

- `+0.0` 与 `-0.0` 的 sign 分歧（`0x00000000` vs `0x80000000` 等）；
- `+inf` 与 `-inf` 的 sign 分歧；
- NaN 的 payload 差异（虽然本基准不引入 NaN，但路径同样按字节严格）。


## 5. 标准 Golden 代码

```python
import torch

def unique(
    x: torch.Tensor,
    return_inverse: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    去除张量中的重复元素

    公式：y, inverse = unique(x, return_inverse)

    Args:
        x: 输入张量
        return_inverse: 是否返回逆索引，用于重建原始张量

    Returns:
        y: 唯一值张量
        inverse: 逆索引，满足 x = y[inverse] (当 return_inverse=True 时)
    """

    if return_inverse:
        y, inverse = torch.unique(x, return_inverse=True)
        return y, inverse
    else:
        y = torch.unique(x, return_inverse=False)
        return y, None
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.tensor([1, 2, 3, 2, 1, 4, 3], dtype=torch.int32, device="npu")
y, inverse = cann_bench.unique(x, True)  # y=[1,2,3,4], inverse=[0,1,2,1,0,3,2]

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y, _ = cann_bench.unique(x, False)  # 仅返回唯一值
```
