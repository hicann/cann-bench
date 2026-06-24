# DynamicMxQuantWithDualAxis 算子 API 描述

## 1. 算子简介

对输入张量同时在 **-1 轴**和 **-2 轴**执行 Microscaling (MX) 动态量化。在给定的两个轴上，每 32 个数计算一组共享指数（scale），将组内元素除以对应 scale 后转换到目标低精度类型（FP4/FP8），分别输出两路量化结果及对应的 scale。

**主要应用场景**：
- LLM 推理中的权重/激活双轴量化压缩（FP4/FP8 低精度部署）
- MX 格式量化训练（OCP Microscaling 标准兼容）
- 大模型 KV Cache 压缩存储
- 混合精度推理中的动态量化前处理

**算子特征**：
- 难度等级：L3（Quantization）
- 1 输入，4 输出，4 个属性参数
- 支持 ND 格式输入，2-7 维
- 支持 3 种 scale 计算算法（scaleAlg=0/1/2），4 种目标量化格式，3 种舍入模式
- 双轴同时量化：-1 轴结果输出为 y1/mxscale1，-2 轴结果输出为 y2/mxscale2
- 分组量化：每轴均按 blocksize=32 独立计算 scale
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

### 数学公式

将输入 x 在 **-1 轴**上按 32 个数分组，一组 32 个数 `\{V_i\}_{i=1}^{32}` 动态量化为 `\{mxscale1, \{P_i\}_{i=1}^{32}\}`：

- 当 scaleAlg 为 0 时：

  $$
  shared\_exp1 = floor(log_2(max_i(|V_i|))) - emax
  $$

  $$
  mxscale1 = 2^{shared\_exp1}
  $$

  $$
  P_i = cast\_to\_dst\_type(V_i / mxscale1, round\_mode)
  $$

同时，将输入 x 在 **-2 轴**上按 32 个数分组，一组 32 个数 `\{V_j\}_{j=1}^{32}` 动态量化为 `\{mxscale2, \{P_j\}_{j=1}^{32}\}`：

  $$
  shared\_exp2 = floor(log_2(max_j(|V_j|))) - emax
  $$

  $$
  mxscale2 = 2^{shared\_exp2}
  $$

  $$
  P_j = cast\_to\_dst\_type(V_j / mxscale2, round\_mode)
  $$

- 当 scaleAlg 为 1 时，仅涉及 FP8 类型，采用 CuBALS 块缩放算法：
  - 分别对 -1 轴和 -2 轴的每组计算块缩放因子 `S_{fp32}^b = Amax(D_{fp32}^b) / Amax(DType)`
  - 将浮点缩放因子转换为 FP8 可表示的 `S_{ue8m0}^b = 2^{E_{int}^b}`，其中指数按需向上取整
  - 最终输出 `\left(S^b, [d^i]\right)` 作为对应轴的量化结果

- 当 scaleAlg 为 2 时，仅涉及 FP4_E2M1 类型：
  - 默认 `dst_type_max=0.0` 时等价于使用 FP4_E2M1 最大表示值 6.0
  - 对每组计算共享指数时，根据尾数位高比特是否进位选择 `ceil(log2(...))` 或 `floor(log2(...))`
  - 当前 torch_npu Python 接口未暴露 `dst_type_max` 参数，评测用例中统一使用 0.0

**emax 取值**：

|   DataType    | emax |
| :-----------: | :--: |
|  FLOAT4_E2M1  |  2   |
|  FLOAT4_E1M2  |  0   |
| FLOAT8_E4M3FN |  8   |
|  FLOAT8_E5M2  |  15  |

### 合轴说明

算子实现时，会对 **-2 轴（不包含）之前的所有轴**进行合轴处理。即对于输入 shape 为 `(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})` 的张量，`-2 轴之前的维度 (d_0, ..., d_{n-3})` 会被合并为一个维度，等效于将输入 reshape 为 `(d_0 * ... * d_{n-3}, d_{n-2}, d_{n-1})` 后再进行量化计算。

### 舍入模式说明

- **rint**: 银行家舍入（tie to even），0.5 取偶数
- **round**: 标准四舍五入（tie away from zero）
- **floor**: 向负无穷方向舍入

注：当 dstType 为 FP8（35/36）时，仅支持 "rint" 模式；当 dstType 为 FP4（40/41）时，支持 "rint"、"floor"、"round" 三种模式。

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_mx_quant_with_dual_axis(
    Tensor x,
    str round_mode="rint",
    int dst_type=40,
    int scale_alg=0,
    float dst_type_max=0.0,
) -> (Tensor y1, Tensor mxscale1, Tensor y2, Tensor mxscale2)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| x | 输入 | 待量化的输入张量 | 目的类型为FP4时最后一维必须是偶数，不支持空Tensor | FLOAT16, BFLOAT16 | ND | 2-7 | 支持 |
| round_mode | 输入 | 数据转换的模式 | FP4 支持 {"rint","floor","round"}；FP8 仅支持 {"rint"} | STRING | - | - | - |
| dst_type | 输入 | 目标量化数据类型 | {35:FP8_E5M2, 36:FP8_E4M3FN, 40:FP4_E2M1, 41:FP4_E1M2} | INT64 | - | - | - |
| scale_alg | 输入 | mxscale 计算算法 | 0=OCP共享指数, 1=FP8块缩放, 2=FP4自定义max | INT64 | - | - | - |
| dst_type_max | 输入 | 自定义量化范围上限 | 支持 0.0 和 6.0~12.0，仅 scaleAlg=2 时有效；当前 torch_npu 接口未暴露，默认 0.0 | DOUBLE | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| y1 (y1Out) | 输出 | -1 轴量化结果 | shape 与输入 x 一致；FP4 时最后一维打包为 uint8 | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 | ND | 2-7 | 支持 |
| mxscale1 (mxscale1Out) | 输出 | -1 轴每个分组对应的 scale | rank = rank(x)+1，详见 shape 规则 | FLOAT8_E8M0 (uint8) | ND | 3-8 | 支持 |
| y2 (y2Out) | 输出 | -2 轴量化结果 | shape 与输入 x 一致；FP4 时最后一维打包为 uint8 | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 | ND | 2-7 | 支持 |
| mxscale2 (mxscale2Out) | 输出 | -2 轴每个分组对应的 scale | rank = rank(x)+1，详见 shape 规则 | FLOAT8_E8M0 (uint8) | ND | 3-8 | 支持 |

### mxscale 输出 shape 规则

- `rank(mxscale1Out) = rank(mxscale2Out) = rank(x) + 1`
- `mxscale1Out.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2`
- `mxscale2Out.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2`
- `mxscale1Out.shape[-1] = mxscale2Out.shape[-1] = 2`（interleaved 格式）
- 其他维度与输入 x 一致
- `mxscale2Out` 输出需要对每两行数据进行交织(interleave)处理

举例：输入 x 的 shape 为 `[B, M, N]`，目的数据类型为 FP8 类时：
- `y1` 和 `y2` 的 shape 均为 `[B, M, N]`
- `mxscale1` 的 shape 为 `[B, M, (ceil(N/32)+2-1)/2, 2]`
- `mxscale2` 的 shape 为 `[B, (ceil(M/32)+2-1)/2, N, 2]`

### 数据类型

| x dtype | y1/y2 dtype | mxscale1/mxscale2 dtype |
|---------|-------------|--------------------------|
| float16 | FP4_E2M1 / FP4_E1M2 / FP8_E4M3FN / FP8_E5M2 | FP8_E8M0 (uint8) |
| bfloat16 | FP4_E2M1 / FP4_E1M2 / FP8_E4M3FN / FP8_E5M2 | FP8_E8M0 (uint8) |

### 规则与约束

- 输入 x 维度必须为 2-7 维
- `scale_alg=1` 仅支持 FP8 目标类型（dst_type=35 或 36）
- `scale_alg=2` 仅支持 FP4_E2M1（dst_type=40）
- FP4_E1M2（dst_type=41）仅支持 scale_alg=0
- FP8 目标类型仅支持 `round_mode="rint"`
- FP4 输出时输入最后一维必须为偶数
- `dst_type_max` 仅在 `scale_alg=2` 时有效；当前 torch_npu Python 接口未透传该参数，统一按 0.0 处理

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(x)`（输入维度数） | 2 ~ 7 | NPU API 硬性要求；cases.yaml 实测 2 ~ 5 维 |
| 各维度大小 `dim_i` | 1 ~ 65536 | cases.yaml 实测最小 3、最大 16384 |
| 张量总元素数 | 1 ~ 2^30 | cases.yaml 实测最大约 16M（16×64×16384） |
| `-1 轴` 维度大小 | >= 32 | 每 32 个数一组计算 mxscale1 |
| `-2 轴` 维度大小 | >= 32 | 每 32 个数一组计算 mxscale2 |
| `dst_type` | 35, 36, 40, 41 | 35=FP8_E5M2, 36=FP8_E4M3FN, 40=FP4_E2M1, 41=FP4_E1M2 |
| `scale_alg` | 0, 1, 2 | 0=OCP共享指数, 1=FP8块缩放, 2=FP4自定义max |
| `dst_type_max` | 0.0 | 当前 NPU Python 接口未透传，cases.yaml 统一使用 0.0 |
| 输入 dtype | float16 / bfloat16 | NPU API 不支持 float32 |
| FP4 最后一维 | 偶数 | dst_type=40/41 时最后一维必须为偶数 |

## 4. 精度要求

本算子输出为低精度量化格式的 uint8 字节表示，采用**字节级精确匹配**进行验证。

**验证方式**：
- 将 golden 和 NPU 的输出均转为 uint8 字节表示
- y1/y2 输出：FP8 直接 view 为 uint8；FP4 两两打包为 uint8（低 nibble + 高 nibble << 4）
- mxscale1/mxscale2 输出：FP8_E8M0 直接 view 为 uint8
- 逐字节精确比较，mismatch_count = 0 判定为通过

**通过标准**：

| 指标 | 通过条件 |
|------|---------|
| mismatch_count (y1) | = 0 |
| mismatch_count (mxscale1) | = 0 |
| mismatch_count (y2) | = 0 |
| mismatch_count (mxscale2) | = 0 |
| max_abs_diff | = 0 |

## 5. 标准 Golden 代码 (完整实现见 `golden.py`)

```python
import torch
import numpy

DST_TYPE_MAP = {
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}

def dynamic_mx_quant_with_dual_axis(
    x: torch.Tensor,
    round_mode: str = "rint",
    dst_type: int = 40,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对输入张量同时在 -1 轴和 -2 轴执行 Microscaling (MX) 动态量化。

    Returns:
        (y1, mxscale1, y2, mxscale2): 均为 uint8 字节表示
    """
    # 分别调用单轴量化逻辑，blocksize 固定为 32
    y1, mxscale1 = dynamic_mx_quant(
        x, axis=-1, round_mode=round_mode, dst_type=dst_type,
        blocksize=32, scale_alg=scale_alg, dst_type_max=dst_type_max,
    )
    y2, mxscale2 = dynamic_mx_quant(
        x, axis=-2, round_mode=round_mode, dst_type=dst_type,
        blocksize=32, scale_alg=scale_alg, dst_type_max=dst_type_max,
    )
    return y1, mxscale1, y2, mxscale2
```

完整实现见 `golden.py`。

## 6. 额外信息

### 算子调用示例

```python
import torch
import torch_npu
import cann_bench

# FP4_E2M1 双轴量化 (scale_alg=0)
x = torch.randn(128, 1024, dtype=torch.float16, device="npu")
y1, mxscale1, y2, mxscale2 = cann_bench.dynamic_mx_quant_with_dual_axis(
    x, round_mode="rint", dst_type=40, scale_alg=0)

# FP8_E4M3FN 双轴量化 (scale_alg=1)
x = torch.randn(32, 4096, dtype=torch.float16, device="npu")
y1, mxscale1, y2, mxscale2 = cann_bench.dynamic_mx_quant_with_dual_axis(
    x, round_mode="rint", dst_type=36, scale_alg=1)

# FP4_E2M1 自定义 max 双轴量化 (scale_alg=2, dst_type_max 默认 0.0)
x = torch.randn(64, 1024, dtype=torch.float16, device="npu")
y1, mxscale1, y2, mxscale2 = cann_bench.dynamic_mx_quant_with_dual_axis(
    x, round_mode="rint", dst_type=40, scale_alg=2)
```
