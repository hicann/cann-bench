# DynamicMxQuant 算子 API 描述

## 1. 算子简介

对输入张量执行 Microscaling (MX) 动态量化。在给定的轴 axis 上，按 blocksize 个数分组，计算每组的量化尺度 mxscale，然后将每组元素除以 mxscale 并转换到目标低精度类型（FP4/FP8），输出量化结果 y 和对应的量化 scale。

**主要应用场景**：
- LLM 推理中的权重/激活量化压缩（FP4/FP8 低精度部署）
- MX 格式量化训练（OCP Microscaling 标准兼容）
- 大模型 KV Cache 压缩存储
- 混合精度推理中的动态量化前处理

**算子特征**：
- 难度等级：L3（Quantization）
- 1 输入，2 输出，6 个属性参数
- 支持 ND 格式输入，1-7 维
- 支持 3 种 scale 计算算法（scaleAlg=0/1/2），4 种目标量化格式，3 种舍入模式
- 分组量化：按 blocksize 将指定轴切分为若干 block，每个 block 独立计算 scale
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

### 数学公式

将输入 x 在 axis 维度上按 $k = \text{blocksize}$ 个数分组，一组 k 个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{\text{mxscale}, \{P_i\}_{i=1}^{k}\}$。

- 场景1，当scaleAlg为0时：
  - 将输入x在axis维度上按k = blocksize个数分组，一组k个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = blocksize

  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
  mxscale = 2^{shared\_exp}\\
  P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
  $$

  - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

  - emax: 对应数据类型的最大正则数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

- 场景2，当scaleAlg为1时，只涉及FP8类型：
  - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
  - 找到该块中数值的最大绝对值:
    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$
  - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值（FP8_E4M3FN = 448.0，FP8_E5M2 = 57344.0）:
    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$
  - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
  - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
  - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$
  - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
  - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子（即$S_{ue8m0}^b$），$[d^i]_{i=1}^k$代表块内量化后的数据。

- 场景3，当scaleAlg为2时，只涉及FP4_E2M1类型：
  - 当dstTypeMax = 0.0/6.0/7.0时：
    - 将输入x在axis维度上按k = blocksize个数分组，一组k个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = blocksize：
    $$
    shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果尾数位的高比特前一/两位为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
    $$
    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$
    - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
  - 当dstTypeMax != 0.0/6.0/7.0时（如8.0, 12.0）：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值:
    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$
    - 将FP32映射到目标数据类型可表示的范围内，其中$Amax(DType)$是dstTypeMax传入值:
    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整：
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
      $$
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
    - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

### 舍入模式说明

- **rint**: 银行家舍入（tie to even），0.5 取偶数
- **round**: 标准四舍五入（tie away from zero）
- **floor**: 向负无穷方向舍入

注：当dstType为FP8（35/36）时，仅支持 "rint" 模式；当dstType为FP4（40/41）时，支持 "rint"、"floor"、"round" 三种模式。

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_mx_quant(Tensor x, int axis=-1, str round_mode="rint", int dst_type=40, int blocksize=32, int scale_alg=0, float dst_type_max=0.0) -> (Tensor y, Tensor mxscale)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| x | 输入 | 待量化的输入张量 | 目的类型为FP4时最后一维必须是偶数，不支持空Tensor | FLOAT16, BFLOAT16 | ND | 1-7 | 支持 |
| axis | 输入 | 量化发生的轴 | 取值范围 [-D, D-1]，D 为 x 的维数 | INT64 | - | - | - |
| round_mode | 输入 | 舍入模式 | FP4 支持 {"rint","floor","round"}；FP8 仅支持 {"rint"} | STRING | - | - | - |
| dst_type | 输入 | 目标量化数据类型 | {35:FP8_E5M2, 36:FP8_E4M3FN, 40:FP4_E2M1, 41:FP4_E1M2} | INT64 | - | - | - |
| blocksize | 输入 | 每组量化的元素个数 | 32 的倍数，不能为 0，不超过 1024 | INT64 | - | - | - |
| scale_alg | 输入 | mxscale 计算算法 | 0=场景1, 1=场景2(仅FP8), 2=场景3(仅FP4_E2M1) | INT64 | - | - | - |
| dst_type_max | 输入 | 自定义量化范围上限 | 支持 0.0 和 6.0~12.0，仅 scaleAlg=2 且 blocksize=32 时有效 | DOUBLE | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| y (yOut) | 输出 | 量化结果 | shape 与输入 x 一致（FP4时最后一维减半，packed uint8） | FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2 | ND | 1-7 | 支持 |
| mxscale (mxscaleOut) | 输出 | 每个分组对应的量化尺度 | rank = rank(x)+1，详见 shape 规则 | FLOAT8_E8M0 (uint8) | ND | 2-8 | 支持 |

### mxscale 输出 shape 规则

- `rank(mxscaleOut) = rank(x) + 1`
- `axis_norm = axis if axis >= 0 else axis + rank(x)`
- `num_blocks = ceil(x.shape[axis_norm] / blocksize)`
- `mxscaleOut.shape[axis_norm] = ceil((num_blocks + (num_blocks % 2)) / 2)`（偶数 pad 后折半）
- `mxscaleOut.shape[-1] = 2`（interleaved 格式）
- 其他维度与输入 x 一致
- 当 axis 为非尾轴时，mxscaleOut 需要对每两行数据进行交织(interleave)处理

### 数据类型

| x dtype | y dtype | mxscale dtype |
|---------|---------|---------------|
| float16 | FP4_E2M1 / FP4_E1M2 / FP8_E4M3FN / FP8_E5M2 | FP8_E8M0 (uint8) |
| bfloat16 | FP4_E2M1 / FP4_E1M2 / FP8_E4M3FN / FP8_E5M2 | FP8_E8M0 (uint8) |

### 规则与约束

- `blocksize` 必须为 32 的倍数，最大 1024，不能为 0
- `scale_alg=1` 仅支持 FP8 目标类型（dst_type=35 或 36）
- `scale_alg=2` 仅支持 FP4_E2M1（dst_type=40）
- FP4_E1M2（dst_type=41）仅支持 scale_alg=0
- FP8 目标类型仅支持 `round_mode="rint"`
- FP4 输出时输入最后一维必须为偶数
- `dst_type_max` 仅在 `scale_alg=2` 且 `blocksize=32` 时有效，取值 0.0 或 [6.0, 12.0]
- 确定性计算：aclnnDynamicMxQuantV2 默认确定性实现

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(x)`（输入维度数） | 1 ~ 7 | NPU API 硬性要求；cases.yaml 实测 2 ~ 5 维 |
| 各维度大小 `dim_i` | 1 ~ 65536 | cases.yaml 实测最小 2、最大 16384 |
| 张量总元素数 | 1 ~ 2^30 | cases.yaml 实测最大约 16M（16×64×16384） |
| axis 维度大小 | >= blocksize | axis 维度需至少为 blocksize 大小 |
| blocksize | 32, 64, 128, 256, 512, 1024 | 32 的倍数，不能为 0 |
| axis | [-rank, rank-1] | 负值自动规范化；cases.yaml 实测 -1, 0, 1, 2, 3 |
| dst_type | 35, 36, 40, 41 | 35=FP8_E5M2, 36=FP8_E4M3FN, 40=FP4_E2M1, 41=FP4_E1M2 |
| scale_alg | 0, 1, 2 | 0=OCP共享指数, 1=FP8块缩放(仅FP8), 2=FP4自定义max(仅FP4_E2M1) |
| dst_type_max | 0.0, 6.0~12.0 | 仅 scaleAlg=2 且 blocksize=32 时有效 |
| 输入 dtype | float16 / bfloat16 | NPU API 不支持 float32 |
| FP4 最后一维 | 偶数 | dst_type=40/41 时最后一维必须为偶数 |

## 4. 精度要求

本算子输出为低精度量化格式的 uint8 字节表示，采用**字节级精确匹配**进行验证。

**验证方式**：
- 将 golden 和 NPU 的输出均转为 uint8 字节表示
- y 输出：FP8 直接 view 为 uint8；FP4 两两打包为 uint8（低 nibble + 高 nibble << 4）
- mxscale 输出：FP8_E8M0 直接 view 为 uint8
- 逐字节精确比较，mismatch_count = 0 判定为通过

**通过标准**：

| 指标 | 通过条件 |
|------|---------|
| mismatch_count (y) | = 0 |
| mismatch_count (mxscale) | = 0 |
| max_abs_diff | = 0 |

## 5. 标准 Golden 代码 (完整实现见 `golden.py`)

```python
import torch
import numpy
import copy

DST_TYPE_MAP = {
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}

def dynamic_mx_quant(
    x: torch.Tensor,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 40,
    blocksize: int = 32,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对输入张量执行 Microscaling (MX) 动态量化。

    Returns:
        (y_tensor, mxscale_tensor): 均为 uint8 字节表示
        - y_tensor: FP8 直接 view uint8; FP4 两两打包 uint8
        - mxscale_tensor: FP8_E8M0 的 uint8 表示
    """
    mx_ele_dtype = DST_TYPE_MAP[dst_type]
    # 转为 numpy 计算
    if x.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16
        fp_array = x.to(torch.float32).numpy().astype(bfloat16)
    elif x.dtype == torch.float16:
        fp_array = x.numpy().astype(numpy.float16)
    else:
        fp_array = x.numpy().astype(numpy.float32)

    axis_norm = len(fp_array.shape) + axis if axis < 0 else axis

    # 1. Reshape to blocks: pad axis to blocksize multiple
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(
        fp_array, axis_norm, blocksize)

    # 2. 计算共享指数 (根据 scale_alg 选择算法)
    share_exp = _compute_share_exp(fp_array, axis_norm, mx_ele_dtype,
                                    scale_alg, dst_type_max, orig_shape)

    # 3. 限制 scale 范围 (E8M0: [-127, 127])
    share_exp[share_exp > 127] = float("NaN")
    share_exp[share_exp < -127] = -127

    # 4. 量化元素到目标格式
    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode)

    # 5. 恢复原始 shape
    ele_array = _mx_undo_reshape_to_blocks(
        ele_array, axis_norm, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis_norm + 1)

    # 6. 构建 scale (2^share_exp) 并编码为 E8M0 uint8
    scale_array = 2 ** share_exp
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)

    # 7. Interleave + pack scale 为 uint8
    mxscale_uint8 = _encode_scale_to_uint8(scale_array, axis_norm, fp_array)

    # 8. 编码量化结果为 uint8
    y_tensor = _encode_y_to_uint8(ele_array, mx_ele_dtype, x.shape)

    return y_tensor, torch.from_numpy(mxscale_uint8)
```

完整实现见 `golden.py`。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# FP4_E2M1 量化 (Algorithm 0, 尾轴)
x = torch.randn(128, 1024, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=-1, dst_type=40, blocksize=32, scale_alg=0)

# FP8_E4M3FN 量化 (Algorithm 1, 尾轴)
x = torch.randn(32, 4096, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=-1, dst_type=36, blocksize=32, scale_alg=1)

# FP4_E2M1 自定义 max 量化 (Algorithm 2)
x = torch.randn(64, 1024, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=-1, dst_type=40, blocksize=32, scale_alg=2, dst_type_max=8.0)

# 非尾轴量化 (axis=0)
x = torch.randn(64, 512, dtype=torch.float16, device="npu")
y, mxscale = cann_bench.dynamic_mx_quant(x, axis=0, dst_type=41, blocksize=32, scale_alg=0)
```

