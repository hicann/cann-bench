# AddRmsNormDynamicMxQuant

## 算子简介

`AddRmsNormDynamicMxQuant` 是面向大模型推理/训练量化场景的**融合算子**，依次完成：

1. **Add**：`x = x1 + x2`
2. **RmsNorm**：`y = x * rsqrt(mean(x^2, axis=-1) + epsilon) * gamma + beta`
3. **Dynamic MX Quantization**：对 `y` 按最后一维每 32 个元素分组计算共享 scale，再量化到 FP8/FP4 目标类型

- 难度等级：L3
- 所属类别：Normalization
- 典型应用：FP8/FP4 量化推理、KV Cache 量化、MOE/Transformer 激活量化
- 支持硬件：Ascend 950PR / 950DT

## 算子定义

给定输入 `x1, x2, gamma, beta`：

```
x = x1 + x2
rstd = rsqrt(mean(x^2, axis=-1, keepdim=True) + epsilon)
y = x * rstd * gamma + beta
y_quant, mxscale = dynamic_mx_quant(y, dst_type, block_size=32)
```

Dynamic MX Quantization 核心步骤：

1. 将 `y` 沿最后一维按 `block_size=32` 分块，不足 32 的位置补 0（或 2^-127）。
2. 每块计算最大绝对值 `abs_max`，共享指数 `share_exp = floor(log2(abs_max)) - ele_emax`，其中 `ele_emax` 由目标类型最大可表示值决定。
3. 对块内元素按 `quant = round(y / 2^share_exp)` 量化到目标 FP8/FP4 类型。
4. `share_exp` 以 FP8_E8M0 格式存储为 `mxscale`；由于每字节可存 2 个 scale，最后一维 scale 数量会 pad 到偶数并按 interleave 排列。

公式写法（使用 floor/ceil 函数）：

```
abs_max = max(|y[block]|)
share_exp = floor(log2(abs_max + FP32_MIN_NORMAL * (abs_max == 0))) - ele_emax
quant = clamp(round((y / 2^share_exp) / 2^private_exp) * 2^private_exp, -max_norm, max_norm)
```

## 接口规范

### 函数原型

```python
add_rms_norm_dynamic_mx_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = 40,
    x1_dtype: str = "float16",
    output_rstd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

### 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x1 | float16 / bfloat16 | 主输入，2D `[M, N]` 或 3D `[B, M, N]` |
| x2 | float16 / bfloat16 | 与 x1 同 shape 的输入 |
| gamma | float16 / bfloat16 | shape 为 `[N]` |
| beta | float16 / bfloat16 / None | shape 为 `[N]`，可选 |
| epsilon | float | RMS 稳定系数 |
| scale_alg | int | 0=OCP 标准；1=cuBLAS（仅 FP8 有效，FP4 强制走 OCP） |
| round_mode | str | rint / floor / round |
| dst_type | int | 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN, 40=FLOAT4_E2M1, 41=FLOAT4_E1M2 |
| x1_dtype | str | golden 计算时恢复原始 dtype 用，与 cases.yaml dtype 一致 |
| output_rstd | bool | 算子层面是否输出 rstd；true 时上层 NPU PTA 接口通过将 x1/x2 的 requires_grad 置为 True 来对应输出 rstd；false 时不输出 |

### 输出参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| y | float8_e4m3fn / float8_e5m2 / uint8 (FP4 bit-pattern) | 量化输出；FP8 按 torch.float8 返回，框架做 MERE/MARE 对比；FP4 按 uint8 bit-pattern 返回（每字节存一个 4-bit 量化值） |
| x | float16 / bfloat16 | 残差 `x1 + x2` |
| mxscale | uint8 | FP8_E8M0 bit-pattern，shape 为 `rank(x1)+1` |
| rstd | float32 | 算子层面受 output_rstd 控制；上层 NPU PTA 接口没有 output_rstd 参数，通过输入 x1/x2 的 requires_grad 来对应；output_rstd=true 时返回有效 rstd，false 时返回空张量 |

## 支持范围

- 维度：2D、3D（最后一维为 hidden_size）
- x1/x2/gamma/beta 数据类型：FLOAT16、BFLOAT16
- y 目标类型：FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2
- mxscale 数据类型：FP8_E8M0（以 uint8 存储）
- FP4 输出要求最后一维为偶数
- cann-bench 算子接口通过 `output_rstd` 属性控制是否输出 rstd；上层 NPU PTA 接口没有 `output_rstd` 参数，`output_rstd=true` 时通过将输入 `x1/x2` 的 `requires_grad` 置为 `True` 来对应输出 rstd；`output_rstd=false` 时不输出

## Golden 实现

`golden.py` 提供 CPU 参考实现，分两个阶段：

1. `add_rms_norm`：使用 PyTorch float32 计算 `x = x1 + x2`、`rstd`、`y = x * rstd * gamma + beta`。
2. `dynamic_mx_quant`：使用 numpy + ml_dtypes/en_dtypes 模拟 MX 量化，按 32 块计算 share_exp、量化到目标 FP8/FP4 类型，并生成 mxscale。

由于 cann-bench 默认将 golden 输入升精度到 FP64， golden 函数通过 `x1_dtype` 参数将中间结果 cast 回原始 dtype，再执行量化，从而与 NPU 的 FP16/BF16 计算路径对齐，保证数值一致。

为适应 FP4 打包/解包差异，golden 与 NPU 提交实现将 FP4 的 `y` 以 uint8 bit-pattern 返回（FP4 解包为每元素 1 字节），`mxscale` 同样以 uint8 bit-pattern 返回。FP8 的 `y` 则按 torch.float8_e4m3fn / torch.float8_e5m2 返回，由框架按浮点精度标准做 MERE/MARE 对比。

## 评测说明

- 共 20 个 case，覆盖 2D/3D、FLOAT16/BFLOAT16、FP8/FP4、对齐/非对齐 hidden_size、有/无 beta、不同 scale_alg/round_mode。
- 性能基线通过 `test_baseline_perf.py` 在 NPU 上采集，取 p50 耗时。
- 框架精度对比时，FP8 的 `y` 按 torch.float8 做 MERE/MARE 对比；FP4 的 `y` 与 `mxscale` 按 uint8 绝对差值 ≤1 进行比对；`x` 按 float16/bfloat16 相对误差标准；`rstd` 按 float32 精度标准对比，`output_rstd=False` 时返回空张量并跳过对比。
