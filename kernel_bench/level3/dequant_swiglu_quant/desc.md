# DequantSwigluQuant 算子 API 描述

## 1. 算子简介

对输入张量 x 进行**反量化（Dequant）→ SwiGLU 激活 → 动态量化（per-token Quant）**
三步融合，输出量化后的结果 y 和 per-token scale。**精简版**仅保留生产推理常用的
5 个参数，**仅动态量化**；不暴露 group_index（MoE 分组路由）、swiglu_mode 变种、
bias、quant_offset、quant_mode（固定为 dynamic per-token）等。

**主要应用场景**
- 大语言模型推理 FFN 层 W8A8 量化加速（int32 + per-token dequant + SwiGLU + 重量化）
- BF16 → INT8 一体化加速（去掉中间 dequant/quant 显存开销）

**算子特征**
- 难度等级：L3（FusedComposite）
- 主输入 `x`，可选 3 个 Tensor（`weight_scale`, `activation_scale`, `quant_scale`）
- 双输出 `(y, scale)`：量化结果 + per-token scale
- 仅支持动态量化（per-token max → scale）
- 输入最后一维 `2H` 必须为偶数

## 2. 算子定义

### 数学公式

设 x ∈ ℝ^[TokensNum, 2H]，按最后一维等分两半：

$$
A = x[\,..., : H],\qquad B = x[\,..., H:\,]
$$

**Step 1 — 反量化**：

$$
\text{dequantOut} = \begin{cases}
\bigl(x \cdot \text{weight\_scale}\bigr) \cdot \text{activation\_scale.unsqueeze(-1)} & x \in \text{int32} \\
x.\text{float}() & x \in \text{bfloat16} \cup \text{float16}
\end{cases}
$$

**Step 2 — SwiGLU 激活**（与官方 API 对齐）：

| `activate_left` | 公式 |
|:--:|---|
| `False`（默认） | `swiglu_out = SiLU(B) * A` |
| `True` | `swiglu_out = SiLU(A) * B` |

**Step 3 — smooth quant 系数（可选）**：

$$
\text{swiglu\_out} = \text{swiglu\_out} \cdot \text{quant\_scale}
$$

`quant_scale` 形状 `[1, H]`，broadcast 到 `[TokensNum, H]`。当其为 None 时跳过。

**Step 4 — 动态量化（per-token）**：

$$
s_i = \max_j |\text{swiglu\_out}_{i,j}|\, /\, 127,\qquad
y_{i,j} = \text{clamp}\bigl(\,\text{round}(\text{swiglu\_out}_{i,j} / s_i),\ -128,\ 127\,\bigr) \in \text{int8}
$$

`scale` 输出每个 token 的 $s_i$（float32）。

## 3. 接口规范

### 算子原型

```python
cann_bench.dequant_swiglu_quant(
    Tensor x,
    Tensor? weight_scale=None,
    Tensor? activation_scale=None,
    Tensor? quant_scale=None,
    bool   activate_left=False,
) -> (Tensor y, Tensor scale)
```

### Shape 变量

| 变量 | 含义 |
|---|---|
| `TokensNum` | token 数（≥ 0）|
| `H` | 输出最后一维大小（`x` 最后一维的一半，> 0）|

### 输入参数

| 参数 | 类型 | shape | dtype | 必选 | 描述 |
|---|---|---|---|:--:|---|
| `x` | Tensor | `[TokensNum, 2H]` | int32 / bfloat16 / float16 | ✓ | 主输入；尾轴必须为偶数 |
| `weight_scale` | Tensor | `[1, 2H]` | float32 | x=int32 时必选 | 权重量化的反量化系数 |
| `activation_scale` | Tensor | `[TokensNum]` | float32 | x=int32 时必选 | per-token 激活反量化系数 |
| `quant_scale` | Tensor | `[1, H]` | float32 | 可选 | smooth 量化系数（仅 float32）|
| `activate_left` | bool | — | — | 可选（默认 False） | False = `SiLU(B)*A`；True = `SiLU(A)*B` |

### 输出

| 参数 | shape | dtype | 描述 |
|---|---|---|---|
| `y` | `[TokensNum, H]` | int8 | 量化后输出 |
| `scale` | `[TokensNum]` | float32 | 每 token 的量化 scale（max abs / 127） |

### 数据类型支持

| `x` 输入 dtype | `weight_scale` / `activation_scale` | `quant_scale` | `y` | `scale` |
|---|---|---|---|---|
| int32 | **必须提供** (float32 / float32) | 可选 float32 | int8 | float32 |
| bfloat16 | **必须为 None** | 可选 float32 | int8 | float32 |
| float16 | **必须为 None** | 可选 float32 | int8 | float32 |

### 规则与约束

- `x.shape[-1]` 必须为**偶数**；`x` 必须是 2D。
- `x` 为 int32 时，`weight_scale` 与 `activation_scale` 必须非 None。
- `x` 为 bfloat16 / float16 时，`weight_scale` 与 `activation_scale` 必须为 None。
- `quant_scale` 仅支持 float32 dtype；fp16 / bf16 在 CANN 850 上调用会运行时报错。
- 输出 `y` 仅 int8，`scale` 仅 float32。
- 该接口仅支持**推理场景**及**图模式**调用。
- 支持芯片：Atlas A2 / Atlas 800I A2 / A200I A2 / Atlas A3 系列。

### 不支持的参数（相对官方 API）

下列参数 `kernel_bench` 精简版**不暴露**，调用底层算子时取默认值：

| 参数 | 默认值 | 用途 / 不暴露原因 |
|---|---|---|
| `quant_mode` | 1（pinned dynamic）| 静态模式在 CANN 850 实测中表现为 identity scale 且 scale 返回未初始化内存，不具备生产可用性 |
| `bias` | None | x 的偏置；可在算子外做 |
| `quant_offset` | None | 量化偏移；生产部署中几乎都为 0 |
| `group_index` | None | MoE 分组路由 |
| `swiglu_mode` | 0 | 0=传统 SwiGLU；1=变种（带 clamp + alpha + bias）|
| `clamp_limit` | 7.0 | 仅变种 SwiGLU 生效 |
| `glu_alpha` | 1.702 | 仅变种 SwiGLU 生效 |
| `glu_bias` | 1.0 | 仅变种 SwiGLU 生效 |

## 4. 精度要求

由于输出 `y` 是 int8（容易出现 ±1 舍入抖动），采用经典量化算子的判定：

| 输出 | 验证方式 | 通过阈值 |
|---|---|---|
| `y` (int8) | 与 golden 逐元素比较，允许 `\|diff\| ≤ 1` 元素占比 < 1e-3 | — |
| `scale` (float32) | 浮点相对误差 | rtol = 1e-3, atol = 1e-5 |

参考[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。

## 5. 标准 Golden 代码

详见同目录 `golden.py`。核心逻辑：

```python
def dequant_swiglu_quant(x, weight_scale=None, activation_scale=None,
                        quant_scale=None, activate_left=False):
    # Step 1: dequant
    if x.dtype == torch.int32:
        d = x.float() * weight_scale.float()
        d = d * activation_scale.float().unsqueeze(-1)
    else:  # bfloat16
        d = x.float()
    # Step 2: swiglu
    A, B = d[..., :d.shape[-1]//2], d[..., d.shape[-1]//2:]
    silu = torch.nn.functional.silu
    out = silu(A)*B if activate_left else silu(B)*A
    # Step 3: smooth quant
    if quant_scale is not None:
        out = out * quant_scale.float()
    # Step 4: dynamic per-token int8 quantize
    s = (out.abs().amax(-1) / 127.0).clamp_min(1e-12)         # [TokensNum]
    y = torch.clamp((out / s.unsqueeze(-1)).round(), -128, 127).to(torch.int8)
    return y, s.to(torch.float32)
```

## 6. 额外信息

### 算子调用示例

```python
import torch, torch_npu

# 路径 A：x = bfloat16，无 weight_scale，纯 SwiGLU + 动态量化
x = torch.randn(2048, 4096, dtype=torch.bfloat16, device="npu")
y, scale = torch_npu.npu_dequant_swiglu_quant(x, quant_mode=1)
# y: [2048, 2048] int8, scale: [2048] float32

# 路径 B：x = int32（W8A8 反量化路径）+ smooth quant
x  = torch.randint(-128, 127, (1024, 4096), dtype=torch.int32, device="npu")
ws = torch.randn(1, 4096, dtype=torch.float32, device="npu")     # weight_scale
as_ = torch.randn(1024,    dtype=torch.float32, device="npu")    # activation_scale
qs = torch.randn(1, 2048, dtype=torch.float32, device="npu")     # quant_scale
y, scale = torch_npu.npu_dequant_swiglu_quant(
    x, weight_scale=ws, activation_scale=as_, quant_scale=qs, quant_mode=1)
```

> 注：`kernel_bench` 暴露的接口不接受 `quant_mode` 参数；调用底层 torch_npu 时
> 由 ref 函数固定传 `quant_mode=1`。

### 参考文档

- 官方 CANN 算子文档：[aclnnDequantSwigluQuant](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/aolapi/context/ops-nn/aclnnDequantSwigluQuant.md)
- torch_npu 绑定：`python -c "import torch_npu; print(torch_npu.npu_dequant_swiglu_quant.__doc__)"`

### 相关算子

- `torch_npu.npu_swiglu_quant` —— 不带反量化的 SwiGLU + 量化
- `torch_npu.npu_grouped_matmul_swiglu_quant_v2` —— 包含 GEMM 的更上层融合算子
