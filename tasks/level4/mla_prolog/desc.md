# MlaProlog 算子 API 描述

## 1. 背景与动机

`MlaProlog` 是 Multi-Head Latent Attention（MLA）前处理拆分算子，对应 DeepSeek-V2 中 MLA 机制的 Query/Key 投影与位置编码阶段。当前仅考虑 bfloat16 非量化路径，过滤掉量化参数、PagedAttention 缓存格式和数据格式转换（FRACTAL_NZ）等配置。

完整的 MLA 算子包含 13 步计算（投影 + RoPE + Attention + Value 聚合），复杂度较高。MlaProlog 将前半段（Query/Key 的投影与位置编码，共 8 步）拆分为独立算子，覆盖 4 次 CUBE 矩阵乘法、2 次 RMSNorm 归一化和 2 次 RoPE 旋转位置编码。

**主要应用场景**:
- DeepSeek-V2/V3 等采用 MLA 机制的 LLM 推理前处理
- MLA 前处理与后处理（Attention 计算）分离优化
- 推理阶段的 Query/Key 准备（含 W_UK 吸收优化）

**相较于完整 MLA 的拆分优势**:
- 独立优化：前处理的 CUBE+VEC 融合可独立于 Attention 计算调优
- 接口简化：统一输入为 token_x，权重合并减少参数数量
- W_UK 吸收：将 Key 上投影权重吸收到 Query 侧，减少后续在线计算量

**算子特征**:
- 难度等级：L4（FusedComposite）

## 2. 算子定义

### 数学公式

**Query 路径**（三步投影 + RMSNorm + RoPE）：
- $c^Q = \text{RMSNorm}(x \cdot W^{DQ},\, \gamma_{cq},\, \epsilon_{cq})$ — query 压缩表示（下投影 + 归一化）
- $q^C,\, q^R_{\text{raw}} = \text{split}(c^Q \cdot W^{UQ\_QR})$ — query 内容分量 + RoPE 原始分量
- $q^N = q^C \cdot W^{UK}$ — 吸收后的 query（per-head 矩阵乘法）
- $q^R = \text{RoPE}(q^R_{\text{raw}},\, \cos,\, \sin)$ — query 位置编码

**Key 路径**（下投影 + RMSNorm + RoPE）：
- $c_{kv\_raw},\, k^R_{\text{raw}} = \text{split}(x \cdot W^{DKV\_KR})$ — KV 压缩表示 + Key RoPE 原始分量
- $c_{kv} = \text{RMSNorm}(c_{kv\_raw},\, \gamma_{ckv},\, \epsilon_{ckv})$ — 归一化后的压缩 KV
- $k^R = \text{RoPE}(k^R_{\text{raw}},\, \cos,\, \sin)$ — key 位置编码

**RMSNorm 公式**：
$$
\text{RMSNorm}(x, \gamma, \epsilon) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon}}
$$

**RoPE 旋转位置编码**：
$$
\text{RoPE}(\mathbf{x}, \cos, \sin) = \mathbf{x} \odot \cos + \text{rotate\_half}(\mathbf{x}) \odot \sin
$$

其中 $\text{rotate\_half}(\mathbf{x}) = [-\mathbf{x}_{d/2:},\, \mathbf{x}_{:d/2}]$。

### 计算步骤（含数据流）

```
=== Query Path ===
1. c_q_raw = token_x @ W_DQ                           # 参考: BF16输入升FP32，结果保留FP32
2. c_q = RMSNorm(c_q_raw, γ_cq, ε_cq)                 # 参考: FP32→FP32
3. qr = c_q @ W_UQ_QR → split + reshape                # 参考: BF16输入升FP32，结果保留FP32; VEC: split
   → q_c [B, S, N, D] (fp32), q_r_raw [B, S, N, Dr] (fp32)
4. q_n = q_c @ W_UK (per-head batched matmul)           # 参考: BF16输入升FP32，结果保留FP32
5. q_r = RoPE(q_r_raw, cos, sin)                        # 参考: FP32→FP32

=== Key Path ===
6. dkv_kr = token_x @ W_DKV_KR → split                 # 参考: BF16输入升FP32，结果保留FP32; VEC: split
   → ckv_raw [B, S, Hckv] (fp32), kr_raw [B, S, Dr] (fp32)
7. c_kv = RMSNorm(ckv_raw, γ_ckv, ε_ckv)               # 参考: FP32→FP32
8. k_r = RoPE(kr_raw, cos, sin)                          # 参考: FP32→FP32
```

### 融合优化特点

- **8 步融合**: 4 次 CUBE MatMul + 2 次 RMSNorm + 2 次 RoPE 全部融合执行
- **权重合并**: W_UQ + W_QR → W_UQ_QR，W_DKV + W_KR → W_DKV_KR，减少独立矩阵乘法次数
- **W_UK 吸收**: 将 Key 上投影权重预融合到 Query 侧（q^N = q^C · W^UK），减少后续在线计算
- **RoPE 预索引**: sin/cos 表按位置预索引后传入（[B,S,Dr]），避免在线查表开销

## 3. 接口规范

### 算子原型

```python
cann_bench.mla_prolog(Tensor token_x, Tensor w_dq, Tensor w_uq_qr, Tensor w_uk, Tensor w_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, int n_heads, float rmsnorm_epsilon_cq=1e-5, float rmsnorm_epsilon_ckv=1e-5) -> (Tensor query, Tensor query_rope, Tensor c_kv, Tensor k_rope)
```

### 输入参数

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| token_x | Tensor | 是 | bfloat16 | [B, S, He] | 输入 hidden states |
| w_dq | Tensor | 是 | bfloat16 | [He, Hcq] | query 下投影权重 W^DQ |
| w_uq_qr | Tensor | 是 | bfloat16 | [Hcq, N\*(D+Dr)] | query 上投影 + RoPE 权重（W^UQ 和 W^QR 合并） |
| w_uk | Tensor | 是 | bfloat16 | [N, D, Hckv] | key 上投影权重 W^UK（吸收到 query 侧） |
| w_dkv_kr | Tensor | 是 | bfloat16 | [He, Hckv+Dr] | KV 下投影 + Key RoPE 权重（W^DKV 和 W^KR 合并） |
| rmsnorm_gamma_cq | Tensor | 是 | bfloat16 | [Hcq] | c_q 的 RMSNorm 缩放参数 γ |
| rmsnorm_gamma_ckv | Tensor | 是 | bfloat16 | [Hckv] | c_kv 的 RMSNorm 缩放参数 γ |
| rope_sin | Tensor | 是 | bfloat16 | [B, S, Dr] | RoPE 正弦值（已按位置索引） |
| rope_cos | Tensor | 是 | bfloat16 | [B, S, Dr] | RoPE 余弦值（已按位置索引） |
| n_heads | int | 是 | - | 标量 | 注意力头数 N |
| rmsnorm_epsilon_cq | float | 否 | - | 标量 | c_q RMSNorm epsilon，默认 1e-5 |
| rmsnorm_epsilon_ckv | float | 否 | - | 标量 | c_kv RMSNorm epsilon，默认 1e-5 |

### 输出

| 名称 | 类型 | dtype | shape | 描述 |
|------|------|-------|-------|------|
| query | Tensor | bfloat16 | [B, S, N, Hckv] | q^N — 吸收 W_UK 后的 query |
| query_rope | Tensor | bfloat16 | [B, S, N, Dr] | q^R — query 位置编码 |
| c_kv | Tensor | bfloat16 | [B, S, Hckv] | k^C — 归一化后的压缩 KV |
| k_rope | Tensor | bfloat16 | [B, S, Dr] | k^R — key 位置编码 |

### 数据类型

- **输入**: bfloat16（所有 Tensor 参数）
- **输出**: bfloat16（4 个输出 Tensor）
- **内部计算**: BF16输入与权重先升FP32，矩阵乘法、RMSNorm、RoPE中间值均保留FP32，仅四个最终输出转回BF16

### 支持范围

以下是本题选取的规格范围。它不等同于所有 CANN 版本的接口支持域；进行官方实现对照时须固定组件版本并完成下述布局与 cache 适配：

| 维度 / 参数 | 支持值 | 备注 |
|---|---|---|
| `B`（batch，token_x[0]） | 1 ~ 128 | CANN 上限 65536；cases.csv 实测 1 ~ 128 |
| `S`（序列长度，token_x[1]） | **1 ~ 16** | 本题限定 S ∈ [1, 16] 以控制评估成本；不宣称这是所有 CANN 版本的上限 |
| `He`（hidden，token_x[2] / w_dq[0] / w_dkv_kr[0]） | **{6144, 7168, 7680}** | CANN 文档枚举值；cases.csv 实测 6144（compact）、7168（DSv3）、7680 |
| `Hcq`（query 压缩维，w_dq[1] / w_uq_qr[0] / γ_cq） | **固定 1536** | CANN 文档固定值 |
| `N`（注意力头数，n_heads / w_uk[0]） | **{1, 2, 4, 8, 16, 32, 64, 128}** | CANN 文档枚举值；cases.csv 实测 64 / 128 |
| `D`（每头 query/key 内容维，w_uk[1]） | **固定 128** | CANN 文档固定值 |
| `Hckv`（KV 压缩维，w_uk[2] / γ_ckv） | **固定 512** | CANN 文档固定值 |
| `Dr`（RoPE 维度，rope_sin[2] / rope_cos[2]） | **固定 64** | CANN 文档固定值 |
| `w_uq_qr[1]` | = N \* (D + Dr) | cases.csv 实测 24576 = 128 \* (128 + 64） |
| `w_dkv_kr[1]` | = Hckv + Dr | cases.csv 实测 576 = 512 + 64 |
| `rmsnorm_epsilon_cq` | 1e-8 ~ 1e-3 | cases.csv 实测 1e-8 ~ 1e-3（默认 1e-5），必须 > 0 |
| `rmsnorm_epsilon_ckv` | 1e-8 ~ 1e-3 | cases.csv 实测 1e-8 ~ 1e-3（默认 1e-5），必须 > 0 |
| 输入数值范围 | [-1, 1] 典型 | cases.csv 实测 [-1, 1]（19 case）和 [0, 1]（非对称 1 case） |

公开用例分别覆盖典型配置与维度边界，避免同时叠加多个维度上界；支持范围由上表定义。

约束：
- 所有 9 个 Tensor 输入 dtype 必须为 `bfloat16`
- 维度一致性：`w_dq[0] == w_dkv_kr[0] == He`、`w_dq[1] == w_uq_qr[0] == len(γ_cq) == Hcq`、`w_uk[2] == len(γ_ckv) == Hckv`、`rope_sin.shape == rope_cos.shape == [B, S, Dr]`
- `w_uq_qr[1]` 必须能整除为 `N * (D + Dr)`，由调用方保证（算子根据 `n_heads` 与 `w_uk` 推断 D、Dr）

> 说明：CANN 内部 KV cache 维度（Nkv 固定 1、BlockSize ∈ {16, 128}）由调用方在工程侧管理，不属于本算子输入；本算子约束仅覆盖前向计算所需的 9 个 Tensor 与 `n_heads` / `epsilon` 参数。

## 4. 计算流程

```
输入: token_x [B, S, He], 权重矩阵, RMSNorm 参数, RoPE sin/cos 表

=== Phase 1 — Query 下投影与归一化 ===
  1. 下投影: c_q_raw = token_x @ W_DQ             [B, S, Hcq], fp32
  2. RMSNorm: c_q = RMSNorm(c_q_raw, γ_cq, ε_cq) [B, S, Hcq], fp32

=== Phase 2 — Query 上投影与分离 ===
  3. 上投影: qr = c_q @ W_UQ_QR                   [B, S, N*(D+Dr)], fp32
     → reshape: [B, S, N, D+Dr]
     → split: q_c [B, S, N, D], q_r_raw [B, S, N, Dr]

=== Phase 3 — Query W_UK 吸收与 RoPE ===
  4. 吸收: q_n = q_c @ W_UK (per-head batched)    [B, S, N, Hckv], fp32
  5. RoPE: q_r = RoPE(q_r_raw, cos, sin)           [B, S, N, Dr], fp32

=== Phase 4 — Key 下投影与分离 ===
  6. 下投影: dkv_kr = token_x @ W_DKV_KR          [B, S, Hckv+Dr], fp32
     → split: ckv_raw [B, S, Hckv], kr_raw [B, S, Dr]

=== Phase 5 — Key 归一化与 RoPE ===
  7. RMSNorm: c_kv = RMSNorm(ckv_raw, γ_ckv, ε_ckv) [B, S, Hckv], fp32
  8. RoPE: k_r = RoPE(kr_raw, cos, sin)               [B, S, Dr], fp32

输出: query [B, S, N, Hckv], query_rope [B, S, N, Dr],
      c_kv [B, S, Hckv], k_rope [B, S, Dr]
      全部 bfloat16
```

**复杂度**: $O(B \cdot S \cdot (He \cdot Hcq + Hcq \cdot N \cdot (D + Dr) + N \cdot D \cdot Hckv + He \cdot (Hckv + Dr)))$，其中前两项为 Query 路径 CUBE 计算的主要贡献。

## 5. 数值特性

### BF16 精度特点

- bfloat16: ~3 位有效数字，动态范围与 float32 相同（8 位指数），推理场景常用
- 低精度输入先升FP32，矩阵乘法结果保留FP32
- RMSNorm、RoPE沿用FP32中间值，仅四个最终输出转回输入dtype

### RMSNorm 数值稳定性

- RMSNorm 相比 LayerNorm 省去均值中心化，仅依赖均方根（RMS）归一化
- epsilon 参数（默认 1e-5）防止除零，保证数值稳定性
- gamma 缩放参数使用 bf16，归一化计算在 fp32 下进行
- 两处 RMSNorm（c_q 和 c_kv）的 epsilon 可独立配置

### RoPE 数值特性

- RoPE 的 sin/cos 值已按位置预索引并以 bf16 传入
- rotate_half 操作不引入数值误差，仅重排和取负
- RoPE 分别作用于 Query 的低维 Dr 分量和 Key 的 Dr 分量
- Query 的 RoPE 在 per-head reshape 后应用（[B, S, N, Dr]）

## 6. 约束与限制

### 输入约束

- 所有 Tensor 输入 dtype 为 bfloat16
- **S ∈ [1, 16]**（本题覆盖/资源范围；S ≥ 1 排除空序列）
- **He ∈ {6144, 7168, 7680}**（CANN 文档枚举值）
- **D 固定 128，Dr 固定 64，Hcq 固定 1536，Hckv 固定 512**（CANN 文档固定值；权重 shape 必须严格匹配）
- N ∈ {1, 2, 4, 8, 16, 32, 64, 128}，且 N \* (D + Dr) 必须与 w_uq_qr 的第二维一致

### 维度一致性

- w_dq: 第一维 == token_x 第三维（He），第二维 == Hcq
- w_uq_qr: 第一维 == Hcq，第二维 == N \* (D + Dr)
- w_uk: shape 为 [N, D, Hckv]
- w_dkv_kr: 第一维 == He，第二维 == Hckv + Dr
- rmsnorm_gamma_cq: 长度 == Hcq
- rmsnorm_gamma_ckv: 长度 == Hckv
- rope_sin / rope_cos: shape 为 [B, S, Dr]，Dr == w_uq_qr 中推断的 Dr

### 特殊值处理

- rmsnorm_epsilon_cq 和 rmsnorm_epsilon_ckv 默认 1e-5，可分别配置
- W_UK 吸收为 per-head batched matmul：对 N 个头分别执行 [D, Hckv] 矩阵乘法

## 7. Golden 定义

参考计算将输入与权重升为FP32，矩阵乘法、RMSNorm和RoPE中间值保持FP32，仅四个最终输出转换回输入dtype。

```python

import torch
from typing import Tuple

"""
MlaProlog算子Torch Golden参考实现

Multi-Head Latent Attention前处理，融合 Query/Key 投影、RMSNorm 和 RoPE
"""


def rms_norm(x, gamma, epsilon):
    """
    RMSNorm: gamma * x / sqrt(mean(x^2) + epsilon).
    """
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f**2, dim=-1, keepdim=True) + epsilon)
    return (gamma.float() * x_f / rms).to(x.dtype)


def apply_rope(x, rope_cos, rope_sin):
    """
    Apply RoPE with pre-indexed sin/cos.
    """
    cos = rope_cos.float()
    sin = rope_sin.float()
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rotated * sin).to(x.dtype)


def mla_prolog(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-Head Latent Attention 前处理

    Args:
        token_x: [B, S, He] 输入 hidden states, bfloat16
        w_dq: [He, Hcq] query 下投影权重, bfloat16
        w_uq_qr: [Hcq, N*(D+Dr)] query 上投影+RoPE 权重(合并), bfloat16
        w_uk: [N, D, Hckv] key 上投影权重(吸收到 query 侧), bfloat16
        w_dkv_kr: [He, Hckv+Dr] KV 下投影+Key RoPE 权重(合并), bfloat16
        rmsnorm_gamma_cq: [Hcq] c_q 的 RMSNorm gamma, bfloat16
        rmsnorm_gamma_ckv: [Hckv] c_kv 的 RMSNorm gamma, bfloat16
        rope_sin: [B, S, Dr] RoPE 正弦(已按位置索引), bfloat16
        rope_cos: [B, S, Dr] RoPE 余弦(已按位置索引), bfloat16
        n_heads: 注意力头数 N
        rmsnorm_epsilon_cq: c_q RMSNorm epsilon
        rmsnorm_epsilon_ckv: c_kv RMSNorm epsilon

    Returns:
        query: [B, S, N, Hckv] 吸收 W_UK 后的 query, bfloat16
        query_rope: [B, S, N, Dr] query 位置编码, bfloat16
        c_kv: [B, S, Hckv] 归一化后的压缩 KV, bfloat16
        k_rope: [B, S, Dr] key 位置编码, bfloat16
    """
    original_dtype = token_x.dtype
    _low_prec = original_dtype in (torch.float16, torch.bfloat16)
    if _low_prec:
        token_x = token_x.float()
        w_dq = w_dq.float()
        w_uq_qr = w_uq_qr.float()
        w_uk = w_uk.float()
        w_dkv_kr = w_dkv_kr.float()
        rmsnorm_gamma_cq = rmsnorm_gamma_cq.float()
        rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.float()
        rope_sin = rope_sin.float()
        rope_cos = rope_cos.float()

    B, S, He = token_x.shape
    N = n_heads
    Hckv = w_uk.shape[2]
    D = w_uk.shape[1]
    Dr = rope_sin.shape[-1]

    # === Query Path ===
    c_q_raw = torch.matmul(token_x, w_dq)
    c_q = rms_norm(c_q_raw, rmsnorm_gamma_cq, rmsnorm_epsilon_cq)
    qr = torch.matmul(c_q, w_uq_qr)
    qr = qr.reshape(B, S, N, D + Dr)
    q_c = qr[..., :D]
    q_r_raw = qr[..., D:]
    query = torch.einsum("bsnd,ndh->bsnh", q_c, w_uk)
    cos_exp = rope_cos.unsqueeze(2).expand(-1, -1, N, -1)
    sin_exp = rope_sin.unsqueeze(2).expand(-1, -1, N, -1)
    query_rope = apply_rope(q_r_raw, cos_exp, sin_exp)

    # === Key Path ===
    dkv_kr = torch.matmul(token_x, w_dkv_kr)
    ckv_raw = dkv_kr[..., :Hckv]
    kr_raw = dkv_kr[..., Hckv:]
    c_kv = rms_norm(ckv_raw, rmsnorm_gamma_ckv, rmsnorm_epsilon_ckv)
    k_rope = apply_rope(kr_raw, rope_cos, rope_sin)

    if _low_prec:
        query = query.to(original_dtype)
        query_rope = query_rope.to(original_dtype)
        c_kv = c_kv.to(original_dtype)
        k_rope = k_rope.to(original_dtype)

    # einsum / slice 可能留下 view, 而输出契约要求 contiguous (issue #146)；
    # 已连续时 .contiguous() 原样返回, 无额外拷贝
    return (
        query.contiguous(),
        query_rope.contiguous(),
        c_kv.contiguous(),
        k_rope.contiguous(),
    )


def get_input(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    n_heads: int = 1,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    **kwargs,
):
    """把 rope_sin / rope_cos 重建为同一角度的正弦与余弦，满足 sin^2 + cos^2 = 1。

    与 level2/apply_rotary_pos_emb 同一问题：RoPE 的 sin / cos 是同一组位置角 theta 的
    正弦与余弦（proto 也写明"已按位置索引"），对应保范数的平面旋转。通用生成器把它们
    当成两个互相独立的随机张量，(cos, sin) 落在正方形而非单位圆上，不对应任何位置角。

    后果同样有两层：利用该恒等式实现的候选（由 cos 还原 sin、融合复数乘）在真实 RoPE 上
    正确却会与 golden 分叉；且精度阈值按保范数旋转标定，独立随机下失去参照。

    **与 level2 的关键差异：本算子的 rope_cos / rope_sin 是全宽 Dr，不是半宽。**
    level2 的 cos 形状为 (S, D/2)，其 golden 内部会 `cos.repeat(1, 1, 1, 2)` 补齐到全宽，
    因此在输入张量上逐元素取角即可。而本算子的 apply_rope（见本文件 L33-42）拿到的就是
    全宽 Dr，直接逐元素相乘、**没有内部 repeat**：

        x1, x2 = xf.chunk(2, dim=-1);  rotated = cat([-x2, x1])
        y = xf * cos + rotated * sin

    展开后（h = Dr/2、j in [0, h)）：

        y[j]   = x[j]   * cos[j]   - x[j+h] * sin[j]
        y[j+h] = x[j+h] * cos[j+h] + x[j]   * sin[j+h]

    这对 (x[j], x[j+h]) 构成保范数旋转，**当且仅当** cos[j] == cos[j+h] 且
    sin[j] == sin[j+h] —— 即全宽张量的前后半必须是同一角度的复制，正是 RoPE
    `emb = cat([freqs, freqs])` 的约定（desc §7 内嵌的 apply_rope 与此一致）。

    所以这里在**半宽 (..., Dr/2) 上取角**，再复制拼接到全宽；若像 level2 那样在全宽上
    逐元素独立取角，虽然 sin^2+cos^2 = 1 处处成立，整体却不是旋转（实测配对模长比在
    0.023 ~ 1.996 之间），反而与本 PR 的目标背道而驰。

    t 取固定种子的 U(-pi, pi)。用随机角而非真实频率结构，是为了让数据不可被提交侧预测
    写死；kernel 对 sin / cos 只做逐元素乘加，频率结构不影响其算术路径。
    其余 7 个输入原样保留。

    kernel_eval 用输入名 + attrs 作为关键字调用本函数，并用返回值（按 golden 签名的
    Tensor 顺序）同时替换 golden 与候选的输入，故比较公平。

    Returns:
        [token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
         rope_sin, rope_cos]，顺序与 mla_prolog 签名一致。
    """
    unchanged = [
        token_x,
        w_dq,
        w_uq_qr,
        w_uk,
        w_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
    ]
    if not isinstance(rope_sin, torch.Tensor) or not isinstance(rope_cos, torch.Tensor):
        return unchanged
    if rope_sin.shape != rope_cos.shape:
        return unchanged

    # 特殊值压力用例原样放行：含 NaN/Inf 或恒为常数（c20 value_range=[0,0]）时，本就
    # 不是"某个位置角的正弦/余弦"，其用意是特殊值与零值边界覆盖，不应被重建抹掉。
    if not bool(torch.isfinite(rope_sin).all()) or not bool(
        torch.isfinite(rope_cos).all()
    ):
        return unchanged
    if bool((rope_sin == rope_sin.reshape(-1)[0]).all()) and bool(
        (rope_cos == rope_cos.reshape(-1)[0]).all()
    ):
        return unchanged

    dr = int(rope_sin.shape[-1])
    if dr < 2 or dr % 2 != 0:
        # apply_rope 的 chunk(2, dim=-1) 本就要求 Dr 为偶数；奇数时配对约定不成立，
        # 不臆测，保持原样
        return unchanged

    g = torch.Generator().manual_seed(0)  # 固定种子：跨 eval 运行必须可复现
    half_shape = list(rope_sin.shape[:-1]) + [dr // 2]
    theta = (
        torch.rand(half_shape, generator=g, dtype=torch.float64) * 2 - 1
    ) * torch.pi
    # 前后半复制同一角度：apply_rope 无内部 repeat，靠输入自身满足配对约定
    sin_full = torch.cat([torch.sin(theta), torch.sin(theta)], dim=-1)
    cos_full = torch.cat([torch.cos(theta), torch.cos(theta)], dim=-1)
    new_sin = sin_full.to(dtype=rope_sin.dtype, device=rope_sin.device)
    new_cos = cos_full.to(dtype=rope_cos.dtype, device=rope_cos.device)
    return [
        token_x,
        w_dq,
        w_uq_qr,
        w_uk,
        w_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        new_sin,
        new_cos,
    ]
```

## 8. 参考文献

**学术参考**:
- DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model". arXiv:2405.04434.
  - 提出 MLA 机制，MlaProlog 覆盖其 Query/Key 投影与位置编码阶段
- Su, J. et al. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding". Neurocomputation 568.
  - Rotary Position Embedding (RoPE) 数学定义

**官方文档**:
- `torch_npu.npu_mla_prolog` — Ascend NPU MLA 前处理融合算子（本算子的设计参考来源）

**相关 CakeBench Case**:
- `level_4_vector_cube_fused/MultiHeadLatentAttention` — 完整 MLA 算子（MlaProlog 为其前半段拆分）
