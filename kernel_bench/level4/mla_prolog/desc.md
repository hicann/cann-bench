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
1. c_q_raw = token_x @ W_DQ                           # CUBE: bf16×bf16→bf16
2. c_q = RMSNorm(c_q_raw, γ_cq, ε_cq)                 # VEC: bf16→(fp32内部)→bf16
3. qr = c_q @ W_UQ_QR → split + reshape                # CUBE: bf16×bf16→bf16; VEC: split
   → q_c [B, S, N, D] (bf16), q_r_raw [B, S, N, Dr] (bf16)
4. q_n = q_c @ W_UK (per-head batched matmul)           # CUBE: bf16×bf16→bf16
5. q_r = RoPE(q_r_raw, cos, sin)                        # VEC: bf16→(fp32内部)→bf16

=== Key Path ===
6. dkv_kr = token_x @ W_DKV_KR → split                 # CUBE: bf16×bf16→bf16; VEC: split
   → ckv_raw [B, S, Hckv] (bf16), kr_raw [B, S, Dr] (bf16)
7. c_kv = RMSNorm(ckv_raw, γ_ckv, ε_ckv)               # VEC: bf16→(fp32内部)→bf16
8. k_r = RoPE(kr_raw, cos, sin)                          # VEC: bf16→(fp32内部)→bf16
```

### 融合优化特点

- **8 步融合**: 4 次 CUBE MatMul + 2 次 RMSNorm + 2 次 RoPE 全部融合执行
- **权重合并**: W_UQ + W_QR → W_UQ_QR，W_DKV + W_KR → W_DKV_KR，减少独立矩阵乘法次数
- **W_UK 吸收**: 将 Key 上投影权重预融合到 Query 侧（q^N = q^C · W^UK），减少后续在线计算
- **RoPE 预索引**: sin/cos 表按位置预索引后传入（[B,S,Dr]），避免在线查表开销

## 3. 接口规范

### 函数签名

```python
def mla_pre(
    token_x,           # [B, S, He] - 输入 hidden states, bfloat16
    w_dq,              # [He, Hcq] - query 下投影权重, bfloat16
    w_uq_qr,           # [Hcq, N*(D+Dr)] - query 上投影+RoPE 权重(合并), bfloat16
    w_uk,              # [N, D, Hckv] - key 上投影权重(吸收到 query 侧), bfloat16
    w_dkv_kr,          # [He, Hckv+Dr] - KV 下投影+Key RoPE 权重(合并), bfloat16
    rmsnorm_gamma_cq,  # [Hcq] - c_q 的 RMSNorm gamma, bfloat16
    rmsnorm_gamma_ckv, # [Hckv] - c_kv 的 RMSNorm gamma, bfloat16
    rope_sin,          # [B, S, Dr] - RoPE 正弦(已按位置索引), bfloat16
    rope_cos,          # [B, S, Dr] - RoPE 余弦(已按位置索引), bfloat16
    n_heads,           # int - 注意力头数 N
    rmsnorm_epsilon_cq=1e-5,   # float - c_q RMSNorm epsilon
    rmsnorm_epsilon_ckv=1e-5,  # float - c_kv RMSNorm epsilon
) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    # query:      [B, S, N, Hckv] - q^N (吸收后的 query), bfloat16
    # query_rope: [B, S, N, Dr]   - q^R (query 位置编码), bfloat16
    # c_kv:       [B, S, Hckv]    - k^C (压缩 KV), bfloat16
    # k_rope:     [B, S, Dr]      - k^R (key 位置编码), bfloat16
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
- **内部计算**: CUBE 矩阵乘法使用 bf16×bf16→bf16；VEC 向量运算（RMSNorm、RoPE）内部使用 fp32，输入输出为 bf16

## 4. 计算流程

```
输入: token_x [B, S, He], 权重矩阵, RMSNorm 参数, RoPE sin/cos 表

=== Phase 1 — Query 下投影与归一化 ===
  1. 下投影: c_q_raw = token_x @ W_DQ             [B, S, Hcq], bf16
  2. RMSNorm: c_q = RMSNorm(c_q_raw, γ_cq, ε_cq) [B, S, Hcq], bf16

=== Phase 2 — Query 上投影与分离 ===
  3. 上投影: qr = c_q @ W_UQ_QR                   [B, S, N*(D+Dr)], bf16
     → reshape: [B, S, N, D+Dr]
     → split: q_c [B, S, N, D], q_r_raw [B, S, N, Dr]

=== Phase 3 — Query W_UK 吸收与 RoPE ===
  4. 吸收: q_n = q_c @ W_UK (per-head batched)    [B, S, N, Hckv], bf16
  5. RoPE: q_r = RoPE(q_r_raw, cos, sin)           [B, S, N, Dr], bf16

=== Phase 4 — Key 下投影与分离 ===
  6. 下投影: dkv_kr = token_x @ W_DKV_KR          [B, S, Hckv+Dr], bf16
     → split: ckv_raw [B, S, Hckv], kr_raw [B, S, Dr]

=== Phase 5 — Key 归一化与 RoPE ===
  7. RMSNorm: c_kv = RMSNorm(ckv_raw, γ_ckv, ε_ckv) [B, S, Hckv], bf16
  8. RoPE: k_r = RoPE(kr_raw, cos, sin)               [B, S, Dr], bf16

输出: query [B, S, N, Hckv], query_rope [B, S, N, Dr],
      c_kv [B, S, Hckv], k_rope [B, S, Dr]
      全部 bfloat16
```

**复杂度**: $O(B \cdot S \cdot (He \cdot Hcq + Hcq \cdot N \cdot (D + Dr) + N \cdot D \cdot Hckv + He \cdot (Hckv + Dr)))$，其中前两项为 Query 路径 CUBE 计算的主要贡献。

## 5. 数值特性

### BF16 精度特点

- bfloat16: ~3 位有效数字，动态范围与 float32 相同（8 位指数），推理场景常用
- 矩阵乘法（CUBE 核心）使用 bf16×bf16→bf16
- 向量运算（VEC 核心，如 RMSNorm、RoPE）内部使用 fp32 计算，输入输出为 bf16

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
- D（每头 query/key 内容维度）需为 **16 的倍数**（CUBE 对齐要求）
- Dr（RoPE 维度）需为**偶数**（rotate_half 对半分割要求）
- n_heads 须为正整数，N \* (D + Dr) 必须与 w_uq_qr 的第二维一致
- S ≥ 1（不支持空序列）

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

```python
import torch


def rms_norm(x, gamma, epsilon):
    """
    RMSNorm: gamma * x / sqrt(mean(x^2) + epsilon).

    Args:
        x: [..., D] - input tensor, bf16
        gamma: [D] - scale parameter, bfloat16
        epsilon: float

    Returns:
        [..., D] - normalized tensor, bf16
    """
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f ** 2, dim=-1, keepdim=True) + epsilon)
    return (gamma.float() * x_f / rms).to(x.dtype)


def apply_rope(x, rope_cos, rope_sin):
    """
    Apply RoPE with pre-indexed sin/cos.

    Args:
        x: [..., Dr] - input tensor, bf16
        rope_cos: [..., Dr] - cosine values, bfloat16
        rope_sin: [..., Dr] - sine values, bfloat16

    Returns:
        [..., Dr] - rotated tensor, bf16
    """
    cos = rope_cos.float()
    sin = rope_sin.float()
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rotated * sin).bfloat16()


def mla_pre_golden(
    token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr,
    rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
    rope_sin, rope_cos, n_heads,
    rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
):
    """
    MlaProlog golden reference.

    Args:
        token_x: [B, S, He], bf16
        w_dq: [He, Hcq], bf16
        w_uq_qr: [Hcq, N*(D+Dr)], bf16
        w_uk: [N, D, Hckv], bf16
        w_dkv_kr: [He, Hckv+Dr], bf16
        rmsnorm_gamma_cq: [Hcq], bf16
        rmsnorm_gamma_ckv: [Hckv], bf16
        rope_sin: [B, S, Dr], bf16
        rope_cos: [B, S, Dr], bf16
        n_heads: int
        rmsnorm_epsilon_cq: float
        rmsnorm_epsilon_ckv: float

    Returns:
        query [B, S, N, Hckv], query_rope [B, S, N, Dr],
        c_kv [B, S, Hckv], k_rope [B, S, Dr] — all bf16
    """
    B, S, He = token_x.shape
    N = n_heads
    Hckv = w_uk.shape[2]
    D = w_uk.shape[1]
    Dr = rope_sin.shape[-1]

    # === Query Path ===
    # Step 1: c_q_raw = token_x @ W_DQ  (bf16)
    c_q_raw = torch.matmul(token_x, w_dq)                           # [B, S, Hcq], bf16
    # Step 2: c_q = RMSNorm(c_q_raw)
    c_q = rms_norm(c_q_raw, rmsnorm_gamma_cq, rmsnorm_epsilon_cq)   # [B, S, Hcq], bf16
    # Step 3: qr = c_q @ W_UQ_QR → split + reshape  (bf16)
    qr = torch.matmul(c_q, w_uq_qr)                                 # [B, S, N*(D+Dr)], bf16
    qr = qr.reshape(B, S, N, D + Dr)                                # [B, S, N, D+Dr]
    q_c = qr[..., :D]                                               # [B, S, N, D], bf16
    q_r_raw = qr[..., D:]                                           # [B, S, N, Dr], bf16
    # Step 4: q_n = q_c @ W_UK (per-head batched matmul, bf16)
    query = torch.einsum('bsnd,ndh->bsnh', q_c, w_uk)               # [B, S, N, Hckv], bf16
    # Step 5: q_r = RoPE(q_r_raw, cos, sin)
    cos_exp = rope_cos.unsqueeze(2).expand(-1, -1, N, -1)
    sin_exp = rope_sin.unsqueeze(2).expand(-1, -1, N, -1)
    query_rope = apply_rope(q_r_raw, cos_exp, sin_exp)              # [B, S, N, Dr], bf16

    # === Key Path ===
    # Step 6: dkv_kr = token_x @ W_DKV_KR → split  (bf16)
    dkv_kr = torch.matmul(token_x, w_dkv_kr)                        # [B, S, Hckv+Dr], bf16
    ckv_raw = dkv_kr[..., :Hckv]                                    # [B, S, Hckv], bf16
    kr_raw = dkv_kr[..., Hckv:]                                     # [B, S, Dr], bf16
    # Step 7: c_kv = RMSNorm(ckv_raw)
    c_kv = rms_norm(ckv_raw, rmsnorm_gamma_ckv, rmsnorm_epsilon_ckv)  # [B, S, Hckv], bf16
    # Step 8: k_r = RoPE(kr_raw, cos, sin)
    k_rope = apply_rope(kr_raw, rope_cos, rope_sin)                   # [B, S, Dr], bf16

    return query, query_rope, c_kv, k_rope
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
