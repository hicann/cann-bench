# LactTttChunk 算子 API 描述

## 1. 算子简介

Large-Chunk Test-Time Training（LaCT）层的前向计算。TTT（test-time training）类序列层把"记忆"参数化为一个小型神经网络（快权重），在前向过程中用自监督损失对它做在线梯度更新。LaCT 的关键改动是把更新粒度从逐 token 放大到数千 token 的大 chunk：块内所有 token 的梯度一次性累加成一步更新，使更新计算从一串小矩阵-向量操作变成少数大矩阵乘。本算子的快权重是逐样本、逐头的 SwiGLU MLP $f_W(x) = W_2\,[\mathrm{SiLU}(W_1 x) \odot (W_3 x)]$，可选用 Muon（Newton–Schulz 正交化）处理梯度，并在每步更新后对权重做逐神经元 L2 归一化。

这类层是长上下文 / 长视频 LLM 中替代 softmax attention 的新方向：复杂度 $O(L)$，且状态容量（快权重参数量）远大于线性注意力的矩阵状态。它的前向由"apply（用快权重回答查询）+ update（闭式梯度 + Muon + 归一化）"交替构成，块间严格串行、块内高度并行，是典型的必须精心编排才能吃满算力的算子。

**主要应用场景**：
- TTT / 快权重类长上下文 LLM 与长视频（新视角合成、自回归视频扩散）模型的训练与 prefill
- 需要大状态容量的线性复杂度序列建模（快权重参数量可达隐藏维度的数倍）
- 跨段携带快权重状态（w*_out 作为下一段初始快权重由框架侧衔接）的分段推理

**算子特征**：
- 难度等级：L5（FusedComposite）
- 七输入（q, k, v, w1, w2, w3, lr）四输出（y, w1_out, w2_out, w3_out）
- 融合 L2 归一化、SwiGLU 前向、闭式反向梯度、可选 Muon（Newton–Schulz 5 步）与逐神经元权重归一化
- `chunk_size` 是算子语义的一部分（每块一步权重更新，块数不同结果不同），与只约束实现的分块参数不同
- lr > 0（由评测 value_range 保证），配合权重归一化使快权重轨迹有界，任意随机输入下数值稳定

**为何是 L5**（约为 L4 FusedComposite 的 2 倍难度）：
- **正确 ≠ 快**：按 §2 的逐块公式用最朴素的方式直译（甚至块内逐 token 累加梯度）即可得到完全正确的结果，但性能只能锚定朴素 baseline（预期得分 ~0.5 档）；要逼近硬件下界，必须按 §2 给出的 dual form 把块内 apply / 梯度全部组织成堆叠大矩阵乘，并把 apply 与 update 的计算流水化。所需数学形式（含全部闭式梯度与 Muon 迭代式）已在 §2 完整给出并经数值验证，难点在工程映射而非推导
- **实现完整规格**：实现必须遵循本文定义的分块、快权重更新、Muon 分支和精度契约，不应依赖公开样本的特定配置。
- **状态是三个矩阵**：快权重 $W_1, W_3 \in \mathbb{R}^{D_h \times D}$、$W_2 \in \mathbb{R}^{D \times D_h}$ 逐 (b, h) 驻留并跨块串行更新，Muon 还要在其上做 5 步迭代矩阵乘，数据驻留与依赖管理远难于向量状态的递归算子
- **精度约束严**：快权重与梯度链必须 fp32 保存/累加、Muon 全程 fp32（实测 matmul 操作数整体降 bf16 时误差放大四个数量级、必不达标，见 §4）

## 2. 算子定义

### 数学定义（逐块更新）

每个 (batch, head) 独立。快权重函数（SwiGLU MLP，无偏置）：

$$
f_W(x) = W_2\,\big[\mathrm{SiLU}(W_1 x) \odot (W_3 x)\big], \qquad
W_1, W_3 \in \mathbb{R}^{D_h \times D},\; W_2 \in \mathbb{R}^{D \times D_h}
$$

查询/键先做 L2 归一化（eps 加在范数上）：

$$
\hat{q}_t = \frac{q_t}{\lVert q_t \rVert_2 + 10^{-6}}, \qquad
\hat{k}_t = \frac{k_t}{\lVert k_t \rVert_2 + 10^{-6}}
$$

序列按 `chunk_size` = $C$ 切块 $I_c$，$c = 0, \dots, \lceil L/C \rceil - 1$（末块按实际长度）。对每个块：

- **apply**：$y_t = f_{W}(\hat{q}_t)$，$t \in I_c$
- **update**：以块内所有 token 的负内积损失

  $$
  \mathcal{L} = \sum_{i \in I_c} lr_i \cdot \big(-f_W(\hat{k}_i)^{\mathsf T} v_i\big)
  $$

  求梯度 $g_{W_1}, g_{W_2}, g_{W_3} = \nabla_W \mathcal{L}$（闭式见下），然后

  $$
  \Delta = \begin{cases} \mathrm{Muon}(g) & \text{use\_muon=True} \\ g & \text{use\_muon=False} \end{cases}
  \qquad
  W \leftarrow \mathrm{RowL2Normalize}(W - \Delta)
  $$

**顺序**：`update_first=False`（默认）时块内先 apply（用进入该块前的权重 $W^{(c)}$）再 update 得 $W^{(c+1)}$——因果；`update_first=True` 时先 update 再 apply——块内可见未来（双向场景）。两种顺序下**权重轨迹相同**，只有 y 不同。

**RowL2Normalize**：对每个输出神经元的权重向量做 L2 归一（eps 加在范数上）——$W_1 / W_3$ 的每一行沿 $D$ 维归一，$W_2$ 的每一行沿 $D_h$ 维归一：

$$
\mathrm{RowL2Normalize}(W)[r, :] = \frac{W[r, :]}{\lVert W[r, :] \rVert_2 + 10^{-6}}
$$

**Muon（Newton–Schulz 5 步 zeropower，fp32 计算）**：

$$
X_0 = \frac{g}{\lVert g \rVert_F + 10^{-7}}
$$

若 $g$ 的行数 > 列数则先转置；迭代 5 次（系数 $a = 3.4445,\; b = -4.7750,\; c = 2.0315$）：

$$
A = X X^{\mathsf T}, \qquad X \leftarrow a X + (b A + c A^2)\, X
$$

结束后若曾转置则转置回。$\mathrm{Muon}(g)$ 是 $g$ 的近似正交化（奇异值收敛到 1 附近，实测 NS5 后落在 (0.68, 1.14) 区间，最大 1.134）。

**输出**：$y$ 逐 token 拼接，以及序列末尾的快权重 `w1_out` / `w2_out` / `w3_out`（shape 与输入相同）。

### 块内 dual form（数学参考）

以下堆叠矩阵形式与上面的定义**数学等价**（已数值验证：fp64 下与逐块定义的最大偏差 < 1e-12，覆盖 chunk 大小 1 / 8 / 29 / 64、末块残缺与 use_muon × update_first 全部四种组合），供实现时直接使用。块内把 $\hat{k}_i$ 按行堆叠为 $X \in \mathbb{R}^{C \times D}$、$v_i$ 堆叠为 $V \in \mathbb{R}^{C \times D}$、$\Lambda = \mathrm{diag}(lr_i)$：

**update 的闭式梯度**（全部为大矩阵乘 + 逐元素门）：

$$
H_1 = X W_1^{\mathsf T}, \quad H_3 = X W_3^{\mathsf T}, \quad
U = \mathrm{SiLU}(H_1) \odot H_3 \in \mathbb{R}^{C \times D_h}
$$

$$
\partial\mathrm{Out} = -\Lambda V \in \mathbb{R}^{C \times D}
$$

$$
g_{W_2} = \partial\mathrm{Out}^{\mathsf T}\, U, \qquad
dU = \partial\mathrm{Out}\, W_2
$$

$$
dH_3 = dU \odot \mathrm{SiLU}(H_1), \qquad
dH_1 = dU \odot H_3 \odot \mathrm{SiLU}'(H_1)
$$

$$
g_{W_1} = dH_1^{\mathsf T} X, \qquad g_{W_3} = dH_3^{\mathsf T} X
$$

其中 $\mathrm{SiLU}(x) = x\,\sigma(x)$，$\mathrm{SiLU}'(x) = \sigma(x)\,\big(1 + x\,(1 - \sigma(x))\big)$，$\sigma$ 为 sigmoid。

**apply 的堆叠形式**（$\hat{q}_i$ 堆叠为 $\hat{Q} \in \mathbb{R}^{C \times D}$）：

$$
Y = \big[\mathrm{SiLU}(\hat{Q} W_1^{\mathsf T}) \odot (\hat{Q} W_3^{\mathsf T})\big]\, W_2^{\mathsf T} \in \mathbb{R}^{C \times D}
$$

## 3. 接口规范

### 算子原型

```python
lact_ttt_chunk(Tensor q, Tensor k, Tensor v, Tensor w1, Tensor w2, Tensor w3, Tensor lr, int chunk_size=2048, bool use_muon=False, bool update_first=False) -> (Tensor y, Tensor w1_out, Tensor w2_out, Tensor w3_out)
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| q | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, D] | 查询张量，算子内部沿 D 做 L2 归一化 |
| k | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, D] | 键张量，算子内部沿 D 做 L2 归一化 |
| v | Tensor | 是 | float32/float16/bfloat16 | [B, L, H, D] | 值张量（快权重更新的回归目标） |
| w1 | Tensor | 是 | float32/float16/bfloat16 | [B, H, Dh, D] | 快权重 W1（SwiGLU 门分支），逐样本、逐头独立 |
| w2 | Tensor | 是 | float32/float16/bfloat16 | [B, H, D, Dh] | 快权重 W2（输出投影） |
| w3 | Tensor | 是 | float32/float16/bfloat16 | [B, H, Dh, D] | 快权重 W3（SwiGLU 线性分支） |
| lr | Tensor | 是 | float32/float16/bfloat16 | [B, L, H] | 逐 token 逐头学习率，恒正（评测取值范围 [0.001, 0.05]） |
| chunk_size | int | 否 | - | 标量 | TTT 块大小 C，默认 2048；**影响数学结果**（每块一步权重更新） |
| use_muon | bool | 否 | - | - | True 时 Δ = Muon(g)，默认 False（Δ = g） |
| update_first | bool | 否 | - | - | False 为 apply-then-update（因果，默认），True 为 update-then-apply（双向） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| y | 与 q 一致 | [B, L, H, D] | 输出序列 y_t = f_W(q̂_t) |
| w1_out | 与 q 一致 | [B, H, Dh, D] | 序列末尾的快权重 W1 |
| w2_out | 与 q 一致 | [B, H, D, Dh] | 序列末尾的快权重 W2 |
| w3_out | 与 q 一致 | [B, H, Dh, D] | 序列末尾的快权重 W3 |

### 数据类型

| q/k/v/w1/w2/w3/lr dtype | 输出 dtype | 内部计算 |
|-------------------------|-----------|----------|
| bfloat16 | bfloat16 | fp32（快权重 fp32 驻留，梯度链/Muon fp32） |
| float16 | float16 | fp32 |
| float32 | float32 | fp32 |

### 规则与约束

- 七个 Tensor 输入 dtype 必须一致
- 维度一致性：q/k/v 的 shape 完全一致（[B, L, H, D]）；w1/w3 shape 一致（[B, H, Dh, D]）；w2 为 [B, H, D, Dh]；lr 为 [B, L, H]
- Dh 与 D 允许任意大小关系（评测覆盖 Dh>D、Dh=D、Dh<D——对应 Muon 里"行 > 列先转置"的三种分支）
- lr > 0 由 cases 的 value_range 保证；算子不对违反此约定的输入负责
- `chunk_size` 为正整数，无需整除 L（末块按实际长度做一步更新），chunk_size ≥ L 时整个序列为单块，chunk_size=1 时退化为逐 token TTT
- 快权重与梯度链以 fp32 保存/累加，Muon 全程 fp32（低精度输入场景下这是精度达标的必要条件，见 §4）
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 16 | cases.csv 实测 1 ~ 4 |
| `L`（序列长度） | 16 ~ 16384 | cases.csv 实测 2048 ~ 8192 |
| `H`（头数） | 1 ~ 32 | cases.csv 实测 8 / 16 |
| `D`（头维） | 8 ~ 256（含质数 / 非对齐值） | 公开样本见 cases.csv；完整支持范围以本表为准 |
| `Dh`（快权重隐藏维） | 8 ~ 1024 | cases.csv 实测 256 / 512；完整支持范围以本表为准 |
| `chunk_size` | ≥ 1 | cases.csv 实测 512 / 1024 / 2048 |
| `use_muon` | {True, False} | cases.csv 实测两种均覆盖 |
| `update_first` | {True, False} | cases.csv 实测两种均覆盖 |
| dtype | bfloat16 / float16 / float32 | cases.csv 实测三种均覆盖 |
| `q` / `k` / `v` / `w1` / `w2` / `w3` 取值 | [-1, 1] 典型 | 鲁棒性 case 使用有限幅值变体（最高至 ±6e4、最低至微幅），全部为有限实数 |
| `lr` 取值 | [0.001, 0.05] | 恒正 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**（本算子放宽阈值）：

块间串行的快权重链（最多 L/chunk_size 步）放大误差，Muon 的 Newton–Schulz 迭代对输入扰动敏感。阈值经评测框架 checker 实测确定（fp32 数据通路的误差下限：bf16 MERE≈1e-6 / MARE≈4.4e-2，fp16 MERE≈1.5e-7 / MARE≈9.7e-4，fp32+muon MERE≈4.8e-6 / 正常值域 MARE≈2.0e-3）：

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.001 | 0.005 | 0.02 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。输出与梯度均为多项内积（存在相消），小值域与相消场景由评测框架的兜底标准处理。**达标前提**：快权重与梯度链以 fp32 保存/累加、Muon 全程 fp32——实测把 matmul 操作数整体降为 bf16 时 MERE 达 1.1e-2、且小值域兜底判定必失败。

## 5. 标准 Golden 代码

```python

import torch
from typing import Tuple

"""
LactTttChunk 算子 Torch Golden 参考实现

Large-Chunk Test-Time Training（LaCT）层的前向：快权重是逐样本、逐头的 SwiGLU MLP
f_W(x) = W2 [SiLU(W1 x) ⊙ (W3 x)]，序列按 chunk_size 切块，每块用块内所有 token 的
(k̂, v, lr) 对快权重做一步梯度更新（每个 (b, h) 独立）:
    q̂, k̂ = q, k 沿 D 做 L2 归一化（eps 1e-6）
    块 c 的损失 ℒ_c = Σ_{i∈I_c} lr_i · (−f_W(k̂_i)ᵀ v_i)，g = ∇_W ℒ_c（W1/W2/W3 三个矩阵）
    Δ = Muon(g)（use_muon=True，Newton–Schulz 5 步 zeropower）或 g（use_muon=False）
    W ← RowL2Normalize(W − Δ)（对每个输出神经元的权重向量做 L2 归一，eps 1e-6）
    update_first=False：块内先 y_t = f_{W^{(c)}}(q̂_t) 再更新得 W^{(c+1)}（因果，apply-then-update）
    update_first=True ：先更新再 apply（块内可见未来）
输出 y = f_W(q̂) 逐 token 拼接，以及最终快权重 w1_out / w2_out / w3_out。
输入约定: lr > 0（由 value_range 保证）；末块可残缺（按实际长度处理）。
plain golden 内部升 fp32 计算（bench 语义），输出转回输入 dtype；
lact_ttt_chunk_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为 fp64 真值）。
"""

# Newton–Schulz 5 步 zeropower 的多项式系数（Keller Jordan Muon）
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315
_NS_STEPS = 5


def _row_l2_normalize(w):
    """对每个输出神经元的权重向量（最后一维）做 L2 归一化，eps 加在范数上。"""
    return w / (w.norm(dim=-1, keepdim=True) + 1e-6)


def _muon_zeropower(g):
    """Muon：Newton–Schulz 5 步 zeropower（对最后两维的每个矩阵独立），返回近似正交化的 g。"""
    x = g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)  # 保证 rows <= cols，A = X Xᵀ 更小
    for _ in range(_NS_STEPS):
        a_mat = x @ x.transpose(-2, -1)
        b_mat = _NS_B * a_mat + _NS_C * (a_mat @ a_mat)
        x = _NS_A * x + b_mat @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x


def _lact_ttt_chunk_core(
    q, k, v, w1, w2, w3, lr, chunk_size, use_muon, update_first, compute_dtype
):
    """核心计算：以 compute_dtype 精度执行块循环（apply / update），返回 (y, w1, w2, w3)。"""
    Bsz, L, H, D = q.shape

    q_f = q.to(compute_dtype)
    k_f = k.to(compute_dtype)
    # 沿 D 做 L2 归一化后转为 [B, H, L, D]，便于逐 (b, h) 做 batched matmul
    q_hat = (q_f / (q_f.norm(dim=-1, keepdim=True) + 1e-6)).permute(0, 2, 1, 3)
    k_hat = (k_f / (k_f.norm(dim=-1, keepdim=True) + 1e-6)).permute(0, 2, 1, 3)
    v_bh = v.to(compute_dtype).permute(0, 2, 1, 3)  # [B, H, L, D]
    lr_bh = lr.to(compute_dtype).permute(0, 2, 1).unsqueeze(-1)  # [B, H, L, 1]
    W1 = w1.to(compute_dtype).clone()  # [B, H, Dh, D]
    W2 = w2.to(compute_dtype).clone()  # [B, H, D, Dh]
    W3 = w3.to(compute_dtype).clone()  # [B, H, Dh, D]

    y = torch.empty(Bsz, H, L, D, dtype=compute_dtype, device=q.device)

    def _apply(s, e):
        # y_t = W2 [SiLU(W1 q̂_t) ⊙ (W3 q̂_t)]，t ∈ [s, e)
        x = q_hat[:, :, s:e]  # [B, H, Lc, D]
        h1 = x @ W1.transpose(-2, -1)  # [B, H, Lc, Dh]
        h3 = x @ W3.transpose(-2, -1)
        u = torch.nn.functional.silu(h1) * h3
        y[:, :, s:e] = u @ W2.transpose(-2, -1)  # [B, H, Lc, D]

    def _update(s, e):
        nonlocal W1, W2, W3
        x = k_hat[:, :, s:e]  # [B, H, Lc, D]
        # 前向：h1 = W1 k̂, h3 = W3 k̂, u = SiLU(h1) ⊙ h3, out = W2 u
        h1 = x @ W1.transpose(-2, -1)  # [B, H, Lc, Dh]
        h3 = x @ W3.transpose(-2, -1)
        sig = torch.sigmoid(h1)
        s1 = h1 * sig  # SiLU(h1)
        u = s1 * h3
        # ℒ = Σ_i lr_i · (−outᵢᵀ vᵢ) 的闭式梯度：∂out = −lr ⊙ v
        d_out = -lr_bh[:, :, s:e] * v_bh[:, :, s:e]  # [B, H, Lc, D]
        d_w2 = d_out.transpose(-2, -1) @ u  # dW2 = Σ ∂out uᵀ   [B, H, D, Dh]
        d_u = d_out @ W2  # du = W2ᵀ ∂out     [B, H, Lc, Dh]
        d_h3 = d_u * s1  # dh3 = du ⊙ SiLU(h1)
        d_h1 = d_u * h3 * (sig * (1.0 + h1 * (1.0 - sig)))  # dh1 = du ⊙ h3 ⊙ SiLU'(h1)
        d_w1 = d_h1.transpose(-2, -1) @ x  # dW1 = Σ dh1 k̂ᵀ    [B, H, Dh, D]
        d_w3 = d_h3.transpose(-2, -1) @ x  # dW3 = Σ dh3 k̂ᵀ    [B, H, Dh, D]
        if use_muon:
            d_w1 = _muon_zeropower(d_w1)
            d_w2 = _muon_zeropower(d_w2)
            d_w3 = _muon_zeropower(d_w3)
        # W ← RowL2Normalize(W − Δ)
        W1 = _row_l2_normalize(W1 - d_w1)
        W2 = _row_l2_normalize(W2 - d_w2)
        W3 = _row_l2_normalize(W3 - d_w3)

    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)  # 末块可残缺
        if update_first:
            _update(s, e)
            _apply(s, e)
        else:
            _apply(s, e)
            _update(s, e)

    # 评测框架要求输出 contiguous，permute 后需实体化
    return y.permute(0, 2, 1, 3).contiguous(), W1, W2, W3


def lact_ttt_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    lr: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    update_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LaCT 大块 TTT golden reference（块循环 + 闭式梯度；plain golden = bench：fp32 计算）

    Args:
        q: [B, L, H, D] 查询，算子内部沿 D 做 L2 归一化
        k: [B, L, H, D] 键，算子内部沿 D 做 L2 归一化
        v: [B, L, H, D] 值（快权重更新的回归目标）
        w1: [B, H, Dh, D] 快权重 W1（SwiGLU 门分支）
        w2: [B, H, D, Dh] 快权重 W2（输出投影）
        w3: [B, H, Dh, D] 快权重 W3（SwiGLU 线性分支）
        lr: [B, L, H] 逐 token 逐头学习率，恒正（评测取值范围 [0.001, 0.05]）
        chunk_size: 块大小 C，序列按 C 切块、末块可残缺
        use_muon: True 时 Δ = Muon(g)（Newton–Schulz 5 步 zeropower），False 时 Δ = g
        update_first: False 为 apply-then-update（因果），True 为 update-then-apply（块内可见未来）

    Returns:
        y: [B, L, H, D] 输出序列，dtype 与 q 一致
        w1_out / w2_out / w3_out: 序列末尾的快权重，shape 与输入同、dtype 与 q 一致
    """
    y, w1_o, w2_o, w3_o = _lact_ttt_chunk_core(
        q,
        k,
        v,
        w1,
        w2,
        w3,
        lr,
        chunk_size,
        use_muon,
        update_first,
        torch.float64 if q.dtype == torch.float64 else torch.float32,
    )
    return y.to(q.dtype), w1_o.to(q.dtype), w2_o.to(q.dtype), w3_o.to(q.dtype)


def lact_ttt_chunk_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    lr: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    update_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值），不硬编码 .float()。"""
    return _lact_ttt_chunk_core(
        q, k, v, w1, w2, w3, lr, chunk_size, use_muon, update_first, q.dtype
    )


def lact_ttt_chunk_bench(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    lr: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    update_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 working math for quantized inputs; same output dtype as the API."""
    return lact_ttt_chunk(q, k, v, w1, w2, w3, lr, chunk_size, use_muon, update_first)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

B, L, H, D, Dh = 2, 4096, 8, 128, 256

q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="npu")
k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="npu")
v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="npu")
w1 = torch.randn(B, H, Dh, D, dtype=torch.bfloat16, device="npu")
w2 = torch.randn(B, H, D, Dh, dtype=torch.bfloat16, device="npu")
w3 = torch.randn(B, H, Dh, D, dtype=torch.bfloat16, device="npu")
lr = torch.empty(B, L, H, dtype=torch.bfloat16, device="npu").uniform_(0.001, 0.05)

y, w1_out, w2_out, w3_out = lact_ttt_chunk(
    q, k, v, w1, w2, w3, lr, chunk_size=2048, use_muon=True, update_first=False)
# y.shape: [B, L, H, D]，w*_out 与输入快权重同 shape
```

### 参考文献

- Zhang, T. et al. (2026). "Test-Time Training Done Right". ICLR 2026, arXiv:2505.23884（本算子来源：LaCT 大块 TTT + SwiGLU 快权重 + Muon）
- Sun, Y. et al. (2024). "Learning to (Learn at Test Time): RNNs with Expressive Hidden States". arXiv:2407.04620（TTT 层）
- Jordan, K. et al. (2024). "Muon: An optimizer for hidden layers in neural networks"（Newton–Schulz 5 步 zeropower 与系数来源）
