# AttentionResidual 算子 API 描述

## 1. 算子简介

Block Attention Residuals(块级注意力残差)前向聚合算子。它把 Transformer 中固定的加法残差连接替换为**沿深度方向(块轴)的、随输入变化的注意力聚合**:当前层的输入不再是 `x + f(x)`,而是对前面各块输出的加权和,权重由一个可学习的伪查询向量对归一化后的块表示打分、再沿块轴做 softmax 得到。

本算子是该机制的**单步融合核心**(无状态,张量进/张量出),对应推理引擎中把四个算子熔成单 kernel 的形态:RMSNorm(键路径)→ 伪查询投影打分 → 深度 softmax → 加权求和,最后与当前层输入做门控混合。

**主要应用场景**:
- 采用 Attention Residuals / Block AttnRes 结构的大模型推理(残差连接替换件)
- 深度方向特征聚合类结构的融合 kernel 验证

**算子特征**:
- 难度等级:L4(FusedComposite)
- 四张量输入(v_stack, x, w, gamma)+ 两标量属性(alpha, epsilon),单输出
- 融合 RMSNorm、逐 token 深度打分、softmax(块轴)、加权求和与门控混合
- 带宽主导:深度堆叠张量 `v_stack` 的整读是主要数据量

**出处说明(provenance,仅溯源,不构成实现依据)**:机制来自 Kimi 团队
《Attention Residuals》技术报告(arXiv:2603.15031)及其参考仓
MoonshotAI/Attention-Residuals 的 Block AttnRes 伪代码;本文档给出的公式为
本任务的**唯一权威定义**,实现只需依据本文档。

## 2. 算子定义

### 数学公式

记输入堆叠块表示 $V \in \mathbb{R}^{N \times B \times T \times D}$(N 个块、批 B、序列 T、隐维 D),当前层输入 $x \in \mathbb{R}^{B \times T \times D}$,伪查询 $w \in \mathbb{R}^{D}$,RMSNorm 权重 $\gamma \in \mathbb{R}^{D}$,门控标量 $\alpha$ 与数值稳定项 $\varepsilon$:

**步骤 1 — 键路径 RMSNorm(沿 D,逐 (n,b,t) 向量)**:

$$
k_{n,b,t,d} \;=\; \frac{v_{n,b,t,d}}{\sqrt{\tfrac{1}{D}\sum_{d'=1}^{D} v_{n,b,t,d'}^{2} \;+\; \varepsilon}}\;\cdot\;\gamma_d
$$

**步骤 2 — 伪查询打分(逐 token、逐块的标量分数,含 $\sqrt{D}$ 温度)**:

$$
s_{n,b,t} \;=\; \frac{1}{\sqrt{D}}\sum_{d=1}^{D} w_d \, k_{n,b,t,d}
$$

**步骤 3 — 深度 softmax(沿块轴 n)**:

$$
p_{n,b,t} \;=\; \frac{e^{\,s_{n,b,t}}}{\sum_{n'=1}^{N} e^{\,s_{n',b,t}}}
$$

**步骤 4 — 对原始(未归一化)块表示加权求和**:

$$
\mathrm{agg}_{b,t,d} \;=\; \sum_{n=1}^{N} p_{n,b,t}\, v_{n,b,t,d}
$$

**步骤 5 — 门控混合(残差替换语义)**:

$$
y_{b,t,d} \;=\; (1-\alpha)\, x_{b,t,d} \;+\; \alpha\, \mathrm{agg}_{b,t,d}
$$

### 数值精度约定

- 输入张量以存储 dtype(bfloat16 / float16)读入后,**全部中间计算在 float32 中进行**
  (平方均值、点积、softmax、加权求和、门控混合),最终输出转换回输入 dtype。
- softmax 实现应满足数值稳定性(等价于先减去沿 n 的最大值;本公式的数学结果与之相同)。
- **数值路径自由(path-freedom)**:满足精度门槛的任何计算路径均合法(含更高/更低精度
  的中间表示、重排求和、分块累加等);精度门槛是唯一判据。

### 定义域与特殊值

- 本算子的输入定义域为**有限实数**(finite):所有张量取值不含 NaN / ±Inf,
  由 case 的 value_range 保证;实现无需定义非有限输入下的行为。
- $N \ge 2$ **为硬约束**:N=1 时 softmax 退化为常数 1、输出与打分路径无关,
  不属于本算子定义域。
- $\alpha \in [0.05,\, 0.95]$ **为硬约束**:保证门控两支(x 支与聚合支)在定义域内
  恒为有效数据依赖,任何输入都不可被跳过。

## 3. 接口规范

### 算子原型

```python
attention_residual(Tensor v_stack, Tensor x, Tensor w, Tensor gamma, float alpha, float epsilon=1e-6) -> Tensor y
```

### 输入参数

| 参数 | 类型 | dtype | Shape | 描述 |
|---|---|---|---|---|
| v_stack | Tensor | bfloat16 / float16 | [N, B, T, D] | 深度堆叠的块表示(含"当前部分块"在内由调用方堆好) |
| x | Tensor | bfloat16 / float16 | [B, T, D] | 当前层输入(门控混合的残差支) |
| w | Tensor | bfloat16 / float16 | [D] | 可学习伪查询向量 |
| gamma | Tensor | bfloat16 / float16 | [D] | 键路径 RMSNorm 的逐维权重 |
| alpha | float | - | 标量 | 门控系数,定义域 [0.05, 0.95] |
| epsilon | float | - | 标量 | RMSNorm 数值稳定项,默认 1e-6,定义域 [1e-8, 1e-3] |

四个张量输入 dtype 必须一致。

### 输出

| 参数 | Shape | dtype | 描述 |
|---|---|---|---|
| y | [B, T, D] | 与输入一致 | 聚合后的层输入(残差替换结果) |

### 支持范围

| 维度 / 参数 | 支持值 | 备注 |
|---|---|---|
| `N`(块数,v_stack[0]) | **2 ~ 64** | N=1 不在定义域(见上);非对齐值合法 |
| `B`(batch,v_stack[1] / x[0]) | 1 ~ 128 | |
| `T`(序列长,v_stack[2] / x[1]) | 1 ~ 32768 | T=1 为 decode 单步场景 |
| `D`(隐维,v_stack[3] / x[2] / w / gamma) | 256 ~ 8192 | 热点值 {2048, 2560, 3584, 4096, 5120, 7168};非对齐值合法 |
| `v_stack` 总元素数 N·B·T·D | ≤ 2^28 | 控制单 case 输入在约 0.5 GiB(bf16)内 |
| `alpha` | [0.05, 0.95] | 硬约束(见定义域) |
| `epsilon` | [1e-8, 1e-3] | 默认 1e-6,必须 > 0 |
| `v_stack` 取值 | [-1, 1] 典型 | 鲁棒性变体:低至 [-1e-3, 1e-3];高至 `[-64, 64]`（bfloat16）/ `[-32, 32]`（float16）——上限按评测比对器的可达性阶梯标定(高幅值 + 加权求和内部相消会使任意两个正确实现的逐元相对差越过阈值) |
| `x` 取值 | [-1, 1] 典型 | 鲁棒性变体:低至 [-1e-3, 1e-3]、高至 `[-1e3, 1e3]`（混合支无相消,高幅可达） |
| `w` 取值 | [-1, 1] 典型 | 鲁棒性 case 含低幅值变体 |
| `gamma` 取值 | [0.5, 1.5] 典型 | 模拟训练后 RMSNorm 权重尺度;鲁棒性变体低至 [0.01, 0.1]、高至 `[4, 12]`（注:分数对 v 的幅值尺度不变——RMSNorm 消去 v 的尺度;softmax 的分布尖锐度由 gamma/w 幅值驱动） |
| dtype | bfloat16 / float16 | 四张量一致;float32 不在本任务计分范围 |

维度一致性:`v_stack[1:] == x.shape`、`len(w) == len(gamma) == D`。

## 4. 精度要求

按评测框架对 bfloat16 / float16 的生态标准阈值执行(逐元素相对/绝对混合误差)。
golden 采用上述 float32 累加语义;由于全部归约(均值、点积、softmax 归一、
加权求和)均在 float32 中进行且 N ≤ 64、D ≤ 8192,正确实现(含分块/重排累加)
均可稳定达标。
