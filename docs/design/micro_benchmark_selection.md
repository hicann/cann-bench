# CANN-Bench Micro-Benchmark 选取策略说明

> 下一版 60 项大集及 16 项 micro 的选题更新见 [2026-09-24 名单与名称迁移](next_suite_20260924.md)。以下保留原 53 项版本的选取策略与历史名单，不代表新版运行配置已经启用。

> 版本: v1.0 | 日期: 2026-06-05 | 状态: 定稿

---

## 1 背景

CANN-Bench 当前评测集包含 **53 个算子**，分布在 4 个难度层级：

| Level | 数量 | 含义 | 典型特征 |
|-------|------|------|----------|
| L1 | 8 | 单核基元算子 | 逐元素激活、TensorList 批量操作 |
| L2 | 16 | 双核复合算子 | 归一化、优化器、索引、量化 |
| L3 | 21 | 多核融合算子 | 卷积、MoE、量化融合、排序、检测 |
| L4 | 8 | 整图级算子 | 注意力变体、RNN、MoE 全融合 |

完整运行 53 个算子的评测周期较长，不利于快速回归验证和增量开发迭代。本文档定义一个 **Micro-Benchmark 子集**（16 个算子），在保持评测覆盖度的前提下大幅缩短评测周期。

---

## 2 选取目标

| 目标 | 优先级 | 说明 |
|------|--------|------|
| **范式全覆盖** | P0 | 每种计算范式至少有 1 个代表算子，无遗漏 |
| **范式零重叠** | P0 | 任意两个入选算子不属于同一计算范式，不可分解为已有范式的组合 |
| **热点路径代表** | P1 | 优先选取 LLM/MoE 推理热点路径上的算子 |
| **硬件通路全覆盖** | P1 | 每种 NPU 硬件执行通路（向量核、float Cube核、INT8 Cube核、归约引擎、DMA）至少有 1 个算子施压 |
| **Level 比例均衡** | P1 | 各 Level 名额占比与原始占比偏差 ≤ ±10% |
| **类别占比均衡** | P1 | 各功能类别（量化、归一化、注意力等）占比与原始占比偏差 ≤ ±2× |
| **评测叙事性** | P2 | 入选算子之间形成可对比的评测链，能回答"融合收益"、"量化闭环"等关键问题 |

> **v2.0 新增**：将"范式零重叠"从严化——不仅范式名不同，核心计算也不可分解为已入选范式的组合。这是 mha、gqa、sfa 被移出的根本原因。
>
> **v3.0 新增**：增加"类别占比均衡"约束，防止某一功能类别（如量化）占据过多名额。量化相关算子在原始集合中占比 9.4%（5/53），入选比例不应超过 2× 即 18.75%（3/16）。

---

## 3 范式分类体系

53 个算子按计算范式分为 **38 种细分范式**：

| 范式编号 | 范式名称 | 硬件压力点 | 全部算子 |
|----------|----------|------------|----------|
| P1 | 逐元素激活（单输入） | 向量核单流 | exp, gelu, mish, sigmoid |
| P2 | 门控激活（多输入/分裂） | 向量核多流 + 分裂 | swi_glu, masked_scale |
| P3 | TensorList 批量调度 | 批量 kernel 发射引擎 | foreach_addcdiv_scalar, foreach_norm |
| P4 | 简单归约→归一化 | 归约引擎 + 向量核除法 | rms_norm, group_norm |
| P5 | 复杂归约→归一化 | 归约引擎(max+sum) + 向量核exp+除法 | softmax |
| P6 | 旋转位置编码 | 向量核三角函数 + 复数旋转 | apply_rotary_pos_emb |
| P7 | 优化器多输入复合 | 高 IO 吞吐 + 多 tensor 依赖 | apply_adam_w |
| P8 | per-token 动态量化 | 归约引擎(per-token max) + 类型转换 | dynamic_quant |
| P9 | 随机索引读（scatter-read） | DMA 非连续读取 | gather |
| P10 | 随机索引写（scatter-write） | DMA 非连续写入 + 冲突处理 | scatter |
| P11 | 批量 float Cube 核调度 | float Cube 核多小矩阵并行 | grouped_matmul |
| P12 | 反量化→激活→量化融合 | 中间精度转换 + 变量消除 | dequant_swiglu_quant |
| P13 | Add+Norm+Quant 三域融合 | 跨域(残差/归一化/量化)融合 | add_rms_norm_dynamic_quant |
| P14 | 排序+TopK 选择 | 控制流密集排序核 | top_k, arg_max |
| P15 | 跨步数据搬运 | DMA 跨步读写 | transpose, strided_slice |
| P16 | 算子内迭代循环 | cache 驻留 + 收敛稳定性 | mhc_sinkhorn |
| P17 | 卷积（im2col + Cube 核） | 数据重排 + 单大矩阵 Cube | conv_2d, depthwise_conv_2d |
| P18 | 梯度卷积 | 反向传播数据流 + Cube 核 | conv_3d_backprop_filter |
| P19 | 自适应池化 | 不均匀分组归约 | adaptive_avg_pool_3d |
| P20 | 形态学膨胀 | 局部最大值 + 非标准邻域 | dilation_2d |
| P21 | 检测条件分支 | IoU 比较 + 过滤循环 | nms |
| P22 | ROI 特征提取 | 非均匀尺寸插值 + 区域索引 | roi_align |
| P23 | 去重 | 哈希/查找表间接访存 | unique |
| P24 | 累积归约（前缀扫描） | 扫描式前缀归约 | cummin |
| P25 | 分段归约 | segment_id 驱动组归约 | unsorted_segment_sum |
| P26 | 整数运算 | 整数核 | gcd |
| P27 | 逐元素数学（双输入） | 向量核双流 | maximum |
| P28 | INT8 Cube 核 + 输出反量化 | INT8 Cube 核 + 类型转换输出 | quant_matmul, weight_quant_batch_matmul |
| P29 | Engram 门控融合 | 双 RMSNorm + 门控 + Conv + 残差 | engram_gate_fusion |
| P30 | MoE 门控选择融合 | Softmax + TopK 融合 | moe_gating_top_k_softmax |
| P31 | MoE 路由重排 | token 重分布 + AlltoAll | moe_re_routing, moe_finalize_routing |
| P32 | 标准注意力 | Cube 核 QKV + FlashAttention tiling | mha |
| P33 | 分组 KV 注意力 | KV head 共享 + Cube 核调度 | gqa |
| P34 | 稀疏注意力 | 稀疏 KV 选择 + tiling | sparse_flash_attention |
| P35 | KV 低秩压缩注意力（注意力计算） | 低秩解压 + 缩放点积注意力 | mla |
| P35a | KV 低秩压缩注意力（前处理编排） | RMSNorm + 投影 + RoPE 九路编排 | mla_prolog |
| P36 | RNN 循环序列 | 序列依赖数据流编排 | gru, lstm |
| P37 | MoE 全融合 | GMM + 激活 + 量化四步合一 | grouped_matmul_swiglu_quant |
| P38 | 插值/采样 | 坐标映射 + 权重插值 | resize_bilinear, grid_sampler_3d |

> 53 个算子映射到 38 种细分范式。其中 P32/P33/P34 可分解为已有范式的组合（详见 §4.2 第 6 轮分析），最终 16 种入选范式均为不可分解的基本范式。

---

## 4 选取过程

### 4.1 四种候选策略

本文档分析了四种选取策略，最终采用**混合策略**。

#### 策略 A：计算类型全覆盖

> 每种计算类型取 1 个代表，确保 NPU 各硬件单元都有压力。

- 优点：硬件覆盖度最高
- 缺点：不考虑热点路径，部分算子实用价值低

#### 策略 B：大模型热点路径优先

> 选取 LLM/MoE 推理中最频繁执行的算子，benchmark 结果直接反映真实模型性能。

- 优点：与模型性能强相关
- 缺点：CV、检测、序列等非 LLM 类型缺失，硬件覆盖不全

#### 策略 C：融合难度梯度

> 按 L1→L4 依次选取，构成"单核→双核→多核→整图"难度梯度。

- 优点：评测编译器/算子库融合能力的梯度清晰
- 缺点：L1 简单算子占比偏高，对推理瓶颈不敏感

#### 策略 D：访存模式多样性

> 选取涵盖不同访存模式的算子，全面压力测试 DMA/Cache 数据通路。

- 优点：内存子系统评测最全面
- 缺点：不考虑计算类型分布

#### 最终策略：A（类型覆盖）为主 + B（热点路径）为辅

- 以范式全覆盖为硬约束（P0）
- 以范式零重叠（含组合分解检查）为硬约束（P0）
- 以类别占比均衡（≤ 2× 原始比例）为约束（P1）
- 在每个范式内优先选热点路径算子（P1）
- 确保每种 NPU 硬件通路至少有 1 个算子施压（P1）
- 通过交叉审查消除范式重叠，形成叙事链（P2）

### 4.2 逐轮筛选与调整（10 轮）

#### 第 1 轮：初始选取（15 个）

基于策略 A+B，每个核心范式取 1 个代表：

| # | 算子 | Level | 范式 |
|---|------|-------|------|
| 1 | gelu | L1 | 逐元素激活 |
| 2 | swi_glu | L1 | 门控激活 |
| 3 | rms_norm | L2 | 简单归约→归一化 |
| 4 | softmax | L2 | 复杂归约→归一化 |
| 5 | apply_adam_w | L2 | 优化器多输入复合 |
| 6 | apply_rotary_pos_emb | L2 | 旋转位置编码 |
| 7 | gather | L2 | 随机索引读 |
| 8 | conv_2d | L3 | 卷积 |
| 9 | grouped_matmul | L3 | 批量 Cube 核调度 |
| 10 | dequant_swiglu_quant | L3 | 反量化→激活→量化融合 |
| 11 | top_k | L3 | 排序+TopK 选择 |
| 12 | nms | L3 | 检测条件分支 |
| 13 | transpose | L3 | 跨步数据搬运 |
| 14 | mhc_sinkhorn | L3 | 算子内迭代循环 |
| 15 | mha | L4 | 标准注意力 |

#### 第 2 轮：Level 分布修正

初始方案 L1 严重不足（1/15 = 7%，原始 15%），L3 偏重。按原始比例加权调整：

- L1 补入 **foreach_addcdiv_scalar**（TensorList 批量调度范式，全集合唯一）
- L4 补入 **gqa** + **lstm** → 形成 mha→gqa→sfa 注意力梯度
- 移出 nms（条件分支可由 mhc_sinkhorn 迭代循环替代）

#### 第 3 轮：范式重叠消除（第一类：同类范式重叠）

审查发现两处同一范式内的重叠：

| 重叠组 | 问题 | 决策 |
|--------|------|------|
| gelu ↔ swi_glu | 同属逐元素激活范式，swi_glu（门控激活）覆盖更全面 | **移出 gelu**，swi_glu 独占该范式 |
| conv_2d ↔ grouped_matmul | 同属 Cube 核矩阵乘范式，conv_2d 不在 LLM 热点路径且 im2col 数据重排可由 transpose 间接覆盖 | **移出 conv_2d** |

移出 gelu 和 conv_2d 后，补入两个高价值遗漏算子：

| 补入算子 | 补入范式 | 理由 |
|----------|----------|------|
| **dynamic_quant** (L2) | per-token 动态量化(P8) | 全集合唯一的量化入口范式；与 dequant_swiglu_quant 形成量化闭环 |
| **add_rms_norm_dynamic_quant** (L3) | Add+Norm+Quant 三域融合(P13) | DeepSeek decode 路径；与 rms_norm 形成"单步 vs 融合"对照 |

#### 第 4 轮：冗余消除（apply_* 多输入复合重叠）

审查发现 apply_adam_w（优化器）与 apply_rotary_pos_emb（位置编码）同属"apply_* 多输入复合运算"范式，压同一个硬件特征（高 IO 吞吐 + 多 tensor 依赖），且 apply_adam_w 是训练优化器，不在推理热点路径。

- **移出 apply_adam_w**
- foreach_addcdiv_scalar 独立代表 TensorList 批量调度范式，不再依赖"对照 apply_adam_w"的理由

#### 第 5 轮：扩展至 16 个，补入随机写范式

当前 15 个算子覆盖 15 种范式，但**随机写（scatter-write）范式**缺失。写入模式光谱不完整：

```
sequential → strided → sorted partial → random index-driven  ← 缺失
(softmax输出)  (transpose)  (top_k)          (?)
```

扩展至 16 个，补入 **scatter**（L2），理由：

- 随机写比随机读更难优化（冲突写入、原子性、写合并），更容易暴露 NPU 写端瓶颈
- scatter 具有独特正确性挑战（重复索引处理、写入顺序依赖）
- 补全写模式光谱后，4 种写入模式完整覆盖

#### 第 6 轮：组合分解检查——mha、gqa、sfa 替换

严化"范式零重叠"定义：**不仅要求范式名称不同，还要求算子核心计算不可分解为已入选算子范式的组合**。

审查发现三个注意力算子均可分解：

| 算子 | 分解 | 已被覆盖 |
|------|------|----------|
| **mha** | QKV投影(matmul) + Softmax + 输出投影(matmul) | grouped_matmul(P11) + softmax(P5) + grouped_matmul(P11) |
| **gqa** | 同 mha，仅 KV head 数减少 | 同上（参数配置差异，非范式差异） |
| **sfa** | sparseIndices选择 + QKV(matmul) + Softmax + 输出(matmul) | scatter(P10) + grouped_matmul(P11) + softmax(P5) |

决策：

- **移出 mha**，替换为 **mla_prolog** (L4)
  - mla_prolog 是全集合唯一的"9-input 多步前处理编排"范式（RMSNorm→Q投影→K投影→RoPE→9路数据流）
  - 涉及归约引擎(Cube核投影) + 向量核(RoPE三角函数) + 归约引擎(RMSNorm) 三域并发
  - 不可分解：9-input 编排模式是独特的系统级挑战

- **移出 gqa**
  - gqa 的唯一特征"KV head 共享"是 mha 的调度优化变体，非独立范式
  - mha 移出后 gqa 成为无 baseline 对照的孤立项

- **移出 sfa**，替换为 **mla** (L4)
  - mla 的 KV 低秩解压+注意力计算范式虽包含 Cube核 + softmax 组合，但"on-the-fly KV 解压"是全集合唯一的：注意力核内部嵌套低秩投影步骤
  - mla_prolog + mla 形成**完整 MLA 流管线**：前处理→注意力计算，评测叙事性远强于 sfa 单打独斗

#### 第 7 轮：补入 INT8 Cube 核范式

移出 mha、gqa、sfa 后，审查发现**INT8 Cube 核硬件通路**无任何算子覆盖。现有 15 个算子的 Cube 核路径全部是 float16/bfloat16，INT8 量化推理场景评测空白。

补入 **quant_matmul** (L3)，理由：

- 全集合唯一使用 INT8 Cube 核的算子——输入 int8 量化矩阵，Cube 核执行整数矩阵乘，输出反量化为 float16/bf16
- INT8 Cube 核是 NPU 上一条独立的物理计算通路（8bit 累加器、独立调度器、2× 吞吐密度），与 float Cube 核完全不同

#### 第 8 轮：最终验证（v2.0 定稿）

16 个算子逐一通过三项检查：

1. **范式唯一性**：每个算子代表一种不可分解的基本范式
2. **硬件通路覆盖**：向量核、float Cube核、INT8 Cube核、归约引擎、DMA 均有施压
3. **Level 分布**：所有 Level 偏差 ≤ ±3%

#### 第 9 轮：量化占比优化（v3.0）

审查发现量化相关算子占比过高：

| | 纯量化算子 | 含量化组件 | 合计 |
|--|-----------|-----------|------|
| v2.0 入选 | 3 (DynamicQuant, QuantMatmul, DequantSwigluQuant) | 1 (AddRmsNormDynamicQuant) | **4/16 = 25%** |
| 原始占比 | 5/53 = 9.4% | — | 9.4% |
| 偏差 | 3.2× | — | **2.7× — 超出 2× 上限** |

分析各量化算子的可替换性：

| 量化算子 | 独特硬件通路 | 移出后果 | 可替换性 |
|----------|------------|----------|----------|
| DynamicQuant | 归约引擎(per-token max) + 类型转换 | 量化叙事失去"入口"，但 QuantMatmul 本身包含入口量化逻辑 | ⚠️ **最高** |
| QuantMatmul | INT8 Cube 核（全集合唯一） | INT8 推理硬件通路空白 | ❌ 不可替换 |
| DequantSwigluQuant | float Cube核 + 向量核融合 | 量化叙事失去"出口" | ❌ 不可替换 |

移出 **DynamicQuant**，理由：

1. per-token max 归约与 rms_norm 的归约范式有结构性重叠（都是"沿维度提取统计量→逐元素缩放"）
2. QuantMatmul 本身已包含量化入口逻辑（INT8输入→Cube核→反量化输出）
3. 移出后量化叙事从三步变为两步，但核心对比维度（INT8 vs float Cube核）保留

补入 **cummin** (L2)，理由：

- 前缀扫描归约(P24) 是全集合唯一的**顺序依赖模式**：output[i] = min(input[0..i])
- 与 rms_norm(全局广播依赖) 和 top_k(无顺序依赖) 构成**归约依赖光谱**
- mhc_sinkhorn 的迭代依赖是双向收敛，与 cummin 的单向递增依赖本质不同，无法间接覆盖

移出后量化占比修正：

| | 纯量化算子 | 含量化组件 | 合计 |
|--|-----------|-----------|------|
| v3.0 入选 | 2 (QuantMatmul, DequantSwigluQuant) | 1 (AddRmsNormDynamicQuant) | **3/16 = 18.75%** |
| 原始占比 | 5/53 = 9.4% | — | 9.4% |
| 偏差 | — | — | **2.0× — 符合 ≤2× 上限** ✅ |

#### 第 10 轮：访存读写闭环（v3.0）

当前 16 个算子中 scatter(P10) 独占随机访存范式，但缺少读端对照。补入 **gather** 可形成 scatter-read + scatter-write 闭环，直接回答"NPU DMA 读写端是否对称"这一关键问题。

需腾出 1 个名额。审查各算子可替换性：

| 算子 | 不可替换性 | 理由 |
|------|-----------|------|
| swi_glu, foreach_addcdiv_scalar, rms_norm, softmax, apply_rotary_pos_emb, cummin, scatter, grouped_matmul, quant_matmul, dequant_swiglu_quant, add_rms_norm_dynamic_quant, top_k, transpose, mla_prolog, mla | 🔒 不可替换 | 各自代表唯一范式，移出导致范式空白 |
| **mhc_sinkhorn** | ⚠️ **最低** | 算子内迭代循环(P16)独特，但**不在任何主流热点路径**（仅 DeepSeek mHC 模块）；归约依赖光谱退为两点(broadcast + prefix-scan)仍具对比价值 |

移出 **mhc_sinkhorn**，补入 **gather**，理由：

1. mhc_sinkhorn 不在主流热点路径，实用价值有限
2. gather + scatter 的**访存读写闭环**叙事价值高于 mhc_sinkhorn 的孤立迭代叙事：
   - scatter-read vs scatter-write 直接揭示 DMA 读写端对称性
   - 单靠 scatter 无法回答"随机读与随机写性能差异有多大"
3. 归约依赖光谱退为两点（broadcast → prefix-scan）仍有效，第三点(iterative)可由扩展候补补入

---

## 5 最终选取结果

### 5.1 算子清单

| # | 算子 | Level | 核心范式 | 热点路径价值 |
|---|------|-------|----------|-------------|
| 1 | **SwiGLU** | L1 | 门控激活(P2) | LLaMA/DeepSeek FFN |
| 2 | **ForeachAddcdivScalar** | L1 | TensorList 批量调度(P3) | 优化器基元 / 批量 kernel 发射 |
| 3 | **RmsNorm** | L2 | 简单归约→归一化(P4) | LLM pre-norm |
| 4 | **Softmax** | L2 | 复杂归约→归一化(P5) | 所有 Transformer attention |
| 5 | **ApplyRotaryPosEmb** | L2 | 旋转位置编码(P6) | RoPE 推理路径 |
| 6 | **Cummin** | L2 | 前缀扫描归约(P24) | 顺序依赖链独特范式 |
| 7 | **Scatter** | L2 | 随机索引写(P10) | 写端随机访存 |
| 8 | **Gather** | L2 | 随机索引读(P9) | 读端随机访存 |
| 9 | **GroupedMatmul** | L3 | 批量 float Cube 核调度(P11) | MoE dispatch |
| 10 | **QuantMatmul** | L3 | INT8 Cube 核 + 输出反量化(P28) | INT8 量化推理核心 |
| 11 | **DequantSwigluQuant** | L3 | 反量化→激活→量化融合(P12) | MoE 量化推理出口 |
| 12 | **AddRmsNormDynamicQuant** | L3 | Add+Norm+Quant 三域融合(P13) | DeepSeek-V3 decode |
| 13 | **TopK** | L3 | 排序+TopK 选择(P14) | MoE gating + sampling |
| 14 | **Transpose** | L3 | 跨步数据搬运(P15) | 访存重排基线 |
| 15 | **MlaProlog** | L4 | 9-input 多步前处理编排(P35a) | DeepSeek-V3 decode 前处理 |
| 16 | **MLA** | L4 | KV 低秩解压+注意力计算(P35) | DeepSeek-V2/V3 核心注意力 |

### 5.2 Level 分布

| Level | 名额 | 占比 | 原始占比 | 偏差 |
|-------|------|------|----------|------|
| L1 | 2 | 12.5% | 15.1% | −2.6% ✅ |
| L2 | 6 | 37.5% | 30.2% | +7.3% ✅ |
| L3 | 5 | 31.3% | 39.6% | −8.3% ✅ |
| L4 | 2 | 12.5% | 15.1% | −2.6% ✅ |

> 所有偏差 ≤ ±10%，分布合理。L2 略重（+7.3%）是因为 L2 原始 16 个算子中有效范式密度高（归一化、编码、索引、扫描），6 个名额合理。

### 5.3 范式覆盖验证

| # | 范式 | 代表算子 | 硬件通路 | 不可分解 |
|---|------|----------|----------|----------|
| 1 | 门控激活(P2) | swi_glu | 向量核 | ✅ 含分裂+门控乘法 |
| 2 | TensorList 批量调度(P3) | foreach_addcdiv_scalar | 批量 kernel 发射引擎 | ✅ 多 tensor 并发发射 |
| 3 | 简单归约→归一化(P4) | rms_norm | 归约引擎 + 向量核 | ✅ mean(x²)归约路径独特 |
| 4 | 复杂归约→归一化(P5) | softmax | 归约引擎 + 向量核 | ✅ max+exp-sum 双归约路径独特 |
| 5 | 旋转位置编码(P6) | apply_rotary_pos_emb | 向量核（三角函数） | ✅ 复数旋转模式独特 |
| 6 | 前缀扫描归约(P24) | cummin | 向量核顺序扫描 | ✅ 逐元素递增依赖链独特 |
| 7 | 随机索引写(P10) | scatter | DMA scatter-write | ✅ 冲突写入+原子性独特 |
| 8 | 随机索引读(P9) | gather | DMA scatter-read | ✅ 非连续收集读取独特 |
| 9 | 批量 float Cube 核调度(P11) | grouped_matmul | float16/bf16 Cube 核 | ✅ 多小矩阵并行调度独特 |
| 10 | INT8 Cube 核+输出反量化(P28) | quant_matmul | **INT8 Cube 核** | ✅ 8bit 累加器+独立调度器 |
| 11 | 反量化→激活→量化融合(P12) | dequant_swiglu_quant | float Cube核 + 向量核 | ✅ 三步融合中间变量消除 |
| 12 | Add+Norm+Quant 三域融合(P13) | add_rms_norm_dynamic_quant | 归约+向量+类型转换 | ✅ 跨残差/归一化/量化三域 |
| 13 | 排序+TopK 选择(P14) | top_k | 控制流密集排序核 | ✅ 排序核+选择输出独特 |
| 14 | 跨步数据搬运(P15) | transpose | DMA 跨步读写 | ✅ 维度重排模式独特 |
| 15 | 9-input 多步前处理编排(P35a) | mla_prolog | Cube+向量+归约+DMA 四域并发 | ✅ 九路数据流编排独特 |
| 16 | KV 低秩解压+注意力计算(P35) | mla | Cube核 + 向量核 | ✅ on-the-fly KV解压嵌套注意力 |

**16 个算子 = 16 种不可分解的基本范式，严格一一对应，零重叠零遗漏。**

### 5.4 硬件通路覆盖验证

| 硬件通路 | 施压算子 | 覆盖 |
|----------|----------|------|
| 向量核（逐元素） | swi_glu | ✅ |
| 向量核（三角函数） | apply_rotary_pos_emb | ✅ |
| 向量核（顺序扫描） | cummin | ✅ |
| float16/bf16 Cube 核 | grouped_matmul | ✅ |
| **INT8 Cube 核** | **quant_matmul** | ✅ |
| 归约引擎（全局） | rms_norm, softmax | ✅ |
| DMA 顺序读写 | transpose | ✅ |
| DMA 随机读 | gather | ✅ |
| DMA 随机写 | scatter | ✅ |
| 批量 kernel 发射引擎 | foreach_addcdiv_scalar | ✅ |
| 控制流密集核 | top_k | ✅ |
| 多域并发编排 | mla_prolog, mla | ✅ |

### 5.5 类别占比验证

| 功能类别 | 入选数 | 入选占比 | 原始占比 | 偏差倍数 | 是否合规 |
|----------|--------|----------|----------|----------|----------|
| 量化相关 | 2 纯量化 + 1 含量化组件 = 3 | 18.75% | 9.4% | 2.0× | ✅ ≤2× |
| 归一化相关 | 2 (rms_norm, softmax) | 12.5% | 5.7% | 2.2× | ✅ ≤2× |
| 注意力相关 | 2 (mla_prolog, mla) | 12.5% | 15.1% | 0.8× | ✅ ≤2× |
| 索引相关 | 2 (scatter, gather) | 12.5% | 5.7% | 2.2× | ✅ ≤2× |
| 其他 | 7 | 43.75% | — | — | ✅ |

---

## 6 评测叙事链

入选算子之间存在五条可对比的评测叙事链，能回答关键性能问题：

### 6.1 量化通路对照

```
QuantMatmul(L3, INT8 Cube核) ↔ GroupedMatmul(L3, float Cube核)
QuantMatmul(L3) → DequantSwigluQuant(L3)
```

| 对比维度 | QuantMatmul | GroupedMatmul | DequantSwigluQuant |
|----------|-------------|---------------|-------------------|
| Cube 核类型 | INT8 | float16/bf16 | float16/bf16 |
| 数据流方向 | int8 → int8 → float | float → float | int8 → float → int8 |
| 评测问题 | INT8 核真实吞吐 vs float 核？ | baseline | 反量化+激活后精度恢复多少？ |

### 6.2 融合收益对照

```
RmsNorm(L2, 单步归一化) ↔ AddRmsNormDynamicQuant(L3, 三域融合)
```

| 对比维度 | RmsNorm | AddRmsNormDynamicQuant |
|----------|---------|----------------------|
| 操作范围 | 单一归一化 | 残差加 + 归一化 + 量化 |
| kernel 调用次数 | 1 | 1（融合后）vs 3（未融合） |
| 评测问题 | 归一化自身性能 | 融合到底比三步分做快多少？中间变量消除有多少收益？ |

### 6.3 MLA 流管线

```
MlaProlog(L4, 前处理) ↔ MLA(L4, 注意力计算)
```

| 对比维度 | MlaProlog | MLA |
|----------|-----------|-----|
| 计算阶段 | RMSNorm + Q/K投影 + RoPE | KV解压 + 缩放点积注意力 + 输出投影 |
| 输入数 | 9 | 5 |
| 评测问题 | 前处理开销占 MLA 全流程多大比例？ | 前处理+注意力融合成单 kernel 的收益？ |

> mla_prolog 内部包含 RMSNorm 和 RoPE，与独立选入的 rms_norm(P4) 和 apply_rotary_pos_emb(P6) 构成第二层对照：**两步分做 vs 九路融合**。

### 6.4 写模式光谱

```
Sequential → Strided → Sorted Partial → Random Index-driven
(softmax输出)  (transpose)  (top_k)        (scatter)
```

| 写入模式 | 代表算子 | DMA 行为 |
|----------|----------|----------|
| 连续顺序写 | softmax 输出 | 全带宽顺序写入，最优情况 |
| 跨步写 | transpose | 固定步长非连续写入 |
| 排序部分写 | top_k | 只写前 K 个位置，写范围缩小 |
| 随机索引写 | scatter | index 驱动的完全随机写入，最劣情况 |

### 6.5 访存读写闭环

```
Gather(L2, 随机读) ↔ Scatter(L2, 随机写)
DMA scatter-read ↔ DMA scatter-write
```

| 对比维度 | Gather | Scatter |
|----------|--------|---------|
| 访存方向 | 读端：从散布位置收集数据 | 写端：向散布位置分发数据 |
| 硬件难度 | DMA 有 read-coalesce 优化空间 | 写合并更难，冲突处理 + 原子性 |
| 正确性挑战 | 低（读不改写目标） | 高（重复索引 + 写入顺序） |
| 评测问题 | NPU 随机读带宽衰减多少？ | 随机写比随机读慢多少？DMA 读写端是否对称？ |

> 这条闭环能揭示 **NPU DMA 读写端的对称性（或不对称性）**——如果 scatter 比 gather 显著慢，说明写端优化不足；如果两者接近，说明 DMA 读写通路均衡。这是单靠 scatter 无法回答的问题。

### 6.6 归约依赖光谱

```
Broadcast依赖 → Prefix-scan依赖
(rms_norm)       (cummin)
```

| 依赖模式 | 算子 | NPU压力点 |
|----------|------|-----------|
| broadcast | rms_norm | 归约→单值广播→逐元素，最易并行 |
| prefix-scan | cummin | 逐元素递增依赖链，需扫描核顺序执行 |

> 两点光谱揭示从"完全并行"到"完全顺序"的依赖处理能力差异。若未来扩展至 17 个，补入 mhc_sinkhorn(iterative 依赖) 可形成三点完整光谱。

---

## 7 未入选算子说明

以下 37 个算子未入选。按 v3.0 的严化标准，分为三类：

### 7.1 范式已被入选算子覆盖（同名范式）

| 未入选算子 | 所属范式 | 已被覆盖 |
|------------|----------|----------|
| exp | 逐元素激活(P1) | swi_glu 覆盖门控激活(P2)，P1 由 P2 包含 |
| mish | 逐元素激活(P1) | 同上 |
| sigmoid | 逐元素激活(P1) | 同上 |
| gelu | 逐元素激活(P1) | 同上（第 3 轮移出） |
| masked_scale | 门控激活(P2) | swi_glu 已覆盖 |
| foreach_norm | TensorList 批量(P3) | foreach_addcdiv_scalar 已覆盖 |
| group_norm | 简单归约→归一化(P4) | rms_norm 已覆盖 |
| apply_adam_w | 多输入复合(P7) | 与 apply_rotary_pos_emb 重叠（第 4 轮移出） |
| arg_max | 排序+选择(P14) | top_k 已覆盖 |
| adaptive_avg_pool_3d | 自适应池化(P19) | rms_norm 覆盖归约范式 |
| dilation_2d | 形态学膨胀(P20) | top_k 覆盖局部最大值选择 |
| roi_align | ROI 特征(P22) | scatter 覆盖区域索引写入 |
| unique | 去重(P23) | scatter 间接覆盖间接访存 |
| unsorted_segment_sum | 分段归约(P25) | rms_norm 覆盖分组归约 |
| maximum | 逐元素数学(P27) | swi_glu 覆盖逐元素双输入 |
| conv_2d | 卷积(P17) | grouped_matmul(P11) + transpose(P15) 组合覆盖（第 3 轮移出） |
| depthwise_conv_2d | 卷积(P17) | 同上 |
| conv_3d_backprop_filter | 梯度卷积(P18) | grouped_matmul 覆盖 Cube 核 |
| weight_quant_batch_matmul | INT8 Cube核(P28) | quant_matmul 已覆盖 |
| engram_gate_fusion | 门控融合(P29) | add_rms_norm_dynamic_quant(P13) 覆盖多域融合 |
| moe_gating_top_k_softmax | MoE门控(P30) | softmax(P5) + top_k(P14) 分别覆盖 |
| moe_re_routing | MoE路由(P31) | scatter(P10) 覆盖随机写 |
| moe_finalize_routing | MoE路由(P31) | 同上 |

### 7.2 可分解为已入选范式组合（v2.0 新增类别）

| 未入选算子 | 分解 | 决策轮次 |
|------------|------|----------|
| **mha** | grouped_matmul(P11) + softmax(P5) + grouped_matmul(P11) | 第 6 轮移出，替换为 mla_prolog(P35a) |
| **gqa** | 同 mha，KV head 数仅为参数差异 | 第 6 轮移出 |
| **sparse_flash_attention** | scatter(P10) + grouped_matmul(P11) + softmax(P5) | 第 6 轮移出，替换为 mla(P35) |
| cross_entropy_loss | softmax(P5) + 向量核(log+NLLLoss) | softmax 覆盖核心归约部分 |
| strided_slice | transpose(P15) 的多维特化 | transpose 覆盖跨步访存基线 |
| grouped_matmul_swiglu_quant | grouped_matmul(P11) + dequant_swiglu_quant(P12) 组合 | 无新范式 |

### 7.3 类别占比约束移出（v3.0 新增类别）

| 未入选算子 | 范式 | 移出理由 |
|------------|------|----------|
| **dynamic_quant** | per-token 动态量化(P8) | 量化占比超出 2× 上限（25% vs 原始 9.4%）；per-token max 归约与 rms_norm 归约有结构性重叠；QuantMatmul 本身包含入口量化逻辑（第 9 轮移出） |
| **mhc_sinkhorn** | 算子内迭代循环(P16) | 为 Gather 腾名额；不在主流热点路径；迭代依赖与 cummin 前缀扫描依赖的硬件压力重叠度高于 gather↔scatter 的读写互补价值（第 10 轮移出） |

### 7.4 范式优先级不足（非热点路径 + 无独特硬件通路）

| 未入选算子 | 范式 | 不入选理由 |
|------------|------|------------|
| lstm | RNN 循环序列(P36) | LLM 时代 RNN 边缘化 |
| gru | RNN 循环序列(P36) | 同上 |
| nms | 检测条件分支(P21) | CV 检测路径，非 LLM 热点 |
| gcd | 整数运算(P26) | 无独特硬件通路 |
| resize_bilinear | 插值(P38) | 非热点路径 |
| grid_sampler_3d | 插值(P38) | 同上 |

---

## 8 使用建议

### 8.1 快速回归评测

本设计对应 Season1 的 53 题，而不是当前 Season2。复现下列 16 题时使用归档路径；CLI 没有 `--task-list` 参数。命令在仓库根目录执行，候选 wheel 及运行环境需事先准备；每题分别生成报告。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for op in \
  swi_glu foreach_addcdiv_scalar rms_norm softmax apply_rotary_pos_emb \
  cummin scatter gather grouped_matmul quant_matmul dequant_swiglu_quant \
  add_rms_norm_dynamic_quant top_k transpose mla_prolog mla; do
  python -m kernel_eval.cli eval --bench-name cann \
    --task-dir archives/season1/tasks/level*/"$op" || exit $?
done
```

### 8.2 叙事链专项分析

| 分析目标 | 对比算子 | 关键指标 |
|----------|----------|----------|
| INT8 vs float Cube核吞吐 | quant_matmul ↔ grouped_matmul | HW 时间 / 数据量，归一化后对比 |
| 量化出口精度恢复 | quant_matmul → dequant_swiglu_quant | 相对误差、绝对误差 |
| 融合收益量化 | rms_norm ↔ add_rms_norm_dynamic_quant | HW 时间比、kernel 调用次数比 |
| MLA 流管线瓶颈 | mla_prolog ↔ mla | 各阶段 HW 时间占比 |
| MLA 前处理融合 | rms_norm + apply_rotary_pos_emb ↔ mla_prolog | 两步分做 vs 九路融合 HW 时间比 |
| DMA 读写端对称性 | gather ↔ scatter | HW 时间 / 数据量，读端 vs 写端对比 |
| 写端性能衰减 | softmax输出 → transpose → top_k → scatter | HW 时间 / 数据量，归一化后对比 |
| 归约依赖对比 | rms_norm ↔ cummin | HW 时间 / 数据量，并行 vs 顺序对比 |

### 8.3 定期扩展验证

建议每季度运行一次完整 53 算子评测，验证 Micro-Benchmark 子集是否仍能有效代表整体评测结果。若 Micro-Benchmark 评分与完整评测评分的相关系数 < 0.9，需重新审视选取策略。

### 8.4 扩展候补清单

若未来需要扩展 Micro-Benchmark（如增至 17~18 个），按优先级依次补入：

| 优先级 | 候补算子 | 范式 | 理由 |
|--------|----------|------|------|
| 1 | **mhc_sinkhorn** (L3) | 算子内迭代循环(P16) | 归约依赖光谱第三点(iterative)，补全 broadcast→prefix-scan→iterative 三点光谱 |
| 2 | **dynamic_quant** (L2) | per-token 动态量化(P8) | 量化叙事恢复完整三步流水线(entry→core→exit) |
| 3 | **conv_2d** (L3) | im2col + 单大 Cube核(P17) | 单大矩阵 Cube核调度与 grouped_matmul 的多小矩阵调度互补 |
| 4 | **depthwise_conv_2d** (L3) | 向量核 per-channel | 不同于 Cube核的卷积硬件通路 |

---

## 附录 A：53 算子完整清单与范式映射

| # | 算子 | Level | 核心范式编号 | 是否入选 | 入选编号/未入选理由 |
|---|------|-------|-------------|----------|---------------------|
| 1 | Exp | L1 | P1 | ❌ | P1 被 swi_glu(P2) 覆盖 |
| 2 | ForeachAddcdivScalar | L1 | P3 | ✅ | #2 |
| 3 | ForeachNorm | L1 | P3 | ❌ | P3 被 foreach_addcdiv_scalar 覆盖 |
| 4 | Gelu | L1 | P1 | ❌ | P1 被 swi_glu(P2) 覆盖，第3轮移出 |
| 5 | MaskedScale | L1 | P2 | ❌ | P2 被 swi_glu 覆盖 |
| 6 | Mish | L1 | P1 | ❌ | P1 被 swi _glu(P2) 覆盖 |
| 7 | Sigmoid | L1 | P1 | ❌ | P1 被 swi_glu(P2) 覆盖 |
| 8 | SwiGLU | L1 | P2 | ✅ | #1 |
| 9 | ApplyAdamW | L2 | P7 | ❌ | 与 apply_rotary_pos_emb 重叠，第4轮移出 |
| 10 | ApplyRotaryPosEmb | L2 | P6 | ✅ | #5 |
| 11 | ArgMax | L2 | P14 | ❌ | P14 被 top_k 覆盖 |
| 12 | CrossEntropyLoss | L2 | P5+P4 | ❌ | 可分解为 softmax(P5) + 逐元素 |
| 13 | Cummin | L2 | P24 | ✅ | #6（第9轮补入） |
| 14 | DynamicQuant | L2 | P8 | ❌ | 量化占比超出2×上限，第9轮移出；扩展候补#2 |
| 15 | Gather | L2 | P9 | ✅ | #8（第10轮补入） |
| 16 | Gcd | L2 | P26 | ❌ | 优先级不足 |
| 17 | GridSampler3D | L2 | P38 | ❌ | 优先级不足 |
| 18 | GroupNorm | L2 | P4 | ❌ | P4 被 rms_norm 覆盖 |
| 19 | Maximum | L2 | P27 | ❌ | P27 被 swi_glu 覆盖 |
| 20 | ResizeBilinear | L2 | P38 | ❌ | 优先级不足 |
| 21 | RmsNorm | L2 | P4 | ✅ | #3 |
| 22 | Scatter | L2 | P10 | ✅ | #7 |
| 23 | Softmax | L2 | P5 | ✅ | #4 |
| 24 | UnsortedSegmentSum | L2 | P25 | ❌ | P25 被 rms_norm 覆盖 |
| 25 | AdaptiveAvgPool3D | L3 | P19 | ❌ | P19 被 rms_norm 归约范式覆盖 |
| 26 | AddRmsNormDynamicQuant | L3 | P13 | ✅ | #12 |
| 27 | Conv2D | L3 | P17 | ❌ | 可分解为 grouped_matmul(P11) + transpose(P15)；第3轮移出 |
| 28 | Conv3DBackpropFilter | L3 | P18 | ❌ | P18 被 grouped_matmul(P11) 覆盖 |
| 29 | DepthwiseConv2D | L3 | P17 | ❌ | 同 Conv2D；扩展候补#4 |
| 30 | DequantSwigluQuant | L3 | P12 | ✅ | #11 |
| 31 | Dilation2D | L3 | P20 | ❌ | P20 被 top_k(P14) 覆盖 |
| 32 | EngramGateFusion | L3 | P29 | ❌ | P29 被 add_rms_norm_dynamic_quant(P13) 覆盖 |
| 33 | GroupedMatmul | L3 | P11 | ✅ | #9 |
| 34 | MhcSinkhorn | L3 | P16 | ❌ | 为 Gather 腾名额，第10轮移出；扩展候补#1 |
| 35 | MoeFinalizeRouting | L3 | P31 | ❌ | P31 被 scatter(P10) 覆盖 |
| 36 | MoeGatingTopKSoftmax | L3 | P30 | ❌ | 可分解为 softmax(P5) + top_k(P14) |
| 37 | MoeReRouting | L3 | P31 | ❌ | P31 被 scatter(P10) 覆盖 |
| 38 | NMS | L3 | P21 | ❌ | 优先级不足 |
| 39 | QuantMatmul | L3 | P28 | ✅ | #10 |
| 40 | ROIAlign | L3 | P22 | ❌ | P22 被 scatter(P10) 覆盖 |
| 41 | StridedSlice | L3 | P15 | ❌ | P15 被 transpose 覆盖 |
| 42 | TopK | L3 | P14 | ✅ | #13 |
| 43 | Transpose | L3 | P15 | ✅ | #14 |
| 44 | Unique | L3 | P23 | ❌ | P23 被 scatter(P10) 间接覆盖 |
| 45 | WeightQuantBatchMatmul | L3 | P28 | ❌ | P28 被 quant_matmul 覆盖 |
| 46 | GQA | L4 | P33 | ❌ | 可分解为 grouped_matmul(P11) + softmax(P5)；第6轮移出 |
| 47 | GroupedMatmulSwigluQuant | L4 | P37 | ❌ | 可分解为 grouped_matmul(P11) + dequant(P12) |
| 48 | GRU | L4 | P36 | ❌ | 优先级不足 |
| 49 | LSTM | L4 | P36 | ❌ | 同 GRU |
| 50 | MHA | L4 | P32 | ❌ | 可分解为 grouped_matmul(P11) + softmax(P5)；第6轮移出 |
| 51 | MLA | L4 | P35 | ✅ | #16 |
| 52 | MlaProlog | L4 | P35a | ✅ | #15 |
| 53 | SparseFlashAttention | L4 | P34 | ❌ | 可分解为 scatter(P10) + grouped_matmul(P11) + softmax(P5)；第6轮移出 |
