# 算子代码生成评测基准规范

## 1. 方案概述

### 1.1 背景与定位

大语言模型在代码生成领域的能力快速提升，正逐步扩展到算子开发等系统级编程场景。Ascend C作为昇腾平台的算子开发语言，具有高度并行化、显式内存管理、多级流水线等复杂编程范式，对AI生成代码的正确性和性能提出了更高要求。

本评测方案建立了一套面向AI生成Ascend C算子代码的评测体系，核心定位如下：

- **量化评估基准**：通过编译正确性、功能正确性、性能优化性三个维度，量化衡量AI生成算子代码的质量
- **能力演进驱动**：以评促建，通过持续评测推动AI在算子生成领域的精度和性能提升
- **模型选型参考**：为不同基础模型、Agent/Skill方案的选择提供客观对比依据
- **社区协作平台**：建立开放的算子评测贡献机制，汇聚社区力量持续扩展评测场景

### 1.2 适用范围

本评测方案适用于以下场景：

| 场景 | 说明 |
|------|------|
| 基础模型评测 | 对比不同基础模型（GLM、Claude、GPT等）在算子代码生成任务上的表现 |
| Agent/Skill评测 | 评估针对算子开发定制的Agent工作流和Skill工具链的效果 |
| 训练效果评估 | 衡量模型经过算子领域微调/训练后的能力提升幅度 |
| 社区贡献验收 | 作为社区新增算子评测任务的审核和验收标准 |

### 1.3 设计理念

- **场景驱动**：以实际算子开发场景为核心，覆盖从简单Elementwise到复杂Attention/RNN的全难度谱系
- **多维评价**：编译、功能精度、性能三维度独立评测，避免单一指标的片面性
- **开放透明**：算子规格、用例设计、Golden实现、评分算法全部开源，评测结果可复现
- **持续演进**：通过社区共建扩展评测场景，建立垂直领域子榜单，评分体系持续完善

### 1.4 版本演进

详细版本变更记录请参阅 [docs/changelog.md](../changelog.md)。

---

### 1.5 演进规划

本评测方案以"场景驱动、社区共建、持续演进"为核心理念，分阶段推进榜单建设与评分体系完善。

#### 1.5.1 阶段一：基础覆盖（当前阶段）

聚焦算子场景覆盖和泛化性，建立基础评测框架：

- **算子覆盖**：完成算子泛化场景下55个算子（L1-L4）的规格定义、用例设计和Golden实现，覆盖Elementwise、Reduce、Matmul、Convolution、Attention等核心计算模式
- **评测维度**：建立编译正确性、功能正确性、性能优化性三大评测维度，形成综合评分体系
- **泛化验证**：每个算子提供20开放用例，覆盖Shape维度、数据类型、属性取值等泛化场景
- **基准建立**：发布第一版算子榜单，为AI生成算子代码能力提供量化评估基准

#### 1.5.2 阶段二：社区共建

吸纳社区贡献，扩展评测场景覆盖：

- **社区算子贡献**：建立算子贡献机制，接纳社区开发者提交的新算子评测任务，包括算子规格定义、测试用例和Golden实现，经审核后纳入评测集
- **垂直领域榜单**：根据不同垂直领域的场景需求，构建独立的子榜单集合：
  - **大模型训练领域**：FlashAttention系列、MLA、MoE路由等Transformer相关算子
  - **推荐系统领域**：EmbeddingHashLookup、Foreach系列等推荐场景算子
  - **量化推理领域**：DynamicQuant、WeightQuantBatchMatmul、DequantSwigluQuant等量化融合算子
  - **视觉处理领域**：Conv2D、ROIAlign、GridSampler3D等图像处理算子
  - **科学计算领域**：社区贡献的科学计算、图计算等特殊场景算子
- **场景化Prompt集**：针对各领域典型使用场景，构建标准化的Prompt输入模板，提升评测的可复现性

#### 1.5.3 阶段三：工程化平台

建设自动化评测基础设施，支撑榜单持续运营：

- **评测网站**：发布在线评测结果展示平台，支持多维度榜单查询、历史版本对比、趋势分析
- **工程平台**：提供在线算子定义管理、用例管理、评测任务提交和结果可视化的一站式平台

#### 1.5.4 阶段四：评分体系持续完善

深化评分算法和评测标准，建立更科学的评价体系：

- **评分算法优化**：
  - 引入算子复杂度因子，根据计算流复杂度、Tiling策略难度、内存访问模式等调整分值权重
  - 引入用例难度因子，区分典型用例和边界/极端用例的评分权重
  - 性能评分从"相比基准加速比"演进到"相比理论性能占比"，更客观衡量优化空间
- **精度标准升级**：对接CANN官方算子精度标准，建立细粒度的分精度评测体系
- **防作弊体系**：完善aclnn路由检测、代码相似度分析等防作弊机制

## 2. 评测体系架构

### 2.1 三层评测框架

评测体系自底向上划分为数据层、评测层、应用层三个层次，各层职责如下：

**数据层**：提供评测所需的全部数据基础，是评测体系的底层支撑。

| 数据项 | 说明 | 文件格式 |
|--------|------|----------|
| 算子规格描述 | 算子原型定义、输入输出schema、属性列表 | proto.yaml |
| 算子分类与难度分级 | L1~L4四级难度体系，不同难度对应不同分值权重 | 元数据标签 |
| 测试用例集 | 每个算子20+开放用例，覆盖Shape/dtype/属性/值域 | cases.csv |
| Golden实现 | 基于PyTorch官方API的标杆计算函数 | golden.py |
| 性能基线 | 每个用例对应的基准执行时间 | baseline_perf_us |
| 内部泛化验证集 | CANN官方CI工程维护的80条泛化用例（不公开） | 内部数据集 |

**评测层**：基于数据层提供的数据，执行三大维度的评测。

| 评测维度 | 评测内容 | 核心指标 | 输出 |
|----------|----------|----------|------|
| 编译正确性 | AI生成的算子代码能否成功编译链接 | Pass/Fail | 是否编译通过（二值） |
| 功能正确性 | 算子输出与Golden结果的精度偏差 | 用例通过数 | 精度通过率 |
| 性能优化性 | 生成算子与基准性能的比例 | SpeedUp加速比 | 性能评分 |

**应用层**：将评测结果转化为可视化的报告、榜单和持续集成的工程化能力。

| 应用项 | 说明 |
|--------|------|
| 评测报告 | 综合得分、子项得分、各算子任务得分明细 |
| 在线评测平台 | 算子评测任务提交、多维度榜单查询、历史版本对比、趋势分析 |

### 2.2 评测流程

评测流程分为AI算子生成、本地评测、CANN官方评测三个阶段，依次推进，前一阶段的结果作为后一阶段的输入。

**阶段一：AI算子生成**

模型基于算子目录中的完整信息生成Ascend C算子代码。

| 输入文件 | 说明 |
|----------|------|
| desc.md | 算子描述（计算公式、功能说明、典型使用场景） |
| proto.yaml | 算子原型定义（输入输出schema、属性列表、数据类型） |
| golden.py | 算子标杆函数（基于PyTorch的参考实现） |
| cases.csv | 算子用例集（Shape、dtype、属性、值域、性能基线） |

| 输出文件 | 说明 |
|----------|------|
| 算子工程源码 | 完整的Ascend C算子实现（含CMakeLists、kernel实现、TorchAPI plugin等） |

**阶段二：本地评测**

本地评测按顺序执行编译、功能、性能三项评测，存在前置依赖关系。

```
┌────────────┐      ┌────────────┐       ┌────────────┐
│  编译评测   │────▶│ 功能精度评测 │────▶│  性能评测   │
└────────────┘      └────────────┘       └────────────┘
  Pass/Fail          用例通过数            SpeedUp
```

| 评测项 | 前置条件 | 输入 | 输出 | 工具 |
|--------|----------|------|------|------|
| 编译评测 | 无 | AI生成的算子工程源码 | 编译是否通过（Pass/Fail） | cmake、bisheng编译器 |
| 功能精度评测 | 编译通过 | 编译后的算子 + cases.csv + golden.py | 用例通过率 | evaluation框架 |
| 性能评测 | 功能精度通过 | 通过精度用例的算子 + cases.csv基线数据 | SpeedUp加速比 | msprof/TorchNPU.prof |

**阶段三：CANN官方评测**

通过本地评测后，按指定格式提交算子源码包至CANN官方CI工程，进行泛化评测（(根据评测集确定是否有私有用例)）。

| 输入 | 说明 |
|------|------|
| 模型信息 | 使用的基础模型（GLM、Claude、GPT等） |
| Prompt信息 | 每个算子任务的完整Prompt |
| 算子工程源码 | 最终提交的算子源码 |

| 输出 | 说明 |
|------|------|
| 官方评测报告 | 发布在评测平台的官方评测报告 |

## 3. 数据层

数据层主要包含了一系列的评测任务集合，当前评测主要针对端到端算子生成任务，发布的评测任务集以各种算子开发任务为主，涉及不同开发难度的算子。
数据层给出的信息，都可以作为Prompt的输入信息给到模型。

### 3.1 算子分类和难度等级定义

**算子分类**

**算子难度等级**

算子根据计算流、计算模式不同，可以大致分为以下几个难度等级：

| 等级 | 特征描述 | AI生成难度 |代表算子 |
|------|----------|------------|------------|
| L1 | 单输入单输出、Elewise操作、无特殊优化 | 简单 |Add、Exp、MaskedScale |
| L2 | 多输入、轻量级Broadcast、需Tiling但策略固定 | 中等 |Gather、ApplyAdamW、Gelu |
| L3 | 多维度归约、多Tiling策略可选 | 较难 | TopK、AvgPool、Matmul、Conv2D、BatchMatMul |
| L4 | 矩阵运算、多算子融合、复杂数据流、需要极致性能调优 | 困难 |FlashAttentionScore、LSTM|

不同level在总分计量中会有不同的分值权重。

### 3.2 评测算子清单

结合Ascend C算子开发特点，以当前CANN仓中算子作为基础评测任务

**L1算子**
- Elewise：Exp、MaskedScale
- 激活函数：Gelu、Sigmoid、SwiGLU、Mish
- Foreach类：ForeachNorm、ForeachAddcdivScalar

**L2算子**
- 优化器算子：ApplyAdamW
- Broadcast：Maximum、Gcd
- 量化算子：DynamicQuant
- 损失函数算子：CrossEntropyLoss
- 索引操作：Gather、Scatter、UnsortedSegmentSum(仅Int)
- 插值类：ResizeBilinear、GridSampler3D
- Reduce：ArgMax、Cummin
- 正则化：Softmax、RMSNorm、GroupNorm
- Transform类：ApplyRotaryPosEmb

**L3算子**
- 池化算子：AdaptiveAvgPool3D
- 张量变换：Transpose、StridedSlice
- 排序类：TopK、Unique
- 图像处理：Dilation2D
- 目标检测：NMSWithMask、ROIAlign
- MoE类：MoeReRouting、MoeFinalizeRouting、MoeGatingTopKSoftmax
- 矩阵运算：GroupedMatmul
- MM量化：QuantBatchMatmul、WeightQuantBatchMatmul
- 卷积：Conv2D、DepthwiseConv2D
- 卷积反向：Conv3DBackpropFilter
- VV融合：AddRmsNormDynamicQuant、DequantSwigluQuant、MhcSinkhorn、Engram
 

**L4算子**
- Transformer类：MHA、GQA、MLA、SparseFlashAttention、MlaProlog
- RNN类：LSTM、GRU
- 量化融合：GroupedMatmulSwigluQuant

**算子定义文件：** proto.yaml
```yaml
# proto.yaml
- name: Exp
    category: Elementwise
    difficulty: L1
    formula: "y = e^((x * scale + shift) * ln(base))"
    description: "计算输入张量的指数函数，支持自定义底数、缩放和偏移"
    shape_support: "输入任意维度，输出与输入相同shape"
    attrs:
      - name: base
        type: float
        default: -1.0
        description: "指数底数，-1.0表示使用自然底数e，正值表示自定义底数"
      - name: scale
        type: float
        default: 1.0
        description: "输入缩放因子"
      - name: shift
        type: float
        default: 0.0
        description: "输入偏移量"
    note: "当base=-1时，公式简化为 y = e^(x * scale + shift)"
    inputs:
      - name: x
        description: 输入张量
        dtype: ["float16", "float32", "bfloat16"]
    outputs:
      - name: y
        description: 指数计算结果
        dtype: ["float16", "float32", "bfloat16"]
    schema: exp(Tensor x, float base, float scale, float shift) -> Tensor y
```
> 算子自定义TorchAPI接口，需要与schema一致, lib空间统一为`cann_bench`
```
import cann_bench
y = cann_bench.exp(x, -1.0, 1.0, 0.0)
```

### 3.3 算子评测用例

该评测体系有两种用例：
- 开放用例：随算子评测标准一起发布，由算子任务集中算子典型场景Shape和Attr属性组合（一般一个算子20条左右）；
- 内部评测用例：不随算子评测标准发布，按照指定工程形式提供自定义算子源码文件，由CANN官方CI工程完成评测。

**算子评测用例设计原则**
每个用例可以用不同的标签，比如泛化、热点网络等，根据标签可以输出不同的评测维度榜单

**泛化维度**
- 用例生成：输入输出（Shape维度/数据类型）、属性泛化、取值范围泛化、特殊值

**热点维度**
常见网络Shape/网络Shape泛化

**算子用例定义文件：** cases.csv
```
operator,case_id,input_shape,dtype,attrs,value_range,baseline_perf_us,note
Exp,1,"[[1024, 1024]]",['float16'],"{'base': -1.0, 'scale': 1.0, 'shift': 0.0}","[-1, 1]",21.05,float16-1M-对齐-对称小值域-base=-1
Exp,2,"[[2048, 2048]]",['float32'],"{'base': -1.0, 'scale': 1.5, 'shift': 0.0}","[-2, 2]",18.27,float32-4M-对齐-对称小值域-scale=1.5
```

### 3.4 Golden脚本

根据proto.yaml中算子的定义，提供相应算子的Golden脚本

实现方式：基于pytorch官方API
```python
def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数（核心Golden计算逻辑）

    公式: y = base^(x * scale + shift)，当base=-1时，y = e^(x * scale + shift)

    Args:
        x: 输入张量
        base: 底数，默认-1.0表示使用e
        scale: 缩放因子，默认1.0
        shift: 偏移量，默认0.0

    Returns:
        输出张量 y
    """
    temp = x * scale + shift
    if base == -1.0:
        y = torch.exp(temp)
    else:
        y = torch.exp(temp * torch.log(torch.tensor(base, dtype=x.dtype, device=x.device)))
    return y
```
---

## 4. 评测层

### 4.1 三大评测维度

| 维度 | 权重 | 评测重点 | 评测工具 | 评分范围 |
|------|------|----------|----------|----------|
| 编译正确性 | Wc=2 | 编译通过 | cmake、gtest | [0, 100] |
| 功能正确性 | Wf=3 | 通过测试用例Golden对比 | cmake、gtest | [0, 100] |
| 性能优化性 | Wp=5 | 相比基准时间的比例| msprof、cannsim | [0, 100] |

### 4.2 核心评测指标

- **编译正确性**: 是否编译通过（Pass/Fail，二值）
- **功能正确性**: 精度用例通过数量 
- **性能优化性**: 相比基准性能的加速比(当前)；相比理论性能的比例(规划)
```
单算子综合评分
├── 编译正确性 (权重 Wc=2)
│   └── compile_pass ∈ {0, 1}（整份提交编译是否通过，与用例数无关）
├── 功能正确性 (权重 Wf=3)
│   └── 用例通过数 (通过精度用例的数量)
└── 性能优化性 (权重 Wp=5)
    └── 加速比 (验证性能 / 基准性能)

计算方式：
编译通过得分 = compile_pass × Wc         # 单算子一次，编译 Pass=1 / Fail=0
单用例功能得分 = case_pass × Wf          # case_pass ∈ {0, 1}，该用例是否通过精度校验
单用例性能得分 = SpeedUp_i × Wp          # 仅对功能通过的用例计入，SpeedUp_i 为该用例实测

单算子综合评分 = 编译通过得分
              + Σ_{功能通过的用例 i} ( Wf + SpeedUp_i × Wp )

即：编译项为单算子一次的标量贡献；功能与性能项按该算子"功能通过用例"逐一累加。
```

**聚合规则**：

```
Level-N 得分       = Σ_{op ∈ Level-N} 单算子综合评分
benchmark 总分     = Σ_{所有算子} 单算子综合评分
                   = Level1 得分 + Level2 得分 + Level3 得分 + Level4 得分
```

### 4.3 编译评测（Pass/Fail）

官方评测中，每个算子只接收一份源码提交，编译评测为**二值结果**：Pass（通过）/ Fail（未通过）。这份源码的编译结果作用于该算子的全部测试用例：

- **编译通过**（`compile_pass = 1`）：所有用例继续进入功能与性能评测；
- **编译失败**（`compile_pass = 0`）：所有用例的编译得分、功能得分、性能得分均为 0，该算子综合得分直接为 0。

> 早期设计曾沿用业界代码生成评测中的 `Pass@k` 命名，但在"单算子单提交"的官方评测约束下，`n=1、k=1`，该指标实际退化为 pass/fail 二值判断，因此本方案不再使用 `Pass@k` 表述。若未来官方评测放开允许一次提交多份候选代码，再行按标准 Pass@k 公式扩展。

### 4.4 精度标准

当前采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)，后续会引入[昇腾算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/commercial_standard.md)

**生态算子精度标准**
##### 误差指标

该标准主要用来衡量生态贡献中（贡献在experimental目录下）的计算类算子精度是否达标，通过该标准作为生态贡献的必要条件。
该标准采用平均相对误差和最大相对误差指标来判断，计算公式如下：$actual$为NPU实际输出的结果；$golden$为参考计算的真值

1. 平均相对误差（Mean Relative Error，MERE）：采样点中相对误差平均值。
   
   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$
   
   计算相对误差的时候引入小值1e-7避免golden出现除0风险。
2. 最大相对误差（Max Relative Error，MARE）：采样点中相对误差最大值。
   
   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

##### 通过标准

**单标杆比对**：与更高精度的实现的单一精度标杆（CPU或昇腾小算子拼接）直接比较。

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

**通过标准：**
当平均相对误差MERE < Threshold ， 最大相对误差MARE < 10 * Threshold判定为通过

##### 小值域通过说明

当算子输出结果为极小值（接近0）时，相对误差计算可能不稳定，因此需要使用小值域通过标准评估精度。

**小值域阈值对应表：**

| 指标类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **小值域阈值(Small Value Threshold)** | 2^-11 | 2^-8 | 2^-14 | 2^-12 | 2^-4 | 2^-3 |
| **小值域error指标** | 2^-16 | 2^-16 | 2^-30 | 2^-28 | 2^-6 | 2^-5 |

当真值小于 Small Value Threshold 时，采用小值域通过标准。定义误差度量指标**小值域数值错误数量（ErrorCount）**：

$$
\mathbf{ErrorCount}=\sum \mathbb{I}\left(
\mathbf{|golden|} < threshold \land
\left|\mathbf{actual} - \mathbf{golden}\right| > \mathbf{error}
\right)
$$

- $\mathbb{I}(⋅)$ 是指示函数（条件成立时为 1，否则为 0）
- $∧$ 表示逻辑"且"
- $error$、$threshold$ 请参考上表

**小值域通过标准：**

$$
\frac{\text{ErrorCount}_{\text{npu}}}{\max(\text{ErrorCount}_{\text{cpu标准精度}}, 1)} \leq 2
$$

**说明：** 此标准适用于所有数据类型。

### 4.5 性能评测规范

**评测原则**：性能评测需保证测量一致性、防止作弊攻击、准确反映算子真实性能。

**评测流程**：

```
功能通过 → NPU升频清cache → 预热执行 → 正式性能测试 → 解析trace → 计算统计结果 → 与基准对比
```

**核心要求**：

| 要求项 | 说明 |
|--------|------|
| Kernel-only测量 | 仅统计 NPU 内核执行时间，剥离 Python 派发开销 |
| NPU升频清cache | 每次测量前执行 MatMul + ReduceMax，保证 NPU 频率稳定并清空 L2 cache |
| 输入池防缓存 | 预分配 clone 输入池轮换使用，防止按 data_ptr 缓存输出 |
| Warmup Kernel过滤 | 自动过滤升频用的 MatMul/ReduceMax kernel，只统计目标算子时间 |

**采集参数标准**：

| 参数 | 标准值 | 说明 |
|------|--------|------|
| warmup | 3 | 预热次数，消除缓存影响 |
| repeat | 5 | 正式采集次数 |
| freq_boost | True | 启用 NPU 升频清 cache |
| ProfilerLevel | Level0 | 采集详细程度（kernel-only） |

**性能指标计算**：

- **Kernel时间**：通过解析 chrome trace 中 `cat="dequeue"` 事件获取 NPU 内核执行时间
- **加速比**：`SpeedUp = baseline_perf_us / kernel_perf_us`
- **几何平均加速比**：对多个用例的加速比取几何平均

> 详细实现请参阅 [evaluator_design.md](../design/evaluator_design.md)

### 4.6 防作弊规范

| 防护项 | 规范要求 |
|--------|----------|
| 禁用内置算子 | 删除环境内内置算子的二进制实现，避免直接路由到内置算子 |
| Timing API防护 | 快照关键 API 身份，安装 wheel 后验证是否被篡改 |
| 返回值类型检查 | 严格检查 `type(output) is torch.Tensor`，拒绝 FakeTensor |
| 二次验证 | 用新鲜输入重跑，防止缓存作弊 |

> 详细实现请参阅 [evaluator_design.md](../design/evaluator_design.md)

## 5. 应用层规范

### 5.1 评测报告规范

**评测报告必须包含的核心要素**：

| 要素 | 说明 |
|------|------|
| 评测集版本号 | 对应明确的算子任务清单、用例集合、验收标准、性能基线 |
| 评测代号 | 自定义提交评测任务的组织代号 |
| 基础模型 | 使用的基础模型（GLM、Claude、GPT等） |
| Agent/Skill | 使用的 Agent/Skill 方案 |
| 综合得分 | 按评分规则计算的综合得分 |
| 子项得分 | 编译、功能精度、性能各维度得分 |
| 算子任务明细 | 各算子任务的详细得分情况 |

### 5.2 评测平台规范

评测平台应提供以下能力：

- 多维度榜单查询（按级别、按领域、按模型）
- 历史版本对比
- 趋势分析
- 在线评测任务提交