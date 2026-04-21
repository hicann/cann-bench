# 算子代码生成评测方案V1.0.0

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

| 版本 | 主要变更 |
|------|----------|
| V1.0.0 | 初版，建立基础评测框架|
| V0.2.0 | 引入Pass@k评测、算子分类体系、三大维度评测 |
| V0.3.0 | 完善算子复杂度定义、规范用例输入输出 |
| V0.4.0 | 增加评测报告结构、优化评测流程 |
| V0.5.0 | 规范算子交付件要求 |
| V0.6.0 | 调整算子清单，明确第一版泛化场景55个算子|
---

### 1.5 演进规划

本评测方案以"场景驱动、社区共建、持续演进"为核心理念，分阶段推进榜单建设与评分体系完善。

#### 1.5.1 阶段一：基础覆盖（当前版本 V1.0.0）

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
| 编译正确性 | AI生成的算子代码能否成功编译链接 | Pass@k | 编译通过率 |
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
   Pass@k            用例通过率            SpeedUp
```

| 评测项 | 前置条件 | 输入 | 输出 | 工具 |
|--------|----------|------|------|------|
| 编译评测 | 无 | AI生成的算子工程源码 | Pass@k指标 | cmake、bisheng编译器 |
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
- 插值类：ResizeBilinearV2、GridSampler3D
- Reduce：ArgMax、Cummin
- 正则化：Softmax、RMSNorm、GroupNorm
- Transform类：ApplyRotaryPosEmb

**L3算子**
- 池化算子：AdaptiveAvgPool3D
- 张量变换：Transpose、StridedSlice
- 排序类：TopK、Unique
- Hash类：EmbeddingHashLookupOrInsert
- 图像处理：Dilation2D
- 目标检测：NMSWithMask、ROIAlign、ROIPooling（待删除）
- MoE类：MoeReRouting、MoeFinalizeRoutingV2、MoeGatingTopKSoftmax
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

### 3.2 算子评测用例

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

### 3.3 Golden脚本

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

- **编译正确性**: Pass@k指标
- **功能正确性**: 精度用例通过数量 
- **性能优化性**: 相比基准性能的加速比(当前)；相比理论性能的比例(规划)
```
综合评分
├── 编译正确性 (权重分 Wc=2)
│   └── Pass@1 (编译通过率)
├── 功能正确性 (权重分 Wf=3)
│   └── 用例通过数 (通过精度用例的数量) 
└── 性能优化性 (权重分 Wp=5)
    └── 加速比 (验证性能/测试基准性能)

计算方式：
编译得分：Pass@1 x Wc
功能得分：Pass@1 x Wf
性能得分：SpeedUp x Wp
综合评分 = 编译通过用例数 * 编译得分 + 功能通过用例数 × (功能得分 + 性能得分)
```

### 4.3 Pass@k评测

业界标准的代码生成评测方法，生成k个候选代码中至少有1个通过所有测试用例的概率。

**简化计算**
```
Pass@k = 1 - C(n-c, k) / C(n, k)
```
- **n**: 生成的候选代码总数
- **c**: 通过测试的候选代码数量
- **k**: 选择的最优候选数量（通常取1、5、10）

**简化计算**
```
Pass@1 = c / n
```
即：单次生成通过率 = 通过数 / 总生成数

> 官方评测中，由于只要求提供一份源码，因此Pass@1只有可能是0或者1两种取值

#### 4.4 精度标准

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

<table style="width: 120%; border-collapse: collapse;">
    <colgroup>
      <col style="width: 25%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
      <col style="width: 12.5%;" />
    </colgroup>
    <thead>
      <tr>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;">数据类型</th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT16</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>BFLOAT16</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT32</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>HiFLOAT32</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT8 E4M3</strong></th>
        <th style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>FLOAT8 E5M2</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>通过阈值<br>(Threshold)</strong></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-7</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-13</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-11</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-3</sup></td>
        <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2<sup>-2</sup></td>
      </tr>
    </tbody>
  </table>

**通过标准：**
当平均相对误差MERE < Threshold ， 最大相对误差MARE < 10 * Threshold判定为通过

### 4.5 性能评测

性能评测流程
```
功能通过 → 预热执行 → 正式性能测试 → 计算统计结果 → 与基准对比 → 计算加速比
```

**性能采集方式1**
预热3次后，执行5次取中位数，使用msprof工具，先执行3次预热消除缓存影响，再执行5次正式测试，取中位数作为最终执行时间，排除异常值影响
```
msprof op --warm-up=3 --launch-count=5 --output=./msprof_output ./your_op arg1 arg2
```
**性能采集方式2**
基于torch_npu.profiler的性能采集方案，通过解析chrome trace JSON获取NPU内核执行时间。

```python
import torch_npu

# 配置 experimental_config
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
)

# 使用 schedule 机制：warmup预热 + repeat采集
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=3, active=5, repeat=1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    experimental_config=experimental_config,
) as prof:
    # 执行 warmup + active 次循环
    for _ in range(warmup + repeat):
        outputs = func(*args, **kwargs)
        prof.step()
```

**性能数据解析**
解析生成的trace_view.json文件，通过cat字段区分Host/Device阶段，获取NPU内核执行时间：
```python
# 解析 chrome trace JSON
events = data.get('traceEvents', [])
for event in events:
    if event.get('ph') != 'X':  # 只处理完整事件
        continue
    dur = event.get('dur', 0)
    name = event.get('name', '')

    # 通过 cat 字段判断：有 cat = Host端，无 cat = Device端
    if 'cat' in event:
        host_ops[name] = host_ops.get(name, 0) + dur
    else:
        device_kernels[name] = device_kernels.get(name, 0) + dur
        total_kernel_us += dur

# 对 repeat 次采集结果取平均
kernel_time_us = total_kernel_us / repeat
```

**采集参数配置**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| warmup | 3 | 预热次数，消除缓存影响 |
| repeat | 5 | 正式采集次数 |
| ProfilerLevel | Level0 | 采集详细程度 |
| export_type | Text | 输出格式 |

**特点**
- 基于Python接口，无需外部命令行工具
- 支持异步解析，不影响后续测试执行
- 自动归档profiling数据到 `test/reports/prof_data/{level}/{op_name}/{caseid}/`
- 可获取Host端和Device端各算子的详细耗时

### 4.6 防作弊
**避免使用内置算子**
  对于aclnn调用，删除环境内内置算子的二进制实现（环境上不安装内置算子kernel），避免直接在aclnn层路由到内置算子实现
> TODO

## 5 应用层
评测报告、评测工程、CANN评测结果网站等
### 5.1 评测报告
**评测报告核心要素**
- 评测集版本号：每个版本号对应明确的算子任务清单和开放和未开放的用例集合，用例验收标准以及性能基线
- 评测代号：自定义提交评测任务的组织代号
- 基础模型：GLM5/Opus等
- Agent/Skill: CANNBot等
- 综合得分：计算综合得分
- 子项得分：评估各个子项维度的得分（编译、功能精度、性能）
- 各算子任务得分情况

### 5.2 Web在线网站


## 6 规划
- 工程平台构建
  - [ ] 完成剩余 Level3/Level4 算子核对验证, 发布第一版算子评测集合
  - [ ] 建立持续评测 CI 流水线
  - [ ] 发布评测网站

- 评测集构建
  - [ ] 增加更多算子类型覆盖
  - [ ] 根据领域场景分类，算子特征等，构建出更多独立榜单集合，覆盖不同评测场景的需求

- 评测标准构建
  - [ ] 评测精度标准，精度衡量方法构建
  - [ ] 评测性能基线，理论性能评估
  - [ ] 评分算法优化（例如算子复杂度、用例难度），科学评价生成能力
  - [ ] 算子分级/分类方法
  - [ ] 防作弊体系构建