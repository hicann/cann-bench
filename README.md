# CANN Bench: CANN 领域评测

评测AI在处理CANN领域代码任务的能力，涵盖算子生成、算子优化等领域，支撑模型选型、训练效果评估，统一量化评估标准，识别Agent能力短板，构建CANN领域评测平台，推动AI能力在CANN领域的持续演进。

📖 [查看详细技术报告](docs/technical-report.pdf)

## 👋 Task Description

评测AI模型生成Ascend C算子代码的能力，按算子复杂度分为4个等级，覆盖算子主流开发范式：

- **Level 1**: 基础算子 (Element-wise, Activation)
  单输入单输出、Elewise操作、无特殊优化，如 Add、Exp、Gelu、Sigmoid、Mish
- **Level 2**: 中级算子 (Normalization, Reduction, Gather/Scatter)
  多输入、轻量级Broadcast、需Tiling但策略固定，如 Gather、ApplyAdamW、Softmax
- **Level 3**: 高级算子 (Conv, Pooling, MoE)
  多维度归约、多Tiling策略可选，如 TopK、Conv2D、Matmul、NMS
- **Level 4**: 复杂算子 (Attention, RNN)
  矩阵运算、多算子融合、复杂数据流，如 FlashAttention、LSTM、GRU

## ⚖️ Evaluation

### 三层评测框架

- **数据层**：评测任务集（算子规格描述、Golden实现、测试用例、泛化验证集）
- **评测层**：评测维度（编译正确性、功能精度、性能优化性）
- **应用层**：评测报告、CI流水线工程、问题改进、评测结果网站

### 核心评测指标

| 维度 | 指标 | 权重 | 说明 |
|------|------|------|------|
| 编译正确性 | Pass/Fail | w_c=0.2 | 是否编译通过（算子级一次） |
| 功能正确性 | 精度用例通过 | w_f=0.3 | 单用例是否通过精度标准测试 |
| 性能优化性 | 性能用例评分（HAP） | w_p=0.5 | 见下方公式 |

**单用例性能得分 HAP（Hardware-Anchored Performance，硬件锚定性能）** ：HAP 以硬件理论性能上界 $T_{\text{HW}}$ 为锚点对候选 kernel 计分，因此得名“硬件锚定”。

$$
\text{HAP}_i = \frac{T_{\text{baseline},i} - T_{\text{HW},i}}{(T_{\text{cand},i} - T_{\text{HW},i}) + (T_{\text{baseline},i} - T_{\text{HW},i})}
$$

其中 `T_HW = t_hw_us` 为硬件理论性能上界，已在 cases.yaml 中提供；`T_baseline = baseline_perf_us` 为CANN基线性能，也已在 cases.yaml 中提供（由于torch接口功能限制，部分基线实现由算子拼接得到）；`T_cand` 为候选 kernel 实测时间。这个公式的设计保证了如果性能低于基线（T_cand > T_baseline），HAP 为 0-0.5；如果性能优于基线（T_cand < T_baseline），HAP 0.5 以上；如果性能达到硬件上界或更高（T_cand <= T_HW），HAP 为 1 以上。

> **HAP 是饱和型指标，不是加速比（speedup）。** 它衡量的是“候选 kernel 逼近硬件理论上界的程度”，而非“比 baseline 快多少倍”。当 baseline 本身很慢（$T_{\text{baseline}} \gg T_{\text{HW}}$）时，即使候选相对 baseline 有巨大的 speedup，HAP 也只会趋近于 1 附近而不会线性放大；只有 $T_{\text{cand}} < T_{\text{HW}}$（快于硬件理论上界）时 HAP 才会 > 1。
>
> **边界与异常取值。** 当锚点非法（$T_{\text{cand}} \le 0$ 或 $T_{\text{HW}} \le 0$），或分母 $(T_{\text{cand}}-T_{\text{HW}}) + (T_{\text{baseline}}-T_{\text{HW}}) \le 0$（罕见，例如 $T_{\text{baseline}} \approx T_{\text{HW}}$ 且 $T_{\text{cand}} < T_{\text{HW}}$）时，当前实现对该用例的 HAP 返回 `None`（该用例不计入性能项），**不会输出 $\pm\infty$ 或负分**。
![HAP 示例曲线](docs/assets/perf_score.png)
**单算子综合评分** ：

$$
\text{EachOperatorScore} = \left[ w_c \cdot \delta_{\text{pass}} + \sum_{i \in \text{cases}} \frac{\delta_{\text{accuracy},i} (w_f + w_p \cdot \text{HAP}_i)}{|\text{cases}|} \right] \cdot 100
$$

- 在 $T_{\text{cand}} \ge T_{\text{HW}}$ 的常规物理有效区间内，单算子满分约 100；若候选快于硬件理论上界（$T_{\text{cand}} < T_{\text{HW}}$），HAP 与算子分**允许超过 100**（不做上界截断）。编译失败时 `δ_pass=0` ⇒ 整算子计 0；某个用例精度不过则只扣除该用例得分（`δ_accuracy,i=0`），HAP 按用例总和计算。
- Level-N 得分 = 该 level 内算子 EachOperatorScore 总合
- benchmark 总分 = 全部算子 EachOperatorScore（= Level1 + Level2 + Level3 + Level4）总和

## 🔍 Directory Structure

```
cann-bench/
├── tasks/                  # 评测主集合（algo desc / golden / cases / proto）
│   ├── level1/             # L1 基础算子（Elementwise / Activation）
│   ├── level2/             # L2 中级算子（Norm / Reduce / Gather/Scatter）
│   ├── level3/             # L3 高级算子（Conv / Pool / MoE）
│   └── level4/             # L4 复杂算子（Attention / RNN）
├── bench_lab/              # 实验/孵化区，本地测试通过后转入 tasks
├── examples/               # 算子工程样例 & 评测任务样例
│   ├── aclnn_launch_example/    # ACLNN 自定义算子样例
│   ├── direct_launch_example/   # Direct launch 算子样例
│   ├── eval_task/               # 评测任务样例（add/sqrt fixture，验证评测流水线）
│   └── stanfordbench_example/   # Stanford benchmark 样例
├── docs/                   # 规范 / 设计 / 指南文档
├── scripts/                # 评测/测试入口脚本
│   ├── run_evaluation.sh   # AI 算子评测入口（编译→安装→评测）
│   ├── run_test.sh         # 测试入口（ut/e2e）
│   └── utils/              # 工具脚本
│       ├── gen_report.sh   # 从 JSON 生成评测报告
├── src/                    # 源代码
│   └── kernel_eval/        # 算子评测模块（CLI / 数据 / 评测 / 报告 / 安全）
├── tests/                  # 测试代码（含 docs/test_report.md）
├── requirements.txt        # Python 依赖
├── LICENSE                 # 许可证
└── README.md               # 本文档
```

## 🔧 Setup

### 环境要求

- 910B / 910_93 / 950 开发环境

## 🚀 Quick Start
### 1. 配置环境
1. **安装依赖**
- 确保已安装CANN开发环境 和TorchNpu扩展
- 确保已安装Ascend C 算子开发工具链。
```bash
# 安装依赖
pip install -r requirements.txt
```

2. **克隆项目仓库**
   ```bash
   git clone https://gitcode.com/cann/cann-bench.git
   cd cann-bench
   ```
### 2. 生成算子代码
参考 examples 中的算子工程样例，根据算子规格描述生成 Ascend C 算子代码工程（例如 `generated_project/`）。
- **ACLNN 算子工程样例**：[examples/aclnn_launch_example/](examples/aclnn_launch_example/)
- **直调算子工程样例**：[examples/direct_launch_example/](examples/direct_launch_example/)

### 3. 运行评测
1. **准备测试用例**
   `tasks/levelN/<op>/` 下已包含 `proto.yaml`、`golden.py`、`cases.yaml`、`cases.csv`、`desc.md`。无需另外生成。

2. **运行评测脚本**
   ```bash
   # 从 AI 生成的源码目录评测（自动扫描、编译、安装、评测）
   ./scripts/run_evaluation.sh --source-dir generated_project

   # 或位置参数形式：
   ./scripts/run_evaluation.sh generated_project
   ```
    评测报告输出到 `reports/`（含 `eval_report.json`、`eval_report.md`、`eval_report.html`、`summary.md`）。

    更多用法见 [docs/guide/quick_start.md](docs/guide/quick_start.md)。

3. **生成评测报告**（从已有 JSON 重新生成）
   ```bash
   # 使用默认模板 (tasks/description.html)
   ./scripts/utils/gen_report.sh reports/cann/eval_xxx.json

   # 指定自定义模板
   ./scripts/utils/gen_report.sh --json eval.json --template custom/index.html
   ```

## 📋 Test Case Structure

每个算子目录下包含以下文件作为开发算子的输入文件：

| 文件 | 说明 |
|------|------|
| `cases.csv` | 测试用例 CSV 格式 |
| `cases.yaml` | 测试用例 YAML 格式（内容与CSV相同，方便Agent读取） |
| `golden.py` | PyTorch 参考实现，用于结果验证 |
| `proto.yaml` | 算子原型定义 |
| `desc.md` | 算子详细说明文档 |

### 待评测算子工程样例与评测任务样例

项目提供了多种示例：

**算子实现样例**（教你"怎么写算子"）：

1. **ACLNN 算子启动示例**：[examples/aclnn_launch_example/](examples/aclnn_launch_example/)
2. **直调算子启动示例**：[examples/direct_launch_example/](examples/direct_launch_example/)

**评测任务样例**（教你"怎么评算子"）：

3. **评测任务 fixture**：[examples/eval_task/](examples/eval_task/) — 提供 Add/Sqrt 两个轻量算子的评测定义（proto.yaml + golden.py + cases.yaml），用于验证评测流水线是否正常工作

使用自定义算子：

```python
import torch
import torch_npu
import cann_bench

x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()
result = cann_bench.add(x, y)
```

## 👥 社区贡献

我们欢迎社区开发者贡献新的评测任务，共同丰富CANN领域的评测体系。以下是贡献新评测任务的详细流程：

### 贡献流程

1. **创建算子目录**
   在 `tasks/levelN/` 下创建算子目录（如 `tasks/level2/my_op/`）；或先在 `bench_lab/` 实验/孵化区暂存，测试通过后 PR 转入主集合。详见 [docs/guide/contributing.md](docs/guide/contributing.md)。

2. **准备核心文件**  
   每个算子目录需包含以下文件作为评测输入：

   | 文件                | 说明                                                                 |
   |---------------------|----------------------------------------------------------------------|
   | `proto.yaml`        | 算子原型定义，包括输入输出张量形状、数据类型、属性参数等             |
   | `golden.py`         | PyTorch参考实现，用于功能正确性验证（需覆盖所有测试用例场景）        |
   | `desc.md`           | 算子详细说明文档，包括功能描述、数学公式、实现约束、参考资料等       |
   | `cases.csv` 或 `cases.yaml` | 测试用例定义，包含输入数据、预期输出、性能基线等信息（推荐Yaml格式） |

3. **提交PR**  
   将算子目录提交至主仓库，PR需包含：
   - 完整的算子评测文件
   - 简要的功能说明和测试验证结果
   - （可选）算子实现难度分级建议（Level 1-4）

### 贡献规范

- **算子原型**：需符合Ascend C算子开发规范，支持动态shape、数据类型兼容性等特性。
- **参考实现**：需通过PyTorch官方接口实现，确保在CPU/NPU环境可正确运行。
- **测试用例**：需覆典型场景。
- **文档描述**：需清晰说明算子功能、适用场景及与同类算子的差异。

## 🛣️ Roadmap

- 工程平台构建
  - [ ] 建立持续评测 CI 流水线
  - [ ] 发布官方评测网站
  - [ ] 评测结果可视化展示，用户友好的分析工具

- 评测集构建
  - [ ] 增加更多算子类型覆盖
  - [ ] 根据领域场景分类，算子特征等，构建出更多独立榜单集合，覆盖不同评测场景的需求

- 评测标准构建
  - [ ] 评测精度标准，精度衡量方法构建
  - [ ] 理论性能评估方法构建
  - [ ] 评分算法优化（算子分级/分类方法），科学评价生成能力
  - [ ] 防作弊体系构建


## 🪪 License

CANN Open Software License Agreement Version 2.0