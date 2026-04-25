# cann-bench: CANN 领域评测

评测AI在处理CANN领域代码任务的能力，涵盖算子生成、算子优化等领域，支撑模型选型、训练效果评估，统一量化评估标准，识别Agent能力短板，构建CANN领域评测平台，推动AI能力在CANN领域的持续演进。

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
| 编译正确性 | Pass/Fail | Wc=2 | 是否编译通过 |
| 功能正确性 | 精度用例通过数 | Wf=3 | 通过精度用例的数量 |
| 性能优化性 | 加速比 (SpeedUp) | Wp=5 | 验证性能/测试基准性能 |

单算子综合评分 = 编译通过得分 + 功能通过用例数 × (功能得分 + 性能得分)
  - 编译通过得分 = compile_pass × Wc    # 单算子一次，整份提交的编译结果
  - 功能得分     = Wf                   # 每个功能通过的用例
  - 性能得分     = SpeedUp × Wp         # 每个功能通过的用例（按该用例实测）

Level-N 得分 = 该 level 内所有算子综合评分之和
benchmark 总分 = 所有算子综合评分之和（= Level1 + Level2 + Level3 + Level4）

## 🔍 Directory Structure

```
cann-bench/
├── kernel_bench/           # 算子生成评测任务
│   ├── level1/             # 基础算子
│   ├── ...                 # 中级算子
│   └── level4/             # 复杂算子
├── bench_lab/              # 实验室级测试用例(后续版本会规划进主评测集)
│   └── kernel_bench/       # 实验室级算子评测任务
├── examples/               # 示例代码工程
│   ├── aclnn_launch_example/        # ACLNN 算子工程样例
│   └── direct_launch_example/       # 直接算子工程样例
├── docs/                   # 设计文档
├── scripts/                # 测试脚本
│   ├── run_test.sh         # 统一测试运行脚本
│   └── run_evaluation.py   # 评测运行脚本
├── src/                    # 源代码
│   └── kernel_eval/        # 算子评测模块
├── test/                   # 测试代码
├── requirements.txt        # Python 依赖
├── LICENSE                 # 许可证文件
└── README.md               # 项目说明文档
```

## 🔧 Setup

### 环境要求

- 910B开发环境

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
   git clone https://github.com/yourusername/cann-bench.git
   cd cann-bench
   ```
### 2. 生成算子代码
参考 examples 中的算子工程样例，根据算子规格描述生成 Ascend C 算子代码工程（例如generated_project）。
- **ACNN 算子工程样例**：[examples/aclnn_launch_example/](examples/aclnn_launch_example/)
- **直接算子工程样例**：[examples/direct_launch_example/](examples/direct_launch_example/)

### 3. 运行评测
1. **准备测试用例**  
   确保 `kernel_bench` 目录下包含测试用例文件 `cases.csv`、`golden.py`、`proto.yaml`、`desc.md`。

2. **运行评测脚本**  
   ```bash
   python scripts/run_evaluation.py --task kernel_bench --source-dir generated_project
   ```
   评测结果将在 `results/` 目录下生成。

## 📋 Test Case Structure

每个算子目录下包含以下文件作为开发算子的输入文件：

| 文件 | 说明 |
|------|------|
| `cases.csv` | 测试用例 CSV 格式 |
| `golden.py` | PyTorch 参考实现，用于结果验证 |
| `proto.yaml` | 算子原型定义 |
| `desc.md` | 算子详细说明文档 |

### 待评测算子工程样例

项目提供了多种算子开发示例：

1. **ACNN 算子启动示例**：[examples/aclnn_launch_example/](examples/aclnn_launch_example/)
2. **直接算子启动示例**：[examples/direct_launch_example/](examples/direct_launch_example/)

这些示例演示如何使用 Ascend C 和 PyTorch Extension 开发自定义 NPU 算子。

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
   在 `bench_lab/` 目录下创建以任务名（例如 `driver_bench`）+算子名称命名的文件夹（例如 `bench_lab/driver_bench/my_new_op/`）。

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