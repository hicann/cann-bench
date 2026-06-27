# CANN 算子代码生成评测文档

**文档版本：参见 [changelog](changelog.md)**

本目录包含算子代码生成评测体系的完整文档，按主题分类组织。

**更新说明参见 [changelog](changelog.md)**：
- Profiling 升级为 Level1/Level2（删除 Level0）
- 提交规则与安全边界补充
- Config 依赖注入（支持多实例并行评测）
- 进程隔离简化（取消用例级，保留算子级）
- YAML 格式校验（WARNING 输出）
- CLI 参数增强（--device、--warmup、--repeat、--profiler-level）

## 文档索引

### 规范类文档 (spec/)

面向算子定义者和贡献者，定义评测基准标准。

| 文档 | 说明 |
|------|------|
| [benchmark_spec.md](spec/benchmark_spec.md) | 评测基准规范：算子定义、用例设计、精度标准、评分规则 |
| [api_spec.md](spec/api_spec.md) | 算子接口规范：proto.yaml schema 说明（待补充） |

### 设计类文档 (design/)

面向评测器开发者，定义评测工程架构设计。

| 文档 | 说明 |
|------|------|
| [evaluator_design.md](design/evaluator_design.md) | 评测工程设计：评测器架构、安全防护、报告生成 |
| [benchmark_orchestrator_architecture.md](design/benchmark_orchestrator_architecture.md) | auto_pipeline 设计：core/prompt/generator/converter 边界与数据流 |
| [perf_collection_design.md](design/perf_collection_design.md) | 性能采集设计：NPU Profiler、Trace 解析、升频清 Cache |

### 指南类文档 (guide/)

面向使用者，提供操作指南。

| 文档 | 说明 |
|------|------|
| [contributing.md](guide/contributing.md) | 算子贡献指南：如何提交新算子评测任务 |
| [quick_start.md](guide/quick_start.md) | 快速入门：评测流程和命令行使用（待补充） |
| [submission_rules.md](guide/submission_rules.md) | 算子提交原则与禁止行为：说明哪些实现方式会被视为无效或作弊 |
| [auto_pipeline_usage.md](guide/auto_pipeline_usage.md) | auto_pipeline 使用指南：CLI、配置、环境变量、输出目录 |
| [auto_pipeline_agent_integration.md](guide/auto_pipeline_agent_integration.md) | auto_pipeline 新 agent 接入指南：通用 code agent + skills 与 LangGraph workflow 两类路径 |

### 版本记录

| 文档 | 说明 |
|------|------|
| [changelog.md](changelog.md) | 版本变更记录 |

## 快速导航

**我是算子贡献者** → [contributing.md](guide/contributing.md) + [submission_rules.md](guide/submission_rules.md) + [benchmark_spec.md](spec/benchmark_spec.md)

**我是评测器开发者** → [evaluator_design.md](design/evaluator_design.md) + [benchmark_orchestrator_architecture.md](design/benchmark_orchestrator_architecture.md) + [auto_pipeline_agent_integration.md](guide/auto_pipeline_agent_integration.md)

**我是使用者** → [quick_start.md](guide/quick_start.md) + [submission_rules.md](guide/submission_rules.md) + [auto_pipeline_usage.md](guide/auto_pipeline_usage.md)

## 评测体系概述

本评测体系用于量化评估 AI 生成的 Ascend C 算子代码质量，涵盖三个核心维度：

- **编译/运行正确性**：算子代码能否成功编译链接，并在用例上按接口约定运行
- **精度正确性**：算子输出与 Golden 结果的数值精度偏差
- **性能优化性**：生成算子与基准性能的比例

详细设计请参阅 [benchmark_spec.md](spec/benchmark_spec.md) 和 [evaluator_design.md](design/evaluator_design.md)。
