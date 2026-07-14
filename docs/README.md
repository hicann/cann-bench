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
| [cases_yaml_spec.md](spec/cases_yaml_spec.md) | cases.yaml 字段与顺序约定（占位/空值规则） |

> proto.yaml 的 schema 说明见 [contributing.md §1](guide/contributing.md)（暂无独立 `api_spec.md`）。

### 设计类文档 (design/)

面向评测器开发者，定义评测工程架构设计。

| 文档 | 说明 |
|------|------|
| [kernel_eval_architecture.md](design/kernel_eval_architecture.md) | kernel_eval 整体架构：分层（base/benches/checkers/data/eval/registry/report/security/utils）与数据流 |
| [module_panorama.md](design/module_panorama.md) | 模块全景：各子包职责与依赖关系 |
| [evaluator_design.md](design/evaluator_design.md) | 评测工程设计：评测器架构、安全防护、报告生成 |
| [precision_comparison_design.md](design/precision_comparison_design.md) | 精度对比设计：MERE/MARE、小值域/相消判定、阈值表 |
| [report_design.md](design/report_design.md) | 报告系统设计：评分计算、summary/HTML 生成 |
| [perf_collection_design.md](design/perf_collection_design.md) | 性能采集设计：NPU Profiler、Trace 解析、升频清 Cache |
| [baseline_collection_design.md](design/baseline_collection_design.md) | baseline 采集设计：collect_baseline 脚本与 metadata 产物 |
| [multi_card_parallel_analysis.md](design/multi_card_parallel_analysis.md) | 多卡并行评测分析 |
| [micro_benchmark_selection.md](design/micro_benchmark_selection.md) | 微基准/算子选型分析 |
| [benchmark_orchestrator_architecture.md](design/benchmark_orchestrator_architecture.md) | auto_pipeline 设计：core/prompt/generator/converter 边界与数据流 |

### 指南类文档 (guide/)

面向使用者，提供操作指南。

| 文档 | 说明 |
|------|------|
| [contributing.md](guide/contributing.md) | 算子贡献指南：如何提交新算子评测任务（含 proto.yaml schema） |
| [quick_start.md](guide/quick_start.md) | 快速入门：评测流程和命令行使用 |
| [submission_rules.md](guide/submission_rules.md) | 算子提交原则与禁止行为：说明哪些实现方式会被视为无效或作弊 |
| [version_policy.md](guide/version_policy.md) | 版本策略：VERSION / tasks 版本与兼容性 |
| [custom_benchmark_integration.md](guide/custom_benchmark_integration.md) | 自定义评测集接入指南 |
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
