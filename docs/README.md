# CANN 算子代码生成评测文档

**文档版本：参见 [changelog](changelog.md)**

本目录包含算子代码生成评测体系的完整文档，按主题分类组织。

## 当前实现与文档范围

- 正式集合为 Season2，60 题、1200 个公开 case；旧 53 题在 Season1 归档。范围与平台排除规则见 [Season2 迁移说明](guide/season2_migration.md)。
- 框架版本位于根 `VERSION`，实际题集版本位于 `<bench_root>/metadata/VERSION`；权重随实际题集大版本选择，见 [版本策略](guide/version_policy.md)。
- 默认 NPU 脚本分阶段评测，并支持批量 case Profiler 与逐 case 回退。调度/隔离边界应以当前实现为准，不能沿用“已取消所有用例级隔离”的旧结论。
- `guide/` 和 `spec/` 为操作与契约入口；`design/`、`refactor/` 中的提案、统计和实验结果有各自历史基线，不能把示意类/API 或尚未实现的方案当作当前接口。具体差异见相应文档开头的现状说明。
- MC2 有独立的参考健康、通信域与硬件资源约定，见 [MC2 指南](guide/mc2_accuracy.md)；可选参考缓存见 [tensor cache](design/tensor_cache.md)。

## 文档索引

### 规范类文档 (spec/)

面向算子定义者和贡献者，定义评测基准标准。

| 文档 | 说明 |
|------|------|
| [benchmark_spec.md](spec/benchmark_spec.md) | 评测基准规范：算子定义、用例设计、精度标准、评分规则 |
| [cases_yaml_spec.md](spec/cases_yaml_spec.md) | cases.yaml 字段与顺序约定（占位/空值规则） |
| [submission_spec.md](spec/submission_spec.md) | 评测输入与接口规范：解包后的源码、构建产物、接口发现和 auto-pipeline 规则编号 |

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
| [triton_ascend_integration_implementation.md](design/triton_ascend_integration_implementation.md)（[HTML 汇报页](design/triton_ascend_integration_implementation.html)） | Triton-Ascend 加载、JIT、缓存、执行与核心源码详解 |

### 指南类文档 (guide/)

面向使用者，提供操作指南。

| 文档 | 说明 |
|------|------|
| [contributing.md](guide/contributing.md) | 算子贡献指南：如何提交新算子评测任务（含 proto.yaml schema） |
| [quick_start.md](guide/quick_start.md) | 快速入门：评测流程和命令行使用 |
| [triton_ascend_quick_start.md](guide/triton_ascend_quick_start.md) | Triton-Ascend 环境、构建和真机验证快速指南 |
| [submission_rules.md](guide/submission_rules.md) | 算子提交行为与禁止规则：提交前必读，统一说明检测范围、提交约束、违规处理、规则编号及示例 |
| [version_policy.md](guide/version_policy.md) | 框架/实际题集版本、权重选择和发布验收 |
| [season2_migration.md](guide/season2_migration.md) | 当前正式题集范围、metadata 状态与平台限制 |
| [mc2_accuracy.md](guide/mc2_accuracy.md) | MC2 多卡参考、精度及资源约定 |
| [custom_benchmark_integration.md](guide/custom_benchmark_integration.md) | 自定义评测集接入指南 |
| [auto_pipeline_usage.md](guide/auto_pipeline_usage.md) | auto_pipeline 使用指南：CLI、配置、环境变量、输出目录 |
| [auto_pipeline_agent_integration.md](guide/auto_pipeline_agent_integration.md) | auto_pipeline 新 agent 接入指南：通用 code agent + skills 与 LangGraph workflow 两类路径 |

### 版本记录

| 文档 | 说明 |
|------|------|
| [changelog.md](changelog.md) | 版本变更记录 |

## 快速导航

**我是评测提交实现者** → [submission_spec.md](spec/submission_spec.md) + [submission_rules.md](guide/submission_rules.md) + [quick_start.md](guide/quick_start.md)

**我是算子贡献者** → [contributing.md](guide/contributing.md) + [benchmark_spec.md](spec/benchmark_spec.md)

**我是评测器开发者** → [evaluator_design.md](design/evaluator_design.md) + [benchmark_orchestrator_architecture.md](design/benchmark_orchestrator_architecture.md) + [auto_pipeline_agent_integration.md](guide/auto_pipeline_agent_integration.md)

**我是使用者** → [quick_start.md](guide/quick_start.md) + [submission_rules.md](guide/submission_rules.md) + [auto_pipeline_usage.md](guide/auto_pipeline_usage.md)

## 评测体系概述

本评测体系用于量化评估 AI 生成的 Ascend C 算子代码质量，涵盖三个核心维度：

- **编译/运行正确性**：算子代码能否成功编译链接，并在用例上按接口约定运行
- **精度正确性**：算子输出与 Golden 结果的数值精度偏差
- **性能优化性**：由候选设备耗时、baseline 和硬件耗时下界计算 HAP；不等同于 speedup

详细设计请参阅 [benchmark_spec.md](spec/benchmark_spec.md) 和 [evaluator_design.md](design/evaluator_design.md)。
