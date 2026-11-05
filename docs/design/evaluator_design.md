# 算子评测工程设计

**文档版本：V0.4.0**

**更新摘要（自旧版主要变化）**：
- 基类层分离：新增 `base/` 目录统一存放抽象基类，特化类移至 `benches/`
- 注册层完善：新增 `registry/` 目录，统一管理所有组件的注册机制
- 评测集插件化：新增 `benches/` 目录，支持 CANN / Stanford 多评测集
- Checker 层独立：从 benches 中拆出 `checkers/`，通用标准不与评测集绑定
- 性能策略模式：`base/perf_strategy.py` 定义策略基类，三种口径策略可按场景切换
- TorchOpGuard：新增 `security/torch_op_guard.py`，检测 AI 算子直接调用 PyTorch 内置计算 API
- BaselineStore：集中式 baseline 存储，从 `metadata/<hardware>.json` 加载，支持多硬件按平台扩展
- HTML 报告：新增 `report/html_generator.py`，生成带 KPI/柱状图/认证印章的完整 HTML 报告
- 渐进设备恢复：Evaluator.run_cases 新增 healthy→recovering→unrecoverable 三阶段恢复机制
- 增量输出 + OOM 恢复：子进程每用例完成后增量写入，OOM Kill 时主进程可恢复部分结果
- Golden 精度策略：`BenchConfig.golden_precision` 三档可选（fp64_cpu / native_cpu / native_npu）
- 确定性种子：`Config.eval_seed` 默认 0，基于 case_id hash 确保跨进程可复现
- ProcessPoolCoordinator：TaskUnit 调度 + eval-child 子命令 + ThreadPoolExecutor 多卡并行
- FailureSynthesizer：统一合成编译失败 / 安全失败 / 子进程失败 / OOM 失败结果

## 目录
- [1. Context](#1-context)
- [2. 方案设计](#2-方案设计)
- [3. 源码目录评测流程](#3-源码目录评测流程)
- [4. 核心能力设计](#4-核心能力设计)
- [5. 实施步骤](#5-实施步骤)
- [6. 验证方案](#6-验证方案)
- [7. 附录](#7-附录)

---

## 1. Context

### 1.1 背景

根据 `docs/spec/benchmark_spec.md` 设计文档，构建一套AI生成Ascend C算子代码评测体系，用于量化评估AI生成的算子代码质量，涵盖编译正确性、功能正确性、性能优化性三个核心维度。

### 1.2 两工程架构设计

本评测体系分为两个独立工程，通过whl包进行传递：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI自动生成工程                                     │
│  参考 examples/fast_kernel_launch_example 结构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  project_root/                                                              │
│  ├── setup.py              # 构建配置（PyTorch Extension）                  │
│  ├── CMakeLists.txt        # CMake构建文件                                  │
│  ├── build.sh              # 编译脚本（可选）                                │
│  ├── cann_bench/           # Python包（约定命名）                            │
│  │   └── __init__.py       # 导入_C扩展模块                                 │
│  ├── csrc/                 # 算子源码目录                                    │
│  │   ├── exp/              # Exp算子                                        │
│  │   │   └ ascend910b/                                                     │
│  │   │     ├── CMakeLists.txt                                              │
│  │   │     └── exp.cpp      # AI生成的Ascend C代码                         │
│  │   └── ...                                                               │
│  └── dist/                   # 构建产物                                     │
│  │   ├── cann_bench_xxx.whl  # Python包                                   │
│  │   └── cann_bench_xxx.run   # NPU内核包（可选）                          │
│                                                                             │
│  算子接口约定：torch.ops.cann_bench.exp(x, ...) 或 cann_bench.exp()        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ cann_bench_xxx.whl + cann_bench_xxx.run
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           评测工程（src/kernel_eval）                         │
│  本方案设计目标                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  src/kernel_eval/                                                            │
│  ├── cli.py                # 命令行入口                                     │
│  ├── config.py             # 配置管理                                       │
│  ├── simulation.py         # CPU 仿真评测（无 NPU 环境下功能验证）           │
│  ├── _version.py           # 版本管理（从 VERSION 文件读取）                │
│  │                                                                          │
│  ├── base/                 # 基类层（抽象基类，无特化依赖）                  │
│  │   ├── enums.py          # 通用枚举（DifficultyLevel/BackendType/…）      │
│  │   ├── models.py         # 数据模型基类（TaskSpec/CaseSpec/…）            │
│  │   ├── loaders.py        # 加载器基类（TaskLoader/CaseLoader/…）          │
│  │   ├── matcher.py        # 匹配器基类（OperatorMatcherBase）             │
│  │   ├── checker.py        # 精度判断器基类（CorrectnessChecker）           │
│  │   ├── result.py         # 评测结果基类（AccuracyResult/PerfResult/…）    │
│  │   ├── scoring.py        # 评分方案基类（ScoringScheme）                  │
│  │   └── perf_strategy.py  # 性能指标策略基类（PerfMetricStrategy）         │
│  │                                                                          │
│  ├── registry/             # 注册层（组件注册表 + 便捷获取函数）             │
│  │   ├── base.py           # BaseRegistry 泛型基类                          │
│  │   ├── bench_registry.py # 评测集配置注册表（BenchConfig）                │
│  │   ├── loader_registry.py# 加载器注册表                                   │
│  │   ├── golden_registry.py# Golden 加载器注册表                            │
│  │   ├── matcher_registry.py# 算子匹配器注册表                              │
│  │   ├── checker_registry.py# Checker 注册表                                │
│  │   ├── scoring_registry.py# 评分方案注册表                                │
│  │   ├── case_spec_registry.py# CaseSpec 子类注册表                         │
│  │   └── perf_strategy_registry.py# 性能策略注册表                          │
│  │                                                                          │
│  ├── benches/              # 评测集插件层（特化实现 + 注册）                 │
│  │   ├── cann.py           # CANN 评测集入口（导出 + 注册）                  │
│  │   ├── cann_loader.py    # CANN 加载器（CannTaskLoader/CannCaseLoader/…） │
│  │   ├── cann_spec.py      # CANN 数据模型（CannTaskSpec/CannCaseSpec/…）   │
│  │   ├── cann_matcher.py   # CANN 算子匹配器                                │
│  │   ├── cann_scoring.py   # CANN 评分方案                                  │
│  │   ├── cann_solution.py  # CANN 解决方案规格                              │
│  │   ├── stanford.py       # Stanford 评测集入口                            │
│  │   ├── stanford_loader.py# Stanford 加载器                                │
│  │   ├── stanford_matcher.py# Stanford 匹配器                               │
│  │   └── stanford_scoring.py# Stanford 评分方案                             │
│  │                                                                          │
│  ├── checkers/             # Checker 层（通用精度判断器，不绑定评测集）      │
│  │   ├── relative_error_checker.py # MERE/MARE + 小值域 + 相消处理          │
│  │   └── allclose_checker.py       # torch.allclose 简化对比               │
│  │                                                                          │
│  ├── data/                 # 数据层                                         │
│  │   ├── data_generator.py # 数据生成（确定性种子，shape/dtype/value_range） │
│  │   └── package_manager.py# 包管理（源码扫描、编译、安装、接口扫描）       │
│  │                                                                          │
│  ├── eval/                 # 评测层                                         │
│  │   ├── evaluator.py      # 综合评测调度器（Config+BenchConfig 依赖注入）  │
│  │   ├── accuracy_eval.py  # 精度评测（Checker 三输入模式）                 │
│  │   ├── perf_eval.py      # 性能评测（Profiler + Strategy 解析）          │
│  │   ├── op_runner.py      # 算子执行器（TorchOpGuard 集成）                │
│  │   ├── input_pool.py     # 输入池管理（防缓存攻击）                       │
│  │   ├── results.py        # 评测结果数据类（增量输出 + failure_type）       │
│  │   ├── failure_synthesizer.py # 失败结果合成器                            │
│  │   ├── process_pool.py   # 多卡并行协调器（TaskUnit 调度）                │
│  │   └── subprocess_utils.py # OOM 保护 + CANN 环境变量 + 部分结果恢复     │
│  │                                                                          │
│  ├── security/             # 安全层                                         │
│  │   ├── api_guard.py      # Timing API 防护                                │
│  │   ├── torch_op_guard.py # Torch 算子守卫（检测 PyTorch 内置 API 调用）   │
│  │   └── type_checker.py   # 返回值类型检查                                 │
│  │                                                                          │
│  ├── report/               # 报告层                                         │
│  │   ├── report_generator.py # 评测报告生成器（JSON+Markdown+Summary+HTML） │
│  │   ├── html_generator.py   # HTML 报告渲染（KPI+柱状图+认证印章）         │
│  │   ├── scoring.py          # 评分计算（hardware-anchored Eq.3/4/5）       │
│  │   ├── summary_generator.py# Summary 生成                                 │
│  │   └── setup_info.py       # 评测配置采集（NPU/CANN/PyTorch 版本信息）    │
│  │                                                                          │
│  └ utils/                  # 工具层                                         │
│  │   ├── device_manager.py  # 设备管理（渐进恢复：light→full→unrecoverable）│
│  │   ├── baseline_resolver.py# Baseline 解析（多硬件 dict + fallback）      │
│  │   ├── baseline_store.py   # BaselineStore（metadata/<hw>.json 加载）     │
│  │   ├── compare.py          # 张量对比引擎（MERE/MARE+小值域+相消）        │
│  │   ├── thresholds.py       # 精度阈值表（单一事实来源）                    │
│  │   ├── param_builder.py    # 参数构建                                     │
│  │   ├── dtype_mapper.py     # 数据类型映射                                 │
│  │   ├── naming.py           # PascalCase→snake_case 转换（多候选模糊匹配） │
│  │   ├── path_resolver.py    # 路径解析                                     │
│  │   ├── tensor_utils.py     # 张量工具（tensors_to_cpu/tensors_to_fp64）   │
│  │                                                                          │
│  评测流程：                                                                  │
│  1. 安全初始化（APIGuard 快照 + TorchOpGuard 配置）                         │
│  2. 扫描源码目录（检查build.sh、dist目录）                                   │
│  3. 检查dist是否有whl包/run包，无则执行build.sh编译                          │
│  4. 迭代隔离编译（失败的算子自动隔离到 _quarantine/）                        │
│  5. 安装run包（NPU内核包）+ whl包（Python包）                                │
│  6. 安全验证（APIGuard完整性检查 + TorchOpGuard 模式配置）                   │
│  7. 扫描cann_bench接口，打印接口信息                                         │
│  8. 加载 tasks 用例数据（通过 Registry 选择 BenchConfig 组件）               │
│  9. 执行Golden函数（精度策略由 BenchConfig.golden_precision 控制）           │
│  10. 执行AI算子（TorchOpGuard 监听 + profiler 采集）                        │
│  11. 精度对比（Checker 三输入：AI/Golden/同精度参考）                        │
│  12. 防作弊二次验证（可选，Config.enable_accuracy_retry）                    │
│  13. 性能评测（Profiler + PerfMetricStrategy 解析）                         │
│  14. 渐进设备恢复（healthy→recovering→unrecoverable）                       │
│  15. 生成评测报告（JSON + Markdown + HTML）                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键约定**：
1. AI生成工程包名统一为 `cann_bench`
2. 算子接口与proto.yaml的schema一致
3. 评测工程通过 `--source-dir` 参数指定源码目录，自动扫描编译安装whl包和run包
4. 安全机制防止作弊攻击（APIGuard + TorchOpGuard + 二次验证 + InputPool）
5. 高精度Golden计算（CPU fp64 / native_cpu / native_npu 三档策略）
6. 评测集通过 `--bench-name` 参数选择，默认 `cann`，支持 `stanford` 等其他评测集
7. Baseline 数据通过 `metadata/<hardware>.json` 集中管理，支持多硬件按平台扩展

**命名说明**：
- `kernel_eval`：评测工程代码目录（src/kernel_eval）
- `tasks`：测试用例数据目录（tasks/level*/op_name/）
- `./scripts/run_evaluation.sh`：CLI命令脚本（推荐使用）

---

## 2. 方案设计

### 2.1 工程架构

```
src/kernel_eval/
├── __init__.py              # 包入口，导出公共API + 版本号
├── cli.py                   # 命令行入口（eval/list/info/config/eval-child）
├── config.py                # 配置管理（Config 数据类 + 全局配置）
├── simulation.py            # CPU 仿真评测模块
├── _version.py              # 版本管理（从 VERSION 文件动态读取）
│
├── base/                    # 基类层
│   ├── __init__.py          # 导出所有基类
│   ├── enums.py             # 通用枚举（DifficultyLevel/BackendType/SourceType/GoldenReference/EvaluationMode）
│   ├── models.py            # 数据模型基类（AttrSpec/InputSpec/OutputSpec/TaskSpec/CaseSpec/SolutionSpec）
│   ├── loaders.py           # 加载器基类（TaskLoader/CaseLoader/OperatorDirMixin/GoldenLoaderBase）
│   ├── matcher.py           # 匹配器基类（OperatorMatcherBase）
│   ├── checker.py           # 精度判断器基类（CorrectnessChecker）
│   ├── result.py            # 评测结果基类（OutputResult/AccuracyResult/PerfResult + 注册表）
│   ├── scoring.py           # 评分方案基类（ScoringScheme/CaseScoreInfo）
│   └── perf_strategy.py     # 性能指标策略基类（PerfMetricStrategy/ProfFileLocations/三种策略实现）
│
├── registry/                # 注册层
│   ├── __init__.py          # 导出所有注册表 + 便捷获取函数
│   ├── base.py              # BaseRegistry 泛型基类（register/get/list_all/clear）
│   ├── bench_registry.py    # BenchConfig + BenchRegistry（评测集配置聚合）
│   ├── loader_registry.py   # TaskLoader/CaseLoader 注册表
│   ├── golden_registry.py   # GoldenLoader 注册表
│   ├── matcher_registry.py  # OperatorMatcher 注册表
│   ├── checker_registry.py  # CorrectnessChecker 注册表
│   ├── scoring_registry.py  # ScoringScheme 注册表
│   ├── case_spec_registry.py # CaseSpec 子类注册表
│   └── perf_strategy_registry.py # PerfMetricStrategy 注册表
│
├── benches/                 # 评测集插件层
│   ├── __init__.py          # 导入所有评测集模块触发注册
│   ├── cann.py              # CANN 评测集入口（导出 + BenchRegistry 注册）
│   ├── cann_loader.py       # CANN 加载器实现
│   ├── cann_spec.py         # CANN 特化数据模型
│   ├── cann_matcher.py      # CANN 算子匹配器实现
│   ├── cann_scoring.py      # CANN 评分方案实现（Eq.3/4/5）
│   ├── cann_solution.py     # CANN 解决方案规格
│   ├── stanford.py          # Stanford 评测集入口
│   ├── stanford_loader.py   # Stanford 加载器实现
│   ├── stanford_matcher.py  # Stanford 匹配器实现
│   └── stanford_scoring.py  # Stanford 评分方案实现
│
├── checkers/                # Checker 层（通用，不绑定评测集）
│   ├── __init__.py
│   ├── relative_error_checker.py # RelativeErrorChecker + RelativeErrorOutputResult
│   ├── allclose_checker.py       # AllCloseChecker + AllCloseOutputResult
│
├── data/                    # 数据层
│   ├── __init__.py
│   ├── data_generator.py    # 输入数据生成（确定性种子 + shape/dtype/value_range）
│   └── package_manager.py   # 包管理（源码扫描、迭代编译、安装、接口扫描）
│
├── eval/                    # 评测层
│   ├── __init__.py
│   ├── evaluator.py         # 综合调度器（Config+BenchConfig 依赖注入，渐进设备恢复）
│   ├── accuracy_eval.py     # 精度评测（Checker 三输入模式：AI/Golden/同精度参考）
│   ├── perf_eval.py         # 性能评测（Profiler 运行 + Strategy 解析）
│   ├── op_runner.py         # 算子执行器（TorchOpGuard + 返回值检查 + 设备迁移）
│   ├── input_pool.py        # 输入池管理（clone 池 + 内存限制）
│   ├── results.py           # 评测结果数据类（failure_type + 增量输出）
│   ├── failure_synthesizer.py # 失败结果合成器（编译/安全/子进程/OOM）
│   ├── process_pool.py      # 多卡并行协调器（TaskUnit + ThreadPoolExecutor）
│   ├── subprocess_utils.py  # OOM 保护 + CANN 环境变量 + 部分结果恢复
│
├── security/                # 安全层
│   ├── __init__.py
│   ├── api_guard.py         # Timing API 防护（快照+验证+恢复）
│   ├── torch_op_guard.py    # Torch 算子守卫（TorchFunctionMode + 多路径规约）
│   ├── type_checker.py      # 返回值类型检查
│
├── report/                  # 报告层
│   ├── __init__.py
│   ├── report_generator.py  # 评测报告生成器（JSON+Markdown+Summary+HTML 多格式）
│   ├── html_generator.py    # HTML 报告渲染（KPI+柱状图+认证印章+4 Section）
│   ├── scoring.py           # 评分计算（hardware-anchored Eq.3/4/5）
│   ├── summary_generator.py # Summary 生成（几何平均加速比）
│   └── setup_info.py        # 评测配置采集（NPU/CANN/PyTorch/Python/OS 版本信息）
│
├── utils/                   # 工具层
│   ├── __init__.py
│   ├── device_manager.py    # 设备管理（渐进恢复：light→full→unrecoverable）
│   ├── dtype_mapper.py      # 数据类型映射
│   ├── param_builder.py     # 参数构建
│   ├── compare.py           # 张量对比引擎（MERE/MARE+小值域+相消处理）
│   ├── thresholds.py        # 精度阈值表（单一事实来源，Config 拷贝可覆盖）
│   ├── baseline_resolver.py # Baseline 解析（多硬件 dict + 平台别名）
│   ├── baseline_store.py    # BaselineStore（metadata/<hw>.json 集中存储 + 多平台 fallback）
│   ├── naming.py            # PascalCase→snake_case 转换（多候选模糊匹配）
│   ├── path_resolver.py     # 路径解析
│   └── tensor_utils.py      # 张量工具（to_cpu / to_fp64_cpu）
```

### 2.2 核心模块职责

#### 2.2.1 基类层（base/）

| 模块 | 职责 |
|------|------|
| `enums.py` | 通用枚举：DifficultyLevel(L1-L4)、BackendType(torch/torch_npu/ascendc/…)、SourceType(file/code/module/generated)、GoldenReference(file/self/fp64_cpu/none)、EvaluationMode(accuracy/performance/full) |
| `models.py` | 统一数据模型基类：AttrSpec、InputSpec、OutputSpec、TaskSpec、CaseSpec、SolutionSpec；支持 to_dict/from_dict 序列化 |
| `loaders.py` | 加载器抽象基类：TaskLoader（list_tasks/get_task/get_statistics）、CaseLoader（scan_all/scan_by_task）、OperatorDirMixin（proto.yaml+cases.yaml+golden.py 三文件检测）、GoldenLoaderBase（get_golden_function/get_input_function/get_output_function） |
| `matcher.py` | 算子匹配器抽象基类：OperatorMatcherBase（load_ai_operator/find_operator_info/clear_cache/find_operator_info_by_snake） |
| `checker.py` | 精度判断器抽象基类：CorrectnessChecker（get_name/check/辅助方法 _normalize_outputs/_ensure_cpu/_check_output_count） |
| `result.py` | 评测结果基类：OutputResult（ABC + 注册表反序列化）、AccuracyResult（passed/threshold/output_results/metadata）、PerfResult（elapsed_us/op_times/metadata + 便捷方法 get_speedup/get_baseline_us/get_t_hw_us）、compute_speedup |
| `scoring.py` | 评分方案抽象基类：ScoringScheme（prepare_baseline/calculate_case_score/aggregate_operator_scores）、CaseScoreInfo |
| `perf_strategy.py` | 性能指标策略：PerfMetricStrategy ABC、ProfFileLocations（文件定位）、KernelDetailsStrategy（默认，CSV唯一源）、TraceViewStrategy（PYPTO口径，待收编）、MsProfSummaryStrategy（基准采集专用） |

#### 2.2.2 注册层（registry/）

| 模块 | 职责 |
|------|------|
| `base.py` | BaseRegistry 泛型基类：register/get/list_all/is_registered/clear |
| `bench_registry.py` | BenchConfig（评测集配置聚合：task_loader/case_loader/golden_loader/operator_matcher/scoring_scheme/checker/case_spec_cls/golden_precision/perf_metric_strategy/precision_thresholds）+ BenchRegistry（注册/获取/列出）+ get_bench_config/get_bench_components 便捷函数 |
| `loader_registry.py` | TaskLoader/CaseLoader 注册表 + get_task_loader/get_case_loader |
| `golden_registry.py` | GoldenLoader 注册表 + get_golden_loader |
| `matcher_registry.py` | OperatorMatcher 注册表 + get_operator_matcher |
| `checker_registry.py` | CorrectnessChecker 注册表 + get_correctness_checker/register_correctness_checker |
| `scoring_registry.py` | ScoringScheme 注册表 + get_scoring_scheme |
| `case_spec_registry.py` | CaseSpec 子类注册表（供 BenchConfig.get_case_spec_cls 反序列化使用） |
| `perf_strategy_registry.py` | PerfMetricStrategy 注册表 + get_perf_metric_strategy |

#### 2.2.3 评测集插件层（benches/）

| 模块 | 职责 |
|------|------|
| `cann.py` | CANN 评测集入口：导出所有 CANN 组件 + 注册 BenchConfig("cann") |
| `cann_loader.py` | CANN 加载器：CannTaskLoader（proto.yaml 解析）、CannCaseLoader（cases.yaml 解析+校验）、GoldenLoader（golden.py 动态导入+PascalCase→snake_case） |
| `cann_spec.py` | CANN 特化数据模型：CannTaskSpec（继承 TaskSpec，增加 category/reference）、CannCaseSpec（继承 CaseSpec，增加 golden_reference/t_hw_us/baseline_perf_us dict）、CannInputSpec/CannOutputSpec/CannSolutionSpec |
| `cann_matcher.py` | CANN 算子匹配器：OperatorMatcher（继承 OperatorMatcherBase，优先 torch.ops.cann_bench 再 cann_bench 模块） |
| `cann_scoring.py` | CANN 评分方案：CannScoringScheme（Eq.3/4/5）、SimpleComparisonScheme（对比基线）、RecordingOnlyScheme（仅记录）、ScoringCalculator/OperatorScoreInfo |
| `cann_solution.py` | CANN 解决方案规格：CannSolutionSpec（继承 SolutionSpec） |
| `stanford.py` | Stanford 评测集入口：导出 Stanford 组件 + 注册 BenchConfig("stanford") |
| `stanford_loader.py` | Stanford 加载器：StanfordTaskLoader/StanfordCaseLoader/StanfordGoldenLoader |
| `stanford_matcher.py` | Stanford 匹配器：StanfordMatcher |
| `stanford_scoring.py` | Stanford 评分方案：StanfordScoringScheme |

#### 2.2.4 Checker 层（checkers/）

| 模块 | 职责 |
|------|------|
| `relative_error_checker.py` | RelativeErrorChecker：MERE/MARE 标准 + 小值域 + 相消处理；多输出支持；RelativeErrorOutputResult（逐输出 detail） |
| `allclose_checker.py` | AllCloseChecker：torch.allclose 简化对比；AllCloseOutputResult |

> **设计原则**：Checker 层不与任何评测集绑定，通过注册机制供 BenchConfig 选择。注册名 `relative_error`（主名）+ `cann_default`（兼容别名）。

#### 2.2.5 数据层（data/）

| 模块 | 职责 |
|------|------|
| `data_generator.py` | 输入数据生成：根据 shape/dtype/value_range 生成张量，支持确定性种子（hashlib SHA256 确保跨进程可复现） |
| `package_manager.py` | 包管理：源码目录扫描、迭代隔离编译（失败的算子自动隔离到 `_quarantine/`）、whl+run 安装、接口扫描（torch.ops.cann_bench + cann_bench 模块） |

#### 2.2.6 评测层（eval/）

| 模块 | 职责 |
|------|------|
| `evaluator.py` | 综合调度器：Config+BenchConfig 依赖注入；Golden 精度策略三档；渐进设备恢复；增量输出（OOM 恢复）；防作弊二次验证；APIGuard+TorchOpGuard 集成 |
| `accuracy_eval.py` | 精度评测：Checker 三输入模式（AI 输出 + Golden 输出 + 同精度参考输出）；Checker 注册机制选择 |
| `perf_eval.py` | 性能评测：Profiler 运行 + 文件定位（_locate_prof_files）；解析委托给 PerfMetricStrategy |
| `op_runner.py` | 算子执行器：TorchOpGuard 监听 + 返回值类型检查 + 设备迁移 |
| `input_pool.py` | 输入池管理：预分配 clone 池 + 内存限制（max_memory_mb） |
| `results.py` | 评测结果数据类：EvalCaseResult（failure_type 标注）、EvalOperatorResult（compile_passed/compilation_error）、EvalSessionResult；增量输出 JSON 格式 |
| `failure_synthesizer.py` | 失败结果合成器：编译失败/安全失败/子进程失败/OOM 失败四种类型 |
| `process_pool.py` | 多卡并行协调器：TaskUnit（算子×用例组×device_id）调度 + ThreadPoolExecutor + eval-child 子命令 |
| `subprocess_utils.py` | OOM Killer 保护（oom_score_adj 设置/检测）、CANN 环境变量继承列表、部分结果恢复 |

#### 2.2.7 安全层（security/）

| 模块 | 职责 |
|------|------|
| `api_guard.py` | Timing API 防护：快照关键 API（torch.npu.Event.elapsed_time/record、torch.npu.synchronize、torch_npu.profiler.profile/schedule）身份，安装后验证是否被篡改 |
| `torch_op_guard.py` | Torch 算子守卫：TorchFunctionMode 上下文管理器，检测 AI 算子直接调用 PyTorch 内置计算 API；多路径规约（aten/@/method/F.* → canonical name）；三种模式（off/warn/block）；pause() 机制排除 harness 自身预热调用 |
| `type_checker.py` | 返回值类型检查：`type(output) is torch.Tensor` 严格检查，拒绝 FakeTensor |

#### 2.2.8 报告层（report/）

| 模块 | 职责 |
|------|------|
| `report_generator.py` | 评测报告生成器：JSON+Markdown+Summary+HTML 四格式；EvalReport/OperatorReport 数据类；语义前缀自动命名 |
| `html_generator.py` | HTML 报告渲染：4 Section（Setup/Results/Details/认证印章）+ KPI 指标卡 + 等级分析表 + 柱状图 + 算子详情表 |
| `scoring.py` | 评分计算：hardware-anchored Eq.3（per_case_sol_score）+ Eq.4（aggregate_eq4）；fallback 基线 `max(t_hw*10, 10)` |
| `summary_generator.py` | Summary 生成：几何平均加速比计算 |
| `setup_info.py` | 评测配置采集：metadata（framework/date/agent_skill/base_model/benchmark）+ environment（NPU/CPU/CANN/Driver/PyTorch/Python/OS/Docker） |

#### 2.2.9 工具层（utils/）

| 模块 | 职责 |
|------|------|
| `device_manager.py` | 设备管理：CPU/NPU 切换；渐进恢复机制（recover_light=empty_cache → recover_full=aclrtResetDevice → unrecoverable） |
| `compare.py` | 张量对比引擎：compare_tensors 主入口；MERE/MARE 计算；小值域兜底判定（ErrorCount 比值）；相消处理（CPU 同精度对照）；整数精确匹配；多输出支持 |
| `thresholds.py` | 精度阈值表：单一事实来源（PRECISION_THRESHOLDS dict）；Config 拷贝后可通过 op_info.precision_thresholds 覆盖单 dtype |
| `baseline_resolver.py` | Baseline 解析：多硬件 dict 格式（{910b2: 40.2, 910b1: 45.1}）；平台别名映射（Ascend910B2→910b2、Ascend910_93 系列→910c） |
| `baseline_store.py` | BaselineStore：从 metadata/<hardware>.json 集中加载；逐级向上查找 metadata/ 目录；多平台 fallback（910b1→910b2）；三级嵌套 JSON 格式（level→op→case_id） |
| `naming.py` | PascalCase→snake_case：camel_to_snake + snake_case_candidates（多候选模糊匹配，处理 3D/ROI/NMS 等特殊情况） |
| `param_builder.py` | 参数构建：合并后的统一方法，根据 golden 签名 + case attrs 构建调用参数 |
| `tensor_utils.py` | 张量工具：tensors_to_cpu（保持精度）、tensors_to_fp64_cpu（升精度到 float64） |

---

## 3. 源码目录评测流程

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           源码目录评测流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: --source-dir /path/to/ai_generated_ops                               │
│                                                                             │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │ 扫描源码   │────▶│ 检查dist   │────▶│ 编译whl    │────▶│ 安装包     │   │
│  │ 目录结构   │     │ whl+run包  │     │ (build.sh) │     │ (whl+run)  │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│        │                  │                  │                  │          │
│        │           有包则跳过编译        无dist则编译           │          │
│        │                  │                  │                  │          │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │ APIGuard   │────▶│ 扫描模块   │────▶│ 匹配用例   │────▶│ 执行评测   │   │
│  │ 验证       │     │ 接口列表   │     │ (BenchConfig│     │            │   │
│  │            │     │            │     │  Registry) │     │            │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│                                                                             │
│  输出: 评测报告 (JSON + Markdown + HTML)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 详细步骤

**Step 1: 扫描源码目录结构**

检查源码目录是否存在以下结构：
```
source_dir/
├── build.sh          # 编译脚本（可选）
├── dist/             # 编译产物目录（可选）
│   ├── cann_bench_xxx.whl   # Python包
│   └── cann_bench_xxx.run   # NPU内核包（可选）
├── cann_bench/       # Python包目录
│   └── __init__.py
├── csrc/             # C++源码
│   └── ops/
│       └── exp/
│           └── ascend910b/
│               └── exp.cpp
├── setup.py          # 构建配置
└── CMakeLists.txt
```

**Step 2: 检查dist目录**

扫描 `dist/` 目录，查找：
- `cann_bench_xxx.whl` - Python包（必须）
- `cann_bench_xxx.run` - NPU内核包（可选）

如果存在这些包，跳过编译步骤；否则检查是否有 `build.sh` 并执行编译。

**Step 3: 迭代隔离编译（如果需要）**

执行 `build.sh` 脔本，如果编译失败，系统自动识别并隔离编译不过的算子到 `_quarantine/` 目录，然后对剩余的算子重新执行编译和评测。这确保部分算子编译失败不会导致整个评测任务失败。

**Step 4: 安装包**

安装顺序：先安装run包，再安装whl包。

**Step 5: APIGuard 验证**

安装后使用 APIGuard 验证关键 Timing API 是否被篡改。若检测到篡改，恢复原始 API 并为所有算子合成安全失败结果。

**Step 6: 扫描模块接口**

导入cann_bench模块，扫描提供的算子接口：
```python
# 优先从 torch.ops.cann_bench 加载，再尝试 cann_bench 模块
import torch
if hasattr(torch.ops, 'cann_bench'):
    for name in dir(torch.ops.cann_bench):
        if not name.startswith('_'):
            interfaces.append(name)
```

**Step 7: 匹配用例（通过 Registry）**

根据 `--bench-name` 参数获取 BenchConfig，进而获取 TaskLoader/CaseLoader/GoldenLoader 等组件实例，加载对应的用例数据。

**Step 8: 执行评测**

对每个匹配到的算子执行评测：
1. APIGuard 验证 + TorchOpGuard 配置
2. 加载用例数据（通过 CaseLoader）
3. 生成确定性种子（SHA256 hash + Config.eval_seed）
4. 生成输入数据（确定性种子确保可复现）
5. 执行Golden函数（精度策略由 BenchConfig.golden_precision 控制）
6. 执行AI算子（TorchOpGuard 监听 + profiler 采集性能）
7. 精度对比（Checker 三输入：AI/Golden/同精度参考）
8. 防作弊二次验证（可选，Config.enable_accuracy_retry=True）
9. 性能数据提取（PerfMetricStrategy 解析 profiler 产出）
10. 渐进设备恢复（连续失败后检测 NPU 健康状态）
11. 增量输出（每用例完成后写入，OOM 恢复）

### 3.3 命令行参数

底层入口为 `python -m kernel_eval.cli <subcommand>`；`scripts/run_evaluation.sh` 是它的 shell 包装。

#### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 子命令 | `eval` / `list` / `info` / `config` / `eval-child`（CPU 模拟走 `eval --device cpu`，无独立 `simulate` 子命令） | 无 |
| `-v, --verbose` | 详细输出 | False |
| `--bench-name <name>` | 评测集名称 | `cann` |

#### 评测（`eval`）参数

| 参数 | 说明 | 默认值 | 通过 shell 暴露 |
|------|------|--------|------|
| `--source-dir <dir>` | AI 生成算子源码目录 | None | ✓ |
| `--task-dir <path>` | 评测目录 | None | ✓ |
| `--operator <name>` | 算子名称筛选 | None | ✓ |
| `--case-id <id>` | 用例编号筛选 | None | ✓ |
| `--device <cpu/npu>` | 设备类型 | `npu` | ✓ |
| `--device-id <id>` | NPU 设备 ID（单卡） | None | ✓ |
| `--processes-per-card <n>` | 多卡并行每卡进程数 | 2 | ✓ |
| `--timeout-per-operator <n>` | 多卡并行单算子超时（秒） | 300 | ✓ |
| `--warmup <n>` | 预热次数 | 3 | ✓ |
| `--repeat <n>` | 采集次数 | 5 | ✓ |
| `--reports-dir <dir>` | 报告输出目录 | `reports` | ✓ |
| `--output <dir>` | 报告输出目录（覆盖 --reports-dir） | None | ✗ |
| `--eval-code <code>` | 评测代号 | None | ✗ |
| `--bench-name <name>` | 评测集名称 | `cann` | ✗ |
| `--op-timeout-sec <n>` | 单进程隔离下 per-op 超时 | 240 | ✗ |
| `--no-iterative-compile` | 关闭迭代隔离编译 | False | ✗ |
| `--no-perf` | 关闭性能采集 | False | ✓ |
| `--profiler-level <level>` | Profiler 级别 | `Level1` | ✓ |
| `--no-freq-boost` | 关闭升频预热（部分卡防挂死） | False | ✗ |
| `--eval-seed <n>` | 确定性种子（0=自动hash, -1=纯随机, N=偏移） | 0 | ✗ |
| `--input-dist <dist>` | 输入数据分布（uniform / normal，别名 u/norm/gaussian/n） | `uniform` | ✗ |
| `--torch-op-guard-mode <mode>` | Torch 算子守卫模式（off/warn/block） | `block` | ✗ |
| `--perf-metric-strategy <name>` | 性能策略覆盖 | None | ✗ |

**多卡并行判定**（`src/kernel_eval/cli.py`）：
```python
use_multi_card = (device == 'npu' and device_id is None and not source_dir)
```
即只要在 NPU 下不指定 `--device-id` 且不带 `--source-dir`，就走 `ProcessPoolCoordinator` 多卡并行模式。

**eval-child 内部子命令**：仅供 ProcessPoolCoordinator 调用，接收 `--cases-file`（JSON 用例列表）和 `--output`（结果文件路径），在子进程中独立执行评测。

#### CPU 仿真

无独立 `simulate` 子命令；无 NPU 环境下的功能验证使用 `eval --device cpu`。

#### 列表（`list`）参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bench-name <name>` | 评测集名称 | `cann` |
| `--operator <name>` | 算子名称筛选 | None |
| `--cases` | 列出用例而非算子 | False |

#### 详情（`info`）参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bench-name <name>` | 评测集名称 | `cann` |
| `--operator <name>` | 算子名称（必填） | — |

#### 配置（`config`）参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--show` | 显示当前配置 | False |

### 3.4 使用示例

```bash
# 从源码目录评测（自动编译安装）
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops

# 使用 Stanford 评测集
python -m kernel_eval.cli eval --bench-name stanford

# 设置确定性种子（确保可复现）
python -m kernel_eval.cli eval --operator Exp --eval-seed 42

# 用正态分布输入评测（默认 uniform，与历史行为逐位一致）
python -m kernel_eval.cli eval --input-dist norm

# 分布是全局开关，评测范围仍由 --operator / --task-dir / --case-id 控制
python -m kernel_eval.cli eval --operator Softmax --case-id 3 --input-dist normal

# Torch 算子守卫 block 模式（默认：检测到 torch.matmul 等直接抛错）
python -m kernel_eval.cli eval --operator Exp --torch-op-guard-mode block

# Torch 算子守卫 warn 模式（仅记日志，便于排查）
python -m kernel_eval.cli eval --operator Exp --torch-op-guard-mode warn

# 使用 Level2 Profiler
./scripts/run_evaluation.sh --action eval --operator Exp --profiler-level Level2

# CPU 仿真评测（无独立 simulate 子命令，用 eval --device cpu）
python -m kernel_eval.cli eval --device cpu --operator Exp

# 使用 MsProfSummary 性能策略（基准采集）
python -m kernel_eval.cli eval --operator Exp --perf-metric-strategy msprof_summary

# 列出所有算子
./scripts/run_evaluation.sh --action list

# 查看算子详情
./scripts/run_evaluation.sh --action info --operator Exp

# 显示当前配置
./scripts/run_evaluation.sh --action config
```

### 3.5 异常处理

| 场景 | 处理方式 |
|------|----------|
| 源码目录不存在 | 报错退出 |
| 无build.sh且无dist | 报错退出 |
| build.sh执行失败 | 迭代隔离编译，失败的算子合成 compile failure 结果 |
| run包安装失败 | 报错退出（NPU模式必须） |
| whl包安装失败 | 报错退出 |
| cann_bench导入失败 | 报错退出 |
| 未匹配到算子 | 警告并退出 |
| Timing API被篡改 | 恢复原始API后为所有算子合成 security failure 结果 |
| AI算子调用torch内置API | TorchOpGuard 检测：warn 模式记日志，block 模式抛 RuntimeError |
| NPU设备异常 | 渐进恢复（light→full→unrecoverable），恢复无效时剩余用例标记 cascade_device |
| 子进程被 OOM Kill | 恢复增量写入的部分结果，剩余用例合成 oom_killed 失败 |
| 子进程超时 | SIGTERM → 10s 宽限 → SIGKILL；恢复部分结果或合成 timeout 失败 |

---

## 4. 核心能力设计

### 4.1 安全防护设计

#### 4.1.1 Timing API防护

**原理**：在submission代码运行前，快照关键Timing API的身份；安装wheel后验证是否被篡改。

**集成位置**：
- `APIGuard.snapshot()` 在 `PackageManager.prepare_from_source()` 内部调用（安装前）
- `APIGuard.verify()` 在 `Evaluator.evaluate_from_source()` 调用 prepare_from_source 后验证（安装后）

**关键API列表**：
- `torch.npu.Event.elapsed_time`
- `torch.npu.Event.record`
- `torch.npu.synchronize`
- `torch_npu.profiler.profile`
- `torch_npu.profiler.schedule`

#### 4.1.2 TorchOpGuard（PyTorch 内置 API 调用检测）

**原理**：使用 `torch.overrides.TorchFunctionMode`（PyTorch ≥1.11）拦截 `with` 块内的所有 torch 函数调用，判断是否为禁止的计算密集型 API。

**禁止 API 集合**（BUILTIN_COMPUTE_OPS）：
```python
BUILTIN_COMPUTE_OPS = {
    "torch.matmul", "torch.mm", "torch.bmm", "torch.einsum",
    "torch.nn.functional.linear",
    "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d",
    "torch.nn.functional.softmax", "torch.nn.functional.log_softmax",
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.nn.functional.silu", "torch.nn.functional.gelu", "torch.nn.functional.relu",
    "torch.nn.functional.layer_norm", "torch.nn.functional.rms_norm", "torch.nn.functional.batch_norm",
    "torch.Tensor.matmul", "torch.Tensor.mm", "torch.Tensor.bmm",
    "torch.Tensor.__matmul__", "torch.Tensor.__rmatmul__",
    "torch.Tensor.softmax", "torch.Tensor.log_softmax", "torch.Tensor.layer_norm",
}
```

**多路径规约**：同一数学运算在 PyTorch 内部有多种调度路径（torch.matmul / torch.ops.aten.matmul / a @ b / a.matmul(b) / F.linear），`_LEAF_TO_CANONICAL` 映射将所有路径规约到同一个 canonical name，防止单条规则被多路径绕过。

**三种模式**：
- `off`：不监听（纯调试）
- `warn`：检测到禁止 API 时打印 [WARN]，不阻断执行
- `block`：检测到禁止 API 时抛 RuntimeError（默认，生产评测使用）

**pause() 机制**：harness 自身的升频清 cache 预热（torch.matmul / torch.max）不是 candidate 的真实计算，通过 `TorchOpGuard.pause()` 上下文管理器临时排除，避免 false positive。

**集成位置**：`OpRunner.run_ai_op()` 中，在执行 AI 算子前后启用/关闭 TorchOpGuard。

#### 4.1.3 返回值类型检查

**原理**：使用 `type(output) is torch.Tensor` 严格检查，拒绝FakeTensor等子类伪装。

#### 4.1.4 二次验证机制

**原理**：用新鲜输入重跑一次，防止缓存作弊。

**实现**：`Evaluator._retry_with_fresh_inputs()`，仅在 `Config.enable_accuracy_retry=True` 且第一轮已通过时触发。二次验证使用偏移种子（case_seed + 1），并对所有浮点张量添加 0.01 微扰，确保输入确实不同。

#### 4.1.5 输入池防缓存

**原理**：预分配一组clone输入，轮换使用，每次调用data_ptr不同，防止按地址缓存输出。支持 max_memory_mb 内存限制。

#### 4.1.6 确定性种子

**原理**：使用 hashlib SHA256 计算 case_id hash（而非 Python hash()，因 PYTHONHASHSEED 随机化），确保跨进程可复现。

```python
if config.eval_seed is not None:
    digest = hashlib.sha256(case_id_str.encode('utf-8')).digest()
    deterministic_hash = int.from_bytes(digest[:8], byteorder='big') % (2**31)
    case_seed = (config.eval_seed + deterministic_hash) % (2**31)
else:
    case_seed = None  # 纯随机模式（不推荐）
```

- `eval_seed=0`（默认）：基于 case_id hash，确保可复现
- `eval_seed=N`（正整数）：用 `(N + case_id_hash) % 2^31`，不同 N 给出不同但可复现的输入
- `eval_seed=None`：纯随机模式（不推荐，会导致 flaky 测试）

### 4.2 精度验证设计

#### 4.2.1 Golden 精度策略（BenchConfig.golden_precision）

| 策略 | 说明 | 同精度参考来源 |
|------|------|------|
| `fp64_cpu`（默认） | 升精度到 float64 + CPU 计算，避免 NPU 溢出污染 | 需单独执行 native_cpu Golden |
| `native_cpu` | 保持原始精度在 CPU 上计算 | 直接复用 Golden 输出 |
| `native_npu` | 保持原始精度在 NPU 上计算 | 直接复用 Golden 输出 |

**精度对比 Checker 三输入模式**：

```python
accuracy_result = checker.check(
    ai_outputs=ai_result.outputs,       # AI 算子输出
    golden_outputs=golden_result.outputs, # Golden 参考输出（可能是 fp64）
    dtype=dtype,
    threshold=threshold,
    native_outputs=native_out,          # 同精度参考输出（用于小值域判断）
    ignore_indices=ignore_output_indices,
    custom_thresholds=merged_thresholds,
)
```

- `fp64_cpu` 时同精度参考需单独执行一次 native_cpu Golden（精度不同）
- `native_cpu/native_npu` 时同精度参考直接复用 Golden（精度相同，避免重复计算）

#### 4.2.2 精度标准实现

**Checker 注册机制**：
- `relative_error`（注册主名）+ `cann_default`（兼容别名）：MERE/MARE + 小值域 + 相消处理
- `allclose`：torch.allclose 简化对比

**compare_tensors 核心逻辑**：
```python
def verify_accuracy(actual, golden, dtype):
    # 整数类型: 精确匹配
    # 浮点类型:
    mere = compute_mere(actual, golden)
    mare = compute_mare(actual, golden)
    threshold = get_threshold(dtype)

    if mere < threshold and mare < 10 * threshold:
        return True

    # 小值域检查（当 golden 接近 0 时）
    # 相消处理（当 output ≈ 0 且 golden 在精度边界附近时）
    return check_small_value_region(actual, golden, native_out, dtype) \
        or check_cancel_boundary(actual, golden, native_out, dtype)
```

**OutputResult 注册表反序列化**：
```python
# 序列化时 OutputResult 子类自带 checker_name
# 反序列化时通过注册表按 checker_name 查找子类
register_output_result('relative_error', RelativeErrorOutputResult)
register_output_result('cann_default', RelativeErrorOutputResult)  # 兼容旧名
```

### 4.3 性能评测设计

#### 4.3.1 PerfMetricStrategy 策略模式

**设计原则**：perf_eval 只负责 profiler 运行和文件定位（`_locate_prof_files`），文件解析逻辑完全交给 strategy，不同策略自主选择读哪些文件、怎么解析。

**三种策略**：

| 策略 | elapsed_us 数据源 | 适用场景 | 注册名 |
|------|------|------|------|
| `KernelDetailsStrategy` | Σ kernel Duration 中位数（kernel_details.csv 唯一源） | 正式评测（默认） | `kernel_details` |
| `TraceViewStrategy` | trace_view.json aicore_e2e（PYPTO 口径） | tilefwk/PYPTO 算子（**待收编**） | `trace_view` |
| `MsProfSummaryStrategy` | 优先 kernel_details.csv，fallback msprof op_summary | 基准性能数据采集 | `msprof_summary` |

**KernelDetailsStrategy（默认）**：
- CSV 不可用 → 明确报错，不 fallback 到 trace_view
- trace_view.json 的用途：补充 tilefwk/PYPTO 指标（aicore_e2e/aicpukernel_gap/aicore_e2e_jitter），写入 metadata；sanity check 对比 trace_view kernel total vs CSV total
- 为什么只用 CSV：trace_view 只识别 `aclnn*AiCore_` 命名，会**静默漏掉**自定义 AscendC kernel（`*_custom`）

**MsProfSummaryStrategy（基准采集专用）**：
- kernel_details.csv 不可用时 fallback 到 msprof op_summary（更完整：包含所有 kernel，不受 Level1 过滤限制）
- 正式评测用 KernelDetailsStrategy，基准采集用 MsProfSummaryStrategy

**策略选择方式**：
- 默认：BenchConfig.perf_metric_strategy（通常为 `kernel_details`）
- 覆盖：Config.perf_metric_strategy_override（CLI `--perf-metric-strategy`）

#### 4.3.2 ProfFileLocations（文件定位结果）

```python
@dataclass
class ProfFileLocations:
    ascend_output_dir: Optional[str] = None  # ASCEND_PROFILER_OUTPUT 目录
    csv_path: Optional[str] = None           # kernel_details.csv 完整路径
    trace_view_path: Optional[str] = None    # trace_view.json 完整路径
    prof_dir: str = ""                        # profiler 输出根目录
    msprof_summary_paths: List[str] = field(default_factory=list)  # msprof op_summary CSV 路径
```

#### 4.3.3 Warmup kernel 过滤

使用 Input Shapes 精确匹配 MatMulV3/ReduceMax 升频清 cache kernel：

```python
WARMUP_MATMUL_SHAPE = '"10240,10240;10240,10240"'
WARMUP_REDUCE_SHAPE = '"96,1024,1024;3"'
```

CSV 是唯一包含 Input Shapes 的数据源，warmup 过滤先从 CSV 提取 warmup kernel 名称集合，再用于 trace_view kernel 过滤。

#### 4.3.4 NPU升频与L2清空

预分配升频清 cache 的 tensors 并固定到目标设备，避免设备不匹配。通过 `TorchOpGuard.pause()` 临时排除守卫。

#### 4.3.5 BaselineStore 集中存储

**文件位置**：`tasks/metadata/<hardware>.json`

**JSON 格式**（三级嵌套）：
```json
{
  "_metadata": { "version": "0.2.0", "hardware": "910b2" },
  "level1": {
    "exp": {
      "1": { "baseline_perf_us": 13.78, "t_hw_us": 1.09 },
      "2": { "baseline_perf_us": 29.06, "t_hw_us": 8.74 }
    }
  }
}
```

**多硬件支持**：值可以是 float（单硬件）或 `{hardware: float}` dict（多硬件）。

**查找策略**：从 bench_root 开始逐级向上查找 metadata/ 目录，确保子目录也能正确定位。

**多平台 fallback**：非默认平台文件缺失时，自动 fallback 到默认平台（910b2）数据，每个硬件只 warn 一次。

**BaselineResolver 平台别名**：
```python
PLATFORM_ALIAS = {
    "Ascend910B2": "910b2",  # 产品型号 → 逻辑名
    "Ascend910_93": "910c",  # Atlas A3 系列前缀
    "Ascend910B1": "910b1",
}
```

### 4.4 进程调度设计

#### 4.4.1 ProcessPoolCoordinator（多卡并行）

**核心设计**：
- 任务单元 = TaskUnit（算子, 用例组, device_id），统一调度粒度
- 子进程通过 `eval-child` 独立子命令执行（纯执行者，不做调度/编译/fork）
- 主进程按算子维度聚合 case 结果 → EvalOperatorResult

**配置**：
```python
processes_per_card = 1      # 每卡并发进程数（profiler 开启时强制为 1）
card_count = 8              # 8 张 NPU 卡
timeout_per_operator = 300  # 单算子超时（秒）
```

**调度流程**：
1. 检测 NPU 卡数（torch.npu.device_count）
2. 按 TaskUnit 拆分任务，均分到各卡
3. ThreadPoolExecutor 并行执行，每个 TaskUnit 启动一个 eval-child 子进程
4. 子进程结果通过 JSON 文件传递
5. 主进程按算子维度聚合 case 结果

**OOM Killer 保护**：
- 子进程自设 `oom_score_adj=1000`（最优先被 OOM Killer 杀死，保护主进程）
- 父进程双保险写入 `oom_score_adj`
- OOM Kill 时恢复增量写入的部分结果，剩余用例合成 oom_killed 失败

#### 4.4.2 渐进设备恢复

**Evaluator.run_cases() 三阶段恢复机制**：

| 阶段 | 说明 | 方法 |
|------|------|------|
| `healthy` | 正常评测 | — |
| `recovering` | 连续 ≥3 失败且设备不健康 → 尝试恢复 | `recover_light`（empty_cache）→ `recover_full`（aclrtResetDevice） |
| `unrecoverable` | 恢复无效或恢复后仍失败 | 剩余用例标记 cascade_device 跳过 |

**注意事项**：profiler 活跃时不尝试 recover_full（aclrtResetDevice 会破坏 profiler 的 ACL profiling 资源）。

#### 4.4.3 增量输出（OOM 恢复）

```python
def _write_incremental_output(self, operator, rel_path, results, total_cases):
    payload = {"case_results": [r.to_dict() for r in results]}
    Path(self.incremental_output_path).write_text(json.dumps(payload))
```

子进程每完成一个用例就刷新写入当前结果，OOM Kill 时主进程可从 output 文件恢复已完成的部分结果。

#### 4.4.4 失败结果合成

**FailureSynthesizer 四种类型**：

| 方法 | error_prefix | error_field | failure_type |
|------|------|------|------|
| `synthesize_compile_failure` | `compile failed:` | `compilation_error` | `cascade_device` |
| `synthesize_security_failure` | `security check failed:` | `subprocess_failure_reason` | `cascade_device` |
| `synthesize_subprocess_failure` | `subprocess failed:` | `subprocess_failure_reason` | `cascade_device` |
| `synthesize_oom_failure` | — | `subprocess_failure_reason` | `oom_killed` |

**设计原则**：失败算子仍然出现在 session 结果里，报告可见原因，而非完全失踪。

### 4.5 报告生成设计

#### 4.5.1 多格式输出

| 格式 | 内容 | 文件命名 |
|------|------|------|
| JSON | 完整结构化数据（可反序列化） | `<prefix>_<timestamp>.json` |
| Markdown | 表格摘要 + 详细用例结果 | `<prefix>_<timestamp>.md` |
| Summary | 简要摘要 | `<prefix>_<timestamp>.md` |
| HTML | 完整可视化报告（4 Section + 认证印章） | `<prefix>_<timestamp>.html` |

**语义前缀**：默认命名自动加入语义前缀（如 `cann_eval_`、`stanford_eval_`、`cpu_sim_cann_`）。

#### 4.5.2 HTML 报告结构

```
Section 1: Abstract（从 description.html 读取）
Section 2: Experiment Setup（setup_info 采集的 metadata + environment）
Section 3: Results Analysis
  3.1 KPI 指标卡（通过率/算子数/用例数/失败数/级联失败/总得分）
  3.2 等级分析表（按 Level 汇总）
  3.3 柱状图（得分/通过率/加速比）
  3.4 Top 算子分析表（按通过率/加速比排序）
Section 4: Operator Details（按 Level 分组的算子详情表）
Certification Seal（CANN-Bench 认证印章）
```

**动态字段替换**：HTML 报告中的通过率/得分/算子数/用例数/Agent/Skill/BaseModel 等字段从评测结果动态替换。

#### 4.5.3 setup_info 采集

```python
setup = collect_setup_info(config)
# setup == {
#     "metadata": {
#         "framework": "CANN-Bench V0.4.0",
#         "tasks_version": "0.4.0",
#         "date": "2026-06-16 14:30:00",
#         "agent_skill": "DeepSeek V4 Pro",
#         "base_model": "...",
#         "benchmark": "CANN-Bench tasks",
#         "license": "CANN Open Software License v2.0",
#     },
#     "environment": {
#         "npu": "Ascend910B2 × 8",
#         "cpu": "aarch64",
#         "cann": "9.0.0",
#         "driver": "cann-9.0.0",
#         "pytorch": "2.1.0",
#         "pytorch_npu": "2.1.0.post3",
#         "python": "3.10.12",
#         "os": "Linux-5.10.0...",
#         "docker": "cann-bench-eval:1.0.0-ascend910b-aarch64-opsnone / CANN 9.1.0",
#     },
# }
```

#### 4.5.4 几何平均加速比

```python
def geometric_mean_speedup(speedups):
    if not speedups:
        return 0.0
    return math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))
```

保留为诊断字段，正式评分使用 hardware-anchored Eq.3/4/5。

---

## 5. 实施步骤

### Phase 1：基类层分离

1. 创建 `base/` 目录，统一存放抽象基类
2. 从 benches 中提取 TaskLoader/CaseLoader/CorrectnessChecker/ScoringScheme 等基类
3. 新增 `enums.py`（DifficultyLevel/BackendType/SourceType/GoldenReference/EvaluationMode）
4. 新增 `perf_strategy.py`（PerfMetricStrategy ABC + KernelDetailsStrategy + MsProfSummaryStrategy）
5. 更新所有导入路径

### Phase 2：注册层完善

1. 创建 `registry/` 目录，统一管理所有组件的注册机制
2. 实现 BaseRegistry 泛型基类
3. 实现各子注册表（loader/golden/matcher/checker/scoring/case_spec/bench/perf_strategy）
4. 实现 BenchConfig 聚合配置 + get_bench_components 便捷函数

### Phase 3：评测集插件化

1. benches/cann 子目录扁平化（删除子目录，文件平铺到 benches/）
2. 新增 `benches/stanford.py` + `stanford_loader.py` + `stanford_matcher.py` + `stanford_scoring.py`
3. Checker 重命名：cann_default → relative_error（保留兼容别名）
4. Checker 层独立为 `checkers/` 目录

### Phase 4：安全层增强

1. 实现 `torch_op_guard.py`（TorchFunctionMode + 多路径规约 + pause 机制）
2. 确定性种子机制（hashlib SHA256 + Config.eval_seed）
3. 二次验证增强（偏移种子 + 全张量微扰）

### Phase 5：性能评测增强

1. PerfMetricStrategy 策略模式（KernelDetailsStrategy/TraceViewStrategy/MsProfSummaryStrategy）
2. BaselineStore 集中存储（metadata/<hardware>.json + 多平台 fallback）
3. ProfFileLocations 文件定位数据类

### Phase 6：进程调度增强

1. ProcessPoolCoordinator（TaskUnit 调度 + eval-child 子命令）
2. FailureSynthesizer（统一失败结果合成）
3. 增量输出 + OOM 恢复机制
4. 渐进设备恢复（light → full → unrecoverable）

### Phase 7：报告层完善

1. HTML 报告生成器（4 Section + KPI + 柱状图 + 认证印章）
2. setup_info 采集模块（NPU/CANN/PyTorch 版本信息）
3. 语义前缀自动命名
4. 动态字段替换

### Phase 8：版本管理

1. _version.py 单点版本真相源（从 VERSION 文件动态读取）
2. FRAMEWORK_VERSION + TASKS_VERSION 双版本管理

---

## 6. 验证方案

### 6.1 安全验证

| 测试项 | 验证方法 |
|--------|----------|
| API篡改检测 | 模拟monkey-patch，验证检测并恢复 |
| TorchOpGuard 检测 | AI算子调用torch.matmul，验证warn/block模式 |
| TorchOpGuard 多路径规约 | 用 a@b / torch.ops.aten.mm / F.linear 等路径调用，验证统一归到 canonical name |
| TorchOpGuard pause | harness 预热调用 torch.matmul，验证不触发 |
| 返回值检查 | 返回FakeTensor，验证拒绝 |
| 二次验证 | 缓存作弊实现，验证二次失败 |
| 输入池轮换 | 验证每次调用data_ptr不同 |
| 确定性种子 | 同一 eval_seed 跨进程重复评测，验证结果一致 |

### 6.2 精度验证

| 测试项 | 验证方法 |
|--------|----------|
| Golden 精度策略 fp64_cpu | 对比fp64和fp32结果差异 |
| Golden 精度策略 native_cpu | 验证同精度参考复用 Golden |
| Golden 精度策略 native_npu | 验证 NPU 上计算 + 同精度参考复用 |
| Checker 三输入模式 | 验证 native_output 在小值域/相消中的作用 |
| MERE/MARE计算 | 手动计算验证 |
| NaN/Inf处理 | 特殊值用例验证 |
| 整数精确匹配 | 整型用例验证 |
| OutputResult 注册表反序列化 | 验证 JSON 报告能正确恢复 Checker 子类 |

### 6.3 性能验证

| 测试项 | 验证方法 |
|--------|----------|
| KernelDetailsStrategy | 验证 CSV 为唯一 elapsed_us 数据源 |
| TraceViewStrategy | 验证 PYPTO 口径，非 tilefwk 算子明确报错 |
| MsProfSummaryStrategy | 验证 CSV fallback 到 msprof op_summary |
| 自定义 kernel 覆盖 | 验证 CSV 包含 `*_custom` kernel，trace_view 不漏掉 |
| BaselineStore 多平台 | 验证 910b1 fallback 到 910b2 数据 |
| BaselineStore 向上查找 | 验证子目录也能找到 metadata/ |
| 升频清cache效果 | 对比有无预热的时间稳定性 |
| 确定性种子数据生成 | 同种子生成相同输入 |

### 6.4 进程调度验证

| 测试项 | 验证方法 |
|--------|----------|
| ProcessPoolCoordinator 多卡 | 8卡并行评测，验证结果聚合 |
| OOM Kill 保护 | 模拟 OOM，验证部分结果恢复 |
| 渐进设备恢复 | 模拟连续失败，验证 light→full→unrecoverable |
| FailureSynthesizer | 验证四种失败类型正确合成 |
| eval-child 子命令 | 验证 JSON 输入/输出正确传递 |

### 6.5 报告验证

| 测试项 | 验证方法 |
|--------|----------|
| HTML 报告完整性 | 验证 4 Section + 认证印章完整 |
| setup_info 采集 | 验证 NPU/CANN/PyTorch 版本信息正确 |
| 动态字段替换 | 验证通过率/得分/Agent/Skill 替换 |
| 语义前缀命名 | 验证不同 bench_name 有不同前缀 |

### 6.6 Registry 验证

| 测试项 | 验证方法 |
|--------|----------|
| BenchConfig 组件获取 | 验证 get_bench_components 返回正确实例 |
| Checker 注册切换 | 验证 checker 注册与切换判断器 |
| Bench 名称切换 | 验证 --bench-name 参数切换评测集 |
| 性能策略切换 | 验证 --perf-metric-strategy 参数切换策略 |

### 6.7 集成验证

- 运行完整评测流程（source-dir + multi-card）
- 验证 JSON/Markdown/HTML/Summary 四格式报告
- 验证 hardware-anchored 评分计算
- 验证安全机制在整个流程中有效
- 验证 OOM Kill 时部分结果恢复

---

## 7. 附录

### 7.1 参考文档

- `docs/spec/benchmark_spec.md`：算子代码生成评测基准规范
- `docs/design/evaluator_design.md`：评测工程设计文档（本文档）
- `docs/guide/quick_start.md`：快速入门指南
- `docs/changelog.md`：版本变更记录
- `../opbase/docs/zh/ops_precision_standard/experimental_standard.md`：精度标准

### 7.2 包命名约定

| 包类型 | 命名格式 | 说明 |
|---------|---------|------|
| whl包 | `cann_bench_xxx.whl` | Python包，包含算子接口 |
| run包 | `cann_bench_xxx.run` | NPU内核二进制包 |

### 7.3 评分公式 (bench.tex §3.3 / Eq. 3, 4, 5)

```
权重: w_c = 0.2, w_f = 0.3, w_p = 0.5  (sum = 1, 单算子满分 100)

单用例 hardware-anchored 性能得分 (Eq. 3):
  score_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))

  T_HW    = metadata/<hw>.json 中 t_hw_us (硬件下界)
  T_baseline = metadata/<hw>.json 中 baseline_perf_us
  T_cand  = 候选 kernel 实测时间

  锚点:
    T_cand = T_baseline ⇒ score = 0.5
    T_cand = T_HW       ⇒ score = 1.0
    T_cand → ∞          ⇒ score → 0

  Fallback: baseline 缺失时 proxy_baseline = max(t_hw*10, 10)

单算子综合评分 (Eq. 4):
  EachOperatorScore = [ w_c · δ_pass + Σ_i δ_acc,i · (w_f + w_p · score_i) / N ] · 100

  δ_pass ∈ {0, 1}      整份提交编译是否通过
  δ_acc,i ∈ {0, 1}     用例 i 是否通过精度门
  N = len(cases)

聚合 (Eq. 5):
  Level-N 得分   = Σ 该 level 内 EachOperatorScore
  benchmark 总分 = Σ 所有算子 EachOperatorScore
```

实现位于 `src/kernel_eval/report/scoring.py`。

### 7.4 BenchConfig 默认配置（CANN 评测集）

```python
BenchConfig(
    name="cann",
    task_loader="cann",
    case_loader="cann",
    golden_loader="cann",
    operator_matcher="cann",
    scoring_scheme="cann",
    checker="relative_error",
    case_spec_cls="cann",
    golden_precision="fp64_cpu",          # 升精度到 float64 + CPU 计算
    perf_metric_strategy="kernel_details", # CSV 为唯一 elapsed_us 数据源
    default_tasks_root="",                # 由 Config.tasks_root 覆盖
)
```

### 7.5 版本演进

详细版本变更记录请参阅 [docs/changelog.md](../changelog.md)。

| 版本 | 日期 | 关键变化 |
|------|------|------|
| V0.1.0 | 2026-04-25 | 初版发布，基础评测框架 |
| V0.1.1 | 2026-04-29 | 文档重组，精度标准完善 |
| V0.2.0 | 2026-05-07 | 评分体系切换为 hardware-anchored 公式 |
| V0.3.0 | 2026-05-19 | 基类层分离 + benches 扁平化 + Registry 完善 |
| V0.4.0 | 2026-06-25 | 多硬件多卡评测 + StanfordBench 集成 + 反作弊加固 + HAP 性能口径完善 + 报告系统重构 |
