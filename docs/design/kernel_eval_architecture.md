# kernel_eval 模块架构图

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI 入口层                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         cli.py                                       │    │
│  │  命令: eval | list | info | config | eval-child | simulate           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      simulation.py                                   │    │
│  │  CPU 仿真评测入口（独立入口 / eval --device cpu 等价）               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              配置 + 版本层                                    │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────┐    │
│  │        config.py         │  │          _version.py                 │    │
│  │  Config 数据类:          │  │  FRAMEWORK_VERSION (VERSION 文件)    │    │
│  │  - device_type/warmup    │  │  TASKS_VERSION (tasks/VERSION)       │    │
│  │  - bench_name/checker    │  │                                      │    │
│  │  - eval_seed             │  │                                      │    │
│  │  - torch_op_guard_mode   │  │                                      │    │
│  │  - perf_metric_strategy  │  │                                      │    │
│  │  - enable_accuracy_retry │  │                                      │    │
│  └──────────────────────────┘  └──────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Registry 层                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │LoaderRegistry│  │GoldenRegistry│  │MatcherRegistry│ │CheckerRegistry│   │
│  │ Loader注册   │  │ Golden注册   │  │ Matcher注册  │  │ Checker注册  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ScoringRegistry│ │CaseSpecRegistry│ │PerfStrategyReg│ │ BenchRegistry │   │
│  │ Scoring注册  │  │ CaseSpec注册 │  │ 性能策略注册 │  │ BenchConfig   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  │ 评测集聚合配置│   │
│                                                        └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Base 层（基类）                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  models.py   │  │  result.py   │  │  loaders.py  │  │  checker.py  │    │
│  │ TaskSpec     │  │ AccuracyResult│  │ TaskLoader   │  │ Checker基类  │    │
│  │ CaseSpec     │  │ PerfResult   │  │ CaseLoader   │  │              │    │
│  │ InputSpec    │  │ OutputResult │  │ GoldenLoader │  │              │    │
│  │ OutputSpec   │  │ (ABC+注册表) │  │ OperatorDir  │  │              │    │
│  │ SolutionSpec │  │              │  │              │  │              │    │
│  │ AttrSpec     │  │              │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────────┐   │
│  │  matcher.py  │  │  scoring.py  │  │        perf_strategy.py          │   │
│  │ MatcherBase  │  │ ScoringBase  │  │  PerfMetricStrategy ABC          │   │
│  │              │  │ CaseScoreInfo│  │  ProfFileLocations               │   │
│  └──────────────┘  └──────────────┘  │  KernelDetailsStrategy (默认)   │   │
│                                       │  TraceViewStrategy (PYPTO)      │   │
│  ┌──────────────┐                     │  MsProfSummaryStrategy (采集)   │   │
│  │  enums.py    │                     │  Warmup过滤 / CSV/trace解析     │   │
│  │ 枚举定义     │                     └─────────────────────────────────┘   │
│  └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Benches 层（CANN + Stanford 多评测集）                      │
│  ┌─────────────────────────────── CANN 评测集 ──────────────────────────┐   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │cann_loader.py│  │ cann_spec.py │  │cann_matcher.py│              │    │
│  │  │CannTaskLoader│  │CannTaskSpec  │  │OperatorMatcher│              │    │
│  │  │CannCaseLoader│  │CannCaseSpec  │  │              │              │    │
│  │  │GoldenLoader  │  │CannInputSpec │  │              │              │    │
│  │  │              │  │CannOutputSpec│  │              │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │    │
│  │  │cann_scoring.py│ │cann_solution │  │       cann.py            │  │    │
│  │  │CannScoringSch│ │CannSolution  │  │  导出 + Bench注册(入口)   │  │    │
│  │  │SimpleCompar. │ │              │  │                          │  │    │
│  │  │RecordingOnly │ │              │  │                          │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │    │
│  └───────────────────────────────────────────────────────────────────┘   │
│  ┌───────────────────── Stanford 评测集 ──────────────────────────────┐   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │stanford_loader│ │stanford_matcher│ │stanford_scoring│             │    │
│  │  │StanfordTaskL │ │StanfordMatch │ │StanfordScoring│             │    │
│  │  │StanfordCaseL │ │              │ │              │              │    │
│  │  │StanfordGolden│ │              │ │              │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │                     stanford.py                               │  │    │
│  │  │            导出 + Registry 注册                                │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Checkers 层（通用精度判断器，不绑定评测集）                  │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐         │
│  │  relative_error_checker.py   │  │    allclose_checker.py      │         │
│  │  RelativeErrorChecker        │  │    AllCloseChecker          │         │
│  │  RelativeErrorOutputResult   │  │    AllCloseOutputResult     │         │
│  │  (MERE/MARE + 小值域 + 相消) │  │    (torch.allclose 简化)    │         │
│  │  注册名: relative_error      │  │    注册名: allclose         │         │
│  │  兼容名: cann_default        │  │                              │         │
│  └──────────────────────────────┘  └──────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              评测核心层                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     eval/evaluator.py                                │    │
│  │              综合评测调度器（BenchConfig 依赖注入）                   │    │
│  │              Golden 精度策略三档 + 渐进设备恢复                       │    │
│  │              增量输出(OOM恢复) + 防作弊二次验证                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│          ┌───────────────────────────┼───────────────────────────┐          │
│          ▼                           ▼                           ▼          │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐      │
│  │ accuracy_eval│          │  perf_eval   │          │  op_runner   │      │
│  │  精度评测    │          │  性能评测    │          │  算子执行    │      │
│  │  Checker三输入│          │  Strategy解析│          │  TorchOpGuard│      │
│  │  AI/Golden/  │          │  ProfFileLoc │          │  返回值检查  │      │
│  │  同精度参考  │          │              │          │              │      │
│  └──────────────┘          └──────────────┘          └──────────────┘      │
│                                      │                                       │
│          ┌───────────────────────────┼───────────────────────────┐          │
│          ▼                           ▼                           ▼          │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐      │
│  │ process_pool │          │input_pool    │          │  results     │      │
│  │ 进程池协调   │          │ 输入池管理   │          │  结果数据    │      │
│  │ (TaskUnit调度│          │ (防缓存攻击) │          │ (failure_type│      │
│  │  eval-child) │          │              │          │  增量输出)   │      │
│  └──────────────┘          └──────────────┘          └──────────────┘      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │       failure_synthesizer + subprocess_utils                     │    │
│  │       失败合成(编译/安全/OOM) + OOM保护 + 部分结果恢复            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据工具层                                      │
│  ┌──────────────┐  ┌──────────────────────────────────────────────────┐    │
│  │data_generator│  │              package_manager                     │    │
│  │  数据生成    │  │           包管理（迭代隔离编译、安装、接口扫描）   │    │
│  │ (确定性种子) │  │                                                  │    │
│  └──────────────┘  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              工具支持层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │device_manager│  │param_builder │  │path_resolver │  │baseline_resolver│  │
│  │  设备管理    │  │  参数构建    │  │  路径解析    │  │  基线解析    │    │
│  │ (渐进恢复)   │  │  (签名解析)  │  │  (task-dir)  │  │  (多硬件)    │    │
│  │ light→full  │  │              │  │              │  │  (平台别名)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │baseline_store│  │   compare    │  │tensor_utils  │  │   naming     │    │
│  │ 集中存储    │  │  张量对比    │  │  Tensor工具  │  │  命名转换    │    │
│  │ metadata/json│  │ (MERE/MARE+ │  │ (FP64/CPU)   │  │ (Pascal→snake│   │
│  │ 多平台fallback│ │ 小值域+相消) │  │              │  │  多候选)     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐                                         │
│  │dtype_mapper  │  │  thresholds  │                                         │
│  │  类型映射    │  │  精度阈值    │                                         │
│  │ (str→dtype)  │  │  (单一来源)  │                                         │
│  └──────────────┘  └──────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              报告生成层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │report_generator│ │html_generator│ │   scoring    │  │ setup_info   │    │
│  │  报告生成    │  │  HTML渲染    │  │  评分计算    │  │  配置采集    │    │
│  │ (JSON/MD/   │  │ (KPI+柱状图 │  │ (Eq.3/4/5)  │  │ (NPU/CANN/  │    │
│  │  Summary)   │  │  认证印章)   │  │              │  │  PyTorch版本)│    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐                                                           │
│  │summary_gen   │                                                           │
│  │  Summary生成 │                                                           │
│  │  (几何平均)  │                                                           │
│  └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              安全防护层                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           api_guard                                  │   │
│  │              Timing API 防护（快照+验证+恢复）                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        torch_op_guard                                │   │
│  │              Torch 算子守卫（TorchFunctionMode + 多路径规约）         │   │
│  │              模式: off / warn / block；pause() 机制排除预热调用       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         type_checker                                 │   │
│  │              返回值类型检查（type() is torch.Tensor，拒绝 FakeTensor）│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
src/kernel_eval/
├── __init__.py            # 包入口，导出 Config + 版本号
├── cli.py                 # CLI 命令入口（eval/list/info/config/eval-child/simulate）
├── config.py              # 全局配置（Config 数据类 + 全局配置实例）
├── simulation.py          # CPU 仿真评测模块
├── _version.py            # 版本管理（从 VERSION 文件动态读取）
│
├── base/                  # 基类层（抽象定义，无特化依赖）
│   ├── __init__.py        # 导出所有基类
│   ├── enums.py           # DifficultyLevel, BackendType, SourceType, GoldenReference, EvaluationMode
│   ├── models.py          # AttrSpec, InputSpec, OutputSpec, TaskSpec, CaseSpec, SolutionSpec
│   ├── result.py          # OutputResult(ABC+注册表), AccuracyResult, PerfResult, compute_speedup
│   ├── loaders.py         # TaskLoader, CaseLoader, GoldenLoaderBase, OperatorDirMixin
│   ├── checker.py         # CorrectnessChecker (抽象基类)
│   ├── matcher.py         # OperatorMatcherBase (抽象基类)
│   ├── scoring.py         # ScoringScheme, CaseScoreInfo (抽象基类)
│   └── perf_strategy.py   # PerfMetricStrategy ABC, ProfFileLocations,
│                           #   KernelDetailsStrategy, TraceViewStrategy, MsProfSummaryStrategy
│                           #   warmup过滤/CSV/trace_view/msprof解析函数
│
├── benches/               # 评测集插件层（CANN + Stanford 多评测集）
│   ├── __init__.py        # 聚合导入，导入所有评测集模块触发注册
│   ├── cann.py            # 导出 + Registry 注册（核心入口）
│   ├── cann_loader.py     # CannTaskLoader, CannCaseLoader, GoldenLoader
│   ├── cann_spec.py       # CannTaskSpec, CannCaseSpec, CannInputSpec, CannOutputSpec
│   ├── cann_matcher.py    # OperatorMatcher（torch.ops.cann_bench 或 cann_bench 模块）
│   ├── cann_scoring.py    # CannScoringScheme, SimpleComparisonScheme, RecordingOnlyScheme
│                           #   ScoringCalculator, OperatorScoreInfo, per_case_sol_score, aggregate_eq4
│   ├── cann_solution.py   # CannSolutionSpec
│   ├── stanford.py        # 导出 + Registry 注册
│   ├── stanford_loader.py # StanfordTaskLoader, StanfordCaseLoader, StanfordGoldenLoader
│   ├── stanford_matcher.py # StanfordMatcher
│   └── stanford_scoring.py # StanfordScoringScheme
│
├── checkers/              # Checker 层（通用精度判断器，不绑定评测集）
│   ├── __init__.py
│   ├── relative_error_checker.py  # RelativeErrorChecker, RelativeErrorOutputResult
│   │                                # 注册名: relative_error + 兼容名 cann_default
│   └── allclose_checker.py        # AllCloseChecker, AllCloseOutputResult
│                                    # 注册名: allclose
│
├── registry/              # 注册机制层
│   ├── __init__.py        # 导出所有 Registry + 便捷获取函数
│   ├── base.py            # BaseRegistry (泛型基类: register/get/list_all/clear)
│   ├── loader_registry.py # LoaderRegistry: TaskLoader/CaseLoader 注册
│   ├── golden_registry.py # GoldenLoaderRegistry: GoldenLoader 注册
│   ├── matcher_registry.py # OperatorMatcherRegistry: Matcher 注册
│   ├── checker_registry.py # CheckerRegistry: Checker 注册
│   ├── scoring_registry.py # ScoringSchemeRegistry: ScoringScheme 注册
│   ├── case_spec_registry.py # CaseSpecRegistry: CaseSpec 子类注册
│   ├── perf_strategy_registry.py # PerfStrategyRegistry: 性能策略注册
│   └── bench_registry.py  # BenchRegistry + BenchConfig（评测集配置聚合）
│                            #   get_bench_config, get_bench_components
│
├── data/                  # 数据工具层
│   ├── __init__.py        # 导出 DataGenerator + PackageManager
│   ├── data_generator.py  # DataGenerator (确定性种子 tensor 生成)
│   └── package_manager.py # PackageManager (迭代隔离编译、安装、接口扫描)
│                           #   PackageInfo, InterfaceInfo
│
├── eval/                  # 评测执行层
│   ├── __init__.py        # 导出所有评测组件
│   ├── evaluator.py       # Evaluator (综合调度: BenchConfig依赖注入 + 渐进设备恢复
│                           #   增量输出 + 防作弊二次验证 + Golden精度策略)
│   ├── op_runner.py       # OpRunner (算子执行器: TorchOpGuard + 返回值检查)
│   ├── accuracy_eval.py   # AccuracyEvaluator (精度评测: Checker三输入模式)
│   ├── perf_eval.py       # PerfEvaluator (性能评测: Profiler + Strategy委托解析)
│   ├── results.py         # EvalCaseResult, EvalOperatorResult, EvalSessionResult
│                           #   summarize_case_results, CaseResultSummary, failure_type
│   ├── input_pool.py      # InputPool (防缓存攻击: clone池 + 内存限制)
│   ├── process_pool.py    # ProcessPoolCoordinator (多卡并行: TaskUnit调度)
│                           #   ProcessConfig, TaskUnit, build_task_units
│   ├── failure_synthesizer.py # FailureSynthesizer (编译/安全/子进程/OOM 失败合成)
│   └── subprocess_utils.py # OOM保护 + CANN环境变量 + 部分结果恢复
│                            #   _write_oom_score_adj, _is_oom_killed, _try_recover_partial_results
│
├── report/                # 报告生成层
│   ├── __init__.py
│   ├── report_generator.py # ReportGenerator (JSON+Markdown+Summary+HTML 多格式)
│   │                          #   EvalReport, OperatorReport; 语义前缀命名
│   ├── html_generator.py   # render_html_report (4Section + KPI + 柱状图 + 认证印章)
│   ├── scoring.py          # per_case_sol_score (Eq.3), aggregate_eq4 (Eq.4)
│   │                          #   _fallback_baseline_from_hw
│   ├── summary_generator.py # SummaryGenerator (几何平均加速比)
│   └── setup_info.py       # collect_setup_info (metadata + environment 采集)
│
├── utils/                 # 工具层
│   ├── __init__.py
│   ├── device_manager.py   # DeviceManager, DeviceConfig (渐进恢复: light→full→unrecoverable)
│   ├── param_builder.py    # ParamBuilder (签名解析 + 参数构建)
│   ├── path_resolver.py    # 路径解析
│   ├── baseline_resolver.py # BaselineResolver (多硬件 dict + 平台别名映射)
│   ├── baseline_store.py   # BaselineStore (metadata/<hw>.json 集中存储 + 多平台 fallback)
│   ├── compare.py          # compare_tensors (MERE/MARE+小值域+相消+整数精确匹配)
│   │                          #   SingleOutputResult, CompareResult
│   ├── thresholds.py       # PRECISION_THRESHOLDS (精度阈值表，单一事实来源)
│   ├── dtype_mapper.py     # dtype 映射 (str→dtype)
│   ├── tensor_utils.py     # tensors_to_cpu, tensors_to_fp64_cpu
│   └── naming.py           # camel_to_snake, snake_case_candidates (多候选模糊匹配)
│
└── security/              # 安全防护层
    ├── __init__.py
    ├── api_guard.py        # APIGuard (Timing API 快照+验证+恢复)
    ├── torch_op_guard.py   # TorchOpGuard (TorchFunctionMode + 多路径规约 + pause)
    │                          #   BUILTIN_COMPUTE_OPS, _LEAF_TO_CANONICAL
    └── type_checker.py     # 返回值类型检查
```

## 模块职责详解

### 1. Base 层 (base/)

基类定义，提供抽象接口和通用数据结构：

```
┌─────────────────────────────────────────────────────────────────┐
│                         base/                                   │
│                       基类定义层                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  enums.py:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DifficultyLevel   - L1/L2/L3/L4 (难度级别)             │   │
│  │  BackendType       - torch/torch_npu/ascendc/… (后端)   │   │
│  │  SourceType        - file/code/module/generated          │   │
│  │  GoldenReference   - file/self/fp64_cpu/none             │   │
│  │  EvaluationMode    - accuracy/performance/full           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  models.py:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AttrSpec         - 算子属性规格                          │   │
│  │  TaskSpec          - 任务规格基类 (含 category/precision_thresholds) │ │
│  │  CaseSpec          - 用例规格基类 (含 baseline_perf_us/t_hw_us) │ │
│  │  InputSpec         - 输入规格基类                        │   │
│  │  OutputSpec        - 输出规格基类 (含 compare 标记)      │   │
│  │  SolutionSpec      - 解决方案规格基类                    │   │
│  │  全部支持 to_dict/from_dict 序列化                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  result.py:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  OutputResult (ABC) - 单输出结果基类                     │   │
│  │    + 注册表: register_output_result / get_output_result_cls│ │
│  │  AccuracyResult    - 精度结果 (passed/output_results/metadata) │ │
│  │  PerfResult        - 性能结果 (elapsed_us/op_times)      │   │
│  │    + 便捷方法: get_speedup/get_baseline_us/get_t_hw_us │   │
│  │  compute_speedup   - 加速比计算函数                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  loaders.py:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TaskLoader        - 任务加载器 ABC (list_tasks/get_task)│   │
│  │  CaseLoader        - 用例加载器 ABC (scan_all/scan_by_task)│ │
│  │  GoldenLoaderBase  - Golden 加载器 ABC                   │   │
│  │    (get_golden_function / get_input_function /           │   │
│  │     get_output_function)                                 │   │
│  │  OperatorDirMixin  - 算子目录扫描 (proto+cases+golden)   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  checker.py:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CorrectnessChecker - 精度判断器 ABC                     │   │
│  │    (get_name / check 三输入模式)                         │   │
│  │    辅助: _normalize_outputs / _ensure_cpu /              │   │
│  │          _check_output_count                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  matcher.py:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  OperatorMatcherBase - AI 算子匹配器 ABC                 │   │
│  │    (load_ai_operator / find_operator_info / clear_cache)│   │
│  │    (find_operator_info_by_snake — snake 反查)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  scoring.py:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ScoringScheme     - 评分方案 ABC                        │   │
│  │    (prepare_baseline / calculate_case_score / aggregate) │   │
│  │  CaseScoreInfo     - 用例级得分信息                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  perf_strategy.py:                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PerfMetricStrategy - 性能指标策略 ABC                   │   │
│  │    (parse / get_strategy_name)                           │   │
│  │  ProfFileLocations  - 文件定位结果 (csv/trace/msprof)   │   │
│  │  KernelDetailsStrategy - 默认策略 (CSV唯一源)            │   │
│  │  TraceViewStrategy   - PYPTO口径 (待收编)               │   │
│  │  MsProfSummaryStrategy - 基准采集 (CSV→msprof fallback) │   │
│  │  共享函数:                                              │   │
│  │    extract_warmup_names_from_csv / parse_trace_view_    │   │
│  │    kernels / parse_csv_kernels / parse_tilefwk_metrics │   │
│  │    parse_msprof_op_summary                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Benches 层 (benches/)

多评测集插件实现，每个评测集独立注册：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           benches/ 模块架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────── CANN 评测集 ────────────────────────────────────┐ │
│   │   cann.py（核心入口）:                                                │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   1. 导入所有 CANN 特化组件                                    │  │ │
│   │   │   2. 注册到 Registry（Loader/Golden/Matcher/Checker/Scoring/  │  │ │
│   │   │      CaseSpec/PerfStrategy/BenchConfig("cann"))                │  │ │
│   │   │   3. 执行注册 _register_cann_components()                     │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   cann_loader.py:                                                      │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   CannTaskLoader   - CANN 任务加载器 (proto.yaml 解析)         │  │ │
│   │   │   CannCaseLoader   - CANN 用例加载器 (cases.yaml 解析+校验)   │  │ │
│   │   │   GoldenLoader     - Golden 函数加载器 (动态导入+命名转换)     │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   cann_spec.py:                                                        │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   CannTaskSpec    - CANN 任务规格 (继承 TaskSpec)              │  │ │
│   │   │   CannCaseSpec    - CANN 用例规格 (继承 CaseSpec,              │  │ │
│   │   │     增加 baseline_perf_us dict / t_hw_us / golden_reference)  │  │ │
│   │   │   CannInputSpec / CannOutputSpec / CannSolutionSpec            │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   cann_matcher.py:                                                     │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   OperatorMatcher - AI 算子匹配器                              │  │ │
│   │   │   (torch.ops.cann_bench 或 cann_bench 模块加载)               │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   cann_scoring.py:                                                     │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   CannScoringScheme       - hardware-anchored Eq.3/4/5         │  │ │
│   │   │   SimpleComparisonScheme  - 加速比评分方案                    │  │ │
│   │   │   RecordingOnlyScheme     - 仅记录方案                        │  │ │
│   │   │   ScoringCalculator / OperatorScoreInfo / per_case_sol_score   │  │ │
│   │   │   aggregate_eq4           - Eq.4 聚合计算                    │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   cann_solution.py:                                                    │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   CannSolutionSpec - CANN 解决方案规格 (继承 SolutionSpec)    │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌───────────────────── Stanford 评测集 ──────────────────────────────┐ │
│   │   stanford.py（核心入口）:                                            │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   注册到 Registry（BenchConfig("stanford"))                    │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   stanford_loader.py:                                                  │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   StanfordTaskLoader / StanfordCaseLoader / StanfordGoldenLoader │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   stanford_matcher.py:                                                 │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   StanfordMatcher                                              │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   │                                                                        │ │
│   │   stanford_scoring.py:                                                 │ │
│   │   ┌───────────────────────────────────────────────────────────────┐  │ │
│   │   │   StanfordScoringScheme                                         │  │ │
│   │   └───────────────────────────────────────────────────────────────┘  │ │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Checkers 层 (checkers/)

通用精度判断器，不与任何评测集绑定，通过注册机制供 BenchConfig 选择：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           checkers/ 模块架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   relative_error_checker.py:                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   RelativeErrorChecker - 相对误差精度判断器                         │  │
│   │   (继承 CorrectnessChecker)                                         │  │
│   │                                                                      │  │
│   │   判断标准:                                                          │  │
│   │   - 浮点: MERE < threshold AND MARE < 10*threshold                  │  │
│   │   - 小值域: ErrorCount 比值标准                                     │  │
│   │   - 相消处理: CPU 同精度对照标准                                    │  │
│   │   - 整数: 精确匹配                                                  │  │
│   │   - 多输出支持                                                      │  │
│   │                                                                      │  │
│   │   RelativeErrorOutputResult: 逐输出详细结果                         │  │
│   │   注册名: relative_error + 兼容名 cann_default                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   allclose_checker.py:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   AllCloseChecker - torch.allclose 简化对比                         │  │
│   │   AllCloseOutputResult                                              │  │
│   │   注册名: allclose                                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Registry 层 (registry/)

注册机制，支持多评测集接入和组件动态选择：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           registry/ 模块架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   base.py:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   BaseRegistry[T] - 泛型注册基类                                    │  │
│   │   register(name, item) / get(name) / list_all() / clear()           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   loader_registry.py:        golden_registry.py:                            │
│   ┌───────────────────┐    ┌───────────────────┐                          │
│   │ LoaderRegistry    │    │ GoldenLoaderReg   │                          │
│   │ TaskLoader 注册   │    │ GoldenLoader 注册 │                          │
│   │ CaseLoader 注册   │    │                   │                          │
│   │ get_task_loader   │    │ get_golden_loader │                          │
│   │ get_case_loader   │    │                   │                          │
│   └───────────────────┘    └───────────────────┘                          │
│                                                                             │
│   matcher_registry.py:      checker_registry.py:                           │
│   ┌───────────────────┐    ┌───────────────────┐                          │
│   │ MatcherRegistry   │    │ CheckerRegistry   │                          │
│   │ Matcher 注册      │    │ Checker 注册      │                          │
│   │ get_operator_match│    │ get_correctness_  │                          │
│   │                   │    │ register_correctn │                          │
│   └───────────────────┘    └───────────────────┘                          │
│                                                                             │
│   scoring_registry.py:      case_spec_registry.py:                         │
│   ┌───────────────────┐    ┌───────────────────┐                          │
│   │ ScoringRegistry   │    │ CaseSpecRegistry  │                          │
│   │ ScoringScheme注册 │    │ CaseSpec 子类注册 │                          │
│   │ get_scoring_scheme│    │ (反序列化用)       │                          │
│   └───────────────────┘    └───────────────────┘                          │
│                                                                             │
│   perf_strategy_registry.py:                                                │
│   ┌───────────────────┐                                                    │
│   │ PerfStrategyReg   │                                                    │
│   │ 性能策略注册      │                                                    │
│   │ get_perf_metric_  │                                                    │
│   │ strategy          │                                                    │
│   └───────────────────┘                                                    │
│                                                                             │
│   bench_registry.py:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   BenchRegistry                                                     │  │
│   │   BenchConfig:                                                      │  │
│   │     - name: str                     (评测集名称)                     │  │
│   │     - task_loader: str              (TaskLoader 注册名)             │  │
│   │     - case_loader: str              (CaseLoader 注册名)             │  │
│   │     - golden_loader: str            (GoldenLoader 注册名)           │  │
│   │     - operator_matcher: str         (Matcher 注册名)                │  │
│   │     - scoring_scheme: str           (ScoringScheme 注册名)          │  │
│   │     - checker: str                  (Checker 注册名)                │  │
│   │     - case_spec_cls: str            (CaseSpec 子类注册名)           │  │
│   │     - golden_precision: str         (fp64_cpu/native_cpu/native_npu) │  │
│   │     - precision_thresholds: dict    (自定义精度阈值)                │  │
│   │     - perf_metric_strategy: str     (kernel_details/trace_view/...) │  │
│   │     - default_tasks_root: str       (默认数据目录)                  │  │
│   │     - description: str                                              │  │
│   │                                                                      │  │
│   │   Methods:                                                           │  │
│   │     - get_task_loader(**kwargs)                                      │  │
│   │     - get_case_loader(**kwargs)                                      │  │
│   │     - get_golden_loader(**kwargs)                                    │  │
│   │     - get_operator_matcher(operator_loader)                          │  │
│   │     - get_scoring_scheme()                                           │  │
│   │     - get_checker()                                                  │  │
│   │     - get_case_spec_cls()                                            │  │
│   │     - get_precision_thresholds()                                     │  │
│   │     - get_perf_metric_strategy()                                    │  │
│   │                                                                      │  │
│   │   便捷函数:                                                          │  │
│   │     - get_bench_config(name)                                         │  │
│   │     - get_bench_components(name, tasks_root)                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. 评测核心层 (eval/)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           eval/ 模块架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                       Evaluator (evaluator.py)                      │  │
│   │                    综合评测调度器                                    │  │
│   │                                                                     │  │
│   │   构造: Config + BenchConfig 依赖注入                               │  │
│   │   通过 Registry 获取所有组件实例                                    │  │
│   │                                                                     │  │
│   │   evaluate_case(case):                                              │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ 1. 确定性种子 (SHA256 hash + eval_seed)                      │  │  │
│   │   │ 2. 生成输入张量 (DataGenerator, seed=case_seed)              │  │  │
│   │   │ 2.5 get_input 预处理 (如有)                                  │  │  │
│   │   │ 3. 构建调用参数 (ParamBuilder 或 golden 签名匹配)            │  │  │
│   │   │ 4. Golden 执行 (精度策略: fp64_cpu/native_cpu/native_npu)   │  │  │
│   │   │ 5. AI 算子执行 (TorchOpGuard 监听 + profiler 采集)           │  │  │
│   │   │ 6. 同精度参考 (fp64_cpu时单独执行, 其他复用golden)           │  │  │
│   │   │ 7. get_output 后处理 (如有, 对 golden/AI/同精度参考三路)     │  │  │
│   │   │ 8. Checker 三输入模式 (AI/Golden/同精度参考)                 │  │  │
│   │   │ 8.5 防作弊二次验证 (enable_accuracy_retry, 偏移种子+微扰)   │  │  │
│   │   │ 9. 性能数据 (从 profiler 运行中直接提取)                    │  │  │
│   │   │ 10. 生成 EvalCaseResult                                     │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   │   run_cases(cases): 渐进设备恢复                                    │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ healthy → recovering → unrecoverable                        │  │  │
│   │   │ 连续≥3失败 → recover_light → recover_full                  │  │  │
│   │   │ profiler活跃时不尝试 recover_full                           │  │  │
│   │   │ 增量输出: 每用例完成后写入 incremental_output_path           │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                    OpRunner (op_runner.py)                         │    │
│   │                      算子执行器                                    │    │
│   │                                                                   │    │
│   │  run(func, params, ...) → OpRunResult (通用执行)                  │    │
│   │  run_ai_op(op_func, params, ...) → TorchOpGuard 监听 + profiler  │    │
│   │  返回值类型检查 + 设备迁移                                        │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌──────────────────┐        ┌──────────────────────────────────┐         │
│   │ AccuracyEvaluator │        │       PerfEvaluator              │         │
│   │   (accuracy_eval) │        │       (perf_eval)                │         │
│   │                  │        │                                  │         │
│   │  精度对比:       │        │  性能采集:                       │         │
│   │  - Checker三输入 │        │  - Profiler 运行                 │         │
│   │  - AI/Golden/   │        │  - 文件定位 (_locate_prof_files) │         │
│   │    同精度参考    │        │  - 解析委托 PerfMetricStrategy   │         │
│   │  - 注册机制选择 │        │  - 升频清cache + pause guard     │         │
│   └──────────────────┘        └──────────────────────────────────┘         │
│                                                                             │
│   ┌──────────────────┐        ┌──────────────────────────────────┐         │
│   │    Results       │        │      FailureSynthesizer          │         │
│   │    (results)     │        │   (failure_synthesizer)          │         │
│   │                  │        │                                  │         │
│   │  数据结构:       │        │  失败结果合成:                   │         │
│   │  - EvalCaseResult│        │  - synthesize_compile_failure    │         │
│   │    (failure_type)│        │  - synthesize_security_failure   │         │
│   │  - EvalOperator  │        │  - synthesize_subprocess_failure │         │
│   │    (compile_pass)│        │  - synthesize_oom_failure       │         │
│   │  - EvalSession   │        │                                  │         │
│   └──────────────────┘        └──────────────────────────────────┘         │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │         ProcessPoolCoordinator (process_pool.py)                  │    │
│   │                       进程池协调器                                │    │
│   │                                                                   │    │
│   │  多卡 × 多进程并行架构 (TaskUnit 调度):                          │    │
│   │  ┌────────────────────────────────────────────────────────────┐  │    │
│   │  │  TaskUnit = (算子, 用例组, device_id)                      │  │    │
│   │  │  按 (算子×用例组) 均分到各卡 → 自然负载均衡                 │  │    │
│   │  │  子进程通过 eval-child 独立子命令执行                       │  │    │
│   │  │  ThreadPoolExecutor + as_completed 动态调度                 │  │    │
│   │  │  profiler 开启时每卡强制 1 进程                             │  │    │
│   │  └────────────────────────────────────────────────────────────┘  │    │
│   │                                                                   │    │
│   │  异常处理:                                                        │    │
│   │  - 超时: SIGTERM → 10s → SIGKILL; 恢复部分结果                  │    │
│   │  - OOM Kill: 恢复增量写入部分结果; 合成 oom_killed 失败          │    │
│   │  - 子进程异常退出: 合成 subprocess_failure 失败                  │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌──────────────────┐        ┌──────────────────────────────────┐         │
│   │    InputPool     │        │      subprocess_utils            │         │
│   │   (input_pool)   │        │   (subprocess_utils)            │         │
│   │                  │        │                                  │         │
│   │  防缓存攻击:     │        │  OOM Killer 保护:               │         │
│   │  - 预分配clone   │        │  - _write_oom_score_adj          │         │
│   │  - 轮换使用      │        │  - _is_oom_killed               │         │
│   │  - 内存限制      │        │  CANN 环境变量继承:             │         │
│   │                  │        │  - _CANN_ENV_VARS                │         │
│   │                  │        │  部分结果恢复:                   │         │
│   │                  │        │  - _try_recover_partial_results  │         │
│   │                  │        │  - _synthesize_failure_cases     │         │
│   └──────────────────┘        └──────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6. 数据工具层 (data/)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           data/ 模块架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   data_generator.py:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │   DataGenerator                                                      │ │
│   │   - generate_input_tensors_from_case(input_shapes, dtypes,          │ │
│   │     value_ranges, seed)                                              │ │
│   │   - generate_tensor(shape, dtype, value_range, seed)                │ │
│   │   - 确定性种子: hashlib SHA256 确保跨进程可复现                      │ │
│   │   - 支持 FP64/CPU 生成                                              │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   package_manager.py:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │   PackageManager                                                     │ │
│   │   - scan_source()        → 扫描源码目录                             │ │
│   │   - check_compile()      → 检查编译状态                             │ │
│   │   - build_whl/run()      → 编译 whl 包和 run 包                    │ │
│   │   - install()            → 安装到环境                               │ │
│   │   - scan_interfaces()    → 扫描算子接口                             │ │
│   │   - match_operators()    → 匹配 tasks 算子定义                      │ │
│   │   - prepare_from_source() → APIGuard.snapshot() + 完整流程           │ │
│   │   - prepare_skip_build()  → 跳过编译直接评测                        │ │
│   │                                                                      │ │
│   │   PackageInfo: 源码包信息                                            │ │
│   │   InterfaceInfo: 算子接口信息                                        │ │
│   │                                                                      │ │
│   │   支持:                                                              │ │
│   │   - 迭代隔离编译（失败的算子隔离到 _quarantine/）                    │ │
│   │   - 多硬件 baseline 解析                                            │ │
│   │   - APIGuard snapshot 集成                                          │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7. 工具支持层 (utils/)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           utils/ 模块架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────┐              ┌───────────────────┐                │
│   │   DeviceManager   │              │   ParamBuilder    │                │
│   │  (device_manager) │              │  (param_builder)  │                │
│   │                   │              │                   │                │
│   │  功能:            │              │  功能:            │                │
│   │  - 检测 NPU/CPU   │              │  - 解析函数签名   │                │
│   │  - 设备切换       │              │  - 构建调用参数   │                │
│   │  - 张量迁移       │              │  - 处理 null 位   │                │
│   │  - health_check   │              │                   │                │
│   │  - 渐进恢复:      │              │                   │                │
│   │    recover_light  │              │                   │                │
│   │    recover_full   │              │                   │                │
│   │                   │              │                   │                │
│   │  API:             │              │  API:             │                │
│   │  get_device()     │              │  build_call_params│                │
│   │  to_device()      │              │                   │                │
│   └───────────────────┘              └───────────────────┘                │
│                                                                             │
│   ┌───────────────────┐              ┌───────────────────┐                │
│   │   PathResolver    │              │ BaselineResolver  │                │
│   │  (path_resolver)  │              │(baseline_resolver)│                │
│   │                   │              │                   │                │
│   │  功能:            │              │  功能:            │                │
│   │  - 解析 task-dir  │              │  - 解析 baseline  │                │
│   │  - 检测算子目录   │              │  - 多硬件 dict    │                │
│   │  - 处理相对路径   │              │  - 平台别名映射   │                │
│   │                   │              │   (Ascend910B2→910b2)              │
│   │                   │              │                   │                │
│   │  API:             │              │  API:             │                │
│   │  resolve_task_dir │              │  resolve_baseline │                │
│   └───────────────────┘              └───────────────────┘                │
│                                                                             │
│   ┌───────────────────┐              ┌───────────────────┐                │
│   │  BaselineStore    │              │     Compare       │                │
│   │ (baseline_store)  │              │    (compare)      │                │
│   │                   │              │                   │                │
│   │  功能:            │              │  功能:            │                │
│   │  - 集中存储       │              │  - compare_tensors │                │
│   │  - metadata/json  │              │  - MERE/MARE 计算 │                │
│   │  - 多平台fallback │              │  - 小值域兜底     │                │
│   │  - 向上查找       │              │  - 相消处理       │                │
│   │  - 三级嵌套JSON   │              │  - 整数精确匹配   │                │
│   │                   │              │  - 多输出支持     │                │
│   │  API:             │              │                   │                │
│   │  get_perf()       │              │  API:             │                │
│   │  get_t_hw()       │              │  compare_tensors() │                │
│   │  has_baseline()   │              │                   │                │
│   └───────────────────┘              └───────────────────┘                │
│                                                                             │
│   ┌───────────────────┐              ┌───────────────────┐                │
│   │   DTypeMapper     │              │    Thresholds     │                │
│   │  (dtype_mapper)   │              │   (thresholds)    │                │
│   │                   │              │                   │                │
│   │  功能:            │              │  功能:            │                │
│   │  - str → dtype    │              │  - PRECISION_     │                │
│   │  - dtype → str    │              │    THRESHOLDS     │                │
│   │                   │              │  - 单一事实来源    │                │
│   │  API:             │              │  - Config 可拷贝   │                │
│   │  str_to_dtype()   │              │    覆盖单dtype    │                │
│   │  dtype_to_str()   │              │                   │                │
│   └───────────────────┘              └───────────────────┘                │
│                                                                             │
│   ┌───────────────────┐              ┌───────────────────┐                │
│   │   TensorUtils     │              │     Naming        │                │
│   │  (tensor_utils)   │              │    (naming)       │                │
│   │                   │              │                   │                │
│   │  功能:            │              │  功能:            │                │
│   │  - FP64 转换      │              │  - Pascal→snake   │                │
│   │  - CPU 迁移       │              │  - 多候选生成     │                │
│   │  - 批量处理       │              │  - 处理 3D/ROI等  │                │
│   │                   │              │                   │                │
│   │  API:             │              │  API:             │                │
│   │  tensors_to_fp64  │              │  camel_to_snake   │                │
│   │  tensors_to_cpu   │              │  snake_case_      │                │
│   │                   │              │  candidates       │                │
│   └───────────────────┘              └───────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8. 报告生成层 (report/)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           report/ 模块架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                     ReportGenerator                                 │ │
│   │                      报告生成器                                      │ │
│   │                                                                     │ │
│   │  输入: EvalSessionResult + setup_info                               │ │
│   │  输出:                                                              │ │
│   │  ┌─────────────────────────────────────────────────────────────┐   │ │
│   │  │  JSON 格式:  详细结果数据 + 统计 + 时间戳                    │   │ │
│   │  └─────────────────────────────────────────────────────────────┘   │ │
│   │  ┌─────────────────────────────────────────────────────────────┐   │ │
│   │  │  Markdown 格式: 表格 + 详细用例结果                          │   │ │
│   │  └─────────────────────────────────────────────────────────────┘   │ │
│   │  ┌─────────────────────────────────────────────────────────────┐   │ │
│   │  │  Summary 格式:  通过率 + 几何平均加速比                      │   │ │
│   │  └─────────────────────────────────────────────────────────────┘   │ │
│   │  ┌─────────────────────────────────────────────────────────────┐   │ │
│   │  │  HTML 格式:  4Section + KPI + 柱状图 + 认证印章              │   │ │
│   │  │  动态字段替换: 通过率/得分/Agent/Skill/BaseModel             │   │ │
│   │  └─────────────────────────────────────────────────────────────┘   │ │
│   │                                                                     │ │
│   │  语义前缀: cann_eval_ / stanford_eval_ / cpu_sim_cann_            │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                   HTML Generator                                  │    │
│   │                   (html_generator.py)                             │    │
│   │                                                                   │    │
│   │   结构:                                                           │    │
│   │   Section 1: Abstract (description.html)                          │    │
│   │   Section 2: Experiment Setup (setup_info)                        │    │
│   │   Section 3: Results Analysis                                    │    │
│   │     3.1 KPI 指标卡 (通过率/算子数/得分/级联失败)                  │    │
│   │     3.2 等级分析表                                               │    │
│   │     3.3 柱状图 (得分/通过率/加速比)                               │    │
│   │     3.4 Top 算子分析表                                            │    │
│   │   Section 4: Operator Details (按 Level 分组)                    │    │
│   │   Certification Seal                                             │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                   Scoring                                        │    │
│   │                   (scoring.py)                                    │    │
│   │                                                                   │    │
│   │   - per_case_sol_score (Eq.3): hardware-anchored                 │    │
│   │     score_i = (T_baseline - T_HW) / ((T_cand - T_HW) +          │    │
│   │              (T_baseline - T_HW))                                 │    │
│   │   - aggregate_eq4 (Eq.4): 算子综合评分                            │    │
│   │   - _fallback_baseline_from_hw: max(t_hw*10, 10)                  │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌──────────────────┐    ┌──────────────────────────────────────────┐    │
│   │SummaryGenerator  │    │           SetupInfo                     │    │
│   │(summary_generator)│    │          (setup_info.py)                │    │
│   │                  │    │                                          │    │
│   │  几何平均加速比  │    │  采集:                                   │    │
│   │  通过率统计      │    │  metadata: framework/date/agent_skill/  │    │
│   │  综合得分汇总    │    │    base_model/benchmark/license         │    │
│   │                  │    │  environment: npu/cpu/cann/pytorch/     │    │
│   │                  │    │    python/os/docker                     │    │
│   └──────────────────┘    └──────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9. 安全防护层 (security/)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           security/ 模块架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   api_guard.py:                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   APIGuard                                                          │  │
│   │   - snapshot() → 快照关键 Timing API 身份                           │  │
│   │     (torch.npu.Event.elapsed_time/record, synchronize,              │  │
│   │      torch_npu.profiler.profile/schedule)                           │  │
│   │   - verify()  → 验证是否被篡改，篡改则恢复原始API                   │  │
│   │   - 集成: PackageManager.prepare_from_source() (snapshot)           │  │
│   │            Evaluator.evaluate_from_source() (verify)                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   torch_op_guard.py:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   TorchOpGuard                                                       │  │
│   │   - TorchFunctionMode 上下文管理器 (PyTorch ≥1.11)                  │  │
│   │   - 检测 AI 算子调用 PyTorch 内置计算 API                            │  │
│   │   - BUILTIN_COMPUTE_OPS: matmul/mm/conv/softmax/attention/...      │  │
│   │   - _LEAF_TO_CANONICAL: 多路径规约                                  │  │
│   │     (torch.matmul / torch.ops.aten.mm / @ / F.linear →             │  │
│   │      同一 canonical name, 防止绕过)                                  │  │
│   │   - 模式: off / warn / block                                       │  │
│   │   - pause() 机制: 排除 harness 自身预热调用                         │  │
│   │   - 集成: OpRunner.run_ai_op() (with TorchOpGuard)                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   type_checker.py:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   type_checker                                                       │  │
│   │   - type(output) is torch.Tensor 严格检查                            │  │
│   │   - 拒绝 FakeTensor / 懒求值包装器                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           评测数据流                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   输入:                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  tasks/                                                              │ │
│   │  ├── level1/exp/                                                    │ │
│   │  │   ├── proto.yaml       # 算子定义                                │ │
│   │  │   ├── cases.yaml       # 测试用例 (含 baseline/t_hw)             │ │
│   │  │   └── golden.py        # Golden 函数                             │ │
│   │  └── ...                                                             │ │
│   │  └── metadata/910b2.json  # Baseline 集中存储                       │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                      CLI (cli.py)                                   │ │
│   │  python -m kernel_eval eval --bench-name cann                      │ │
│   │                                     --operator Exp                  │ │
│   │                                     --eval-seed 0                   │ │
│   │                                     --torch-op-guard-mode block     │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                  Evaluator (evaluator.py)                           │ │
│   │                                                                     │ │
│   │  for each case:                                                     │ │
│   │  ┌─────────────────────────────────────────────────────────────┐   │ │
│   │  │                                                             │   │ │
│   │  │  ① BenchRegistry.get('cann') → BenchConfig                 │   │ │
│   │  │     (golden_precision=fp64_cpu, checker=relative_error,    │   │ │
│   │  │      perf_metric_strategy=kernel_details)                   │   │ │
│   │  │                                                             │   │ │
│   │  │  ② LoaderRegistry.get_case_loader('cann') → CannCaseLoader │   │ │
│   │  │     → CannCaseSpec(input_shapes, dtypes, attrs,            │   │ │
│   │  │       baseline_perf_us, t_hw_us, golden_reference)         │   │ │
│   │  │                                                             │   │ │
│   │  │  ③ GoldenLoaderRegistry.get('cann') → GoldenLoader          │   │ │
│   │  │     → golden_func (Callable)                                │   │ │
│   │  │                                                             │   │ │
│   │  │  ④ 确定性种子 (SHA256 + eval_seed) → case_seed              │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑤ DataGenerator.generate_input_tensors(case, seed)        │   │ │
│   │  │     → input_tensors (List[Tensor])                          │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑥ _apply_golden_precision(inputs) → fp64_cpu inputs       │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑦ OpRunner.run(golden_func, params, to_device=False)      │   │ │
│   │  │     → golden_output (Tensor, CPU fp64)                      │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑧ OperatorMatcher.load_ai_operator("exp")                 │   │ │
│   │  │     → op_func (torch.ops.cann_bench.exp)                    │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑨ OpRunner.run_ai_op(op_func, params, TorchOpGuard=block) │   │ │
│   │  │     → ai_output (Tensor, NPU) + perf_result                │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑩ 同精度参考 (fp64_cpu 时单独执行 native_cpu Golden)       │   │ │
│   │  │     → native_output (Tensor, CPU fp32)                      │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑪ CheckerRegistry.get('relative_error') → checker         │   │ │
│   │  │     .check(ai_output, golden_output, native_output)         │   │ │
│   │  │     → AccuracyResult(passed, output_results, threshold)     │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑫ [可选] 二次验证 (_retry_with_fresh_inputs)              │   │ │
│   │  │     偏移种子 + 0.01微扰 → 两次都通过才记pass               │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑬ PerfMetricStrategy.parse(prof_files, perf_result)       │   │ │
│   │  │     → PerfResult(elapsed_us, op_times, metadata)            │   │ │
│   │  │                                                             │   │ │
│   │  │  ⑭ EvalCaseResult(success, accuracy, perf, failure_type)   │   │ │
│   │  │                                                             │   │ │
│   │  └─────────────────────────────────────────────────────────────┘   │ │
│   │                                                                     │ │
│   │  run_cases 渐进设备恢复:                                            │ │
│   │  healthy → (连续≥3失败+NPU不健康) → recovering → unrecoverable     │ │
│   │  每用例完成后 → 增量写入 incremental_output_path                    │ │
│   │                                                                     │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│   输出:                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  reports/                                                            │ │
│   │  ├── cann_eval_20260616_143000.json    # JSON 结果                  │ │
│   │  ├── cann_eval_20260616_143000.md      # Markdown 报告              │ │
│   │  ├── cann_eval_20260616_143000.html    # HTML 可视化报告            │ │
│   │  └ prof_data/              # Profiling 数据归档                     │ │
│   │      └ level1/exp/1/                                                │ │
│   │        └ kernel_details.csv                                        │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 多卡并行架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ProcessPoolCoordinator                                 │
│                         进程池协调器                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   主进程 (Coordinator):                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                                                                     │ │
│   │  ① 检测 NPU 卡: torch.npu.device_count()                           │ │
│   │  ② 构建 TaskUnit: (算子×用例组) 均分到各卡                         │ │
│   │  ③ ThreadPoolExecutor + as_completed 动态调度                       │ │
│   │  ④ 每个 TaskUnit 启动 eval-child 子进程                            │ │
│   │  ⑤ 监控进程状态，处理超时/OOM                                      │ │
│   │  ⑥ 收集结果，aggregate_by_operator 聚合                            │ │
│   │                                                                     │ │
│   │  OOM 保护:                                                          │ │
│   │  - 子进程 oom_score_adj=1000 (最优先被杀，保护主进程)               │ │
│   │  - OOM Kill 时恢复增量写入的部分结果                                │ │
│   │  - 剩余用例合成 oom_killed 失败                                    │ │
│   │                                                                     │ │
│   │  超时处理:                                                          │ │
│   │  - SIGTERM → 10s 宽限 → SIGKILL                                   │ │
│   │  - 恢复部分结果或合成 timeout 失败                                  │ │
│   │                                                                     │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   子进程分布 (profiler 开启时每卡 1 进程):                                │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                                                                     │ │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │ │
│   │  │   Card 0    │    │   Card 1    │    │   Card 2    │           │ │
│   │  │  NPU:0      │    │  NPU:1      │    │  NPU:2      │           │ │
│   │  └─────────────┘    └─────────────┘    └─────────────┘           │ │
│   │       │                  │                  │                      │ │
│   │       ▼                  ▼                  ▼                      │ │
│   │  ┌─────────┐        ┌─────────┐        ┌─────────┐              │ │
│   │  │eval-child│        │eval-child│        │eval-child│              │ │
│   │  │ 算子 A   │        │ 算子 C   │        │ 算子 E   │              │ │
│   │  │ 增量输出 │        │ 增量输出 │        │ 增量输出 │              │ │
│   │  └─────────┘        └─────────┘        └─────────┘              │ │
│   │                                                                     │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   eval-child 子命令:                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                                                                     │ │
│   │  python -m kernel_eval.cli eval-child                                │ │
│   │      --bench-name cann                                              │ │
│   │      --device-id 0                                                  │ │
│   │      --cases-file /tmp/cases_0.json (CaseSpec JSON 列表)            │ │
│   │      --output /tmp/cannbench_0.json  (结果输出路径)                  │ │
│   │      --warmup 3 --repeat 5                                          │ │
│   │      [--no-perf] [--profiler-level Level1]                          │ │
│   │      [--torch-op-guard-mode block] [--eval-seed 0]                  │ │
│   │                                                                     │ │
│   │  ① torch.npu.set_device(device_id)                                 │ │
│   │  ② Evaluator(incremental_output_path=output)                        │ │
│   │  ③ Evaluator.run_cases(cases)                                       │ │
│   │  ④ 每用例完成后增量写入 output 文件                                 │ │
│   │  ⑤ 所有用例完成后写入完整结果                                      │ │
│   │                                                                     │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           模块依赖图                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   cli.py                                                                   │
│       │                                                                    │
│       ├── config.py                                                        │
│       ├── _version.py                                                      │
│       ├── simulation.py                                                    │
│       ├── registry/bench_registry.py                                       │
│       ├── registry/loader_registry.py                                      │
│       ├── eval/evaluator.py                                                │
│       ├── eval/process_pool.py                                             │
│       ├── eval/results.py                                                  │
│       ├── eval/failure_synthesizer.py                                      │
│       ├── report/report_generator.py                                       │
│       ├── report/html_generator.py                                         │
│       ├── report/scoring.py                                                │
│       ├── report/setup_info.py                                             │
│       └── utils/path_resolver.py                                           │
│                                                                             │
│   simulation.py                                                             │
│       │                                                                    │
│       ├── config.py                                                        │
│       ├── eval/evaluator.py                                                │
│       ├── report/report_generator.py                                       │
│       ├── security/api_guard.py                                            │
│       └── registry/loader_registry.py                                      │
│                                                                             │
│   benches/cann.py                                                          │
│       │                                                                    │
│       ├── benches/cann_loader.py                                           │
│       ├── benches/cann_spec.py                                             │
│       ├── checkers/relative_error_checker.py                               │
│       ├── benches/cann_matcher.py                                          │
│       ├── benches/cann_scoring.py                                          │
│       ├── benches/cann_solution.py                                         │
│       ├── registry/loader_registry.py                                      │
│       ├── registry/golden_registry.py                                      │
│       ├── registry/matcher_registry.py                                     │
│       ├── registry/checker_registry.py                                     │
│       ├── registry/scoring_registry.py                                     │
│       ├── registry/case_spec_registry.py                                   │
│       ├── registry/perf_strategy_registry.py                               │
│       └── registry/bench_registry.py                                       │
│                                                                             │
│   benches/stanford.py                                                      │
│       │                                                                    │
│       ├── benches/stanford_loader.py                                       │
│       ├── benches/stanford_matcher.py                                      │
│       ├── benches/stanford_scoring.py                                      │
│       └── registry/* (同上)                                                │
│                                                                             │
│   eval/evaluator.py                                                        │
│       │                                                                    │
│       ├── config.py                                                        │
│       ├── _version.py (通过 benches 导入)                                  │
│       ├── registry/* (获取所有组件实例)                                    │
│       ├── data/data_generator.py                                           │
│       ├── data/package_manager.py                                          │
│       ├── eval/op_runner.py                                                │
│       ├── eval/accuracy_eval.py                                            │
│       ├── eval/perf_eval.py                                                │
│       ├── eval/results.py                                                  │
│       ├── eval/failure_synthesizer.py                                      │
│       ├── utils/device_manager.py                                          │
│       ├── utils/param_builder.py                                           │
│       ├── utils/tensor_utils.py                                            │
│       ├── utils/baseline_resolver.py                                       │
│       ├── utils/baseline_store.py                                          │
│       ├── security/api_guard.py                                            │
│       ├── security/torch_op_guard.py                                       │
│       ├── base/models.py                                                   │
│       └── benches (触发注册)                                              │
│                                                                             │
│   eval/op_runner.py                                                        │
│       │                                                                    │
│       ├── utils/device_manager.py                                          │
│       ├── eval/perf_eval.py                                                │
│       ├── eval/input_pool.py                                               │
│       ├── utils/tensor_utils.py                                            │
│       ├── security/torch_op_guard.py                                       │
│       └── security/type_checker.py                                         │
│                                                                             │
│   eval/accuracy_eval.py                                                    │
│       │                                                                    │
│       ├── base/checker.py                                                  │
│       ├── base/result.py                                                   │
│       ├── registry/checker_registry.py                                     │
│       └── eval/input_pool.py                                               │
│                                                                             │
│   eval/perf_eval.py                                                        │
│       │                                                                    │
│       ├── config.py                                                        │
│       ├── base/perf_strategy.py                                            │
│       ├── utils/device_manager.py                                          │
│       ├── eval/input_pool.py                                               │
│       ├── security/torch_op_guard.py                                       │
│       └── registry/perf_strategy_registry.py                               │
│                                                                             │
│   eval/process_pool.py                                                     │
│       │                                                                    │
│       ├── config.py                                                        │
│       ├── base/models.py                                                   │
│       ├── eval/results.py                                                  │
│       ├── eval/subprocess_utils.py                                         │
│       └── registry/loader_registry.py                                      │
│                                                                             │
│   checkers/relative_error_checker.py                                       │
│       │                                                                    │
│       ├── base/checker.py                                                  │
│       ├── base/result.py                                                   │
│       ├── utils/compare.py                                                 │
│       └── utils/thresholds.py                                              │
│                                                                             │
│   checkers/allclose_checker.py                                             │
│       │                                                                    │
│       ├── base/checker.py                                                  │
│       ├── base/result.py                                                   │
│       └── utils/compare.py                                                │
│                                                                             │
│   utils/compare.py                                                          │
│       │                                                                    │
│       ├── utils/thresholds.py                                              │
│                                                                             │
│   benches/cann_scoring.py                                                  │
│       │                                                                    │
│       ├── base/scoring.py                                                  │
│       ├── base/result.py                                                   │
│       ├── eval/results.py                                                  │
│       └── utils/baseline_resolver.py                                       │
│                                                                             │
│   report/report_generator.py                                               │
│       │                                                                    │
│       ├── eval/results.py                                                  │
│       ├── report/scoring.py                                                │
│       ├── report/summary_generator.py                                      │
│       ├── report/html_generator.py                                         │
│       ├── report/setup_info.py                                             │
│       ├── registry/scoring_registry.py                                     │
│       └── config.py                                                        │
│                                                                             │
│   security/torch_op_guard.py                                               │
│       │ (独立模块, 无外部依赖)                                             │
│                                                                             │
│   utils/baseline_store.py                                                 │
│       │                                                                    │
│       ├── utils/baseline_resolver.py                                       │
│                                                                             │
│   utils/device_manager.py                                                 │
│       │                                                                    │
│       ├── config.py (DeviceConfig)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 文件职责汇总

| 模块 | 文件 | 核心职责 |
|------|------|----------|
| **root** | __init__.py | 包入口，导出 Config + 版本号 |
| **root** | cli.py | 命令行入口，参数解析，子命令路由（eval/list/info/config/eval-child/simulate） |
| **root** | config.py | Config 数据类（含 eval_seed/torch_op_guard_mode/perf_metric_strategy_override/checker_name 等） |
| **root** | simulation.py | CPU 仿真评测模块（独立入口 / eval --device cpu 等价） |
| **root** | _version.py | 版本管理（从 VERSION + tasks/metadata/VERSION 动态读取） |
| **base** | enums.py | 通用枚举（DifficultyLevel/BackendType/SourceType/GoldenReference/EvaluationMode） |
| **base** | models.py | 数据模型基类（AttrSpec/InputSpec/OutputSpec/TaskSpec/CaseSpec/SolutionSpec + to_dict/from_dict） |
| **base** | result.py | 结果基类 + OutputResult 注册表（AccuracyResult/PerfResult/OutputResult ABC + register/get_output_result_cls） |
| **base** | loaders.py | 加载器基类（TaskLoader/CaseLoader/GoldenLoaderBase/OperatorDirMixin） |
| **base** | checker.py | CorrectnessChecker ABC（三输入模式 + _normalize_outputs/_ensure_cpu/_check_output_count） |
| **base** | matcher.py | OperatorMatcherBase ABC（load_ai_operator/find_operator_info/clear_cache/find_operator_info_by_snake） |
| **base** | scoring.py | ScoringScheme ABC + CaseScoreInfo（prepare_baseline/calculate_case_score/aggregate） |
| **base** | perf_strategy.py | PerfMetricStrategy ABC + 三策略实现 + ProfFileLocations + warmup/CSV/trace/msprof解析函数 |
| **benches** | cann.py | CANN 组件导出 + BenchConfig("cann") 注册 |
| **benches** | cann_loader.py | CANN 加载器（CannTaskLoader/CannCaseLoader/GoldenLoader） |
| **benches** | cann_spec.py | CANN 特化数据模型（CannTaskSpec/CannCaseSpec 等） |
| **benches** | cann_matcher.py | CANN Matcher（torch.ops.cann_bench / cann_bench 模块） |
| **benches** | cann_scoring.py | CANN 评分（CannScoringScheme/SimpleComparisonScheme/RecordingOnlyScheme + Eq.3/4/5） |
| **benches** | cann_solution.py | CANN 解决方案规格 |
| **benches** | stanford.py | Stanford 组件导出 + BenchConfig("stanford") 注册 |
| **benches** | stanford_loader.py | Stanford 加载器 |
| **benches** | stanford_matcher.py | Stanford Matcher |
| **benches** | stanford_scoring.py | Stanford 评分 |
| **checkers** | relative_error_checker.py | RelativeErrorChecker + RelativeErrorOutputResult（注册名: relative_error + 兼容名 cann_default） |
| **checkers** | allclose_checker.py | AllCloseChecker + AllCloseOutputResult（注册名: allclose） |
| **registry** | base.py | BaseRegistry 泛型基类 |
| **registry** | loader_registry.py | TaskLoader/CaseLoader 注册 + 便捷函数 |
| **registry** | golden_registry.py | GoldenLoader 注册 + 便捷函数 |
| **registry** | matcher_registry.py | OperatorMatcher 注册 + 便捷函数 |
| **registry** | checker_registry.py | Checker 注册 + register_correctness_checker 装饰器 |
| **registry** | scoring_registry.py | ScoringScheme 注册 + 便捷函数 |
| **registry** | case_spec_registry.py | CaseSpec 子类注册（反序列化用） |
| **registry** | perf_strategy_registry.py | PerfMetricStrategy 注册 + 便捷函数 |
| **registry** | bench_registry.py | BenchConfig 聚合配置 + BenchRegistry + get_bench_config/get_bench_components |
| **data** | data_generator.py | 确定性种子输入张量生成 |
| **data** | package_manager.py | 迭代隔离编译、安装、接口扫描、APIGuard snapshot 集成 |
| **eval** | evaluator.py | 综合调度（BenchConfig 依赖注入 + Golden精度策略 + 渐进设备恢复 + 增量输出 + 二次验证） |
| **eval** | op_runner.py | 算子执行（TorchOpGuard + 返回值检查 + profiler 同步采集） |
| **eval** | accuracy_eval.py | 精度评测（Checker 三输入模式：AI/Golden/同精度参考） |
| **eval** | perf_eval.py | 性能评测（Profiler 运行 + PerfMetricStrategy 委托解析） |
| **eval** | results.py | 结果数据结构（failure_type + 增量输出 JSON 格式 + from_dict 反序列化） |
| **eval** | input_pool.py | 输入池（clone 池 + 内存限制） |
| **eval** | process_pool.py | 多卡并行（TaskUnit 调度 + eval-child 子命令 + OOM 保护） |
| **eval** | failure_synthesizer.py | 失败结果合成（编译/安全/子进程/OOM 四种类型） |
| **eval** | subprocess_utils.py | OOM 保护 + CANN 环境变量 + 部分结果恢复 |
| **report** | report_generator.py | 报告生成（JSON+Markdown+Summary+HTML 四格式 + 语义前缀命名） |
| **report** | html_generator.py | HTML 渲染（4Section + KPI + 柱状图 + 认证印章 + 动态字段替换） |
| **report** | scoring.py | 评分计算（hardware-anchored Eq.3/4/5 + fallback baseline） |
| **report** | summary_generator.py | Summary 生成（几何平均加速比） |
| **report** | setup_info.py | 评测配置采集（metadata + environment） |
| **utils** | device_manager.py | 设备管理（渐进恢复：light→full→unrecoverable + health_check） |
| **utils** | param_builder.py | 参数构建（签名解析 + null 位处理） |
| **utils** | path_resolver.py | 路径解析 |
| **utils** | baseline_resolver.py | Baseline 解析（多硬件 dict + 平台别名映射） |
| **utils** | baseline_store.py | BaselineStore（metadata/<hw>.json 集中存储 + 向上查找 + 多平台 fallback） |
| **utils** | compare.py | 张量对比引擎（compare_tensors + MERE/MARE + 小值域 + 相消 + 整数匹配 + 多输出） |
| **utils** | thresholds.py | PRECISION_THRESHOLDS 精度阈值表（单一事实来源，Config 可覆盖） |
| **utils** | dtype_mapper.py | dtype 字符串映射 |
| **utils** | tensor_utils.py | tensors_to_cpu / tensors_to_fp64_cpu |
| **utils** | naming.py | camel_to_snake + snake_case_candidates（多候选模糊匹配） |
| **security** | api_guard.py | APIGuard（Timing API 快照+验证+恢复） |
| **security** | torch_op_guard.py | TorchOpGuard（TorchFunctionMode + 多路径规约 + pause 机制） |
| **security** | type_checker.py | 返回值类型检查 |

## 导入路径示例

```python
# 版本
from kernel_eval._version import FRAMEWORK_VERSION, TASKS_VERSION

# 基类
from kernel_eval.base import TaskSpec, CaseSpec, TaskLoader, CaseLoader
from kernel_eval.base import AccuracyResult, PerfResult, OutputResult
from kernel_eval.base import CorrectnessChecker, OperatorMatcherBase, ScoringScheme
from kernel_eval.base import DifficultyLevel, BackendType, GoldenReference

# CANN 特化组件
from kernel_eval.benches.cann import CannTaskLoader, CannCaseLoader, RelativeErrorChecker
from kernel_eval.benches.cann import OperatorMatcher, CannScoringScheme

# 或从 benches 直接导入（重新导出）
from kernel_eval.benches import CannTaskLoader, RelativeErrorChecker
from kernel_eval.benches import StanfordTaskLoader, StanfordMatcher

# Checker（通用，不绑定评测集）
from kernel_eval.checkers import RelativeErrorChecker, AllCloseChecker
from kernel_eval.checkers import RelativeErrorOutputResult, AllCloseOutputResult

# Registry
from kernel_eval.registry import LoaderRegistry, BenchRegistry, BenchConfig
from kernel_eval.registry import get_bench_config, get_bench_components

# 性能策略
from kernel_eval.base.perf_strategy import KernelDetailsStrategy, MsProfSummaryStrategy

# 评测组件
from kernel_eval.eval import Evaluator, OpRunner, AccuracyEvaluator, PerfEvaluator
from kernel_eval.eval import EvalCaseResult, EvalOperatorResult, EvalSessionResult
from kernel_eval.eval.process_pool import ProcessPoolCoordinator
from kernel_eval.eval import FailureSynthesizer

# 安全
from kernel_eval.security import APIGuard
from kernel_eval.security.torch_op_guard import TorchOpGuard

# 数据工具
from kernel_eval.data import DataGenerator, PackageManager

# 工具
from kernel_eval.utils import DeviceManager, compare_tensors
from kernel_eval.utils.baseline_store import BaselineStore
from kernel_eval.utils import tensors_to_fp64_cpu, camel_to_snake

# 报告
from kernel_eval.report import ReportGenerator
from kernel_eval.report.html_generator import render_html_report
from kernel_eval.report.setup_info import collect_setup_info

# 配置
from kernel_eval.config import Config, get_config, set_config
```
