# 版本变更记录

## Season2 题集修订 tasks-v2.1.0

- 更新部分公开用例配置、参考实现及相关接口说明，保持 60 题、每题 20 条公开用例。
- 更新受影响用例的性能参考记录，按平台应用最小耗时下限并保留待设备复验标记。
- 将 DynamicMxQuant 在 910B2 标记为不支持，更新平台排除名单。
- 明确 Unique 的可选返回值及 LSTM 推理模式和投影维度。
- SparseFlashAttention 支持范围表仅说明算子约束，公开用例配置见同目录 `cases.yaml` 和 `cases.csv`。
- 题集版本升级至 `2.1.0`，同步公开文件校验清单；框架版本与评分公式保持不变。
- 修订前后的用例及性能参考记录存在差异，评测结果应按题集版本区分，比较成绩时使用同一版本重新评测。

## Season2 题集切换 tasks-v2.0.0 待性能验收

- 主线 `tasks/` 切换为 60 个算子的公开题包，每题 20 个用例，覆盖 L1 到 L5。
- 原 53 题及原始 metadata 完整归档至 `archives/season1/tasks/`。
- 公开 910B2、910C 及用户提供的 950PR/950DT Season2 metadata，禁止跨平台回退；950 两平台各有 1200 条公开记录，使用提交文件的封顶评分基线，原始层不对外同步。
- 对现有支持平台记录中的 404 条缺失基线按 `3 × t_hw_us` 补齐，表达理论性能的三分之一，逐条标注为估算来源；不修改已有正数基线、理论值或评分公式。
- 补齐 MoE `value_topk` 和 TopK `distinct` 索引契约，统一公开入口和等级描述；KV Cache 使用显式返回输出的公开契约。
- 本次不改变框架版本或评分公式，不包含隐藏用例；平台支持和性能验收状态见 `docs/guide/season2_migration.md`。

## V1.0.0 (2026-07-18) · git tag: v1.0.0 / tasks-v1.0.0

**里程碑：首个正式版 —— 反作弊 V3 加固 + oracle/bench golden 拆分 + 性能阶段精度复检 + 文档全面对齐**

### 框架变更

- **反作弊 V3 加固**：新增 `cann_bench_utils` C++ 扩展包（自定义 `cache_clean` / `warmup` AscendC 算子）+ V3 脚本（`disable_builtin_kernels_v3.sh` / `restore_builtin_kernels_v3.sh`），禁用整个内置 kernel 树并由自定义 warmup 算子替代，杜绝缓存命中/硬编码结果作弊；`run_evaluation.sh` 集成 V3 启用流程；`perf_strategy` 增加 warmup 阶段配置（替换旧 V1/V2 脚本）
- **oracle/bench golden 拆分**：新增可选 loader 钩子 `get_oracle_function` / `get_bench_function`（向后兼容，缺失时回退 golden 本身）；`evaluator.py` 接线，保留 fp64 数学真值（oracle）与同精度参考（bench）双轨，修复 `|bench−oracle|=0` 退化导致的小值域/相消误杀
- **性能阶段精度复检**：`staged_eval.py` 将 performance 阶段精度翻车判为精度错误（`failure_type=precision_mismatch`），扣精度分、无性能分、不扣编译分；新增 `EvalCaseResult.perf_recheck` 诊断字段写入 `results.json`，标记疑似非确定算子
- **精度判定对齐文档**：`compare.py` 小值域/相消兜底统一为 `npu / max(cpu,1) ≤ 2` 比值判定（对齐 `benchmark_spec.md`，移除 `cpu==0→npu==0` 特例）
- **输入确定性**：`DataGenerator` 整型分支直接透传 generator 给 `torch.randint`；5 个算子 `get_input` 钩子补固定种子，确保精度/性能阶段输入一致、结果可复现
- **技术报告更新**：`docs/technical-report.pdf` 重写反作弊章节，采用"提交规范 / 运行时检查 / 测量完整性 / NPU 执行归因"分层防御模型
- **依赖对齐**：`requirements.txt` 对齐到 `pyproject.toml`（torch/torch_npu 钉 `==2.10.0`，补齐缺失依赖，修正版本冲突）
- **文档全面对齐**：修正评测使用文档（删除已移除的 CLI 参数/子命令、报告文件名对齐实现）；baseline 外置说明（移除 cases.yaml/csv 内嵌 baseline 字段）；contributing 补齐 dtype SoC 红线、proto 输出字段（compare/index_gather）、golden 承重规则；清理文档索引与失效引用；对齐 baseline_collection_design 与 collect_baseline.py 实现

### 评测集变更 (tasks-v1.0.0)

- **golden 拆分**：`weight_quant_batch_matmul` 收敛为 plain(=bench) + `_oracle`（反量化改 bf16/fp16 + fp32 累加，删冗余 `_bench`）；`grouped_matmul` 新增 dtype-agnostic `_oracle`（910B2 实测 14/20 → 17/20，修相消退化）
- **baseline 元数据**：`tasks/metadata/910b2.json` 更新；新增 metadata 性能基线校验（防止 `t_hw_us` 高于可比较的 `baseline_perf_us`）

### bench_lab（孵化区，不纳入版本管理）

- **新增算子**：tilelang_ascend_bench 新增 `flash_attention`（20 case + 910b2 baseline）
- **baseline 补全**：填充 `bench_lab/kernel_bench` 缺失的 `t_hw_us`（910b2 120 条 + 950pr 180 条，共 300 条，floor 1μs）
- **一致性修复**：tilelang flash_attention/gemm 的 cases.yaml/csv 一致性（note 后缀 + dtype JSON 引号）；修复 tilelang 与 910b2 metadata 无效 JSON/尾随逗号

### 示例

- **pypto example**：新增 swi_glu pypto 生成示例（c1–c8 共 8 个候选实现）；修复性能评测 `trace_view` 参数传递；pypto 生成任务增加精度要求传递

### 测试

- 新增 `test_process_pool`（+467 行）、`test_perf_eval`（+217 行）、`test_device_visibility`、`test_get_output`、`test_retry_mechanism`、`test_weight_quant_oracle_bench`、`test_reference_func_selection`、`test_golden_loader` 等

## V0.4.0 (2026-06-25) · git tag: v0.4.0 / tasks-v0.4.0

**里程碑：多硬件多卡评测 + StanfordBench 集成 + 反作弊加固 + HAP 性能口径完善 + 报告系统重构**

### 框架变更

- **多硬件支持**：新增 Ascend 950 / 910_93 全栈支持；SOC 自动检测（修复 910B2 被误识别为 910_93）
- **多卡评测**：支持 MC2 分布式算子与多卡并行评测；修复 eval-child 设备可见性串扰、reports_dir 未继承、子进程失败生成空壳报告等问题
- **StanfordBench 集成**：迁移至 `bench_lab/stanford_bench`；按 dtype 注入精度容差对齐原生 KernelBench（fp32=1e-4，fp16/bf16=1e-2），取代原先不分 dtype 的固定 1e-2
- **反作弊加固**：dispatch 层守卫拦截 CPU offload 与 C++ 内置算子绕过；补全内置算子禁用清单（修复 TopK 等可绕过漏洞）；TorchOpGuard 误判自定义 `torch.ops` 修复；golden ST 以 warn 模式运行避免误杀参考实现
- **HAP 性能口径**：性能评分正式命名 HAP（hardware-anchored performance）；性能口径计入自定义/direct-launch kernel（以 `kernel_details.csv` 为权威源）；弃用墙钟计时；新增 MsProfSummaryStrategy 及 baseline 采集框架；集成 PyPTO orchestrator 与多 kernel 开发能力
- **报告系统重构**：`description.html` + `setup_info` + HTML 输出；支持 JSON 评测报告；默认命名自动加入语义前缀
- **Checker 解耦**：checker 与 benchmark suites 解耦
- **评测流程**：编译失败整批计 0 分（支持并行化评测）；`cann_scoring` 增加 NaN guard；run package 安装失败 fail fast 并移除 `libopapi.so` fallback；精度指标显示/判定逻辑/错误日志修复
- **Docker**：新增 cann-bench 参考执行镜像（参数化 CANN 版本与硬件型号，`PYPI_INDEX_URL` build-arg 可覆盖默认 PyPI 源）
- **依赖升级**：torch / torch_npu 升至 2.10.0（torchvision 0.25.0）

### 评测集变更 (tasks-v0.4.0)

- **新增算子评测集**：vector 算子、dynamic_mx_quant、omni-ops、moe_gating_top_k_backward、cv_agent、MC2 分布式算子
- **baseline 数据**：新增 950PR baseline 与 `t_hw_us`（hardware-anchored 性能下界）；910b2 baseline 统一为 10×t_hw（封顶实测 + 填充缺失 + 10× 代理）；亚微秒 t_hw 下限设为 1us
- **算子修复**：LSTM 支持 LSTMP 投影（`weight_hr` 入参）；moe_finalize_routing 向量化核心循环 + drop_pad 语义修正；apply_adam_w epsilon 移回 sqrt 外对齐 `torch.optim.AdamW`；quant_matmul 用 int64 累加避免 fp32 尾数溢出；roi_align 按 ROI 分块 auto-sampling 避免 OOM；gather int32 index 在 CPU cast 为 int64；conv_3d_backprop_filter 等 7 case 补齐 t_hw_us
- **算子原型一致性**：对齐 desc.md / proto.yaml / golden.py 原型；proto.yaml 类型一致性（drop_pad_mode int 默认、dilation_2d float 标注）

## V0.3.0 (2026-05-19)

**架构重构：基类层分离 + benches 扁平化 + Registry 完善**

- **Base 层创建**：新增 `base/` 目录，统一存放抽象基类
  - `models.py`: TaskSpec, CaseSpec, InputSpec, OutputSpec, SolutionSpec（合并原 models/ 目录）
  - `result.py`: AccuracyResult, PerfResult, OutputResult（合并原 models/result.py）
  - `loaders.py`: TaskLoader, CaseLoader, GoldenLoaderBase, OperatorDirMixin
  - `checker.py`: CorrectnessChecker 抽象基类
  - `matcher.py`: OperatorMatcherBase 抽象基类
  - `scoring.py`: ScoringScheme 抽象基类

- **Benches 层扁平化**：`benches/cann/` 子目录删除，文件平铺到 `benches/`
  - `cann_loader.py`: CannTaskLoader, CannCaseLoader, GoldenLoader
  - `cann_spec.py`: CANN 特化数据模型
  - `cann_checker.py`: RelativeErrorChecker, RelativeErrorOutputResult（CannDefaultChecker/CannOutputResult 为兼容别名）
  - `cann_matcher.py`: OperatorMatcher（重命名自 operator_matcher.py）
  - `cann_scoring.py`: CannScoringScheme, ScoringCalculator 等
  - `cann_solution.py`: CannSolutionSpec
  - `cann.py`: 导出所有组件 + Registry 注册（核心入口）

- **Registry 层完善**：新增 `registry/` 目录，统一管理注册机制
  - `loader_registry.py`: TaskLoader/CaseLoader 注册
  - `golden_registry.py`: GoldenLoader 注册
  - `matcher_registry.py`: OperatorMatcher 注册
  - `checker_registry.py`: Checker 注册
  - `scoring_registry.py`: ScoringScheme 注册
  - `bench_registry.py`: BenchConfig 聚合配置

- **BREAKING CHANGE — Checker 重命名**：
  - 注册名 `cann_default` → `relative_error`（保留 `cann_default` 注册别名，`get_correctness_checker("cann_default")` 仍可用）
  - 类名 `CannDefaultChecker` → `RelativeErrorChecker`，`CannOutputResult` → `RelativeErrorOutputResult`（保留 re-export 兼容别名）
  - 类名 `AllcloseChecker` → `AllCloseChecker`，`AllcloseOutputResult` → `AllCloseOutputResult`
  - 模块路径 `kernel_eval.eval.allclose_checker` → `kernel_eval.checkers.allclose_checker`
  - **迁移方式**：外部配置/脚本中将 `checker: cann_default` 改为 `checker: relative_error`；旧名兼容别名将在后续版本移除

- **向后兼容别名删除**：删除所有 `models/` 目录下的兼容导入
  - 用户应使用 `from kernel_eval.base import TaskSpec` 或 `from kernel_eval.benches.cann import CannTaskSpec`

- **导入路径更新**：
  - 基类：`from kernel_eval.base import TaskSpec, CaseSpec, TaskLoader`
  - CANN 特化：`from kernel_eval.benches.cann import CannTaskLoader, RelativeErrorChecker`
  - Registry：`from kernel_eval.registry import LoaderRegistry, BenchRegistry`
  - 通用 Checker：`from kernel_eval.checkers import AllCloseChecker`

- **文档更新**：
  - `docs/guide/custom_benchmark_integration.md`: 更新架构图、导入路径、接入示例
  - `docs/design/kernel_eval_architecture.md`: 更新目录结构、模块职责、数据流图

## V0.2.0 (2026-05-07)

**评分体系切换为 hardware-anchored 公式 (对齐 bench.tex)**

- 评分公式改版：单用例性能得分由原始 `SpeedUp = baseline / candidate` 改为 `score_i = (T_baseline − T_HW) / ((T_cand − T_HW) + (T_baseline − T_HW))`（bench.tex Eq. 3）
- 单算子综合评分改版：`EachOperatorScore = [w_c·δ_pass + Σ δ_acc,i (w_f + w_p·score_i) / N] · 100`，归一化到 [0, 100]（bench.tex Eq. 4）
- 权重调整：`(w_c, w_f, w_p) = (0.2, 0.3, 0.5)`（原 `(2, 3, 5)`）
- 新增字段 `t_hw_us`：每个用例新增硬件下界 `T_HW`，写入 cases.yaml 与 cases.csv，加载链路 (case_loader / EvalCaseResult / report_generator) 同步打通
- 工具更新：`scripts/utils/yaml_to_csv.py` 加入新字段；`src/kernel_eval/report/scoring.py` 与 `summary_generator.py` 重写为新公式
- 几何平均加速比保留为诊断字段

## V0.1.1 (2026-04-29)

**文档重组与内容完善**

- 文档目录重组：建立 spec/、design/、guide/ 分层结构
- 文档职责分离：benchmark_spec.md 定义规范，evaluator_design.md 定义实现
- 精度标准完善：新增小值域通过标准（ErrorCount 计算公式）
- 性能评测完善：更新 Trace 解析逻辑（`cat="dequeue"` 事件）、Warmup Kernel 过滤机制、InputPool 防缓存攻击
- 设备同步优化：目标设备同步而非默认设备
- 安全防护：Timing API 防护、返回值类型检查、二次验证机制
- Golden 计算：CPU fp64 Golden 计算流程
- 多硬件支持：多硬件 baseline 解析
- 报告生成：几何平均加速比计算、JSON/Markdown/Summary 多格式输出

---

## V0.1.0 (2026-04-25)

**初版发布**

- 建立基础评测框架
- 定义 L1-L4 四级难度体系
- 完成 55 个算子规格定义和用例设计
- 建立编译正确性、功能正确性、性能优化性三大评测维度
- 定义 MERE/MARE 精度标准和阈值表
- 基础评测架构：编译、功能、性能三维度评测
- JSON + Markdown 报告生成
- Profiler kernel-only 测量
- 目录结构：src/kernel_eval 评测工程
