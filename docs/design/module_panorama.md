# Module Panorama - NPU Operator Benchmark Framework

> **范围**：文件数、类数及依赖统计为设计时快照，不代表当前仓库的实时清单。现行契约与运行方式从[文档索引](../README.md)进入，模块签名以当前 `src/` 实现为准。

> Auto-generated: 2026-06-03 | Sources: `src/kernel_eval/`, `src/auto_pipeline/`, `scripts/`, `tests/`

---

## 1. Architecture Overview

```
+--------------------------------------------------------------------+
|                        CLI / YAML Config                           |
+-------------------+--------------------------------+---------------+
                    |                                |
        +-----------v-----------+        +-----------v-----------+
        |    auto_pipeline      |        |     kernel_eval       |
        |  (Code Generation)    |        |   (Evaluation)        |
        +-----------+-----------+        +-----------+-----------+
                    |                                |
    +---------------+---------------+   +------------+------------+
    | Generator     | Converter     |   | Evaluator  | Report    |
    | (akg/pypto/   | (to_cann/    |   | (accuracy/ | (html/    |
    |  opencode)    |  to_stanford) |   |  perf/sub) |  scoring) |
    +-------+-------+-------+------+   +-----+------+-----+----+
            |               |               |            |
            +-------+-------+-------+-------+            |
                    |               |                    |
              subprocess: python -m kernel_eval.cli eval |
                    |                                    |
            +-------v-------+                            |
            |   Registry    |<---------------------------+
            |  (7 tables)   |
            +-------+-------+
                    |
        +-----------+-----------+
        | benches/cann.py       |
        | benches/stanford.py   |
        +-----------------------+
```

**Two-domain design**: `auto_pipeline` (code generation) and `kernel_eval` (evaluation) are fully decoupled -- they interact only through **subprocess CLI calls** and **directory conventions**, sharing no in-process state.

---

## 2. Registered Modules

### 2.1 kernel_eval Registries (7 tables)

| Registry | Key Type | Registered Items |
|----------|----------|-----------------|
| `LoaderRegistry` | `eval_system` | `cann` (CannTaskLoader/CannCaseLoader), `stanford` (StanfordTaskLoader/StanfordCaseLoader) |
| `GoldenLoaderRegistry` | `eval_system` | `cann` (GoldenLoader), `stanford` (StanfordGoldenLoader) |
| `OperatorMatcherRegistry` | `eval_system` | `cann` (OperatorMatcher), `stanford` (StanfordMatcher) |
| `CheckerRegistry` | `name` | `relative_error` (RelativeErrorChecker), `cann_default` (alias), `allclose` (AllCloseChecker) |
| `ScoringSchemeRegistry` | `name` | `cann` (CannScoringScheme), `simple_comparison`, `recording_only`, `stanford` (StanfordScoringScheme) |
| `CaseSpecRegistry` | `name` | `cann` (CannCaseSpec), `stanford` (CaseSpec base) |
| `BenchRegistry` | `bench_name` | `cann`, `stanford` |

### 2.2 auto_pipeline Registries (3 tables)

| Registry | Key | Registered Items |
|----------|-----|-----------------|
| **Runner** | `name` | `opencode` (OpenCodeAgent) |
| **Generator** | `type` | `akg-agent` (AkgAgent), `pypto` (PyptoOrchestratorAgent) |
| **Converter** | `(source, target)` | `pypto->cann`, `pypto->stanford`, `akg-agent->cann`, `akg-agent->stanford` |
| **PromptBuilder** | `bench_name` | `cann` (CannPromptBuilder), `stanford` (StanfordPromptBuilder) |

---

## 3. BenchConfig: Component Binding

BenchConfig is the **composition core** -- it binds 6 component names into a coherent evaluation stack:

### CANN Bench (`cann`)

| Component | Registry | Registered Name | Implementation |
|-----------|----------|-----------------|----------------|
| TaskLoader | LoaderRegistry | `cann` | `CannTaskLoader` |
| CaseLoader | LoaderRegistry | `cann` | `CannCaseLoader` |
| GoldenLoader | GoldenLoaderRegistry | `cann` | `GoldenLoader` |
| OperatorMatcher | OperatorMatcherRegistry | `cann` | `OperatorMatcher` |
| Checker | CheckerRegistry | `relative_error` | `RelativeErrorChecker` |
| ScoringScheme | ScoringSchemeRegistry | `cann` | `CannScoringScheme` |
| CaseSpec | CaseSpecRegistry | `cann` | `CannCaseSpec` |
| Golden Precision | - | `fp64_cpu` | float64 + CPU |
| Tasks Root | - | `tasks` | - |

### Stanford Bench (`stanford`)

| Component | Registry | Registered Name | Implementation |
|-----------|----------|-----------------|----------------|
| TaskLoader | LoaderRegistry | `stanford` | `StanfordTaskLoader` |
| CaseLoader | LoaderRegistry | `stanford` | `StanfordCaseLoader` |
| GoldenLoader | GoldenLoaderRegistry | `stanford` | `StanfordGoldenLoader` |
| OperatorMatcher | OperatorMatcherRegistry | `stanford` | `StanfordMatcher` |
| Checker | CheckerRegistry | `allclose` | `AllCloseChecker` |
| ScoringScheme | ScoringSchemeRegistry | `stanford` | `StanfordScoringScheme` |
| CaseSpec | CaseSpecRegistry | `stanford` | `CaseSpec` (base) |
| Golden Precision | - | `native_npu` | native dtype + NPU |
| Tasks Root | - | `thirdparty/KernelBench/...` | - |

---

## 4. Combination Matrix

### 4.1 End-to-End Pipeline Combinations

| Generator | Benchmark | PromptBuilder | Converter | conversion_runner | Status |
|-----------|-----------|---------------|-----------|-------------------|--------|
| `akg-agent` | cann | CannPromptBuilder | AkgToCannConverter | None | Implemented |
| `akg-agent` | stanford | StanfordPromptBuilder | AkgToStanfordConverter | None | Implemented |
| `pypto` | cann | CannPromptBuilder | PyptoToCannConverter | OpenCodeAgent | Implemented |
| `pypto` | stanford | StanfordPromptBuilder | PyptoToStanfordConverter | OpenCodeAgent | Implemented |
| `opencode` | cann | CannPromptBuilder | - | - | Runner only, no Generator |
| `opencode` | stanford | StanfordPromptBuilder | - | - | Runner only, no Generator |

### 4.2 kernel_eval Component Cross-Bench Reuse

| Component | CANN Bench | Stanford Bench | Cross-reusable? |
|-----------|:----------:|:--------------:|:---------------:|
| RelativeErrorChecker | Default | - | Yes |
| AllCloseChecker | - | Default | Yes |
| CannScoringScheme | Default | - | Yes (interface-compatible) |
| StanfordScoringScheme | - | Default | Yes (interface-compatible) |
| SimpleComparisonScheme | Available | - | Yes |
| RecordingOnlyScheme | Available | - | Yes |
| CannTaskLoader | Bound | - | **No** (dir format differs) |
| StanfordTaskLoader | - | Bound | **No** (dir format differs) |
| OperatorMatcher (cann) | Bound | - | **No** (whl/pkg format) |
| StanfordMatcher | - | Bound | **No** (ai_op.py format) |

**Summary**: Checker and Scoring are **freely composable** across benches. Loader and Matcher are **strongly bound** to their bench's directory structure and file format.

### 4.3 Generator x Converter (Cartesian Product)

| | **to_cann** | **to_stanford** |
|---|:-----------:|:---------------:|
| **akg-agent** | Implemented | Implemented |
| **pypto** | Implemented | Implemented |

All 4 combinations are **fully implemented**.

---

## 5. Invalid Combinations

| Combination | Reason |
|-------------|--------|
| `opencode` as Generator | Only registered as Runner; `create_generator("opencode")` raises ValueError |
| Generator + unregistered bench | `create_converter` KeyError on `_CONVERTER_FACTORIES` |
| Stanford submission -> CANN bench | CANN requires `build.sh + cann_bench/`; Stanford only produces `ai_op.py` |
| CANN submission -> Stanford bench | Stanford expects `ai_op.py`; CANN produces whl packages |
| Mix Loader from bench A + Matcher from bench B | Directory format mismatch |
| GoldenLoader cross-bench | Each bench has unique golden file layout |

---

## 6. Potential Combinations (Not Yet Implemented)

| Opportunity | Description | Effort |
|-------------|-------------|--------|
| `opencode` as Generator | Wrap via `PromptGenerator` (already in core.py), add `(opencode, cann)` and `(opencode, stanford)` converter registrations | Medium |
| New bench domain | BenchRegistry supports dynamic registration; add Loader/Checker/Scoring + Converter | Medium |
| Cross-bench Checker swap | Use `allclose` for CANN or `relative_error` for Stanford -- just change BenchConfig.checker field | Trivial |
| Cross-bench Scoring swap | Use `cann` scoring for Stanford -- change BenchConfig.scoring_scheme field | Trivial |

---

## 7. Dependency Graph

### 7.1 kernel_eval Eval Layer

```
Evaluator (orchestrator)
 +-- OpRunner
 |    +-- PerfEvaluator
 |    |    +-- InputPool (anti-cache-attack)
 |    |    +-- DeviceManager
 |    +-- DeviceManager
 +-- AccuracyEvaluator
 |    +-- CheckerRegistry -> RelativeErrorChecker / AllCloseChecker
 |    +-- compare.py (MERE/MARE engine)
 |    +-- thresholds.py (5 precision tables)
 |    +-- type_checker.py (FakeTensor guard)
 +-- SubprocessRunner
 |    +-- FailureSynthesizer -> CaseLoader
 +-- ProcessPoolCoordinator (multi-card parallel)
 +-- PackageManager -> compile/install whl -> APIGuard
 +-- DataGenerator -> dtype_mapper
 +-- ParamBuilder
```

### 7.2 auto_pipeline Layer

```
BenchmarkPipeline (orchestrator)
 +-- CannBenchClient (case loading)
 |    +-- build_case_material -> PromptBuilder (cann/stanford)
 +-- Generator (Protocol)
 |    +-- AkgAgent (in-process AKG SDK)
 |    +-- PyptoOrchestratorAgent (7-stage orchestrator)
 |         +-- OpenCodeAgent (run_opencode)
 +-- Converter (Protocol)
 |    +-- BaseConverter -> optional Runner (for LLM conversion)
 |    +-- AkgToCannConverter / AkgToStanfordConverter
 |    +-- PyptoToCannConverter / PyptoToStanfordConverter
 +-- Submission -> subprocess: python -m kernel_eval.cli eval
```

### 7.3 Cross-Domain Data Flow

```
YAML Config
  |
  v
auto_pipeline CLI (cli.py)
  |-- _parse_config: agent.type + benchmark.name
  |-- create_generator(agent.type) -> Generator
  |-- create_converter(agent.type, benchmark.name) -> Converter
  |
  v
BenchmarkPipeline.run_case:
  1. CannBenchClient.load_case -> CannBenchCase
  2. PromptBuilder.build_case_material -> CaseMaterial
  3. Generator.generate(GeneratorInput) -> Artifact
  4. Converter.convert(bench_name, case, artifact) -> Submission(source_dir)
  5. CannBenchClient.eval_submission
       |
       |  subprocess boundary (directory + JSON only)
       v
     kernel_eval CLI (cli.py eval --bench-name <name> --source-dir <dir>)
       |-- Evaluator(bench_name) -> BenchConfig -> full component stack
       |-- evaluate_case x N -> EvalSessionResult -> JSON report
```

---

## 8. Supporting Modules

### 8.1 Independence Ratings

| Module | Independence | Notes |
|--------|:-----------:|-------|
| `utils/thresholds` | 5/5 | Pure data tables + query functions, zero deps |
| `utils/naming` | 5/5 | Pure string transforms, zero deps |
| `utils/dtype_mapper` | 4/5 | Only depends on torch |
| `utils/tensor_utils` | 4/5 | Only depends on torch |
| `utils/baseline_resolver` | 4/5 | Only stdlib + logging |
| `utils/param_builder` | 4/5 | Only depends on inspect |
| `utils/compare` | 3/5 | Depends on thresholds, self-contained |
| `utils/path_resolver` | 3/5 | Depends on base.loaders |
| `security/api_guard` | 4/5 | Only depends on torch |
| `security/torch_op_guard` | 4/5 | Only depends on torch |
| `security/type_checker` | 4/5 | Only depends on torch |
| `data/data_generator` | 3/5 | Depends on dtype_mapper |
| `data/package_manager` | 2/5 | Depends on config + security + registry |
| `report/*` | 2/5 | Depends on eval.results data structures + config |
| `utils/device_manager` | 2/5 | Depends on config singleton |
| `config.py` | 3/5 | Depends on thresholds (lazy import) |

### 8.2 Cross-Domain Reuse Potential

| Priority | Module | Reuse Method | Use Case |
|:--------:|--------|-------------|----------|
| High | `thresholds` | Direct import | Pipeline precision pre-check / result parsing |
| High | `naming` | Direct import | Operator name matching & directory mapping |
| High | `baseline_resolver` | Direct import | Baseline query & speedup calculation |
| Medium | `security/*` | Direct import | If pipeline installs whl in-process, needs APIGuard |
| Medium | `data_generator` | Direct import | Pipeline self-test / golden verification |
| Medium | `compare` | Direct import | In-pipeline precision comparison |
| Low | `device_manager` | Via CLI args | Current subprocess model suffices |
| Low | `report/*` | Via CLI args | Report generated by eval subprocess |
| Low | `package_manager` | Via CLI args | Compile/install handled by eval subprocess |

**Key insight**: Pure-function/pure-data modules (thresholds, naming, baseline_resolver, dtype_mapper) can be directly imported by auto_pipeline with zero coupling risk. Stateful modules (device_manager, config) and workflow modules (package_manager, report) should remain isolated behind the CLI subprocess boundary.

---

## 9. Configuration Sharing

The two domains maintain **separate configurations** unified through **CLI argument contracts**:

| Domain | Config Source | Key Fields |
|--------|--------------|------------|
| `auto_pipeline` | YAML config (`config/` dir) + `core.py` dataclass | `agent.type`, `benchmark.name`, `repo_root`, `device_id` |
| `kernel_eval` | `Config` global singleton (`config.py`) | `tasks_root`, `device_type`, `warmup`, `repeat`, `checker_name`, `precision_thresholds` |
| **Bridge** | CLI args: `--bench-name`, `--source-dir`, `--device`, `--device-id` | `_create_config_from_args()` translates args -> Config |

The `bench_name` string serves as the **configuration anchor** for both domains:
- auto_pipeline: selects Converter, PromptBuilder, submission format validation
- kernel_eval: `get_bench_config(bench_name)` pulls the full component stack

---

## 10. Module Count Summary

| Domain | Modules | Classes | Registries | Registered Items |
|--------|:-------:|:-------:|:----------:|:----------------:|
| kernel_eval/base | 7 files | 16 classes | 0 | - |
| kernel_eval/registry | 8 files | 8 classes | 7 | 18 items |
| kernel_eval/benches | 10 files | 20 classes | - | (self-registering) |
| kernel_eval/eval | 9 files | 12 classes | - | - |
| kernel_eval/checkers | 2 files | 4 classes | - | (via CheckerRegistry) |
| kernel_eval/utils | 9 files | 15+ functions | - | - |
| kernel_eval/security | 3 files | 4 classes | - | - |
| kernel_eval/data | 2 files | 3 classes | - | - |
| kernel_eval/report | 5 files | 6 classes | - | - |
| auto_pipeline/generator | 7 files | 4 classes | 2 | 3 items |
| auto_pipeline/converter | 7 files | 6 classes | 1 | 4 items |
| auto_pipeline/prompt | 3 files | 3 classes | 1 | 2 items |
| auto_pipeline/core | 1 file | 10 classes | - | - |
| **Total** | **~73 files** | **~111 classes** | **11 registries** | **27 registered items** |
