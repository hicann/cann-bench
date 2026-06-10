# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CANN Bench — Ascend C Operator Evaluation Framework

## Project Overview

CANN Bench evaluates AI-generated Ascend C operator code quality across three dimensions: compilation correctness (w_c=0.2), functional accuracy (w_f=0.3), and performance optimization (w_p=0.5). Operators are organized into 4 difficulty levels (L1-L4), covering element-wise to attention-level complexity.

Performance is scored via HAP (Hardware-Anchored Performance): `HAP_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))`, anchored against hardware theoretical upper bound.

Version: 0.3.0 (see `VERSION` and `tasks/metadata/VERSION`).

## Build & Run

**No pip-install needed** — the project runs via `PYTHONPATH=src`:
```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

Or with uv (preferred for dependency locking):
```bash
uv sync          # lock deps for linux-aarch64 only (torch_npu has no other platform wheels)
```

**Evaluation** (the primary workflow):
```bash
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops         # full pipeline: scan → compile → install → eval
./scripts/run_evaluation.sh --source-dir ... --task-dir examples/tasks --no-perf  # fixture test, no perf
PYTHONPATH=src python -m kernel_eval.cli eval --source-dir ...   # direct CLI
```

**Testing**:
```bash
./scripts/run_test.sh              # all (ut + e2e)
./scripts/run_test.sh ut           # unit tests only
./scripts/run_test.sh ut -k "scoring"  # filter by keyword
PYTHONPATH=src pytest tests/ut/    # direct pytest
```

**Linting / formatting**:
```bash
black --line-length=120 src/ tests/   # formatter (pyproject.toml config)
isort --profile=black --line-length=120 src/ tests/  # import sorter
flake8 src/ tests/                     # linter
```

CLI entry point: `python -m kernel_eval.cli` (prog name: `kernel-bench`). Subcommands: `eval`, `list`, `info`, `config`, `eval-process`.

## Architecture

### Two-Layer Structure

```
src/kernel_eval/        — Evaluation framework (data loading, eval logic, scoring, reporting, security)
src/auto_pipeline/      — AI agent orchestration (prompt generation, code generation, submission conversion)
```

`kernel_eval` is the core; `auto_pipeline` orchestrates AI agents to produce operator code that `kernel_eval` then evaluates.

### kernel_eval Module Map

| Submodule | Role |
|-----------|------|
| `base/` | Abstract base classes (`TaskLoader`, `CaseLoader`, `GoldenLoaderBase`, `ScoringScheme`, `CorrectnessChecker`, `PerfMetricStrategy`) + shared data models (`TaskSpec`, `CaseSpec`, `SolutionSpec`, enums) |
| `benches/` | Bench-specific implementations. Currently `cann` (CANN NPU operators) and `stanford` (Stanford KernelBench). Each registers its Loader, Matcher, Checker, ScoringScheme, PerfStrategy, and BenchConfig into global registries. |
| `registry/` | Global registries (`BenchRegistry`, `LoaderRegistry`, `CheckerRegistry`, `ScoringSchemeRegistry`, `PerfMetricStrategyRegistry`, `GoldenLoaderRegistry`, `OperatorMatcherRegistry`, `CaseSpecRegistry`). BenchConfig is the "one-stop config" — a single `--bench-name` parameter resolves all components. |
| `eval/` | Evaluation execution: `Evaluator` (top-level orchestrator), `AccuracyEvaluator`, `PerfEvaluator`, `OpRunner`, `SubprocessRunner` (fork+exec isolation), `ProcessPoolCoordinator` (multi-card parallel), `DataGenerator` (input generation), `InputPool`. |
| `data/` | Data loading: `CannTaskLoader` (proto.yaml), `CannCaseLoader` (cases.yaml), `GoldenLoader` (golden.py), `PackageManager` (compile/install/scan whl/run packages). |
| `checkers/` | Accuracy checkers: `RelativeErrorChecker` (MERE/MARE + small-value domain + cancellation handling, default) and `AllCloseChecker` (torch.allclose, for quick validation). |
| `report/` | Report generation: JSON + Markdown + HTML reports, `ScoringCalculator`, `SetupInfo`, summary generator. |
| `security/` | Anti-cheat: `APIGuard` (snapshot/verify torch.npu timing APIs against monkey-patching), `TorchOpGuard` (detect if AI op calls forbidden torch builtins like torch.matmul), `TypeChecker`. |
| `utils/` | `DeviceManager` (NPU/CPU switch, health check, aclrtResetDevice recovery), `BaselineResolver` (multi-hardware baseline with platform alias mapping), `Thresholds` (per-dtype precision thresholds), `DtypeMapper`, `ParamBuilder`, `TensorUtils`, `Naming`, `PathResolver`. |

### Evaluation Flow

1. **Load**: `CannTaskLoader`/`CannCaseLoader` scan `tasks/levelN/<op>/` directories, parse proto.yaml + cases.yaml + metadata JSON
2. **Compile & Install** (if `--source-dir`): `PackageManager` scans source dir, runs `build.sh`, installs whl + run package. Iterative compilation: failing operators quarantined, remaining continue.
3. **Match**: `OperatorMatcher` discovers available AI operators from installed `cann_bench` module, matches them to task definitions
4. **Generate Inputs**: `DataGenerator` creates input tensors from case specs (shapes, dtypes, value ranges, attrs), seeded by `eval_seed` for reproducibility
5. **Run**: `OpRunner` executes golden (fp64-CPU) and AI operator on NPU, with `TorchOpGuard` monitoring
6. **Accuracy**: `AccuracyEvaluator` compares AI output vs golden using `RelativeErrorChecker` (MERE < threshold, MARE < 10x threshold)
7. **Performance**: `PerfEvaluator` runs warmup+repeat cycles with `torch_npu.profiler`, `PerfMetricStrategy` parses profiler output (kernel_details.csv is the authoritative source; trace_view.json supplements tilefwk/PYPTO metrics)
8. **Score**: `ScoringCalculator` computes per-case HAP score, then aggregates per-operator composite score (Eq. 4)
9. **Report**: `ReportGenerator` outputs JSON + Markdown + HTML to `reports/`

### Subprocess Isolation

Operators are evaluated in forked subprocesses by default (`--no-subprocess-isolation` to disable). Benefits: one kernel crash/hang doesn't poison subsequent operators; timeout enforced via SIGTERM → SIGKILL. `APIGuard` snapshot/verify hooks protect timing APIs across subprocess boundaries.

### Multi-Card Parallel

When `--device-id` is unspecified and device is NPU, `ProcessPoolCoordinator` distributes operators across all available NPU cards (2 processes per card default). Each subprocess runs `kernel-bench eval-process` independently.

## Task Structure

Each operator directory under `tasks/levelN/<op>/` contains:
```
proto.yaml      — Operator prototype (inputs, outputs, attrs, schema)
cases.yaml      — Test case definitions (shapes, dtypes, baseline_perf_us, t_hw_us, attrs)
cases.csv       — Same data in CSV format (for legacy compatibility)
golden.py       — PyTorch reference implementation (executed on fp64-CPU for accuracy baseline)
desc.md         — Operator description (API documentation for AI agent consumption)
metadata/       — Baseline performance data per hardware platform (910b2.json, etc.)
```

`baseline_perf_us` and `t_hw_us` are stored per-hardware in `tasks/metadata/<hardware>.json` (e.g., `910b2.json`). Each file contains scalar values keyed by level → op → case_id. New platforms add a new JSON file; sparse coverage falls back to the default platform. These fields have been removed from `cases.yaml`; `baseline_resolver.py`'s dict-form parsing is legacy code no longer exercised by real data.

## Platform / SOC Detection

- `baseline_resolver.PLATFORM_ALIAS` maps product model names to logical names: `Ascend910_9362` → `910b2`, `Ascend910_9361` → `910b1`, `Ascend310P` → `310p`
- `CANN_BENCH_HARDWARE` env var overrides default hardware (default: `910b2`)
- `build.sh --soc=ascend910b|ascend910_93|ascend950` selects target chip for operator compilation
- `DeviceManager._check_npu()` auto-detects NPU availability via `torch.npu.is_available()`
- Anti-cheat `disable_builtin_kernels.sh --soc=<dir>` auto-detects SOC from `acl.get_soc_name()`

## Examples

- `examples/aclnn_launch_example/` — ACLNN operator project template (registry-invoke mode, L2 API + op_plugin)
- `examples/direct_launch_example/` — Direct launch operator project template (<<<>>> syntax, kernel + tiling + launch in one file)
- `examples/tasks/` — Evaluation task fixtures (add/sqrt with placeholder baselines, for pipeline validation)
- `examples/stanfordbench_example/` — Stanford KernelBench integration sample
- `examples/tilelang_cann_example/` — TileLang CANN operator template

Operator examples teach "how to write operators"; task examples teach "how to evaluate operators".

## auto_pipeline

Orchestrates AI agents to generate operator code:
- `core.py` — `CannBenchClient` loads case context, `run_from_config` executes pipeline from YAML config
- `prompt/` — Builds prompts from case material (proto, cases, desc) for AI agents
- `generator/` — Runs AI code generation: `pypto` (PYPTO/tilefwk), `akg` (AKG agent), `opencode` (OpenCode live bridge)
- `converter/` — Converts generated code to cann-bench submission format

CLI: `python -m auto_pipeline.cli --config pipeline.yaml --workspace /path`

## Security / Anti-Cheat

1. **APIGuard**: Snapshots `torch.npu.Event.elapsed_time`, `torch.npu.synchronize`, `torch_npu.profiler.*` before AI code runs; verifies they haven't been monkey-patched after. `ALLOW_TIMING_TAMPERING` env var for debugging.
2. **TorchOpGuard**: Detects AI operators calling forbidden torch builtins (matmul, conv, softmax, etc.) during execution. Modes: `warn` (default, log only) or `block` (raise RuntimeError).
3. **disable_builtin_kernels.sh**: Moves stock AiCore kernel binaries out of CANN OPP directory (mv, not rm; reversible via restore script). Never auto-invoked by evaluation scripts.
4. **enable_accuracy_retry**: Re-evaluates with fresh inputs to detect cached-output cheating.

## Key Configuration

- `Config` dataclass (`src/kernel_eval/config.py`): tasks_root, reports_dir, device_type/id, warmup/repeat, precision_thresholds, checker_name, eval_seed, perf_metric_strategy_override, torch_op_guard_mode, enable_acl_launch_mode
- `BenchConfig` dataclass: full component wiring (loader, matcher, checker, scoring, perf strategy, case spec class, precision thresholds)
- `pyproject.toml`: `[tool.uv] package = false` — uv manages deps only, no wheel build; runs via `PYTHONPATH=src`
- `requirements.txt`: torch 2.4+, torch_npu, pyyaml, numpy, scipy, etc.
- Environment variables: `CANN_BENCH_HARDWARE`, `ASCEND_OPP_PATH`, `ASCEND_HOME_PATH`, `TASKS_ROOT`, `ALLOW_TIMING_TAMPERING`

## Coding Conventions

- Chinese comments and log messages are common (this is a Huawei CANN project)
- Copyright header on every Python file: CANN Open Software License v2.0
- Dataclasses throughout (models, config, results, scoring info)
- Registry pattern: global `_items: Dict[str, T]` class vars on `BaseRegistry` subclasses; `register()` + `get()` API
- Import `kernel_eval.benches` to trigger auto-registration of all bench components
- Test files follow `test_<module>.py` naming in `tests/ut/` and `tests/e2e/`
- pytest config in pyproject.toml: `testpaths = ["tests"]`, `python_files = ["test_*.py"]`
- black: line-length=120; isort: profile=black, line_length=120

## Key File Paths

| Path | Purpose |
|------|---------|
| `VERSION` | Framework version (single source of truth) |
| `tasks/metadata/VERSION` | Task dataset version |
| `tasks/metadata/910b2.json` | Baseline perf data for 910B2 hardware |
| `src/kernel_eval/_version.py` | Reads VERSION files dynamically |
| `src/kernel_eval/cli.py` | CLI entry point (kernel-bench) |
| `src/kernel_eval/config.py` | Global Config dataclass + get_config/set_config |
| `src/kernel_eval/eval/evaluator.py` | Top-level evaluation orchestrator |
| `src/kernel_eval/benches/cann.py` | CANN bench registration + exports |
| `src/kernel_eval/benches/cann_loader.py` | CannTaskLoader, CannCaseLoader, GoldenLoader |
| `src/kernel_eval/benches/cann_scoring.py` | HAP scoring, weights (w_c=0.2, w_f=0.3, w_p=0.5) |
| `src/kernel_eval/base/perf_strategy.py` | PerfMetricStrategy ABC + KernelDetailsStrategy + TraceViewStrategy + MsProfSummaryStrategy |
| `src/kernel_eval/security/api_guard.py` | Timing API tampering protection |
| `src/kernel_eval/security/torch_op_guard.py` | Forbidden torch op detection |
| `src/kernel_eval/utils/baseline_resolver.py` | Multi-hardware baseline resolution + PLATFORM_ALIAS |
| `src/kernel_eval/utils/thresholds.py` | Per-dtype precision thresholds (PRECISION_THRESHOLDS dict) |
| `src/kernel_eval/data/package_manager.py` | Source scan, compile, install, iterative quarantine |
| `scripts/run_evaluation.sh` | Main evaluation shell entry |
| `scripts/run_test.sh` | Test runner (ut/e2e) |
| `scripts/anti_cheat/` | Kernel binary disable/restore scripts |