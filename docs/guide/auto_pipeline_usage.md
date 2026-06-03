# auto_pipeline 使用指南

本文说明如何运行 `auto_pipeline`。架构和模块边界见
[`docs/design/benchmark_orchestrator_architecture.md`](../design/benchmark_orchestrator_architecture.md)；
新增 agent 接入见
[`docs/guide/auto_pipeline_agent_integration.md`](auto_pipeline_agent_integration.md)。

## 快速入口

```bash
./scripts/run_auto_pipeline.sh --config path/to/config.yaml --workspace /path/to/agent/repo
```

`scripts/run_auto_pipeline.sh` 只设置 `PYTHONPATH=src`，然后转发到：

```bash
python -m auto_pipeline.cli
```

## 当前注册项

Generators：

- `akg-agent`
- `pypto`

Converters：

- `pypto -> cann`
- `pypto -> stanford`
- `akg-agent -> cann`
- `akg-agent -> stanford`

Runners：

- `opencode`

## 配置结构

YAML 只描述实验意图：用什么 agent、跑哪些任务、agent 的静态策略是什么。
机器相关和本次运行相关的值通过 CLI 参数传入，包括源码仓路径、输出目录、卡号、
并发和 model。workdir、convert、submission、kernel_eval 和报告路径都由 pipeline
从 runtime `output` 固定推导。

```yaml
agent:
  type: pypto

benchmark:
  name: cann
  tasks:
    - bench_lab/pypto_cann_bench/exp
    - bench_lab/pypto_cann_bench/sigmoid
```

当前保留的示例配置：

```text
src/auto_pipeline/config/akg_cann_exp_sigmoid.yaml
src/auto_pipeline/config/pypto_cann_exp_sigmoid.yaml
```

旧的 `bench`、`generation`、`submission`、`converter`、`agent_output`、`dsl`、
`generator`、`convert`、`eval`、`cases` 都不是当前配置入口。

## CLI 参数

```text
--config <path>       pipeline YAML config
--workspace <path>    runtime agent source workspace
--model <name>        model for PyPTO generation and conversion
--output <path>       runtime output root
--devices <ids>       device ids, e.g. 0,1 or 1-7
--parallel <n>        maximum number of benchmark tasks to run in parallel
```

不要把 `repo_root`、卡号、`output`、model、临时输出路径写进提交的 YAML。

## PyPTO 示例

```bash
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export PYPTO_PERF_ROUND=0
./scripts/run_auto_pipeline.sh \
  --config src/auto_pipeline/config/pypto_cann_exp_sigmoid.yaml \
  --workspace /path/to/pypto_src \
  --model deepseek/deepseek-v4-pro \
  --output benchmark_runs/pypto_cann_exp_sigmoid \
  --devices 0,1 \
  --parallel 2
```

PyPTO 所需的 `PTO_TILE_LIB_CODE_PATH` 必须由外部环境提前设置，pipeline 只检查它非空，
不负责配置。`--model` 用于 PyPTO generation 和 PyPTO conversion 的 OpenCode runner。
PyPTO stage7 性能优化轮次默认是 3，可用环境变量 `PYPTO_PERF_ROUND` 覆盖。

对于 `agent.type: pypto` 和 `benchmark.name: cann`，pipeline 自动设置：

- converter runner: `opencode`
- eval perf source: `trace_view`
- eval args: `--no-subprocess-isolation --op-timeout-sec 3600 --verbose`
- device id: 来自 `--devices` 调度，同时注入 PyPTO generator 和 kernel_eval

## AKG 示例

```yaml
agent:
  type: akg-agent
  backend: ascend
  arch: ascend910b4
  framework: torch
  codegen_target: triton_ascend
  workflow: kernelgen_only_workflow

benchmark:
  name: cann
  tasks:
    - tasks/level1/gelu
```

```bash
./scripts/run_auto_pipeline.sh \
  --config src/auto_pipeline/config/akg_cann_exp_sigmoid.yaml \
  --workspace /path/to/akg_repo \
  --output benchmark_runs/akg_cann_exp_sigmoid \
  --devices 0,1 \
  --parallel 2
```

## 输出目录

`--output` 可选；不配置时自动生成：

```text
benchmark_runs/run_<timestamp>_<random>/
```

每个 task 的目录固定为：

```text
<output>/<task_name>/work
<output>/<task_name>/convert
<output>/<task_name>/submission
<output>/<task_name>/kernel_eval
<output>/<task_name>/benchmark_result.json
```

批量报告固定为：

```text
<output>/batch_result.json
```

报告包含 case、generator prompt、generated artifact、conversion artifact、submission
和 `kernel_eval` 命令/返回码/report 文件等信息。`PipelineRunResult.ok` 只有在
pipeline 状态为 `success` 且 `kernel_eval` 返回码为 0 时为真。
