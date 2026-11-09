# 快速入门

**文档版本：参见 [changelog](../changelog.md)**

本文档介绍如何使用评测工程进行算子代码生成评测。

## 前置条件

- Python 3.10+
- PyTorch 2.0+
- torch_npu（NPU 模式）
- CANN 9.1.0 及以上版本（NPU 模式）

> CANN 版本不得低于 9.1.0：ACLNN 示例（examples/aclnn_launch_example）引用的 `aclnn_kernels/contiguous.h` 头文件仅在 CANN 9.1.0+ 提供。使用前请 source 对应版本的环境脚本并确认 `ASCEND_HOME_PATH` 指向 9.1.0+ 的 CANN 安装目录：
>
> ```bash
> source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 按实际安装路径调整
> ```
>
> ⚠️ **已知问题（CANN 9.2.0）**：在 CANN 9.2.0 环境下，ACLNN 示例构建出的 `.run` 包安装验证可能失败，报 `undefined symbol: aclnnInnerMlaPrologV3V1`（该符号在 9.2.0 中由 `libopapi_transformer.so` 而非 `libopapi.so` 提供），即使加 `--force` 跳过验证，`import cann_bench` 也会失败。规避方法：`--force` 安装后在运行前预加载该库：
>
> ```bash
> ./dist/cann_bench_*.run --quiet --force
> export LD_PRELOAD="$ASCEND_HOME_PATH/$(uname -m)-linux/lib64/libopapi_transformer.so:${LD_PRELOAD}"
> ```
>
> 在 CANN 9.1.0 环境下无此问题；如条件允许，快速体验建议使用 CANN 9.1.0。

## 安装

仓内运行通常不需要安装本项目包，只需安装依赖并配置 `PYTHONPATH`：

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

> 仓库根目录已提供 `pyproject.toml`，但推荐的仓内运行方式仍是 `PYTHONPATH=src`。
> 若使用 `pip install -e .`，pip 会解析完整依赖（包括 `torch` / `torch_npu`），请确保当前 Python、CANN 和 NPU 依赖版本匹配。

## 评测命令

评测前请先阅读[评测输入与接口规范](../spec/submission_spec.md)。该规范列出解包后 `source_dir`、`build.sh`/wheel 约束、接口发现方式以及 auto-pipeline 校验规则编号。代码行为和反作弊边界见[算子提交行为与禁止规则](submission_rules.md)；通过 `cannbench.com` 网站提交时还需满足网站自身的 ZIP 上传规则。

### 快速体验（评测任务样例）

项目提供了 Add/Sqrt 两个轻量评测任务 fixture，可用于快速验证评测流水线是否正常工作：

```bash
# 评测 add 算子
./scripts/run_evaluation.sh --source-dir examples/aclnn_launch_example \
    --task-dir examples/tasks/level2/add --no-perf

# 评测目录下所有 fixture 算子（add + sqrt）
./scripts/run_evaluation.sh --source-dir examples/aclnn_launch_example \
    --task-dir examples/tasks --no-perf
```

样例的 `metadata/VERSION` 为 `1.0.0`，默认使用评分权重档 `1`。
权重档取自评测集大版本；对旧版或自定义 fixture，可在命令前设置
`CANN_BENCH_SCORE_WEIGHTS=1` 显式选择档位。样例性能数据为占位值，
仅用于 `--no-perf` 流程验证，不用于衡量性能。

详见 [examples/tasks/README.md](../../examples/tasks/README.md)。

### 从源码目录评测（推荐）

自动扫描、编译、安装 AI 生成的算子源码：

```bash
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops
```

### 评测指定算子

正式评测集会随版本变化，先运行 `./scripts/run_evaluation.sh -a list` 查看当前算子。
以下命令中的 `TASK_DIR` 填写实际的 `tasks/levelN/<op>` 目录，`OPERATOR_NAME`
填写该目录 `proto.yaml` 中的 `operator.name`；后续命令沿用这两个变量。

```bash
# 评测指定目录
./scripts/run_evaluation.sh --task-dir tasks/level1

# 评测单个算子目录
./scripts/run_evaluation.sh --task-dir "$TASK_DIR"

# 按算子名称筛选
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME"

# 评测单个用例
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME" --case-id 1

# 仅精度验证（不采集性能）
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME" --no-perf

# 设置 warmup/repeat 参数
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME" --warmup 5 --repeat 10
```

### 多卡并行评测

不指定 `--device-id` 时自动使用全部可用 NPU 卡：

```bash
# 多卡并行（自动检测）
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME"

# 指定每卡进程数
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME" --processes-per-card 4

# 指定进程超时
./scripts/run_evaluation.sh --operator "$OPERATOR_NAME" --timeout-per-process 600
```

### 单卡评测

```bash
# 单卡模式（指定设备 ID）
./scripts/run_evaluation.sh --device-id 0 --operator "$OPERATOR_NAME"
```

### 查看算子信息

```bash
# 列出所有算子
./scripts/run_evaluation.sh -a list

# 查看算子详情
./scripts/run_evaluation.sh -a info --operator "$OPERATOR_NAME"

# 查看配置
./scripts/run_evaluation.sh -a config
```

## 高级选项

`./scripts/run_evaluation.sh` 支持的参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device <type>` | 设备类型 (cpu/npu) | npu |
| `--device-id <id>` | NPU 设备 ID（不指定则多卡并行） | None |
| `--processes-per-card <n>` | 每卡进程数（多卡模式） | 2 |
| `--timeout-per-process <n>` | 单进程超时（秒，等价于 cli `--timeout-per-operator`） | 300 |
| `--warmup <n>` | 预热次数 | 3 |
| `--repeat <n>` | 采集次数 | 5 |
| `--no-perf` | 关闭性能采集（仅精度验证） | False |
| `--perf-batch-cases` | 同一 TaskUnit 的 case 共用一次 Profiler；归属不确定时自动逐 case 回退 | True |
| `--no-perf-batch-cases` | 关闭批量 case，恢复逐 case Profiler | False |
| `--perf-batch-case-memory-limit-mb <mb>` | correctness 阶段单 case VmRSS 峰值上限；超限或采样失败的 case 改走逐 case Profiler | 32768 |
| `--profiler-level <level>` | Profiler 级别 (Level1/Level2) | Level1 |

> shell 暴露的是最常用子集。若需要 `--op-timeout-sec` / `--no-iterative-compile` / `--eval-code` / `--eval-seed` / `--input-dist` 等更精细控制，请直调 cli：
> ```bash
> PYTHONPATH=src python -m kernel_eval.cli eval --source-dir /path/to/ai_ops \
>     --op-timeout-sec 480
> ```
> cli 完整参数表见 [evaluator_design.md §3.3](../design/evaluator_design.md#33-命令行参数)。

## Golden 自验证

将 golden 函数打包成 `cann_bench` whl，通过 `run_evaluation.sh` 验证 golden(NPU) 与 golden(CPU fp64) 的精度一致性：

```bash
# 1. 构建 golden whl 并安装
./scripts/utils/build_golden_wheel.sh --install

# 2. 评测（golden 作为 AI 算子，golden(CPU fp64) 作为参考）
./scripts/run_evaluation.sh --task-dir "$TASK_DIR" --no-perf
```

> ⚠️ **`cann_bench` 包名冲突**：golden whl 与 [examples/aclnn_launch_example](../../examples/aclnn_launch_example/) 构建的 wheel 同名（都叫 `cann_bench`），二者互斥。golden wheel 是纯 Python 包，只提供 `torch.ops.cann_bench` 下的 golden 算子（取决于当前评测集），**不包含**示例的 `cann_bench.add` / `cann_bench.sqrt`。golden 自验证完成后如需使用示例算子，请先卸载 golden wheel，再安装示例 wheel 或将示例目录加入 `PYTHONPATH`：
>
> ```bash
> pip uninstall -y cann_bench
> pip install examples/aclnn_launch_example/dist/cann_bench-*.whl
> # 或：export PYTHONPATH="$(pwd)/examples/aclnn_launch_example:${PYTHONPATH}"
> ```

## 评测报告

评测完成后，报告输出到 `reports/`（其它评测集为 `reports/<bench_name>/`）目录。文件名由“语义前缀 + 阶段 + 时间戳”组成，形如 `<bench>_<stage>_eval_<时间戳>.{json,md,html}`（cann 分阶段评测会分别产出 `cann_correctness_eval_*`、`cann_performance_eval_*`、`cann_final_eval_*`）：

- `<...>.json`：JSON 格式详细报告
- `<...>.md`：Markdown 格式报告
- `<...>.html`：HTML 格式报告
- `reports/prof_data/`：性能采集数据

## 下一步

- [评测输入与接口规范](../spec/submission_spec.md)：解包后的目录、构建、接口和校验规则编号
- [算子提交行为与禁止规则](submission_rules.md)：哪些实现方式会被视为无效或作弊
- [贡献指南](contributing.md)：如何提交新算子评测任务
- [评测基准规范](../spec/benchmark_spec.md)：算子定义和精度标准
- [评测工程设计](../design/evaluator_design.md)：评测器架构设计
- [性能采集设计](../design/perf_collection_design.md)：性能采集机制设计
