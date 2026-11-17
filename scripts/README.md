# Scripts

本目录包含评测工程运行脚本和工具脚本。

**目录结构**：

```
scripts/
├── setup_ascend_arena_env.sh    # 用户级 Conda 环境一键安装与 NPU 检查
├── activate_ascend_arena_env.sh # 加载 Conda、CANN 与项目环境
├── run_evaluation.sh        # AI算子源码评测（完整流程：编译→安装→评测）
├── run_auto_pipeline.sh     # auto_pipeline 的 python -m 包装入口
├── batch_optimize.py          # 可续传的多卡批量优化
├── generate_batch_report.py   # 汇总手动/agent 优化结果为单文件 HTML
├── generate_acceptance_report.py # 生成可归档的验收报告与证据包
├── run_ut.sh                # 单元测试运行
├── utils/
│   ├── build_golden_wheel.sh # Golden whl 打包（Shell 入口）
│   ├── build_golden_wheel.py # Golden whl 打包（Python 逻辑）
│   ├── yaml_to_csv.py       # YAML → CSV 转换
│   └── yaml_block_to_flow.py  # YAML block → flow 格式转换
└── README.md
```

---

## batch_optimize.py

按 NPU 卡数并发优化当前注册的 7 个 Ascend Arena 算子。每张卡固定一个 worker，同一卡上的算子串行执行：

```bash
cp -n src/ascend_arena_e2e/agents/.env.example src/ascend_arena_e2e/agents/.env
# 编辑 .env 中的 codex profile
python scripts/batch_optimize.py \
  --agent codex \
  --devices 0,1,2,3
```

`.env` 使用 YAML 格式，包含 harness、API key、API 地址、模型和推理强度。Claude Code
只需使用对应的 profile：

```bash
python scripts/batch_optimize.py --agent claude-code --devices 0,1,2,3
```

默认输出为 `benchmark_runs/<agent>-<model>/`；`--output` 可覆盖。目录中的 `config.json`
记录复现参数，`progress.json` 原子记录每个算子的状态，`report.html` 内嵌整批及逐算子详情。
Agent 输出会实时显示在终端，并同步刷新到各 workspace 的 `artifact/<agent>.log`。
workspace 包含源码、binding、reference 和 smoke cases；complete/profile cases 由评测入口从仓库任务目录读取。`AGENTS.md` 由所选 Agent 类生成。
再次执行会跳过已完成算子；`--restart` 会重新运行全部所选算子，`--force` 还会重建 workspace。
增加 `--docker` 后，每个算子在独立临时容器中运行，`--docker-image` 可指定镜像。

省略 `--agent` 时，创建全部 7 个算子的 workspace 并完成原始基线的五阶段评测：

```bash
python scripts/batch_optimize.py --devices 0,1 --output benchmark_runs/manual
```

## generate_batch_report.py

手动优化并执行各 workspace 的 `validate.sh` 后，重新生成唯一的批次报告：

```bash
python scripts/generate_batch_report.py benchmark_runs/manual
```

报告数据直接嵌入 `report.html`，不依赖任何算子子目录中的 HTML 文件。自定义 agent 接口见
[根目录 README](../README.md#external-agent)。

## generate_acceptance_report.py

将路径中的占位内容替换为实际目录或运行 ID，再执行示例。

批量功能测试完成后，使用下面的命令生成仓库级功能完整性报告。报告不逐个展开算子；
算子结果只汇总为样本数量、编译/正确性/性能通过数和平均加速比，详细文件仍归档到 `evidence/`：

```bash
python scripts/generate_acceptance_report.py \
  "benchmark_runs/ascend-arena-e2e-<时间>" \
  --command "python tests/run_ascend_arena_batch_e2e.py --output benchmark_runs/ascend-arena-e2e-<时间>"
```

`run_root` 可以是批量目录（含 `e2e_summary.json`）、单算子 workspace（含
`kernel_eval/`），也可以直接是一次 candidate 评测目录（含 `results/status.json`）。
脚本按 README 功能生成验收矩阵；默认从 `<run_root>/feature_results.json` 读取人工执行的
功能命令和截图，也可以显式指定：

```bash
python scripts/generate_acceptance_report.py "<run-root>" \
  --feature-results /path/to/feature_results.json \
  --screenshots-dir /path/to/screenshots \
  --command "<实际执行命令>"
```

`feature_results.json` 按功能 ID 填写，示例如下。将执行结果和证据路径替换为本次实际值：

```json
{
  "features": {
    "workspace_only": {
      "status": "PASS",
      "actual": "工作区创建成功",
      "evidence": ["workspace-proof.txt"]
    }
  }
}
```
`PASS` 必须同时填写 `actual` 并提供可读的 `evidence` 文件；否则脚本会降为
`INCOMPLETE`。一次只跑批量算子评测属于 `runtime-only` 证据范围；没有执行的功能标记 `N/A`，
功能完整性结论保持 `INCOMPLETE`，不会把一次算子评测冒充为全仓库验收。
例如：

```bash
python scripts/generate_acceptance_report.py benchmark_runs/evaluate/adaptive_avg_pool3d
python scripts/generate_acceptance_report.py "benchmark_runs/evaluate/adaptive_avg_pool3d/kernel_eval/ascend_arena_adaptive_avg_pool3d_<run>"
```

默认生成目录为 `<run_root>/acceptance_report/`，包括：

- `acceptance_report.md`：环境、第三方骨架版本、README 功能验收矩阵、算子总体摘要和审核清单；
- `report_data.json`：机器可解析的同一份数据；
- `evidence/`：汇总、功能结果、固定版本骨架清单、状态、逐 case 对比、HTML/轨迹和日志等小型证据，不复制 Profiler 原始目录；
- `screenshots/`：截图目录（默认为空）。

把任意机器上截取的图片放入目录后重新生成，脚本会复制并在报告中使用相对链接：

```bash
python scripts/generate_acceptance_report.py \
  "benchmark_runs/ascend-arena-e2e-<时间>" \
  --screenshots-dir /path/to/screenshots \
  --command "<实际执行命令>"
```

归档前请检查执行命令、截图和日志中不含 API Key、Token 或密码。

建议截图命名为 `01_environment.png`、`02_command.png`、`03_batch_summary.png` 和
`04_report.png`。`report_data.json` 中的证据链接均为相对路径，可和
`acceptance_report/` 一起复制到其他机器；环境中的 CANN 路径在 JSON 中只保留文件名，
并标记为测试主机路径。逐算子 `comparison.json` 只在需要复核具体性能时打开。

---

## run_auto_pipeline.sh

auto_pipeline 的 shell 包装，等价于在仓库根目录设置 `PYTHONPATH=src` 后执行
`python -m auto_pipeline.cli`。

### 用法

```bash
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
./scripts/run_auto_pipeline.sh \
  --config path/to/config.yaml \
  --workspace /path/to/agent/repo \
  --model deepseek/deepseek-v4-pro \
  --devices 0 \
  --parallel 1
```

`run_auto_pipeline.sh` 只设置 `PYTHONPATH=src` 并转发到 `python -m auto_pipeline.cli`。
当前 `auto_pipeline` 的使用说明见
[`docs/guide/auto_pipeline_usage.md`](../docs/guide/auto_pipeline_usage.md)，实现设计见
[`docs/design/benchmark_orchestrator_architecture.md`](../docs/design/benchmark_orchestrator_architecture.md)。

准备一个可手动编辑的 Ascend Arena workspace：

```bash
./scripts/run_auto_pipeline.sh prepare \
  --operator adaptive_avg_pool3d \
  --output benchmark_runs/adaptive_avg_pool3d_workspace \
  --device-id 0
```

可编辑源码位于 `benchmark_runs/adaptive_avg_pool3d_workspace/adaptive_avg_pool3d/work/adaptive_avg_pool3d/`，修改完成后
在 `work/` 下运行 `./validate.sh`。

workspace 已复制 benchmark 用例和 binding，`validate.sh` 使用这份副本，因此整个
workspace 可以移动到其它目录。请先激活安装了 CANN、torch_npu 和 pybind11 的 Python
环境（例如 `conda activate cannbench`），再执行验证脚本。

---

## run_evaluation.sh

AI 生成算子源码评测脚本，支持完整流程（扫描→编译→安装→评测）。

### 用法

```bash
./scripts/run_evaluation.sh [源码目录] [选项]
```

### 命令行选项

#### 操作选项

| 选项 | 说明 |
|------|------|
| `-a, --action <action>` | 操作类型：`eval`(评测)、`list`(列表)、`info`(详情)、`config`(配置)，默认：`eval` |
| `--source-dir <dir>` | AI 生成的算子源码目录（自动扫描编译安装） |

#### 目录配置

| 选项 | 说明 |
|------|------|
| `--task-dir <path>` | 指定评测目录，默认：`tasks`。支持：`tasks`、`tasks/level1`、`tasks/level1/exp` 等 |

#### 设备配置

| 选项 | 说明 |
|------|------|
| `--device <type>` | 设备类型：`cpu`、`npu`（默认：`npu`） |
| `--device-id <id>` | 指定 NPU 设备 ID（单卡模式）。不指定则自动使用全部可用卡（多卡并行） |

#### 多进程并行配置

| 选项 | 说明 |
|------|------|
| `--processes-per-card <n>` | 每卡进程数（默认：2） |
| `--timeout-per-process <n>` | 单进程超时（秒，默认：300）。**等价于 cli 的 `--timeout-per-operator`**；shell 通过环境变量 `KERNEL_BENCH_TIMEOUT_PER_PROCESS` 传给底层 |

#### 用例筛选

| 选项 | 说明 |
|------|------|
| `--operator <name>` | 按算子名称筛选 |
| `--case-id <id>` | 按用例编号筛选 |

#### 性能配置

| 选项 | 说明 |
|------|------|
| `--warmup <n>` | 预热次数（默认：3） |
| `--repeat <n>` | 采集次数（默认：5） |
| `--no-perf` | 关闭性能采集，仅做精度验证 |
| `--profiler-level <level>` | Profiler 级别：`Level1`、`Level2`（默认：`Level1`） |

#### 其他选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 详细输出 |
| `-h, --help` | 显示帮助信息 |

> shell 只透传最常用参数子集。若需要 `--no-subprocess-isolation` / `--op-timeout-sec` / `--no-iterative-compile` / `--reports-dir` / `--eval-code` 等，请直调 cli（`python -m kernel_eval.cli eval ...`），完整参数表见 [docs/design/evaluator_design.md §3.3](../docs/design/evaluator_design.md#33-命令行参数)。

Ascend Arena 直接评测使用独立流程：

```bash
./scripts/run_evaluation.sh --run_optimize \
  --source-dir /path/to/adaptive_avg_pool3d \
  --task-dir bench_lab/ascend_arena_bench \
  --operator adaptive_avg_pool3d --device-id 0
```

结果默认写入 `reports/ascend_arena/<run>/`，可用 `--reports-dir` 指定目录。

### 使用示例

#### 从源码目录评测（推荐）

```bash
# 位置参数方式
./scripts/run_evaluation.sh /path/to/ai_ops

# 显式指定 --source-dir
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops

# 评测指定算子
./scripts/run_evaluation.sh /path/to/ai_ops --operator Exp

# 评测单个用例
./scripts/run_evaluation.sh /path/to/ai_ops --operator Exp --case-id 1
```

#### 指定评测目录

```bash
# 评测 tasks 目录（默认）
./scripts/run_evaluation.sh

# 评测指定级别目录
./scripts/run_evaluation.sh --task-dir tasks/level1

# 评测单个算子目录
./scripts/run_evaluation.sh --task-dir tasks/level1/exp

# 按算子名称筛选
./scripts/run_evaluation.sh --operator Exp
```

#### 设备配置

```bash
# CPU 评测
./scripts/run_evaluation.sh --device cpu --operator Exp

# 单卡 NPU 评测
./scripts/run_evaluation.sh --device-id 0 --operator Exp

# 多卡并行评测（自动检测全部可用卡）
./scripts/run_evaluation.sh --operator Exp
```

#### 性能配置

```bash
# 仅精度验证（关闭性能采集）
./scripts/run_evaluation.sh --no-perf --operator Exp

# 自定义预热和采集次数
./scripts/run_evaluation.sh --warmup 5 --repeat 10 --operator Exp

# 使用 Level2 Profiler（更详细）
./scripts/run_evaluation.sh --profiler-level Level2 --operator Exp
```

#### 查看算子信息

```bash
# 列出算子
./scripts/run_evaluation.sh -a list

# 查看算子详情
./scripts/run_evaluation.sh -a info --operator Exp

# 查看配置
./scripts/run_evaluation.sh -a config
```

### 输出

评测报告保存在 `reports/` 目录：
- `reports/eval_report.json`：JSON 格式详细报告
- `reports/eval_report.md`：Markdown 格式报告
- `reports/summary.md`：摘要报告

---

## build_golden_wheel.sh

Golden whl 打包脚本，将 tasks/ 下 golden.py 函数收集打包成纯 Python `cann_bench` whl 包，
注册到 `torch.ops.cann_bench` 命名空间，用于验证 golden(NPU) 与 golden(CPU fp64) 的精度一致性。

### 用法

```bash
# 构建所有算子并安装
./scripts/utils/build_golden_wheel.sh --install

# 只打包指定算子
./scripts/utils/build_golden_wheel.sh --operator Mish --install

# 只打包指定级别
./scripts/utils/build_golden_wheel.sh --level 1 --install

# 构建后通过 run_evaluation 评测
./scripts/run_evaluation.sh --task-dir tasks/level1/mish --no-perf
```

详见 `scripts/utils/build_golden_wheel.py` 中的完整参数说明。

---

## run_ut.sh

单元测试运行脚本，基于 pytest。

### 用法

```bash
./scripts/run_ut.sh [选项]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 详细模式，显示每个测试名称和结果 |
| `-q, --quiet` | 静默模式，只显示最终统计 |
| `-k, --keyword <kw>` | 按关键字筛选测试（pytest -k） |
| `-f, --file <name>` | 指定测试文件（如 test_config.py） |
| `-t, --test <spec>` | 指定测试方法（如 TestConfig::test_default_config） |
| `-x, --fail-fast` | 首次失败即停止 |
| `-s, --no-capture` | 不捕获 stdout/stderr（调试用） |
| `-j, --jobs <n>` | 并行执行（pytest-xdist） |
| `--pdb` | 失败时进入 pdb 调试器 |
| `-h, --help` | 显示帮助 |

### 使用示例

```bash
# 全部单元测试
./scripts/run_ut.sh

# 详细模式
./scripts/run_ut.sh -v

# 按关键字筛选
./scripts/run_ut.sh -k "config"

# 指定测试文件
./scripts/run_ut.sh -f test_config.py

# 指定测试方法
./scripts/run_ut.sh -f test_config.py -t TestConfig::test_default_config

# 失败即停 + 详细输出
./scripts/run_ut.sh -x -v

# 无捕获 + 失败调试
./scripts/run_ut.sh -s --pdb
```

---

## utils/yaml_to_csv.py

将 cases.yaml 转换为 cases.csv 格式。

### 用法

```bash
python scripts/utils/yaml_to_csv.py <input.yaml> [-o <output.csv>]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `input_file` | 输入 YAML 文件路径 |
| `-o, --output <path>` | 输出 CSV 文件路径，默认与输入同名但扩展名改为 .csv |

### 使用示例

```bash
# 转换为同名 CSV
python scripts/utils/yaml_to_csv.py tasks/level1/exp/cases.yaml

# 指定输出路径
python scripts/utils/yaml_to_csv.py tasks/level1/exp/cases.yaml -o output.csv
```

---

## utils/yaml_block_to_flow.py

将 YAML 文件中的 block format 数组转换为 flow format。

### 用法

```bash
python scripts/utils/yaml_block_to_flow.py <input.yaml> [-o <output.yaml>] [-s <style>]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `input_file` | 输入 YAML 文件路径 |
| `-o, --output <path>` | 输出 YAML 文件路径，默认输出到控制台 |
| `-i, --indent <n>` | 缩进空格数（默认：2） |
| `-s, --style <style>` | 转换风格：`all`（所有列表）、`selective`（仅简单列表）、`smart`（智能选择），默认：`selective` |

### 使用示例

```bash
# 转换为 flow format（输出到控制台）
python scripts/utils/yaml_block_to_flow.py cases.yaml

# 保存到文件
python scripts/utils/yaml_block_to_flow.py cases.yaml -o cases_flow.yaml

# 转换所有列表为 flow format
python scripts/utils/yaml_block_to_flow.py cases.yaml -s all

# 智能转换
python scripts/utils/yaml_block_to_flow.py cases.yaml -s smart
```

---

## 脚本对比

| 脚本 | 用途 | 底层入口 | 输入 | 输出 |
|------|------|----------|------|------|
| `run_evaluation.sh` | AI 算子完整评测 | `src/kernel_eval/cli.py` | AI 源码目录 / tasks 目录 | 评测报告 |
| `run_test.sh` | 测试（ut/e2e） | pytest | tests/ut + tests/e2e 目录 | pytest 输出 |
| `utils/build_golden_wheel.sh` | Golden whl 打包 | `scripts/utils/build_golden_wheel.py` | tasks 目录 | cann_bench whl |
| `utils/gen_report.sh` | 评测报告生成 | `kernel_eval.report.html_generator` | eval JSON 文件 | HTML 报告 |

**底层入口职责**：

| 入口 | 职责 | 核心能力 |
|------|------|----------|
| `cli.py` | **评测核心** | `--task-dir`、多卡并行、`--no-perf`、`--source-dir`、`--profiler-level` |

**参数统一说明**：

`run_evaluation.sh` 支持以下参数：

| 参数 | 说明 |
|------|------|
| `--task-dir <path>` | 指定评测目录. 按级别评测写 `tasks/levelN` (本脚本没有 `--level`, 见下方说明) |
| `--device-id <id>` | 单卡模式：指定设备；不指定则多卡并行 |
| `--processes-per-card <n>` | 多卡并行时每卡进程数 |
| `--warmup <n>` | 预热次数 |
| `--repeat <n>` | 采集次数 |
| `--no-perf` | 关闭性能采集，仅精度验证 |
| `--operator <name>` | 按算子名称筛选 |
| `--profiler-level <level>` | Profiler 级别（Level1/Level2） |

**关于 `--level`**: `kernel_eval` 的 `eval` / `list` / `info` 三个子命令有 `--level`, 但
`run_evaluation.sh` **没有** -- NPU + cann 这一默认组合下它分发到 `kernel_eval.staged_eval`,
而后者的参数表里不含 `--level`. 经本脚本按级别评测请用 `--task-dir tasks/levelN`; 需要
`--level` 时直接调 `python -m kernel_eval.cli eval --level N`.

---

## 相关文档

- [评测基准规范](../docs/spec/benchmark_spec.md)
- [评测工程设计](../docs/design/evaluator_design.md)
- [性能采集设计](../docs/design/perf_collection_design.md)
- [快速入门](../docs/guide/quick_start.md)
