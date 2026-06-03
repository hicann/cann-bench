# auto_pipeline agent 接入指南

本文说明如何把新的生成 agent 接入 `auto_pipeline`。当前流水线边界固定为：

```text
Core load case -> Prompt -> Generator -> Artifact -> Converter -> Submission -> Core eval
```

接入新 agent 时，先判断它属于哪一类：

| 类型 | 代表 | 适用场景 |
|------|------|----------|
| 通用 code agent + skills | PyPTO | 生成逻辑主要由外部 code agent 执行，cann-bench 只负责准备 workspace、prompt、env 和结果收集 |
| LangGraph workflow agent | AKG | agent 仓库本身暴露 Python workflow/API，cann-bench 直接调用 workflow 得到生成文件 |

这两类都必须落到同一个 `Generator` contract：消费 `GeneratorInput`，返回 `Artifact`。

## 公共接入点

新增 agent 通常涉及这些文件：

```text
src/auto_pipeline/generator/<agent_name>/
src/auto_pipeline/generator/registry.py
src/auto_pipeline/converter/registry.py
src/auto_pipeline/core.py
src/auto_pipeline/config/*.yaml
tests/unit/test_<agent_name>_agent.py
tests/unit/test_<agent_name>_to_<benchmark>_converter.py
```

核心 contract：

```python
from auto_pipeline.core import GeneratorInput
from auto_pipeline.core import Artifact


class MyAgent:
    type = "my-agent"

    def generate(self, task: GeneratorInput) -> Artifact:
        ...
```

`GeneratorInput` 由 core 加载 case 后通过 `prompt` package 构造，包含：

| 字段 | 含义 |
|------|------|
| `case` | 当前 benchmark case 元数据 |
| `material` | task 文件、op name、REQUIRE 文本、prompt context |
| `workdir` | 该 case 的生成工作目录 |
| `output_dir` | generator artifact 输出目录 |
| `env` | pipeline 注入的运行环境 |
| `metadata` | device、task selector 等运行信息 |

`Artifact` 是 generator 和 converter 之间的唯一生成产物 contract。常用字段：

| 字段 | 含义 |
|------|------|
| `status` | `success` / `failed` 等状态 |
| `workdir` | artifact 根目录 |
| `files` | 结构化文件表，例如 `{"source_dir": path}` |
| `log_file` | generator 日志 |
| `metadata` | agent-specific 元数据 |
| `message` | 失败原因或简短说明 |

`auto_pipeline` 不要求 generator 产物直接可评测。只要 converter 能把 `Artifact`
转成目标 benchmark 的 `Submission` 即可。

## 配置边界

YAML 只描述实验意图和 agent 静态策略：

```yaml
agent:
  type: my-agent
  mode: some_static_strategy

benchmark:
  name: cann
  tasks:
    - tasks/level1/exp
```

机器相关、本次运行相关的值走 CLI args：

```bash
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
./scripts/run_auto_pipeline.sh \
  --config src/auto_pipeline/config/my_agent_cann_exp.yaml \
  --workspace /path/to/my-agent-repo \
  --model deepseek/deepseek-v4-pro \
  --output benchmark_runs/my_agent_exp \
  --devices 0,1 \
  --parallel 2
```

不要把 `repo_root`、卡号、`output`、model、临时输出路径写进提交的 YAML。
PyPTO 所需的 `PTO_TILE_LIB_CODE_PATH` 由外部环境提供，pipeline 只检查它非空。
PyPTO stage7 性能优化轮次默认是 3，可用环境变量 `PYPTO_PERF_ROUND` 覆盖。

## 路径一：通用 code agent + skills

这类 agent 的特点是：cann-bench 不直接 import 生成系统内部 workflow，而是调用一个
通用 runner，例如 `opencode run --agent <skill>`。PyPTO 属于这一类。

推荐分层：

```text
Generator:
  准备 workspace / prompt / env / artifact 判定

Runner:
  执行外部命令，负责 PROMPT.md、日志、timeout、session export

Converter:
  把 raw artifact 整理成 benchmark submission
```

### PyPTO 示例

PyPTO 链路：

```text
CaseMaterial
  -> PyptoOrchestratorAgent
  -> PyPTO workspace: custom/<op>/
  -> OpenCodeAgent runner
  -> opencode run --agent pypto-op-orchestrator
  -> raw PyPTO artifact
  -> pypto -> cann converter
  -> kernel_eval
```

对应实现：

| 文件 | 职责 |
|------|------|
| `generator/pypto/orchestrator.py` | generator，准备 PyPTO workspace、渲染 prompt、判断 raw artifact |
| `generator/opencode/runner.py` | runner，执行 `opencode run` |
| `generator/pypto/templates/orchestrator.j2` | PyPTO code agent prompt |
| `generator/pypto/converter/to_cann.py` | raw PyPTO artifact -> CANN submission |
| `generator/pypto/converter/to_stanford.py` | raw PyPTO artifact -> Stanford submission |
| `generator/registry.py` | 注册 `pypto` generator 和 `opencode` runner |
| `converter/registry.py` | 注册 `pypto -> cann`、`pypto -> stanford` |

PyPTO generator 做的事情：

1. 从 `GeneratorInput.material` 读取 `proto.yaml`、`cases.yaml`、`desc.md`、`golden.py`。
2. 在 PyPTO repo 下准备 `custom/<op>/` 或隔离 worktree。
3. 写入 `REQUIRE.md` 和必要 task 文件。
4. 渲染 `orchestrator.j2`，构造 `RunnerPrompt`。
5. 调用 `OpenCodeAgent.run()`。
6. 检查 raw artifact 是否包含预期文件，例如 `<op>_impl.py`、`SPEC.md`。
7. 返回 `Artifact(files={"source_dir": raw_dir}, ...)`。

这种方式的重点是：generator 不重复实现外部 code agent 的执行细节；runner 不理解
PyPTO、CANN 或 benchmark 语义。

### 使用 Claude Code 的 agent

如果新 agent 需要调用 Claude Code，不建议在 generator 里直接拼 `claude` 命令。
正确边界是新增一个 runner，例如：

```text
src/auto_pipeline/generator/claude_code/
├── __init__.py
└── runner.py
```

`ClaudeCodeRunner` 应实现通用 `Runner` contract：

```python
from auto_pipeline.generator.base import Runner
from auto_pipeline.core import Artifact, RunnerPrompt


class ClaudeCodeRunner:
    type = "claude-code"

    def run(self, prompt: RunnerPrompt) -> Artifact:
        ...
```

runner 负责所有 Claude Code 执行细节：

- 解析 `claude` 可执行文件路径。
- 写入 `PROMPT.md`。
- 构造 Claude Code 非交互执行命令。
- 合并 `RunnerPrompt.env`、runtime env 和必要的隔离环境。
- 设置 cwd、timeout、日志文件。
- 捕获 return code、timeout、启动失败。
- 收集输出文件并返回 `Artifact`。
- 必要时导出 session/transcript，但这仍然属于 runner 细节。

generator 仍然只做 agent 语义：

1. 从 `GeneratorInput.material` 构造 Claude Code 要看的 task 文件和 prompt。
2. 准备 workspace。
3. 构造 `RunnerPrompt(text=..., cwd=..., output_dir=..., env=...)`。
4. 调用 `ClaudeCodeRunner.run(prompt)`。
5. 检查 raw artifact 是否满足该 agent 的产物约定。
6. 返回结构化 `Artifact` 给 converter。

注册方式：

```python
from auto_pipeline.generator.claude_code import ClaudeCodeRunner


def _create_claude_code(cfg):
    return ClaudeCodeRunner(
        claude_bin=str(cfg.get("bin") or cfg.get("claude_bin") or "claude"),
        model=str(cfg.get("model") or ""),
        extra_args=_string_list(cfg.get("extra_args")),
    )


register_runner("claude-code", _create_claude_code)
```

如果 Claude Code 用在 generation 阶段，通常由该 agent 的 factory 创建并注入 runner：

```python
from auto_pipeline.generator.registry import create_runner


def _create_my_agent(cfg):
    runner = create_runner("claude-code", cfg.get("runner") or {})
    return MyAgent(repo_root=cfg["repo_root"], runner=runner)
```

如果希望在 YAML 中写 `agent.runner` 这类静态 runner 配置，需要同时把 `runner`
加入 `core.py` 的 `_allowed_agent_keys()`，并确保它不包含机器私有路径或卡号。

如果 Claude Code 用在 conversion 阶段，则在 `core.py` 的
`_conversion_runner()` 中按 `agent.type + benchmark.name` 选择 `create_runner("claude-code", ...)`。

测试至少覆盖：

- `ClaudeCodeRunner` 能写 `PROMPT.md`、构造命令、传入 env、记录日志。
- `claude` 不存在时返回 `AGENT_NOT_FOUND`。
- timeout 和非零返回码能变成 failed `Artifact`。
- generator 不直接调用 `subprocess` 执行 Claude Code。
- converter 不依赖 Claude Code 私有输出目录，仍只消费 `Artifact.files`。

### 新 code-agent 接入步骤

1. 新增 `src/auto_pipeline/generator/<name>/agent.py`。
2. 在 generator 中把 `CaseMaterial` 映射成该 agent 需要的 workspace 和 prompt。
3. 复用 `OpenCodeAgent`，或者为 Claude Code 等外部工具新增一个只负责执行命令的 `Runner`。
4. 返回 raw artifact，保留日志和 metadata。
5. 新增 converter，把 raw artifact 打包成目标 benchmark submission。
6. 在 registry 注册 generator、runner、converter。
7. 写 unit tests 覆盖 workspace、prompt、env、device_id、artifact 判定和 converter。

## 路径二：LangGraph workflow agent

这类 agent 的特点是：外部 agent 仓库已经提供 Python workflow/API，cann-bench 直接
import 或 subprocess 调用它。AKG 属于这一类。

推荐分层：

```text
Generator:
  生成 workflow 输入配置
  设置 agent repo 的 Python path/env
  调用 workflow
  收集 workflow 输出文件

Runner:
  通常不需要

Converter:
  把 workflow 输出代码打包成 benchmark submission
```

### AKG 示例

AKG Triton Ascend 链路：

```text
CaseMaterial
  -> AkgAgent
  -> AKG LangGraph workflow: kernelgen_only_workflow
  -> generated Triton Ascend code
  -> akg-agent -> cann converter
  -> kernel_eval
```

对应实现：

| 文件 | 职责 |
|------|------|
| `generator/akg/agent.py` | generator，加载 AKG repo、构造 workflow 配置、调用 AKG workflow |
| `generator/akg/converter/to_cann.py` | AKG generated code -> CANN submission |
| `generator/akg/converter/to_stanford.py` | AKG generated code -> Stanford submission |
| `generator/registry.py` | 注册 `akg-agent` generator |
| `converter/registry.py` | 注册 `akg-agent -> cann`、`akg-agent -> stanford` |

AKG generator 做的事情：

1. 解析 `GeneratorInput.material`，得到 op name、schema、task 文件。
2. 从 CLI runtime `--workspace` 找到 AKG repo。
3. 把 AKG Python package 路径注入当前进程环境。
4. 构造 workflow 配置，例如 `backend`、`arch`、`framework`、`codegen_target`、
   `workflow`、`device_id`。
5. 调用 AKG workflow，并让 workflow 自己完成 kernel generation、code check、
   verifier。
6. 从 workflow 输出目录收集 generated code。
7. 返回 `Artifact(files={"source_dir": generated_dir}, metadata={"akg_task_config": ...})`。

这种方式的重点是：cann-bench 不接管 LangGraph 内部节点，也不把 verifier 结果当成
最终 benchmark 得分。最终通过与其它 agent 相同的 converter 和 `kernel_eval` 得到
统一报告。

### 新 workflow-agent 接入步骤

1. 新增 `src/auto_pipeline/generator/<name>/agent.py`。
2. 明确外部 repo 的 Python import root，不要依赖调用者手动设置 `PYTHONPATH`。
3. 将 `GeneratorInput.material` 转成 workflow 输入配置。
4. 将 `device_id`、timeout、env 透传给 workflow。
5. 捕获 workflow 异常，返回 failed `Artifact`，不要让错误绕过 pipeline report。
6. 收集 generated source，返回结构化 `files`。
7. 新增 converter，负责 submission 打包，而不是让 generator 直接拼评测目录。
8. 注册 generator/converter，并补 unit tests。

## Registry 规则

generator 注册在 `src/auto_pipeline/generator/registry.py`：

```python
def _create_my_agent(cfg):
    repo_root = cfg.get("repo_root")
    if not repo_root:
        raise ValueError("my-agent generator requires repo_root")
    return MyAgent(repo_root=repo_root, device_id=_optional_int(cfg.get("device_id")))


register_generator("my-agent", _create_my_agent)
```

converter 注册在 `src/auto_pipeline/converter/registry.py`：

```python
register_converter("my-agent", "cann", lambda cfg: MyAgentToCannConverter(...))
```

`core.py` 会根据 `agent.type + benchmark.name` 自动选择 converter。
如果该 agent 需要额外静态配置，加入 `_allowed_agent_keys()`，并在 YAML 的 `agent`
下声明。运行时路径、卡号、并发不要加入 YAML schema。

## Converter 规则

converter 是 agent 和 benchmark 之间的适配层。它负责：

- 校验 generated artifact 是否可用于目标 benchmark。
- 创建 submission 目录。
- 写 `cann_bench/<op>.py`、`benchmark_submission.json`、`build.sh` 等目标 benchmark
  需要的文件。
- 必要时构建 wheel 或运行 packaging command。
- 返回 `Submission(kind=<benchmark>, operator=<op>, source_dir=<submission_dir>)`。

converter 不应该：

- 重新调用 generator。
- 修改 benchmark task 定义。
- 静默吞掉 generation 失败。
- 把 agent 私有 workspace 当成最终 submission。

## 测试 checklist

新增 agent 至少覆盖：

- registry 能创建 generator。
- 缺少 `--workspace` 或必要 runtime 值时报错清晰。
- `device_id` 能从 `--devices` 注入 generator。
- generator 成功时返回 `Artifact.ok == True` 和正确 `files`。
- generator 失败时返回 failed `Artifact`，pipeline report 可读。
- converter 能从最小 artifact 构造 submission。
- `core.py` 能把 simplified YAML + runtime args 串到 generator/converter/eval。
- 并发时每个 task 使用预期 device。

建议先用 fake repo/fake workflow 做 unit test，再在真实机器上跑一个小 task 做 smoke。
