# CANNBench 赛题接入模板

`skeleton` 分支面向想接入自有赛题的团队。它提供任务包、题集和评测适配器的起点。接入新赛题时，可以从最接近的示例复制任务包，再替换评测资产和适配器。

| 示例 | 可参考的部分 |
|---|---|
| **整网模型优化** | 用同一批输入验证基线和候选的整网输出，再比较端到端耗时。示例模型有两层固定权重的仿射变换，示例答案将两层合并。[分类模型任务包](tasks/example.model.classifier/) |
| **算子实现** | 根据输入输出契约实现 ReLU，以参考实现验证每个公开用例。[ReLU 任务包](tasks/example.operator.relu/) |

## 1. 克隆并跑通示例

两个示例可在 CPU 上执行，需要 Git 和 Python 3.10+。以下命令适用于 Bash。PowerShell 中用 `$env:PYTHONPATH = "$PWD/src"` 设置模块路径，其余命令相同。

```bash
git clone --depth 1 --single-branch --branch skeleton https://gitcode.com/cann/cann-bench.git cannbench-template
cd cannbench-template
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"

python -m cannbench validate --schema
python -m cannbench tasks --suite examples
python -m cannbench show example.model.classifier
python -m cannbench run example.model.classifier --suite examples --source-dir tasks/example.model.classifier/answer --reports-dir reports/model
python -m cannbench run example.operator.relu --suite examples --source-dir tasks/example.operator.relu/answer --reports-dir reports/relu
```

报告分别写入 `reports/model/example_result.json` 和 `reports/relu/example_result.json`。每次运行使用新的报告目录。将 `run` 换成 `plan` 可先查看评测器收到的参数。

在模型报告里，`correct` 表示已执行用例全部通过，`complete` 表示覆盖了全部用例。完整通过后，报告逐用例记录基线耗时、候选耗时和加速比，`score` 是用例加速比的几何平均。若只想排查一条用例，可加 `--case-id 1 --no-perf`，报告仍保留正确性结果。示例耗时会随主机和运行环境变化。

如果候选 `solution.py` 加载失败，CLI 返回退出码 `1`，并在 `example_result.json` 中通过 `error.stage`（`candidate_import`）、`error.type` 和 `error.message` 记录原因。此时 `correct` 和 `complete` 都为 `false`，`score` 为 `null`。

在仓库根目录运行 `python -m unittest discover -s tests`，可重跑两个 CPU 示例并检查导入失败的报告。分支的 CPU 冒烟工作流还会运行 `python -m cannbench validate --schema`，校验任务和题集清单。

## 2. 新增自己的赛题

复制接近目标的示例任务包，填好题目契约、工作负载、验收门槛和评分规则。用答案示例及典型失败提交验证评测后，再将固定任务版本加入题集。[新增赛题指南](docs/guide/adding_tasks.md)

若赛题是在已有模型上优化性能，可参考分类模型示例如何组织基线、端到端计时和答案。[整网模型优化接入指南](docs/guide/model_optimization.md)

## 3. 配置和代码各放在哪里

| 目录 | 团队应在这里维护什么 |
|---|---|
| `tasks/<task-id>/` | 该题的目标、资产引用、可改范围、最低资源和评分规则引用。[任务包契约](docs/spec/task_package.md) |
| `scoring/` | 可复用的计分公式、用例聚合和失败处理。用 `evaluation.scoring_rule` 引用固定版本。[端到端耗时比规则](scoring/latency-ratio.v1.yaml) |
| `profiles/` | 硬件、软件环境和测量边界。同一评分规则可由多道题使用，每道题选自己的测量配置。[CPU 端到端测量配置](profiles/measurement/python-e2e.yaml) |
| `suites/` | 题集成员及任务版本。需要榜单总分时，由团队定义跨题权重和聚合。[示例题集](suites/examples.yaml) |
| `schemas/` | 任务、题集、提交和结果的结构约束。[任务 Schema](schemas/task.schema.json) |
| `src/cannbench/` | 任务加载、CLI、执行编排和适配器注册。[执行入口](src/cannbench/engine.py) · [CPU 示例适配器](src/cannbench/adapters/python_example.py) |

CPU 示例适配器用于运行可信的本地 Python 提交。部署昇腾或多卡赛题时，接入团队自己的执行和隔离环境。共享评分规则描述公开口径，具体计算由评测器实现。

[许可证](LICENSE)
