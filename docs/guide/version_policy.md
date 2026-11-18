# CANN-Bench 版本管理策略

## 1. 框架与题集独立版本

| 文件 | 管理范围 | 当前主线 |
|---|---|---|
| 根目录 `VERSION` | `kernel_eval` 与 `auto_pipeline` 框架 | 1.0.0 |
| `tasks/metadata/VERSION` | 当前正式题集 | 2.1.0（Season2） |
| `<bench_root>/metadata/VERSION` | 实际传给评测入口的其他题集 | 以该文件为准 |

当前正式题集的范围、摘要及平台限制见 [Season2 迁移说明](season2_migration.md)与 [suite_manifest.json](../../tasks/suite_manifest.json)。Season1 位于 `archives/season1/tasks/`，使用归档题集自己的 metadata；不得用框架安装目录下的主线题集版本替代实际被评测 bundle 的版本。

## 2. 运行时读取和评分选择

[`_version.py`](../../src/kernel_eval/_version.py) 提供：

```python
import kernel_eval
from kernel_eval._version import read_tasks_version

framework_version = kernel_eval.__version__  # 根 VERSION
bundled_tasks_version = kernel_eval.TASKS_VERSION  # 安装目录中 tasks/metadata/VERSION
actual_tasks_version = read_tasks_version("tasks")  # 显式指定本次题集根目录
```

`read_tasks_version` 严格读取 `<bench_root>/metadata/VERSION` 的第一行，要求 `X.Y.Z`；缺失、非法格式会报错。后续兼容性注释不参与版本解析。`FRAMEWORK_VERSION` / `TASKS_VERSION` 常量则用于框架自报及安装内容版本，不能与本次评测 bundle 的版本混用。

权重选择优先级为：显式 profile > `CANN_BENCH_SCORE_WEIGHTS` > 实际题集版本的大版本。支持的档位见 [score_weights.py](../../src/kernel_eval/base/score_weights.py)：

| 档位 | 编译/运行 | 精度 | 全通过 bonus | 性能 |
|---|---:|---:|---:|---:|
| 1 | 0.20 | 0.30 | 0 | 0.50 |
| 2 | 0.05 | 0.20 | 0.05 | 0.70 |

没有有效版本且未显式覆盖、或解析出未知档位时，不会静默回落。缺 VERSION 的历史/自定义数据复算可显式选择支持的档位，但报告应标明版本未知。`eq3` / `eq3_log` 公式选择由独立的 `CANN_BENCH_PERF_METRIC` 控制，不能从权重档推断。报告中的 `tasks_version`、`score_weight_profile`、`perf_metric` 应一并归档。

## 3. 版本变更和发布

框架与题集采用语义版本，分别审批与发布：不兼容契约变更用 MAJOR，兼容能力/题目扩展用 MINOR，修复用 PATCH。具体升级范围需结合发布影响确认，不能仅根据修改文件数量自动决定。

题集变更除版本外，还需同步公开范围、case ID、YAML/CSV、文件摘要及分平台 metadata。当前固定题集测试也要随经批准的新范围更新。修改版本文件只会更新运行时的版本读取结果，**不会自动刷新 manifest、数据或测试**。

发布检查顺序：

1. 确认本次框架/题集版本和兼容范围。
2. 更新根 `VERSION` 或实际题集的 `metadata/VERSION`，同步 `suite_manifest.json` 及对应 metadata。
3. 更新 [changelog](../changelog.md)，区分框架与题集变更。
4. 在仓库根目录运行相关检查：

   ```bash
   PYTHONPATH=src python -m pytest tests/ut/test_version_consistency.py \
     tests/ut/test_season2_public_suite.py
   ```

5. 完成目标硬件的软件配套、编译、精度和性能发布验收；CPU 检查不替代 NPU 验证。
6. 通过 PR 合入后，由发布维护者按批准的提交创建 `vX.Y.Z`（框架）、`tasks-vX.Y.Z`（题集）标签。两层独立变更，不要求每次同时打双标签。

## 4. 兼容性声明与其他目录

题集版本文件示例：

```text
2.1.0
# requires-framework: >=1.0.0
```

`requires-framework` 目前是兼容性声明；`read_tasks_version` 不解析或强制执行这个条件，发布者与使用者仍需核对配套版本，不能声称框架已经自动完成兼容性检查。

`bench_lab/` 和 `examples/` 不随正式 `tasks/` 统一发布，但其中被实际送入评分入口的集合仍需满足版本/权重选择约定。例如 `examples/tasks/metadata/VERSION` 为 1.0.0。实验集合不得因没有正式发布版本而静默借用主线的评分档位。

Docker tag 当前由框架 VERSION、SoC、CPU 架构及 ops 模式组成；它不单独标识题集版本。成绩复现还需镜像 digest / 构建提交、题集版本、硬件与实际评分口径，详见 [docker/eval](../../docker/eval/README.md)。
