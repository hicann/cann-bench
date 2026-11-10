# 评测输入与接口规范

本文档是 `cann/cann-bench` 仓库对评测输入的公开规范，适用于 Runner 已经解包并确定提交根目录之后的构建、安装、接口发现、auto-pipeline 转换和运行行为检查。

本仓库不定义网站上传 ZIP 的大小、条目数、路径安全、允许扩展名、用户额度、硬件在线状态或 Benchmark 版本选择规则。这些是通过 `cannbench.com` 网站提交时需要满足的额外要求，由网站独立定义。

## 1. 规则编号与所有权

| 前缀 | 本仓执行层 | 主要实现位置 |
| --- | --- | --- |
| `EVAL-CANN` | CANN 构建、安装与接口发现 | `src/kernel_eval/data/package_manager.py` |
| `AUTO-CANN` | auto-pipeline CANN 转换结果 | `src/auto_pipeline/converter/submission.py` |
| `AUTO-STANFORD` | auto-pipeline Stanford 转换结果 | `src/auto_pipeline/converter/submission.py` |
| `SUB-BEH` | 运行行为与反作弊边界 | `docs/guide/submission_rules.md` |

auto-pipeline 校验错误包含规则编号、问题位置、期望、实际值和修复建议。同一阶段能够独立判断的问题应一次性返回；依赖构建或导入成功的检查可以在后续阶段报告。

## 2. CANN 评测输入约束

评测器接收一个已解包的 `source_dir`。该目录可以包含可复现构建源码，也可以包含预构建产物。

| 规则编号 | 约束 | 失败结果 |
| --- | --- | --- |
| `EVAL-CANN-001` | `source_dir` 必须存在。 | 提交准备失败。 |
| `EVAL-CANN-002` | 若 `dist/cann_bench*.whl` 不存在，根目录必须有 `build.sh`。 | 无法编译。 |
| `EVAL-CANN-003` | `build.sh` 由 `bash build.sh` 执行，必须在构建超时内以 0 退出并在 `dist/` 生成 wheel。 | 本次提交相关算子全部按编译失败计 0 分。 |
| `EVAL-CANN-004` | wheel 必须可由 pip 使用 `--force-reinstall --no-deps` 安装，并可导入为 `cann_bench`。 | 安装或接口扫描失败。 |
| `EVAL-CANN-005` | `cann_bench` 至少公开一个非 class 的 callable，待评测接口名称应与任务算子名或 schema 函数名匹配。 | 没有匹配算子，无法进入评测。 |

推荐的可复现源码结构：

```text
source_dir/
├── build.sh
├── setup.py
├── cann_bench/
│   ├── __init__.py
│   └── <operator>.py
├── csrc/                 # Ascend C / C++ 提交按需提供
└── scripts/              # 构建辅助脚本按需提供
```

预构建结构：

```text
source_dir/
└── dist/
    ├── cann_bench*.whl
    └── cann_bench*.run   # 可选；存在时先于 wheel 安装
```

预构建 wheel 可以跳过构建，但建议提交可复现源码。若同时存在 run 包和 wheel，评测器先安装 run 包，再安装 wheel；任一安装失败都会终止准备流程。

## 3. auto-pipeline CANN 转换校验

auto-pipeline 必须产出可复现的标准源码提交，因此比评测器接受预构建产物的最低要求更严格。

| 规则编号 | 约束 |
| --- | --- |
| `AUTO-SUB-001` | `source_dir` 必须是现有目录。 |
| `AUTO-CANN-001` | 根目录必须包含普通文件 `build.sh`。 |
| `AUTO-CANN-002` | 必须包含 `cann_bench/` 目录，或 `dist/cann_bench*.whl`。 |

校验器会在一次错误中同时报告所有可判定的结构问题，并列出提交根目录实际内容。

## 4. auto-pipeline Stanford 转换校验

| 规则编号 | 约束 |
| --- | --- |
| `AUTO-SUB-001` | `source_dir` 必须是现有目录。 |
| `AUTO-STANFORD-001` | 根目录必须包含普通文件 `ai_op.py`。 |
| `AUTO-STANFORD-002` | `ai_op.py` 必须是 UTF-8 编码的合法 Python。 |
| `AUTO-STANFORD-003` | `ai_op.py` 不得使用相对 import；本地模块应使用可从 `source_dir` 解析的绝对 import。 |
| `AUTO-STANFORD-004` | Benchmark task 模块必须可导入并定义 `Model`。 |
| `AUTO-STANFORD-005` | `ai_op.py` 必须可导入，依赖应包含在 `source_dir` 或运行环境中。 |
| `AUTO-STANFORD-006` | task 必须定义 `Model`，提交必须定义 `ModelNew`。 |
| `AUTO-STANFORD-007` | `ModelNew.__init__` 的参数顺序、种类和默认值必须与 `Model.__init__` 一致。 |
| `AUTO-STANFORD-008` | `ModelNew.forward` 的参数顺序、种类和默认值必须与 `Model.forward` 一致。 |
| `AUTO-STANFORD-009` | 两个模型必须能用 `get_init_inputs()` 构造，并支持 `state_dict()`。 |
| `AUTO-STANFORD-010` | 两个 `state_dict` 的 key 和顺序必须一致。 |
| `AUTO-STANFORD-011` | 同名 state tensor 的 shape 必须一致。 |
| `AUTO-STANFORD-012` | 同名 state tensor 的 dtype 必须一致。 |

转换器会在 `ai_op.py` 前加入本地模块搜索路径 prelude。提交不得依赖父目录、历史 run 或未打包的工作区文件。

## 5. 运行行为约束

文件和接口约束通过不代表提交有效。评测期间还会检查或审查：

- `SUB-BEH-001`：不得调用 PyTorch / torch_npu 内置计算 API 代算。
- `SUB-BEH-002`：不得在包装层完成实质性输入输出 tensor 变换。
- `SUB-BEH-003`：不得路由到 CANN 内置同名算子。
- `SUB-BEH-004`：不得 CPU fallback，且必须实际执行提交 NPU kernel。
- `SUB-BEH-005`：不得缓存、固定输出或按输入地址/公开 case 命中。
- `SUB-BEH-006`：不得篡改 profiler、同步或 timing API。
- `SUB-BEH-007`：必须返回真实 `torch.Tensor`。

检测范围、完整提交约束、违规处理及行为示例统一见[算子提交行为与禁止规则](../guide/submission_rules.md)。

## 6. 规则变更要求

新增或修改本仓提交约束时，必须同时：

1. 分配或更新 `EVAL-*`、`AUTO-*` 或 `SUB-BEH-*` 规则编号。
2. 更新本文档或行为规则文档。
3. 更新拥有该规则的 evaluator 或 auto-pipeline 实现。
4. 添加合法与非法样例测试。
5. 不在本仓复制 `cannbench.com` 网站的上传限制；网站约束变化应由网站独立维护。

没有文档和测试的隐式约束不应合入。
