# 快速入门

本文档介绍如何使用评测工程进行算子代码生成评测。

## 前置条件

- Python 3.8+
- PyTorch 2.0+
- torch_npu（NPU 模式）
- CANN 环境（NPU 模式）

## 安装

```bash
pip install -e .
```

## 评测命令

### 从源码目录评测（推荐）

自动扫描、编译、安装 AI 生成的算子源码：

```bash
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops
```

### 评测指定算子

```bash
# 评测指定级别
./scripts/run_evaluation.sh --action eval --level 1

# 评测指定算子
./scripts/run_evaluation.sh --action eval --operator Exp --level 1

# 评测单个用例
./scripts/run_evaluation.sh --action eval --operator Exp --level 1 --case-id 1
```

### 查看算子信息

```bash
# 列出所有算子
./scripts/run_evaluation.sh --action list

# 列出指定级别的算子
./scripts/run_evaluation.sh --action list --level 1

# 查看算子详情
./scripts/run_evaluation.sh --action info --operator Exp
```

## 高级选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no-subprocess-isolation` | 关闭子进程隔离 | False（开启） |
| `--op-timeout-sec` | 算子评测超时时间（秒） | 240 |
| `--no-iterative-compile` | 关闭迭代隔离编译 | False（开启） |

## 评测报告

评测完成后，报告输出到 `reports/` 目录：

- `reports/eval_report.json`：JSON 格式详细报告
- `reports/eval_report.md`：Markdown 格式报告
- `reports/summary.md`：摘要报告
- `reports/prof_data/`：性能采集数据

## 下一步

- [贡献指南](contributing.md)：如何提交新算子评测任务
- [评测基准规范](../spec/benchmark_spec.md)：算子定义和精度标准
- [评测工程设计](../design/evaluator_design.md)：评测器架构设计