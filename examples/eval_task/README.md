# 评测任务示例 (eval_task)

> **定位**：本目录是评测流水线的**示例 fixture**，非生产评测目标。
> mish 等正式评测算子在 `tasks/` 目录下，本目录仅提供 add/sqrt 两个轻量算子，
> 供开发者理解评测任务目录结构、验证流水线是否正常工作。

## 目录结构

```
eval_task/
├── add/                    # Add 算子（双输入 Elementwise）
│   ├── proto.yaml          # 算子原型定义
│   ├── golden.py           # PyTorch 参考实现
│   ├── cases.yaml          # 评测用例定义
│   └── desc.md             # 算子 API 文档
├── sqrt/                   # Sqrt 算子（单输入 Elementwise）
│   ├── proto.yaml
│   ├── golden.py
│   ├── cases.yaml
│   └── desc.md
└── README.md               # 本文件
```

## 用例说明

这两个算子的 `cases.yaml` 中所有 `baseline_perf_us` 和 `t_hw_us` 均设为 `0.0`（placeholder），因为本目录用于验证评测流水线机制，不做真实性能评测。

| 算子 | 输入 | dtype | 用例数 | 特点 |
|------|------|-------|--------|------|
| Add | x, y (双输入) | float16, float32, bfloat16, int32 | 5 | 演示双输入算子的评测流程 |
| Sqrt | x (单输入) | float16, float32, bfloat16 | 4 | 演示单输入算子的评测流程 |

## 使用方式

### 1. 通过评测脚本

```bash
# 评测 add 算子（CPU 模式 + 仅精度验证）
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops --task-dir examples/eval_task/add --device cpu --no-perf

# 评测 sqrt 算子
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops --task-dir examples/eval_task/sqrt --device cpu --no-perf

# 评测目录下所有算子（add + sqrt）
./scripts/run_evaluation.sh --source-dir /path/to/ai_ops --task-dir examples/eval_task --device cpu --no-perf
```

### 2. 通过 kernel_eval CLI

```bash
PYTHONPATH=src python -m kernel_eval.cli eval \
    --source-dir /path/to/ai_ops \
    --task-dir examples/eval_task/add \
    --device cpu --no-perf --case-id 1
```

### 3. 通过 auto_pipeline

```python
from auto_pipeline.core import CannBenchClient

client = CannBenchClient()
add_case = client.load_case("cann", "examples/eval_task/add")
sqrt_case = client.load_case("cann", "examples/eval_task/sqrt")
```

## 与生产评测任务的关系

| 目录 | 定位 | baseline/t_hw | 用例数 |
|------|------|---------------|--------|
| `tasks/levelN/<op>/` | 正式评测集 | 真实测量值 | 20+ |
| `examples/eval_task/<op>/` | 流水线 fixture | `0.0` placeholder | 4-5 |

如需新增生产评测算子，请参照 [`docs/guide/contributing.md`](../../docs/guide/contributing.md) 将算子目录放入 `tasks/levelN/` 下。

## 与算子工程样例的关系

| 目录 | 定位 | 内容 |
|------|------|------|
| `examples/direct_launch_example/` | 算子**实现**样例 | C++ kernel + Python bindings（编译→安装→调用） |
| `examples/aclnn_launch_example/` | ACLNN 算子**实现**样例 | ACLNN 模式算子工程 |
| `examples/eval_task/` | 算子**评测**样例 | proto.yaml + golden.py + cases.yaml（定义→评测） |

简单理解：`direct_launch_example` / `aclnn_launch_example` 教你"怎么写算子"，`eval_task` 教你"怎么评算子"。