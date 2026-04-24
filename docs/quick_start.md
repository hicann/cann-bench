# Kernel Eval 评测工程快速入门

本文档介绍如何使用 `kernel_eval` 评测工程验证AI生成的Ascend C算子代码。

## 1. 环境准备

### 1.1 安装依赖

```bash
pip install torch pyyaml numpy
```

NPU环境还需安装：
```bash
pip install torch_npu
```

### 1.2 设置Python路径

```bash
export PYTHONPATH=src:$PYTHONPATH
```

或在代码中添加：
```python
import sys
sys.path.insert(0, 'src')
```

## 2. 基本使用

### 2.1 快速脚本方式（推荐）

使用 `scripts/run_test.sh` 脚本执行测试：

```bash
# 查看帮助
./scripts/run_test.sh --help

# 运行所有测试
./scripts/run_test.sh

# CPU 设备测试（默认）
./scripts/run_test.sh --cpu

# NPU 设备测试
./scripts/run_test.sh --npu

# Level 1 测试
./scripts/run_test.sh --level 1

# 指定算子测试
./scripts/run_test.sh --operator gelu

# NPU 测试 + 性能采集
./scripts/run_test.sh --npu --prof

# 详细输出模式
./scripts/run_test.sh --level 1 -v

# 组合选项
./scripts/run_test.sh --npu --operator matmul --level 2 -v
```

脚本参数说明：
| 参数 | 说明 |
|------|------|
| `--cpu` | 使用 CPU 设备测试（默认） |
| `--npu` | 使用 NPU 设备测试 |
| `--level <1-4>` | 按难度级别筛选 |
| `--operator <name>` | 按算子名称筛选（模糊匹配） |
| `--case-id <num>` | 按用例编号筛选 |
| `--prof` | 启用性能采集 |
| `-v, --verbose` | 详细输出 |
| `-o, --output <path>` | 指定结果输出文件路径 |

### 2.2 命令行方式

**列出算子**
```bash
# 列出所有level的算子
PYTHONPATH=src python -m kernel_eval.cli list

# 列出指定level的算子
PYTHONPATH=src python -m kernel_eval.cli list --level 1

# 列出指定算子的用例
PYTHONPATH=src python -m kernel_eval.cli list --level 1 --operator Exp --cases
```

**查看算子详情**
```bash
# 查看算子详细信息
PYTHONPATH=src python -m kernel_eval.cli info --operator Exp

# 指定level查看
PYTHONPATH=src python -m kernel_eval.cli info --operator Softmax --level 2
```

**执行评测**
```bash
# 从源码目录评测（自动扫描编译安装）
PYTHONPATH=src python -m kernel_eval.cli eval --source-dir /path/to/ai_ops

# 仅执行Golden验证（不安装whl）
PYTHONPATH=src python -m kernel_eval.cli eval --operator Exp --level 1

# 评测单个用例
PYTHONPATH=src python -m kernel_eval.cli eval --operator Exp --level 1 --case-id 1
```

**配置管理**
```bash
# 查看当前配置
PYTHONPATH=src python -m kernel_eval.cli config --show
```

### 2.3 Python API方式

**加载算子信息**
```python
from kernel_eval.data import OperatorLoader, CaseLoader

# 加载算子定义
loader = OperatorLoader()
op_info = loader.get_operator('Exp', level=1)
print(f"算子: {op_info.name}")
print(f"接口: {op_info.schema}")
print(f"属性: {[a.name for a in op_info.attrs]}")

# 加载测试用例
case_loader = CaseLoader('kernel_bench')  # kernel_bench是数据目录名
cases = case_loader.scan_by_operator(1, 'Exp')
print(f"用例数: {len(cases)}")
```

**从源码目录执行评测**
```python
from kernel_eval.eval import Evaluator
from kernel_eval.report import ReportGenerator

# 创建评测器
evaluator = Evaluator()

# 从源码目录评测（自动扫描编译安装）
session_result = evaluator.evaluate_from_source(
    source_dir='/path/to/ai_ops',
    verbose=True
)

# 打印结果
for op_result in session_result.operators:
    print(f"算子: {op_result.operator}")
    print(f"通过率: {op_result.pass_rate:.2%}")
    print(f"平均加速比: {op_result.avg_speedup:.2f}x")

# 生成报告
report_gen = ReportGenerator()
for op_result in session_result.operators:
    report_gen.add_operator_result(op_result)
report = report_gen.generate()
report_gen.save_all(report)

# 关闭评测器
evaluator.shutdown()
```

## 3. 评测流程详解

### 3.1 源码目录结构要求

AI生成的算子源码目录应包含以下结构：
```
source_dir/
├── build.sh          # 编译脚本（可选）
├── dist/             # 编译产物目录（可选）
│   ├── cann_bench_xxx.whl   # Python包
│   └── cann_bench_xxx.run   # NPU内核包（可选）
├── cann_bench/       # Python包目录
│   └── __init__.py
├── csrc/             # C++源码
│   └── ops/
│       └── exp/
│           └── ascend910b/
│               └── exp.cpp
├── setup.py          # 构建配置
└── CMakeLists.txt
```

### 3.2 评测执行流程

评测工程执行以下步骤：
1. **扫描源码目录** - 检查build.sh、dist目录是否存在
2. **检查包** - 如果dist目录有whl/run包则跳过编译，否则执行build.sh
3. **安装包** - 先安装run包（NPU内核包），再安装whl包（Python包）
4. **扫描接口** - 导入cann_bench，扫描算子接口并打印
5. **匹配用例** - 根据接口名称匹配kernel_bench中的算子定义
6. **执行评测** - 加载用例数据，执行Golden和AI算子，精度验证，性能评测
7. **生成报告** - 输出JSON和Markdown格式报告

### 3.3 输出示例

```bash
$ ./scripts/run_test.sh --npu --level 1 --operator Exp

========================================
设备: --npu
级别: Level 1
算子: Exp
========================================

[INFO] 加载算子 Exp (L1), 用例数: 20
[1/20] L1_Exp_1: ✅ (耗时: 125.50μs, 加速比: 1.25x)
[2/20] L1_Exp_2: ✅ (耗时: 118.30μs, 加速比: 1.30x)
...

========================================
测试完成
========================================
```

## 4. 评测报告

### 4.1 报告输出位置

默认输出到 `test/reports/` 目录：
```
test/reports/
├── test_results.json      # JSON格式报告
├── traces/                # 性能采集数据（NPU + --prof）
```

### 4.2 报告内容

**JSON报告结构**
```json
{
  "summary": {
    "total": 100,
    "passed": 98,
    "failed": 2,
    "pass_rate": "98.00%",
    "timestamp": "2026-04-22T17:00:00"
  },
  "results": [
    {
      "level": 1,
      "operator": "Exp",
      "case_id": 1,
      "status": "success",
      "elapsed_us": 125.50,
      "device": "npu",
      "speedup": 1.25
    }
  ]
}
```

### 4.3 评分公式

```
编译通过得分 = compile_pass × Wc  (Wc=2，compile_pass ∈ {0, 1}，整份提交编译是否通过)
功能得分     = case_pass × Wf     (Wf=3，case_pass ∈ {0, 1}，单个用例是否通过精度校验)
性能得分     = SpeedUp × Wp       (Wp=5，仅对功能通过的用例计入，SpeedUp 按该用例实测)

单算子综合评分 = 编译通过得分 + Σ_{功能通过的用例} (Wf + SpeedUp × Wp)

Level-N 得分   = Σ 该 level 内算子综合评分
benchmark 总分 = Σ 所有算子综合评分 (= Level1 + Level2 + Level3 + Level4)
```

## 5. 精度验证标准

采用生态算子开源精度标准（MERE/MARE指标）：

### 误差计算公式
- **平均相对误差 (MERE)**: avg(|actual - golden| / (|golden| + 1e-7))
- **最大相对误差 (MARE)**: max(|actual - golden| / (|golden| + 1e-7))

### 精度阈值表
| 数据类型 | 阈值 (Threshold) |
|---------|-----------------|
| float16 | 2^-10 ≈ 0.000976 |
| bfloat16 | 2^-7 ≈ 0.007812 |
| float32 | 2^-13 ≈ 0.000122 |
| hifloat32 | 2^-11 ≈ 0.000488 |
| float8_e4m3 | 2^-3 ≈ 0.125 |
| float8_e5m2 | 2^-2 ≈ 0.25 |
| int8/16/32/64 | 0 (完全相等) |

### 通过标准
当 **MERE < Threshold** 且 **MARE < 10 × Threshold** 时判定为通过

## 6. 评测模式说明

当不指定 `--source-dir` 参数时，评测工程默认跳过编译安装步骤，直接扫描已安装的 `cann_bench` 模块接口并执行评测。

适用场景：
- 包已经安装
- 只想验证特定算子
- 调试测试

## 7. 包命名约定

AI生成的包应遵循以下命名规范：

| 包类型 | 命名格式 | 说明 |
|---------|---------|------|
| whl包 | `cann_bench_xxx.whl` | Python包，包含算子接口 |
| run包 | `cann_bench_xxx.run` | NPU内核二进制包 |

## 8. 常见问题

### Q1: 如何只验证Golden函数正确性？

不提供source-dir参数，仅指定算子：
```bash
./scripts/run_test.sh --cpu --operator Exp --level 1
```

### Q2: 如何查看详细错误信息？

使用 `-v` 参数启用详细输出：
```bash
./scripts/run_test.sh --npu --operator Exp -v
```

### Q3: 如何更新基线性能数据？

使用 `update_baseline_perf.py` 脚本：
```bash
python scripts/update_baseline_perf.py test/reports/test_results.json kernel_bench
```

### Q4: dist目录下需要哪些包？

- **whl包**（必须）：`cann_bench_xxx.whl` - Python包，包含算子接口
- **run包**（可选）：`cann_bench_xxx.run` - NPU内核二进制包

## 9. 示例代码

快速测试：
```bash
# CPU 测试 Golden 函数
./scripts/run_test.sh --cpu --operator Exp --level 1 --case-id 1

# NPU 性能评测
./scripts/run_test.sh --npu --operator Exp --prof

# 查看报告
cat test/reports/test_results.json
```

更多设计细节见：
- `docs/kernel_bench_eval_engine_design.md` - 评测工程设计文档（V2.0）
- `scripts/README.md` - 脚本使用说明

---

**命名说明**：
- `kernel_eval`：评测工程代码目录（src/kernel_eval）
- `kernel_bench`：测试用例数据目录（kernel_bench/level*/op_name/）
- `scripts/run_test.sh`：统一测试运行脚本