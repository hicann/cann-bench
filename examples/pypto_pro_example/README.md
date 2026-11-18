# PyPTO-Pro CANN Example

基于 PyPTO-Pro (Python-to-NPU compiler, Pro 版) 的 NPU 算子示例，打包为 `cann_bench` Python 包供 CANN Bench 评测。本示例由 PyPTO-Pro 算子开发工作流（pypto-pro-op-orchestrator）生成，作为后续 PyPTO-Pro 算子产物提交给 CANN Bench 的样板。

本示例包含一个算子 `rms_norm`，隔离在 `cann_bench/rms_norm/` 子包中。该结构同时演示了多算子提交的目录约定（见下文）。

## 目录结构

```
pypto_pro_example/
├── cann_bench/                         # Python 包
│   ├── __init__.py                     # 顶层 thin forwarder（聚合所有算子入口）
│   └── rms_norm/                       # 算子子包：RmsNorm (L2)
│       ├── __init__.py                 # from .dispatcher import rms_norm
│       ├── dispatcher.py               # 多 class (ndim,dtype) 分发器
│       └── c1/ ~ c7/
│           └── test_rms_norm.py        # @pl.jit kernel + host wrapper
├── build.sh                            # 构建脚本
├── setup.py                            # 打包配置
└── README.md
```

## 多算子隔离约定

每个算子独占一个 `cann_bench/<op>/` 子包，内含：
- `__init__.py`：从 dispatcher 导出入口函数
- `dispatcher.py`：多 class 分发器（按 `(ndim, dtype)` 签名路由）
- `c{N}/test_<op>.py`：每个 class 的 `@pl.jit` kernel + host wrapper

顶层 `cann_bench/__init__.py` 为每个算子写一个 thin forwarder，评测框架通过 `dir(cann_bench)` 扫描到所有算子接口。新增算子时只需：
1. 新建 `cann_bench/<op>/` 子包（复制上述结构）
2. 顶层 `__init__.py` 追加一个 forwarder
3. `setup.py` 的 `packages` 和 `package_data` 追加该子包

## 算子说明

### RmsNorm (L2 / Normalization)

$$y = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \cdot \gamma$$

- 输入：x（fp16/fp32/bf16，任意前导维度+末维 D）、gamma（1D [D]）
- 输出：y（同输入 shape/dtype）
- attrs：epsilon（float，默认 1e-6）
- 7 个 class (c1~c7) 覆盖不同 (ndim, dtype) 签名

## 快速评测（推荐）

### 前置条件

1. Python 3.10+，且已进入配置好 PyPTO-Pro 的 Python/Conda 环境
2. `import pypto_pro.language as pl` 可用
3. CANN 环境已初始化，`import torch_npu` 可用且 NPU 设备可见

### 全新检出后执行

首次使用时，在上述同一 Python 环境中安装 CANN Bench 依赖，然后从仓根目录启动评测：

```bash
git clone https://gitcode.com/cann/cann-bench.git
cd cann-bench
python -m pip install -r examples/pypto_pro_example/requirements.txt
./scripts/run_evaluation.sh examples/pypto_pro_example
```

如果仓库及其依赖已经准备好，后续只需在 cann-bench 仓根目录执行最后一条命令：

```bash
./scripts/run_evaluation.sh examples/pypto_pro_example
```

最终报告输出到 `reports/`，含 `cann_final_eval_<时间戳>.md` 汇总（编译分 + 精度分 + 性能分）。

> **工作目录**：`--source-dir examples/...` 为相对路径，必须在 **cann-bench 仓根目录**下执行，即先 `cd <path-to>/cann-bench`。
>
> **环境一致性**：依赖必须安装到能够导入 `pypto_pro` 的同一 Python 环境中。示例专用的 `requirements.txt` 只补齐评测框架依赖，不安装或替换 PyPTO-Pro 运行环境中的 `torch`、`torch_npu` 和 CANN；评测脚本会自动构建并安装 `cann_bench_utils`，无需手工准备该组件，也无需手工设置 `PYTHONPATH`。

### 多卡并行

脚本默认不传 `--device-id`，走多卡并行模式，调度参数默认为每卡 2 个 worker。PyPTO-Pro 的每个 case 均由独立的 `eval-child` 执行，自动把 20 个 case 分发到所有可见 NPU；可通过 `--processes-per-card` 调整每卡并发度。性能阶段开启 profiler 时，仍遵循框架原有的 profiler 独占规则，每卡运行 1 个 worker。单卡调试可加 `--device-id 0`。

### 报告输出位置

报告保存到 `reports/`（绝对路径，由脚本自动设置），包含：
- `cann_final_eval_<时间戳>.json` — 完整结构化数据（含 per-case elapsed_us、accuracy、kernel details 指标）
- `cann_final_eval_<时间戳>.md`   — Markdown 摘要（概览表 + 每算子详情表）
- `cann_final_eval_<时间戳>.html` — 独立可视化报告
- `cann_correctness_eval_<时间戳>.*` / `cann_performance_eval_<时间戳>.*` — 各阶段中间报告
- `cann_compile_failed_eval_<时间戳>.*` — 仅编译失败时产出（成功则无）

### 评测流程

```
./scripts/run_evaluation.sh examples/pypto_pro_example
  │  → python -m kernel_eval.staged_eval --source-dir examples/pypto_pro_example ...
  │
  ├─ stage1 compile:
  │    ├─ bash examples/pypto_pro_example/build.sh → cann_bench-1.0.0-py3-none-any.whl
  │    └─ 记录编译结果
  │
  ├─ stage2 correctness (enable_profiler=False):
  │    ├─ pip install --force-reinstall --no-deps cann_bench-1.0.0-py3-none-any.whl
  │    ├─ import cann_bench → 扫描接口: rms_norm
  │    ├─ 匹配 tasks/level2/rms_norm 算子定义
  │    └─ 每个 case 由独立 eval-child 跑精度（不采性能，耗时显示 N/A 属正常）
  │
  ├─ stage3 performance (enable_profiler=True):
  │    ├─ 只跑 stage2 通过的 case
  │    ├─ 每个 case 由独立 eval-child 执行 + ASCEND_RT_VISIBLE_DEVICES 隔离
  │    ├─ 测量前执行 NPU 升频和初始 L2 清理，每个 active repeat 前再次清 L2
  │    ├─ 继承 --profiler-level 配置，仅采集 NPU activity
  │    └─ 采集 kernel_details.csv 性能数据
  │
  └─ merge → cann_final_eval_<时间戳>.{json,md,html}
```

## 手动评测（备选）

如需绕过 staged-eval、直接单步评测（例如单算子调试），可手动两步：

```bash
# 1. 构建 wheel（staged-eval 会自动做这步，手动模式才需执行）
bash examples/pypto_pro_example/build.sh

# 2. 单步评测（单卡 device 0，默认采集性能）
PYTHONPATH=src python -m kernel_eval.cli eval \
  --bench-name cann \
  --task-dir tasks/level2/rms_norm \
  --source-dir examples/pypto_pro_example \
  --device-id 0 \
  --reports-dir "$PWD/reports"
```

> PyPTO-Pro 的每个用例由独立的 `eval-child` 执行，并在独立临时工作目录中完成 JIT 编译。调度器会先把 `--reports-dir`、`--task-dir` 和 `--source-dir` 转换为绝对路径，因此相对路径也能正确使用；命令中保留 `"$PWD/reports"` 是为了让实际报告位置更直观。

## 调用链

```
cann_bench/__init__.py
  → cann_bench.<op>  (子包)
    → <op>/__init__.py
      → <op>/dispatcher.py            # 按 (ndim, dtype) 签名匹配 c{N}
        → c{N}/test_<op>.py           # 实际 kernel (<op>_wrapper)
```

## PyPTO-Pro Kernel 结构

每个 `c{N}/test_<op>.py` 包含三部分：

| 部分 | 职责 | 示例 |
|------|------|------|
| **vector_function** | `@pl.vector_function` 装饰的 vector 计算函数 | `rms_norm_rows_vf(in_tile, out_tile, gamma_tile, ...)` |
| **Kernel 定义** | `@pl.jit(auto_mutex=True)` 装饰的 JIT kernel | `rms_norm_kernel(x, gamma, epsilon, y)` |
| **Host wrapper** | 折叠前导维、分配输出、调用 JIT | `rms_norm_wrapper(x, gamma, epsilon=1e-6)` |

JIT kernel 通过 `pl.TileType` + `pl.make_tile_group` 构造 tile 分组，用 SPMD 跨步循环（`pl.get_block_num` / `pl.get_block_idx`）遍历 M 维 tile，`pl.load` / `pl.store` 完成 GM↔UB 搬运。

## 多 Class 分发机制

各算子 `dispatcher.py` 中的 `_CLASSES` 表将输入 tensor 的 `(ndim, dtype)` 签名路由到对应 class 的 kernel。dispatcher 通过 `importlib` 按文件路径懒加载对应 class 的 `test_<op>.py`，入口函数查找顺序为 `<op>` → `<op>_wrapper`。

### RmsNorm

| Class | x ndim | x dtype  | gamma ndim | gamma dtype |
|-------|--------|----------|------------|-------------|
| c1    | 3D     | float16  | 1D         | float16     |
| c2    | 3D     | float32  | 1D         | float32     |
| c3    | 3D     | bfloat16 | 1D         | bfloat16    |
| c4    | 2D     | bfloat16 | 1D         | bfloat16    |
| c5    | 4D     | float16  | 1D         | float16     |
| c6    | 4D     | float32  | 1D         | float32     |
| c7    | 5D     | float32  | 1D         | float32     |

> 各 class 的 tile 参数（MAX_N/TILE_ROWS）、地址布局、dtype 转换策略、计算顺序均按 (ndim, dtype) 特化，不可合并为单一实现。
