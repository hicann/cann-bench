# 性能采集设计文档

> **当前实现范围**：性能执行入口在 `eval/perf_eval.py`，解析策略已分离至 `base/perf_strategy.py`，不是正文图中单个 `_parse_kernel_details` 方法。默认候选路径使用 `torch_npu.profiler`，warmup=3、repeat=5；脚本默认支持批量 case 采集及逐 case 回退。默认 `kernel_details` 指标统计设备 kernel 耗时，不是 Python 调用端到端墙钟。
>
> `trace_view`、`msprof_summary` 等策略有不同适用范围，不能混称同一性能口径；baseline 脚本的采集模式见[baseline 设计顶部现状说明](baseline_collection_design.md)。参数以[使用指南](../guide/quick_start.md)和实际 `--help` 为准。

**文档版本：参见 [changelog](../changelog.md)**

本文档详细描述评测工程的性能采集机制设计，包括 NPU Profiler 采集、kernel_details.csv 解析、升频清 Cache、Warmup Kernel 精确过滤等核心机制。

**更新说明参见 [changelog](../changelog.md)**：
- Profiling 升级为 Level1（默认）/ Level2（可选），删除 Level0 支持
- 数据源改为 kernel_details.csv（47列），支持 Input Shapes 精确形状匹配过滤 warmup
- 删除 `_suppress_cann_profiler_errors()`（Level1/Level2 有完整数据，无需抑制）
- 统计量改为中位数

---

## 1. 设计目标

性能采集模块的目标是获取 **NPU kernel-only 执行时间**，用于计算生成算子与基准性能的加速比。

设计原则：

| 目标 | 说明 |
|------|------|
| **精准性** | 只测量 NPU 内核执行时间，排除 API 调用、Host 端开销 |
| **一致性** | 通过升频清 Cache 保证多次测量结果稳定 |
| **防作弊** | InputPool 防止按 data_ptr 缓存输出的攻击 |
| **容错性** | Trace 解析失败时降级到简单计时 |

---

## 2. 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpRunner.run()                            │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  1. to_device_batch(input_tensors) → NPU tensors          │  │
│   │  2. _update_params(params, device_tensors)                │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│              ┌───────────────▼───────────────┐                   │
│              │    PerfEvaluator.run_profiled │                   │
│              │    ┌────────────────────────┐ │                   │
│              │    │ _prepare_warmup_tensors│ │ ← 10240×10240 fp16 │
│              │    │    MatMul + ReduceMax  │ │    升频清 Cache    │
│              │    └────────────────────────┘ │                   │
│              │              │                │                   │
│              │    ┌─────────▼─────────────┐  │                   │
│              │    │ torch_npu.profiler    │  │ ← Level1/Level2   │
│              │    │ schedule(warmup,repeat│  │                   │
│              │    │   + freq_boost per step│  │                   │
│              │    └─────────┬─────────────┘  │                   │
│              │              │                │                   │
│              │    ┌─────────▼─────────────┐  │                   │
│              │    │ kernel_details.csv    │  │ ← 47列详细数据    │
│              │    │    (Input Shapes)     │  │                   │
│              │    └─────────┬─────────────┘  │                   │
│              │              │                │                   │
│              │    ┌─────────▼─────────────┐  │                   │
│              │    │ _parse_kernel_details │  │ ← 精确形状匹配    │
│              │    │    + _is_warmup_kernel │  │   过滤 warmup     │
│              │    └─────────┬─────────────┘  │                   │
│              │              │                │                   │
│              │         PerfResult            │                   │
│              └──────────────┴────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 采集流程

### 3.1 主流程

```python
def run_profiled(case_id, func, *args, warmup=3, repeat=5):
    # 1. 准备升频清 Cache tensors
    if freq_boost:
        _prepare_warmup_tensors()  # MatMul + ReduceMax tensors

    # 2. 创建 profiling 目录
    prof_dir = f"reports/prof_data/{level}/{op_name}/{caseid}/"

    # 3. 执行 warmup + repeat 周期（Level1/Level2）
    with torch_npu.profiler.profile(
        schedule=schedule(wait=0, warmup=warmup, active=repeat),
        on_trace_ready=tensorboard_trace_handler(prof_dir),
        experimental_config=_ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,  # 默认 Level1
        ),
    ):
        for _ in range(warmup + repeat):
            _boost_freq_and_clear_cache()  # 每次调用前升频
            func(*args)
            prof.step()

    # 4. 定位并解析 kernel_details.csv（47列）
    csv_file = locate_kernel_details_csv(prof_dir)
    op_times, total_us = _parse_kernel_details_csv(csv_file)

    # 5. 按 repeat 次数归一化（使用中位数）
    elapsed_us = median(op_times.values())
    return outputs, PerfResult(elapsed_us, op_times)
```

### 3.2 时序图

```
时间 ─────────────────────────────────────────────────────────────▶

    ┌─────────────────────────────────────────────────────────────┐
    │ torch_npu.profiler.profile() 上下文                         │
    └─────────────────────────────────────────────────────────────┘
          │
          │  ┌────────────┐  ┌────────────┐  ┌────────────┐
          │  │  Warmup 1  │  │  Warmup 2  │  │  Warmup 3  │
          │  │────────────│  │────────────│  │────────────│
          │  │ boost_freq │  │ boost_freq │  │ boost_freq │
          │  │ func()     │  │ func()     │  │ func()     │
          │  │ prof.step()│  │ prof.step()│  │ prof.step()│
          │  └────────────┘  └────────────┘  └────────────┘
          │
          │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
          │  │  Active 1  │  │  Active 2  │  │  Active 3  │  │  Active 4  │  │  Active 5  │
          │  │────────────│  │────────────│  │────────────│  │────────────│  │────────────│
          │  │ boost_freq │  │ boost_freq │  │ boost_freq │  │ boost_freq │  │ boost_freq │
          │  │ func()     │  │ func()     │  │ func()     │  │ func()     │  │ func()     │
          │  │ prof.step()│  │ prof.step()│  │ prof.step()│  │ prof.step()│  │ prof.step()│
          │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
          │
          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ on_trace_handler → trace_view.json                          │
    └─────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ _parse_trace_file() → PerfResult                            │
    └─────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 PerfEvaluator

**职责**：NPU 性能采集与 Trace 解析。

**关键参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | False | 是否启用 Profiler |
| `warmup` | int | 3 | 预热次数 |
| `repeat` | int | 5 | 采集次数 |
| `archive_prof` | bool | True | 是否归档 profiling 数据 |
| `freq_boost` | bool | True | 是否启用升频清 Cache |

**核心方法**：

| 方法 | 职责 |
|------|------|
| `run_profiled()` | 主入口，执行 profiling 并返回结果 |
| `_prepare_warmup_tensors()` | 创建升频清 Cache 的 MatMul + ReduceMax tensors |
| `_boost_freq_and_clear_cache()` | 执行 MatMul + ReduceMax 升频清 L2 Cache |
| `_profile()` | 执行 torch_npu.profiler.profile() 上下文 |
| `_parse_trace_file()` | 解析 Chrome Trace JSON |
| `_normalize_result()` | 按 repeat 次数归一化结果 |

### 4.2 OpRunner

**职责**：算子执行与性能采集协调。

**核心方法**：

| 方法 | 职责 |
|------|------|
| `run()` | 执行算子，自动选择 Profiler 或简单计时 |
| `run_ai_op()` | 执行 AI 算子，支持 enable_perf 参数 |
| `_run_simple()` | 简单计时执行（Profiler 禁用时） |
| `_update_params()` | 将设备张量替换参数中的 CPU 张量引用 |

### 4.3 批量会话的内存保护与回退

三阶段评测在 correctness 阶段以 50 ms 周期采样每个 case 的进程
`VmRSS` 峰值，并随 case 结果序列化。性能阶段仅把峰值不超过
`perf_batch_case_memory_limit_mb` 的 case 放入共享 Profiler 会话；超限或
无法读取 `/proc/self/status` 的 case 仍参与性能评测，但使用独立会话。

若多个单独合格的 case 在共享会话内叠加后触发 OOM，父调度进程会按
`start_new_session=True` 建立的进程组同时清理 eval-child、msprof 和 parser
后代，丢弃该批不完整的性能结果，并把整批拆成单 case TaskUnit，以
`--no-perf-batch-cases` 各重试一次。这样 correctness 结果不会因性能采集
OOM 丢失，也不会遗留占用 NPU 的 Profiler 会话。

### 4.4 InputPool

**职责**：防止 data_ptr 缓存攻击。

**原理**：攻击者可能按 tensor.data_ptr() 缓存输出，相同输入地址返回预计算结果。InputPool 预分配多个 clone 输入，轮换使用，保证每次调用的 data_ptr 都不同。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_pool_size` | 8 | 最大池大小 |
| `max_memory_mb` | 512 | 最大内存占用（MB） |

**使用方式**：

```python
pool = InputPool(inputs, pool_size=warmup + repeat)
for _ in range(warmup + repeat):
    inputs = pool.get_next()  # 每次 data_ptr 不同
    output = func(*inputs)
```

---

## 5. Profiler 配置

### 5.1 torch_npu.profiler 参数

```python
# Level1（默认）或 Level2 配置
profiler_level = torch_npu.profiler.ProfilerLevel.Level1  # 默认
if config.profiler_level == "Level2":
    profiler_level = torch_npu.profiler.ProfilerLevel.Level2

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=profiler_level,
    aic_metrics=torch_npu.profiler.AiCMetrics.AicPipeUtilization,
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=warmup, active=repeat, repeat=1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    experimental_config=experimental_config,
) as prof:
    for _ in range(warmup + repeat):
        _boost_freq_and_clear_cache()
        func(*args)
        prof.step()
```

### 5.2 Profiler Level 选择

| Level | 数据量 | CSV 列数 | 适用场景 |
|-------|--------|----------|----------|
| **Level1** | 中等 | 47 | 默认级别，包含 Input Shapes 用于精确 warmup 过滤 |
| Level2 | 详细 | 47 | 全量采集，更详细的 AICPU 指标 |

**选择 Level1 的原因**：

1. 实验验证：Level1/Level2 与 Level0 耗时无差异
2. 提供 Input Shapes 列，支持精确形状匹配过滤 warmup kernel
3. 47 列数据维度丰富（包含 AICPU 等）
4. 不会产生 CANN parser ERROR（数据完整）

---

## 6. CSV 解析

### 6.1 kernel_details.csv 结构

Level1/Level2 profiling 自动产出 `kernel_details.csv`，包含 47 列详细数据：

**关键列**：

| 列名 | 说明 | 用途 |
|------|------|------|
| **OP Type** | 算子类型（如 MatMul, ReduceMax） | warmup 过滤 |
| **Input Shapes** | 输入形状（如 `"10240,10240;10240,10240"`） | **精确 warmup 过滤** |
| **Task Duration (us)** | NPU kernel 执行时间 | 核心测量对象 |
| **OP Name** | 具体算子名（如 aclnnExp） | 结果记录 |

### 6.2 Warmup Kernel 精确过滤

使用 OP Type + Input Shapes 精确匹配过滤 warmup kernels：

```python
# Warmup kernel 形状特征
WARMUP_MATMUL_SHAPE = '"10240,10240;10240,10240"'
WARMUP_REDUCE_SHAPE = '"96,1024,1024;3"'

def _is_warmup_kernel(op_type: str, input_shapes: str) -> bool:
    """精确形状匹配判断 warmup kernel"""
    if op_type == 'MatMul' and input_shapes == WARMUP_MATMUL_SHAPE:
        return True
    if op_type == 'ReduceMax' and input_shapes == WARMUP_REDUCE_SHAPE:
        return True
    return False
```

**优势**：
- 精确匹配，不会误过滤同名算子（如测试名为 "Max" 的算子）
- 基于 Input Shapes 列，数据可靠
- 替代了脆弱的名称子串匹配

### 6.3 解析逻辑

```python
def _parse_kernel_details_csv(csv_file):
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    
    device_kernels = {}
    durations = []
    
    for row in df.itertuples():
        op_type = getattr(row, 'OP Type', '')
        input_shapes = getattr(row, 'Input Shapes', '')
        duration = getattr(row, 'Task Duration (us)', 0)
        
        # 精确形状匹配过滤 warmup
        if _is_warmup_kernel(op_type, input_shapes):
            continue
        
        op_name = getattr(row, 'OP Name', op_type)
        device_kernels[op_name] = device_kernels.get(op_name, 0) + duration
        durations.append(duration)
    
    # 中位数统计（更稳定）
    elapsed_us = median(durations) if durations else 0
    return device_kernels, elapsed_us
```

---

## 7. 升频清 Cache

### 7.1 目的

每次测量前执行 MatMul + ReduceMax：

1. **升频**：将 NPU 频率提升到稳定状态
2. **清 Cache**：清空 L2 Cache，保证测量一致性

### 7.2 实现细节

```python
def _prepare_warmup_tensors():
    device = self.device_manager.get_device()
    # MatMul tensors: 10240×10240 fp16
    mm1 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
    mm2 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
    # Reduce tensor: 96×1024×1024 fp16
    reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).to(device)
    self._warmup_tensors = (mm1, mm2, reduce_input)

def _boost_freq_and_clear_cache():
    mm1, mm2, reduce_input = self._warmup_tensors
    torch.matmul(mm1, mm2)
    torch.npu.synchronize(mm1.device)  # 同步目标设备而非默认设备
    torch.max(reduce_input)
    torch.npu.synchronize(mm1.device)
```

### 7.3 设备同步

**设计要点**：同步目标设备（`mm1.device`）而非默认设备。

```python
torch.npu.synchronize(mm1.device)  # 正确：同步 warmup tensor 所在设备
torch.npu.synchronize()            # 错误：同步当前设备，可能与 warmup tensor 不同
```

**原因**：`.npu()` 无参数时迁移到当前设备（默认 0），可能导致 warmup tensor 与测试 tensor 在不同设备上。

## 8. CPU Fallback

当 Profiler 禁用或 NPU 不可用时，使用简单计时：

```python
def _measure_simple(func, warmup, repeat):
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Measure
    times = []
    for _ in range(repeat):
        torch.npu.synchronize() if npu_available
        t0 = time.perf_counter()
        func(*args)
        torch.npu.synchronize() if npu_available
        times.append((time.perf_counter() - t0) * 1_000_000)

    elapsed_us = sum(times) / len(times)
    return PerfResult(elapsed_us=elapsed_us)
```

**差异对比**：

| 方式 | 测量内容 | 精度 |
|------|----------|------|
| **Profiler** | NPU kernel-only | 高（排除 API 开销） |
| **Simple** | Wall-clock（含 API） | 低（作为兜底） |

---

## 9. 数据归档

### 9.1 目录结构

```
reports/prof_data/
├── level1/
│   ├── Exp/
│   │   ├── 1/
│   │   │   └── trace_view.json
│   │   ├── 2/
│   │   │   └── trace_view.json
│   │   └── ...
│   ├── Add/
│   └── ...
├── level2/
│   └── ...
└── level3/
    └── ...
```

### 9.2 case_id 解析

```python
def _parse_case_id(case_id):
    # "L2_Gcd_5" → ("level2", "Gcd", "5")
    m = re.match(r"^L(?P<level>\d+)_(?P<op>.+)_(?P<case>\d+)$", case_id)
    if not m:
        return "level_unknown", case_id, "0"
    return f"level{m['level']}", m["op"], m["case"]
```

---

## 10. 配置参数

| 参数 | 配置来源 | 默认值 | 说明 |
|------|----------|--------|------|
| `enable_profiler` | Config / CLI `--no-perf` | True | 是否启用 Profiler |
| `warmup` | Config / CLI `--warmup` | 3 | 预热次数 |
| `repeat` | Config / CLI `--repeat` | 5 | 采集次数 |
| `freq_boost` | PerfEvaluator 初始化 | True | 是否升频清 Cache |
| `archive_prof` | PerfEvaluator 初始化 | True | 是否归档 Trace 数据 |
| `use_input_pool` | run_profiled 参数 | False | 是否启用 InputPool |
| `perf_batch_cases` | Config / CLI `--perf-batch-cases` / `--no-perf-batch-cases` | True | 同一 TaskUnit 复用一次 Profiler；解析或反作弊归属不确定时逐 case 回退 |
| `perf_batch_case_memory_limit_mb` | Config / CLI `--perf-batch-case-memory-limit-mb` | 32768 | correctness 阶段单 case VmRSS 峰值上限；超限或采样缺失时不进入批量会话 |
| `profiler_level` | Config / CLI `--profiler-level` | `Level1` | Profiler 级别，可选 `Level1` / `Level2` |

> CLI 完整参数与默认值，见 [evaluator_design.md §3.3](./evaluator_design.md#33-命令行参数)。

---

## 11. 设计问题与改进方向

### 11.1 Warmup Kernel 过滤（已解决）

**历史问题**：使用名称子串匹配过滤 warmup kernels，可能误过滤同名算子。

**已实施改进**：升级到 Level1 profiling，使用 `kernel_details.csv` 的 Input Shapes 列精确形状匹配：

```python
WARMUP_MATMUL_SHAPE = '"10240,10240;10240,10240"'
WARMUP_REDUCE_SHAPE = '"96,1024,1024;3"'

def _is_warmup_kernel(op_type: str, input_shapes: str) -> bool:
    if op_type == 'MatMul' and input_shapes == WARMUP_MATMUL_SHAPE:
        return True
    if op_type == 'ReduceMax' and input_shapes == WARMUP_REDUCE_SHAPE:
        return True
    return False
```

**效果**：不再误过滤测试名为 "Max" 或 "Matmul" 的算子。

### 11.2 统计量改为中位数

**现状**：已从平均值改为中位数统计。

**原因**：中位数对异常值更稳定。

---

## 12. Level1 Profiling 升级（已完成）

### 12.1 升级内容


**已完成升级**（2026-04-30）：

| 变更项 | 原方案 | 新方案 |
|--------|--------|--------|
| Profiler Level | Level0 | Level1（默认）/ Level2（可选） |
| 数据源 | trace_view.json | kernel_details.csv（47列） |
| Warmup 过滤 | 名称子串匹配 | Input Shapes 精确形状匹配 |
| 统计量 | 平均值 | 中位数 |
| ERROR 抑制 | `_suppress_cann_profiler_errors()` | 已删除（Level1/Level2 无 ERROR） |

### 12.2 实验验证结论

| 指标 | Level0 | Level1 | Level2 |
|------|---------|---------|---------|
| 耗时 | 5.69s | 5.67s | 5.69s |
| CSV 列数 | 9 | 47 | 47 |
| Input Shapes | ❌ 无 | ✓ 有 | ✓ 有 |

**结论**：Level1/Level2 耗时无增加，提供丰富数据用于精确 warmup 过滤。

### 12.3 配置参数

profiler 相关配置见 §10；CLI 完整参数表（多卡 / 子进程隔离 / Profiler 级别等）见 [evaluator_design.md §3.3](./evaluator_design.md#33-命令行参数)。

---

## 13. CLI 入口统一

### 13.1 职责划分

评测入口统一后，各脚本职责明确：

| 脚本 | 职责 | 独特能力 |
|------|------|----------|
| `src/kernel_eval/cli.py` | **核心评测入口** | 所有评测能力：`--task-dir`、多卡并行、`--no-perf`、`--source-dir` |
| `scripts/utils/build_golden_wheel.sh` | **Golden whl 打包** | 将 golden 函数打包成 cann_bench whl，注册到 torch.ops.cann_bench |
| `scripts/run_evaluation.sh` | CLI 封装 | 无（纯参数转发） |

### 13.2 CLI 核心能力

`python -m kernel_eval.cli eval` 主要参数（完整表见 [evaluator_design.md §3.3](./evaluator_design.md#33-命令行参数)）：

**目录与筛选**：
- `--task-dir <path>`：评测目录（`tasks` / `tasks/level1` / `tasks/level1/exp` 等）
- `--operator <name>`：按算子名称筛选
- `--case-id <id>`：按用例编号筛选

**设备配置**：
- `--device cpu|npu`：设备类型（默认 npu）
- `--device-id <id>`：单卡模式；不指定则多卡并行
- `--processes-per-card <n>`：多卡并行时每卡进程数（默认 2）
- `--timeout-per-operator <n>`：多卡并行下单算子超时（秒，默认 300）

**性能配置**：
- `--warmup <n>`：预热次数（默认 3）
- `--repeat <n>`：采集次数（默认 5）
- `--no-perf`：关闭性能采集
- `--perf-batch-cases`：合并同一 TaskUnit 的 case 到一个 Profiler 会话（默认开启）
- `--no-perf-batch-cases`：关闭批量 case，恢复逐 case Profiler 会话
- `--profiler-level Level1|Level2`：Profiler 级别（默认 Level1）

**源码评测**：
- `--source-dir <dir>`：AI 生成算子源码目录（自动编译安装）

### 13.3 多卡并行模式

CLI 自动判断多卡并行模式：

```python
# 多卡并行条件：NPU + 未指定 device_id + 无 source_dir（Golden 模式）
use_multi_card = (
    args.device == 'npu'
    and args.device_id is None
    and not args.source_dir
)
```

多卡并行使用 `ProcessPoolCoordinator`：
- 每张 NPU 卡运行 `processes_per_card` 个进程
- 进程间无通信，通过文件传递结果
- 每进程独立初始化 `torch_npu.profiler`

### 13.4 Golden 自验证

Golden 自验证（验证 golden(NPU) 与 golden(CPU fp64) 精度一致性）通过 `build_golden_wheel.sh` 打包 golden 函数为 `cann_bench` whl，再用 `run_evaluation.sh` 评测：

```bash
# 构建 golden whl 并安装
./scripts/utils/build_golden_wheel.sh --install

# 评测（golden 作为 AI 算子，golden(CPU fp64) 作为参考）
./scripts/run_evaluation.sh --task-dir tasks/level1/exp --no-perf
```

---

## 14. 相关文件

| 文件 | 职责 |
|------|------|
| `src/kernel_eval/eval/perf_eval.py` | 性能采集核心实现 |
| `src/kernel_eval/eval/op_runner.py` | 算子执行与性能协调 |
| `src/kernel_eval/eval/input_pool.py` | InputPool 防缓存攻击 |
| `src/kernel_eval/utils/device_manager.py` | 设备管理与同步 |
| `src/kernel_eval/config.py` | 全局配置（warmup/repeat 等） |
