# 算子评测工程设计

## 目录
- [1. Context](#1-context)
- [2. 方案设计](#2-方案设计)
- [3. 源码目录评测流程](#3-源码目录评测流程)
- [4. 核心能力设计](#4-核心能力设计)
- [5. 实施步骤](#5-实施步骤)
- [6. 验证方案](#6-验证方案)
- [7. 附录](#7-附录)

---

## 1. Context

### 1.1 背景

根据 `docs/spec/benchmark_spec.md` 设计文档，构建一套AI生成Ascend C算子代码评测体系，用于量化评估AI生成的算子代码质量，涵盖编译正确性、功能正确性、性能优化性三个核心维度。

### 1.2 两工程架构设计

本评测体系分为两个独立工程，通过whl包进行传递：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI自动生成工程                                     │
│  参考 examples/fast_kernel_launch_example 结构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  project_root/                                                              │
│  ├── setup.py              # 构建配置（PyTorch Extension）                  │
│  ├── CMakeLists.txt        # CMake构建文件                                  │
│  ├── build.sh              # 编译脚本（可选）                                │
│  ├── cann_bench/           # Python包（约定命名）                            │
│  │   └── __init__.py       # 导入_C扩展模块                                 │
│  ├── csrc/                 # 算子源码目录                                    │
│  │   ├── exp/              # Exp算子                                        │
│  │   │   └ ascend910b/                                                     │
│  │   │     ├── CMakeLists.txt                                              │
│  │   │     └── exp.cpp      # AI生成的Ascend C代码                         │
│  │   └── ...                                                               │
│  └ dist/                   # 构建产物                                       │
│  │   ├── cann_bench_xxx.whl  # Python包                                   │
│  │   └── cann_bench_xxx.run   # NPU内核包（可选）                          │
│                                                                             │
│  算子接口约定：torch.ops.cann_bench.exp(x, ...) 或 cann_bench.exp()        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ cann_bench_xxx.whl + cann_bench_xxx.run
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           评测工程（src/kernel_eval）                         │
│  本方案设计目标                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  src/kernel_eval/                                                            │
│  ├── cli.py                # 命令行入口                                     │
│  ├── config.py             # 配置管理                                       │
│  ├── data/                 # 数据层（加载kernel_bench用例）                 │
│  ├── eval/                 # 评测层（精度+性能+安全验证）                   │
│  ├── report/               # 报告层（JSON+Markdown+Summary）               │
│  ├── security/             # 安全层（防篡改检查）                           │
│  └── utils/                # 工具层                                         │
│                                                                             │
│  评测流程：                                                                  │
│  1. 安全初始化（Timing API快照）                                            │
│  2. 扫描源码目录（检查build.sh、dist目录）                                   │
│  3. 检查dist是否有whl包/run包，无则执行build.sh编译                          │
│  4. 安装run包（NPU内核包）+ whl包（Python包）                                │
│  5. 安全验证（Timing API完整性检查）                                        │
│  6. 扫描cann_bench接口，打印接口信息                                         │
│  7. 加载 kernel_bench 用例数据                                              │
│  8. 执行精度验证（CPU fp64 Golden + 二次验证）                              │
│  9. 执行性能评测（Profiler kernel-only + 升频清cache）                      │
│  10. 生成评测报告                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键约定**：
1. AI生成工程包名统一为 `cann_bench`
2. 算子接口与proto.yaml的schema一致
3. 评测工程通过 `--source-dir` 参数指定源码目录，自动扫描编译安装whl包和run包
4. 安全机制防止作弊攻击
5. 高精度Golden计算（CPU fp64）

**命名说明**：
- `kernel_eval`：评测工程代码目录（src/kernel_eval）
- `kernel_bench`：测试用例数据目录（kernel_bench/level*/op_name/）
- `./scripts/run_evaluation.sh`：CLI命令脚本（推荐使用）

---

## 2. 方案设计

### 2.1 工程架构

```
src/kernel_eval/
├── __init__.py              # 包入口，导出公共API
├── cli.py                   # 命令行入口
├── config.py                # 配置管理（含多硬件baseline）
│
├── data/                    # 数据层
│   ├── __init__.py
│   ├── operator_loader.py   # 算子定义加载（proto.yaml解析）
│   ├── case_loader.py       # 测试用例加载
│   ├── golden_loader.py     # Golden函数加载
│   ├── data_generator.py    # 数据生成（含特殊值、tensor list）
│   └── package_manager.py   # 包管理（源码扫描、编译、安装、接口扫描）
│
├── eval/                    # 评测层
│   ├── __init__.py
│   ├── accuracy_eval.py     # 功能精度评测（CPU fp64 Golden + 二次验证）
│   ├── perf_eval.py         # 性能评测（Profiler kernel-only + 升频清cache）
│   ├── op_runner.py         # 算子执行器（返回值检查）
│   ├── evaluator.py         # 综合评测调度器
│   └── input_pool.py        # 输入池管理（防缓存攻击）
│
├── security/                # 安全层
│   ├── __init__.py
│   ├── api_guard.py         # Timing API防护（快照+验证+恢复）
│   └── type_checker.py      # 返回值类型检查
│
├── report/                  # 报告层
│   ├── __init__.py
│   ├── report_generator.py  # 评测报告生成器（JSON + Markdown）
│   ├── summary_generator.py # Summary生成（几何平均加速比）
│   └── scoring.py           # 评分计算
│
├── utils/                   # 工具层
│   ├── __init__.py
│   ├── device_manager.py    # 设备管理（CPU/NPU）
│   ├── dtype_mapper.py      # 数据类型映射
│   ├── param_builder.py     # 参数构建（函数签名解析）
│   ├── precision.py         # 精度验证工具（MERE/MARE）
│   └── baseline_resolver.py # Baseline解析（多硬件支持）
```

### 2.2 核心模块职责

#### 2.2.1 数据层（data/）

| 模块 | 职责 |
|------|------|
| `operator_loader.py` | 解析proto.yaml，提供算子schema、attrs、inputs、outputs信息 |
| `case_loader.py` | 扫描cases.yaml，返回CaseInfo数据结构 |
| `golden_loader.py` | 动态导入golden函数，支持PascalCase→snake_case转换 |
| `data_generator.py` | 根据shape/dtype/value_range生成输入张量，支持特殊值 |
| `package_manager.py` | 扫描源码目录、检查/编译whl/run包、安装包、扫描接口 |

#### 2.2.2 评测层（eval/）

| 模块 | 职责 |
|------|------|
| `accuracy_eval.py` | CPU fp64 Golden计算、MERE/MARE精度验证、二次验证 |
| `perf_eval.py` | Profiler kernel-only测量、NPU升频清L2 cache、Trace解析 |
| `op_runner.py` | 算子执行、返回值类型检查、设备迁移 |
| `input_pool.py` | 预分配clone输入池，防止data_ptr缓存攻击 |
| `evaluator.py` | 综合调度，协调精度和性能评测 |

#### 2.2.3 安全层（security/）

| 模块 | 职责 |
|------|------|
| `api_guard.py` | Timing API快照+验证+恢复，防止monkey-patch攻击 |
| `type_checker.py` | 严格类型检查（type() is torch.Tensor），拒绝FakeTensor |

#### 2.2.4 报告层（report/）

| 模块 | 职责 |
|------|------|
| `report_generator.py` | JSON + Markdown双格式报告生成 |
| `summary_generator.py` | Summary生成，几何平均加速比计算 |
| `scoring.py` | 功能得分+性能得分计算 |

---

## 3. 源码目录评测流程

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           源码目录评测流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: --source-dir /path/to/ai_generated_ops                               │
│                                                                             │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │ 扫描源码   │────▶│ 检查dist   │────▶│ 编译whl    │────▶│ 安装包     │   │
│  │ 目录结构   │     │ whl+run包  │     │ (build.sh) │     │ (whl+run)  │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│        │                  │                  │                  │          │
│        │           有包则跳过编译        无dist则编译           │          │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │ 扫描模块   │────▶│ 打印接口   │────▶│ 匹配用例   │────▶│ 执行评测   │   │
│  │ 接口列表   │     │ 信息       │     │ (level/op) │     │            │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│                                                                             │
│  输出: 评测报告 (JSON + Markdown + Summary)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 详细步骤

**Step 1: 扫描源码目录结构**

检查源码目录是否存在以下结构：
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

**Step 2: 检查dist目录**

扫描 `dist/` 目录，查找：
- `cann_bench_xxx.whl` - Python包（必须）
- `cann_bench_xxx.run` - NPU内核包（可选）

如果存在这些包，跳过编译步骤；否则检查是否有 `build.sh` 并执行编译。

**Step 3: 编译whl/run包（如果需要）**

执行 `build.sh` 脚本：
```bash
cd source_dir && bash build.sh
```

**迭代隔离编译**（默认开启）：如果编译失败，系统会自动识别并隔离编译不过的算子到 `_quarantine/` 目录，然后对剩余的算子重新执行编译和评测。这可以确保部分算子编译失败不会导致整个评测任务失败。

编译后检查 `dist/` 目录是否生成whl包和run包。

**Step 4: 安装包**

安装顺序：先安装run包，再安装whl包。

**安装run包（NPU内核包）**：
```bash
chmod +x dist/cann_bench_xxx.run
./dist/cann_bench_xxx.run --install
```

**安装whl包（Python包）**：
```bash
# 先卸载旧版本，再安装新版本（不使用force-reinstall，避免重装依赖）
pip uninstall cann_bench -y
pip install dist/cann_bench_xxx.whl
```

**Step 5: 扫描模块接口**

导入cann_bench模块，扫描提供的算子接口：
```python
import cann_bench

interfaces = []
for name in dir(cann_bench):
    if not name.startswith('_'):
        attr = getattr(cann_bench, name)
        if callable(attr):
            interfaces.append(name)

# 同时扫描 torch.ops.cann_bench
import torch
if hasattr(torch.ops, 'cann_bench'):
    for name in dir(torch.ops.cann_bench):
        if not name.startswith('_'):
            interfaces.append(name)
```

**Step 6: 打印接口信息**

显示扫描到的接口：
```
============================================================
扫描到的 cann_bench 接口:
============================================================
  1. exp(x, base=-1.0, scale=1.0, shift=0.0) -> Tensor

共 1 个算子接口
============================================================
```

**Step 7: 匹配用例**

根据接口名称，匹配kernel_bench中的算子用例：
- 查找 proto.yaml 中对应的算子定义
- 加载对应的 cases.yaml 用例

**Step 8: 执行评测**

**子进程隔离**（默认开启）：每个算子在独立的子进程中执行评测，避免一个算子的挂死或崩溃影响其他算子的评测。每个子进程都有独立的超时控制（默认240秒），超时后会先发送SIGTERM信号，10秒宽限期后发送SIGKILL信号强制终止。

对每个匹配到的算子执行评测：
1. 安全验证（Timing API完整性）
2. 加载用例数据
3. 执行Golden函数（CPU fp64）
4. 执行AI算子（返回值类型检查）
5. 精度验证（MERE/MARE）
6. 二次验证（新鲜输入重跑）
7. 性能评测（Profiler kernel-only）
8. 计算评分

### 3.3 命令行参数

#### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-a, --action <action>` | 操作类型: eval(评测), list(列表), info(详情), config(配置) | eval |
| `-v, --verbose` | 详细输出 | False |

#### 评测(eval)相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source-dir <dir>` | AI生成的算子源码目录（自动扫描编译安装） | 无 |
| `-l, --level <level>` | 算子难度级别 (1/2/3/4) | 无 |
| `-o, --operator <name>` | 算子名称 (如 Exp, Softmax) | 无 |
| `-c, --case-id <id>` | 用例编号 | 无 |
| `--no-subprocess-isolation` | 关闭子进程隔离（默认开启）。开启后每个算子在独立子进程评测，一个kernel挂死/崩溃不会污染后面的算子 | False |
| `--op-timeout-sec` | 子进程隔离下 per-op 超时。超时先 SIGTERM，10s 宽限后 SIGKILL | 240秒 |
| `--no-iterative-compile` | 关闭迭代隔离编译（默认开启）。开启时build.sh失败会自动识别并隔离编译不过的算子到_quarantine/，剩下的算子继续编译和评测 | False |

当不指定 `--source-dir` 时，默认跳过编译安装，直接使用已安装的cann_bench模块。

#### 列表(list)相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-l, --level <level>` | 按级别筛选 (1/2/3/4) | 无 |
| `-o, --operator <name>` | 按算子筛选 | 无 |

#### 详情(info)相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o, --operator <name>` | 算子名称（必填） | 无 |
| `-l, --level <level>` | 难度级别 | 无 |

#### 配置(config)相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 无 | 显示当前配置 | - |

### 3.4 使用示例

```bash
# 从源码目录评测（自动编译安装）
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops

# 使用子进程隔离评测（默认开启）
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops

# 关闭子进程隔离以提高速度
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops --no-subprocess-isolation

# 设置算子超时时间
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops --op-timeout-sec 300

# 关闭迭代隔离编译（严格模式）
./scripts/run_evaluation.sh --action eval --source-dir /path/to/ai_ops --no-iterative-compile

# 仅执行指定算子的评测
./scripts/run_evaluation.sh --action eval --operator Exp --level 1

# 评测单个用例
./scripts/run_evaluation.sh --action eval --operator Exp --level 1 --case-id 1

# 列出所有level 1的算子
./scripts/run_evaluation.sh --action list --level 1

# 列出指定算子的所有用例
./scripts/run_evaluation.sh --action list --operator Exp

# 查看算子详情
./scripts/run_evaluation.sh --action info --operator Exp

# 显示当前配置
./scripts/run_evaluation.sh --action config
```

### 3.5 异常处理

| 场景 | 处理方式 |
|------|----------|
| 源码目录不存在 | 报错退出 |
| 无build.sh且无dist | 报错退出（无法编译） |
| build.sh执行失败 | 报错退出 |
| run包安装失败 | 报错退出（NPU模式必须） |
| whl包安装失败 | 报错退出 |
| cann_bench导入失败 | 报错退出 |
| 未匹配到算子 | 警告并退出 |
| Timing API被篡改 | 恢复原始API后报错退出 |

---

## 4. 核心能力设计

### 4.1 安全防护设计

#### 4.1.1 Timing API防护

**原理**：在submission代码运行前，快照关键Timing API的身份；安装wheel后验证是否被篡改。

**关键API列表**：
- `torch.npu.Event.elapsed_time`
- `torch.npu.Event.record`
- `torch.npu.synchronize`
- `torch_npu.profiler.profile`
- `torch_npu.profiler.schedule`

**防护流程**：
```python
api_guard = APIGuard()
api_guard.snapshot()           # 1. 安装wheel前快照
install_wheel(path)            # 2. 安装submission
api_guard.verify()             # 3. 验证完整性
# ... 执行评测 ...
api_guard.restore()            # 4. 程序退出前恢复
```

#### 4.1.2 返回值类型检查

**原理**：使用 `type(output) is torch.Tensor` 严格检查，拒绝FakeTensor等子类伪装。

```python
def check_output_type(output):
    if type(output) is not torch.Tensor:
        raise RuntimeError("算子必须返回torch.Tensor，拒绝FakeTensor/懒求值包装器")
```

#### 4.1.3 二次验证机制

**原理**：用新鲜输入重跑一次，防止缓存作弊。如果算子缓存第一次结果，第二次用不同输入会产生错误结果。

```python
# 第一轮验证
result1 = evaluate(golden_fn, custom_fn, inputs)

# 第二轮验证（新鲜输入+微扰）
fresh_inputs = generate_inputs(case)
perturb_inputs(fresh_inputs)  # 添加0.01微扰
result2 = evaluate(golden_fn, custom_fn, fresh_inputs)
```

#### 4.1.4 输入池防缓存

**原理**：预分配一组clone输入，轮换使用，每次调用data_ptr不同，防止按地址缓存输出。

```python
pool = InputPool(inputs, pool_size=warmup+repeat)
for _ in range(warmup + repeat):
    inputs = pool.get_next()  # 每次data_ptr不同
    output = fn(*inputs)
```

### 4.2 精度验证设计

#### 4.2.1 CPU fp64 Golden计算

**原理**：Golden函数在CPU fp64精度下计算，比NPU原生dtype精度更高，避免溢出/下溢污染参考值。精度对比时双方都cast回fp32计算MERE/MARE。

```python
def compute_golden_fp64(golden_fn, inputs, param_builder, case):
    fp64_inputs = [t.cpu().double() for t in inputs]
    params = param_builder.build(golden_fn, case, fp64_inputs)
    with torch.no_grad():
        return golden_fn(**params)
```

#### 4.2.2 精度标准实现

精度标准（MERE/MARE误差指标、精度阈值表、小值域通过标准）已在 [benchmark_spec.md](../spec/benchmark_spec.md) 中定义。本工程实现精度验证流程：

```python
def verify_accuracy(actual, golden, dtype):
    # 计算MERE/MARE
    mere = compute_mere(actual, golden)
    mare = compute_mare(actual, golden)
    
    # 获取精度阈值
    threshold = get_precision_threshold(dtype)
    
    # 判断是否通过
    if mere < threshold and mare < 10 * threshold:
        return True
    
    # 小值域检查（当golden接近0时）
    return check_small_value_region(actual, golden, dtype)
```

### 4.3 性能评测设计

#### 4.3.1 Profiler Kernel-Only测量

**原理**：使用 `torch_npu.profiler` 采集NPU端chrome trace，使用 `ProfilerLevel.Level0` 采集最小开销的 kernel 执行时间。通过解析 `cat="dequeue"` 事件获取 NPU 内核执行时间，自动过滤升频用的 MatMul/ReduceMax kernel。

```python
def measure_kernel_us(fn, warmup=3, repeat=5, freq_boost=True):
    import torch_npu
    
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
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
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(warmup + repeat):
            if freq_boost:
                boost_freq_and_clear_cache()  # 升频清cache
            fn()
            prof.step()
    
    # 解析trace，提取 cat="dequeue" 事件（NPU内核执行时间）
    # 过滤 warmup kernels (matmul, max, reducemax, reduced)
    return parse_trace_dequeue_events(trace_file) / repeat
```

**Trace解析逻辑**：

```python
# Warmup kernel关键词（需过滤）
warmup_keywords = ("matmul", "max", "reducemax", "reduced")

for event in events:
    if event.get('ph') != 'X':
        continue
    dur = event.get('dur', 0)
    if dur <= 0:
        continue
    name = event.get('name', '')
    
    # cat="dequeue" 是 CANN 权威分类，表示 NPU 内核执行时间
    if event.get('cat') == 'dequeue':
        # 过滤 warmup kernels
        if any(kw in name.lower() for kw in warmup_keywords):
            continue
        device_kernels[name] = device_kernels.get(name, 0) + dur
        total_kernel_us += dur
    
    # cat="cpu_op" 表示 CPU 侧 API 调用时间（仅供参考）
    elif event.get('cat') == 'cpu_op':
        if any(kw in name.lower() for kw in warmup_keywords):
            continue
        host_ops[name] = host_ops.get(name, 0) + dur
```

#### 4.3.2 NPU升频与L2清空

**原理**：每次profiler step前执行MatMul+ReduceMax，保证NPU频率稳定并清空L2 cache，确保测量一致性。Warmup tensors预分配并固定到目标设备，避免设备不匹配。

```python
def prepare_warmup_tensors(device):
    """预分配升频清cache的tensors"""
    mm1 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
    mm2 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
    reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).to(device)
    return (mm1, mm2, reduce_input)

def boost_freq_and_clear_cache(warmup_tensors):
    """NPU升频 + 清L2 cache"""
    mm1, mm2, reduce_input = warmup_tensors
    torch.matmul(mm1, mm2)
    torch.npu.synchronize(mm1.device)  # 同步目标设备
    torch.max(reduce_input)
    torch.npu.synchronize(mm1.device)
```

#### 4.3.3 InputPool防缓存攻击

**原理**：预分配一组clone输入，轮换使用，每次调用data_ptr不同，防止按地址缓存输出。

```python
class InputPool:
    """预分配clone输入池，防止data_ptr缓存攻击"""
    
    def __init__(self, inputs, pool_size, max_memory_mb=512):
        self.pool = []
        # 根据内存限制计算实际池大小
        actual_size = min(pool_size, max_pool_size, max_sets_by_memory)
        # 预分配clone池
        for _ in range(actual_size):
            cloned = [item.clone() if isinstance(item, torch.Tensor) else item 
                      for item in inputs]
            self.pool.append(cloned)
    
    def get_next(self):
        """获取下一个输入集（每次data_ptr不同）"""
        inputs = self.pool[self.idx % len(self.pool)]
        self.idx += 1
        return inputs
```

#### 4.3.4 多硬件Baseline支持

**cases.yaml格式**：
```yaml
cases:
  - case_id: 1
    baseline_perf_us: 40.2            # 单硬件baseline（默认910b2）
    
  - case_id: 2
    baseline_perf_us:                  # 多硬件baseline字典
      910b2: 40.2
      910b1: 45.1
      910a: 50.0
```

### 4.4 报告生成设计

#### 4.4.1 几何平均加速比

使用几何平均计算多个case的加速比：

```python
def geometric_mean_speedup(speedups):
    if not speedups:
        return 0.0
    return math.exp(sum(math.log(max(s, 1e-9)) for s in speedups) / len(speedups))
```

#### 4.4.2 报告输出格式

**JSON格式**：
```json
{
  "hardware": "910b2",
  "total_operators": 1,
  "total_cases": 20,
  "total_passed": 18,
  "overall_geometric_mean_speedup": 1.25,
  "operators": [...]
}
```

**Markdown格式**：包含表格摘要和详细用例结果。

**Summary格式**：
```markdown
# 算子评测报告
**评测代号**: eval_20260422_xxx
**硬件**: 910b2

## 总体结果
- **通过率**: 90.00%
- **几何平均加速比**: 1.25x
```

---

## 5. 实施步骤

### Phase 1：目录重命名与结构调整

1. 将 `src/kernel_bench/` 重命名为 `src/kernel_eval/`
2. 创建 `security/` 目录
3. 创建 `eval/input_pool.py`
4. 创建 `utils/baseline_resolver.py`
5. 创建 `report/summary_generator.py`
6. 更新所有导入路径

### Phase 2：安全层实现

1. 实现 `security/api_guard.py`（Timing API防护）
2. 实现 `security/type_checker.py`（返回值类型检查）
3. 在 `evaluator.py` 中集成安全检查流程
4. 测试安全防护有效性

### Phase 3：精度验证增强

1. 修改 `accuracy_eval.py` 支持CPU fp64 Golden计算
2. 实现二次验证机制
3. 实现 `input_pool.py` 输入池管理
4. 验证精度测试结果正确性

### Phase 4：性能评测增强

1. 实现Profiler kernel-only测量逻辑
2. 实现NPU升频+清L2 cache逻辑
3. 实现Trace解析逻辑
4. 实现 `baseline_resolver.py` 多硬件支持
5. 验证性能测量准确性

### Phase 5：报告层完善

1. 实现几何平均加速比计算
2. 实现summary生成器
3. 完善JSON和Markdown输出格式
4. 验证报告输出完整性

### Phase 6：CLI与集成

1. 更新CLI命令适配新模块路径
2. 实现完整评测流程串联
3. 编写使用文档
4. 集成测试

---

## 6. 验证方案

### 6.1 安全验证

| 测试项 | 验证方法 |
|--------|----------|
| API篡改检测 | 模拟monkey-patch，验证检测并恢复 |
| 返回值检查 | 返回FakeTensor，验证拒绝 |
| 二次验证 | 缓存作弊实现，验证二次失败 |
| 输入池轮换 | 验证每次调用data_ptr不同 |

### 6.2 精度验证

| 测试项 | 验证方法 |
|--------|----------|
| CPU fp64 Golden | 对比fp32和fp64结果差异 |
| MERE/MARE计算 | 手动计算验证 |
| NaN/Inf处理 | 特殊值用例验证 |
| 整数精确匹配 | 整型用例验证 |

### 6.3 性能验证

| 测试项 | 验证方法 |
|--------|----------|
| Profiler kernel-only | 对比wall-clock和profiler时间 |
| 升频清cache效果 | 对比有无预热的时间稳定性 |
| Trace解析 | 检查解析结果与trace内容一致 |
| 多硬件baseline | 验证不同硬件baseline解析 |

### 6.4 集成验证

- 运行完整评测流程
- 验证JSON/Markdown/Summary报告
- 验证几何平均加速比计算
- 验证安全机制在整个流程中有效

---

## 7. 附录

### 7.1 参考文档

- `docs/spec/benchmark_spec.md`：算子代码生成评测基准规范
- `docs/design/evaluator_design.md`：评测工程设计文档（本文档）
- `docs/guide/quick_start.md`：快速入门指南
- `docs/changelog.md`：版本变更记录
- `../opbase/docs/zh/ops_precision_standard/experimental_standard.md`：精度标准

### 7.2 包命名约定

| 包类型 | 命名格式 | 说明 |
|---------|---------|------|
| whl包 | `cann_bench_xxx.whl` | Python包，包含算子接口 |
| run包 | `cann_bench_xxx.run` | NPU内核二进制包 |

### 7.3 评分公式

```
编译通过得分 = compile_pass × Wc  (Wc=2，compile_pass ∈ {0, 1}，整份提交编译是否通过，与用例数无关)
功能得分     = case_pass × Wf     (Wf=3，case_pass ∈ {0, 1}，单个用例是否通过精度校验)
性能得分     = SpeedUp × Wp       (Wp=5，仅对功能通过的用例计入，SpeedUp 按该用例实测)

单算子综合评分 = 编译通过得分 + Σ_{功能通过的用例 i} (Wf + SpeedUp_i × Wp)

Level-N 得分   = Σ 该 level 内算子综合评分
benchmark 总分 = Σ 所有算子综合评分 (= Level1 + Level2 + Level3 + Level4)
```

### 7.4 版本演进

详细版本变更记录请参阅 [docs/changelog.md](../changelog.md)。