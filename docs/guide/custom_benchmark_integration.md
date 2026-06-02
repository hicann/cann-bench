# 自定义评测集接入指南

## 概述

cann-bench 支持接入其他评测体系。本文档介绍如何接入自定义评测集。

**重要说明**：
- cann-bench 是针对 **CANN/NPU** 的评测体系，性能采集使用 `torch_npu.profiler`
- 仅支持 **Ascend NPU** 后端，不支持 CUDA 或其他硬件平台
- 接入其他评测集时，正确性验证和评分计算可适配，但性能采集必须在 NPU 上执行

## 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│  BenchRegistry Layer（评测集配置）                            │
│  ├─ BenchConfig（统一配置）                                   │
│  └─ 自动组装：Loader + ScoringScheme + Checker               │
├─────────────────────────────────────────────────────────────┤
│  Base Layer（基类定义）                                       │
│  ├─ models.py: TaskSpec / CaseSpec / InputSpec / OutputSpec  │
│  ├─ result.py: AccuracyResult / PerfResult                   │
│  ├─ loaders.py: TaskLoader / CaseLoader / GoldenLoaderBase   │
│  ├─ checker.py: CorrectnessChecker (基类)                    │
│  ├─ matcher.py: OperatorMatcherBase                          │
│  └─ scoring.py: ScoringScheme (基类)                         │
├─────────────────────────────────────────────────────────────┤
│  Benches Layer（CANN 特化）                                   │
│  ├─ cann_loader.py: CannTaskLoader / CannCaseLoader          │
│  ├─ cann_spec.py: CannTaskSpec / CannCaseSpec                │
│  ├─ relative_error_checker.py: RelativeErrorChecker / RelativeErrorOutputResult   │
│  ├─ cann_matcher.py: OperatorMatcher                         │
│  ├─ cann_scoring.py: CannScoringScheme / SOL-Score           │
│  └─ cann.py: 导出 + Registry 注册                            │
├─────────────────────────────────────────────────────────────┤
│  Registry Layer（注册机制）                                   │
│  ├─ LoaderRegistry: TaskLoader / CaseLoader 注册             │
│  ├─ GoldenLoaderRegistry: GoldenLoader 注册                  │
│  ├─ OperatorMatcherRegistry: Matcher 注册                    │
│  ├─ CheckerRegistry: Checker 注册                            │
│  ├─ ScoringSchemeRegistry: ScoringScheme 注册                │
│  └─ BenchRegistry: 评测集配置聚合                             │
├─────────────────────────────────────────────────────────────┤
│  Eval Layer（评测执行 - 仅NPU）                               │
│  ├─ PerfEvaluator (性能采集 - torch_npu.profiler)             │
│  ├─ CorrectnessChecker (精度判断 - 可插拔)                     │
│  └─ Evaluator (协调器)                                         │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
src/kernel_eval/
├── base/                  # 基类层
│   ├── models.py          # TaskSpec, CaseSpec, InputSpec, OutputSpec, SolutionSpec
│   ├── result.py          # AccuracyResult, PerfResult, OutputResult
│   ├── loaders.py         # TaskLoader, CaseLoader, GoldenLoaderBase, OperatorDirMixin
│   ├── checker.py         # CorrectnessChecker (基类)
│   ├── matcher.py         # OperatorMatcherBase (基类)
│   ├── scoring.py         # ScoringScheme (基类)
│   └── enums.py           # 枚举定义
│
├── benches/               # CANN 特化层（扁平结构）
│   ├── cann_loader.py     # CannTaskLoader, CannCaseLoader, GoldenLoader
│   ├── cann_spec.py       # CannTaskSpec, CannCaseSpec, CannInputSpec, CannOutputSpec
│   ├── cann_solution.py   # CannSolutionSpec
│   ├── relative_error_checker.py    # RelativeErrorChecker, RelativeErrorOutputResult
│   ├── cann_matcher.py    # OperatorMatcher
│   ├── cann_scoring.py    # CannScoringScheme, SimpleComparisonScheme, RecordingOnlyScheme
│   ├── cann.py            # 导出所有 CANN 组件 + Registry 注册
│   └── __init__.py        # 聚合导入
│
├── registry/              # 注册机制
│   ├── loader_registry.py
│   ├── golden_registry.py
│   ├── matcher_registry.py
│   ├── checker_registry.py
│   ├── scoring_registry.py
│   └── bench_registry.py
│
├── data/                  # 数据工具层
│   ├── data_generator.py  # DataGenerator
│   └── package_manager.py # PackageManager
│
├── eval/                  # 评测执行层
│   ├── evaluator.py
│   ├── op_runner.py
│   ├── accuracy_eval.py
│   ├── perf_eval.py
│   └── results.py
│
├── report/                # 报告生成层
│   └── report_generator.py
│
├── utils/                 # 工具层
│   ├── precision.py
│   ├── tensor_utils.py
│   └── ...
│
└── cli.py                 # CLI 入口
```

## BenchRegistry：统一配置机制

通过 BenchRegistry，只需指定 `--bench-name` 即可自动组装所有依赖组件：

```python
from kernel_eval.registry.bench_registry import BenchRegistry, BenchConfig

# 注册自定义评测集
BenchRegistry.register('my_bench', BenchConfig(
    task_loader='my_task_loader',
    case_loader='my_case_loader',
    scoring_scheme='simple_comparison',
    checker='relative_error',
    golden_precision='fp64_cpu',  # Golden 精度策略（见下文）
    precision_thresholds={'float16': 0.001, 'float32': 0.0001},
    description='自定义 NPU 评测集',
))
```

**CLI 使用**：

```bash
# 使用评测集（自动加载所有配置）
kernel-bench eval --bench-name my_bench

# 使用默认 CANN 评测集
kernel-bench eval --bench-name cann
```

### golden_precision 精度策略

`BenchConfig.golden_precision` 控制 Golden 参考输出的计算精度和设备。策略通过 `Evaluator._apply_golden_precision()` 在 golden 执行前转换输入张量。

| 策略 | 输入变换 | 执行设备 | 说明 |
|------|---------|---------|------|
| `fp64_cpu`（默认） | `tensors_to_fp64_cpu` | CPU, float64 | 升精度到 fp64 避免 NPU 溢出污染 |
| `native_cpu` | `tensors_to_cpu` | CPU, 原始精度 | 保持原始精度在 CPU 上计算 |
| `native_npu` | 不变 | NPU, 原始精度 | 保持原始精度在 NPU 上计算 |

**checker 三输入对比**：

checker 接收三路数据做精度判断：

| 输入 | 含义 | 来源 |
|------|------|------|
| `ai_output` | AI 算子输出 | NPU 上原始精度执行 |
| `golden_output` | Golden 参考输出 | 由 `golden_precision` 策略决定 |
| `native_output` | 同精度参考输出 | 用于小值域判断，与 AI 输出同精度 |

`native_output` 的获取策略：
- `fp64_cpu` 时 golden 是 fp64（精度不同），需单独跑一次 CPU 同精度 golden
- `native_cpu` / `native_npu` 时 golden 已是同精度，直接复用 `golden_result.outputs` 避免重复计算

```python
# 示例：为不同评测集选择合适的 golden_precision
BenchRegistry.register('kernel_triton', BenchConfig(
    ...
    golden_precision='fp64_cpu',   # Triton kernel 对比 fp64 golden，严格精度验证
))

BenchRegistry.register('quick_smoke', BenchConfig(
    ...
    golden_precision='native_npu', # 快速冒烟测试，golden 直接在 NPU 上跑
))
```

## 接入场景

### 场景 1：新用例格式 + 内置评分方案

适用：用例定义格式不同，但评分需求与内置方案一致

#### Step 1: 定义数据模型（可选）

如果内置的 `TaskSpec` / `CaseSpec` 满足需求，可直接使用 metadata 存储特化字段：

```python
from kernel_eval.base import TaskSpec, CaseSpec

# 直接使用基类，特化字段存入 metadata
task = TaskSpec(
    task_id="my_op_001",
    name="MyOperator",
    metadata={
        'custom_field1': 'value1',
        'custom_field2': 'value2',
    }
)

case = CaseSpec(
    case_id="my_op_001_1",
    input_shapes=[[1024, 1024]],
    dtypes=["float32"],
    metadata={
        'baseline_perf_us': 100.0,  # 性能基线
    }
)
```

如果需要特化字段强类型，定义子类：

```python
from dataclasses import dataclass
from kernel_eval.base import TaskSpec, CaseSpec

@dataclass
class MyTaskSpec(TaskSpec):
    """自定义任务规格"""
    custom_field1: str = ""
    custom_perf_baseline: float = 0.0

@dataclass
class MyCaseSpec(CaseSpec):
    """自定义用例规格"""
    custom_perf_baseline: float = 0.0
```

#### Step 2: 实现 Loader 子类

```python
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

from kernel_eval.base import TaskLoader, CaseLoader, TaskSpec, CaseSpec

class MyTaskLoader(TaskLoader):
    """自定义任务加载器"""
    
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
        self._cache: Dict[str, TaskSpec] = {}
    
    def list_tasks(self) -> List[TaskSpec]:
        """列出所有任务"""
        tasks = []
        for task_file in self.bench_root.glob("**/my_task.yaml"):
            try:
                task_spec = self.get_task(str(task_file.parent.relative_to(self.bench_root)))
                if task_spec:
                    tasks.append(task_spec)
            except Exception as e:
                print(f"[WARN] 加载失败: {task_file}: {e}")
        return tasks
    
    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        """获取指定任务"""
        if task_id in self._cache:
            return self._cache[task_id]
        
        task_file = self.bench_root / task_id / "my_task.yaml"
        if not task_file.exists():
            return None
        
        with open(task_file) as f:
            data = yaml.safe_load(f)
        
        task_spec = TaskSpec(
            task_id=task_id,
            name=data.get('name', ''),
            description=data.get('description', ''),
            metadata={
                'baseline_perf_us': data.get('baseline_us', 0.0),
            }
        )
        self._cache[task_id] = task_spec
        return task_spec
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        tasks = self.list_tasks()
        return {'total': len(tasks)}


class MyCaseLoader(CaseLoader):
    """自定义用例加载器"""
    
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
    
    def scan_all(self) -> List[CaseSpec]:
        """扫描所有用例"""
        cases = []
        for case_file in self.bench_root.glob("**/my_cases.yaml"):
            cases.extend(self._load_cases(case_file))
        return cases
    
    def scan_by_task(self, task_name: str) -> List[CaseSpec]:
        """扫描指定任务的用例"""
        all_cases = self.scan_all()
        return [c for c in all_cases if c.metadata.get('task_name', '').lower() == task_name.lower()]
    
    def _load_cases(self, case_file: Path) -> List[CaseSpec]:
        """加载用例文件"""
        with open(case_file) as f:
            data = yaml.safe_load(f)
        
        rel_path = str(case_file.parent.relative_to(self.bench_root))
        cases = []
        
        for case_data in data.get('cases', []):
            case_spec = CaseSpec(
                case_id=f"{rel_path}_{case_data.get('id', 0)}",
                input_shapes=case_data.get('input_shapes', []),
                dtypes=case_data.get('dtypes', []),
                attrs=case_data.get('attrs', {}),
                value_ranges=case_data.get('value_ranges', []),
                metadata={
                    'task_name': case_data.get('task_name', ''),
                    'baseline_perf_us': case_data.get('baseline_us', 0.0),
                }
            )
            cases.append(case_spec)
        
        return cases
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        cases = self.scan_all()
        return {'total': len(cases)}
```

#### Step 3: 注册 Loader

```python
from kernel_eval.registry.loader_registry import LoaderRegistry

# 注册自定义评测集
LoaderRegistry.register_task_loader("my_bench", MyTaskLoader)
LoaderRegistry.register_case_loader("my_bench", MyCaseLoader)
```

#### Step 4: 使用

```python
from kernel_eval.registry.loader_registry import LoaderRegistry
from kernel_eval.benches.cann_scoring import ScoringCalculator

# 获取自定义 Loader
task_loader = LoaderRegistry.get_task_loader("my_bench", bench_root="/path/to/my_bench")
case_loader = LoaderRegistry.get_case_loader("my_bench", bench_root="/path/to/my_bench")

# 列出任务和用例
tasks = task_loader.list_tasks()
cases = case_loader.scan_all()

# 使用内置评分方案
from kernel_eval.registry.scoring_registry import ScoringSchemeRegistry
scheme = ScoringSchemeRegistry.get("simple_comparison")  # 加速比方案
```

---

### 场景 2：新评分方案

适用：评分公式不同（如自定义 baseline 来源、自定义评分公式）

#### Step 1: 实现评分方案子类

```python
from typing import Any, List, Optional
from kernel_eval.base import ScoringScheme, ScoreInfo, PerfResult

class HistoricalComparisonScheme(ScoringScheme):
    """历史数据对比评分方案
    
    baseline 从历史运行数据获取，而不是从用例定义获取
    评分公式：score = baseline / elapsed（加速比）
    """
    
    def get_scheme_name(self) -> str:
        return "historical_comparison"
    
    def get_scheme_description(self) -> str:
        return "历史数据对比：baseline从历史运行获取"
    
    def prepare_baseline(self, case_spec: Any) -> float:
        """从历史数据文件获取基线"""
        # 从历史 JSON 文件读取
        import json
        from pathlib import Path
        
        history_file = Path("reports/history_perf.json")
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
            case_id = case_spec.case_id if hasattr(case_spec, 'case_id') else str(case_spec)
            return history.get(case_id, {}).get('avg_elapsed_us', 0.0)
        
        # 或从用例 metadata 获取
        if hasattr(case_spec, 'metadata'):
            return case_spec.metadata.get('baseline_perf_us', 0.0)
        
        return 0.0
    
    def calculate_case_score(self, perf_result: PerfResult, baseline_us: float) -> Optional[float]:
        """计算加速比"""
        elapsed_us = perf_result.elapsed_us
        
        if elapsed_us <= 0 or baseline_us <= 0:
            return None
        
        return baseline_us / elapsed_us
    
    def aggregate_operator_scores(
        self,
        case_scores: List[ScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None
    ) -> float:
        """聚合算子得分"""
        if total_cases is None:
            total_cases = len(case_scores)
        
        passed_cases = [s for s in case_scores if s.passed]
        if not passed_cases:
            return 0.0
        
        scores = [s.score for s in passed_cases if s.score is not None]
        if not scores:
            return 0.0
        
        # 平均加速比 * pass_rate * 100
        avg_speedup = sum(scores) / len(scores)
        pass_rate = len(passed_cases) / total_cases
        
        return min(avg_speedup * pass_rate * 100, 100)
```

#### Step 2: 注册评分方案

```python
from kernel_eval.registry.scoring_registry import ScoringSchemeRegistry

ScoringSchemeRegistry.register("historical_comparison", HistoricalComparisonScheme())
```

---

### 场景 3：完整新评测集接入示例

以下示例接入一个自定义格式的 NPU 评测集：

```python
# my_npu_bench.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from kernel_eval.base import TaskSpec, CaseSpec, TaskLoader, CaseLoader, ScoringScheme, ScoreInfo, PerfResult
from kernel_eval.registry.loader_registry import LoaderRegistry
from kernel_eval.registry.scoring_registry import ScoringSchemeRegistry


# === 数据模型（使用 metadata 存储特化字段）===

# 无需定义子类，直接使用 TaskSpec/CaseSpec + metadata


# === Loader 实现 ===

class LegacyTaskLoader(TaskLoader):
    """旧版用例格式加载器
    
    旧版格式：每个算子一个 JSON 文件，包含算子信息和用例列表
    """
    
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
        self._cache: Dict[str, TaskSpec] = {}
    
    def list_tasks(self) -> List[TaskSpec]:
        tasks = []
        for op_file in self.bench_root.glob("**/operator.json"):
            rel_path = str(op_file.parent.relative_to(self.bench_root))
            task = self.get_task(rel_path)
            if task:
                tasks.append(task)
        return tasks
    
    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        if task_id in self._cache:
            return self._cache[task_id]
        
        op_file = self.bench_root / task_id / "operator.json"
        if not op_file.exists():
            return None
        
        with open(op_file) as f:
            data = json.load(f)
        
        task_spec = TaskSpec(
            task_id=task_id,
            name=data.get('name', ''),
            description=data.get('description', ''),
            metadata={
                'baseline_perf_us': data.get('baseline_us', 0.0),
                't_hw_us': data.get('t_hw_us', 0.0),
                'category': data.get('category', ''),
            }
        )
        self._cache[task_id] = task_spec
        return task_spec
    
    def get_statistics(self) -> Dict[str, Any]:
        tasks = self.list_tasks()
        return {'total': len(tasks)}


class LegacyCaseLoader(CaseLoader):
    """旧版用例加载器"""
    
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
    
    def scan_all(self) -> List[CaseSpec]:
        cases = []
        for op_file in self.bench_root.glob("**/operator.json"):
            rel_path = str(op_file.parent.relative_to(self.bench_root))
            cases.extend(self._load_cases(op_file, rel_path))
        return cases
    
    def scan_by_task(self, task_name: str) -> List[CaseSpec]:
        all_cases = self.scan_all()
        return [c for c in all_cases if c.metadata.get('operator', '').lower() == task_name.lower()]
    
    def _load_cases(self, op_file: Path, rel_path: str) -> List[CaseSpec]:
        with open(op_file) as f:
            data = json.load(f)
        
        cases = []
        for i, case_data in enumerate(data.get('test_cases', [])):
            cases.append(CaseSpec(
                case_id=f"{rel_path}_{i}",
                input_shapes=case_data.get('shapes', []),
                dtypes=case_data.get('dtypes', []),
                attrs=case_data.get('attrs', {}),
                value_ranges=case_data.get('value_ranges', []),
                metadata={
                    'operator': data.get('name', ''),
                    'rel_path': rel_path,
                    'baseline_perf_us': case_data.get('baseline_us', data.get('baseline_us', 0.0)),
                    't_hw_us': case_data.get('t_hw_us', data.get('t_hw_us', 0.0)),
                }
            ))
        return cases
    
    def get_statistics(self) -> Dict[str, Any]:
        cases = self.scan_all()
        return {'total': len(cases)}


# === 注册 ===

LoaderRegistry.register_task_loader("legacy", LegacyTaskLoader)
LoaderRegistry.register_case_loader("legacy", LegacyCaseLoader)

# 使用 CANN 内置评分方案（SOL-Score）
# 或注册自定义评分方案
```

## 内置评分方案

| 方案名 | 描述 | 适用场景 |
|--------|------|----------|
| `cann` | SOL-Score（baseline + t_hw） | CANN 评测集（默认） |
| `simple_comparison` | 加速比 = baseline / elapsed | 有 baseline 的评测集 |
| `recording_only` | 仅记录，不评分 | 无 baseline 的评测集 |

---

## Baseline 适配方法

Baseline（性能基线）是评分的关键锚点。不同评测集的 baseline 来源各异，推荐统一通过 `CaseSpec.metadata` 存储和获取。

### Baseline 来源分类

| 来源 | 说明 | 适配方式 |
|------|------|----------|
| **用例预制** | 在配置文件中预先标定 | 存入 `cases.yaml`/`cases.csv`，Loader 解析到 `metadata` |
| **历史数据** | 从历史运行记录获取 | ScoringScheme 的 `prepare_baseline()` 中读取历史文件 |
| **参考实现运行** | 实时运行参考实现采集 | 评测前预处理，或 ScoringScheme 中动态计算 |
| **第三方库基准** | 高性能库执行时间 | 作为"最佳可达 baseline"存入配置 |

### 场景 A：用例预制 Baseline

适用：baseline 已预先采集并固化到配置文件中（最推荐方式）

#### Step 1：配置文件格式

```yaml
# cases.yaml 示例（baseline 预制）
cases:
  - case_id: 1
    input_shape: [[1024, 1024]]
    dtype: [float32]
    baseline_perf_us: 1250.0    # 预制的参考实现执行时间
    baseline_optimized_us: 800.0  # 预制的优化实现执行时间（可选）
    
  - case_id: 2
    input_shape: [[2048, 2048]]
    dtype: [float16]
    baseline_perf_us: 450.0
```

#### Step 2：Loader 解析到 metadata

```python
from kernel_eval.base import CaseLoader, CaseSpec

class MyCaseLoader(CaseLoader):
    def _load_cases(self, case_file: Path) -> List[CaseSpec]:
        with open(case_file) as f:
            data = yaml.safe_load(f)
        
        cases = []
        for case_data in data.get('cases', []):
            case = CaseSpec(
                case_id=case_data.get('case_id', ''),
                input_shapes=case_data.get('input_shape', []),
                dtypes=case_data.get('dtype', []),
                # baseline 存入 metadata（统一接口）
                metadata={
                    'baseline_perf_us': case_data.get('baseline_perf_us', 0.0),
                    'baseline_optimized_us': case_data.get('baseline_optimized_us', 0.0),
                }
            )
            cases.append(case)
        return cases
```

#### Step 3：ScoringScheme 获取 baseline

```python
from kernel_eval.base import ScoringScheme, PerfResult

class SpeedupScoringScheme(ScoringScheme):
    def prepare_baseline(self, case_spec) -> float:
        """从 CaseSpec.metadata 获取 baseline"""
        if hasattr(case_spec, 'metadata'):
            return case_spec.metadata.get('baseline_perf_us', 0.0)
        return 0.0
    
    def calculate_case_score(self, perf_result: PerfResult, baseline_us: float) -> Optional[float]:
        """计算 Speedup = baseline / elapsed"""
        elapsed_us = perf_result.elapsed_us
        if elapsed_us <= 0 or baseline_us <= 0:
            return None
        return baseline_us / elapsed_us
```

### 场景 B：多 Baseline 来源（动态选择）

适用：同一评测集需要支持多种 baseline 来源（如不同参考实现的执行时间）

```python
from kernel_eval.base import ScoringScheme, PerfResult

class MultiBaselineScoringScheme(ScoringScheme):
    """支持多 baseline 来源的评分方案"""
    
    def __init__(self, baseline_source: str = 'default'):
        self.baseline_source = baseline_source  # 'default' 或 'optimized'
    
    def prepare_baseline(self, case_spec) -> float:
        """根据配置选择 baseline 来源"""
        if not hasattr(case_spec, 'metadata'):
            return 0.0
        
        metadata = case_spec.metadata
        if self.baseline_source == 'optimized':
            # 使用优化后的 baseline（如高性能库）
            return metadata.get('baseline_optimized_us', 
                                metadata.get('baseline_perf_us', 0.0))
        else:
            # 默认使用标准 baseline
            return metadata.get('baseline_perf_us', 0.0)
    
    def calculate_case_score(self, perf_result: PerfResult, baseline_us: float) -> Optional[float]:
        if perf_result.elapsed_us <= 0 or baseline_us <= 0:
            return None
        return baseline_us / perf_result.elapsed_us
```

### 场景 C：历史数据 Baseline

适用：baseline 从历史运行数据动态获取

```python
from kernel_eval.base import ScoringScheme

class HistoricalBaselineScheme(ScoringScheme):
    """从历史 JSON 文件获取 baseline"""
    
    def __init__(self, history_file: str = "reports/history_perf.json"):
        self.history_file = Path(history_file)
        self._history_cache = None
    
    def _load_history(self) -> Dict[str, float]:
        if self._history_cache is None and self.history_file.exists():
            with open(self.history_file) as f:
                self._history_cache = json.load(f)
        return self._history_cache or {}
    
    def prepare_baseline(self, case_spec) -> float:
        history = self._load_history()
        case_id = case_spec.case_id if hasattr(case_spec, 'case_id') else str(case_spec)
        return history.get(case_id, {}).get('avg_elapsed_us', 0.0)
```

---

## Golden 函数加载适配

Golden（参考实现）的来源同样因评测体系而异。当前架构已支持 `GoldenLoaderRegistry` 注册机制。

### Golden 来源分类

| 评测体系 | Golden 来源 | 文件/函数 |
|----------|-------------|-----------|
| **cann-bench** | 算子目录内的 golden.py | `tasks/{level}/{op}/golden.py` 中的函数 |
| **外部评测集 A** | PyTorch 标准实现 | `torch.matmul`、`torch.nn.functional.relu` 等 |
| **外部评测集 B** | Model 类的 forward | `validation/module.py` 中的 `Model.forward` |
| **外部评测集 C** | 单独的 reference.py | `reference.py` 中的实现函数 |

### 场景 A：继承 GoldenLoaderBase 子类

适用：Golden 文件位置或命名与 cann-bench 不同

```python
from pathlib import Path
import importlib.util
from kernel_eval.base import GoldenLoaderBase

class ExternalGoldenLoader(GoldenLoaderBase):
    """从 validation/module.py 加载 Model.forward"""
    
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
    
    def get_golden_function(self, task_id: str) -> Callable:
        """加载 Model 类的 forward 方法"""
        module_path = self.bench_root / task_id / "validation" / "module.py"
        if not module_path.exists():
            raise ImportError(f"模块不存在: {module_path}")
        
        # 动态加载
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取 Model 类
        if not hasattr(module, 'Model'):
            raise AttributeError(f"模块中找不到 Model 类")
        
        model_cls = getattr(module, 'Model')
        
        # 返回 forward 方法
        return model_cls().forward
    
    def get_input_function(self, task_id: str) -> Optional[Callable]:
        """从 prepare_inputs.py 加载 get_inputs"""
        prep_path = self.bench_root / task_id / "validation" / "prepare_inputs.py"
        if not prep_path.exists():
            return None
        
        spec = importlib.util.spec_from_file_location("prepare_inputs", prep_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return getattr(module, 'get_inputs', None)
```

### 场景 B：PyTorch 标准实现作为 Golden

适用：Golden 直接使用 PyTorch 内置函数

```python
from typing import Callable, Dict
import torch

from kernel_eval.base import GoldenLoaderBase

class PyTorchRefLoader(GoldenLoaderBase):
    """PyTorch 标准实现作为 Golden"""
    
    # 标准函数映射表
    REFERENCE_MAP: Dict[str, Callable] = {
        'matmul': torch.matmul,
        'add': torch.add,
        'mul': torch.mul,
        'relu': torch.nn.functional.relu,
        'gelu': torch.nn.functional.gelu,
        'softmax': torch.nn.functional.softmax,
        'layer_norm': torch.nn.functional.layer_norm,
    }
    
    def __init__(self, bench_root: str = None):
        self.bench_root = bench_root  # 可能不需要
    
    def get_golden_function(self, op_name: str) -> Callable:
        """根据算子名返回 PyTorch 标准实现"""
        op_key = op_name.lower().replace('_', '')
        if op_key in self.REFERENCE_MAP:
            return self.REFERENCE_MAP[op_key]
        
        # 尝试从 torch 直接获取
        if hasattr(torch, op_key):
            return getattr(torch, op_key)
        if hasattr(torch.nn.functional, op_key):
            return getattr(torch.nn.functional, op_key)
        
        raise ImportError(f"未找到 {op_name} 的 PyTorch 标准实现")
    
    def get_input_function(self, task_id: str) -> Optional[Callable]:
        """PyTorch 标准实现通常不需要自定义输入生成"""
        return None
```

### 场景 C：注册自定义 GoldenLoader

```python
from kernel_eval.registry.golden_registry import GoldenLoaderRegistry

# 注册自定义 GoldenLoader
GoldenLoaderRegistry.register('pytorch_ref', PyTorchRefLoader)

# 在 BenchConfig 中使用
from kernel_eval.registry.bench_registry import BenchRegistry, BenchConfig

BenchRegistry.register('kernelbench', BenchConfig(
    task_loader='kernelbench',
    case_loader='kernelbench',
    golden_loader='pytorch_ref',  # 使用 PyTorch 标准实现
    scoring_scheme='speedup',
    checker='allclose',
    ...
))
```

---

## OperatorMatcher 适配

OperatorMatcher 负责 AI 算子函数的加载和匹配。不同评测体系的 AI 算子来源各异，需要适配。

### AI 算子来源分类

| 评测体系 | AI 算子来源 | 加载方式 |
|----------|-------------|----------|
| **cann-bench** | `torch.ops.cann_bench` 或 `cann_bench` 模块 | 通过 `torch.ops` 或 `import` 加载 |
| **KernelBench** | Triton kernel 或 `kernel_gen_ops` 模块 | Triton 编译或模块导入 |
| **自定义评测集 A** | 动态编译的 CUDA kernel | JIT 编译后加载 |
| **自定义评测集 B** | 用户提交的 Python 函数 | 从提交目录动态导入 |

### 场景 A：继承 OperatorMatcherBase 子类

适用：AI 算子来源与 cann-bench 不同（如 Triton kernel）

```python
from typing import Callable, Dict, Optional
import importlib.util

from kernel_eval.base import OperatorMatcherBase
from kernel_eval.benches.cann_loader import CannTaskLoader

class TritonOperatorMatcher(OperatorMatcherBase):
    """从 Triton kernel 文件加载 AI 算子"""

    def __init__(self, operator_loader: CannTaskLoader):
        self.operator_loader = operator_loader
        self._kernel_cache: Dict[str, Callable] = {}

    def load_ai_operator(self, kernel_name: str) -> Callable:
        """加载 Triton kernel 函数

        Args:
            kernel_name: kernel 名称（如 problem_id）

        Returns:
            Triton kernel 函数
        """
        cache_key = kernel_name.lower()
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # 从提交目录加载 Triton kernel
        kernel_file = self.operator_loader.bench_root / kernel_name / "submission" / "kernel.py"
        if not kernel_file.exists():
            raise ImportError(f"Triton kernel 不存在: {kernel_file}")

        spec = importlib.util.spec_from_file_location("kernel", kernel_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 获取 kernel 函数（假设命名规则）
        kernel_func = getattr(module, 'triton_kernel', None)
        if kernel_func is None:
            # 尝试其他命名
            kernel_func = getattr(module, kernel_name.lower(), None)
            if kernel_func is None:
                raise AttributeError(f"无法在 {kernel_file} 中找到 kernel 函数")

        self._kernel_cache[cache_key] = kernel_func
        return kernel_func

    def find_operator_info(self, operator_name: str) -> Optional:
        """查找算子信息"""
        return self.operator_loader.get_task(operator_name)

    def clear_cache(self):
        """清空缓存"""
        self._kernel_cache.clear()
```

### 场景 B：从 torch.ops 加载（通用模式）

适用：AI 算子通过 `torch.ops` 注册

```python
import torch
from typing import Callable, Dict, List, Optional

from kernel_eval.base import OperatorMatcherBase

class TorchOpsMatcher(OperatorMatcherBase):
    """从 torch.ops 加载 AI 算子"""

    def __init__(self, operator_loader, ops_namespace: str = "custom_ops"):
        self.operator_loader = operator_loader
        self.ops_namespace = ops_namespace
        self._op_cache: Dict[str, Callable] = {}

    def load_ai_operator(self, op_name: str) -> Callable:
        """从 torch.ops.{namespace} 加载算子"""
        cache_key = op_name.lower()
        if cache_key in self._op_cache:
            return self._op_cache[cache_key]

        # 获取 namespace
        if not hasattr(torch.ops, self.ops_namespace):
            raise ImportError(f"torch.ops.{self.ops_namespace} 未注册")

        namespace = getattr(torch.ops, self.ops_namespace)

        # 尝试多种命名变体
        candidates = self._get_name_candidates(op_name)
        for name in candidates:
            if hasattr(namespace, name):
                func = getattr(namespace, name)
                self._op_cache[cache_key] = func
                return func

        raise AttributeError(f"无法在 torch.ops.{self.ops_namespace} 中找到 {op_name}")

    def _get_name_candidates(self, op_name: str) -> List[str]:
        """生成命名候选列表"""
        return [
            op_name.lower(),
            op_name.replace('_', ''),
            op_name.replace(' ', '_').lower(),
        ]

    def find_operator_info(self, operator_name: str) -> Optional:
        return self.operator_loader.get_task(operator_name)

    def clear_cache(self):
        self._op_cache.clear()
```

### 场景 C：使用 OperatorMatcherRegistry 注册

适用：将自定义 OperatorMatcher 注册到评测体系

```python
from kernel_eval.registry.matcher_registry import OperatorMatcherRegistry
from kernel_eval.base import OperatorMatcherBase

# 1. 定义自定义 Matcher（如上场景 A/B）

# 2. 注册到 Registry
OperatorMatcherRegistry.register("triton", TritonOperatorMatcher)
OperatorMatcherRegistry.register("torch_ops", TorchOpsMatcher)

# 3. BenchConfig 中指定 operator_matcher
from kernel_eval.registry.bench_registry import BenchRegistry, BenchConfig

BenchRegistry.register('kernelbench', BenchConfig(
    task_loader='kernelbench',
    case_loader='kernelbench',
    golden_loader='torch_compile',  # torch.compile 作为 Golden
    operator_matcher='triton',      # Triton kernel 作为 AI 算子
    scoring_scheme='pass_at_k',     # Pass@k 评分
    checker='torch_compile',        # torch.compile 正确性基准
    description='KernelBench Triton kernel 评测',
))
```

### OperatorMatcherBase 抽象接口

```python
# src/kernel_eval/base/matcher.py

from abc import ABC, abstractmethod
from typing import Callable, Optional

class OperatorMatcherBase(ABC):
    """AI 算子匹配器抽象基类"""

    @abstractmethod
    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载 AI 生成的算子函数

        Args:
            operator_name: 算子名称或标识

        Returns:
            可调用的算子函数
        """
        pass

    @abstractmethod
    def find_operator_info(self, operator_name: str) -> Optional:
        """查找算子定义信息

        Args:
            operator_name: 算子名称

        Returns:
            TaskSpec 或类似对象，包含算子信息
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """清空算子缓存"""
        pass
```

### OperatorMatcherRegistry API

```python
# src/kernel_eval/registry/matcher_registry.py

class OperatorMatcherRegistry:
    """AI 算子匹配器注册表"""

    @classmethod
    def register(cls, name: str, matcher_cls: Type[OperatorMatcherBase]) -> None:
        """注册匹配器类"""

    @classmethod
    def get(cls, name: str = None, operator_loader = None) -> OperatorMatcherBase:
        """获取匹配器实例（自动实例化）"""

    @classmethod
    def get_cls(cls, name: str = None) -> Optional[Type[OperatorMatcherBase]]:
        """获取匹配器类（不实例化）"""

    @classmethod
    def list_matchers(cls) -> List[str]:
        """列出已注册的匹配器"""

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查是否已注册"""

# 便捷函数
def get_operator_matcher(eval_system: str = "cann", operator_loader = None) -> OperatorMatcherBase:
    """获取 OperatorMatcher 实例"""
```

---

## 正确性基准适配

正确性验证的"黄金参考"来源同样可适配。除了内置的 `relative_error` 和 `allclose`，还可接入其他参考来源。

### 场景 A：自定义正确性基准

适用：使用自定义的正确性验证逻辑

```python
from kernel_eval.base import CorrectnessChecker, AccuracyResult

class CustomReferenceChecker(CorrectnessChecker):
    """使用自定义参考实现作为正确性基准"""
    
    def __init__(self, rtol: float = 1e-2, atol: float = 1e-2):
        self.rtol = rtol
        self.atol = atol
    
    def check(self, ai_output, ref_func, inputs, **kwargs) -> AccuracyResult:
        rtol = kwargs.get('rtol', self.rtol)
        atol = kwargs.get('atol', self.atol)
        
        import torch
        
        # 执行参考实现获取 golden 输出
        with torch.no_grad():
            golden_output = ref_func(*inputs)
        
        # allclose 比较
        passed = torch.allclose(ai_output, golden_output, rtol=rtol, atol=atol)
        
        # 计算误差统计（可选）
        max_abs_err = 0.0
        max_rel_err = 0.0
        if isinstance(ai_output, torch.Tensor) and isinstance(golden_output, torch.Tensor):
            abs_diff = torch.abs(ai_output - golden_output)
            max_abs_err = abs_diff.max().item()
            rel_diff = abs_diff / (torch.abs(golden_output) + 1e-7)
            max_rel_err = rel_diff.max().item()
        
        return AccuracyResult(
            passed=passed,
            threshold=rtol,
            metadata={
                'max_abs_error': max_abs_err,
                'max_rel_error': max_rel_err,
            }
        )
```

注册：

```python
from kernel_eval.registry.checker_registry import CheckerRegistry

CheckerRegistry.register("custom_ref", CustomReferenceChecker(rtol=1e-2, atol=1e-2))
```

**注意**：参考实现 `ref_func` 的执行必须在 NPU 上完成，以确保与 AI 算子的输出在同一设备上进行比较。

### 场景 B：自定义精度阈值（按 dtype）

适用：不同数据类型使用不同的精度阈值

```python
from kernel_eval.base import CorrectnessChecker, AccuracyResult

class DtypeAwareChecker(CorrectnessChecker):
    """按 dtype 使用不同精度阈值"""
    
    DEFAULT_THRESHOLDS = {
        'float16': {'rtol': 1e-2, 'atol': 1e-2},
        'bfloat16': {'rtol': 1e-2, 'atol': 1e-2},
        'float32': {'rtol': 1e-4, 'atol': 1e-4},
        'float64': {'rtol': 1e-5, 'atol': 1e-5},
    }
    
    def check(self, ai_output, ref_output, inputs, dtype: str = None, **kwargs) -> AccuracyResult:
        # 根据 dtype 选择阈值
        thresholds = self.DEFAULT_THRESHOLDS.get(dtype or 'float32', 
                                                   {'rtol': 1e-4, 'atol': 1e-4})
        rtol = kwargs.get('rtol', thresholds['rtol'])
        atol = kwargs.get('atol', thresholds['atol'])
        
        passed = torch.allclose(ai_output, ref_output, rtol=rtol, atol=atol)
        
        return AccuracyResult(passed=passed, threshold=rtol)
```

---

## Pass@k 指标计算

适用于需要多次采样统计的场景（如模型生成多个候选 kernel）。

### 概念说明

Pass@k：在 n 次尝试中，至少有 k 次成功的概率估计。

公式：`Pass@k = 1 - C(n-c, k) / C(n, k)`，其中 c 为成功次数。

### 实现

```python
from math import comb
from kernel_eval.base import ScoringScheme, ScoreInfo, PerfResult

class PassAtKScoringScheme(ScoringScheme):
    """支持 Pass@k 指标的评分方案"""
    
    def compute_pass_at_k(self, results: List[bool], k: int) -> float:
        """计算 Pass@k
        
        Args:
            results: 每次尝试的成功/失败列表
            k: 需要至少成功的次数
            
        Returns:
            Pass@k 概率估计 [0, 1]
        """
        n = len(results)
        c = sum(results)  # 成功次数
        
        if n < k:
            return 0.0
        if c >= n - k + 1:  # 失败次数 < k，必然满足
            return 1.0
        
        # Pass@k = 1 - C(n-c, k) / C(n, k)
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    def aggregate_operator_scores(
        self,
        case_scores: List[ScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None
    ) -> float:
        """聚合算子得分，计算 Pass@1, Pass@5, Pass@10"""
        results = [s.passed for s in case_scores]
        
        # 返回多个 Pass@k 指标
        return {
            'pass@1': self.compute_pass_at_k(results, 1),
            'pass@5': self.compute_pass_at_k(results, 5) if len(results) >= 5 else 0.0,
            'pass@10': self.compute_pass_at_k(results, 10) if len(results) >= 10 else 0.0,
        }
```

---

## 自定义输入生成

部分评测集需要自定义输入生成逻辑（如特定的初始化方式）。

### 场景 A：从配置文件加载输入生成函数

```python
from pathlib import Path
from typing import Callable, List
import importlib.util
from kernel_eval.base import CaseSpec

class CustomInputGenerator:
    """从 prepare_inputs.py 加载输入生成函数"""
    
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
    
    def get_input_func(self, task_id: str) -> Optional[Callable]:
        """加载指定任务的输入生成函数"""
        prep_file = self.bench_root / task_id / "prepare_inputs.py"
        if not prep_file.exists():
            return None
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("prepare_inputs", prep_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取 get_inputs 函数
        return getattr(module, 'get_inputs', None)
    
    def generate_inputs(self, case_spec: CaseSpec) -> List:
        """生成输入数据"""
        task_id = case_spec.metadata.get('task_id', '')
        input_func = self.get_input_func(task_id)
        
        if input_func:
            # 使用自定义输入生成
            param = {
                'shape': str(case_spec.input_shapes),
                'dtype': case_spec.dtypes[0] if case_spec.dtypes else 'float32',
            }
            return input_func(param, device='npu')
        
        # 使用默认 DataGenerator
        from kernel_eval.data import DataGenerator
        generator = DataGenerator()
        return generator.generate_input_tensors_from_case(
            input_shapes=case_spec.input_shapes,
            dtypes=case_spec.dtypes,
            value_ranges=case_spec.value_ranges,
        )
```

---

## 完整接入示例（带 Baseline 和 Pass@k）

以下示例展示一个完整的评测集接入，包含预制 baseline、Pass@k 指标、自定义正确性基准：

```python
# complete_bench_integration.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from math import comb
import yaml
import json

from kernel_eval.base import TaskSpec, CaseSpec, TaskLoader, CaseLoader, ScoringScheme, ScoreInfo, CorrectnessChecker, AccuracyResult, PerfResult
from kernel_eval.registry.loader_registry import LoaderRegistry
from kernel_eval.registry.scoring_registry import ScoringSchemeRegistry
from kernel_eval.registry.checker_registry import CheckerRegistry
import torch


# === Loader 实现（baseline 预制）===

class ExternalBenchTaskLoader(TaskLoader):
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
        self._cache: Dict[str, TaskSpec] = {}
    
    def list_tasks(self) -> List[TaskSpec]:
        tasks = []
        for task_file in self.bench_root.glob("**/problem.yaml"):
            rel_path = str(task_file.parent.relative_to(self.bench_root))
            task = self.get_task(rel_path)
            if task:
                tasks.append(task)
        return tasks
    
    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        if task_id in self._cache:
            return self._cache[task_id]
        
        task_file = self.bench_root / task_id / "problem.yaml"
        if not task_file.exists():
            return None
        
        with open(task_file) as f:
            data = yaml.safe_load(f)
        
        task_spec = TaskSpec(
            task_id=task_id,
            name=data.get('name', ''),
            description=data.get('description', ''),
            metadata={
                'level': data.get('level', 1),
                'category': data.get('category', ''),
            }
        )
        self._cache[task_id] = task_spec
        return task_spec
    
    def get_statistics(self) -> Dict[str, Any]:
        return {'total': len(self.list_tasks())}


class ExternalBenchCaseLoader(CaseLoader):
    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
    
    def scan_all(self) -> List[CaseSpec]:
        cases = []
        for task_file in self.bench_root.glob("**/problem.yaml"):
            rel_path = str(task_file.parent.relative_to(self.bench_root))
            cases.extend(self._load_cases(task_file, rel_path))
        return cases
    
    def scan_by_task(self, task_name: str) -> List[CaseSpec]:
        all_cases = self.scan_all()
        return [c for c in all_cases if c.metadata.get('task_name', '').lower() == task_name.lower()]
    
    def _load_cases(self, task_file: Path, rel_path: str) -> List[CaseSpec]:
        with open(task_file) as f:
            data = yaml.safe_load(f)
        
        cases = []
        for i, baseline_info in enumerate(data.get('baselines', [])):
            case = CaseSpec(
                case_id=f"{data['problem_id']}_{i}",
                input_shapes=[baseline_info.get('shape', [])],
                dtypes=[baseline_info.get('dtype', 'float32')],
                # baseline 存入 metadata
                metadata={
                    'task_name': data.get('name', ''),
                    'rel_path': rel_path,
                    'baseline_perf_us': baseline_info.get('baseline_us', 0.0),
                    'baseline_optimized_us': baseline_info.get('optimized_us', 0.0),
                    'level': data.get('level', 1),
                }
            )
            cases.append(case)
        return cases
    
    def get_statistics(self) -> Dict[str, Any]:
        return {'total': len(self.scan_all())}


# === Checker 实现（自定义正确性基准）===

class CustomReferenceChecker(CorrectnessChecker):
    """使用参考实现作为正确性基准"""
    
    def __init__(self, rtol: float = 1e-2, atol: float = 1e-2):
        self.rtol = rtol
        self.atol = atol
    
    def check(self, ai_output, ref_func, inputs, **kwargs) -> AccuracyResult:
        rtol = kwargs.get('rtol', self.rtol)
        atol = kwargs.get('atol', self.atol)
        
        with torch.no_grad():
            golden_output = ref_func(*inputs)
        
        passed = torch.allclose(ai_output, golden_output, rtol=rtol, atol=atol)
        
        return AccuracyResult(passed=passed, threshold=rtol)


# === Scoring 实现（Speedup + Pass@k）===

class SpeedupWithPassAtKScheme(ScoringScheme):
    """Speedup 评分 + Pass@k 指标"""
    
    def get_scheme_name(self) -> str:
        return "speedup_pass_at_k"
    
    def prepare_baseline(self, case_spec) -> float:
        if hasattr(case_spec, 'metadata'):
            return case_spec.metadata.get('baseline_perf_us', 0.0)
        return 0.0
    
    def calculate_case_score(self, perf_result: PerfResult, baseline_us: float) -> Optional[float]:
        elapsed_us = perf_result.elapsed_us
        if elapsed_us <= 0 or baseline_us <= 0:
            return None
        return baseline_us / elapsed_us
    
    def compute_pass_at_k(self, results: List[bool], k: int) -> float:
        n = len(results)
        c = sum(results)
        if n < k:
            return 0.0
        if c >= n - k + 1:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    def aggregate_operator_scores(
        self,
        case_scores: List[ScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None
    ) -> Dict[str, Any]:
        results = [s.passed for s in case_scores]
        scores = [s.score for s in case_scores if s.score is not None]
        
        return {
            'avg_speedup': sum(scores) / len(scores) if scores else 0.0,
            'pass@1': self.compute_pass_at_k(results, 1),
            'pass@5': self.compute_pass_at_k(results, 5) if len(results) >= 5 else 0.0,
            'pass@10': self.compute_pass_at_k(results, 10) if len(results) >= 10 else 0.0,
        }


# === 注册 ===

LoaderRegistry.register_task_loader("external_bench", ExternalBenchTaskLoader)
LoaderRegistry.register_case_loader("external_bench", ExternalBenchCaseLoader)
CheckerRegistry.register("custom_ref", CustomReferenceChecker(rtol=1e-2, atol=1e-2))
ScoringSchemeRegistry.register("speedup_pass_at_k", SpeedupWithPassAtKScheme())

from kernel_eval.registry.bench_registry import BenchRegistry, BenchConfig

BenchRegistry.register('external_bench', BenchConfig(
    task_loader='external_bench',
    case_loader='external_bench',
    scoring_scheme='speedup_pass_at_k',
    checker='custom_ref',
    precision_thresholds={'rtol': 1e-2, 'atol': 1e-2},
    default_tasks_root='external_bench/problems',
    description='External benchmark with pre-computed baselines and Pass@k metrics',
))
```

---

## 关键设计说明

### Baseline 为何统一存储在 metadata？

- **来源多样**：用例预制、历史数据、参考实现运行、框架编译器
- **评分公式可定制**：SOL-Score、加速比、自定义公式
- **统一接口**：`CaseSpec.metadata['baseline_perf_us']` 让所有 ScoringScheme 用相同方式获取
- **扩展友好**：新增 baseline 来源只需修改 Loader 或 ScoringScheme.prepare_baseline()

### 正确性基准为何要抽象？

- **参考来源多样**：
  - 预制的 golden 函数（cann-bench）
  - PyTorch 标准实现（torch.matmul）
  - 自定义参考实现
- **精度标准不同**：MERE/MARE、allclose、自定义阈值
- **按 dtype 阈值**：float16/float32/bfloat16 可能需要不同精度要求

### PerfResult 为何是统一类？

- **性能采集统一**：NPU 使用 `torch_npu.profiler`，采集方式一致
- **elapsed_us 是通用指标**：所有评测集都有运行时间
- **特化指标通过 metadata 存储**：baseline、t_hw、roofline 等

### 评分方案为何要抽象？

- **评分公式差异**：SOL-Score、加速比、Pass@k
- **baseline 来源多样**：用例定义、历史数据、框架编译器
- **聚合方式不同**：平均加速比、Pass@k、加权组合

### Loader 为何要抽象？

- **用例格式差异**：YAML、JSON、CSV、自定义格式
- **字段命名差异**：不同评测集的命名约定不同
- **baseline 预制位置不同**：用例文件、历史数据、单独配置

### Golden 为何要适配？

- **来源位置差异**：
  - 算子目录内（`golden.py`）
  - 验证目录内（`validation/module.py`）
  - PyTorch 标准实现（`torch.matmul`）
  - 单独参考文件（`reference.py`）
- **命名差异**：函数名、类方法（`Model.forward`）
- **加载方式差异**：动态导入、直接引用、映射表查找

### OperatorMatcher 为何要适配？

- **AI 算子来源差异**：
  - torch.ops 注册（`torch.ops.cann_bench`）
  - Triton kernel 编译
  - 用户提交目录动态导入
  - 自定义模块（`cann_bench`）
- **命名规则差异**：PascalCase、snake_case、自定义命名
- **加载方式差异**：torch.ops 查找、importlib 动态导入、Triton JIT 编译
- **缓存策略差异**：不同来源可能有不同的缓存需求

### 性能采集为何仅支持 NPU？

- **cann-bench 定位**：专为 CANN/Ascend NPU 设计的评测体系
- **性能采集依赖**：使用 `torch_npu.profiler` 采集 NPU 性能数据
- **不支持其他平台**：CUDA、TPU 等硬件平台不在支持范围内
- **正确性验证可适配**：Checker 层可支持 torch.compile 等多种正确性基准，但最终性能采集必须在 NPU 上执行

## 接入成本

| 场景 | 代码量 | 时间 |
|------|--------|------|
| 新用例格式（内置评分） | 100-200 行 | 1-2 小时 |
| 新评分方案 | 50-100 行 | 0.5-1 小时 |
| 完整新评测集 | 200-300 行 | 2-3 小时 |

## CLI 使用方式

接入新的评测集后，CLI 命令通过 `--bench-name` 参数指定评测集：

```bash
# 列出已注册的评测集（含描述）
kernel-bench config --list-benches

# 列出已注册的评分方案
kernel-bench config --list-scoring-schemes

# 列出已注册的精度判断器
kernel-bench config --list-checkers

# 列出 CANN 评测集的算子
kernel-bench list --bench-name cann

# 列出自定义评测集的算子
kernel-bench list --bench-name my_bench

# 列出用例
kernel-bench list --bench-name my_bench --cases

# 查看算子详情
kernel-bench info --bench-name my_bench --operator MyOperator

# 执行评测（自动加载评测集配置）
kernel-bench eval --bench-name my_bench

# 执行评测（筛选算子）
kernel-bench eval --bench-name my_bench --operator MyOperator
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bench-name` | 评测集名称（自动加载 Loader、ScoringScheme、Checker） | `cann` |
| `--task-dir` | 评测目录 | 配置的 tasks_root |
| `--operator` | 算子名称筛选 | 无 |
| `--level` | 难度级别筛选（CANN） | 无 |