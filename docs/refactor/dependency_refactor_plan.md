# 依赖重构方案

> **状态说明（归档）**：本文为早期重构规划，最终落地与此处“目标结构”有出入，阅读时以实际代码树为准。已知差异：未创建 `core/` 包（`evaluator.py` / `perf_eval.py` / `process_pool.py` 在 `eval/`，`data_generator.py` / `package_manager.py` 在 `data/`）；`GoldenLoaderBase` 在 `base/loaders.py` 而非 `base/golden.py`；`benches/cann.py` 已拆为多个 `cann_*.py`（`cann_loader.py` / `cann_matcher.py` / `cann_scoring.py` / …）。

## 一、当前依赖问题分析

### 1.1 依赖方向错误

**正确方向**：特化类 → 依赖 → 公共基类
**当前问题**：公共模块 → 依赖 → 特化类（错误）

### 1.2 违规依赖清单（共 14 处）

| 文件 |违规 import | 特化类 | 修正方案 |
|------|------------|--------|----------|
| `data/loader_registry.py` | `from .cann_task_loader import CannTaskLoader` | CannTaskLoader | 移除硬编码注册，改由 `benches/cann.py` 注册 |
| `data/loader_registry.py` | `from .cann_case_loader import CannCaseLoader` | CannCaseLoader | 同上 |
| `data/golden_loader_registry.py` | `from .golden_loader import GoldenLoader` | GoldenLoader | 移除硬编码注册 |
| `data/bench_registry.py` | `from .cann_task_loader import CannTaskLoader` | CannTaskLoader | 移除直接 import，通过 Registry 获取 |
| `data/package_manager.py` | 依赖 CannTaskLoader | CannTaskLoader | 改用 TaskLoader 基类类型 |
| `eval/operator_matcher_registry.py` | 类型注解 `CannTaskLoader` | CannTaskLoader | 改用 `TaskLoader` 基类类型 |
| `eval/evaluator.py` | `from ..data import CannCaseLoader, CannTaskLoader` | CannCaseLoader, CannTaskLoader | 通过 `get_bench_components()` 获取 |
| `eval/evaluator.py` | `from ..models import CannTaskSpec, CannCaseSpec` | CannTaskSpec, CannCaseSpec | 改用 `TaskSpec`, `CaseSpec` 基类类型 |
| `eval/process_pool.py` | `from ..models import CannCaseSpec` | CannCaseSpec | 改用 `CaseSpec` 基类类型 |
| `eval/process_pool.py` | `from ..data import CannCaseLoader` | CannCaseLoader | 通过 Registry 获取 |
| `eval/failure_synthesizer.py` | `from ..models import CannTaskSpec` | CannTaskSpec | 改用 `TaskSpec` 基类类型 |

---

## 二、重构后目录结构

```
src/kernel_eval/
├── base/                       #【新增】所有基类集中
│   ├── __init__.py             # 导出所有基类
│   ├── loaders.py              # TaskLoader + CaseLoader 基类
│   ├── golden.py               # GoldenLoaderBase 基类
│   ├── matcher.py              # OperatorMatcherBase 基类
│   ├── checker.py              # CorrectnessChecker 基类
│   ├── scoring.py              # ScoringScheme 基类
│   ├── models.py               # TaskSpec + CaseSpec + Solution 基类
│   └── result.py               # PerfResult + AccuracyResult 基类
│
├── registry/                   # 【新增】所有注册表集中
│   ├── __init__.py             # 导出所有 Registry +便捷函数
│   ├── loader_registry.py      # TaskLoader/CaseLoader 注册
│   ├── golden_registry.py      # GoldenLoader 注册
│   ├── matcher_registry.py     # OperatorMatcher 注册
│   ├── checker_registry.py     # CorrectnessChecker 注册
│   ├── scoring_registry.py     # ScoringScheme 注册
│   └── bench_registry.py       # BenchConfig 注册
│
├── benches/                    # 【新增】特化实现聚合
│   ├── __init__.py             # 自动加载所有 bench
│   └── cann.py                 # CANN 全部特化 + 注册
│
├── core/                       # 公共核心逻辑（无特化依赖）
│   ├── evaluator.py            # 评测执行器
│   ├── perf_eval.py            # 性能采集
│   ├── process_pool.py         # 进程池管理
│   ├── data_generator.py       # 数据生成
│   ├── package_manager.py      # 包管理
│   └── report_generator.py     # 报告生成
│
├── utils/                      # 公共工具（不变）
├── security/                   # 安全检查（不变）
└── cli.py                      # CLI入口
```

---

## 三、依赖方向修正方案

### 3.1 Registry 层修正

**原则**：Registry 不直接 import 特化类，只定义注册接口

```python
# registry/loader_registry.py（修正后）
from typing import Dict, Type, Optional
from kernel_eval.base.loaders import TaskLoader, CaseLoader

class LoaderRegistry:
    _task_loaders: Dict[str, Type[TaskLoader]] = {}
    _case_loaders: Dict[str, Type[CaseLoader]] = {}

    @classmethod
    def register_task_loader(cls, name: str, loader_cls: Type[TaskLoader]):
        cls._task_loaders[name] = loader_cls

    @classmethod
    def get_task_loader(cls, name: str, **kwargs) -> TaskLoader:
        if name not in cls._task_loaders:
            raise ValueError(f"Task loader '{name}' 未注册")
        return cls._task_loaders[name](**kwargs)

# 移除末尾的硬编码注册
# 由 benches/cann.py 负责注册
```

### 3.2 Evaluator 层修正

**原则**：通过 Registry 获取组件，类型注解使用基类

```python
# core/evaluator.py（修正后）
from typing import List, Dict, Any, Callable
from kernel_eval.base.models import TaskSpec, CaseSpec
from kernel_eval.base.result import PerfResult, AccuracyResult
from kernel_eval.registry import get_bench_components, BenchRegistry

class Evaluator:
    def __init__(self, config, bench_name: str = 'cann'):
        self.config = config
        self.bench_name = bench_name

        # 通过 Registry 获取组件（不直接 import 特化类）
        components = get_bench_components(bench_name, tasks_root=config.tasks_root)
        self.task_loader = components['task_loader']
        self.case_loader = components['case_loader']
        self.golden_loader = components['golden_loader']
        self.operator_matcher = components['operator_matcher']
        self.scoring_scheme = components['scoring_scheme']
        self.checker = components['checker']

    def evaluate_case(self, case: CaseSpec, ai_op_func: Callable = None) -> EvalCaseResult:
        # 类型注解使用基类 CaseSpec
        ...

    def _filter_cases(self, cases: List[CaseSpec], filter_dict: Dict) -> List[CaseSpec]:
        # 类型注解使用基类 CaseSpec
        ...
```

### 3.3 benches/cann.py 结构

```python
# benches/cann.py（特化实现聚合）

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import yaml
import torch

# 导入基类
from kernel_eval.base.loaders import TaskLoader, CaseLoader
from kernel_eval.base.golden import GoldenLoaderBase
from kernel_eval.base.matcher import OperatorMatcherBase
from kernel_eval.base.checker import CorrectnessChecker
from kernel_eval.base.scoring import ScoringScheme
from kernel_eval.base.models import TaskSpec, CaseSpec
from kernel_eval.base.result import AccuracyResult

# 导入 Registry
from kernel_eval.registry import (
    LoaderRegistry, GoldenLoaderRegistry, OperatorMatcherRegistry,
    CheckerRegistry, ScoringSchemeRegistry, BenchRegistry, BenchConfig
)

# === TaskLoader ===
class CannTaskLoader(TaskLoader):
    ...

# === CaseLoader ===
class CannCaseLoader(CaseLoader):
    ...

# === GoldenLoader ===
class CannGoldenLoader(GoldenLoaderBase):
    ...

# === OperatorMatcher ===
class CannOperatorMatcher(OperatorMatcherBase):
    ...

# === Checker ===
class RelativeErrorChecker(CorrectnessChecker):
    ...

class CannAccuracyResult(AccuracyResult):
    ...

# === ScoringScheme ===
class CannScoringScheme(ScoringScheme):
    ...

# === Models ===
class CannTaskSpec(TaskSpec):
    ...

class CannCaseSpec(CaseSpec):
    ...

# === 注册（文件末尾）===
LoaderRegistry.register_task_loader('cann', CannTaskLoader)
LoaderRegistry.register_case_loader('cann', CannCaseLoader)
GoldenLoaderRegistry.register('cann', CannGoldenLoader)
OperatorMatcherRegistry.register('cann', CannOperatorMatcher)
CheckerRegistry.register('relative_error', RelativeErrorChecker)
ScoringSchemeRegistry.register('cann', CannScoringScheme)

BenchRegistry.register('cann', BenchConfig(
    task_loader='cann',
    case_loader='cann',
    golden_loader='cann',
    operator_matcher='cann',
    scoring_scheme='cann',
    checker='relative_error',
    description='CANN NPU 算子评测集',
))
```

---

## 四、修正后依赖关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    benches/                                  │
│  benches/cann.py──→ 注册特化类到 Registry                     │
│  benches/kernelbench.py──→ 注册特化类到 Registry              │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ import 基类
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     base/                                    │
│  base/loaders.py──→ TaskLoader, CaseLoader 基类              │
│  base/models.py──→ TaskSpec, CaseSpec 基类                   │
│  base/checker.py──→ CorrectnessChecker 基类                  │
│  ...                                                         │
│  （无任何特化依赖）                                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ 被 Registry 引用（类型注解）
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    registry/                                 │
│  registry/loader_registry.py──→ 注册接口（无硬编码）           │
│  registry/bench_registry.py──→ BenchConfig 管理              │
│  ...                                                         │
│  （无任何特化类 import）                                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ 通过 bench_name 获取组件
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     core/                                    │
│  core/evaluator.py──→ 通过 get_bench_components() 获取       │
│  core/process_pool.py──→ 类型注解用基类                       │
│  ...                                                         │
│  （无任何特化类 import）                                       │
└─────────────────────────────────────────────────────────────┘
```

**依赖方向验证**：
- `benches/` → `base/`：正确（特化依赖基类）
- `benches/` → `registry/`：正确（特化注册到 Registry）
- `registry/` → `base/`：正确（类型注解引用基类）
- `core/` → `registry/`：正确（通过 Registry 获取）
- `core/` → `base/`：正确（类型注解用基类）
- `registry/` → `benches/`：**无**（正确，无反向依赖）
- `core/` → `benches/`：**无**（正确，无反向依赖）

---

## 五、重构执行步骤

### Phase 1：创建新目录结构（不影响现有代码）

1. 创建 `base/` 目录，集中所有基类
2. 创建 `registry/` 目录，集中所有 Registry
3. 创建 `benches/` 目录

### Phase 2：修正 Registry（移除硬编码注册）

1. `loader_registry.py` 移除 `from .cann_task_loader import CannTaskLoader`
2. `golden_loader_registry.py` 移除 `from .golden_loader import GoldenLoader`
3. `operator_matcher_registry.py` 移除 CannTaskLoader 类型注解

### Phase 3：修正 Core 模块（通过 Registry 获取）

1. `evaluator.py` 通过 `get_bench_components()` 获取组件
2. `process_pool.py` 类型注解改用 `CaseSpec`
3. `failure_synthesizer.py` 类型注解改用 `TaskSpec`

### Phase 4：创建 benches/cann.py（聚合特化 + 注册）

1. 合并所有 CANN 特化类到单文件
2. 文件末尾执行注册

### Phase 5：更新 __init__.py 导出

1. `base/__init__.py` 导出所有基类
2. `registry/__init__.py` 导出所有 Registry + 便捷函数
3. `benches/__init__.py` 自动加载所有 bench

### Phase 6：运行测试验证

1. 运行现有 404 个测试确保功能不变
2. 新增依赖方向测试确保无反向依赖

---

## 六、测试策略

### 6.1 现有测试基线

- 当前：404 passed
- 重构后：必须保持 404 passed

### 6.2 新增依赖方向测试

```python
# tests/ut/test_dependency_direction.py

def test_registry_no_cann_import():
    """Registry 不应直接 import CannTaskLoader"""
    import kernel_eval.registry.loader_registry as lr
    # 检查模块的 import 语句
    ...

def test_evaluator_no_cann_import():
    """Evaluator 不应直接 import 特化类"""
    import kernel_eval.core.evaluator as ev
    ...

def test_benches_import_base():
    """benches 应 import base 模块"""
    import kernel_eval.benches.cann as cann
    # 验证 CannTaskLoader 继承 TaskLoader
    assert issubclass(cann.CannTaskLoader, TaskLoader)
```

---

## 七、风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| import 路径变化 | 现有代码可能报错 | 保留旧路径别名（兼容过渡） |
| 类型注解变化 | IDE 类型提示可能失效 | 基类保持兼容字段 |
| 注册时机变化 | 未导入 benches 时 Registry 为空 | `benches/__init__.py` 自动导入 |
| 循环导入风险 | 模块加载顺序问题 | 延迟导入 + 注册在文件末尾 |

---

## 八、验收标准

1. ✓ 所有 404 个现有测试通过
2. ✓ Registry 文件无 `from .cann_xxx import` 语句
3. ✓ Core 文件类型注解使用基类
4. ✓ `benches/cann.py` 包含所有 CANN 特化类并完成注册
5. ✓ 依赖方向测试通过（无反向依赖）