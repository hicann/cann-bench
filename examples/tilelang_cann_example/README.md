# TileLang CANN Example

基于 TileLang DSL 的 Ascend NPU 算子示例，通过 JIT 编译实现高性能算子，打包为 `cann_bench` Python 包供 CANN Bench 评测。

## 目录结构

```
tilelang_cann_example/
├── cann_bench/                 # Python包
│   ├── __init__.py             # 导出所有算子接口
│   ├── _common.py              # 共享配置（pass_configs, dtype映射）
│   ├── softmax.py              # Softmax算子（online softmax）
│   └── exp.py                  # Exp算子（逐元素指数）
├── dist/                       # 构建输出
│   └── cann_bench-1.0.0-py3-none-any.whl
├── build.sh                    # 构建脚本
└── setup.py                    # 打包配置
```

## 构建方法

```bash
bash build.sh           # 构建wheel包
```

---

## 评测方法

### 前置条件

1. 安装 TileLang-Ascend（昇腾适配版 TileLang）：

> **注意**：本示例使用的 `tilelang.PassConfigKey.TL_ASCEND_*` 配置项为 TileLang-Ascend 特有，
> PyPI 上的上游 `tilelang`（当前 0.1.x，如 0.1.13）**不包含** Ascend 后端，
> 直接 `pip install tilelang` 会在 `import cann_bench` 时报错。必须安装 TileLang-Ascend：

```bash
# 方式一：官方预编译 wheel（推荐）
# 从 https://github.com/tile-ai/tilelang-ascend 获取 wheel 包后：
pip install tilelang-*.whl

# 方式二：源码编译安装
git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
cd tilelang-ascend
bash install_ascend.sh

# 方式三：通过 PYTHONPATH 注入已有的 tilelang-ascend 源码目录
export PYTHONPATH=/path/to/tilelang-ascend:$PYTHONPATH
```

环境要求（TileLang-Ascend 官方文档）：CANN ≥ 8.3.RC1，torch-npu ≥ 2.6.0.RC1。

2. 安装 cann-bench 依赖：
```bash
pip install -r requirements.txt
```

### 评测单个算子

```bash
# 评测 Exp 算子
PYTHONPATH=/path/to/tilelang-ascend:$PYTHONPATH \
./scripts/run_evaluation.sh \
  --bench-name cann \
  --task-dir tasks/level1 \
  --operator Exp \
  --source-dir examples/tilelang_cann_example \
  --device-id 0

# 评测 Softmax 算子
PYTHONPATH=/path/to/tilelang-ascend:$PYTHONPATH \
./scripts/run_evaluation.sh \
  --bench-name cann \
  --task-dir tasks/level2 \
  --operator Softmax \
  --source-dir examples/tilelang_cann_example \
  --device-id 0
```

### 评测全部算子

```bash
PYTHONPATH=/path/to/tilelang-ascend:$PYTHONPATH \
./scripts/run_evaluation.sh \
  --bench-name cann \
  --task-dir tasks \
  --source-dir examples/tilelang_cann_example \
  --device-id 0
```

评测框架会自动扫描 `cann_bench` 包中导出的算子，与 `tasks/` 目录下的评测任务匹配，逐算子评测。

---

## 新增算子详细步骤

### 第一步：创建算子文件

```bash
# 以新增 GELU 算子为例
touch cann_bench/gelu.py
```

### 第二步：编写 TileLang Kernel

#### gelu.py

每个算子文件包含三部分：Kernel 定义、Kernel 缓存、Python 接口函数。

```python
import torch
import tilelang
from tilelang import language as T
from ._common import PASS_CONFIGS, torch_dtype_to_tl

_kernel_cache = {}


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def _gelu_kernel(M, N, block_M, block_N, dtype="float16"):
    """TileLang JIT 编译的 GELU kernel"""
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)

            row_start = bx * block_M + vid * sub_block_M
            col_start = by * block_N

            T.copy(
                A[row_start : row_start + sub_block_M, col_start : col_start + block_N],
                a,
            )
            # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            # ... kernel 实现逻辑
            T.copy(
                b,
                B[row_start : row_start + sub_block_M, col_start : col_start + block_N],
            )

    return main


def _get_kernel(M, N, tl_dtype):
    """按 (M, N, dtype) 缓存编译后的 kernel"""
    key = (M, N, tl_dtype)
    if key not in _kernel_cache:
        block_M = 128
        block_N = 128
        _kernel_cache[key] = _gelu_kernel(M, N, block_M, block_N, dtype=tl_dtype)
    return _kernel_cache[key]


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU 算子 Python 接口

    签名必须与 tasks/level1/gelu/proto.yaml 中的 schema 一致：
        gelu(Tensor x) -> Tensor y
    """
    original_shape = x.shape
    x_flat = x.contiguous().reshape(-1, x.size(-1))
    M, N = x_flat.shape

    tl_dtype = torch_dtype_to_tl(x.dtype)
    kernel = _get_kernel(M, N, tl_dtype)
    out_flat = kernel(x_flat)

    return out_flat.reshape(original_shape)
```

### 第三步：注册导出

编辑 `cann_bench/__init__.py`：

```python
__version__ = "1.0.0"

from .softmax import softmax
from .exp import exp
from .gelu import gelu       # 新增
```

### 第四步：重新构建

```bash
bash build.sh
```

**无需修改 setup.py 或 build.sh！** `find_packages()` 会自动发现 `cann_bench/` 下的所有模块。

---

## 算子文件结构规范

每个算子文件遵循统一的三段式结构：

| 部分 | 职责 | 示例 |
|------|------|------|
| **Kernel 定义** | `@tilelang.jit` 装饰的 JIT 编译函数 | `_gelu_kernel(M, N, ...)` |
| **Kernel 缓存** | 按 shape/dtype 缓存编译结果，避免重复 JIT | `_get_kernel(M, N, tl_dtype)` |
| **Python 接口** | 与 `proto.yaml` schema 匹配的公开函数 | `gelu(x) -> Tensor` |

### Kernel 编写要点（Ascend NPU）

```python
@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def my_kernel(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),    # 输入
        B: T.Tensor([M, N], dtype),    # 输出（out_idx=[1] 表示第2个参数是输出）
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 1. 分配 UB 缓冲区
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            b = T.alloc_ub([sub_block_M, block_N], dtype)

            # 2. GM → UB 数据搬运
            T.copy(A[...], a)

            # 3. UB 上执行计算（向量指令）
            T.tile.exp(b, a)

            # 4. UB → GM 数据搬运
            T.copy(b, B[...])

    return main
```

**关键 API 对照（Ascend vs CUDA）**：

| 操作 | Ascend TileLang | CUDA TileLang |
|------|----------------|---------------|
| 内存分配 | `T.alloc_ub()` | `T.alloc_shared()` |
| Kernel 启动 | `T.Kernel(..., is_npu=True)` | `T.Kernel(..., threads=128)` |
| 数据搬运 | `T.copy(src, dst)` 显式 | 隐式（shared memory） |
| 向量化计算 | `T.tile.add/exp/sub/...` | `T.Parallel()` |
| 归约 | `T.reduce_max/sum` | `T.reduce_sum` |
| 广播 | `T.tile.broadcast()` | 手动索引 |
| 填充 | `T.tile.fill()` | 手动赋值 |

---

## Python API

```python
import cann_bench

# Softmax 算子
y = cann_bench.softmax(x, dim=-1)

# Exp 算子
y = cann_bench.exp(x, base=-1.0, scale=1.0, shift=0.0)

# 新增算子
y = cann_bench.gelu(x)
```

---

## 评测流程

```
run_evaluation.sh --source-dir examples/tilelang_cann_example
  │
  ├─ build.sh → cann_bench-1.0.0.whl
  ├─ pip install --force-reinstall --no-deps cann_bench-1.0.0.whl
  ├─ import cann_bench → 扫描接口: softmax, exp
  ├─ 匹配 tasks/ 中的算子定义
  │
  └─ 逐算子评测:
      ├─ 加载 cases.yaml 用例
      ├─ 生成输入数据
      ├─ 执行 golden 参考（CPU fp64）
      ├─ 执行 AI 算子（NPU + Profiler）
      ├─ 精度对比（MERE/MARE）
      └─ 性能评分（SOL-Score）
```

---

## 文件职责总结

| 文件 | 职责 |
|------|------|
| `_common.py` | 共享 `PASS_CONFIGS`、`torch_dtype_to_tl()` 等基础配置 |
| `<op>.py` | Kernel 定义 + 缓存 + Python 接口（签名匹配 proto.yaml schema） |
| `__init__.py` | 导出所有算子，评测框架通过 `import cann_bench` 发现接口 |
| `setup.py` | 纯 Python 包配置，`find_packages()` 自动发现模块 |
| `build.sh` | 构建 wheel 包 |
