# CANN Bench Utils - Quick Start

## 🎯 Overview

Two custom NPU operators for cann-bench v3 anti-cheat:
- **cann_bench_warmup**: MatMul (10240×10240, fp16) → boost NPU frequency
- **cann_bench_cache_clean**: ReduceMax (96×1024×1024, fp16) → flush L2 cache

**Key Feature**: Uses specialized profiling names (`CannBenchWarmup` / `CannBenchCacheClean`) for simple filtering without shape matching.

---

## 📦 Build & Install

### Prerequisites

```bash
# Ensure CANN environment is sourced
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Verify
python3 -c "import torch; import torch_npu; import acl; print('✓ Ready')"
```

### Build

```bash
cd "$(git rev-parse --show-toplevel)/src/cann_bench_utils"

# Build (auto-detects NPU architecture)
bash build.sh

# Or specify architecture
bash build.sh --soc=ascend910b
bash build.sh --soc=ascend910_93
bash build.sh --soc=ascend950
```

### Install

Install the matching `torch` / `torch_npu` pair from the repository's
`requirements.txt` before building. Keep that same environment for installation
and execution: these native wheels use its ABI. `--no-deps` prevents reinstalling
an operator wheel from replacing that pair (also use it for the ACLNN example).

```bash
python -m pip install dist/*.whl --force-reinstall --no-deps
```

### Test

```bash
python3 test_ops.py
```

Expected output:
```
============================================================
CANN Bench Utils - Operator Tests
============================================================
Testing cann_bench_warmup...
✓ cann_bench_warmup passed
Testing cann_bench_cache_clean...
✓ cann_bench_cache_clean passed

============================================================
All tests passed! ✓
============================================================
```

---

## 🧪 Manual Testing

```python
import torch
import torch_npu
from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean

# Test warmup
x = torch.randn(10240, 10240, dtype=torch.float16).npu()
y = torch.randn(10240, 10240, dtype=torch.float16).npu()
out = cann_bench_warmup(x, y)
print(f"Warmup output shape: {out.shape}")  # (10240, 10240)

# Test cache clean
cache_tensor = torch.randn(96, 1024, 1024, dtype=torch.float16).npu()
out = cann_bench_cache_clean(cache_tensor)
print(f"Cache clean output shape: {out.shape}")  # torch.Size([])
```

---

## 🔧 Troubleshooting

### Build fails with "bisheng not found"

```bash
# Check CANN environment
which bisheng
# Should output: /usr/local/Ascend/.../bisheng

# Re-source environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 链接失败：cce-ld 报 "ignoring unknown argument: --dependency-file=..."

**原因**：CMake 3.31 起默认向链接器传递 `--dependency-file` 参数
（`CMAKE_LINK_DEPENDS_USE_LINKER=ON`），而毕昇（bisheng）工具链的链接器
cce-ld 不支持该参数，导致 `_C.so` 链接失败。

**解决**：本项目 `CMakeLists.txt` 已内置修复（在所有 `add_library()` 之前设置
`CMAKE_LINK_DEPENDS_USE_LINKER=FALSE`，对旧版本 CMake 无副作用），
拉取最新代码后重新执行 `bash build.sh --clean` 即可。

**版本兼容性说明**：

| 组件 | 要求 |
| --- | --- |
| CMake | >= 3.16（3.31+ 需使用含上述修复的 CMakeLists.txt） |
| 毕昇编译器（bisheng） | 随 CANN toolkit 提供，建议 CANN >= 8.0 |

若你自行维护基于本工程修改的 CMake 工程，请在 `project()` 之后、
任何 `add_library()`/`add_executable()` 之前加上：

```cmake
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
```

### Build fails with "torch_npu not found"

```bash
pip install torch_npu
```

### Wrong NPU architecture detected

```bash
# Check detected SoC
python3 -c "import acl; print(acl.get_soc_name())"

# Force specific architecture
bash build.sh --soc=ascend910_93
```

### Import error after install

```bash
# Reinstall
python -m pip install dist/*.whl --force-reinstall --no-deps

# Verify
python3 -c "from cann_bench_utils import cann_bench_warmup; print('✓ OK')"
```

---

## 📝 Implementation Notes

### Kernel Simplification

These are **NOT** full implementations:
- **cann_bench_warmup**: Elementwise multiply (placeholder for matmul)
- **cann_bench_cache_clean**: Reads all data (placeholder for reduce)

**Why it's OK**: These operators are only for warmup, not in the measurement path. Performance requirements are minimal.

### Profiling Naming

Critical for v3 anti-cheat filtering:

```cpp
// warmup_plugin.cpp
at_npu::native::OpCommand::RunOpApi("CannBenchWarmup", acl_call);

// cache_clean_plugin.cpp
at_npu::native::OpCommand::RunOpApi("CannBenchCacheClean", acl_call);
```

This sets the `Type` field in `kernel_details.csv`, allowing simple filtering:

```python
def _is_warmup_kernel(op_type: str, input_shapes: str = None) -> bool:
    return op_type in ('CannBenchWarmup', 'CannBenchCacheClean')
```

### Fixed Shapes

Operators enforce fixed shapes at compile time:
- Warmup: `M=K=N=10240` (constants in `warmup_kernel.cpp`)
- Cache clean: `TOTAL_SIZE=96*1024*1024` (constant in `cache_clean_kernel.cpp`)

Runtime shape validation in `*_meta()` functions ensures correct usage.

---

## 🚀 Next Steps

After successful build and test:

1. **Integrate into perf_eval** - see `scripts/anti_cheat/v3_code_changes.md`
2. **Update perf_strategy filtering** - simplify `_is_warmup_kernel()`
3. **Test full evaluation flow** - run with v3 kernel disable script
4. **Verify profiling** - check `kernel_details.csv` contains correct Type names

---

## 📚 Reference

- **Full proposal**: `scripts/anti_cheat/v3_proposal.md`
- **Code changes**: `scripts/anti_cheat/v3_code_changes.md`
- **Naming example**: `scripts/anti_cheat/v3_kernel_naming_example.cpp`
- **Direct launch reference**: `examples/direct_launch_example/`

---

## ✅ Status

- [x] Minimal kernel implementations
- [x] Specialized profiling naming
- [x] Build system (CMake + setup.py)
- [x] Structure verification script
- [x] Basic functional tests
- [ ] Integration with perf_eval (next)
- [ ] Full evaluation regression test (next)
- [ ] v3 kernel disable script (next)
