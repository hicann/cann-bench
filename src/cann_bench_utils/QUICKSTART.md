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
cd /mnt/workspace/gitCode/cann/cann-bench/src/cann_bench_utils

# Build (auto-detects NPU architecture)
bash build.sh

# Or specify architecture
bash build.sh --soc=ascend910b
bash build.sh --soc=ascend910_93
bash build.sh --soc=ascend950
```

### Install

```bash
pip install dist/*.whl --force-reinstall
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
pip install dist/*.whl --force-reinstall

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
