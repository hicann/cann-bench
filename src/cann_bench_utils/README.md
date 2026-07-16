# CANN Bench Utils

Framework utilities for cann-bench v3 anti-cheat.

## Overview

Provides two custom NPU operators:
- **cann_bench_warmup**: MatMul warmup for NPU frequency boost
- **cann_bench_cache_clean**: ReduceMax for L2 cache flush

These operators replace `torch.matmul` and `torch.max` in the evaluation framework,
allowing the entire built-in kernel tree to be disabled.

## Key Features

- **Specialized naming**: Profiling Type = `CannBenchWarmup` / `CannBenchCacheClean`
- **Simple filtering**: No shape matching required
- **Minimal implementation**: Just enough to boost frequency and flush cache
- **Fixed shapes**: 
  - Warmup: (10240, 10240) @ (10240, 10240), fp16
  - Cache clean: (96, 1024, 1024), fp16

## Build

```bash
# Auto-detect NPU architecture
bash build.sh

# Specify architecture
bash build.sh --soc=ascend910b
bash build.sh --soc=ascend910_93
bash build.sh --soc=ascend950

# Clean build
bash build.sh --clean
```

## Install

```bash
pip install dist/*.whl
```

## Usage

```python
import torch
from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean

# Warmup (boost NPU frequency)
x = torch.randn(10240, 10240, dtype=torch.float16).npu()
y = torch.randn(10240, 10240, dtype=torch.float16).npu()
out = cann_bench_warmup(x, y)

# Cache clean (flush L2)
cache_tensor = torch.randn(96, 1024, 1024, dtype=torch.float16).npu()
out = cann_bench_cache_clean(cache_tensor)
```

## Integration with perf_eval

Replace in `src/kernel_eval/eval/perf_eval.py`:

```python
# Before
torch.matmul(mm1, mm2)
torch.max(reduce_input)

# After
from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean
cann_bench_warmup(mm1, mm2)
cann_bench_cache_clean(reduce_input)
```

## Profiling Filtering

Update `src/kernel_eval/base/perf_strategy.py`:

```python
def _is_warmup_kernel(op_type: str, input_shapes: str = None) -> bool:
    return op_type in ('CannBenchWarmup', 'CannBenchCacheClean')
```

## Directory Structure

```
src/cann_bench_utils/
├── csrc/
│   ├── extension.cpp                      # Python extension entry
│   ├── CMakeLists.txt                     # Operator registration
│   └── ops/
│       ├── warmup/
│       │   ├── op_kernel/
│       │   │   ├── warmup_kernel.cpp      # AscendC kernel
│       │   │   └── warmup_launch.h
│       │   └── op_plugin/
│       │       └── warmup_plugin.cpp      # Torch binding
│       └── cache_clean/
│           ├── op_kernel/
│           │   ├── cache_clean_kernel.cpp
│           │   └── cache_clean_launch.h
│           └── op_plugin/
│               └── cache_clean_plugin.cpp
├── cann_bench_utils/
│   └── __init__.py                        # Python API
├── cmake/                                 # CMake utilities (from direct_launch_example)
├── CMakeLists.txt                         # Top-level CMake
├── setup.py                               # Package setup
├── build.sh                               # Build script
└── README.md                              # This file
```

## Requirements

- CANN toolkit (with bisheng compiler)
- torch >= 2.0
- torch_npu
- Python >= 3.8

## License

Copyright (c) 2026 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
