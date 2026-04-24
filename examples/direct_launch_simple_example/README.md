# Direct Launch Simple Example

简化版 AscendC 自定义算子示例，支持算子自注册机制。相比 direct_launch_example，此版本去掉了 torch.library 的 Meta/NPU 分层注册，更简洁。

## 目录结构

```
direct_launch_simple_example/
├── cann_bench/         # Python包
├── cmake/              # 公共CMake配置(不感知算子)
│   ├── func.cmake      # 注册宏定义
│   └── ...
├── csrc/
│   ├── extension.cpp   # Python扩展入口
│   └── ops/            # 算子目录
│       ├── CMakeLists.txt  # 自动发现算子
│       ├── add/            # Add算子
│       │   ├── CMakeLists.txt  # 算子自注册
│       │   ├── op_kernel/
│       │   │   ├── add_kernel.cpp  # Kernel+Tiling+Launch(bisheng)
│       │   │   └ add_launch.h      # Launch声明
│       │   └── op_plugin/
│       │       └ add_plugin.cpp    # Python bindings(g++)
│       └── sqrt/           # Sqrt算子(结构相同)
├── dist/               # 输出目录
│   └── cann_bench_ops-1.0.0-cp38-abi3-linux_aarch64.whl
├── scripts/
│   └── build_wheel.sh
├── tests/
├── build.sh            # 统一构建入口
└and setup.py
```

## 构建方法

```bash
bash build.sh           # 仅构建
bash build.sh --install # 构建+安装
```

---

## 新增算子详细步骤

### 第一步：创建算子目录结构

```bash
# 以新增 Mul 算子为例
cd csrc/ops
mkdir -p mul/op_kernel
mkdir -p mul/op_plugin
```

**目录说明：**
| 目录 | 用途 | 编译器 |
|------|------|--------|
| `op_kernel/` | Kernel实现 + Tiling计算 + Launch函数 | bisheng (-xasc) |
| `op_plugin/` | Python bindings (PYBIND11) | g++ |

### 第二步：编写op_kernel文件

#### 2.1 mul_kernel.cpp (bisheng编译)

```cpp
/**
 * Mul算子 - Kernel + Tiling + Launch
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

// ========== Kernel实现 ==========
template <typename T>
__global__ __aicore__ void mul_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());
    
    // ... Kernel实现
}

// ========== Tiling计算 ==========
std::tuple<int64_t, int64_t, int64_t> calc_mul_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static int64_t BUFFER_NUM = 3;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

// ========== Launch函数 ==========
extern "C" {

void launch_mul_kernel_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mul_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

void launch_mul_kernel_half(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream)
{
    mul_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, z, totalLength, blockLength, tileSize);
}

}
```

#### 2.2 mul_launch.h

```cpp
#ifndef MUL_LAUNCH_H
#define MUL_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_mul_tiling_params(int64_t totalLength);

extern "C" {
void launch_mul_kernel_float(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_mul_kernel_half(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}

#endif
```

### 第三步：编写op_plugin文件

#### 3.1 mul_plugin.cpp (g++编译)

```cpp
// Mul API - Simple version

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/mul_launch.h"

torch::Tensor mul_npu(const torch::Tensor &x, const torch::Tensor &y) {
    TORCH_CHECK(x.sizes() == y.sizes(), "shapes must match");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "input must be on NPU");

    auto z = torch::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t len = x.numel();
    int64_t blocks, blkLen, tileSz;
    std::tie(blocks, blkLen, tileSz) = calc_mul_tiling_params(len);

    auto launch = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) 
            launch_mul_kernel_float(x.data_ptr(), y.data_ptr(), z.data_ptr(), len, blocks, blkLen, tileSz, stream);
        else if (x.scalar_type() == torch::kFloat16) 
            launch_mul_kernel_half(x.data_ptr(), y.data_ptr(), z.data_ptr(), len, blocks, blkLen, tileSz, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Mul", launch);
    return z;
}

// PYBIND11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mul", &mul_npu, "Mul two tensors on NPU");
}
```

### 第四步：编写算子CMakeLists.txt

**csrc/ops/mul/CMakeLists.txt**：
```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mul算子自注册

# Kernel源文件(bisheng编译)
set(MUL_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mul_kernel.cpp
)

# Plugin源文件(g++编译)
set(MUL_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mul_plugin.cpp
)

# 注册到全局列表
register_simple_op(
    "${MUL_KERNEL_SRCS}"    # Kernel源文件
    op_kernel               # Kernel include目录
    "${MUL_API_SRCS}"       # Plugin源文件
    op_kernel               # Plugin include目录
    "--npu-arch=dav-2201"   # bisheng编译参数
)
```

### 第五步：更新Python包导出

在 `cann_bench/__init__.py` 中添加：

```python
from ._C import mul

# 或直接绑定
def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _C.mul(x, y)
```

### 第六步：重新构建

```bash
bash build.sh --install
```

**无需修改任何公共CMakeLists.txt文件！**

---

## 算子自注册机制原理

### 注册宏定义 (cmake/func.cmake)

```cmake
macro(register_simple_op KERNEL_SRCS KERNEL_INCLUDE_DIR API_SRCS API_INCLUDE_DIR KERNEL_ARGS)
    # 将kernel源文件添加到 ALL_KERNEL_SRCS
    # 将API源文件添加到 ALL_API_SRCS
    # 将include目录添加到全局列表
endmacro()
```

### 自动发现算子 (csrc/ops/CMakeLists.txt)

```cmake
file(GLOB SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${SUB_DIRS})
    if(IS_DIRECTORY ${SUB_DIR})
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

### 公共CMakeLists.txt不感知算子

```cmake
# Kernel(bisheng)
set_source_files_properties(${ALL_KERNEL_SRCS} PROPERTIES COMPILE_FLAGS "--npu-arch=dav-2201 -xasc")
add_library(all_kernels_obj OBJECT ${ALL_KERNEL_SRCS})

# Plugin(g++)
add_library(all_api_obj OBJECT ${ALL_API_SRCS})

# 合并为共享库
add_library(_C SHARED ${OBJECTS_LIST} $<TARGET_OBJECTS:all_kernels_obj> $<TARGET_OBJECTS:all_api_obj>)
```

---

## Python API

```python
import cann_bench

# 简化版API - 直接调用
z = cann_bench.add(x, y)  # Add算子
r = cann_bench.sqrt(x)    # Sqrt算子
m = cann_bench.mul(x, y)  # 新增Mul算子
```

---

## 与 direct_launch_example 的区别

| 特性 | simple_example | direct_launch_example |
|------|----------------|----------------------|
| Python绑定 | PYBIND11_MODULE | torch.library |
| Meta注册 | 无 | 有 |
| Dispatch机制 | 直接函数调用 | torch.ops.cann_bench.* |
| 适用场景 | 快速原型验证 | 生产级封装 |

---

## 文件职责总结

| 文件 | 编译器 | 职责 |
|------|--------|------|
| `op_kernel/*.cpp` | bisheng | Kernel + Tiling + Launch extern "C" |
| `op_kernel/*.h` | bisheng/g++ | Launch函数声明 |
| `op_plugin/*.cpp` | g++ | PYBIND11绑定 + NPU实现 |
| `CMakeLists.txt` | cmake | 调用register_simple_op() |