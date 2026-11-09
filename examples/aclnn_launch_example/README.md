# ACLNN Launch Example

基于 registry-invoke 模式的 ACLNN 自定义算子示例工程，支持算子自注册机制和Python API接口。

## 目录结构

```
aclnn_launch_example/
├── cann_bench/         # Python包
├── cmake/              # 公共CMake配置(不感知算子)
│   ├── func.cmake      # 注册宏定义
│   └── ...
├── csrc/
│   ├── extension.cpp   # Python扩展入口
│   └── ops/            # 算子目录
│       ├── CMakeLists.txt  # 自动发现算子
│       ├── op_kernel/      # 统一kernel目录(硬链接)
│       ├── add/            # Add算子
│       │   ├── CMakeLists.txt  # 算子自注册
│       │   ├── op_host/
│       │   │   ├── add_def.cpp      # 算子定义
│       │   │   ├── add_infershape.cpp # Shape推导
│       │   │   └ arch22/add_tiling.cpp  # Tiling实现
│       │   ├── op_kernel/
│       │   │   └ arch22/
│       │   │       ├── add.h            # Kernel头文件
│       │   │       ├── add_tiling_data.h # Tiling数据结构
│       │   │       └ add_tiling_key.h    # Tiling Key定义
│       │   ├── op_api/
│       │   │   ├── aclnn_add.cpp    # L2 API
│       │   │   └ add.cpp            # L0 API
│       │   └── op_plugin/
│       │       └ add_plugin.cpp     # Python bindings
│       └── sqrt/           # Sqrt算子(结构相同)
├── dist/               # 输出目录
│   ├── cann_bench_YYYYMMDD_linux_aarch64.run  # run包
│   └── cann_bench-1.0.0-*.whl                  # wheel包
├── scripts/            # 打包脚本
│   ├── build_run.sh
│   └── build_wheel.sh
├── CMakeLists.txt      # 顶层CMake配置
├── requirements.txt    # Python依赖
├── build.sh            # 统一构建入口
└── setup.py
```

## 构建方法

> **环境要求：CANN 9.1.0 及以上版本。** 本示例的算子代码引用了 `aclnn_kernels/contiguous.h` 头文件（见 `csrc/ops/*/op_api/aclnn_*.cpp`），该头文件仅存在于 CANN 9.1.0+；编译前请 source 对应版本的环境脚本（如 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`），确保 `ASCEND_HOME_PATH` 指向 9.1.0+ 的 CANN 安装目录，否则编译失败。

```bash
bash build.sh --soc=ascend910b
# 或指定其他 SoC 如 --soc=ascend910_93 / --soc=ascend950
```

输出到 `dist/` 目录:
- `cann_bench_YYYYMMDD_linux_aarch64.run` - ACLNN算子安装包
- `cann_bench-1.0.0-cp38-abi3-linux_aarch64.whl` - Python wheel包

---

## 新增算子详细步骤

> ℹ️ **关于待评测算子与内置算子**
>
> 为保证公平，评测机会禁用被评测算子的 CANN 内置 kernel 二进制（见 `scripts/anti_cheat/`），
> 因此提交无法通过调用同名 stock 算子（`aclnn<Op>`）"蹭"内置实现——必须自带 kernel。
> 注意区分：kernel 内的 `AscendC::Add` 等是**设备侧 intrinsic**，编译进你的 kernel，
> 与内置算子二进制无关，不受影响。下面以 `Mul` 为例。

### 第一步：创建算子目录结构

```bash
# 以新增 Mul 算子为例
cd csrc/ops
mkdir -p mul/op_host/arch22
mkdir -p mul/op_kernel/arch22
mkdir -p mul/op_api
mkdir -p mul/op_plugin
```

**目录说明：**
| 目录 | 用途 | 编译器 |
|------|------|--------|
| `op_host/` | 算子定义、Shape推导、Tiling | bisheng |
| `op_kernel/arch22/` | Kernel头文件(tiling_data.h, tiling_key.h, *.h) | - |
| `op_api/` | ACLNN L2/L0 API | bisheng |
| `op_plugin/` | Python bindings (torch.library) | g++ |

### 第二步：编写算子源文件

#### 2.1 op_host目录

**op_host/mul_def.cpp** - 算子定义：
```cpp
#include "register/op_def_registry.h"

namespace ops {
OP_DEF(Mul)
    .Input("x1", TensorType({DT_FLOAT, DT_FLOAT16}))
    .Input("x2", TensorType({DT_FLOAT, DT_FLOAT16}))
    .Output("y", TensorType({DT_FLOAT, DT_FLOAT16}))
    .ExtendCfgInfo("opFile.value", "mul_arch22")
    .ExtendCfgInfo("opComputeUnit.value", "AiCore");
}
```

**op_host/arch22/mul_tiling.cpp** - Tiling实现：
```cpp
#include "register/op_def_registry.h"
#include "../../op_kernel/arch22/mul_tiling_data.h"
#include "../../op_kernel/arch22/mul_tiling_key.h"

namespace optiling {
IMPL_OP_OPTILING(Mul).Tiling(MulTilingFunc).TilingParse<MulCompileInfo>(TilingParseForMul);
}
```

#### 2.2 op_kernel/arch22目录

**mul_tiling_data.h** - Tiling数据结构：
```cpp
struct MulTilingData {
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;
};
```

**mul_tiling_key.h** - Tiling Key定义：
```cpp
// 定义kernel选择Key
```

**mul.h** - Kernel头文件：
```cpp
#include "kernel_operator.h"
#include "mul_tiling_data.h"

namespace NsMul {
template <typename T, int BUFFER_MODE>
class Mul {
    // Kernel实现
};
}
```

#### 2.3 op_api目录

**op_api/aclnn_mul.cpp** - L2 API：
```cpp
#include "aclnn_mul.h"
// aclnnMulGetWorkspaceSize, aclnnMul 实现
```

**op_api/mul.cpp** - L0 API：
```cpp
#include "mul.h"
namespace l0op {
const aclTensor* Mul(const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor);
}
```

#### 2.4 op_plugin目录

**op_plugin/mul_plugin.cpp** - Python bindings：
```cpp
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_api/aclnn_mul.h"

namespace cann_bench {
TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("mul(Tensor x, Tensor y) -> Tensor");
}
// Meta和NPU实现...
}
```

### 第三步：创建硬链接到统一kernel目录

```bash
# 在 csrc/ops/op_kernel/ 创建硬链接
cd csrc/ops/op_kernel

# 链接kernel入口文件(假设为 mul_arch22.cpp)
ln -f ../mul/op_kernel/arch22/mul_arch22.cpp mul_arch22.cpp

# 链接头文件
ln -f ../mul/op_kernel/arch22/mul.h mul.h
ln -f ../mul/op_kernel/arch22/mul_tiling_data.h mul_tiling_data.h
ln -f ../mul/op_kernel/arch22/mul_tiling_key.h mul_tiling_key.h
```

### 第四步：编写算子CMakeLists.txt

**csrc/ops/mul/CMakeLists.txt**：
```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mul算子自注册

# Host源文件
set(MUL_HOST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mul_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/mul_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/arch22/mul_tiling.cpp
)

# API源文件
set(MUL_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/mul.cpp
)

# 注册ACLNN算子
register_aclnn_op(
    Mul                      # 算子类型名
    "${MUL_HOST_SRCS}"       # Host源文件列表
    "${MUL_API_SRCS}"        # API源文件列表
    op_kernel                # Kernel目录名(相对于算子目录)
    mul_arch22.cpp           # Kernel入口文件名
    op_kernel/arch22         # Tiling include目录
    op_api                   # API include目录
)

# 注册Python插件(可选)
set(MUL_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mul_plugin.cpp
)
register_aclnn_plugin("${MUL_PLUGIN_SRCS}" op_api)
```

### 第五步：重新构建

```bash
bash build.sh --soc=ascend910b
# 或指定其他 SoC 如 --soc=ascend910_93 / --soc=ascend950
```

**无需修改任何公共CMakeLists.txt文件！**

---

## 算子自注册机制原理

### 注册宏定义 (cmake/func.cmake)

```cmake
# 注册ACLNN算子到全局列表
macro(register_aclnn_op OP_TYPE HOST_SRCS API_SRCS KERNEL_DIR KERNEL_FILE TILING_INCLUDE_DIR API_INCLUDE_DIR)
    # 将源文件添加到全局变量 ALL_HOST_OPS_SRCS, ALL_API_OPS_SRCS
    # 将kernel信息添加到 ALL_KERNEL_OPS_INFO
    # 将include目录添加到全局列表
endmacro()

# 注册Python插件
macro(register_aclnn_plugin PLUGIN_SRCS PLUGIN_INCLUDE_DIR)
    # 将插件源文件添加到 ALL_PLUGIN_SRCS
endmacro()
```

### 自动发现算子 (csrc/ops/CMakeLists.txt)

```cmake
# 遍历所有子目录，排除op_kernel统一目录
file(GLOB SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${SUB_DIRS})
    if(IS_DIRECTORY ${SUB_DIR} AND NOT DIR_NAME STREQUAL "op_kernel")
        add_subdirectory(${SUB_DIR})  # 调用算子的CMakeLists.txt
    endif()
endforeach()
```

### 公共CMakeLists.txt不感知算子

顶层 `CMakeLists.txt` 仅使用全局变量：
```cmake
# 使用注册的源文件构建
npu_op_code_gen(SRC ${ALL_HOST_OPS_SRCS} ...)
npu_op_library(cust_optiling TILING ${ALL_HOST_OPS_SRCS})
npu_op_library(cust_opapi ACLNN ${ALL_API_OPS_SRCS})
foreach(KERNEL_INFO ${ALL_KERNEL_OPS_INFO})
    npu_op_kernel_sources(all_kernels ...)  # 为每个kernel生成ini
endforeach()
```

---

## 安装使用

### run包安装
```bash
# --quiet：非交互安装(Makeself 选项，自动接受 EULA)。勿用 --install(非法选项，会跳过安装)
./dist/cann_bench_*.run --quiet
# 安装后按提示将 op_api/lib 加入库搜索路径：
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_ops/op_api/lib/:${LD_LIBRARY_PATH}
```

### wheel包安装
```bash
python -m pip install dist/cann_bench-*.whl --no-deps
```

> 构建前请按仓库根目录 `requirements.txt` 安装匹配的 `torch` / `torch_npu`，并在同一环境中构建、安装和运行。wheel 复用该环境的原生 ABI，不自行安装这两个依赖；重装时也请保留 `--no-deps`，避免替换已匹配的版本。

> ⚠️ **包名冲突提示**：本示例构建的 wheel 与仓库 golden 自验证 wheel（`scripts/utils/build_golden_wheel.sh` 产物）同名，都叫 `cann_bench`，二者互斥、不能同时安装。若当前环境已安装 golden wheel，调用 `cann_bench.add()` 会报 `module 'cann_bench' has no attribute 'add'`，请先 `pip uninstall -y cann_bench` 卸载后再安装本示例 wheel；或将本工程目录加入 `PYTHONPATH` 以优先导入本包。

### Python API
```python
import cann_bench
z = cann_bench.add(x, y)          # Add算子
r = cann_bench.sqrt(x)            # Sqrt算子
m = cann_bench.mul(x, y)          # 新增Mul算子
# 或通过torch.ops调用
z = torch.ops.cann_bench.add(x, y)
```