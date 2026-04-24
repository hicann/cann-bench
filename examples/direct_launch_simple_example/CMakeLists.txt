# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(cann_bench_ops VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

# 输出目录
set(DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)

set(INCLUDE_DIRS ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_INCLUDE_PATH} ${ASCEND_INCLUDE_DIRS})
set(LINK_DIRS ${TORCH_NPU_LIB_PATH} ${ASCEND_DIR}/lib64)
set(LINK_LIBS ${TORCH_LIBRARIES} torch_npu ascendcl platform register tiling_api runtime)
set(COMPILE_OPTS ${TORCH_CXX_FLAGS} -O3 -w)

# 初始化全局变量
set(ALL_KERNEL_SRCS "" CACHE INTERNAL "All kernel sources")
set(ALL_KERNEL_INCLUDE_DIRS "" CACHE INTERNAL "All kernel include dirs")
set(ALL_API_SRCS "" CACHE INTERNAL "All API sources")
set(ALL_API_INCLUDE_DIRS "" CACHE INTERNAL "All API include dirs")
set(ALL_KERNEL_ARGS "" CACHE INTERNAL "Kernel compile args")

# 添加算子目录(自注册)
set(OBJECTS_LIST "" CACHE INTERNAL "objects")
add_subdirectory(csrc)

message(STATUS "Registered kernel sources: ${ALL_KERNEL_SRCS}")
message(STATUS "Registered API sources: ${ALL_API_SRCS}")

# ========== Kernel编译(bisheng) ==========
set(_SAVED_C ${CMAKE_C_COMPILER})
set(_SAVED_CX ${CMAKE_CXX_COMPILER})
set(_SAVED_L ${CMAKE_LINKER})
set(_SAVED_F ${CMAKE_CXX_FLAGS})
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})
unset(CMAKE_CXX_FLAGS)

set_source_files_properties(${ALL_KERNEL_SRCS} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "--npu-arch=dav-2201 -xasc")
add_library(all_kernels_obj OBJECT ${ALL_KERNEL_SRCS})
target_include_directories(all_kernels_obj PRIVATE ${ALL_KERNEL_INCLUDE_DIRS} ${INCLUDE_DIRS})

set(CMAKE_C_COMPILER ${_SAVED_C})
set(CMAKE_CXX_COMPILER ${_SAVED_CX})
set(CMAKE_LINKER ${_SAVED_L})
set(CMAKE_CXX_FLAGS ${_SAVED_F})

# ========== API编译(g++) ==========
add_library(all_api_obj OBJECT ${ALL_API_SRCS})
target_compile_options(all_api_obj PRIVATE ${COMPILE_OPTS})
target_include_directories(all_api_obj PRIVATE ${ALL_KERNEL_INCLUDE_DIRS} ${ALL_API_INCLUDE_DIRS} ${INCLUDE_DIRS})

# ========== 合并到OBJECTS_LIST ==========
set(TEMP ${OBJECTS_LIST})
list(APPEND TEMP $<TARGET_OBJECTS:all_kernels_obj> $<TARGET_OBJECTS:all_api_obj>)
set(OBJECTS_LIST ${TEMP} CACHE INTERNAL "objects" FORCE)

# ========== 创建共享库 ==========
add_library(_C SHARED ${OBJECTS_LIST})
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX ".so")
target_compile_options(_C PRIVATE ${COMPILE_OPTS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRS})
target_link_directories(_C PRIVATE ${LINK_DIRS})
target_link_libraries(_C PRIVATE ${LINK_LIBS})