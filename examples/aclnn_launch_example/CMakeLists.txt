# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(custom_ops_prj VERSION 1.0.0)

set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)

# ================================== ACLNN Operator Build ==================================
find_package(ASC REQUIRED)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    message(FATAL_ERROR "ASCEND_COMPUTE_UNIT is not set. Pass -DASCEND_COMPUTE_UNIT=<soc_version> to cmake.")
endif()
set(package_name custom_ops)

# Output to dist directory
set(DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${DIST_DIR}
        PACKAGE_NAME cann_bench
)

# 包含注册宏定义
include(cmake/func.cmake)

# 初始化全局变量(在func.cmake中已定义，这里确保清空)
set(ALL_HOST_OPS_SRCS "" CACHE INTERNAL "All host source files")
set(ALL_API_OPS_SRCS "" CACHE INTERNAL "All API source files")
set(ALL_KERNEL_OPS_INFO "" CACHE INTERNAL "All kernel info")
set(ALL_TILING_INCLUDE_DIRS "" CACHE INTERNAL "All tiling include directories")
set(ALL_API_INCLUDE_DIRS "" CACHE INTERNAL "All API include directories")
set(ALL_PLUGIN_SRCS "" CACHE INTERNAL "All plugin source files")
set(ALL_PLUGIN_INCLUDE_DIRS "" CACHE INTERNAL "All plugin include directories")

# 添加算子目录，算子自注册到全局列表
add_subdirectory(csrc/ops)

message(STATUS "Building custom ops for: ${ASCEND_COMPUTE_UNIT}")
message(STATUS "Registered host sources: ${ALL_HOST_OPS_SRCS}")
message(STATUS "Registered API sources: ${ALL_API_OPS_SRCS}")
message(STATUS "Registered kernel info: ${ALL_KERNEL_OPS_INFO}")
message(STATUS "Registered tiling include dirs: ${ALL_TILING_INCLUDE_DIRS}")
message(STATUS "Registered API include dirs: ${ALL_API_INCLUDE_DIRS}")

# ================================== 处理注册的算子 ==================================

# Run autogen
npu_op_code_gen(
    SRC ${ALL_HOST_OPS_SRCS}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include
        -I$ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
        -I$ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc
        -I$ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
        -I$ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

# Tiling library
npu_op_library(cust_optiling TILING ${ALL_HOST_OPS_SRCS})
target_include_directories(cust_optiling PRIVATE
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
    ${ALL_TILING_INCLUDE_DIRS}
)

# ACLNN API library
npu_op_library(cust_opapi ACLNN ${ALL_API_OPS_SRCS})
target_include_directories(cust_opapi PRIVATE
    ${ALL_API_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

# Graph proto library (InferShape + OpDef registration)
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH
    ${ALL_HOST_OPS_SRCS}
    ${proto_src}
)
target_include_directories(cust_op_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
    ${ALL_TILING_INCLUDE_DIRS}
)

# Kernel sources - 从注册信息生成
foreach(KERNEL_INFO ${ALL_KERNEL_OPS_INFO})
    # 解析格式: OP_TYPE|KERNEL_DIR|KERNEL_FILE
    string(REPLACE "|" ";" KERNEL_PARTS ${KERNEL_INFO})
    list(GET KERNEL_PARTS 0 KERNEL_OP_TYPE)
    list(GET KERNEL_PARTS 1 KERNEL_DIR)
    list(GET KERNEL_PARTS 2 KERNEL_FILE)
    message(STATUS "Adding kernel: ${KERNEL_OP_TYPE}, dir=${KERNEL_DIR}, file=${KERNEL_FILE}")

    npu_op_kernel_sources(all_kernels
        OP_TYPE ${KERNEL_OP_TYPE}
        KERNEL_DIR ${KERNEL_DIR}
        KERNEL_FILE ${KERNEL_FILE}
    )
endforeach()

# Kernel library
npu_op_kernel_library(all_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY cust_optiling cust_opapi cust_op_proto all_kernels
)

# 将 libopapi.so 声明为 libcust_opapi.so 的 NEEDED 依赖。
# 自定义算子的 L2 API（如 aclnn_add.cpp）调用内置 L0 算子
# l0op::Contiguous / l0op::ViewCopy 等，这些符号定义在 libopapi.so。
# CANN 构建系统默认不把 opapi 加入 NEEDED 链，导致 .run 包的
# install.sh 验证步骤（dlopen libcust_opapi.so + RTLD_NOW）
# 无法解析这些符号而失败。显式声明依赖后，动态链接器在加载
# libcust_opapi.so 时会自动加载 libopapi.so，验证可正常通过。
if(TARGET ${package_name}_ascendc_cust_opapi)
    target_link_directories(${package_name}_ascendc_cust_opapi PRIVATE
        $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
    )
    target_link_libraries(${package_name}_ascendc_cust_opapi PRIVATE opapi)
    message(STATUS "Added libopapi.so as NEEDED dependency of libcust_opapi.so")
endif()

# ================================== Python Extension Build ==================================
option(BUILD_PYTHON_EXT "Build Python extension module" OFF)

if(BUILD_PYTHON_EXT)
    message(STATUS "Building Python extension module")

    include(cmake/ascend.cmake)
    include(cmake/python.cmake)
    include(cmake/torch.cmake)
    include(cmake/torch_npu.cmake)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(EXTENSION_MODULE_NAME "cann_bench" CACHE STRING "Extension module name")

    set(INCLUDE_DIRECTORIES
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${TORCH_NPU_INCLUDE_PATH}
        ${ASCEND_INCLUDE_DIRS}
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
    )

    set(LINK_DIRECTORIES
        ${TORCH_NPU_LIB_PATH}
        ${ASCEND_DIR}/lib64
        $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
    )

    set(LINK_LIBRARIES
        ${TORCH_LIBRARIES}
        torch_npu
        ascendcl
        opapi
        platform
        register
        tiling_api
        runtime
    )

    set(COMPILE_OPTIONS
        ${TORCH_CXX_FLAGS}
        -O3
        -fdiagnostics-color=always
        -w
        -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
    )

    # Build plugin from registered sources
    message(STATUS "Registered plugin sources: ${ALL_PLUGIN_SRCS}")
    message(STATUS "Registered plugin include dirs: ${ALL_PLUGIN_INCLUDE_DIRS}")

    set(PLUGIN_TARGET cann_bench_plugin_obj)
    add_library(${PLUGIN_TARGET} OBJECT ${ALL_PLUGIN_SRCS})
    target_compile_options(${PLUGIN_TARGET} PRIVATE ${COMPILE_OPTIONS})
    target_include_directories(${PLUGIN_TARGET} PRIVATE
        ${ALL_PLUGIN_INCLUDE_DIRS}
        ${INCLUDE_DIRECTORIES}
        $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include
        $ENV{ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/include/aclnn
    )

    # Create shared library
    set(EXTENSION_CPP ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp)
    add_library(_C SHARED
        ${EXTENSION_CPP}
        $<TARGET_OBJECTS:${PLUGIN_TARGET}>
    )
    set_target_properties(_C PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PREFIX ""
        SUFFIX ".abi3.so"
        OUTPUT_NAME "_C"
    )
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
    target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
    target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
    target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
    target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})

    add_custom_command(TARGET _C POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:_C>
        ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
        COMMENT "Copying compiled extension to ${EXTENSION_MODULE_NAME}/"
    )
endif()