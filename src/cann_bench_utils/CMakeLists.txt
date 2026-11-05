# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME cann_bench_utils)
project(${PKG_NAME} VERSION 1.0.0)

# CMake >= 3.31 passes --dependency-file to the linker by default
# (CMAKE_LINK_DEPENDS_USE_LINKER=ON). The Bisheng linker (cce-ld) does not
# support this flag and fails the link step with
# "ignoring unkown argurment: --dependency-file=...", so disable it here.
# Must be set before any add_library()/add_executable(): it initializes the
# LINK_DEPENDS_USE_LINKER target property. Harmless on CMake < 3.31.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

# NPU architecture
set(NPU_ARCH "ascend910b" CACHE STRING "NPU architecture")
set_property(CACHE NPU_ARCH PROPERTY STRINGS "ascend910b" "ascend910_93" "ascend950")

# Map NPU_ARCH to Bisheng --npu-arch flag
unset(BISHENG_NPU_ARCH)
if(NPU_ARCH STREQUAL "ascend910b" OR NPU_ARCH STREQUAL "ascend910_93")
    set(BISHENG_NPU_ARCH "dav-2201")
elseif(NPU_ARCH STREQUAL "ascend950")
    set(BISHENG_NPU_ARCH "dav-3510")
else()
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}")
endif()
message(STATUS "NPU_ARCH: ${NPU_ARCH} -> Bisheng --npu-arch=${BISHENG_NPU_ARCH}")

set(EXTENSION_MODULE_NAME "cann_bench_utils" CACHE STRING "Extension module name")

# include directories
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
)

# link directories
set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
)

# link libraries
set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

# compile options
set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
)

# Initialize global variables
set(ALL_KERNEL_SRCS "" CACHE INTERNAL "All kernel source files")
set(ALL_KERNEL_INCLUDE_DIRS "" CACHE INTERNAL "All kernel include directories")
set(ALL_PLUGIN_SRCS "" CACHE INTERNAL "All plugin source files")
set(ALL_PLUGIN_INCLUDE_DIRS "" CACHE INTERNAL "All plugin include directories")

# Add operator directories
add_subdirectory(csrc)

message(STATUS "Registered kernel sources: ${ALL_KERNEL_SRCS}")
message(STATUS "Registered plugin sources: ${ALL_PLUGIN_SRCS}")

# ========== Kernel compilation (bisheng) ==========
set(_SAVED_CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
set(_SAVED_CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
set(_SAVED_CMAKE_LINKER ${CMAKE_LINKER})
set(_SAVED_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})
unset(CMAKE_CXX_FLAGS)

set(KERNEL_COMPILE_FLAGS "--npu-arch=${BISHENG_NPU_ARCH} -xasc")
set_source_files_properties(${ALL_KERNEL_SRCS} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "${KERNEL_COMPILE_FLAGS}")

set(KERNEL_TARGET all_kernels_obj)
add_library(${KERNEL_TARGET} OBJECT ${ALL_KERNEL_SRCS})
target_include_directories(${KERNEL_TARGET} PRIVATE ${ALL_KERNEL_INCLUDE_DIRS} ${INCLUDE_DIRECTORIES})

set(CMAKE_C_COMPILER ${_SAVED_CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${_SAVED_CMAKE_CXX_COMPILER})
set(CMAKE_LINKER ${_SAVED_CMAKE_LINKER})
set(CMAKE_CXX_FLAGS ${_SAVED_CMAKE_CXX_FLAGS})

# ========== Plugin compilation (g++) ==========
set(PLUGIN_TARGET all_plugins_obj)
add_library(${PLUGIN_TARGET} OBJECT ${ALL_PLUGIN_SRCS})
target_compile_options(${PLUGIN_TARGET} PRIVATE ${COMPILE_OPTIONS})
target_include_directories(${PLUGIN_TARGET} PRIVATE ${ALL_PLUGIN_INCLUDE_DIRS})
target_include_directories(${PLUGIN_TARGET} SYSTEM PRIVATE ${INCLUDE_DIRECTORIES})

# ========== Create shared library ==========
set(EXTENSION_CPP ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp)
add_library(_C SHARED
    ${EXTENSION_CPP}
    $<TARGET_OBJECTS:${KERNEL_TARGET}>
    $<TARGET_OBJECTS:${PLUGIN_TARGET}>
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "_C"
 )
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C SYSTEM PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})

# Bisheng may otherwise select a libc linker script referring to /lib64/libc.so.6
# on hosts where glibc lives in a multiarch directory. Keep this explicit -L
# before the toolkit library directories; link_directories alone can omit
# directories that CMake considers implicit.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(HOST_GCC_EXECUTABLE NAMES gcc)
    if(NOT HOST_GCC_EXECUTABLE)
        message(WARNING "gcc not found; keeping the default libc search paths")
    else()
        execute_process(
            COMMAND "${HOST_GCC_EXECUTABLE}" -print-file-name=libc.so
            OUTPUT_VARIABLE HOST_LIBC_SCRIPT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE HOST_LIBC_RESULT
        )
        if(HOST_LIBC_RESULT STREQUAL "0" AND IS_ABSOLUTE "${HOST_LIBC_SCRIPT}"
           AND EXISTS "${HOST_LIBC_SCRIPT}")
            get_filename_component(HOST_LIBC_SCRIPT "${HOST_LIBC_SCRIPT}" REALPATH)
            get_filename_component(HOST_LIBC_DIR "${HOST_LIBC_SCRIPT}" DIRECTORY)
            target_link_options(_C BEFORE PRIVATE "-L${HOST_LIBC_DIR}")
            message(STATUS "Using host libc directory: ${HOST_LIBC_DIR}")
        else()
            message(WARNING "Cannot locate host libc.so using gcc: ${HOST_LIBC_SCRIPT}; keeping the default libc search paths")
        endif()
    endif()
endif()

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying compiled extension to ${EXTENSION_MODULE_NAME}/"
)
