# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# 简化版注册宏
# 算子目录通过调用register_simple_op()将自己注册到全局列表

# 全局变量
set(ALL_KERNEL_SRCS "" CACHE INTERNAL "All kernel sources")
set(ALL_KERNEL_INCLUDE_DIRS "" CACHE INTERNAL "All kernel include dirs")
set(ALL_API_SRCS "" CACHE INTERNAL "All API sources")
set(ALL_API_INCLUDE_DIRS "" CACHE INTERNAL "All API include dirs")
set(ALL_KERNEL_ARGS "" CACHE INTERNAL "Kernel compile args")

# 注册简化版算子
macro(register_simple_op KERNEL_SRCS KERNEL_INCLUDE_DIR API_SRCS API_INCLUDE_DIR KERNEL_ARGS)
    get_filename_component(OP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(STATUS "Registering simple op: ${OP_NAME}")

    # Kernel源文件
    set(_TEMP ${ALL_KERNEL_SRCS})
    list(APPEND _TEMP ${KERNEL_SRCS})
    set(ALL_KERNEL_SRCS ${_TEMP} CACHE INTERNAL "All kernel sources")

    # Kernel include
    set(_TEMP_INC ${ALL_KERNEL_INCLUDE_DIRS})
    list(APPEND _TEMP_INC ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_INCLUDE_DIR})
    set(ALL_KERNEL_INCLUDE_DIRS ${_TEMP_INC} CACHE INTERNAL "All kernel include dirs")

    # API源文件
    set(_TEMP_API ${ALL_API_SRCS})
    list(APPEND _TEMP_API ${API_SRCS})
    set(ALL_API_SRCS ${_TEMP_API} CACHE INTERNAL "All API sources")

    # API include
    set(_TEMP_API_INC ${ALL_API_INCLUDE_DIRS})
    list(APPEND _TEMP_API_INC ${CMAKE_CURRENT_SOURCE_DIR}/${API_INCLUDE_DIR})
    set(ALL_API_INCLUDE_DIRS ${_TEMP_API_INC} CACHE INTERNAL "All API include dirs")

    # Kernel args
    set(_TEMP_ARGS ${ALL_KERNEL_ARGS})
    list(APPEND _TEMP_ARGS "${KERNEL_ARGS}")
    set(ALL_KERNEL_ARGS ${_TEMP_ARGS} CACHE INTERNAL "Kernel compile args")

    message(STATUS "Registered ${OP_NAME}: kernel=${KERNEL_SRCS}, api=${API_SRCS}")
endmacro()