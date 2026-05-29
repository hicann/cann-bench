# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ACLNN算子注册宏
# 算子目录通过调用register_aclnn_op()将自己注册到全局列表

# 全局变量定义
set(ALL_HOST_OPS_SRCS "" CACHE INTERNAL "All host source files")
set(ALL_API_OPS_SRCS "" CACHE INTERNAL "All API source files")
set(ALL_KERNEL_OPS_INFO "" CACHE INTERNAL "All kernel info (OP_TYPE|KERNEL_DIR|KERNEL_FILE)")
set(ALL_TILING_INCLUDE_DIRS "" CACHE INTERNAL "All tiling include directories")
set(ALL_API_INCLUDE_DIRS "" CACHE INTERNAL "All API include directories")
set(ALL_PLUGIN_SRCS "" CACHE INTERNAL "All plugin source files")
set(ALL_PLUGIN_INCLUDE_DIRS "" CACHE INTERNAL "All plugin include directories")

# 注册ACLNN算子
# 参数:
#   OP_TYPE: 算子类型名(如Add, Sqrt)
#   HOST_SRCS: host源文件列表
#   API_SRCS: API源文件列表
#   KERNEL_DIR: kernel目录名(相对于算子目录，如op_kernel)
#   KERNEL_FILE: kernel文件名(如add_kernel.cpp)
#   TILING_INCLUDE_DIR: tiling需要的include目录(如op_kernel)
#   API_INCLUDE_DIR: API需要的include目录(如op_api)
macro(register_aclnn_op OP_TYPE HOST_SRCS API_SRCS KERNEL_DIR KERNEL_FILE TILING_INCLUDE_DIR API_INCLUDE_DIR)
    message(STATUS "Registering ACLNN op: ${OP_TYPE}")

    # 添加host源文件到全局列表
    set(_TEMP_HOST ${ALL_HOST_OPS_SRCS})
    list(APPEND _TEMP_HOST ${HOST_SRCS})
    set(ALL_HOST_OPS_SRCS ${_TEMP_HOST} CACHE INTERNAL "All host source files")

    # 添加API源文件到全局列表
    set(_TEMP_API ${ALL_API_OPS_SRCS})
    list(APPEND _TEMP_API ${API_SRCS})
    set(ALL_API_OPS_SRCS ${_TEMP_API} CACHE INTERNAL "All API source files")

    # 添加kernel信息到全局列表
    set(_KERNEL_INFO "${OP_TYPE}|${KERNEL_DIR}|${KERNEL_FILE}")
    set(_TEMP_KERNEL ${ALL_KERNEL_OPS_INFO})
    list(APPEND _TEMP_KERNEL ${_KERNEL_INFO})
    set(ALL_KERNEL_OPS_INFO ${_TEMP_KERNEL} CACHE INTERNAL "All kernel info")

    # 添加include目录
    set(_TEMP_TILING_INC ${ALL_TILING_INCLUDE_DIRS})
    list(APPEND _TEMP_TILING_INC ${CMAKE_CURRENT_SOURCE_DIR}/${TILING_INCLUDE_DIR})
    set(ALL_TILING_INCLUDE_DIRS ${_TEMP_TILING_INC} CACHE INTERNAL "All tiling include directories")

    set(_TEMP_API_INC ${ALL_API_INCLUDE_DIRS})
    list(APPEND _TEMP_API_INC ${CMAKE_CURRENT_SOURCE_DIR}/${API_INCLUDE_DIR})
    set(ALL_API_INCLUDE_DIRS ${_TEMP_API_INC} CACHE INTERNAL "All API include directories")

    message(STATUS "Registered ${OP_TYPE}: host=${HOST_SRCS}, api=${API_SRCS}, kernel=${KERNEL_FILE}")
endmacro()

# 注册Python插件(可选)
# 参数:
#   PLUGIN_SRCS: 插件源文件列表
#   PLUGIN_INCLUDE_DIR: 插件需要的include目录(如op_api)
macro(register_aclnn_plugin PLUGIN_SRCS PLUGIN_INCLUDE_DIR)
    set(_TEMP_PLUGIN ${ALL_PLUGIN_SRCS})
    list(APPEND _TEMP_PLUGIN ${PLUGIN_SRCS})
    set(ALL_PLUGIN_SRCS ${_TEMP_PLUGIN} CACHE INTERNAL "All plugin source files")

    set(_TEMP_PLUGIN_INC ${ALL_PLUGIN_INCLUDE_DIRS})
    list(APPEND _TEMP_PLUGIN_INC ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_INCLUDE_DIR})
    set(ALL_PLUGIN_INCLUDE_DIRS ${_TEMP_PLUGIN_INC} CACHE INTERNAL "All plugin include directories")

    message(STATUS "Registered plugin: ${PLUGIN_SRCS}")
endmacro()