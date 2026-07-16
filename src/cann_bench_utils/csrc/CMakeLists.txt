# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

# 根目录 CMakeLists.txt：遍历 ops/ 下的所有算子目录并注册

# extension.cpp (不参与算子注册)
set(EXTENSION_CPP ${CMAKE_CURRENT_SOURCE_DIR}/extension.cpp)

# 遍历 ops/ 下的所有子目录
file(GLOB OP_DIRS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ops/*)
foreach(OP_DIR ${OP_DIRS})
    if(IS_DIRECTORY ${OP_DIR})
        get_filename_component(OP_NAME ${OP_DIR} NAME)
        message(STATUS "Processing operator: ${OP_NAME}")

        # 每个算子目录必须包含 op_kernel/ 和 op_plugin/
        set(KERNEL_DIR ${OP_DIR}/op_kernel)
        set(PLUGIN_DIR ${OP_DIR}/op_plugin)

        if(EXISTS ${KERNEL_DIR} AND EXISTS ${PLUGIN_DIR})
            # 收集 kernel 源文件 (bisheng 编译)
            file(GLOB_RECURSE KERNEL_SRCS CONFIGURE_DEPENDS ${KERNEL_DIR}/*.cpp)
            if(KERNEL_SRCS)
                set(ALL_KERNEL_SRCS ${ALL_KERNEL_SRCS} ${KERNEL_SRCS} CACHE INTERNAL "All kernel source files")
                set(ALL_KERNEL_INCLUDE_DIRS ${ALL_KERNEL_INCLUDE_DIRS} ${KERNEL_DIR} CACHE INTERNAL "All kernel include dirs")
                message(STATUS "  Registered kernel: ${KERNEL_SRCS}")
            endif()

            # 收集 plugin 源文件 (g++ 编译)
            file(GLOB_RECURSE PLUGIN_SRCS CONFIGURE_DEPENDS ${PLUGIN_DIR}/*.cpp)
            if(PLUGIN_SRCS)
                set(ALL_PLUGIN_SRCS ${ALL_PLUGIN_SRCS} ${PLUGIN_SRCS} CACHE INTERNAL "All plugin source files")
                set(ALL_PLUGIN_INCLUDE_DIRS ${ALL_PLUGIN_INCLUDE_DIRS} ${PLUGIN_DIR} ${KERNEL_DIR} CACHE INTERNAL "All plugin include dirs")
                message(STATUS "  Registered plugin: ${PLUGIN_SRCS}")
            endif()
        else()
            message(WARNING "Operator ${OP_NAME} missing op_kernel/ or op_plugin/ subdirectory, skipped.")
        endif()
    endif()
endforeach()
