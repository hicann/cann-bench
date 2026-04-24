# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# 自动发现并添加所有算子目录
# 算子目录通过自己的CMakeLists.txt调用register_aclnn_op()注册到全局列表

# 获取所有子目录
file(GLOB SUB_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(SUB_DIR ${SUB_DIRS})
    # 排除 op_kernel 统一目录，以及下划线开头的辅助目录(如 _common/ 存放
    # 共享的 plugin helper header — 不是算子目录，没有 CMakeLists.txt)。
    if(IS_DIRECTORY ${SUB_DIR})
        get_filename_component(DIR_NAME ${SUB_DIR} NAME)
        if(NOT DIR_NAME STREQUAL "op_kernel" AND NOT DIR_NAME MATCHES "^_")
            message(STATUS "Adding op directory: ${DIR_NAME}")
            add_subdirectory(${SUB_DIR})
        endif()
    endif()
endforeach()