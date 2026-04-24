# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# 算子目录自动发现，不需要修改此文件

# Extension entry point (g++)
add_library(extension_obj OBJECT extension.cpp)
target_compile_options(extension_obj PRIVATE ${COMPILE_OPTS} -DTORCH_EXTENSION_NAME=_C)
target_include_directories(extension_obj PRIVATE ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
set(TEMP ${OBJECTS_LIST})
list(APPEND TEMP $<TARGET_OBJECTS:extension_obj>)
set(OBJECTS_LIST ${TEMP} CACHE INTERNAL "objects" FORCE)

add_subdirectory(ops)