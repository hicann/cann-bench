# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mish算子自注册

set(MISH_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mish_kernel.cpp
)

set(MISH_API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mish_plugin.cpp
)

register_simple_op(
    "${MISH_KERNEL_SRCS}"
    op_kernel
    "${MISH_API_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
