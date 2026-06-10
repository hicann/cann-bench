#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
工具层模块

职责：
1. 设备管理（CPU/NPU切换、张量迁移）
2. 数据类型映射（字符串与torch.dtype转换）
3. 参数构建（根据函数签名构建调用参数）
4. 精度验证（MERE/MARE标准）
5. Baseline硬件名称解析（平台别名映射）
6. 命名转换（PascalCase→snake_case）
7. 张量处理（精度转换、设备迁移）
8. 目录解析（task-dir 参数处理）
"""

from .device_manager import DeviceManager, DeviceConfig
from .dtype_mapper import str_to_torch_dtype, torch_dtype_to_str, is_float_dtype, is_int_dtype
from .param_builder import ParamBuilder
from .compare import compare_tensors, CompareResult
from .thresholds import PRECISION_THRESHOLDS
from .baseline_resolver import (
    resolve_hardware, PLATFORM_ALIAS,
)
from .naming import camel_to_snake, snake_case_candidates
from .tensor_utils import (
    tensor_to_fp64_cpu, tensors_to_cpu,
    tensors_to_fp64_cpu, tensors_to_device,
)
from .path_resolver import (
    resolve_task_dir,
    is_operator_directory,
    find_bench_root,
)

__all__ = [
    "DeviceManager", "DeviceConfig",
    "str_to_torch_dtype", "torch_dtype_to_str", "is_float_dtype", "is_int_dtype",
    "ParamBuilder",
    "compare_tensors", "CompareResult", "PRECISION_THRESHOLDS",
    "resolve_hardware", "PLATFORM_ALIAS",
    "camel_to_snake", "snake_case_candidates",
    "tensor_to_fp64_cpu", "tensors_to_cpu",
    "tensors_to_fp64_cpu", "tensors_to_device",
    "resolve_task_dir", "is_operator_directory", "find_bench_root",
]