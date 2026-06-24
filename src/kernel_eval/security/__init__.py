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
安全防护层

职责：
1. Timing API防护（防止monkey-patch攻击）
2. 返回值类型检查（防止懒求值/子类伪装攻击）
3. 二次验证支持

使用方法：
    from kernel_eval.security import APIGuard, check_output_type

    # Timing API防护
    guard = APIGuard()
    guard.snapshot()  # 安装wheel前快照
    install_wheel(path)
    guard.verify()    # 验证完整性
    guard.restore()   # 程序退出前恢复
"""

from .api_guard import APIGuard, snapshot_timing_apis, verify_timing_apis, restore_timing_apis
from .type_checker import check_output_type, check_tensor_validity
from .device_residency_guard import (
    DeviceResidencyGuard, DeviceEgressError, BuiltinComputeError,
)

__all__ = [
    "APIGuard",
    "snapshot_timing_apis",
    "verify_timing_apis",
    "restore_timing_apis",
    "check_output_type",
    "check_tensor_validity",
    "DeviceResidencyGuard",
    "DeviceEgressError",
    "BuiltinComputeError",
]