#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
精度判断器模块

通用对比标准实现，不与任何评测集绑定。
各评测集通过 BenchConfig.checker 选用需要的判断器。

- RelativeErrorChecker: MERE/MARE + 小值域 + 相消处理（CANN 选用）
- AllCloseChecker: torch.allclose(atol/rtol)（StanfordBench 选用）
"""

from .relative_error_checker import RelativeErrorChecker, RelativeErrorOutputResult
from .allclose_checker import AllCloseChecker, AllCloseOutputResult

# 兼容旧名
CannDefaultChecker = RelativeErrorChecker
CannOutputResult = RelativeErrorOutputResult

__all__ = [
    "RelativeErrorChecker",
    "RelativeErrorOutputResult",
    "AllCloseChecker",
    "AllCloseOutputResult",
    "CannDefaultChecker",
    "CannOutputResult",
]