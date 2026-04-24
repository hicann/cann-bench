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

# test/utils/__init__.py
"""Utils module for operator testing framework."""

from .device_manager import DeviceManager, DeviceConfig
from .dtype_mapper import str_to_torch_dtype, is_float_dtype, is_int_dtype
from .golden_importer import GoldenImporter
from .param_builder import ParamBuilder

__all__ = [
    'DeviceManager', 'DeviceConfig',
    'str_to_torch_dtype', 'is_float_dtype', 'is_int_dtype',
    'GoldenImporter', 'ParamBuilder'
]
