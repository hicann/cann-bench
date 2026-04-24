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
数据层模块

职责：
1. 算子定义加载（proto.yaml解析）
2. 测试用例加载（cases.yaml解析）
3. Golden函数加载（动态导入）
4. 数据生成（根据shape/dtype生成输入张量）
5. 包管理（源码扫描、编译、安装、接口扫描）
"""

from .operator_loader import OperatorLoader, OperatorInfo
from .case_loader import CaseLoader, CaseInfo
from .golden_loader import GoldenLoader
from .data_generator import DataGenerator
from .package_manager import PackageManager, PackageInfo, InterfaceInfo

__all__ = [
    "OperatorLoader", "OperatorInfo",
    "CaseLoader", "CaseInfo",
    "GoldenLoader",
    "DataGenerator",
    "PackageManager", "PackageInfo", "InterfaceInfo",
]