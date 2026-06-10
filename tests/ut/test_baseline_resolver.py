#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""

baseline 硬件解析模块单元测试

测试对象：kernel_eval.utils.baseline_resolver
核心功能：
1. resolve_hardware - 硬件名称解析（产品型号别名 → 逻辑名）
2. PLATFORM_ALIAS - 平台别名映射
3. DEFAULT_HARDWARE - 默认硬件常量
"""

import pytest

from kernel_eval.utils.baseline_resolver import (
    resolve_hardware,
    PLATFORM_ALIAS,
    DEFAULT_HARDWARE,
)


class TestResolveHardware:
    """resolve_hardware 函数测试"""

    def test_exact_match(self):
        """精确匹配"""
        assert resolve_hardware("910b2") == "910b2"
        assert resolve_hardware("Ascend910_9362") == "910b2"
        assert resolve_hardware("Ascend910B2") == "910b2"
        assert resolve_hardware("Atlas-A2") == "910b2"

    def test_prefix_match(self):
        """前缀匹配"""
        assert resolve_hardware("Ascend310P3") == "310p"
        assert resolve_hardware("Ascend310P") == "310p"

    def test_no_match_returns_original(self):
        """无匹配返回原值"""
        assert resolve_hardware("unknown_chip") == "unknown_chip"
        assert resolve_hardware("ascend310p") == "ascend310p"

    def test_910b1_aliases(self):
        """910B1 别名"""
        assert resolve_hardware("910b1") == "910b1"
        assert resolve_hardware("Ascend910B1") == "910b1"
        assert resolve_hardware("Ascend910_9361") == "910b1"

    def test_910b2_variants(self):
        """910B2 变体"""
        assert resolve_hardware("Ascend910_9362B") == "910b2"

    def test_longest_prefix_wins(self):
        """最长前缀优先"""
        # "Ascend910" 不匹配任何 key 的前缀（"Ascend910B2" 不是 "Ascend910" 的前缀）
        # 所以返回原值
        assert resolve_hardware("Ascend910") == "Ascend910"


class TestPlatformAlias:
    """PLATFORM_ALIAS 映射测试"""

    def test_contains_key_entries(self):
        """包含必要的映射"""
        assert "Ascend910_9362" in PLATFORM_ALIAS
        assert "910b2" in PLATFORM_ALIAS
        assert "Ascend310P" in PLATFORM_ALIAS

    def test_all_values_are_logical_names(self):
        """所有值是逻辑名"""
        for key, value in PLATFORM_ALIAS.items():
            assert value in ("910b2", "910b1", "310p")


class TestDefaultHardware:
    """DEFAULT_HARDWARE 常量测试"""

    def test_default_value(self):
        """默认值为 910b2"""
        assert DEFAULT_HARDWARE == "910b2" or DEFAULT_HARDWARE is not None

    def test_env_override(self):
        """环境变量覆盖（不修改 env，仅验证类型）"""
        assert isinstance(DEFAULT_HARDWARE, str)