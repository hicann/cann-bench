#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
验证 checker 结果类 getter 方法已提升到 OutputResult 基类

AllCloseOutputResult 和 RelativeErrorOutputResult 的 4 个 getter 完全相同,
应继承自 OutputResult 基类的具体实现（不含 get_threshold，该方法在 AccuracyResult 上）。
"""

import pytest


class TestOutputResultGettersLifted:
    """OutputResult 基类应提供具体的 getter 实现"""

    def test_output_result_has_concrete_getters(self):
        """OutputResult 的 4 个 getter 不应该是 abstractmethod"""
        from kernel_eval.base.result import OutputResult
        import inspect

        for name in ('get_index', 'is_passed', 'get_dtype', 'get_error_msg'):
            method = getattr(OutputResult, name)
            assert not hasattr(method, '__isabstractmethod__'), \
                f"OutputResult.{name} should be concrete, not abstract"

    def test_allclose_does_not_override_getters(self):
        """AllCloseOutputResult 不应覆盖基类 getter"""
        from kernel_eval.checkers.allclose_checker import AllCloseOutputResult
        for name in ('get_index', 'is_passed', 'get_dtype', 'get_error_msg'):
            assert name not in AllCloseOutputResult.__dict__, \
                f"AllCloseOutputResult should not define {name}"

    def test_relative_error_does_not_override_getters(self):
        """RelativeErrorOutputResult 不应覆盖基类 getter"""
        from kernel_eval.checkers.relative_error_checker import RelativeErrorOutputResult
        for name in ('get_index', 'is_passed', 'get_dtype', 'get_error_msg'):
            assert name not in RelativeErrorOutputResult.__dict__, \
                f"RelativeErrorOutputResult should not define {name}"

    def test_allclose_getters_still_work(self):
        """AllCloseOutputResult 的 getter 应正常工作（通过继承）"""
        from kernel_eval.checkers.allclose_checker import AllCloseOutputResult
        r = AllCloseOutputResult(index=0, passed=True, dtype='float32',
                                 metadata={'threshold': 1e-5})
        assert r.get_index() == 0
        assert r.is_passed() is True
        assert r.get_dtype() == 'float32'
        assert r.get_error_msg() == ''

    def test_relative_error_getters_still_work(self):
        """RelativeErrorOutputResult 的 getter 应正常工作（通过继承）"""
        from kernel_eval.checkers.relative_error_checker import RelativeErrorOutputResult
        r = RelativeErrorOutputResult(index=1, passed=False, dtype='float16',
                                     error_msg='MARE exceeded',
                                     metadata={'threshold': 1e-3})
        assert r.get_index() == 1
        assert r.is_passed() is False
        assert r.get_dtype() == 'float16'
        assert r.get_error_msg() == 'MARE exceeded'