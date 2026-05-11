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
summarize_case_results 单元测试

测试用例结果统计函数的各种场景：
- 全部通过
- 全部失败
- 混合结果
- 空列表
- 加速比计算
"""

import pytest
from unittest.mock import MagicMock

from kernel_eval.eval.results import summarize_case_results, CaseResultSummary
from kernel_eval.eval.perf_eval import PerfResult
from kernel_eval.eval.accuracy_eval import AccuracyResult


def create_mock_case_result(success: bool, has_accuracy: bool = False, speedup: float = 0.0):
    """创建模拟的 EvalCaseResult"""
    mock = MagicMock()
    mock.success = success
    mock.accuracy_result = AccuracyResult(
        passed=False, dtype='float32', threshold=0.001, mare=0.01, mere=0.01
    ) if has_accuracy else None
    mock.get_speedup = MagicMock(return_value=speedup)
    return mock


class TestSummarizeCaseResults:
    """summarize_case_results 函数测试"""

    def test_all_passed(self):
        """测试全部通过的用例"""
        cases = [
            create_mock_case_result(success=True, speedup=2.0),
            create_mock_case_result(success=True, speedup=3.0),
            create_mock_case_result(success=True, speedup=4.0),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 3
        assert summary.failed == 0
        assert summary.skipped == 0
        assert summary.avg_speedup == 3.0  # (2+3+4)/3
        assert summary.pass_rate == 1.0

    def test_all_failed_with_accuracy(self):
        """测试全部精度失败的用例"""
        cases = [
            create_mock_case_result(success=False, has_accuracy=True),
            create_mock_case_result(success=False, has_accuracy=True),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 0
        assert summary.failed == 2
        assert summary.skipped == 0
        assert summary.avg_speedup == 0.0
        assert summary.pass_rate == 0.0

    def test_all_skipped(self):
        """测试全部跳过的用例（执行失败）"""
        cases = [
            create_mock_case_result(success=False, has_accuracy=False),
            create_mock_case_result(success=False, has_accuracy=False),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.skipped == 2
        assert summary.avg_speedup == 0.0
        assert summary.pass_rate == 0.0

    def test_mixed_results(self):
        """测试混合结果"""
        cases = [
            create_mock_case_result(success=True, speedup=2.0),      # passed
            create_mock_case_result(success=False, has_accuracy=True),  # failed (精度)
            create_mock_case_result(success=False, has_accuracy=False),  # skipped (执行)
            create_mock_case_result(success=True, speedup=4.0),      # passed
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 2
        assert summary.failed == 1
        assert summary.skipped == 1
        assert summary.avg_speedup == 3.0  # (2+4)/2
        assert summary.pass_rate == 0.5

    def test_empty_list(self):
        """测试空列表"""
        summary = summarize_case_results([])

        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.skipped == 0
        assert summary.avg_speedup == 0.0
        assert summary.pass_rate == 0.0

    def test_speedup_zero_excluded(self):
        """测试 speedup=0 的用例被排除在平均值计算外"""
        cases = [
            create_mock_case_result(success=True, speedup=2.0),
            create_mock_case_result(success=True, speedup=0.0),  # 无 baseline
            create_mock_case_result(success=True, speedup=4.0),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 3
        assert summary.avg_speedup == 3.0  # (2+4)/2, 排除 0

    def test_all_zero_speedup(self):
        """测试所有 speedup 为 0"""
        cases = [
            create_mock_case_result(success=True, speedup=0.0),
            create_mock_case_result(success=True, speedup=0.0),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 2
        assert summary.avg_speedup == 0.0

    def test_failed_cases_excluded_from_speedup(self):
        """测试失败用例不参与 speedup 计算"""
        cases = [
            create_mock_case_result(success=True, speedup=2.0),
            create_mock_case_result(success=False, has_accuracy=True),
            create_mock_case_result(success=True, speedup=4.0),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 2
        assert summary.failed == 1
        assert summary.avg_speedup == 3.0  # 只有 passed 参与


class TestCaseResultSummary:
    """CaseResultSummary 数据类测试"""

    def test_dataclass_fields(self):
        """测试数据类字段"""
        summary = CaseResultSummary(
            passed=10,
            failed=2,
            skipped=1,
            avg_speedup=3.5,
            pass_rate=10/13,
        )

        assert summary.passed == 10
        assert summary.failed == 2
        assert summary.skipped == 1
        assert summary.avg_speedup == 3.5
        assert summary.pass_rate == pytest.approx(10/13)


class TestSummarizeCaseResultsIntegration:
    """集成测试：使用真实 EvalCaseResult"""

    def test_with_real_objects(self):
        """测试使用真实的 EvalCaseResult"""
        from kernel_eval.eval.results import EvalCaseResult

        cases = [
            EvalCaseResult(
                case_id="case_1",
                rel_path="level1/exp",
                operator="Exp",
                case_num=1,
                success=True,
                perf_result=PerfResult(case_id="case_1", elapsed_us=100),
                baseline_perf_us=200,  # speedup = 2.0
            ),
            EvalCaseResult(
                case_id="case_2",
                rel_path="level1/exp",
                operator="Exp",
                case_num=2,
                success=True,
                perf_result=PerfResult(case_id="case_2", elapsed_us=50),
                baseline_perf_us=200,  # speedup = 4.0
            ),
            EvalCaseResult(
                case_id="case_3",
                rel_path="level1/exp",
                operator="Exp",
                case_num=3,
                success=False,
                accuracy_result=AccuracyResult(
                    passed=False, dtype='float32', threshold=0.001, mare=0.1, mere=0.1
                ),
            ),
        ]
        summary = summarize_case_results(cases)

        assert summary.passed == 2
        assert summary.failed == 1
        assert summary.skipped == 0
        assert summary.avg_speedup == 3.0
        assert summary.pass_rate == pytest.approx(2/3)