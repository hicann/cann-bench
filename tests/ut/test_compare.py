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
张量对比模块单元测试

测试对象：kernel_eval.utils.compare
"""

import pytest
import torch

from kernel_eval.utils.compare import (
    SingleOutputResult,
    CompareResult,
    compare_tensors,
)


class TestSingleOutputResult:
    """SingleOutputResult 数据类测试"""

    def test_creation_float(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=True,
                               metadata={"dtype_category": "float", "threshold": 2**-13,
                                         "mere": 0.001, "mare": 0.005})
        assert r.index == 0
        assert r.dtype == "float32"
        assert r.passed is True

    def test_creation_int(self):
        r = SingleOutputResult(index=1, dtype="int64", passed=True,
                               mismatch_count=0, total_count=100,
                               metadata={"dtype_category": "int", "threshold": 0})
        assert r.metadata["dtype_category"] == "int"
        assert r.mismatch_count == 0

    def test_to_dict(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=True,
                               metadata={"threshold": 0.001, "mere": 1e-5, "mare": 2e-5})
        d = r.to_dict()
        assert d["index"] == 0
        assert d["passed"] is True
        assert d["mere"] == 1e-5

    def test_format_summary_float_pass(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=True,
                               metadata={"dtype_category": "float", "threshold": 2**-13,
                                         "mere": 1e-5, "mare": 2e-5})
        s = r.format_summary()
        assert "✅" in s

    def test_format_summary_float_fail(self):
        r = SingleOutputResult(index=0, dtype="float32", passed=False,
                               error_msg="MARE exceeded threshold",
                               metadata={"dtype_category": "float", "threshold": 2**-13,
                                         "mere": 0.01, "mare": 0.05})
        s = r.format_summary()
        assert "❌" in s

    def test_format_summary_int_pass(self):
        r = SingleOutputResult(index=0, dtype="int64", passed=True,
                               mismatch_count=0, total_count=100,
                               metadata={"dtype_category": "int", "threshold": 0})
        s = r.format_summary()
        assert "✅" in s

    def test_format_summary_int_fail(self):
        r = SingleOutputResult(index=0, dtype="int64", passed=False,
                               mismatch_count=5, total_count=100,
                               metadata={"dtype_category": "int", "threshold": 0})
        s = r.format_summary()
        assert "❌" in s


class TestCompareResult:
    """CompareResult 数据类测试"""

    def test_creation(self):
        result = CompareResult(
            passed=True, dtype="float32", threshold=2**-13,
            mere=1e-5, mare=2e-5,
        )
        assert result.passed is True
        assert result.dtype == "float32"

    def test_to_dict(self):
        result = CompareResult(
            passed=True, dtype="float32", threshold=0.0001,
            mere=1e-5, mare=2e-5, mismatch_count=0, total_count=100,
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["mismatch_count"] == 0
        assert "output_results" in d

    def test_failed_result(self):
        result = CompareResult(
            passed=False, dtype="float32", threshold=0.0001,
            mere=0.01, mare=0.05, error_msg="MARE exceeded threshold",
        )
        assert result.passed is False
        assert result.error_msg == "MARE exceeded threshold"

    def test_accuracy_type_check_preserves_diagnostic_context(self):
        from kernel_eval.eval.accuracy_eval import AccuracyEvaluator

        evaluator = AccuracyEvaluator()
        result = evaluator.evaluate(
            ai_output=None,
            golden_output=torch.ones(1),
            dtype="float32",
            diagnostic_context="raw call failed: unexpected keyword x",
        )

        assert result.passed is False
        assert "输出类型不支持: NoneType" in result.error_msg
        assert "raw call failed: unexpected keyword x" in result.error_msg

    def test_default_values(self):
        result = CompareResult(passed=True, dtype="float32", threshold=0.0001)
        assert result.mere == 0.0
        assert result.mare == 0.0

    def test_format_all_outputs(self):
        sr = SingleOutputResult(index=0, dtype="float32", passed=True,
                                metadata={"dtype_category": "float", "threshold": 2**-13,
                                          "mere": 1e-5, "mare": 2e-5})
        result = CompareResult(passed=True, dtype="float32", threshold=2**-13,
                               output_results=[sr])
        s = result.format_all_outputs()
        assert "float32" in s


class TestCompareTensors:
    """compare_tensors 函数测试"""

    def test_identical_tensors_pass(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden.clone()
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True
        assert result.mere == pytest.approx(0.0, abs=1e-10)

    def test_small_difference_pass(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 1e-6
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True

    def test_large_difference_fail(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 0.1
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_int_tensors_exact_match(self):
        golden = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = golden.clone()
        result = compare_tensors(output, golden, "int32")
        assert result.passed is True

    def test_int_tensors_mismatch(self):
        golden = torch.tensor([1, 2, 3], dtype=torch.int64)
        output = torch.tensor([1, 2, 4], dtype=torch.int64)
        result = compare_tensors(output, golden, "int32")
        assert result.passed is False

    def test_multiple_outputs(self):
        golden = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        output = [golden[0].clone(), golden[1].clone()]
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True

    def test_output_count_mismatch(self):
        golden = [torch.tensor([1.0]), torch.tensor([2.0])]
        output = [torch.tensor([1.0])]
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_shape_mismatch(self):
        golden = torch.tensor([1.0, 2.0, 3.0])
        output = torch.tensor([1.0, 2.0])
        result = compare_tensors(output, golden, "float32")
        assert result.passed is False

    def test_float16_threshold(self):
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        output = golden + 0.0005
        result = compare_tensors(output, golden, "float16")
        assert result.passed is True

    def test_empty_tensor(self):
        golden = torch.tensor([], dtype=torch.float64)
        output = torch.tensor([], dtype=torch.float64)
        result = compare_tensors(output, golden, "float32")
        assert result.passed is True


class TestSmallValueFallback:
    """小值域兜底判定测试

    判定标准对齐 docs/spec/benchmark_spec.md「小值域通过标准」：
        ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    当 CPU 无小值域错误（cpu_error=0，如 native_output=None 的完美截断）时分母取 1，
    NPU 至多 2 个小值域错误仍判通过、≥3 个才判失败。
    """

    def test_small_value_over_2_errors_fail_when_cpu_clean(self):
        """CPU 无错误 + NPU 小值域错误 >2（ratio>2）→ 应失败"""
        # 构造场景：约一半 golden 值很小（小值域区域）
        n = 1000
        golden = torch.zeros(n, dtype=torch.float64)
        small_count = 500
        golden[:small_count] = 1e-6   # 小值域区域
        golden[small_count:] = 1.0    # 正常区域

        # output 基本匹配，但在 3 个小值域位置有巨大偏差（ratio = 3 / max(0,1) = 3 > 2）
        output = golden.half()
        output[0] = 65504.0  # fp16 max
        output[1] = 65504.0
        output[2] = 65504.0

        # native_output=None → CPU 参考是 golden_truncated（完美截断，cpu_diff=0）
        result = compare_tensors(output, golden, "float16")

        assert result.passed is False
        assert result.small_value_error_count > 2
        assert result.small_value_cpu_error_count == 0
        assert result.mismatch_count > 0

    def test_small_value_le2_errors_pass_when_cpu_clean(self):
        """CPU 无错误 + NPU 小值域错误 ≤2（ratio = 2/max(0,1) = 2 ≤ 2）→ 应通过

        这是对齐文档公式后的行为：cpu_error=0 时分母取 1，容忍最多 2 个 NPU 小值域错误。
        （旧实现在此场景要求 NPU 零错误，与文档不一致，已改为以文档为准。）
        """
        n = 1000
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:500] = 1e-6   # 小值域区域
        golden[500:] = 1.0    # 正常区域

        output = golden.half()
        output[0] = 65504.0   # 2 个小值域巨大偏差
        output[1] = 65504.0

        result = compare_tensors(output, golden, "float16")

        assert result.small_value_error_count == 2
        assert result.small_value_cpu_error_count == 0
        assert result.small_value_passed is True
        # 正常值域全部匹配 → 整体判定由小值域兜底决定，应通过
        assert result.passed is True

    def test_small_value_passes_when_both_npu_and_cpu_have_errors(self):
        """NPU 和 CPU 都有小值域误差（ratio ≤ 2）→ 应通过"""
        # 构造 native_output 使得 CPU 也有小值域误差
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6   # 小值域
        golden[50:] = 1.0    # 正常

        output = golden.half()
        # NPU 在小值域有 2 个错误
        output[0] = 0.001   # 误差 > small_value_error
        output[1] = 0.002

        # native_output 也有小值域误差（模拟 CPU 精度截断也会犯错）
        native = golden.half()
        native[0] = 0.0015  # CPU 也有误差
        native[1] = 0.0025

        result = compare_tensors(output, golden, "float16", native_output=native)

        # NPU=2, CPU=2, ratio=1 ≤ 2 → 通过
        assert result.passed is True

    def test_small_value_fails_when_ratio_exceeds_2(self):
        """NPU 误差远多于 CPU（ratio > 2）→ 应失败"""
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6
        golden[50:] = 1.0

        output = golden.half()
        # NPU 在小值域有 10 个错误
        for i in range(10):
            output[i] = 0.001 + i * 0.001

        # native_output 只有 2 个错误
        native = golden.half()
        native[0] = 0.001
        native[1] = 0.002

        result = compare_tensors(output, golden, "float16", native_output=native)

        # NPU=10, CPU=2, ratio=5 > 2 → 失败
        assert result.passed is False

    def test_small_value_no_errors_passes(self):
        """小值域区域无任何误差 → 应通过"""
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6
        golden[50:] = 1.0

        output = golden.half()  # 完美截断

        result = compare_tensors(output, golden, "float16")

        assert result.passed is True
        assert result.small_value_error_count == 0


class TestCancelFallback:
    """相消区域兜底判定测试（与小值域相同标准：npu / max(cpu,1) ≤ 2）

    构造纯相消场景以真正走到 cancel 兜底路径：golden 全部落在相消区间
    [small_value_threshold, cancel_boundary)（fp16 为 [4.88e-4, 0.03125)），
    output 全部 near-zero（< cancel_zero_threshold），从而所有位置都进入 cancel_mask、
    无 normal_mismatch —— 整体判定完全由 cancel 兜底决定。
    """

    def _make_cancel_case(self, n=100, n_err=0):
        """golden=0.01（相消区间内），output≈0.01（near-zero 但非错误），
        前 n_err 个位置 output=0（相消错误）。native_output=None → CPU 无相消错误。"""
        golden = torch.full((n,), 0.01, dtype=torch.float64)
        output = golden.half()          # ≈0.01，near-zero 区间内，rel_err 极小→非错误
        output[:n_err] = 0.0            # n_err 个 near-zero 巨大相对误差→相消错误
        return output, golden

    def test_cancel_over_2_errors_fail_when_cpu_clean(self):
        """CPU 无相消错误 + NPU 相消错误 >2（ratio>2）→ 应失败"""
        output, golden = self._make_cancel_case(n_err=3)
        result = compare_tensors(output, golden, "float16")

        assert result.cancel_error_count > 2
        assert result.cancel_cpu_error_count == 0
        assert result.cancel_passed is False
        # 无正常值域误差 → 判定确由 cancel 兜底决定
        assert result.passed is False

    def test_cancel_le2_errors_pass_when_cpu_clean(self):
        """CPU 无相消错误 + NPU 相消错误 ≤2（ratio = 2/max(0,1) = 2 ≤ 2）→ 应通过

        对齐文档 npu / max(cpu,1) ≤ 2；旧实现在 cpu_error=0 时要求零错误，与文档不一致，已改。
        """
        output, golden = self._make_cancel_case(n_err=2)
        result = compare_tensors(output, golden, "float16")

        assert result.cancel_error_count == 2
        assert result.cancel_cpu_error_count == 0
        assert result.cancel_passed is True
        assert result.passed is True


class TestBitExactFloat:
    """threshold=0 触发的浮点 bit-exact 路径"""

    BIT_EXACT = {"float16": 0, "bfloat16": 0, "float32": 0}

    def test_identical_fp32_passes(self):
        a = torch.tensor([1.0, -1.0, 0.0, float("inf"), -float("inf")], dtype=torch.float32)
        result = compare_tensors(a.clone(), a, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is True
        assert result.mismatch_count == 0

    def test_signed_zero_divergence_fails_fp32(self):
        out = torch.tensor([+0.0, 1.0, 2.0], dtype=torch.float32)
        gold = torch.tensor([-0.0, 1.0, 2.0], dtype=torch.float32)
        result = compare_tensors(out, gold, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is False
        assert result.mismatch_count == 1

    def test_signed_inf_divergence_fails(self):
        out = torch.tensor([float("inf"), 1.0], dtype=torch.float32)
        gold = torch.tensor([-float("inf"), 1.0], dtype=torch.float32)
        result = compare_tensors(out, gold, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is False

    def test_one_ulp_off_fails(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = a.clone()
        b[1] = torch.nextafter(b[1], torch.tensor(100.0))
        result = compare_tensors(a, b, "float32", custom_thresholds=self.BIT_EXACT)
        assert result.passed is False

    def test_fp64_golden_against_target_dtype_output(self):
        for target_dtype, dtype_str in [
            (torch.bfloat16, "bfloat16"),
            (torch.float16, "float16"),
            (torch.float32, "float32"),
        ]:
            x = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0], dtype=target_dtype)
            x_fp64 = x.double()
            golden_fp64, _ = torch.unique(x_fp64, return_inverse=True)
            output, _ = torch.unique(x, return_inverse=True)
            result = compare_tensors(output, golden_fp64, dtype_str,
                                     custom_thresholds=self.BIT_EXACT)
            assert result.passed, f"{target_dtype} round-trip via fp64 should pass bit-exact"


class TestCompareResultFallbackFlags:
    """CompareResult 兜底判定标志传播测试"""

    def test_small_value_passed_and_cancel_passed_in_result(self):
        """CompareResult 应包含 small_value_passed / cancel_passed 字段"""
        # 正常通过的场景：默认 True
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        output = golden.clone()
        result = compare_tensors(output, golden, "float32")
        assert result.small_value_passed is True
        assert result.cancel_passed is True

    def test_small_value_passed_false_propagated(self):
        """小值域兜底未通过时，small_value_passed 应为 False 并传播。

        构造 CPU 无错误、NPU 有 50 个小值域错误：ratio = 50 / max(0,1) = 50 > 2 →
        兜底判定失败（按文档公式 npu / max(cpu,1) ≤ 2）。
        """
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6  # 小值域
        golden[50:] = 1.0
        # NPU 在小值域有巨大误差（50 个 >> 2）
        output = golden.half()
        output[:50] = 100.0  # 小值域位置严重偏离

        result = compare_tensors(output, golden, "float16")

        # 50 个 NPU 错误、CPU 0 错误 → ratio 50 > 2 → small_value_passed 必为 False
        assert result.small_value_error_count == 50
        assert result.small_value_cpu_error_count == 0
        assert result.small_value_passed is False

    def test_small_value_passed_reflected_in_output_metadata(self):
        """small_value_passed 应通过 output_results metadata 传播"""
        from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker
        n = 100
        golden = torch.zeros(n, dtype=torch.float64)
        golden[:50] = 1e-6
        golden[50:] = 1.0
        output = golden.half()
        output[:50] = 100.0

        checker = RelativeErrorChecker()
        accuracy = checker.check(
            ai_outputs=output,
            golden_outputs=golden,
            dtype='float16',
            threshold=2**-10,
        )

        # 验证 output_results metadata 包含 small_value_passed
        for or_ in accuracy.output_results:
            assert 'small_value_passed' in or_.metadata
            assert 'cancel_passed' in or_.metadata

    def test_format_summary_shows_small_value_failure_when_both_fail(self):
        """当相对误差和兜底判定都失败时，format_summary 应同时展示两者"""
        from kernel_eval.checkers.relative_error_checker import RelativeErrorOutputResult
        # 构造一个 passed=False 且 small_value_passed=False 的结果
        result = RelativeErrorOutputResult(
            index=0,
            dtype='float16',
            passed=False,
            error_msg='',
            mismatch_count=10,
            total_count=100,
            metadata={
                'dtype_category': 'float',
                'threshold': 2**-10,
                'mere': 0.5,
                'mare': 2.0,
                'small_value_error_count': 5,
                'small_value_cpu_error_count': 0,
                'small_value_total_count': 10,
                'cancel_error_count': 0,
                'cancel_cpu_error_count': 0,
                'cancel_total_count': 0,
                'small_value_passed': False,
                'cancel_passed': True,
            },
        )
        summary = result.format_summary()
        # 验证摘要中包含小值域兜底失败信息
        assert '❌' in summary
        assert '小值域兜底❌' in summary
        assert 'NPU/CPU错误=5/0' in summary

    def test_format_summary_shows_cancel_failure_when_both_fail(self):
        """当相对误差和相消兜底判定都失败时，format_summary 应同时展示两者"""
        from kernel_eval.checkers.relative_error_checker import RelativeErrorOutputResult
        result = RelativeErrorOutputResult(
            index=0,
            dtype='float16',
            passed=False,
            error_msg='',
            mismatch_count=5,
            total_count=100,
            metadata={
                'dtype_category': 'float',
                'threshold': 2**-10,
                'mere': 0.3,
                'mare': 1.5,
                'small_value_error_count': 0,
                'small_value_cpu_error_count': 0,
                'small_value_total_count': 0,
                'cancel_error_count': 3,
                'cancel_cpu_error_count': 0,
                'cancel_total_count': 5,
                'small_value_passed': True,
                'cancel_passed': False,
            },
        )
        summary = result.format_summary()
        assert '❌' in summary
        assert '相消兜底❌' in summary
        assert 'NPU/CPU错误=3/0' in summary

    def test_format_summary_no_fallback_info_when_all_passed(self):
        """当只有相对误差失败但兜底都通过时，不追加兜底信息"""
        from kernel_eval.checkers.relative_error_checker import RelativeErrorOutputResult
        result = RelativeErrorOutputResult(
            index=0,
            dtype='float32',
            passed=False,
            error_msg='',
            metadata={
                'dtype_category': 'float',
                'threshold': 2**-13,
                'mere': 0.5,
                'mare': 2.0,
                'small_value_passed': True,
                'cancel_passed': True,
                'small_value_error_count': 0,
                'small_value_cpu_error_count': 0,
                'small_value_total_count': 0,
                'cancel_error_count': 0,
                'cancel_cpu_error_count': 0,
                'cancel_total_count': 0,
            },
        )
        summary = result.format_summary()
        assert '小值域兜底❌' not in summary
        assert '相消兜底❌' not in summary
        # 只显示 MERE/MARE
        assert 'MERE=' in summary


class TestNormalRegionSamePrecisionGate:
    """issue #92：正常值域同精度兜底门（深归约 fp32 固有误差场景）。

    fp32 阈值：mare_threshold = 10*2^-13 ≈ 1.22e-3；|golden|=1.0 远高于
    small_value_threshold(6.1e-5)/cancel_boundary(3.9e-3)，注入点均落在正常值域。
    """

    @staticmethod
    def _mk(n=1000, k=0, err=3e-3):
        """fp32 张量：全 1.0，前 k 个点相对误差 err（> mare_threshold）。"""
        t = torch.ones(n, dtype=torch.float32)
        if k > 0:
            t[:k] = float(1.0 + err)
        return t

    def _cmp(self, output, native, n=1000):
        golden = torch.ones(n, dtype=torch.float64)  # fp64 oracle
        return compare_tensors(output, golden, dtype="float32", native_output=native)

    def test_candidate_equals_native_passes(self):
        """候选 == 同精度参考（issue #92 复现）→ npu==cpu → ratio 1 → 通过。"""
        ref = self._mk(k=100)
        r = self._cmp(ref, ref)
        assert r.normal_error_count == 100 and r.normal_cpu_error_count == 100
        assert r.normal_passed is True and r.passed is True

    def test_candidate_within_2x_reference_passes(self):
        r = self._cmp(self._mk(k=150), self._mk(k=100))  # 1.5x 参考
        assert r.normal_error_count == 150 and r.normal_cpu_error_count == 100
        assert r.normal_passed is True and r.passed is True

    def test_candidate_worse_than_2x_reference_fails(self):
        r = self._cmp(self._mk(k=300), self._mk(k=100))  # 3x 参考
        assert r.normal_error_count == 300 and r.normal_cpu_error_count == 100
        assert r.normal_passed is False and r.passed is False

    def test_clean_reference_keeps_strict(self):
        """守卫：参考干净(0 错点)、候选却超标 → 仍失败（非病态场景零放宽）。"""
        r = self._cmp(self._mk(k=5), self._mk(k=0))
        assert r.normal_cpu_error_count == 0 and r.normal_error_count == 5
        assert r.normal_passed is False and r.passed is False

    def test_candidate_better_than_reference_passes(self):
        """候选比参考还好(0 错点) → 通过。"""
        r = self._cmp(self._mk(k=0), self._mk(k=100))
        assert r.normal_error_count == 0
        assert r.normal_passed is True and r.passed is True
