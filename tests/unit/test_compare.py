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
